import AVFoundation
import Foundation
import Speech
import TurboSpark

enum AudioTranscriptionError: LocalizedError {
    case unavailable(String)
    case recognizerUnavailable
    case onDeviceUnavailable
    case noSpeech

    var errorDescription: String? {
        switch self {
        case .unavailable(let reason):
            return reason
        case .recognizerUnavailable:
            return String(localized: "Speech recognition is not available right now.", bundle: .module)
        case .onDeviceUnavailable:
            return String(
                localized: "On-device speech recognition is not available for this language.",
                bundle: .module)
        case .noSpeech:
            return String(localized: "No speech was detected in the recording.", bundle: .module)
        }
    }
}

/// A finished transcription and who made it.
struct RoutedTranscript: Sendable, Equatable {
    var text: String
    var provider: SpeechProvider
}

/// Speech to text, engine first.
///
/// Every input is normalized by the ENGINE to 16 kHz mono WAV before either
/// provider sees it (`AudioEngineBridge.speechCopy`), so the Rust path and
/// the Apple helper transcribe byte-identical audio and swapping providers
/// changes nothing upstream.
enum SpeechToTextRouter {
    static func transcribe(fileURL: URL, locale: Locale = .current) async throws -> RoutedTranscript {
        let availability = await MainActor.run { AudioCapabilities.shared.snapshot.transcription }
        guard let provider = availability.provider else {
            throw AudioTranscriptionError.unavailable(
                availability.reason
                    ?? String(localized: "Speech recognition is not available right now.", bundle: .module))
        }
        switch provider {
        case .engine:
            let session = try await TurboSparkAudioSession(
                modelPath: AudioPreferences.engineSpeechModelPath)
            let transcript = try await session.transcribe(
                fileURL, language: locale.language.languageCode?.identifier)
            return RoutedTranscript(text: transcript.text, provider: .engine)
        case .apple:
            let normalized = try await Task.detached(priority: .userInitiated) {
                try AudioEngineBridge.speechCopy(of: fileURL)
            }.value
            defer { try? FileManager.default.removeItem(at: normalized) }
            let text = try await AppleSpeechHelper.transcribe(fileURL: normalized, locale: locale)
            return RoutedTranscript(text: text, provider: .apple)
        }
    }
}

/// The native helper: `SFSpeechRecognizer`, ON DEVICE ONLY.
///
/// TurboSpark runs models locally and says so; a fallback that quietly
/// uploaded the user's voice to a server would contradict the product. A
/// locale without an on-device model is refused with a reason instead.
enum AppleSpeechHelper {
    static func transcribe(fileURL: URL, locale: Locale) async throws -> String {
        guard let recognizer = SFSpeechRecognizer(locale: locale) ?? SFSpeechRecognizer(),
            recognizer.isAvailable
        else {
            throw AudioTranscriptionError.recognizerUnavailable
        }
        guard recognizer.supportsOnDeviceRecognition else {
            throw AudioTranscriptionError.onDeviceUnavailable
        }
        let request = SFSpeechURLRecognitionRequest(url: fileURL)
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = false
        request.addsPunctuation = true

        let gate = ResumeOnce()
        let holder = RecognitionTaskHolder()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<String, Error>) in
                holder.task = recognizer.recognitionTask(with: request) { result, error in
                    if let result, result.isFinal {
                        let text = result.bestTranscription.formattedString
                            .trimmingCharacters(in: .whitespacesAndNewlines)
                        gate.run {
                            if text.isEmpty {
                                continuation.resume(throwing: AudioTranscriptionError.noSpeech)
                            } else {
                                continuation.resume(returning: text)
                            }
                        }
                    } else if let error {
                        gate.run { continuation.resume(throwing: error) }
                    }
                }
            }
        } onCancel: {
            holder.task?.cancel()
        }
    }
}

/// Resumes a continuation at most once. The recognizer can call its
/// handler several times (a final result, then a cancellation error).
private final class ResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false

    func run(_ body: () -> Void) {
        lock.lock()
        defer { lock.unlock() }
        guard !done else { return }
        done = true
        body()
    }
}

private final class RecognitionTaskHolder: @unchecked Sendable {
    var task: SFSpeechRecognitionTask?
}
