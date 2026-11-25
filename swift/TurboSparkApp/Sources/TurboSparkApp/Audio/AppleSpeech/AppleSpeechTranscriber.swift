#if compiler(>=6.2)
import AVFoundation
import CoreMedia
import Speech
#endif
import Foundation
import TurboSpark

extension AppleSpeech {
    /// `SpeechTranscriber.isAvailable` behind the OS gate, for routing that
    /// runs on every macOS the app supports.
    nonisolated static var isSupportedOnThisMac: Bool {
        isSupported(osMajor: ProcessInfo.processInfo.operatingSystemVersion.majorVersion) {
            #if compiler(>=6.2)
            if #available(macOS 26, *) { return SpeechTranscriber.isAvailable }
            #endif
            return false
        }
    }
}

enum AppleSpeechError: LocalizedError {
    case unsupportedLanguage
    case unavailable

    var errorDescription: String? {
        switch self {
        case .unsupportedLanguage: return AppStrings.localized("Apple speech recognition does not support this language on this Mac.")
        case .unavailable: return AppStrings.localized("Apple speech recognition is not available on this Mac.")
        }
    }
}

/// The Speech framework calls behind the built-in engine. Recognition runs on
/// this Mac: `SpeechTranscriber` has no server mode, unlike `SFSpeechRecognizer`.
@available(macOS 26, *)
enum AppleSpeechTranscriber {
#if compiler(>=6.2)
    /// The recognizer locale for `preferred`, else US English, else none.
    static func locale(preferred: Locale = .current) async throws -> Locale {
        if let match = await SpeechTranscriber.supportedLocale(equivalentTo: preferred) { return match }
        if let english = await SpeechTranscriber.supportedLocale(equivalentTo: Locale(identifier: "en-US")) { return english }
        throw AppleSpeechError.unsupportedLanguage
    }

    static func module(_ locale: Locale) -> SpeechTranscriber {
        // Final results only, with per-run audio time ranges for word timing.
        SpeechTranscriber(locale: locale, transcriptionOptions: [], reportingOptions: [], attributeOptions: [.audioTimeRange])
    }

    /// Whether macOS still has to download the language model for `locale`.
    static func needsAssets(_ locale: Locale) async -> Bool {
        await AssetInventory.status(forModules: [module(locale)]) < .installed
    }

    /// Asks macOS to download and install the language model, if it has not.
    /// macOS stores it outside the app and shares it with other apps. Called
    /// only from a run the user started.
    static func installAssets(_ locale: Locale, progress: @escaping @Sendable (Double) -> Void) async throws {
        guard let request = try await AssetInventory.assetInstallationRequest(supporting: [module(locale)]) else { return }
        let observation = request.progress.observe(\.fractionCompleted, options: [.initial, .new]) { value, _ in
            progress(value.fractionCompleted)
        }
        defer { observation.invalidate() }
        try await request.downloadAndInstall()
    }

    /// Transcribes 16 kHz mono PCM. Times in the result are seconds from the
    /// first sample.
    static func transcribe(_ pcm: [Float], locale: Locale) async throws -> [AppleSpeech.Piece] {
        guard !pcm.isEmpty else { return [] }
        let transcriber = module(locale)
        guard let source = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16_000, channels: 1, interleaved: false),
              let input = buffer(pcm, format: source) else { throw AppleSpeechError.unavailable }
        let target = await SpeechAnalyzer.bestAvailableAudioFormat(compatibleWith: [transcriber], considering: source) ?? source
        let converted = try convert(input, to: target)

        let analyzer = SpeechAnalyzer(modules: [transcriber])
        let collect = Task { () -> [AppleSpeech.Piece] in
            var pieces: [AppleSpeech.Piece] = []
            for try await result in transcriber.results where result.isFinal {
                pieces.append(piece(result))
            }
            return pieces
        }
        let (stream, continuation) = AsyncStream<AnalyzerInput>.makeStream()
        continuation.yield(AnalyzerInput(buffer: converted))
        continuation.finish()
        do {
            try await withTaskCancellationHandler {
                try await analyzer.prepareToAnalyze(in: target)
                if let end = try await analyzer.analyzeSequence(stream) {
                    try await analyzer.finalizeAndFinish(through: end)
                } else {
                    await analyzer.cancelAndFinishNow()
                }
            } onCancel: {
                Task { await analyzer.cancelAndFinishNow() }
            }
        } catch {
            collect.cancel()
            throw error
        }
        try Task.checkCancellation()
        return try await collect.value
    }

    private static func piece(_ result: SpeechTranscriber.Result) -> AppleSpeech.Piece {
        var words: [AudioWord] = []
        for run in result.text.runs {
            guard let range = run.audioTimeRange else { continue }
            let text = String(result.text[run.range].characters).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty, range.start.isNumeric, range.end.isNumeric else { continue }
            words.append(AudioWord(text: text, startSeconds: range.start.seconds, endSeconds: range.end.seconds))
        }
        let start = result.range.start.isNumeric ? result.range.start.seconds : (words.first?.startSeconds ?? 0)
        let end = result.range.end.isNumeric ? result.range.end.seconds : (words.last?.endSeconds ?? start)
        return AppleSpeech.Piece(text: String(result.text.characters), start: start, end: end, words: words)
    }

    private static func buffer(_ samples: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else { return nil }
        samples.withUnsafeBufferPointer { channel.update(from: $0.baseAddress!, count: samples.count) }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        return buffer
    }

    private static func convert(_ input: AVAudioPCMBuffer, to format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        if input.format == format { return input }
        guard let converter = AVAudioConverter(from: input.format, to: format) else { throw AppleSpeechError.unavailable }
        let ratio = format.sampleRate / input.format.sampleRate
        let capacity = AVAudioFrameCount((Double(input.frameLength) * ratio).rounded(.up)) + 1024
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: capacity) else { throw AppleSpeechError.unavailable }
        var supplied = false
        var failure: NSError?
        let status = converter.convert(to: output, error: &failure) { _, state in
            if supplied { state.pointee = .endOfStream; return nil }
            supplied = true
            state.pointee = .haveData
            return input
        }
        if status == .error { throw failure ?? AppleSpeechError.unavailable }
        return output
    }
#else
    static func locale(preferred: Locale = .current) async throws -> Locale {
        throw AppleSpeechError.unavailable
    }

    static func needsAssets(_ locale: Locale) async -> Bool {
        false
    }

    static func installAssets(_ locale: Locale, progress: @escaping @Sendable (Double) -> Void) async throws {
        throw AppleSpeechError.unavailable
    }

    static func transcribe(_ pcm: [Float], locale: Locale) async throws -> [AppleSpeech.Piece] {
        throw AppleSpeechError.unavailable
    }
#endif
}
