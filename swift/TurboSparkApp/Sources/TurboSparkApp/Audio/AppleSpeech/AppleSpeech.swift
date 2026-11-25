import Foundation
import TurboSpark

/// The macOS built-in speech recognizer (Speech framework `SpeechAnalyzer`)
/// as a Speech-to-Text engine that needs no TurboSpark download. macOS owns
/// its language assets (`AssetInventory`), so the row has no repository and
/// counts as installed whenever this Mac can run it.
///
/// This file does not import Speech: routing, ranking and tests read it on
/// any macOS. `AppleSpeechTranscriber` holds the framework calls.
enum AppleSpeech {
    nonisolated static let alias = "apple-speech"
    nonisolated static let family = "apple_speech"
    /// The catalog `backend` string `AudioRuntimeProvider` routes on.
    nonisolated static let backend = "swift_apple_speech"
    /// Oldest macOS with `SpeechAnalyzer`. Older Macs get no row at all.
    /// `SFSpeechRecognizer` is not a fallback: it needs a separate privacy
    /// grant and can send audio to Apple's servers.
    nonisolated static let minimumMacOSMajor = 26

    /// Whether this Mac can run the engine. `frameworkAvailable` is
    /// `SpeechTranscriber.isAvailable`, passed in so tests and this file stay
    /// free of the framework.
    nonisolated static func isSupported(osMajor: Int, frameworkAvailable: () -> Bool) -> Bool {
        osMajor >= minimumMacOSMajor && frameworkAvailable()
    }

    nonisolated static func isBuiltIn(_ alias: String?) -> Bool { alias == Self.alias }

    /// The row the workspace adds to the catalog's profiles. Rust `canRun` is
    /// false on purpose: the Rust worker cannot run it, so draft and Rust-only
    /// paths skip it, and `AudioRuntimeProvider` gives the real verdict.
    nonisolated static func profile(displayName: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: alias, repository: "", revision: "", assetFingerprint: ""),
            family: family, displayName: displayName,
            capabilities: AudioCapabilities(operations: ["transcribe"], timing: "word", backend: backend,
                                            cancellation: "window", canRun: false),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1),
            // Not measured against this repo's WER gates, so the picker marks
            // it experimental (`readiness != "qualified"`).
            readiness: "implemented_unqualified")
    }

    /// One finalized recognizer result: its text, its extent, and the words
    /// inside it, all in seconds from the start of the PCM it heard.
    struct Piece: Equatable, Sendable {
        var text: String
        var start: Double
        var end: Double
        var words: [AudioWord]
    }

    /// A window's pieces as the transcript the workspace saves. Only pieces
    /// that start inside the part of the recording the window owns are kept,
    /// so overlapping windows do not repeat a sentence. Word times stay
    /// window-relative, as the shared window loop expects.
    nonisolated static func transcript(_ pieces: [Piece], in window: AudioTranscriptionWindow, language: String?) -> AudioTranscript {
        let owned = pieces.filter {
            let start = window.start + $0.start
            return start >= window.ownedFrom && start < window.ownedUntil
        }
        let segments: [AudioSegment] = owned.compactMap { piece in
            let text = piece.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !text.isEmpty else { return nil }
            // Word timing only when every word has it: a partial list would
            // misplace speaker labels that read word ranges.
            let words = piece.words.isEmpty ? nil : piece.words
            return AudioSegment(startSeconds: piece.start, endSeconds: max(piece.start, piece.end), text: text,
                                timing: words == nil ? "segment" : "word", words: words)
        }
        return AudioTranscript(text: segments.map(\.text).joined(separator: " "), language: language, segments: segments)
    }
}
