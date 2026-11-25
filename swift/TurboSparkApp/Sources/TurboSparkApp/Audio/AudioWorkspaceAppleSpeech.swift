import Foundation
import TurboSpark

extension AudioWorkspaceController {
    /// The built-in rows this Mac adds to the catalog's profiles. Empty before
    /// macOS 26, so older Macs see exactly the catalog.
    static var builtInAudioProfiles: [AudioProfile] {
        AppleSpeech.isSupportedOnThisMac
            ? [AppleSpeech.profile(displayName: AppStrings.localized("Apple Speech (built-in)"))]
            : []
    }

    /// Transcribes a saved recording with the macOS recognizer through the
    /// workspace's window loop. The recognizer runs in a system process, so
    /// it never replaces the resident audio model or counts against its
    /// memory budget. On first use macOS downloads the language model,
    /// which only happens here, inside a run the user started.
    @discardableResult func transcribeWithAppleSpeech(item: AudioLibraryItem, into id: UUID,
                                                      source: AudioTranscriptRevision.Source, token: UUID) async throws -> Transcription {
        guard #available(macOS 26, *), AppleSpeech.isSupportedOnThisMac else { throw AppleSpeechError.unavailable }
        let locale = try await AppleSpeechTranscriber.locale()
        if await AppleSpeechTranscriber.needsAssets(locale) {
            status = AppStrings.localized("Getting the Apple speech model from macOS")
            progress = nil
            try await AppleSpeechTranscriber.installAssets(locale) { [weak self] fraction in
                Task { @MainActor in
                    guard let self, self.valid(token) else { return }
                    self.progress = fraction
                }
            }
            guard valid(token) else { throw CancellationError() }
        }
        // Long windows: the recognizer keeps its own context and gains
        // nothing from short ones. The 2 s overlap lets a sentence cut at a
        // window edge be heard whole by the next window.
        let windows = AudioTranscriptionPlan.windows(duration: item.duration, length: 60, overlap: 2, skipsCoveredTail: true)
        let language = locale.language.languageCode?.identifier
        return try await transcribe(item: item, into: id, windows: windows, source: source, token: token) { pcm, window in
            AppleSpeech.transcript(try await AppleSpeechTranscriber.transcribe(pcm, locale: locale), in: window, language: language)
        }
    }
}
