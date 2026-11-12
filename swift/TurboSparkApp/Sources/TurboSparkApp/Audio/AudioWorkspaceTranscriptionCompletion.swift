import Foundation

extension AudioWorkspaceController {
    /// Finishes a final-pass transcription the way the Rust session path in
    /// `beginRun` does: speaker labels are extra and best effort, then the
    /// take is marked complete and the status strip reports the outcome. The
    /// FluidAudio path calls this too, so no engine skips the labels a mode
    /// asked for.
    func completeFinalTranscription(item: AudioLibraryItem, into id: UUID, transcription: Transcription, token: UUID) async throws {
        let labelFailure = try await labelSpeakersBestEffort(
            item: item, into: id, revision: transcription.revision, words: transcription.words, token: token)
        guard valid(token) else { return }
        try persistMutation(id) { $0.status = .completed; $0.updatedAt = Date() }
        if let labelFailure {
            status = AppStrings.localized("Transcript saved. Speakers could not be labeled: \(labelFailure)")
        } else {
            status = AppStrings.localized("Saved in this profile")
        }
    }
}
