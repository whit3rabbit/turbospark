import Foundation
import UniformTypeIdentifiers

/// A reference a past clone take used, offered again in the voice picker.
/// Its clips are pinned by the take, so it survives deleting the source.
struct AudioRecentVoiceReference: Identifiable, Equatable, Sendable {
    struct Key: Hashable, Sendable { let sourceID: UUID; let start: Double; let duration: Double }
    let id: Key
    let take: AudioLibraryItem
    let reference: AudioVoiceReference
    let referenceText: String
    var usedAt: Date { take.createdAt }
}

/// Seams that serve the Voice Cloning page: choosing, recording, reusing and
/// drawing the reference voice. Consent and the run request are untouched.
extension AudioWorkspaceController {
    /// A reference recording stops here and becomes the reference. Matches
    /// the longest reference `AudioVoiceReference.isValid` accepts.
    static let voiceReferenceRecordingLimit: Double = 30

    /// With no cloning model selected the recipe can still hold another
    /// mode's speech settings, so a reference set now would land there.
    var canEditVoiceReference: Bool {
        speechSettings.mode == .clone && !isBusy && !hasRecordingActivity && !isShutDown
    }

    var isRecordingVoiceReference: Bool {
        voiceReferenceRecordingID != nil && voiceReferenceRecordingID == recordingID
    }

    func startVoiceReferenceRecording() {
        guard page == .voiceClone, canEditVoiceReference else { return }
        startRecording(asVoiceReference: true)
    }

    func enforceVoiceReferenceRecordingLimit() {
        guard isRecording, isRecordingVoiceReference, recordingSeconds >= Self.voiceReferenceRecordingLimit else { return }
        stopRecording()
    }

    /// Called by `stopRecording` once a reference recording finished with
    /// audio. Skipped when the user left the page or a run took the recipe.
    func attachRecordedVoiceReference(_ id: UUID) {
        guard page == .voiceClone, speechSettings.mode == .clone, activeGeneration == nil,
              let item = items.first(where: { $0.id == id }), item.status == .completed, item.hasPlayableAudio else { return }
        assignVoiceReference(item)
    }

    /// Restores the reference region and transcript a take was generated
    /// from. Consent and clone options stay as they are: consent covers the
    /// region the user confirmed on this page, never one copied from history.
    func useVoiceReference(fromTake take: AudioLibraryItem) {
        guard !isBusy, !hasRecordingActivity, !isShutDown,
              let saved = take.recipe.speech, let reference = saved.reference else { return }
        stopReferencePreview()
        var settings = speechSettings
        settings.reference = reference
        settings.referenceText = saved.referenceText
        speechSettings = settings
        flushPreferences()
    }

    /// Normalized peaks of the trimmed region, for the reference waveform.
    func referencePeaks(for reference: AudioVoiceReference, count: Int) async throws -> [Float] {
        let assets = self.assets; let lifetime = self.lifetime
        let drawing = AudioOperationLifetime()
        return try await withTaskCancellationHandler {
            try await Task.detached {
                // 16 kHz is plenty for a picture, and accepts the sub-second
                // trims the editor can pass through on the way to a valid one.
                let samples = try AudioMediaIO.readMono16k(
                    reference.clips, store: assets, start: reference.start, duration: min(30, reference.duration),
                    cancel: { try lifetime.check(); try drawing.check() })
                return Self.peaks(samples, count: count)
            }.value
        } onCancel: {
            drawing.end()
        }
    }

    /// Absolute peak per bucket, scaled so the loudest bucket is 1.
    nonisolated static func peaks(_ samples: [Float], count: Int) -> [Float] {
        guard count > 0, !samples.isEmpty else { return Array(repeating: 0, count: max(0, count)) }
        let buckets = min(count, samples.count)
        var result = [Float](repeating: 0, count: buckets)
        for index in 0..<buckets {
            let lower = index * samples.count / buckets
            let upper = max(lower + 1, (index + 1) * samples.count / buckets)
            var peak: Float = 0
            for sample in samples[lower..<upper] where sample.isFinite { peak = max(peak, abs(sample)) }
            result[index] = peak
        }
        guard let loudest = result.max(), loudest > 0 else { return result }
        return result.map { $0 / loudest }
    }

    /// The same filter as the import panel: a local file macOS reads as audio.
    nonisolated static func acceptsAudioFile(_ url: URL) -> Bool {
        guard url.isFileURL else { return false }
        let type = (try? url.resourceValues(forKeys: [.contentTypeKey]).contentType)
            ?? UTType(filenameExtension: url.pathExtension)
        return type?.conforms(to: .audio) == true
    }

    var recentVoiceReferences: [AudioRecentVoiceReference] { Self.recentVoiceReferences(in: items) }

    /// Valid references of completed clone takes, newest first, one per
    /// source region.
    nonisolated static func recentVoiceReferences(in items: [AudioLibraryItem], limit: Int = 8) -> [AudioRecentVoiceReference] {
        var seen = Set<AudioRecentVoiceReference.Key>()
        var result: [AudioRecentVoiceReference] = []
        for take in items.sorted(by: { $0.createdAt > $1.createdAt }) where result.count < limit {
            guard take.kind == .voiceover, take.status == .completed, let speech = take.recipe.speech,
                  speech.mode == .clone, let reference = speech.reference, reference.isValid else { continue }
            let key = AudioRecentVoiceReference.Key(sourceID: reference.sourceID, start: reference.start, duration: reference.duration)
            guard seen.insert(key).inserted else { continue }
            result.append(AudioRecentVoiceReference(id: key, take: take, reference: reference, referenceText: speech.referenceText))
        }
        return result
    }

    /// Library audio that can become a reference, favorites first and then
    /// newest, matching the library list. Never touches the library search.
    nonisolated static func voiceReferenceCandidates(in items: [AudioLibraryItem], query: String, favoritesOnly: Bool) -> [AudioLibraryItem] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        return items.filter {
            $0.hasPlayableAudio && $0.status == .completed && (!favoritesOnly || $0.favorite)
                && (query.isEmpty || $0.title.localizedCaseInsensitiveContains(query)
                    || $0.collection.localizedCaseInsensitiveContains(query))
        }.sorted { $0.favorite == $1.favorite ? $0.createdAt > $1.createdAt : $0.favorite }
    }
}
