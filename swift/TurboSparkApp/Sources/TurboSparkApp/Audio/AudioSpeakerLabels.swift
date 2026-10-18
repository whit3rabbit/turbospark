import Foundation
import TurboSpark

/// Speaker labels for a finished take. The diarization model reports who
/// spoke when as turns; each transcript segment takes the speaker whose
/// turns cover most of its words.
enum AudioSpeakerLabels {
    struct Turn: Equatable, Sendable {
        var speaker: String
        var start: Double
        var end: Double
    }

    /// The runtime names speakers `S1`, `S2`, ... in order of first
    /// appearance. The app shows them as "Speaker 1" in its language; any
    /// other id passes through unchanged.
    static func displayName(_ id: String) -> String {
        guard id.first == "S", let number = Int(id.dropFirst()), number > 0 else { return id }
        return AppStrings.localized("Speaker \(number)")
    }

    /// The turns of a diarization result, with display names.
    static func turns(_ transcript: AudioTranscript) -> [Turn] {
        transcript.segments.compactMap { segment in
            guard let speaker = segment.speaker, let start = segment.startSeconds,
                  let end = segment.endSeconds, end > start else { return nil }
            return Turn(speaker: displayName(speaker), start: start, end: end)
        }
    }

    /// Fills in each unlabeled segment's speaker. A segment's words vote with
    /// the seconds each turn overlaps them; a segment without word timings
    /// votes with its own extent. A segment no turn overlaps stays unlabeled,
    /// and a label already set (a manual one) is kept.
    static func assign(_ segments: [AudioLibrarySegment], words: [ClosedRange<Double>], turns: [Turn]) -> [AudioLibrarySegment] {
        guard !turns.isEmpty else { return segments }
        // First appearance breaks ties, so a tie never depends on hashing.
        var order: [String: Int] = [:]
        for turn in turns where order[turn.speaker] == nil { order[turn.speaker] = order.count }
        return segments.map { segment in
            guard segment.speaker == nil, segment.end > segment.start else { return segment }
            var spans = words.filter {
                let middle = ($0.lowerBound + $0.upperBound) / 2
                return middle >= segment.start && middle <= segment.end
            }
            if spans.isEmpty { spans = [segment.start...segment.end] }
            var votes: [String: Double] = [:]
            for span in spans {
                for turn in turns {
                    let overlap = min(span.upperBound, turn.end) - max(span.lowerBound, turn.start)
                    if overlap > 0 { votes[turn.speaker, default: 0] += overlap }
                }
            }
            guard let best = votes.max(by: { lhs, rhs in
                lhs.value != rhs.value ? lhs.value < rhs.value : order[lhs.key, default: 0] > order[rhs.key, default: 0]
            }) else { return segment }
            var labeled = segment
            labeled.speaker = best.key
            return labeled
        }
    }
}

extension AudioMediaIO {
    /// A whole take for speaker diarization. Every other inference reads one
    /// bounded window, but diarization clusters voices across the recording,
    /// and the native job holds all of its PCM anyway. So this reads bounded
    /// 30 s blocks and joins them, up to `limit` seconds (the model's
    /// 30-minute input is 115 MB of float samples).
    static func readMono16kTake(_ clips: [AudioLibraryClip], store: ManagedAssetStore, duration: Double, limit: Double, cancel: () throws -> Void = {}) throws -> [Float] {
        guard duration.isFinite, duration > 0, duration <= limit else { throw MediaError.tooLong }
        var output: [Float] = []
        output.reserveCapacity(Int(ceil(duration * 16_000)))
        var start = 0.0
        while start < duration {
            try cancel()
            output += try readMono16k(clips, store: store, start: start, duration: min(30, duration - start), cancel: cancel)
            start += 30
        }
        return output
    }
}

extension AudioWorkspaceController {
    /// An installed speaker diarization model this Mac can run, if any.
    var installedDiarizationProfile: AudioProfile? {
        profiles.first {
            $0.identity.task == .diarization && $0.capabilities.canRun
                && (installedPaths[$0.identity.alias] != nil || preferences.localModels[$0.identity.alias] != nil)
        }
    }

    /// The diarization model that labels an item, or nil to skip labels. A
    /// take or import that belongs to a Speech-to-Text mode is labeled only
    /// while that mode's switch is on, with the mode's own choice, and not at
    /// all while that model is not installed. Other takes (the Record page's
    /// Sound and hand-picked sources) keep the first installed model.
    func diarizationProfile(for item: AudioLibraryItem) -> AudioProfile? {
        guard let mode = speechToTextMode(for: item) else { return installedDiarizationProfile }
        guard labelsSpeakers(in: mode) else { return nil }
        guard let profile = speechToTextProfile(mode, .speakers), profile.capabilities.canRun, isInstalled(profile) else { return nil }
        return profile
    }

    /// Labels the speakers of the final transcript `revision` of take `id`.
    /// Does nothing when no diarization model is installed or the take is
    /// longer than the model accepts. The labels land only on that revision,
    /// so an edit made meanwhile (a newer user revision) is never touched.
    func labelSpeakers(item: AudioLibraryItem, into id: UUID, revision: UUID, words: [ClosedRange<Double>], token: UUID) async throws {
        guard let profile = diarizationProfile(for: item) else { return }
        let limit = Double(profile.capabilities.maxInputSeconds ?? 1800)
        guard item.duration <= limit else { return }
        status = AppStrings.localized("Labeling speakers")
        progress = nil
        let session = try await residentSession(for: profile)
        let assets = self.assets; let lifetime = self.lifetime
        let pcm = try await Task.detached { try AudioMediaIO.readMono16kTake(item.clips, store: assets, duration: item.duration, limit: limit, cancel: lifetime.check) }.value
        let result = try await session.execute(AudioRequest(task: .diarization, language: nil), pcm: pcm) { [weak self] update in
            Task { @MainActor in
                guard let self, self.valid(token), let total = update.total, total > 0 else { return }
                self.progress = min(1, Double(update.completed) / Double(total))
            }
        }
        try Task.checkCancellation()
        guard valid(token), let transcript = result.transcript else { return }
        let turns = AudioSpeakerLabels.turns(transcript)
        try persistMutation(id) { current in
            guard let index = current.transcripts.firstIndex(where: { $0.id == revision }) else { return }
            current.transcripts[index].segments = AudioSpeakerLabels.assign(current.transcripts[index].segments, words: words, turns: turns)
            current.updatedAt = Date()
        }
    }

    /// `labelSpeakers` once the transcript is saved, where labels are extra:
    /// a failure comes back as its message and leaves the transcript
    /// unlabeled. A cancel still throws. It reaches the native job, which
    /// reports it as its own error rather than CancellationError.
    func labelSpeakersBestEffort(item: AudioLibraryItem, into id: UUID, revision: UUID, words: [ClosedRange<Double>], token: UUID) async throws -> String? {
        do {
            try await labelSpeakers(item: item, into: id, revision: revision, words: words, token: token)
            return nil
        } catch {
            if error is CancellationError || Task.isCancelled { throw error }
            return error.localizedDescription
        }
    }
}
