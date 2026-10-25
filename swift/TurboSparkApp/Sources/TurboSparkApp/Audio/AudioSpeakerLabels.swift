import AVFoundation
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

    /// The turns of a diarization result with the runtime's ids, moved by
    /// `offset` seconds onto the take's timeline.
    static func turns(_ transcript: AudioTranscript, offset: Double = 0) -> [Turn] {
        transcript.segments.compactMap { segment in
            guard let speaker = segment.speaker, let start = segment.startSeconds,
                  let end = segment.endSeconds, end > start else { return nil }
            return Turn(speaker: speaker, start: start + offset, end: end + offset)
        }
    }

    /// Fills in each unlabeled segment's speaker. A segment's words vote with
    /// the seconds each turn overlaps them; a segment without word timings
    /// votes with its own extent. A segment no turn overlaps stays unlabeled,
    /// and a label already set (a manual one) is kept. The winning id goes to
    /// `speakerID` and its display name to `speaker`.
    static func assign(_ segments: [AudioLibrarySegment], words: [ClosedRange<Double>], turns: [Turn]) -> [AudioLibrarySegment] {
        guard !turns.isEmpty else { return segments }
        // First appearance breaks ties, so a tie never depends on hashing.
        var order: [String: Int] = [:]
        for turn in turns where order[turn.speaker] == nil { order[turn.speaker] = order.count }
        return segments.map { segment in
            guard segment.speaker == nil, segment.speakerID == nil, segment.end > segment.start else { return segment }
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
            return AudioSpeakerRoster.assigning(segment, to: best.key)
        }
    }
}

extension AudioMediaIO {
    /// A whole take for speaker diarization. Every other inference reads one
    /// bounded window, but diarization clusters voices across the recording,
    /// and the native job holds all of its PCM anyway. So this reads bounded
    /// 30 s blocks and joins them, up to `limit` seconds (the model's
    /// 30-minute input is 115 MB of float samples), from `start` on.
    static func readMono16kTake(_ clips: [AudioLibraryClip], store: ManagedAssetStore, start: Double = 0, duration: Double, limit: Double, cancel: () throws -> Void = {}) throws -> [Float] {
        guard start.isFinite, start >= 0, duration.isFinite, duration > 0, duration <= limit else { throw MediaError.tooLong }
        var output: [Float] = []
        output.reserveCapacity(Int(ceil(duration * 16_000)))
        var read = 0.0
        while read < duration {
            try cancel()
            output += try readMono16k(clips, store: store, start: start + read, duration: min(30, duration - read), cancel: cancel)
            read += 30
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
    /// Does nothing when no diarization model is installed. A take longer
    /// than the model accepts is labeled in overlapping parts whose voices
    /// are joined (`AudioSpeakerParts`). The labels land only on that
    /// revision, so an edit made meanwhile (a newer user revision) is never
    /// touched.
    func labelSpeakers(item: AudioLibraryItem, into id: UUID, revision: UUID, words: [ClosedRange<Double>], token: UUID) async throws {
        guard let profile = diarizationProfile(for: item) else { return }
        let limit = Double(profile.capabilities.maxInputSeconds ?? 1800)
        let parts = AudioSpeakerParts.plan(duration: item.duration, limit: limit)
        guard !parts.isEmpty else { return }
        status = AppStrings.localized("Labeling speakers")
        progress = nil
        let session = try await residentSession(for: profile)
        let assets = self.assets; let lifetime = self.lifetime
        var labeled: [(part: AudioSpeakerParts.Part, turns: [AudioSpeakerLabels.Turn])] = []
        for (index, part) in parts.enumerated() {
            // One part's samples at a time: each is up to the model's limit.
            let pcm = try await Task.detached {
                try AudioMediaIO.readMono16kTake(item.clips, store: assets, start: part.start, duration: part.duration, limit: limit, cancel: lifetime.check)
            }.value
            let result = try await session.execute(AudioRequest(task: .diarization, language: nil), pcm: pcm) { [weak self] update in
                Task { @MainActor in
                    guard let self, self.valid(token), let total = update.total, total > 0 else { return }
                    self.progress = min(1, (Double(index) + Double(update.completed) / Double(total)) / Double(parts.count))
                }
            }
            try Task.checkCancellation()
            guard valid(token), let transcript = result.transcript else { return }
            labeled.append((part, AudioSpeakerLabels.turns(transcript, offset: part.start)))
        }
        let turns = labeled.count == 1 ? labeled[0].turns : AudioSpeakerParts.join(labeled)
        let voiceprints = AudioSpeakerMemory.extractVoiceprints(for: item, turns: turns, store: assets)
        try persistMutation(id) { current in
            guard let index = current.transcripts.firstIndex(where: { $0.id == revision }) else { return }
            current.transcripts[index].segments = AudioSpeakerLabels.assign(current.transcripts[index].segments, words: words, turns: turns)
            current.speakerLabelParts = parts.count > 1 ? parts.count : nil
            current.speakerVoiceprints = voiceprints.isEmpty ? nil : voiceprints
            current.updatedAt = Date()
        }
    }

    /// Whether a finished final pass asks for names: a Meeting take whose
    /// `revision` got speaker labels, while its pane is still showing. The
    /// transcript's Name Speakers button opens the same sheet any time.
    func offersSpeakerNaming(_ id: UUID, revision: UUID) -> Bool {
        guard page == .transcribe, speechToTextMode == .meeting, !hasRecordingActivity,
              let item = items.first(where: { $0.id == id }), speechToTextMode(for: item) == .meeting,
              let labeled = item.transcripts.first(where: { $0.id == revision }) else { return false }
        return labeled.segments.contains { $0.speakerID != nil }
    }

    /// Saves the names and merges of take `id`.
    func applySpeakerRoster(_ roster: AudioSpeakerRoster, to id: UUID) {
        update(id) { roster.apply(to: &$0); $0.updatedAt = Date() }
    }

    /// Plays a few seconds of one voice through the preview player, or
    /// stops it when that voice is already playing. Reads only `range`, so a
    /// long take is never rendered whole for a sample.
    func playSpeakerSample(_ item: AudioLibraryItem, key: String, range: ClosedRange<Double>) {
        if speakerSampleKey == key, referencePlaying || referencePreviewTask != nil { stopReferencePreview(); return }
        guard !item.clips.isEmpty, !hasRecordingActivity, !isShutDown, range.upperBound > range.lowerBound else { return }
        stopPlayback(); stopReferencePreview()
        speakerSampleKey = key
        let assets = self.assets; let lifetime = self.lifetime; let token = epoch
        let duration = min(30, range.upperBound - range.lowerBound)
        referencePreviewTask = Task { [weak self] in
            do {
                let bytes = try await Task.detached {
                    let samples = try AudioMediaIO.readMono24k(item.clips, store: assets, start: range.lowerBound, duration: duration, cancel: lifetime.check)
                    return try AudioMediaIO.wav(samples: samples, sampleRate: 24_000, channels: 1)
                }.value
                try Task.checkCancellation()
                guard let self, self.valid(token), !self.hasRecordingActivity, self.speakerSampleKey == key else { return }
                let player = try AVAudioPlayer(data: bytes)
                self.referencePlayer = player; self.referencePlaying = player.play(); self.referencePreviewTask = nil
            } catch {
                guard let self, self.valid(token), !Task.isCancelled else { return }
                self.referencePreviewTask = nil; self.speakerSampleKey = nil; self.error = error.localizedDescription
            }
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
