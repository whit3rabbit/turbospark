import Foundation

/// Speaker labels for a take longer than the diarization model accepts at
/// once. The take is labeled in parts that overlap, and each part's voices
/// are matched to the voices before it where both parts labeled the same
/// audio. The engine reports no voice embeddings, so the overlap is the
/// only evidence. A voice matches only when most of its overlap time agrees
/// with one earlier voice; any other voice, including one silent in the
/// overlap, starts a new speaker. A miss shows as an extra speaker to merge
/// by hand, never as two people joined.
enum AudioSpeakerParts {
    struct Part: Equatable, Sendable {
        var start: Double
        var end: Double
        var duration: Double { end - start }
    }

    /// Seconds two neighboring parts share.
    static let overlap: Double = 300
    /// Seconds a voice must agree with an earlier one to take its label.
    static let minimumAgreement: Double = 8
    /// The share of a voice's overlap time that must agree with one earlier
    /// voice. Below it the voice is split between voices, so it stays new.
    static let dominance: Double = 0.7

    /// Parts of at most `limit` seconds covering `duration`, sharing
    /// `overlap` seconds with each neighbor and starting on whole seconds, so
    /// both parts window the shared audio on the same grid. One part when the
    /// take fits; none for an empty or invalid take.
    static func plan(duration: Double, limit: Double, overlap: Double = overlap) -> [Part] {
        guard duration.isFinite, duration > 0, limit.isFinite, limit > 0 else { return [] }
        guard duration > limit else { return [Part(start: 0, end: duration)] }
        // Whole seconds, so rounding each step up never makes a part longer
        // than the limit.
        let stride = floor(limit - overlap)
        guard stride >= 1 else { return [] }
        let count = Int(ceil((duration - overlap) / stride))
        let step = ceil((duration - overlap) / Double(count))
        return (0..<count).map { index in
            let start = Double(index) * step
            return Part(start: start, end: index == count - 1 ? duration : min(duration, start + step + overlap))
        }
    }

    /// One timeline from each part's turns (on the take's timeline, speakers
    /// as the runtime numbered them in that part). Each shared span is cut at
    /// its middle: the earlier part's turns before, the later part's after.
    /// Speakers come back as `S1`, `S2`, ... in order of first appearance.
    static func join(_ parts: [(part: Part, turns: [AudioSpeakerLabels.Turn])],
                     minimumAgreement: Double = minimumAgreement,
                     dominance: Double = dominance) -> [AudioSpeakerLabels.Turn] {
        var joined: [AudioSpeakerLabels.Turn] = []
        var previous: (part: Part, turns: [AudioSpeakerLabels.Turn])?
        var created = 0
        func newSpeaker() -> String { created += 1; return "G\(created)" }
        for (part, turns) in parts {
            var labels: [String: String] = [:]
            let order = firstAppearance(turns)
            if let previous {
                let shared = part.start...previous.part.end
                for speaker in order {
                    let mine = turns.filter { $0.speaker == speaker }
                    let own = mine.reduce(0) { $0 + length($1, within: shared) }
                    var agreement: [String: Double] = [:]
                    for turn in mine {
                        for earlier in previous.turns {
                            let seconds = length(turn, earlier, within: shared)
                            if seconds > 0 { agreement[earlier.speaker, default: 0] += seconds }
                        }
                    }
                    if let best = agreement.max(by: { $0.value != $1.value ? $0.value < $1.value : $0.key > $1.key }),
                       best.value >= minimumAgreement, best.value >= dominance * own {
                        labels[speaker] = best.key
                    } else {
                        labels[speaker] = newSpeaker()
                    }
                }
            } else {
                for speaker in order { labels[speaker] = newSpeaker() }
            }
            let relabeled = turns.compactMap { turn -> AudioSpeakerLabels.Turn? in
                guard let label = labels[turn.speaker] else { return nil }
                return AudioSpeakerLabels.Turn(speaker: label, start: turn.start, end: turn.end)
            }
            if let previous {
                let cut = (part.start + previous.part.end) / 2
                joined = joined.compactMap { clip($0, to: -Double.infinity...cut) }
                joined += relabeled.compactMap { clip($0, to: cut...Double.infinity) }
            } else {
                joined = relabeled
            }
            previous = (part, relabeled)
        }
        joined.sort { $0.start < $1.start }
        var numbers: [String: String] = [:]
        for turn in joined where numbers[turn.speaker] == nil { numbers[turn.speaker] = "S\(numbers.count + 1)" }
        return joined.map { AudioSpeakerLabels.Turn(speaker: numbers[$0.speaker] ?? $0.speaker, start: $0.start, end: $0.end) }
    }

    private static func firstAppearance(_ turns: [AudioSpeakerLabels.Turn]) -> [String] {
        var seen: Set<String> = []
        return turns.sorted { $0.start < $1.start }.map(\.speaker).filter { seen.insert($0).inserted }
    }

    private static func length(_ turn: AudioSpeakerLabels.Turn, within span: ClosedRange<Double>) -> Double {
        max(0, min(turn.end, span.upperBound) - max(turn.start, span.lowerBound))
    }

    private static func length(_ a: AudioSpeakerLabels.Turn, _ b: AudioSpeakerLabels.Turn, within span: ClosedRange<Double>) -> Double {
        max(0, min(a.end, b.end, span.upperBound) - max(a.start, b.start, span.lowerBound))
    }

    private static func clip(_ turn: AudioSpeakerLabels.Turn, to span: ClosedRange<Double>) -> AudioSpeakerLabels.Turn? {
        let start = max(turn.start, span.lowerBound), end = min(turn.end, span.upperBound)
        return end > start ? AudioSpeakerLabels.Turn(speaker: turn.speaker, start: start, end: end) : nil
    }
}
