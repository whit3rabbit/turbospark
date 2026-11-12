import Foundation

/// A take's speakers as its transcript shows them. A segment's voice is its
/// key: the diarization id, or for a take saved before ids, its label. Names
/// and merges are kept on the take by key, never written into segments, so
/// a rename reaches every segment of every revision and every export at
/// once, and undoing one only removes an entry.
struct AudioSpeakerRoster: Equatable, Sendable {
    var names: [String: String] = [:]
    var merges: [String: String] = [:]

    init(names: [String: String] = [:], merges: [String: String] = [:]) {
        self.names = names
        self.merges = merges
    }

    init(_ item: AudioLibraryItem) {
        self.init(names: item.speakerNames ?? [:], merges: item.speakerMerges ?? [:])
    }

    /// Writes the roster to `item`. Empty maps are stored as nil, so a take
    /// nobody renamed stays as it was saved.
    func apply(to item: inout AudioLibraryItem) {
        item.speakerNames = names.isEmpty ? nil : names
        item.speakerMerges = merges.isEmpty ? nil : merges
    }

    static func key(_ segment: AudioLibrarySegment) -> String? {
        if let id = segment.speakerID, !id.isEmpty { return id }
        if let label = segment.speaker, !label.isEmpty { return label }
        return nil
    }

    /// "Speaker 2" for a diarization id, the saved label for any other key.
    static func defaultName(_ key: String) -> String { AudioSpeakerLabels.displayName(key) }

    /// The voice `key` shows as: the end of its merges. A cycle, which only a
    /// damaged record holds, leaves the key as itself.
    func root(_ key: String) -> String {
        var current = key
        var seen: Set<String> = [key]
        while let next = merges[current] {
            guard seen.insert(next).inserted else { return key }
            current = next
        }
        return current
    }

    func isMerged(_ key: String) -> Bool { root(key) != key }

    func name(_ key: String) -> String {
        let root = root(key)
        return names[root] ?? Self.defaultName(root)
    }

    func label(_ segment: AudioLibrarySegment) -> String? { Self.key(segment).map(name) }

    /// Segments with the label each one shows in `speaker`, for exports and
    /// prompts. The raw id stays in `speakerID`.
    func labeled(_ segments: [AudioLibrarySegment]) -> [AudioLibrarySegment] {
        segments.map { segment in
            var labeled = segment
            labeled.speaker = label(segment)
            return labeled
        }
    }

    /// Names the voice `key` itself, merged or not: a merged voice's name
    /// shows again once it is separated. An empty name, or one equal to the
    /// default label, removes the entry so the default comes back.
    mutating func rename(_ key: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == Self.defaultName(key) { names.removeValue(forKey: key) }
        else { names[key] = trimmed }
    }

    /// Shows `key` as `target`. Refused for the same voice and for a target
    /// that already shows as `key`, which would make a cycle. The merged
    /// voice keeps its own name entry, so separating it restores that too.
    @discardableResult mutating func merge(_ key: String, into target: String) -> Bool {
        var current = target
        var seen: Set<String> = []
        while true {
            if current == key { return false }
            guard seen.insert(current).inserted, let next = merges[current] else { break }
            current = next
        }
        merges[key] = target
        return true
    }

    mutating func separate(_ key: String) { merges.removeValue(forKey: key) }

    /// Each key in `segments`, in order of first appearance.
    static func keys(in segments: [AudioLibrarySegment]) -> [String] {
        var seen: Set<String> = []
        return segments.compactMap(key).filter { seen.insert($0).inserted }
    }

    /// The voices a transcript shows, in order of first appearance: each
    /// one others are merged into, then each merged voice.
    func roots(in segments: [AudioLibrarySegment]) -> [String] {
        var seen: Set<String> = []
        return Self.keys(in: segments).map(root).filter { seen.insert($0).inserted }
    }

    /// The id a new voice gets: one above every `S<n>` the take uses,
    /// including `segments` the editor has not saved yet.
    static func nextID(in item: AudioLibraryItem, also segments: [AudioLibrarySegment] = []) -> String {
        var keys = (item.transcripts.flatMap(\.segments) + segments).compactMap(key)
        keys += (item.speakerNames ?? [:]).keys
        for (key, target) in item.speakerMerges ?? [:] { keys += [key, target] }
        let highest = keys.compactMap { key -> Int? in
            guard key.first == "S" else { return nil }
            return Int(key.dropFirst())
        }.max() ?? 0
        return "S\(highest + 1)"
    }

    /// `segment` reassigned to the voice `key`, or to none. `speaker` takes
    /// the default label, as speaker labeling writes it.
    static func assigning(_ segment: AudioLibrarySegment, to key: String?) -> AudioLibrarySegment {
        var assigned = segment
        assigned.speakerID = key
        assigned.speaker = key.map(defaultName)
        return assigned
    }
}

/// One voice in the Name the Speakers sheet.
struct AudioSpeakerSummary: Identifiable, Equatable, Sendable {
    var id: String
    /// The voice this one is merged into, by key.
    var mergedInto: String?
    /// Seconds of speech: a voice's own segments plus every voice merged
    /// into it, or its own for a merged voice.
    var talkTime: Double
    /// The text of the voice's longest segment.
    var quote: String?
    /// A few seconds from the start of its longest timed segment.
    var sample: ClosedRange<Double>?

    /// Voices in order of first appearance, each merged voice after the
    /// voices that are shown. `sampleSeconds` bounds every sample.
    static func summaries(_ segments: [AudioLibrarySegment], roster: AudioSpeakerRoster, sampleSeconds: Double = 6) -> [AudioSpeakerSummary] {
        let keys = AudioSpeakerRoster.keys(in: segments)
        var own: [String: [AudioLibrarySegment]] = [:]
        for segment in segments { if let key = AudioSpeakerRoster.key(segment) { own[key, default: []].append(segment) } }
        let seconds = { (list: [AudioLibrarySegment]) in list.reduce(0) { $0 + max(0, $1.end - $1.start) } }
        func summary(_ key: String) -> AudioSpeakerSummary {
            let merged = roster.isMerged(key)
            let counted = merged ? own[key] ?? [] : segments.filter { AudioSpeakerRoster.key($0).map(roster.root) == key }
            // A voice's own lines represent it best. One that only others
            // were merged into falls back to theirs.
            let mine = own[key] ?? counted
            let longest = { (list: [AudioLibrarySegment]) in list.max { ($0.end - $0.start) < ($1.end - $1.start) } }
            let quoted = longest(mine.filter { !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty })
            let timed = longest(mine.filter { $0.end > $0.start && $0.start.isFinite && $0.end.isFinite })
            return AudioSpeakerSummary(
                id: key, mergedInto: merged ? roster.root(key) : nil, talkTime: seconds(counted),
                quote: quoted?.text.trimmingCharacters(in: .whitespacesAndNewlines),
                sample: timed.map { $0.start...min($0.end, $0.start + sampleSeconds) })
        }
        let shown = roster.roots(in: segments).map(summary)
        let mergedKeys = keys.filter(roster.isMerged)
        return shown + mergedKeys.map(summary)
    }
}

extension AudioTranscriptExport {
    /// What a transcript export writes for `item`: its preferred transcript
    /// with the take's speaker names. JSON keeps each raw id beside the name.
    static func data(_ item: AudioLibraryItem, format: AudioTranscriptExportFormat) throws -> Data? {
        guard let transcript = item.preferredTranscript else { return nil }
        let roster = AudioSpeakerRoster(item)
        let segments = roster.labeled(transcript.segments)
        switch format {
        case .text: return Data(text(segments).utf8)
        case .srt: return Data(srt(segments).utf8)
        case .vtt: return Data(vtt(segments).utf8)
        case .json:
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            var portable = item
            portable.localModelBookmark = nil
            for index in portable.transcripts.indices {
                portable.transcripts[index].segments = roster.labeled(portable.transcripts[index].segments)
            }
            return try encoder.encode(portable)
        }
    }
}
