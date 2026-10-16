import Foundation

/// What a take on the Record page is for. Picking one sets the sources and
/// live text, and files new takes into its collection. Named apart from
/// `AudioPreset`, which saves a whole recipe.
enum AudioRecordingPreset: String, CaseIterable, Identifiable, Sendable {
    case voiceNote, meeting, sound
    var id: String { rawValue }
    var titleKey: String {
        switch self {
        case .voiceNote: return "Voice note"
        case .meeting: return "Meeting or call"
        case .sound: return "Sound"
        }
    }
    var symbol: String {
        switch self {
        case .voiceNote: return "mic"
        case .meeting: return "person.2.wave.2"
        case .sound: return "waveform"
        }
    }
    var capturesMicrophone: Bool { true }
    /// A call's other side comes from the app playing it.
    var capturesSystemAudio: Bool { self == .meeting }
    /// A sound recording has no speech to read.
    var liveText: Bool { self != .sound }
    /// The name a new take starts with, before its date.
    var takeTitle: String {
        switch self {
        case .voiceNote: return AppStrings.localized("Voice note")
        case .meeting: return AppStrings.localized("Meeting")
        case .sound: return AppStrings.localized("Sound")
        }
    }
    /// The collection name used the first time, in the current language.
    var defaultCollection: String {
        switch self {
        case .voiceNote: return AppStrings.localized("Voice notes")
        case .meeting: return AppStrings.localized("Meetings")
        case .sound: return AppStrings.localized("Sounds")
        }
    }
    func matches(microphone: Bool, systemAudio: Bool) -> Bool {
        capturesMicrophone == microphone && capturesSystemAudio == systemAudio
    }
}

enum AudioLibraryFilter: String, CaseIterable, Identifiable, Sendable {
    case all, favorites, recordings, imported, needsTranscript
    var id: String { rawValue }
    var titleKey: String {
        switch self {
        case .all: return "All"
        case .favorites: return "Favorites"
        case .recordings: return "Recordings"
        case .imported: return "Imported"
        case .needsTranscript: return "Needs transcript"
        }
    }
    func includes(_ item: AudioLibraryItem) -> Bool {
        switch self {
        case .all: return true
        case .favorites: return item.favorite
        case .recordings: return item.kind == .recording
        case .imported: return item.kind == .imported
        case .needsTranscript:
            // Audio someone would read, still without a transcript. A sound
            // recording has no speech, and a take still recording is not done.
            return (item.kind == .recording || item.kind == .imported) && item.hasPlayableAudio
                && item.preferredTranscript == nil && item.status != .recording
                && item.recordingPreset != AudioRecordingPreset.sound.rawValue
        }
    }
}

enum AudioDateGroup: Int, CaseIterable, Identifiable, Sendable {
    case today, thisWeek, earlier
    var id: Int { rawValue }
    var titleKey: String {
        switch self {
        case .today: return "Today"
        case .thisWeek: return "This week"
        case .earlier: return "Earlier"
        }
    }
    /// "This week" is the current calendar week, today excluded, in the
    /// user's calendar (its first weekday).
    static func group(_ date: Date, now: Date, calendar: Calendar) -> AudioDateGroup {
        if calendar.isDate(date, inSameDayAs: now) { return .today }
        if calendar.isDate(date, equalTo: now, toGranularity: .weekOfYear), date < now { return .thisWeek }
        return .earlier
    }
}

struct AudioFacet: Identifiable, Equatable, Sendable {
    let name: String
    let count: Int
    var id: String { name }
}

enum AudioLibraryOrganizer {
    static let maxTags = 20
    static let maxTagLength = 40

    /// Trimmed, single-spaced, deduplicated ignoring case (the first
    /// spelling wins), at most `maxTags` of at most `maxTagLength` characters.
    static func normalizedTags(_ raw: [String]) -> [String] {
        var seen = Set<String>()
        var out: [String] = []
        for tag in raw {
            let cleaned = tag.split(whereSeparator: \.isWhitespace).joined(separator: " ")
            guard !cleaned.isEmpty, cleaned.count <= maxTagLength, out.count < maxTags else { continue }
            if seen.insert(cleaned.lowercased()).inserted { out.append(cleaned) }
        }
        return out
    }

    static func filter(_ items: [AudioLibraryItem], by filter: AudioLibraryFilter, collection: String?, tag: String?) -> [AudioLibraryItem] {
        items.filter { item in
            filter.includes(item)
                && (collection.map { item.collection == $0 } ?? true)
                && (tag.map { wanted in item.tagList.contains { $0.caseInsensitiveCompare(wanted) == .orderedSame } } ?? true)
        }
    }

    /// Newest first within each group; empty groups are left out.
    static func grouped(_ items: [AudioLibraryItem], now: Date = Date(), calendar: Calendar = .current) -> [(group: AudioDateGroup, items: [AudioLibraryItem])] {
        let byGroup = Dictionary(grouping: items) { AudioDateGroup.group($0.createdAt, now: now, calendar: calendar) }
        return AudioDateGroup.allCases.compactMap { group in
            guard let members = byGroup[group], !members.isEmpty else { return nil }
            return (group, members.sorted { $0.createdAt > $1.createdAt })
        }
    }

    static func collections(_ items: [AudioLibraryItem]) -> [AudioFacet] {
        facets(items.map(\.collection).filter { !$0.isEmpty })
    }

    /// Tags counted ignoring case, named by their first spelling.
    static func tags(_ items: [AudioLibraryItem]) -> [AudioFacet] {
        var names: [String: String] = [:]
        var counts: [String: Int] = [:]
        for tag in items.flatMap(\.tagList) {
            let key = tag.lowercased()
            names[key] = names[key] ?? tag
            counts[key, default: 0] += 1
        }
        return counts.map { AudioFacet(name: names[$0.key]!, count: $0.value) }
            .sorted { $0.count != $1.count ? $0.count > $1.count : $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }

    private static func facets(_ names: [String]) -> [AudioFacet] {
        Dictionary(grouping: names, by: { $0 }).map { AudioFacet(name: $0.key, count: $0.value.count) }
            .sorted { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
    }
}

extension AudioLibraryItem {
    mutating func setTags(_ tags: [String]) {
        let normalized = AudioLibraryOrganizer.normalizedTags(tags)
        self.tags = normalized.isEmpty ? nil : normalized
        updatedAt = Date()
    }
}
