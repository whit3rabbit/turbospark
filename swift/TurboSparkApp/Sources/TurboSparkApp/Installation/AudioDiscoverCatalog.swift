import Foundation
import TurboSpark

/// Pinned audio profiles for the jobs exposed by the audio workspaces.
enum AudioDiscoverCatalog {
    enum Group: Int, CaseIterable, Comparable, Identifiable {
        case liveText
        case finalTranscript
        case speakers
        case music
        case voiceCloning

        var id: Int { rawValue }

        var titleKey: String {
            switch self {
            case .liveText: return "Live Text"
            case .finalTranscript: return "Final Transcript"
            case .speakers: return "Speakers"
            case .music: return "Music"
            case .voiceCloning: return "Voice Cloning"
            }
        }

        static func < (lhs: Group, rhs: Group) -> Bool { lhs.rawValue < rhs.rawValue }
    }

    struct Entry: Identifiable, Equatable {
        var group: Group
        var profile: AudioProfile
        var id: String { profile.identity.alias }

        var installedEntry: AudioInstalledEntry {
            AudioInstalledEntry(group: .available, alias: id, profile: profile)
        }
    }

    private static let recommendedAliases: [(String, Group)] = [
        ("parakeet-unified-en-coreml", .liveText),
        ("parakeet-ultra-coreml", .finalTranscript),
        ("parakeet-redux-coreml", .finalTranscript),
        ("speaker-diarization-coreml", .speakers),
        (AudioWorkspaceController.defaultMusicProfileAlias, .music),
    ] + AudioWorkspaceController.curatedCloneAliases.map { ($0, Group.voiceCloning) }

    /// Returns the pinned recommendations supported by this Mac, grouped by
    /// job. Rust Core ML profiles remain on the Rust audio-session route.
    static func entries(
        profiles: [AudioProfile],
        searchText: String = "",
        sort: ModelHubFilter.SortOption = .recommended,
        canRun: (AudioProfile) -> Bool
    ) -> [Entry] {
        let byAlias = Dictionary(profiles.map { ($0.identity.alias, $0) }, uniquingKeysWith: { first, _ in first })
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        var result = recommendedAliases.compactMap { alias, group -> Entry? in
            guard let profile = byAlias[alias], supports(group, profile), canRun(profile) else { return nil }
            let entry = Entry(group: group, profile: profile)
            guard query.isEmpty || matches(query, entry) else { return nil }
            return entry
        }

        switch sort {
        case .recommended:
            break
        case .name:
            result.sort { $0.profile.displayName.localizedStandardCompare($1.profile.displayName) == .orderedAscending }
        case .size:
            result.sort {
                if $0.profile.downloadBytes != $1.profile.downloadBytes {
                    return $0.profile.downloadBytes < $1.profile.downloadBytes
                }
                return $0.id < $1.id
            }
        }
        return result
    }

    static func grouped(_ entries: [Entry]) -> [(group: Group, entries: [Entry])] {
        Group.allCases.compactMap { group in
            let rows = entries.filter { $0.group == group }
            return rows.isEmpty ? nil : (group, rows)
        }
    }

    private static func supports(_ group: Group, _ profile: AudioProfile) -> Bool {
        switch group {
        case .liveText: return SpeechToTextJob.live.accepts(profile)
        case .finalTranscript: return SpeechToTextJob.final.accepts(profile)
        case .speakers: return SpeechToTextJob.speakers.accepts(profile)
        case .music: return profile.identity.task == .music && profile.capabilities.operations.contains("generate_music")
        case .voiceCloning:
            // The Voice Cloning page's own test, so Discover offers only what the page lists.
            return AudioWorkspaceController.curatedCloneProfiles([profile]).first != nil
                && (AudioWorkspaceController.clonesNatively(profile) || profile.family == "qwen3_tts")
        }
    }

    private static func matches(_ query: String, _ entry: Entry) -> Bool {
        let profile = entry.profile
        return profile.identity.alias.lowercased().contains(query)
            || profile.displayName.lowercased().contains(query)
            || profile.family.lowercased().contains(query)
            || profile.identity.repository.lowercased().contains(query)
            || entry.group.titleKey.lowercased().contains(query)
    }
}
