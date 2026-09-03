import Foundation

/// Which skills and agents the user has switched off (state#57).
///
/// **UNDER `AppStorageRoot`, NOT `UserDefaults.standard`.** Two things follow
/// from where this used to live, and both were real.
///
/// The test suite writes real user preferences. `AppStorageRoot` exists
/// because seven stores each spelled their own Application Support path and a
/// one-chat fixture replaced somebody's whole archive (`swift/CLAUDE.md`
/// Gotcha 37); `UserDefaults.standard` is the same hazard through a different
/// API, and every `AgentManager`/`SkillManager` case in the suite was
/// mutating the developer's own disabled lists.
///
/// And the preferences move when the BUNDLE IDENTITY does. A `swift run`
/// build writes `~/Library/Preferences/TurboSparkApp.plist`, the shipped
/// bundle writes `com.whit3rabbit.turbospark.plist` (Gotcha 12), so
/// installing the app silently re-enabled everything the user had turned off
/// -- including, on the agent side, whatever they had turned off for a
/// reason. The three JSON stores were never affected because their paths are
/// hardcoded; this is now one of them.
enum DisabledItemStore {
    /// The two independent lists. Keys are `scope:name`, lowercased.
    struct Archive: Codable {
        var skills: [String] = []
        var agents: [String] = []
    }

    enum Kind {
        case skills
        case agents

        /// The `UserDefaults.standard` key this list used to live under, read
        /// once on migration so nobody's existing preference is silently
        /// forgotten.
        var legacyDefaultsKey: String {
            switch self {
            case .skills: return "TurboSpark.disabledSkillNames"
            case .agents: return "TurboSpark.disabledAgentNames"
            }
        }
    }

    private static var fileURL: URL { AppStorageRoot.file("disabled_items.json") }

    private static nonisolated(unsafe) var cache: Archive?

    private static func load() -> Archive {
        if let cache { return cache }
        var archive =
            AppJSONStore.load(Archive.self, from: fileURL, label: "disabled items") ?? Archive()
        // One-way migration: fold in whatever the old defaults hold, then
        // write once so the next launch reads only this file. The legacy keys
        // are left in place rather than deleted -- removing them would strand
        // a user who moves back to an older build mid-upgrade.
        var migrated = false
        for kind in [Kind.skills, Kind.agents] {
            let legacy = UserDefaults.standard.stringArray(forKey: kind.legacyDefaultsKey) ?? []
            guard !legacy.isEmpty else { continue }
            switch kind {
            case .skills:
                let merged = Set(archive.skills).union(legacy)
                if merged.count != archive.skills.count {
                    archive.skills = Array(merged)
                    migrated = true
                }
            case .agents:
                let merged = Set(archive.agents).union(legacy)
                if merged.count != archive.agents.count {
                    archive.agents = Array(merged)
                    migrated = true
                }
            }
        }
        cache = archive
        if migrated { AppJSONStore.save(archive, to: fileURL, label: "Disabled items") }
        return archive
    }

    static func names(for kind: Kind) -> Set<String> {
        let archive = load()
        switch kind {
        case .skills: return Set(archive.skills)
        case .agents: return Set(archive.agents)
        }
    }

    static func setNames(_ names: Set<String>, for kind: Kind) {
        var archive = load()
        switch kind {
        case .skills: archive.skills = names.sorted()
        case .agents: archive.agents = names.sorted()
        }
        cache = archive
        AppJSONStore.save(archive, to: fileURL, label: "Disabled items")
    }

    /// Drops the in-memory copy. For tests that write the file directly.
    static func invalidateCache() {
        cache = nil
    }
}
