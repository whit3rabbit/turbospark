import Foundation
import Combine

struct AppHookDispatchSnapshot: Sendable {
    let hooks: [AppHookCommand]
    let trustedHashes: Set<String>
    let optionValues: [String: [String: String]]
    let sensitiveOptionKeys: [String: Set<String>]
}

/// Manages persistence, discovery, trusted command hashes, and options for lifecycle hooks.
@MainActor
public final class AppHookStore: ObservableObject {
    public static let shared = AppHookStore()

    @Published public internal(set) var hooks: [AppHookCommand] = []
    @Published public internal(set) var trustedHashes: Set<String> = []
    @Published public internal(set) var optionValues: [String: [String: String]] = [:] // [SourceID: [OptionKey: OptionValue]]
    @Published public internal(set) var sourceGroups: [AppHookSourceGroup] = []
    /// Config entries discovery could not parse (unknown `type`, missing
    /// `command`), so a broken settings.json reads as "hooks are silently
    /// missing" no longer -- surfaced here for the settings UI to show.
    @Published public internal(set) var discoveryDiagnostics: [String] = []

    /// The project directory `refresh(projectDirectory:)` was last called
    /// with. Every mutation that calls `recomputeSourceGroups` afterward
    /// (toggling a hook, trusting a group, adding a custom hook, ...) reads
    /// this instead of passing `nil`, or the project config group's title
    /// and membership would revert to "no project" on the very next edit.
    public internal(set) var lastProjectDirectory: String?

    /// Whether `refresh(projectDirectory:)` has populated `hooks` yet.
    ///
    /// `init()` loads trusted hashes and option values but NOT `hooks`, so
    /// every custom hook on disk is absent from memory until a refresh --
    /// and `saveCustomHooks` writes `hooks.filter { .custom }`, i.e. `[]`.
    /// Any mutation before the first refresh therefore truncated the file.
    public internal(set) var didRefreshAtLeastOnce = false

    /// The plugins hook discovery last saw, kept so `recomputeSourceGroups`
    /// can derive each plugin group's `optionSpecs` from the plugin's own
    /// `userConfig` manifest instead of guessing.
    public internal(set) var lastLoadedPlugins: [LoadedPlugin] = []

    /// Injectable plugin source for tests. When set, hook discovery reads
    /// enabled plugins from here instead of `PluginManager.shared`, whose
    /// default roots point at the real home directory.
    var pluginProvider: ((String?) -> [LoadedPlugin])?

    let fileManager = FileManager.default

    public init() {
        loadTrustedHashes()
        loadOptionValues()
    }

    /// Captures one project's complete hook policy atomically.
    func dispatchSnapshot(projectDirectory: String?) -> AppHookDispatchSnapshot {
        if lastProjectDirectory != projectDirectory || !didRefreshAtLeastOnce {
            refresh(projectDirectory: projectDirectory)
        }
        let sensitive = Dictionary(uniqueKeysWithValues: sourceGroups.map { group in
            (group.id, Set(group.optionSpecs.filter(\.isSensitive).map(\.key)))
        })
        return AppHookDispatchSnapshot(
            hooks: hooks, trustedHashes: trustedHashes, optionValues: optionValues,
            sensitiveOptionKeys: sensitive)
    }

    // MARK: - Trust & Review Checks

    /// Returns true if the hook's command and parameters have been explicitly reviewed and trusted by the user.
    public func isHookTrusted(_ hook: AppHookCommand) -> Bool {
        // Custom in-app created hooks are trusted by default
        if hook.sourceType == .custom {
            return true
        }
        return trustedHashes.contains(hook.contentHash)
    }

    /// Marks a hook as trusted by storing its SHA-256 content hash.
    public func trustHook(_ hook: AppHookCommand) {
        trustedHashes.insert(hook.contentHash)
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: lastProjectDirectory)
    }

    /// Revokes trust for a hook.
    public func untrustHook(_ hook: AppHookCommand) {
        trustedHashes.remove(hook.contentHash)
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: lastProjectDirectory)
    }

    /// Trusts all currently unreviewed hooks in a source group.
    public func trustAllInGroup(_ groupID: String) {
        if let group = sourceGroups.first(where: { $0.id == groupID }) {
            for hook in group.hooks {
                trustedHashes.insert(hook.contentHash)
            }
            saveTrustedHashes()
            recomputeSourceGroups(projectDirectory: lastProjectDirectory)
        }
    }

    // MARK: - Toggle & Option Updates

    public func toggleHookEnabled(id: UUID) {
        if let index = hooks.firstIndex(where: { $0.id == id }) {
            hooks[index].isEnabled.toggle()
            if hooks[index].sourceType == .custom {
                saveCustomHooks()
            }
            recomputeSourceGroups(projectDirectory: lastProjectDirectory)
        }
    }

    public func updateOptionValue(sourceID: String, key: String, value: String) {
        var current = optionValues[sourceID] ?? [:]
        current[key] = value
        optionValues[sourceID] = current
        saveOptionValues()
    }

    public func getOptionValue(sourceID: String, key: String, defaultVal: String? = nil) -> String {
        optionValues[sourceID]?[key] ?? defaultVal ?? ""
    }

    // MARK: - Custom Hook Creation & Deletion

    public func addCustomHook(_ hook: AppHookCommand) {
        // A project-bound dispatch refreshes the store before taking its
        // snapshot. Discover first so a hook added during app startup is
        // persisted and survives that refresh; otherwise the pre-refresh
        // save is intentionally skipped and the in-memory hook disappears.
        if !didRefreshAtLeastOnce {
            refresh(projectDirectory: lastProjectDirectory)
        }
        var newHook = hook
        newHook.sourceType = .custom
        trustedHashes.insert(newHook.contentHash)
        hooks.append(newHook)
        saveCustomHooks()
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: lastProjectDirectory)
    }

    public func updateCustomHook(_ hook: AppHookCommand) {
        if let index = hooks.firstIndex(where: { $0.id == hook.id }) {
            hooks[index] = hook
            saveCustomHooks()
            recomputeSourceGroups(projectDirectory: lastProjectDirectory)
        }
    }

    public func deleteCustomHook(id: UUID) {
        hooks.removeAll { $0.id == id }
        saveCustomHooks()
        recomputeSourceGroups(projectDirectory: lastProjectDirectory)
    }
}
