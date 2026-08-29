import Foundation
import Combine

/// Manages persistence, discovery, trusted command hashes, and options for lifecycle hooks.
@MainActor
public final class AppHookStore: ObservableObject {
    public static let shared = AppHookStore()

    @Published public internal(set) var hooks: [AppHookCommand] = []
    @Published public internal(set) var trustedHashes: Set<String> = []
    @Published public internal(set) var optionValues: [String: [String: String]] = [:] // [SourceID: [OptionKey: OptionValue]]
    @Published public internal(set) var sourceGroups: [AppHookSourceGroup] = []

    let fileManager = FileManager.default

    public init() {
        loadTrustedHashes()
        loadOptionValues()
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
        recomputeSourceGroups(projectDirectory: nil)
    }

    /// Revokes trust for a hook.
    public func untrustHook(_ hook: AppHookCommand) {
        trustedHashes.remove(hook.contentHash)
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: nil)
    }

    /// Trusts all currently unreviewed hooks in a source group.
    public func trustAllInGroup(_ groupID: String) {
        if let group = sourceGroups.first(where: { $0.id == groupID }) {
            for hook in group.hooks {
                trustedHashes.insert(hook.contentHash)
            }
            saveTrustedHashes()
            recomputeSourceGroups(projectDirectory: nil)
        }
    }

    // MARK: - Toggle & Option Updates

    public func toggleHookEnabled(id: UUID) {
        if let index = hooks.firstIndex(where: { $0.id == id }) {
            hooks[index].isEnabled.toggle()
            if hooks[index].sourceType == .custom {
                saveCustomHooks()
            }
            recomputeSourceGroups(projectDirectory: nil)
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
        var newHook = hook
        newHook.sourceType = .custom
        trustedHashes.insert(newHook.contentHash)
        hooks.append(newHook)
        saveCustomHooks()
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: nil)
    }

    public func updateCustomHook(_ hook: AppHookCommand) {
        if let index = hooks.firstIndex(where: { $0.id == hook.id }) {
            hooks[index] = hook
            saveCustomHooks()
            recomputeSourceGroups(projectDirectory: nil)
        }
    }

    public func deleteCustomHook(id: UUID) {
        hooks.removeAll { $0.id == id }
        saveCustomHooks()
        recomputeSourceGroups(projectDirectory: nil)
    }
}
