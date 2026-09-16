import Foundation

/// One named user of this app on this machine.
///
/// Profiles are FOLDERS, not accounts: no password, no login, and the only
/// thing a profile owns is a directory under the machine root plus a row in
/// `profiles.json`. Everything the app stores first-party (settings, chats,
/// projects, global MCP servers, model favorites, appearance, hooks, custom
/// tools) resolves through `AppStorageRoot`, so pointing that one seam at the
/// folder is the whole per-user mechanism; the user-scope content that lives
/// OUTSIDE the root (skills, agents, tools, marketplaces) follows
/// `userScopeSubdirectory` instead.
///
/// The Default user is the machine's existing setup, not a folder of its own:
/// its stores stay at the machine root and its user-scope content stays in the
/// shared `~/.turbospark` tree, so nothing migrates and every installation
/// that predates profiles already is one.
public struct UserProfile: Codable, Equatable, Identifiable {
    public var id: String
    public var name: String
    public var createdAt: Date

    public init(id: String = UUID().uuidString, name: String, createdAt: Date = Date()) {
        self.id = id
        self.name = name
        self.createdAt = createdAt
    }

    private enum CodingKeys: String, CodingKey { case id, name, createdAt }

    /// Tolerant like every store (`swift/CLAUDE.md` Gotcha 13): a row missing
    /// a field decodes as defaults rather than failing the whole registry.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? "Untitled"
        createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
    }
}

/// The on-disk `profiles.json` shape: the ADDITIONAL users, plus which one
/// this installation opens as. The Default user is implicit and never listed,
/// so an empty registry IS the default-only state -- what every existing
/// installation already is on disk, with no migration.
public struct UserProfileRegistry: Codable, Equatable {
    public var profiles: [UserProfile]
    public var activeProfileID: String

    public init(profiles: [UserProfile] = [], activeProfileID: String = UserProfileStore.defaultProfileID) {
        self.profiles = profiles
        self.activeProfileID = activeProfileID
    }

    /// The registry with no additional users and the Default user active.
    public static let empty = UserProfileRegistry()

    private enum CodingKeys: String, CodingKey { case profiles, activeProfileID }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        profiles = try container.decodeIfPresent([UserProfile].self, forKey: .profiles) ?? []
        activeProfileID = try container.decodeIfPresent(String.self, forKey: .activeProfileID)
            ?? UserProfileStore.defaultProfileID
    }
}

/// Loads and resolves the profile registry, and owns the two path helpers the
/// rest of the app reads: `storeDirectory` (via `AppStorageRoot`) for the
/// first-party stores and `userScopeSubdirectory` for the shared-home content.
///
/// `active` is resolved ONCE per process, because `AppStorageRoot.directory`
/// is a cached `static let` every store hangs off: switching profiles is a
/// save-and-relaunch, not a live re-pointing. Everything a test needs is in
/// the pure functions below, which take their inputs rather than reading the
/// process.
public enum UserProfileStore {
    public static let defaultProfileID = "default"

    /// `-TurboSparkProfile <id>` on the command line, or `TURBOSPARK_PROFILE`
    /// in the environment, pins this run to a profile id without editing the
    /// registry. Precedence: launch argument, then environment, then the
    /// persisted `activeProfileID`.
    public static let launchArgumentName = "TurboSparkProfile"
    public static let environmentKey = "TURBOSPARK_PROFILE"

    private static let registryFileName = "profiles.json"
    private static let profilesDirectoryName = "profiles"

    /// The Default user: not a row in the registry, and not a folder. Its
    /// stores stay at the machine root and its user-scope content stays in
    /// the shared `~/.turbospark` tree, exactly as every pre-profile
    /// installation left them.
    public static let defaultProfile = UserProfile(id: defaultProfileID, name: "Default")

    /// This run's user. Must depend on nothing beyond `machineRoot` and
    /// `ProcessInfo`: `AppStorageRoot.directory` reads it, so a dependency on
    /// any store would be a cycle.
    public static let active: UserProfile = {
        let registry = loadRegistry()
        let info = ProcessInfo.processInfo
        let id = resolveActiveProfileID(
            registry: registry,
            arguments: info.arguments,
            environment: info.environment)
        return profile(withID: id, in: registry)
    }()

    public static var isDefault: Bool { active.id == defaultProfileID }

    // MARK: - Pure resolution (the testable half)

    /// Which profile id this run should open as. An override naming nothing
    /// (empty string, a missing value after the flag) falls through to the
    /// next source rather than resolving to a blank id.
    public static func resolveActiveProfileID(
        registry: UserProfileRegistry,
        arguments: [String],
        environment: [String: String]
    ) -> String {
        if let index = arguments.firstIndex(of: "-\(launchArgumentName)"),
           index + 1 < arguments.count,
           !arguments[index + 1].isEmpty {
            return arguments[index + 1]
        }
        if let fromEnvironment = environment[environmentKey], !fromEnvironment.isEmpty {
            return fromEnvironment
        }
        return registry.activeProfileID
    }

    /// The profile an id names, or the Default user when it names nothing.
    /// An id whose row is gone (deleted elsewhere, a hand-edited file) means
    /// the Default user; it never means "invent a folder".
    public static func profile(withID id: String, in registry: UserProfileRegistry) -> UserProfile {
        registry.profiles.first(where: { $0.id == id }) ?? defaultProfile
    }

    /// Where a profile's first-party stores live: nil for the Default user
    /// (machine root, as always) and `profiles/<id>/` under the machine root
    /// for anyone else. Pure; the instance wrapper `storeDirectory()` feeds
    /// it this run's values.
    public static func storeDirectory(profileID: String, machineRoot: URL) -> URL? {
        guard profileID != defaultProfileID else { return nil }
        return machineRoot
            .appendingPathComponent(profilesDirectoryName, isDirectory: true)
            .appendingPathComponent(profileID, isDirectory: true)
    }

    /// The user-scope root for content the Default user shares across
    /// harnesses: `~/.turbospark/<relative>`. Any other profile keeps that
    /// content inside its own folder and never touches the shared tree, and
    /// never reads the cross-agent roots either -- that isolation is the
    /// point of a profile.
    public static func userScopeSubdirectory(
        _ relative: String, profileID: String, homeDirectory: URL, machineRoot: URL
    ) -> URL {
        let base: URL
        if profileID == defaultProfileID {
            base = homeDirectory
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent(relative, isDirectory: true)
        } else {
            base = storeDirectory(profileID: profileID, machineRoot: machineRoot)!
                .appendingPathComponent(relative, isDirectory: true)
        }
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    // MARK: - Registry IO

    public static var registryURL: URL {
        AppStorageRoot.machineRoot.appendingPathComponent(registryFileName)
    }

    /// Loads `profiles.json`. An absent file is a first run and an undecodable
    /// one has already been quarantined by `AppJSONStore`; both come back as
    /// the default-only registry, because a broken registry must not take the
    /// stores down with it.
    public static func loadRegistry() -> UserProfileRegistry {
        AppJSONStore.load(UserProfileRegistry.self, from: registryURL, label: "user profiles")
            ?? .empty
    }

    @discardableResult
    public static func saveRegistry(_ registry: UserProfileRegistry) -> Bool {
        AppJSONStore.save(registry, to: registryURL, label: "user profiles")
    }

    /// Applies `body` to a freshly loaded registry and saves it back. The
    /// registry is small and written whole, so read-modify-write without a
    /// lock is fine for the interactive pace a human edits profiles at.
    @discardableResult
    public static func updateRegistry(_ body: (inout UserProfileRegistry) -> Void) -> Bool {
        var registry = loadRegistry()
        body(&registry)
        return saveRegistry(registry)
    }

    // MARK: - Registry mutations (pure; AppModel wraps these with UI state)

    public enum MutationError: Error, Equatable {
        case emptyName
        case duplicateName
        case reservedName
        case reservedDefault
        case isActive
        case notFound
    }

    /// The built-in Default user's name is not available to registry rows: an
    /// additional profile called "Default" would read as the built-in in
    /// every picker that shows names. Compared case-insensitively, like the
    /// duplicate check. A backup of the Default user imports under a
    /// different name because of this rule.
    public static func isReservedName(_ name: String) -> Bool {
        name.caseInsensitiveCompare(defaultProfile.name) == .orderedSame
    }

    public static func adding(_ profile: UserProfile, to registry: inout UserProfileRegistry) throws {
        let trimmed = profile.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MutationError.emptyName }
        guard !isReservedName(trimmed) else { throw MutationError.reservedName }
        guard !registry.profiles.contains(where: { $0.name.caseInsensitiveCompare(trimmed) == .orderedSame })
        else { throw MutationError.duplicateName }
        var named = profile
        named.name = trimmed
        registry.profiles.append(named)
    }

    public static func renaming(_ id: String, to newName: String, in registry: inout UserProfileRegistry) throws {
        let trimmed = newName.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw MutationError.emptyName }
        guard let index = registry.profiles.firstIndex(where: { $0.id == id }) else {
            throw MutationError.notFound
        }
        // Renaming to the profile's own name is a no-op, not a duplicate:
        // the rename sheet prefills the current name, so refusing it would
        // toast an error for a change nobody made.
        guard registry.profiles[index].name.caseInsensitiveCompare(trimmed) != .orderedSame else {
            return
        }
        guard !isReservedName(trimmed) else { throw MutationError.reservedName }
        guard !registry.profiles.contains(where: {
            $0.id != id && $0.name.caseInsensitiveCompare(trimmed) == .orderedSame
        })
        else { throw MutationError.duplicateName }
        registry.profiles[index].name = trimmed
    }

    public static func deleting(_ id: String, from registry: inout UserProfileRegistry) throws {
        guard id != defaultProfileID else { throw MutationError.reservedDefault }
        // Refusing the ACTIVE profile is what keeps a deleted folder from
        // staying in use: the switch-to-Default path moves off it first.
        guard id != registry.activeProfileID else { throw MutationError.isActive }
        guard let index = registry.profiles.firstIndex(where: { $0.id == id }) else {
            throw MutationError.notFound
        }
        registry.profiles.remove(at: index)
    }

    public static func setActive(_ id: String, in registry: inout UserProfileRegistry) {
        registry.activeProfileID = profile(withID: id, in: registry).id
    }

    // MARK: - This run's paths

    /// `AppStorageRoot.directory` is this plus the machine root.
    public static func storeDirectory() -> URL? {
        storeDirectory(profileID: active.id, machineRoot: AppStorageRoot.machineRoot)
    }

    public static func userScopeSubdirectory(_ relative: String) -> URL {
        userScopeSubdirectory(
            relative,
            profileID: active.id,
            homeDirectory: FileManager.default.homeDirectoryForCurrentUser,
            machineRoot: AppStorageRoot.machineRoot)
    }

    /// The folder a NOT-currently-active profile keeps its stores in,
    /// created on demand. `storeDirectory()` above only answers for THIS
    /// run's profile; create/delete need the others too.
    public static func folder(of profile: UserProfile) -> URL {
        let url = storeDirectory(profileID: profile.id, machineRoot: AppStorageRoot.machineRoot)
            ?? AppStorageRoot.machineRoot
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
}
