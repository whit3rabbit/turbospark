import Foundation
@testable import TurboSparkApp
import XCTest

/// The profile registry, its resolution precedence, and the path math that
/// decides what is per profile and what is shared.
///
/// The resolution and path helpers are PURE functions taking their inputs
/// rather than reading the process, because `UserProfileStore.active` and
/// `AppStorageRoot.directory` are cached `static let`s a test cannot
/// re-point; what the tests exercise here is everything those cached values
/// are computed FROM.
final class UserProfileTests: XCTestCase {
    private var tempRoot: URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("UserProfileTests-\(ProcessInfo.processInfo.processIdentifier)",
                                    isDirectory: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: tempRoot)
        super.tearDown()
    }

    // MARK: - Registry file semantics

    func testAnAbsentRegistryFileIsTheDefaultOnlyState() {
        // In a test host the registry lives under the per-process temp root,
        // so "absent" is simply the fresh-state file.
        try? FileManager.default.removeItem(at: UserProfileStore.registryURL)
        let registry = UserProfileStore.loadRegistry()
        XCTAssertTrue(registry.profiles.isEmpty)
        XCTAssertEqual(registry.activeProfileID, UserProfileStore.defaultProfileID)
    }

    func testACorruptRegistryFileQuarantinesAndYieldsDefaults() throws {
        let fileManager = FileManager.default
        let url = UserProfileStore.registryURL
        try fileManager.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data("not json at all".utf8).write(to: url)

        let registry = UserProfileStore.loadRegistry()
        XCTAssertTrue(registry.profiles.isEmpty, "a broken registry must not take the stores down")

        let quarantined = try fileManager.contentsOfDirectory(
            at: url.deletingLastPathComponent(), includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.contains("corrupt-") }
        XCTAssertFalse(quarantined.isEmpty, "the unreadable bytes are preserved, not dropped")
        for file in quarantined {
            try? fileManager.removeItem(at: file)
        }
        // The latch is meant for the app's toast; left set, it would surface
        // in whichever later test next calls `surfaceStorageIssues()`.
        XCTAssertNotNil(AppJSONStore.lastReadError)
        AppJSONStore.clearLastReadError()
    }

    func testARegistryRoundTripsThroughCodable() throws {
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "Round Trip")
        try UserProfileStore.adding(profile, to: &registry)
        UserProfileStore.setActive(profile.id, in: &registry)

        let data = try JSONEncoder().encode(registry)
        let decoded = try JSONDecoder().decode(UserProfileRegistry.self, from: data)
        XCTAssertEqual(decoded, registry)
    }

    // MARK: - Resolution precedence

    private func makeRegistry(active: String) -> UserProfileRegistry {
        var registry = UserProfileRegistry()
        registry.activeProfileID = active
        return registry
    }

    func testThePersistedChoiceIsTheFallback() {
        let resolved = UserProfileStore.resolveActiveProfileID(
            registry: makeRegistry(active: "persisted"),
            arguments: [],
            environment: [:])
        XCTAssertEqual(resolved, "persisted")
    }

    func testTheEnvironmentVariableWinsOverThePersistedChoice() {
        let resolved = UserProfileStore.resolveActiveProfileID(
            registry: makeRegistry(active: "persisted"),
            arguments: [],
            environment: [UserProfileStore.environmentKey: "from-env"])
        XCTAssertEqual(resolved, "from-env")
    }

    func testTheLaunchArgumentWinsOverEverything() {
        let resolved = UserProfileStore.resolveActiveProfileID(
            registry: makeRegistry(active: "persisted"),
            arguments: ["app", "-\(UserProfileStore.launchArgumentName)", "from-arg"],
            environment: [UserProfileStore.environmentKey: "from-env"])
        XCTAssertEqual(resolved, "from-arg")
    }

    func testAnOverrideNamingNothingFallsThroughToTheNextSource() {
        // A flag with no value after it, and an empty environment value, are
        // both silence rather than an id naming nothing.
        let fromMissingValue = UserProfileStore.resolveActiveProfileID(
            registry: makeRegistry(active: "persisted"),
            arguments: ["app", "-\(UserProfileStore.launchArgumentName)"],
            environment: [:])
        XCTAssertEqual(fromMissingValue, "persisted")

        let fromEmptyEnv = UserProfileStore.resolveActiveProfileID(
            registry: makeRegistry(active: "persisted"),
            arguments: [],
            environment: [UserProfileStore.environmentKey: ""])
        XCTAssertEqual(fromEmptyEnv, "persisted")
    }

    func testAnUnknownProfileIDResolvesToTheDefaultUser() throws {
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "Known")
        try UserProfileStore.adding(profile, to: &registry)

        XCTAssertEqual(
            UserProfileStore.profile(withID: profile.id, in: registry), profile)
        XCTAssertEqual(
            UserProfileStore.profile(withID: "no-such-id", in: registry),
            UserProfileStore.defaultProfile,
            "an id naming nothing means the Default user, never an invented folder")
        XCTAssertEqual(
            UserProfileStore.profile(withID: UserProfileStore.defaultProfileID, in: registry),
            UserProfileStore.defaultProfile)
    }

    // MARK: - Path math: store directory

    func testTheDefaultProfileHasNoStoreDirectoryOfItsOwn() {
        XCTAssertNil(
            UserProfileStore.storeDirectory(
                profileID: UserProfileStore.defaultProfileID, machineRoot: tempRoot),
            "the Default user's stores stay at the machine root, as they always have")
    }

    func testAProfileStoreDirectoryIsProfilesIDUnderTheMachineRoot() {
        let directory = UserProfileStore.storeDirectory(
            profileID: "abc123", machineRoot: tempRoot)
        XCTAssertEqual(
            directory?.standardizedFileURL.path,
            tempRoot.appendingPathComponent("profiles", isDirectory: true)
                .appendingPathComponent("abc123", isDirectory: true)
                .standardizedFileURL.path)
    }

    func testProfilesDoNotRedirectAPinnedRoot() {
        // Under a test host (or a TURBOSPARK_STATE_DIR override) profile
        // resolution is bypassed entirely: the root is what was pinned. If
        // this ever fails, a profile folder could swallow a redirected run.
        XCTAssertEqual(
            AppStorageRoot.directory.standardizedFileURL,
            AppStorageRoot.machineRoot.standardizedFileURL)
    }

    // MARK: - Path math: user scope

    func testTheDefaultUserScopeIsTheSharedHomeTree() {
        let home = tempRoot.appendingPathComponent("home", isDirectory: true)
        let scope = UserProfileStore.userScopeSubdirectory(
            "skills",
            profileID: UserProfileStore.defaultProfileID,
            homeDirectory: home,
            machineRoot: tempRoot)
        XCTAssertEqual(
            scope.standardizedFileURL.path,
            home.appendingPathComponent(".turbospark/skills", isDirectory: true)
                .standardizedFileURL.path)
    }

    func testAProfileUserScopeIsInsideItsOwnFolder() {
        let home = tempRoot.appendingPathComponent("home", isDirectory: true)
        let scope = UserProfileStore.userScopeSubdirectory(
            "skills",
            profileID: "abc123",
            homeDirectory: home,
            machineRoot: tempRoot)
        XCTAssertFalse(scope.path.hasPrefix(home.path),
                       "a non-default profile never touches the shared home tree")
        XCTAssertEqual(
            scope.standardizedFileURL.path,
            tempRoot.appendingPathComponent("profiles", isDirectory: true)
                .appendingPathComponent("abc123", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
                .standardizedFileURL.path)
    }

    // MARK: - Registry mutations

    func testAddingRejectsEmptyNamesAndCaseInsensitiveDuplicates() throws {
        var registry = UserProfileRegistry()
        try UserProfileStore.adding(UserProfile(name: "Work"), to: &registry)

        XCTAssertThrowsError(
            try UserProfileStore.adding(UserProfile(name: "   "), to: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .emptyName)
        }
        XCTAssertThrowsError(
            try UserProfileStore.adding(UserProfile(name: "work"), to: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .duplicateName)
        }
    }

    func testAddingTrimsTheNameItStores() throws {
        var registry = UserProfileRegistry()
        try UserProfileStore.adding(UserProfile(name: "  Padded  "), to: &registry)
        XCTAssertEqual(registry.profiles.first?.name, "Padded")
    }

    func testRenamingRejectsEmptyNamesAndCaseInsensitiveDuplicates() throws {
        var registry = UserProfileRegistry()
        let first = UserProfile(name: "First")
        let second = UserProfile(name: "Second")
        try UserProfileStore.adding(first, to: &registry)
        try UserProfileStore.adding(second, to: &registry)

        XCTAssertThrowsError(
            try UserProfileStore.renaming(second.id, to: "  ", in: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .emptyName)
        }
        XCTAssertThrowsError(
            try UserProfileStore.renaming(second.id, to: "FIRST", in: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .duplicateName)
        }
        try UserProfileStore.renaming(second.id, to: "Renamed", in: &registry)
        XCTAssertEqual(registry.profiles.first(where: { $0.id == second.id })?.name, "Renamed")
    }

    func testRenamingAProfileToItsOwnNameIsANoOp() throws {
        // The rename sheet prefills the current name, so confirming without
        // editing must not toast an error for a change nobody made.
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "First")
        try UserProfileStore.adding(profile, to: &registry)

        try UserProfileStore.renaming(profile.id, to: "First", in: &registry)
        XCTAssertEqual(registry.profiles.first?.name, "First")
        try UserProfileStore.renaming(profile.id, to: "first", in: &registry)
        XCTAssertEqual(registry.profiles.first?.name, "First")
    }

    func testTheReservedNameIsRefusedOnAddAndRename() throws {
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "First")
        try UserProfileStore.adding(profile, to: &registry)

        XCTAssertThrowsError(
            try UserProfileStore.adding(UserProfile(name: "Default"), to: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .reservedName)
        }
        XCTAssertThrowsError(
            try UserProfileStore.adding(UserProfile(name: "DEFAULT"), to: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .reservedName)
        }
        XCTAssertThrowsError(
            try UserProfileStore.renaming(profile.id, to: "default", in: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .reservedName)
        }
        XCTAssertEqual(registry.profiles.count, 1, "refused mutations add nothing")
        XCTAssertEqual(registry.profiles.first?.name, "First", "refused renames change nothing")
    }

    func testDeletingRefusesTheDefaultAndTheActiveProfile() throws {
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "Deletable")
        try UserProfileStore.adding(profile, to: &registry)
        UserProfileStore.setActive(profile.id, in: &registry)

        XCTAssertThrowsError(
            try UserProfileStore.deleting(UserProfileStore.defaultProfileID, from: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .reservedDefault)
        }
        XCTAssertThrowsError(
            try UserProfileStore.deleting(profile.id, from: &registry)
        ) { error in
            XCTAssertEqual(error as? UserProfileStore.MutationError, .isActive)
        }

        // Off the active profile, the deletion goes through.
        UserProfileStore.setActive(UserProfileStore.defaultProfileID, in: &registry)
        try UserProfileStore.deleting(profile.id, from: &registry)
        XCTAssertFalse(registry.profiles.contains(where: { $0.id == profile.id }))
    }

    func testSetActiveNamesOnlyKnownProfiles() throws {
        var registry = UserProfileRegistry()
        let profile = UserProfile(name: "Known")
        try UserProfileStore.adding(profile, to: &registry)

        UserProfileStore.setActive(profile.id, in: &registry)
        XCTAssertEqual(registry.activeProfileID, profile.id)

        UserProfileStore.setActive("no-such-id", in: &registry)
        XCTAssertEqual(registry.activeProfileID, UserProfileStore.defaultProfileID)
    }
}
