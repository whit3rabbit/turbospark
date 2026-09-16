import AppKit
import Foundation

// User profiles: list management and switching. The storage seam itself
// lives in `UserProfileStore` + `AppStorageRoot`; this file is the UI-facing
// half, so every failure surfaces as a toast rather than being swallowed.
extension AppModel {
    /// Populates the published list from the registry. The ACTIVE profile was
    /// resolved once, before any store opened (`UserProfileStore.active`);
    /// loading here neither chooses nor re-chooses it.
    func loadProfiles() {
        profiles = UserProfileStore.loadRegistry().profiles
    }

    /// The user this run belongs to. The id is fixed for the process (a
    /// switch is a relaunch); the name resolves through the published list so
    /// renaming the active profile is visible without one.
    public var currentProfile: UserProfile {
        UserProfileStore.profile(
            withID: UserProfileStore.active.id,
            in: UserProfileRegistry(
                profiles: profiles,
                activeProfileID: UserProfileStore.active.id))
    }

    public var isDefaultProfileActive: Bool { UserProfileStore.isDefault }

    // MARK: - Registry mutations

    public func createProfile(named rawName: String) {
        let profile = UserProfile(name: rawName)
        var registry = UserProfileStore.loadRegistry()
        do {
            try UserProfileStore.adding(profile, to: &registry)
        } catch {
            showToast(profileMutationMessage(error), style: .error)
            return
        }
        guard UserProfileStore.saveRegistry(registry) else {
            surfaceStorageIssues()
            return
        }
        // Materialize the folder now, so "reveal in Finder" after creating
        // shows something and the first store write cannot be the thing that
        // discovers a problem with the path.
        _ = UserProfileStore.folder(of: profile)
        profiles = registry.profiles
        showToast("Profile \"\(profile.name)\" created. Switch to it to start using it.")
    }

    public func renameProfile(_ profile: UserProfile, to newName: String) {
        var registry = UserProfileStore.loadRegistry()
        do {
            try UserProfileStore.renaming(profile.id, to: newName, in: &registry)
        } catch {
            showToast(profileMutationMessage(error), style: .error)
            return
        }
        guard UserProfileStore.saveRegistry(registry) else {
            surfaceStorageIssues()
            return
        }
        profiles = registry.profiles
    }

    public func deleteProfile(_ profile: UserProfile) {
        var registry = UserProfileStore.loadRegistry()
        do {
            try UserProfileStore.deleting(profile.id, from: &registry)
        } catch {
            showToast(profileMutationMessage(error), style: .error)
            return
        }
        // Trash the folder BEFORE saving the registry: if the trash fails,
        // the registry still names the profile and nothing is forgotten. The
        // reverse order loses a folder that failed to move.
        do {
            try FileManager.default.trashItem(
                at: UserProfileStore.folder(of: profile), resultingItemURL: nil)
        } catch {
            showToast(
                "Could not move the profile folder to the Trash: \(error.localizedDescription)",
                style: .error, duration: 8)
            return
        }
        guard UserProfileStore.saveRegistry(registry) else {
            surfaceStorageIssues()
            return
        }
        profiles = registry.profiles
        showToast("Profile \"\(profile.name)\" deleted (its folder moved to the Trash).")
    }

    // MARK: - Switching

    /// Whether a switch may run right now. A generation holds in-memory
    /// state the flush below would cut short, and an install is writing a
    /// shared model directory the new process would scan mid-write.
    public var canSwitchProfile: Bool {
        !generating && !submitting && !isInstallingModel
    }

    /// Flushes every store, records the target profile, and relaunches the
    /// process into it. There is deliberately no live switch: every store is
    /// a singleton cached against `AppStorageRoot.directory`, a `static let`
    /// computed once per process.
    public func switchToProfile(_ profile: UserProfile) {
        guard profile.id != UserProfileStore.active.id else { return }
        guard canSwitchProfile else {
            showToast(
                "Finish or cancel the running work before switching profiles.",
                style: .error)
            return
        }
        var registry = UserProfileStore.loadRegistry()
        UserProfileStore.setActive(profile.id, in: &registry)
        guard UserProfileStore.saveRegistry(registry) else {
            surfaceStorageIssues()
            return
        }
        // The same ordered flush the quit path uses. The exit below fires
        // `applicationWillTerminate` a second time, which is safe: every
        // step is idempotent.
        shutdown()
        relaunchIntoSelectedProfile(profile)
    }

    /// Reveals the machine root: `profiles.json`, the Default user's stores,
    /// and every `profiles/<id>/` folder are all under here.
    public func revealProfilesInFinder() {
        NSWorkspace.shared.activateFileViewerSelecting([AppStorageRoot.machineRoot])
    }

    /// Re-execs this binary with the target profile pinned in the child's
    /// environment, so the fresh process opens as that user even if the
    /// registry read races the old process's exit. Works for the bare
    /// SwiftPM binary and the .app bundle alike: both launch through
    /// `Bundle.main.executableURL`.
    private func relaunchIntoSelectedProfile(_ profile: UserProfile) {
        guard let executable = Bundle.main.executableURL else {
            showToast(
                "Could not locate the app executable to relaunch. The profile is saved; reopen the app to use it.",
                style: .error, duration: 10)
            return
        }
        let process = Process()
        process.executableURL = executable
        var environment = ProcessInfo.processInfo.environment
        environment[UserProfileStore.environmentKey] = profile.id
        process.environment = environment
        do {
            try process.run()
        } catch {
            showToast(
                "Relaunch failed: \(error.localizedDescription). The profile is saved; reopen the app to use it.",
                style: .error, duration: 10)
            return
        }
        exit(0)
    }

    func profileMutationMessage(_ error: Error) -> String {
        switch error as? UserProfileStore.MutationError {
        case .emptyName:
            return "A profile needs a name."
        case .duplicateName:
            return "A profile with that name already exists."
        case .reservedName:
            return "\"Default\" is reserved for the built-in user."
        case .reservedDefault:
            return "The Default profile cannot be deleted."
        case .isActive:
            return "Switch to another profile before deleting this one."
        case .notFound:
            return "That profile no longer exists."
        case nil:
            return error.localizedDescription
        }
    }
}
