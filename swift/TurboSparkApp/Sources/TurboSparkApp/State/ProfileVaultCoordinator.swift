import AppKit
import Foundation
import LocalAuthentication

@MainActor
public final class ProfileVaultCoordinator: ObservableObject {
    public enum State: Equatable {
        case locked
        case unlocking
        case unlocked
        case error(String)
    }

    public static let shared = ProfileVaultCoordinator()

    @Published public private(set) var state: State = .locked
    @Published public private(set) var model: AppModel?
    @Published public private(set) var lastAuthenticationAt: Date?

    private let store: ProfileVaultStore

    init(store: ProfileVaultStore = .shared) {
        self.store = store
        do {
            if try store.prepareForLaunch() != nil {
                state = .unlocked
                constructModel()
            } else {
                state = .locked
            }
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    public var isProtected: Bool { store.isProtected }
    public var quickUnlockEnabled: Bool { store.manifest?.quickUnlockEnabled == true }
    public var canUseQuickUnlock: Bool { store.canUseQuickUnlock }
    public var publicLabel: String { store.manifest?.publicLabel ?? "Protected Profile" }

    public func unlock(passphrase: String) async {
        guard state != .unlocking else { return }
        state = .unlocking
        do {
            try await Task.detached(priority: .userInitiated) { [store] in
                _ = try store.unlock(passphrase: passphrase)
            }.value
            finishUnlock()
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    public func unlockWithSystemAuthentication() async {
        guard state != .unlocking, quickUnlockEnabled else { return }
        state = .unlocking
        do {
            try await Task.detached(priority: .userInitiated) { [store] in
                _ = try store.unlockWithSystemAuthentication(context: LAContext())
            }.value
            finishUnlock()
        } catch {
            state = .error(error.localizedDescription)
        }
    }

    public func protect(passphrase: String, quickUnlock: Bool) async throws {
        try await model?.migratePrivateAssetsIntoVault()
        model?.persistChats()
        model?.persistProjects()
        model?.persistSettings()
        AppChatFileStore.flush()
        let displayName = model?.currentProfile.name ?? UserProfileStore.active.name
        try ProfileRepository.shared.save(displayName, key: "profile:display-name")
        try await Task.detached(priority: .userInitiated) { [store] in
            try store.protect(passphrase: passphrase, enableQuickUnlock: quickUnlock)
        }.value
        setActiveRegistryPrivacy(
            name: store.manifest?.publicLabel ?? "Protected Profile", isProtected: true)
        let legacyImages = AppStorageRoot.directory
            .appendingPathComponent("image-artifacts", isDirectory: true)
        if FileManager.default.fileExists(atPath: legacyImages.path) {
            try FileManager.default.removeItem(at: legacyImages)
        }
        model?.loadProfiles()
        lastAuthenticationAt = Date()
        objectWillChange.send()
    }

    public func changePassphrase(current: String, replacement: String) async throws {
        try await Task.detached(priority: .userInitiated) { [store] in
            try store.changePassphrase(current: current, replacement: replacement)
        }.value
        lastAuthenticationAt = Date()
        objectWillChange.send()
    }

    public func setQuickUnlock(enabled: Bool) async throws {
        try await Task.detached(priority: .userInitiated) { [store] in
            try store.setQuickUnlock(enabled: enabled)
        }.value
        objectWillChange.send()
    }

    public func disableProtection(passphrase: String) async throws {
        let displayName = try ProfileRepository.shared.load(
            String.self, key: "profile:display-name")
        try await Task.detached(priority: .userInitiated) { [store] in
            try store.disableProtection(passphrase: passphrase)
        }.value
        if let displayName { setActiveRegistryPrivacy(name: displayName, isProtected: false) }
        model?.loadProfiles()
        lastAuthenticationAt = Date()
        objectWillChange.send()
    }

    public func lockNow() {
        guard store.isProtected else { return }
        model?.shutdown()
        AppShutdownCoordinator.shared.onTerminate = nil
        ManagedAssetStore.shared.purgeDecryptedCache()
        model = nil
        store.lockVault()
        state = .locked
        lastAuthenticationAt = nil
    }

    public func authenticateRecently(with passphrase: String) async -> Bool {
        do {
            _ = try await Task.detached(priority: .userInitiated) { [store] in
                try store.unlock(passphrase: passphrase)
            }.value
            lastAuthenticationAt = Date()
            return true
        } catch {
            return false
        }
    }

    private func finishUnlock() {
        state = .unlocked
        lastAuthenticationAt = Date()
        AppearanceManager.shared.reloadFromProfile()
        constructModel()
    }

    private func constructModel() {
        let opened = AppModel()
        model = opened
        Task { @MainActor [weak opened] in
            guard let opened else { return }
            do {
                try await opened.migratePrivateAssetsIntoVault()
                let legacyImages = AppStorageRoot.directory
                    .appendingPathComponent("image-artifacts", isDirectory: true)
                if FileManager.default.fileExists(atPath: legacyImages.path) {
                    try FileManager.default.removeItem(at: legacyImages)
                }
            } catch {
                opened.showToast(
                    "Private file migration is incomplete: \(error.localizedDescription)",
                    style: .error, duration: 10)
            }
        }
    }

    private func setActiveRegistryPrivacy(name: String, isProtected: Bool) {
        guard UserProfileStore.active.id != UserProfileStore.defaultProfileID else { return }
        var registry = UserProfileStore.loadRegistry()
        guard let index = registry.profiles.firstIndex(where: {
            $0.id == UserProfileStore.active.id
        }) else { return }
        registry.profiles[index].name = name
        registry.profiles[index].isProtected = isProtected
        _ = UserProfileStore.saveRegistry(registry)
    }
}
