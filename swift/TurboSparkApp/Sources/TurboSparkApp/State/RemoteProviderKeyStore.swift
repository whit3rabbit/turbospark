import Foundation
import Security

/// Where the Keychain items live. A seam so tests count reads and prove the
/// Keychain is never touched without the opt-in.
protocol RemoteSecretBackend: Sendable {
    /// Attributes only: never reads the secret, never prompts.
    func exists(account: String) -> Bool
    func read(account: String) -> KeychainReadResult
    func write(account: String, secret: String) -> OSStatus
    func delete(account: String) -> Bool
}

/// Generic-password items, one per provider, in the `ServerKeychain` pattern.
struct SystemKeychainBackend: RemoteSecretBackend {
    static let service = "TurboSpark.remote-provider"

    private func base(_ account: String) -> [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: Self.service,
         kSecAttrAccount as String: account]
    }

    func exists(account: String) -> Bool {
        var query = base(account)
        query[kSecReturnAttributes as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        return SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess
    }

    func read(account: String) -> KeychainReadResult {
        KeychainReadResult.read(query: base(account))
    }

    func write(account: String, secret: String) -> OSStatus {
        let data = Data(secret.utf8)
        let update = SecItemUpdate(base(account) as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return update }
        guard update == errSecItemNotFound else { return update }
        var add = base(account)
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(add as CFDictionary, nil)
    }

    func delete(account: String) -> Bool {
        let status = SecItemDelete(base(account) as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }
}

/// Provider API keys. The rules, from swift/AGENTS.md and ZP-22:
///
/// * A key typed in Settings lives in memory for the life of the process.
///   Nothing is stored by default.
/// * Keychain storage is an opt-in PER CREDENTIAL: the user's explicit "Save
///   to Keychain" action writes the item and records the opt-in for that
///   provider. Deleting the item clears it.
/// * The Keychain is never touched at launch, in `init`, or for a provider
///   no one sends to. `isKeyAvailable` asks by attributes only. The secret is
///   read at the first send to that provider in a session, then held in
///   memory so the system does not prompt again within the session.
/// * A failed read is a named error, never an empty key, and never a reason
///   to delete the item.
final class RemoteProviderKeyStore: @unchecked Sendable {
    static let optInPrefix = "remoteProviderKeyInKeychain."

    private let backend: any RemoteSecretBackend
    private let defaults: UserDefaults
    private let lock = NSLock()
    private var session: [UUID: String] = [:]

    /// No Keychain access happens here.
    init(backend: any RemoteSecretBackend = SystemKeychainBackend(), defaults: UserDefaults = .standard) {
        self.backend = backend
        self.defaults = defaults
    }

    private func account(_ id: UUID) -> String { "provider-key-\(id.uuidString.lowercased())" }
    private func optInKey(_ id: UUID) -> String { Self.optInPrefix + id.uuidString.lowercased() }

    func isKeychainEnabled(for id: UUID) -> Bool { defaults.bool(forKey: optInKey(id)) }

    func setSessionKey(_ key: String?, for id: UUID) {
        let trimmed = key?.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.lock()
        session[id] = (trimmed?.isEmpty ?? true) ? nil : trimmed
        lock.unlock()
    }

    /// Whether a key can be had, WITHOUT reading the secret.
    func isKeyAvailable(for id: UUID) -> Bool {
        lock.lock()
        let inMemory = session[id] != nil
        lock.unlock()
        if inMemory { return true }
        guard isKeychainEnabled(for: id) else { return false }
        return backend.exists(account: account(id))
    }

    /// The user's explicit save. `errSecSuccess` on success so the pane can
    /// show the system's reason otherwise.
    @discardableResult
    func saveToKeychain(_ key: String, for id: UUID) -> OSStatus {
        let trimmed = key.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return errSecParam }
        let status = backend.write(account: account(id), secret: trimmed)
        if status == errSecSuccess {
            defaults.set(true, forKey: optInKey(id))
            setSessionKey(trimmed, for: id)
        }
        return status
    }

    /// Removes the item and the opt-in, and forgets the in-memory key.
    @discardableResult
    func forget(for id: UUID) -> Bool {
        setSessionKey(nil, for: id)
        let removed = backend.delete(account: account(id))
        if removed { defaults.removeObject(forKey: optInKey(id)) }
        return removed
    }

    /// The key for one send, or the named reason there is none.
    func keyForSend(provider id: UUID) throws -> String {
        lock.lock()
        let memory = session[id]
        lock.unlock()
        if let memory { return memory }
        guard isKeychainEnabled(for: id) else { throw RemoteProviderError.notConfigured }
        switch backend.read(account: account(id)) {
        case .found(let stored):
            let trimmed = stored.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { throw RemoteProviderError.notConfigured }
            setSessionKey(trimmed, for: id)
            return trimmed
        case .notFound:
            throw RemoteProviderError.notConfigured
        case .error(let status):
            let text = (SecCopyErrorMessageString(status, nil) as String?) ?? "Keychain error"
            throw RemoteProviderError.keyUnavailable("\(text) (\(status))")
        }
    }
}
