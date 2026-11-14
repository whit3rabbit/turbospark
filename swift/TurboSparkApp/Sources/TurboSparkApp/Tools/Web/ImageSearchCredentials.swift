import Foundation
import Security

/// Where `image_search` gets its Brave Search API key.
///
/// Nothing is stored by default. A key typed in Settings > Network > Image
/// search lives in memory for the life of the process. Keychain storage is an
/// opt-in per credential (swift/AGENTS.md): with the opt-in off, this type
/// never touches the Keychain, and with it on, the Keychain is read only when
/// an `image_search` call needs the key, never at launch.
enum ImageSearchCredentials {
    static let optInDefaultsKey = "imageSearchKeyInKeychain"
    private static let service = "TurboSpark.image-search"
    private static let account = "brave-api-key"

    private static let lock = NSLock()
    nonisolated(unsafe) private static var sessionKey: String?

    static func setSessionKey(_ key: String?) {
        let trimmed = key?.trimmingCharacters(in: .whitespacesAndNewlines)
        lock.lock()
        sessionKey = (trimmed?.isEmpty ?? true) ? nil : trimmed
        lock.unlock()
    }

    static var hasSessionKey: Bool {
        lock.lock(); defer { lock.unlock() }
        return sessionKey != nil
    }

    static func isKeychainEnabled(_ defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: optInDefaultsKey)
    }

    static func setKeychainEnabled(_ enabled: Bool, defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: optInDefaultsKey)
    }

    /// The key for one call: the in-memory key first, then the Keychain item
    /// only for a user who opted in. `keychainRead` is injectable so a test
    /// can prove the Keychain is not consulted without the opt-in.
    static func resolveKey(
        defaults: UserDefaults = .standard,
        keychainRead: () -> String? = { KeychainReadResult.read(query: baseQuery).value }
    ) -> String? {
        lock.lock()
        let memory = sessionKey
        lock.unlock()
        if let memory { return memory }
        guard isKeychainEnabled(defaults) else { return nil }
        let stored = keychainRead()?.trimmingCharacters(in: .whitespacesAndNewlines)
        return (stored?.isEmpty ?? true) ? nil : stored
    }

    /// `errSecSuccess` on success; otherwise the status, so the pane can show
    /// the system's own reason.
    static func saveToKeychain(_ key: String) -> OSStatus {
        let data = Data(key.utf8)
        let update = SecItemUpdate(baseQuery as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return update }
        var query = baseQuery
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(query as CFDictionary, nil)
    }

    @discardableResult
    static func deleteFromKeychain() -> Bool {
        let status = SecItemDelete(baseQuery as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }
}
