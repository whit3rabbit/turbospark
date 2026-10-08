import Foundation
import Security

/// Keychain storage for the in-process server's API key.
///
/// The key deliberately does NOT live in `MacAppSettings`/`settings.json`:
/// it gates access to the server, and the hooks area's own rule ("never
/// leak secrets") argues against a credential sitting in a plaintext JSON
/// file beside every other preference. Generic-password item, service
/// scoped to this app; a Keychain read failure degrades to "no key restored"
/// (and blocks the destructive save path, see KeychainSecretSync) rather
/// than blocking the server, which then simply starts without one.
enum ServerKeychain {
    private static let service = "TurboSpark.server"
    private static let account = "server-api-key"

    /// Reads the stored API key, distinguishing "none stored" from a read
    /// error (locked or denied Keychain) so callers never treat the latter as
    /// an empty key.
    static func readKey() -> KeychainReadResult {
        KeychainReadResult.read(query: baseQuery)
    }

    /// Convenience for callers without a destructive follow-up: nil for both
    /// "none stored" and a read error.
    static func loadKey() -> String? {
        readKey().value
    }

    /// Stores (or replaces) the API key. An empty string deletes the item
    /// rather than storing emptiness.
    @discardableResult
    static func saveKey(_ key: String) -> Bool {
        guard !key.isEmpty else { return deleteKey() }
        let data = Data(key.utf8)

        let update: [String: Any] = [kSecValueData as String: data]
        let updateStatus = SecItemUpdate(baseQuery as CFDictionary, update as CFDictionary)
        if updateStatus == errSecSuccess { return true }
        guard updateStatus == errSecItemNotFound else { return false }

        var add: [String: Any] = baseQuery
        add[kSecValueData as String] = data
        add[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(add as CFDictionary, nil) == errSecSuccess
    }

    /// Removes the stored key; reports true when nothing is stored after.
    @discardableResult
    static func deleteKey() -> Bool {
        let status = SecItemDelete(baseQuery as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    private static var baseQuery: [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]
    }
}
