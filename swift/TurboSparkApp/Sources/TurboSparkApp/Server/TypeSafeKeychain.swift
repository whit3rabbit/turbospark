import Foundation
import Security
import OpenKind

enum TypeSafeKeychain {
    private static let service = "TurboSpark.typesafe"
    private static let account = "typesafe-api-key"

    /// Generates a random API key with format "ok_" + 64 hex characters (256 bits).
    static func generateKey() -> String {
        (try? OpenKindAPIKey.generate()) ?? ServerAPIKeyGenerator.generate()
    }

    static func readKey() -> KeychainReadResult {
        KeychainReadResult.read(query: baseQuery)
    }

    static func loadKey() -> String? {
        readKey().value
    }

    /// Reads the stored API key, or generates and stores a fresh random key on first use.
    /// If Keychain read encounters an error (locked or denied), returns an empty string with loadFailed: true.
    static func loadOrCreateKey() -> (key: String, loadFailed: Bool) {
        let read = readKey()
        switch read {
        case let .found(stored):
            let trimmed = stored.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                return (trimmed, false)
            }
            let fresh = generateKey()
            _ = saveKey(fresh)
            return (fresh, false)
        case .notFound:
            let fresh = generateKey()
            let saved = saveKey(fresh)
            return (fresh, !saved)
        case .error:
            return ("", true)
        }
    }

    @discardableResult
    static func saveKey(_ key: String) -> Bool {
        guard !key.isEmpty else {
            return deleteKey()
        }
        let data = Data(key.utf8)
        if SecItemUpdate(baseQuery as CFDictionary, [kSecValueData as String: data] as CFDictionary) == errSecSuccess {
            return true
        }
        var query = baseQuery
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(query as CFDictionary, nil) == errSecSuccess
    }

    @discardableResult
    static func deleteKey() -> Bool {
        let status = SecItemDelete(baseQuery as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }
}

