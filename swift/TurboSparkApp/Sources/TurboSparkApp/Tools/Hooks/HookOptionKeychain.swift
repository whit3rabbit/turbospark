import CryptoKit
import Foundation
import Security

protocol HookOptionSecretStoring {
    func load(sourceID: String, key: String, storageDirectory: URL) -> String?
    @discardableResult
    func save(_ value: String, sourceID: String, key: String, storageDirectory: URL) -> Bool
}

/// Keychain storage for hook options declared sensitive by a plugin manifest.
/// The account includes the profile-aware storage path so two profiles using
/// the same plugin do not share credentials.
struct HookOptionKeychain: HookOptionSecretStoring {
    private static let service = "TurboSpark.hook-options"

    func load(sourceID: String, key: String, storageDirectory: URL) -> String? {
        var query = baseQuery(sourceID: sourceID, key: key, storageDirectory: storageDirectory)
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data
        else { return nil }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    func save(_ value: String, sourceID: String, key: String, storageDirectory: URL) -> Bool {
        let query = baseQuery(sourceID: sourceID, key: key, storageDirectory: storageDirectory)
        if value.isEmpty {
            let status = SecItemDelete(query as CFDictionary)
            return status == errSecSuccess || status == errSecItemNotFound
        }

        let data = Data(value.utf8)
        let status = SecItemUpdate(
            query as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if status == errSecSuccess { return true }
        guard status == errSecItemNotFound else { return false }

        var item = query
        item[kSecValueData as String] = data
        item[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
    }

    private func baseQuery(
        sourceID: String, key: String, storageDirectory: URL
    ) -> [String: Any] {
        let identity = "\(storageDirectory.standardizedFileURL.path)\u{0}\(sourceID)\u{0}\(key)"
        let account = SHA256.hash(data: Data(identity.utf8))
            .map { String(format: "%02x", $0) }.joined()
        return [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: Self.service,
            kSecAttrAccount as String: account
        ]
    }
}
