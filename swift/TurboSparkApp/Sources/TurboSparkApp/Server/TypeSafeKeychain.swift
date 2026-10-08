import Foundation
import Security

enum TypeSafeKeychain {
    private static let service = "TurboSpark.typesafe"
    private static let account = "typesafe-api-key"

    static func readKey() -> KeychainReadResult {
        KeychainReadResult.read(query: baseQuery)
    }

    static func loadKey() -> String? {
        readKey().value
    }

    @discardableResult
    static func saveKey(_ key: String) -> Bool {
        guard !key.isEmpty else {
            let status = SecItemDelete(baseQuery as CFDictionary)
            return status == errSecSuccess || status == errSecItemNotFound
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

    private static var baseQuery: [String: Any] {
        [kSecClass as String: kSecClassGenericPassword,
         kSecAttrService as String: service,
         kSecAttrAccount as String: account]
    }
}
