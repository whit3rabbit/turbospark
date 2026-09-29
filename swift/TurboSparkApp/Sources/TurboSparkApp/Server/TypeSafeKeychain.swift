import Foundation
import Security

enum TypeSafeKeychain {
    private static let service = "TurboSpark.typesafe"
    private static let account = "typesafe-api-key"

    static func loadKey() -> String? {
        var query = baseQuery
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data else { return nil }
        return String(data: data, encoding: .utf8)
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
