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

    /// UserDefaults key for the opt-in. Keychain storage of the TypeSafe key is
    /// OFF until the user turns it on in the TypeSafe pane. It used to be on
    /// for everyone: the key was generated and written to the Keychain at
    /// every first launch. An ad-hoc signed app has no stable identity, so each
    /// new build was then asked to prove itself against the old build's item,
    /// which is a Keychain prompt on every update, and a launch with no usable
    /// default keychain left the TypeSafe pane unable to start at all.
    static let optInDefaultsKey = "typeSafeKeyInKeychain"

    static func isKeychainEnabled(_ defaults: UserDefaults = .standard) -> Bool {
        defaults.bool(forKey: optInDefaultsKey)
    }

    static func setKeychainEnabled(_ enabled: Bool, defaults: UserDefaults = .standard) {
        defaults.set(enabled, forKey: optInDefaultsKey)
    }

    /// The key to start with at launch. Without the opt-in this is a fresh
    /// random key held in memory only and the Keychain is never touched, so
    /// the daemon is still authenticated by default and nothing is prompted
    /// or persisted. With it, the stored key (created on first use).
    static func initialKey(useKeychain: Bool) -> (key: String, loadFailed: Bool) {
        guard useKeychain else { return (generateKey(), false) }
        return loadOrCreateKey()
    }

    /// The system's own wording for a Keychain failure, with its code, e.g.
    /// "A default keychain could not be found. (-25307)". `OSStatus` alone
    /// names nothing a person can act on.
    static func describe(status: OSStatus) -> String {
        let text = (SecCopyErrorMessageString(status, nil) as String?) ?? "OSStatus"
        return "\(text) (\(status))"
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
        saveKeyStatus(key) == errSecSuccess
    }

    /// `saveKey` with the status kept, so a failure can be shown with the
    /// system's reason instead of a bare "could not save".
    static func saveKeyStatus(_ key: String) -> OSStatus {
        guard !key.isEmpty else {
            let status = SecItemDelete(baseQuery as CFDictionary)
            return status == errSecItemNotFound ? errSecSuccess : status
        }
        let data = Data(key.utf8)
        let update = SecItemUpdate(baseQuery as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if update == errSecSuccess { return update }
        var query = baseQuery
        query[kSecValueData as String] = data
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        return SecItemAdd(query as CFDictionary, nil)
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

