import Foundation
import Security

/// Outcome of a Keychain read. "Nothing stored" and "could not read" are
/// different facts: treating a locked or denied Keychain as an empty one
/// let a later save compare "" against the real item and delete it.
enum KeychainReadResult: Equatable {
    case found(String)
    case notFound
    case error(OSStatus)

    /// The stored value, or nil for both notFound and error. Only for callers
    /// that have no destructive follow-up.
    var value: String? {
        if case let .found(value) = self { return value }
        return nil
    }

    var isError: Bool {
        if case .error = self { return true }
        return false
    }

    /// Pure classification of a `SecItemCopyMatching` outcome, so the mapping
    /// is testable without a Keychain.
    static func classify(status: OSStatus, data: Data?) -> KeychainReadResult {
        switch status {
        case errSecSuccess:
            guard let data, let text = String(data: data, encoding: .utf8) else {
                return .error(errSecDecode)
            }
            return .found(text)
        case errSecItemNotFound:
            return .notFound
        default:
            return .error(status)
        }
    }

    static func read(query: [String: Any]) -> KeychainReadResult {
        var query = query
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        return classify(status: status, data: item as? Data)
    }
}

/// Decides what to do with a Keychain-backed secret edited through a text
/// field. The destructive case (delete) is only reachable when the last read
/// succeeded, so a failed read can never erase a stored secret.
enum KeychainSecretSync {
    enum Action: Equatable {
        case none
        case save(String)
        case delete
        /// The earlier read failed but the Keychain is readable now and the
        /// user has typed nothing: show the stored value, write nothing.
        case adopt(String)
    }

    /// - Parameters:
    ///   - current: a fresh Keychain read.
    ///   - input: what the field holds now.
    ///   - loadFailed: whether the read at launch failed (field started empty
    ///     for that reason, not because nothing is stored).
    static func action(current: KeychainReadResult, input: String, loadFailed: Bool) -> Action {
        switch current {
        case let .found(stored):
            if input == stored { return .none }
            if loadFailed && input.isEmpty { return .adopt(stored) }
            return input.isEmpty ? .delete : .save(input)
        case .notFound:
            return input.isEmpty ? .none : .save(input)
        case .error:
            // Cannot compare. Writing a non-empty value is an explicit user
            // edit; an empty field is indistinguishable from "failed to
            // load", so never delete.
            return input.isEmpty ? .none : .save(input)
        }
    }
}
