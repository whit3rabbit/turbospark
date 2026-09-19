import CommonCrypto
import CryptoKit
import Foundation
import LocalAuthentication
import Security

enum ProfileVaultCrypto {
    static let masterKeyByteCount = 32
    static let saltByteCount = 16
    static let pbkdf2Rounds: UInt32 = 600_000
    static let minimumPassphraseLength = 15
    static let maximumPassphraseLength = 1_024

    enum CryptoError: Error, LocalizedError, Equatable {
        case invalidPassphraseLength
        case randomGenerationFailed(OSStatus)
        case keyDerivationFailed(Int32)
        case malformedEnvelope
        case authenticationFailed

        var errorDescription: String? {
            switch self {
            case .invalidPassphraseLength:
                return "Use a passphrase between 15 and 1,024 characters."
            case .randomGenerationFailed:
                return "Secure random data could not be generated."
            case .keyDerivationFailed:
                return "The passphrase key could not be derived."
            case .malformedEnvelope:
                return "The encrypted key envelope is malformed."
            case .authenticationFailed:
                return "The passphrase is incorrect or the encrypted key was modified."
            }
        }
    }

    static func normalizedPassphrase(_ passphrase: String) throws -> Data {
        let normalized = passphrase.precomposedStringWithCanonicalMapping
        guard normalized.count >= minimumPassphraseLength,
              normalized.count <= maximumPassphraseLength
        else { throw CryptoError.invalidPassphraseLength }
        return Data(normalized.utf8)
    }

    static func randomBytes(count: Int) throws -> Data {
        var data = Data(count: count)
        let status = data.withUnsafeMutableBytes { bytes in
            SecRandomCopyBytes(kSecRandomDefault, count, bytes.baseAddress!)
        }
        guard status == errSecSuccess else {
            throw CryptoError.randomGenerationFailed(status)
        }
        return data
    }

    static func derivePassphraseKey(
        passphrase: String,
        salt: Data,
        rounds: UInt32 = pbkdf2Rounds
    ) throws -> Data {
        let password = try normalizedPassphrase(passphrase)
        var derived = Data(count: masterKeyByteCount)
        let status = password.withUnsafeBytes { passwordBytes in
            salt.withUnsafeBytes { saltBytes in
                derived.withUnsafeMutableBytes { outputBytes in
                    CCKeyDerivationPBKDF(
                        CCPBKDFAlgorithm(kCCPBKDF2),
                        passwordBytes.bindMemory(to: Int8.self).baseAddress,
                        password.count,
                        saltBytes.bindMemory(to: UInt8.self).baseAddress,
                        salt.count,
                        CCPseudoRandomAlgorithm(kCCPRFHmacAlgSHA256),
                        rounds,
                        outputBytes.bindMemory(to: UInt8.self).baseAddress,
                        masterKeyByteCount)
                }
            }
        }
        guard status == kCCSuccess else {
            throw CryptoError.keyDerivationFailed(status)
        }
        return derived
    }

    static func wrapMasterKey(
        _ masterKey: Data,
        with wrappingKey: Data,
        context: String = "TurboSpark profile key v1"
    ) throws -> Data {
        let sealed = try AES.GCM.seal(
            masterKey,
            using: SymmetricKey(data: wrappingKey),
            authenticating: Data(context.utf8))
        guard let combined = sealed.combined else { throw CryptoError.malformedEnvelope }
        return combined
    }

    static func unwrapMasterKey(
        _ envelope: Data,
        with wrappingKey: Data,
        context: String = "TurboSpark profile key v1"
    ) throws -> Data {
        do {
            let box = try AES.GCM.SealedBox(combined: envelope)
            return try AES.GCM.open(
                box,
                using: SymmetricKey(data: wrappingKey),
                authenticating: Data(context.utf8))
        } catch {
            throw CryptoError.authenticationFailed
        }
    }

    static func deriveKey(masterKey: Data, purpose: String, salt: Data = Data()) -> Data {
        let key = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: SymmetricKey(data: masterKey),
            salt: salt,
            info: Data("TurboSpark/\(purpose)/v1".utf8),
            outputByteCount: masterKeyByteCount)
        return key.withUnsafeBytes { Data($0) }
    }

    static func contentID(for data: Data, masterKey: Data) -> String {
        let key = SymmetricKey(data: deriveKey(masterKey: masterKey, purpose: "asset-id"))
        let digest = HMAC<SHA256>.authenticationCode(for: data, using: key)
        return Data(digest).hexString
    }
}

extension Data {
    fileprivate var hexString: String {
        map { String(format: "%02x", $0) }.joined()
    }

    mutating func wipe() {
        resetBytes(in: startIndex..<endIndex)
        removeAll(keepingCapacity: false)
    }
}

protocol ProfileVaultKeychainProtocol: Sendable {
    func save(masterKey: Data, profileID: String) throws
    func load(profileID: String, context: LAContext) throws -> Data
    func delete(profileID: String)
    func canUseSystemAuthentication() -> Bool
}

struct ProfileVaultKeychain: ProfileVaultKeychainProtocol {
    private let service = "TurboSpark.profile-vault"

    enum KeychainError: Error, LocalizedError {
        case unavailable
        case accessControl
        case status(OSStatus)

        var errorDescription: String? {
            switch self {
            case .unavailable:
                return "Touch ID or Mac login authentication is not available."
            case .accessControl:
                return "The protected Keychain item could not be created."
            case .status(let status):
                return "Keychain operation failed (\(status))."
            }
        }
    }

    func canUseSystemAuthentication() -> Bool {
        let context = LAContext()
        var error: NSError?
        return context.canEvaluatePolicy(.deviceOwnerAuthentication, error: &error)
    }

    func save(masterKey: Data, profileID: String) throws {
        guard canUseSystemAuthentication() else { throw KeychainError.unavailable }
        delete(profileID: profileID)

        var accessError: Unmanaged<CFError>?
        guard let access = SecAccessControlCreateWithFlags(
            nil,
            kSecAttrAccessibleWhenUnlockedThisDeviceOnly,
            .userPresence,
            &accessError)
        else { throw KeychainError.accessControl }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: profileID,
            kSecAttrAccessControl as String: access,
            kSecUseDataProtectionKeychain as String: true,
            kSecValueData as String: masterKey,
        ]
        let status = SecItemAdd(query as CFDictionary, nil)
        guard status == errSecSuccess else { throw KeychainError.status(status) }
    }

    func load(profileID: String, context: LAContext) throws -> Data {
        context.localizedReason = "Unlock this TurboSpark profile"
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: profileID,
            kSecUseDataProtectionKeychain as String: true,
            kSecUseAuthenticationContext as String: context,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var value: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &value)
        guard status == errSecSuccess, let data = value as? Data else {
            throw KeychainError.status(status)
        }
        return data
    }

    func delete(profileID: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: profileID,
            kSecUseDataProtectionKeychain as String: true,
        ]
        SecItemDelete(query as CFDictionary)
    }
}
