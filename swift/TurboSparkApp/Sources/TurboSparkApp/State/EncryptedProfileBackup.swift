import CryptoKit
import Foundation

enum EncryptedProfileBackup {
    static let formatVersion = 1
    static let manifestFileName = "exact-backup-manifest.json"
    static let fileExtension = "turbospark-profile"

    struct Checksum: Codable, Equatable, Sendable {
        var path: String
        var byteCount: UInt64
        var sha256: String
    }

    struct Manifest: Codable, Equatable, Sendable {
        var formatVersion: Int
        var kind: String
        var sourceProfileID: String
        var profileName: String
        var profileCreatedAt: Date
        var exportedAt: Date
        var appVersion: String
        var vaultFormatVersion: Int
        var databaseSchemaVersion: Int
        var contents: [Checksum]
    }

    enum BackupError: Error, LocalizedError, Equatable {
        case locked
        case wrongProfile
        case malformed
        case unsupportedVersion(Int)
        case checksumMismatch(String)
        case unexpectedEntry(String)
        case assetInventoryMismatch
        case integrityCheckFailed

        var errorDescription: String? {
            switch self {
            case .locked: return "Unlock this profile before exporting or restoring it."
            case .wrongProfile: return "Only the active unlocked profile can be exported."
            case .malformed: return "The encrypted profile backup is malformed."
            case .unsupportedVersion(let version):
                return "This backup uses unsupported format \(version)."
            case .checksumMismatch(let path): return "Backup checksum verification failed for \(path)."
            case .unexpectedEntry(let path): return "The backup contains an unexpected entry: \(path)."
            case .assetInventoryMismatch: return "The encrypted asset inventory does not match the database."
            case .integrityCheckFailed: return "The restored encrypted database failed its integrity check."
            }
        }
    }

    static let kind = "turbospark-encrypted-profile"

    static func export(
        profile: UserProfile,
        destination: URL,
        passphrase: String,
        appVersion: String,
        store: ProfileVaultStore = .shared,
        exportedAt: Date = Date()
    ) throws -> Manifest {
        guard let session = store.session else { throw BackupError.locked }
        guard session.profileID == profile.id else { throw BackupError.wrongProfile }
        let manager = FileManager.default
        let scratch = manager.temporaryDirectory
            .appendingPathComponent("turbospark-exact-export-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: scratch, withIntermediateDirectories: false)
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: scratch.path)
        defer { try? manager.removeItem(at: scratch) }
        let snapshot = scratch.appendingPathComponent("profile.sqlite3")
        let databaseKey = ProfileVaultCrypto.deriveKey(
            masterKey: session.masterKey, purpose: "database")
        try session.database.backup(to: snapshot, key: databaseKey)

        let encrypted = try ProfileEncryptedChunkWriter(
            destination: destination, passphrase: passphrase, masterKey: session.masterKey)
        let zip = ProfileZipStreamWriter { try encrypted.write($0) }
        do {
            var checksums: [Checksum] = []
            let databaseDigest = try zip.add(path: "vault/profile.sqlite3", fileURL: snapshot)
            checksums.append(Checksum(databaseDigest))
            for (root, prefix) in [(store.assetsURL, "vault/assets"),
                                   (store.recoveryURL, "vault/recovery")] {
                for file in try recursiveFiles(at: root) {
                    let relative = file.path.dropFirst(root.path.count)
                        .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                    let digest = try zip.add(path: "\(prefix)/\(relative)", fileURL: file)
                    checksums.append(Checksum(digest))
                }
            }
            let manifest = Manifest(
                formatVersion: formatVersion,
                kind: kind,
                sourceProfileID: profile.id,
                profileName: profile.name,
                profileCreatedAt: profile.createdAt,
                exportedAt: exportedAt,
                appVersion: appVersion,
                vaultFormatVersion: ProfileSecurityManifest.currentFormatVersion,
                databaseSchemaVersion: 1,
                contents: checksums.sorted { $0.path < $1.path })
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            _ = try zip.add(path: manifestFileName, data: encoder.encode(manifest))
            try zip.finish()
            try encrypted.finish()
            return manifest
        } catch {
            try? manager.removeItem(at: destination)
            throw error
        }
    }

    /// Restores into a fresh profile folder. The returned row uses the
    /// non-sensitive public label; the requested display name is encrypted
    /// inside the restored database before the folder becomes visible.
    static func restore(
        archive: URL,
        destination: URL,
        newProfileID: String,
        displayName: String,
        passphrase: String
    ) async throws -> (manifest: Manifest, profile: UserProfile) {
        let manager = FileManager.default
        let scratch = manager.temporaryDirectory
            .appendingPathComponent("turbospark-exact-import-\(UUID().uuidString)", isDirectory: true)
        try manager.createDirectory(at: scratch, withIntermediateDirectories: false)
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: scratch.path)
        defer { try? manager.removeItem(at: scratch) }
        let zipURL = scratch.appendingPathComponent("payload.zip")
        let masterKey = try ProfileEncryptedChunkWriter.decrypt(
            archive: archive, passphrase: passphrase, destination: zipURL)
        defer { var key = masterKey; key.wipe() }
        let entries = try await ProfileBackupImport.listEntries(archive: zipURL)
        try ProfileBackupImport.validateArchiveEntries(entries)
        for entry in entries where !isAllowedEntry(entry) {
            throw BackupError.unexpectedEntry(entry)
        }
        let extraction = scratch.appendingPathComponent("extracted", isDirectory: true)
        try manager.createDirectory(at: extraction, withIntermediateDirectories: false)
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: extraction.path)
        try await ProfileBackup.runDitto(
            arguments: ["-x", "-k", zipURL.path, extraction.path], step: "extract exact backup")

        let manifestURL = extraction.appendingPathComponent(manifestFileName)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let manifest = try? decoder.decode(
            Manifest.self, from: Data(contentsOf: manifestURL)), manifest.kind == kind
        else { throw BackupError.malformed }
        guard manifest.formatVersion == formatVersion else {
            throw BackupError.unsupportedVersion(manifest.formatVersion)
        }
        let declared = Set(manifest.contents.map(\.path))
        let actual = Set(entries.filter { $0 != manifestFileName && !$0.hasSuffix("/") })
        guard declared == actual else { throw BackupError.malformed }
        for item in manifest.contents {
            let file = extraction.appendingPathComponent(item.path)
            let digest = try fileDigest(file)
            guard digest.byteCount == item.byteCount, digest.sha256 == item.sha256 else {
                throw BackupError.checksumMismatch(item.path)
            }
        }

        let extractedVault = extraction.appendingPathComponent("vault", isDirectory: true)
        let temporaryManifest = ProfileSecurityManifest(
            formatVersion: ProfileSecurityManifest.currentFormatVersion,
            profileID: newProfileID,
            publicLabel: "Protected Profile",
            protectionMode: .local,
            kdf: nil,
            localMasterKey: masterKey,
            wrappedMasterKey: nil,
            quickUnlockEnabled: false,
            createdAt: manifest.profileCreatedAt,
            updatedAt: Date())
        try writeSecurityManifest(temporaryManifest, to: extractedVault)
        let verificationStore = ProfileVaultStore(
            rootProvider: { extractedVault },
            profileIDProvider: { newProfileID },
            migrateLegacyData: false)
        guard let verificationSession = try verificationStore.prepareForLaunch(),
              try verificationSession.database.integrityCheck()
        else { throw BackupError.integrityCheckFailed }
        let assets = try verificationSession.database.allAssetMetadata()
        let assetStore = ManagedAssetStore(vault: verificationStore)
        for asset in assets {
            try assetStore.streamDecrypted(reference: ManagedAssetDescriptor(
                id: asset.id,
                fileName: asset.fileName,
                mimeType: asset.mimeType,
                byteCount: asset.byteCount).storedReference) { _ in }
        }
        let assetIDsOnDisk = Set(try recursiveFiles(
            at: extractedVault.appendingPathComponent("assets", isDirectory: true))
            .filter { $0.pathExtension == "tsasset" }
            .map { $0.deletingPathExtension().lastPathComponent })
        guard assetIDsOnDisk == Set(assets.map(\.id)) else {
            throw BackupError.assetInventoryMismatch
        }
        try ProfileRepository(store: verificationStore).save(
            displayName, key: "profile:display-name")
        try verificationSession.database.checkpoint()
        verificationStore.lockVault()

        let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
        let wrappingKey = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: passphrase, salt: salt)
        let finalManifest = ProfileSecurityManifest(
            formatVersion: ProfileSecurityManifest.currentFormatVersion,
            profileID: newProfileID,
            publicLabel: "Protected Profile",
            protectionMode: .passphrase,
            kdf: .init(
                algorithm: "PBKDF2-HMAC-SHA256",
                rounds: ProfileVaultCrypto.pbkdf2Rounds,
                salt: salt),
            localMasterKey: nil,
            wrappedMasterKey: try ProfileVaultCrypto.wrapMasterKey(masterKey, with: wrappingKey),
            quickUnlockEnabled: false,
            createdAt: manifest.profileCreatedAt,
            updatedAt: Date())
        try writeSecurityManifest(finalManifest, to: extractedVault)

        try manager.createDirectory(at: destination, withIntermediateDirectories: false)
        try manager.setAttributes([.posixPermissions: 0o700], ofItemAtPath: destination.path)
        let installedVault = destination.appendingPathComponent("private-vault", isDirectory: true)
        do {
            try manager.moveItem(at: extractedVault, to: installedVault)
            try hardenTree(at: installedVault)
        } catch {
            try? manager.removeItem(at: destination)
            throw error
        }
        return (
            manifest,
            UserProfile(
                id: newProfileID,
                name: finalManifest.publicLabel,
                createdAt: manifest.profileCreatedAt,
                isProtected: true))
    }

    private static func isAllowedEntry(_ entry: String) -> Bool {
        if entry == manifestFileName { return true }
        guard entry.hasPrefix("vault/") else { return false }
        return entry == "vault/profile.sqlite3"
            || entry.hasPrefix("vault/assets/")
            || entry.hasPrefix("vault/recovery/")
    }

    private static func recursiveFiles(at root: URL) throws -> [URL] {
        guard FileManager.default.fileExists(atPath: root.path) else { return [] }
        let keys: [URLResourceKey] = [.isRegularFileKey]
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: keys,
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        var result: [URL] = []
        while let url = enumerator?.nextObject() as? URL {
            if (try url.resourceValues(forKeys: Set(keys))).isRegularFile == true {
                result.append(url)
            }
        }
        return result.sorted { $0.path < $1.path }
    }

    private static func fileDigest(_ url: URL) throws -> Checksum {
        var hash = SHA256()
        var count: UInt64 = 0
        let input = try FileHandle(forReadingFrom: url)
        defer { try? input.close() }
        while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
            hash.update(data: chunk)
            count += UInt64(chunk.count)
        }
        return Checksum(
            path: "", byteCount: count,
            sha256: Data(hash.finalize()).map { String(format: "%02x", $0) }.joined())
    }

    private static func writeSecurityManifest(
        _ manifest: ProfileSecurityManifest,
        to vault: URL
    ) throws {
        try FileManager.default.createDirectory(at: vault, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: vault.path)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let url = vault.appendingPathComponent("security.json")
        try encoder.encode(manifest).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }

    private static func hardenTree(at root: URL) throws {
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        let enumerator = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.isDirectoryKey], options: [])
        while let url = enumerator?.nextObject() as? URL {
            let isDirectory = try url.resourceValues(forKeys: [.isDirectoryKey]).isDirectory == true
            try FileManager.default.setAttributes(
                [.posixPermissions: isDirectory ? 0o700 : 0o600], ofItemAtPath: url.path)
        }
    }
}

private extension EncryptedProfileBackup.Checksum {
    init(_ digest: ProfileZipStreamWriter.EntryDigest) {
        self.init(path: digest.path, byteCount: digest.byteCount, sha256: digest.sha256)
    }
}
