import CryptoKit
import Foundation

public struct ManagedAssetDescriptor: Codable, Equatable, Hashable, Sendable {
    public var id: String
    public var fileName: String
    public var mimeType: String?
    public var byteCount: Int64

    public var storedReference: String { ManagedAssetStore.referencePrefix + id }
}

public final class ManagedAssetStore: @unchecked Sendable {
    public static let shared = ManagedAssetStore()
    static let referencePrefix = "turbospark-asset:"

    private static let magic = Data("TSASET01".utf8)
    private static let chunkSize = 1_048_576
    private static let tagSize = 16
    private let lock = NSRecursiveLock()
    private let vault: ProfileVaultStore
    /// Test seam: called after each chunk is hashed or sealed, so a test can
    /// request a vault lock at a deterministic point in a large import.
    var chunkObserver: (() -> Void)?

    enum AssetError: Error, LocalizedError {
        case locked
        case malformed
        case tooLarge
        case missing
        case writeFailed

        var errorDescription: String? {
            switch self {
            case .locked: return "The profile must be unlocked to use managed files."
            case .malformed: return "The encrypted asset is malformed or was modified."
            case .tooLarge: return "The managed file is too large for this asset format."
            case .missing: return "The managed file is missing."
            case .writeFailed: return "The encrypted asset could not be written."
            }
        }
    }

    init(vault: ProfileVaultStore = .shared) {
        self.vault = vault
    }

    public func store(
        data: Data,
        fileName: String,
        mimeType: String? = nil
    ) throws -> ManagedAssetDescriptor {
        try lock.withLock {
            try withOperation { operation in
                let idKey = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
                    masterKey: operation.masterKey, purpose: "asset-id"))
                var hash = HMAC<SHA256>(key: idKey)
                for offset in stride(from: 0, to: data.count, by: Self.chunkSize) {
                    try operation.checkNotCancelled()
                    hash.update(data: data[(data.startIndex + offset)..<(data.startIndex + min(data.count, offset + Self.chunkSize))])
                    chunkObserver?()
                }
                let assetID = Data(hash.finalize()).hexEncodedString
                let descriptor = ManagedAssetDescriptor(
                    id: assetID, fileName: fileName, mimeType: mimeType, byteCount: Int64(data.count))
                let destination = assetURL(id: assetID)
                if FileManager.default.fileExists(atPath: destination.path) {
                    try operation.checkNotCancelled()
                    try operation.database.retainAsset(
                        id: assetID, fileName: fileName, mimeType: mimeType, byteCount: Int64(data.count))
                    return descriptor
                }
                guard data.count / Self.chunkSize < Int(UInt32.max) else { throw AssetError.tooLarge }
                let directory = destination.deletingLastPathComponent()
                try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
                try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
                let temporary = directory.appendingPathComponent(".\(assetID).\(UUID().uuidString).tmp")
                guard FileManager.default.createFile(
                    atPath: temporary.path, contents: nil, attributes: [.posixPermissions: 0o600])
                else { throw AssetError.writeFailed }
                do {
                    let output = try FileHandle(forWritingTo: temporary)
                    defer { try? output.close() }
                    let noncePrefix = try ProfileVaultCrypto.randomBytes(count: 8)
                    let header = makeHeader(plainByteCount: UInt64(data.count), noncePrefix: noncePrefix)
                    try output.write(contentsOf: header)
                    let key = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
                        masterKey: operation.masterKey, purpose: "asset", salt: Data(assetID.utf8)))
                    for offset in stride(from: 0, to: data.count, by: Self.chunkSize) {
                        try operation.checkNotCancelled()
                        let index = UInt32(offset / Self.chunkSize)
                        let sealed = try AES.GCM.seal(
                            data[(data.startIndex + offset)..<(data.startIndex + min(data.count, offset + Self.chunkSize))], using: key,
                            nonce: AES.GCM.Nonce(data: noncePrefix + index.bigEndianData),
                            authenticating: aad(header: header, assetID: assetID, chunkIndex: index))
                        try output.write(contentsOf: sealed.ciphertext)
                        try output.write(contentsOf: sealed.tag)
                        chunkObserver?()
                    }
                    try operation.checkNotCancelled()
                    try output.synchronize()
                    try FileManager.default.moveItem(at: temporary, to: destination)
                    do {
                        try operation.database.retainAsset(
                            id: assetID, fileName: fileName, mimeType: mimeType, byteCount: Int64(data.count))
                    } catch {
                        try? FileManager.default.removeItem(at: destination)
                        throw error
                    }
                    return descriptor
                } catch {
                    try? FileManager.default.removeItem(at: temporary)
                    throw error
                }
            }
        }
    }

    public func store(
        fileURL requestedURL: URL,
        fileName: String? = nil,
        mimeType: String? = nil
    ) throws -> ManagedAssetDescriptor {
        try lock.withLock {
            // The operation holds a key snapshot and keeps the vault from
            // locking or rotating until it returns; a lock request instead
            // cancels it at the next chunk boundary.
            try withOperation { operation in
            // attributesOfItem does not follow a terminal symlink, so stat the
            // resolved target: the header length must describe the bytes the
            // encrypt pass will actually read through FileHandle.
            let fileURL = requestedURL.resolvingSymlinksInPath()
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            // Both passes below read at most byteCount bytes, so a file that
            // grows mid-store cannot make the header disagree with the body.
            let assetID = try contentID(
                fileURL: fileURL, masterKey: operation.masterKey, limit: UInt64(byteCount),
                cancel: operation.checkNotCancelled)
            let name = fileName ?? requestedURL.lastPathComponent
            let descriptor = ManagedAssetDescriptor(
                id: assetID, fileName: name, mimeType: mimeType, byteCount: byteCount)
            let destination = assetURL(id: assetID)

            if FileManager.default.fileExists(atPath: destination.path) {
                try operation.checkNotCancelled()
                try operation.database.retainAsset(
                    id: assetID, fileName: name, mimeType: mimeType, byteCount: byteCount)
                return descriptor
            }

            guard UInt64(byteCount) / UInt64(Self.chunkSize) < UInt64(UInt32.max) else {
                throw AssetError.tooLarge
            }
            let directory = destination.deletingLastPathComponent()
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)

            let noncePrefix = try ProfileVaultCrypto.randomBytes(count: 8)
            let header = makeHeader(plainByteCount: UInt64(byteCount), noncePrefix: noncePrefix)
            let temporary = directory.appendingPathComponent(".\(assetID).\(UUID().uuidString).tmp")
            FileManager.default.createFile(atPath: temporary.path, contents: nil)
            do {
                let output = try FileHandle(forWritingTo: temporary)
                defer { try? output.close() }
                try output.write(contentsOf: header)
                let input = try FileHandle(forReadingFrom: fileURL)
                defer { try? input.close() }
                let assetKey = ProfileVaultCrypto.deriveKey(
                    masterKey: operation.masterKey,
                    purpose: "asset",
                    salt: Data(assetID.utf8))
                var index: UInt32 = 0
                var remaining = UInt64(byteCount)
                while remaining > 0,
                      let chunk = try input.read(upToCount: Int(min(UInt64(Self.chunkSize), remaining))),
                      !chunk.isEmpty {
                    try operation.checkNotCancelled()
                    remaining -= UInt64(chunk.count)
                    let nonceData = noncePrefix + index.bigEndianData
                    let nonce = try AES.GCM.Nonce(data: nonceData)
                    let sealed = try AES.GCM.seal(
                        chunk,
                        using: SymmetricKey(data: assetKey),
                        nonce: nonce,
                        authenticating: aad(
                            header: header, assetID: assetID, chunkIndex: index))
                    try output.write(contentsOf: sealed.ciphertext)
                    try output.write(contentsOf: sealed.tag)
                    index &+= 1
                    chunkObserver?()
                }
                // A file that shrank mid-store would leave a header promising
                // bytes the body does not have: refuse instead of storing it.
                guard remaining == 0 else { throw AssetError.writeFailed }
                try operation.checkNotCancelled()
                try output.synchronize()
                try FileManager.default.moveItem(at: temporary, to: destination)
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o600], ofItemAtPath: destination.path)
                do {
                    try operation.database.retainAsset(
                        id: assetID, fileName: name, mimeType: mimeType, byteCount: byteCount)
                } catch {
                    // No row will ever point at this ciphertext (it was created by
                    // this call; the dedup path returned earlier), and an orphan
                    // would poison every later exact backup.
                    try? FileManager.default.removeItem(at: destination)
                    throw error
                }
                return descriptor
            } catch {
                try? FileManager.default.removeItem(at: temporary)
                throw error
            }
            }
        }
    }

    public func descriptor(for reference: String) throws -> ManagedAssetDescriptor? {
        guard let id = Self.assetID(from: reference) else { return nil }
        let metadata: ProfileDatabase.AssetMetadata?
        do {
            metadata = try withOperation { try $0.database.assetMetadata(id: id) }
        } catch AssetError.locked {
            return nil
        }
        return metadata.map {
            ManagedAssetDescriptor(
                id: $0.id, fileName: $0.fileName, mimeType: $0.mimeType, byteCount: $0.byteCount)
        }
    }

    public func materializedURL(for reference: String) throws -> URL {
        try lock.withLock {
            guard let id = Self.assetID(from: reference) else { throw AssetError.missing }
            return try withOperation { operation in
                guard let metadata = try operation.database.assetMetadata(id: id)
                else { throw AssetError.missing }
                let cache = try decryptedCacheURL(profileID: operation.profileID)
                let safeName = ProfileBackup.sanitizedFileName(metadata.fileName, fallback: "asset")
                let destination = cache.appendingPathComponent("\(id)-\(safeName)")
                if FileManager.default.fileExists(atPath: destination.path) { return destination }
                try decrypt(id: id, to: destination, operation: operation)
                return destination
            }
        }
    }

    /// Decrypts one managed asset into a bounded consumer. Export uses this
    /// path so plaintext bytes flow directly into the ZIP entry and never
    /// exist as a staging file.
    public func streamDecrypted(
        reference: String,
        consume: (Data) throws -> Void
    ) throws {
        try lock.withLock {
            guard let id = Self.assetID(from: reference) else { throw AssetError.missing }
            try withOperation { operation in
                guard try operation.database.assetMetadata(id: id) != nil
                else { throw AssetError.missing }
                try decrypt(
                    id: id, masterKey: operation.masterKey,
                    cancel: operation.checkNotCancelled, consume: consume)
            }
        }
    }

    public func release(reference: String) throws {
        try lock.withLock {
            guard let id = Self.assetID(from: reference) else { return }
            do {
                try withOperation { operation in
                    if try operation.database.releaseAsset(id: id) {
                        try? FileManager.default.removeItem(at: assetURL(id: id))
                        removeCachedCopies(id: id, profileID: operation.profileID)
                    }
                }
            } catch AssetError.locked {
                // A locked vault has nothing to release.
            }
        }
    }

    /// Removes payloads whose metadata was deleted by a committed database
    /// transaction. Failure leaves only an unreadable orphan ciphertext.
    func garbageCollect(ids: [String]) {
        guard !ids.isEmpty else { return }
        lock.withLock {
            for id in ids {
                try? FileManager.default.removeItem(at: assetURL(id: id))
                if let profileID = try? withOperation({ $0.profileID }) {
                    removeCachedCopies(id: id, profileID: profileID)
                }
            }
        }
    }

    private func removeCachedCopies(id: String, profileID: String) {
        guard let cache = try? decryptedCacheURL(profileID: profileID) else { return }
        let prefix = id + "-"
        for item in (try? FileManager.default.contentsOfDirectory(
            at: cache, includingPropertiesForKeys: nil)) ?? []
        where item.lastPathComponent.hasPrefix(prefix) {
            try? FileManager.default.removeItem(at: item)
        }
    }

    public func purgeDecryptedCache() {
        guard let profileID = vault.session?.profileID else { return }
        Self.purgeDecryptedCache(profileID: profileID)
    }

    static func purgeDecryptedCache(profileID: String) {
        guard let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else { return }
        let cache = caches.appendingPathComponent("TurboSpark", isDirectory: true)
            .appendingPathComponent(profileID, isDirectory: true)
            .appendingPathComponent("decrypted", isDirectory: true)
        try? FileManager.default.removeItem(at: cache)
    }

    public static func assetID(from reference: String) -> String? {
        guard reference.hasPrefix(referencePrefix) else { return nil }
        let id = String(reference.dropFirst(referencePrefix.count))
        guard id.count == 64, id.allSatisfy({ $0.isHexDigit }) else { return nil }
        return id.lowercased()
    }

    /// Maps "no session" to the asset-level error callers already handle.
    private func withOperation<T>(_ body: (ProfileVaultOperation) throws -> T) throws -> T {
        do {
            return try vault.withOperation(body)
        } catch ProfileVaultStore.VaultError.locked {
            throw AssetError.locked
        }
    }

    private func decrypt(id: String, to destination: URL, operation: ProfileVaultOperation) throws {
        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        do {
            let output = try FileHandle(forWritingTo: temporary)
            defer { try? output.close() }
            try decrypt(
                id: id, masterKey: operation.masterKey, cancel: operation.checkNotCancelled
            ) { chunk in
                try output.write(contentsOf: chunk)
            }
            try output.synchronize()
            try FileManager.default.moveItem(at: temporary, to: destination)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: destination.path)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            // A lock request is not corruption; keep it distinguishable.
            if let vaultError = error as? ProfileVaultStore.VaultError, vaultError == .locked {
                throw AssetError.locked
            }
            throw AssetError.malformed
        }
    }

    private func decrypt(
        id: String,
        masterKey: Data,
        sourceURL: URL? = nil,
        cancel: () throws -> Void = {},
        consume: (Data) throws -> Void
    ) throws {
        let source = sourceURL ?? assetURL(id: id)
        guard FileManager.default.fileExists(atPath: source.path) else { throw AssetError.missing }
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let headerLength = Self.magic.count + 2 + 4 + 8 + 8
        guard let header = try input.read(upToCount: headerLength), header.count == headerLength,
              let parsed = parseHeader(header)
        else { throw AssetError.malformed }
        let assetKey = ProfileVaultCrypto.deriveKey(
            masterKey: masterKey, purpose: "asset", salt: Data(id.utf8))
        var remaining = parsed.plainByteCount
        var index: UInt32 = 0
        while remaining > 0 {
            try cancel()
            let plainCount = Int(min(UInt64(parsed.chunkSize), remaining))
            let sealedCount = plainCount + Self.tagSize
            guard let sealed = try input.read(upToCount: sealedCount), sealed.count == sealedCount else {
                throw AssetError.malformed
            }
            let nonce = try AES.GCM.Nonce(data: parsed.noncePrefix + index.bigEndianData)
            let box = try AES.GCM.SealedBox(
                nonce: nonce,
                ciphertext: sealed.prefix(plainCount),
                tag: sealed.suffix(Self.tagSize))
            let opened: Data
            do {
                opened = try AES.GCM.open(
                    box,
                    using: SymmetricKey(data: assetKey),
                    authenticating: aad(header: header, assetID: id, chunkIndex: index))
            } catch {
                throw AssetError.malformed
            }
            // Consumer failures, such as a full export volume, are not
            // authentication failures and must retain their original error.
            try consume(opened)
            remaining -= UInt64(plainCount)
            index &+= 1
        }
        guard (try input.read(upToCount: 1) ?? Data()).isEmpty else {
            throw AssetError.malformed
        }
    }

    /// Authenticates every chunk of a ciphertext file (the live copy or a
    /// staged snapshot of it) under `masterKey` without keeping plaintext.
    func authenticate(fileAt url: URL, id: String, masterKey: Data) throws {
        try decrypt(id: id, masterKey: masterKey, sourceURL: url) { _ in }
    }

    /// Re-encrypts every managed asset from `oldKey` to `newKey` for a vault
    /// key rotation. Replayable: an asset whose first chunk already opens
    /// under the new key is skipped, and each rewrite is an atomic rename, so
    /// every file is wholly under one key at any instant. The caller holds
    /// the vault gate exclusively, so no import runs concurrently.
    func rewrapAssets(from oldKey: Data, to newKey: Data, beforeEach: () throws -> Void) throws {
        for file in try Self.assetFiles(under: vault.assetsURL) {
            try beforeEach()
            let id = file.deletingPathExtension().lastPathComponent
            guard let outcome = try? rewrap(file: file, id: id, from: oldKey, to: newKey) else {
                // Unreadable under both keys (or the write failed): surface a
                // write failure, tolerate corruption that predates rotation.
                throw AssetError.writeFailed
            }
            if outcome == .unreadable {
                NSLog("Managed asset %@ opens under neither key; left unchanged", id)
            }
        }
    }

    private enum RewrapOutcome { case rewritten, alreadyCurrent, unreadable }

    private func rewrap(
        file: URL, id: String, from oldKey: Data, to newKey: Data
    ) throws -> RewrapOutcome {
        let headerLength = Self.magic.count + 2 + 4 + 8 + 8
        let probe = try FileHandle(forReadingFrom: file)
        let header = try probe.read(upToCount: headerLength) ?? Data()
        guard header.count == headerLength, let parsed = parseHeader(header) else {
            try? probe.close()
            return .unreadable
        }
        // An empty asset has no sealed chunk, hence nothing keyed.
        guard parsed.plainByteCount > 0 else {
            try? probe.close()
            return .alreadyCurrent
        }
        let firstPlain = Int(min(UInt64(parsed.chunkSize), parsed.plainByteCount))
        let sealed = try probe.read(upToCount: firstPlain + Self.tagSize) ?? Data()
        try? probe.close()
        guard sealed.count == firstPlain + Self.tagSize else { return .unreadable }
        func opens(_ master: Data) -> Bool {
            let key = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
                masterKey: master, purpose: "asset", salt: Data(id.utf8)))
            guard let nonce = try? AES.GCM.Nonce(data: parsed.noncePrefix + UInt32(0).bigEndianData),
                  let box = try? AES.GCM.SealedBox(
                    nonce: nonce, ciphertext: sealed.prefix(firstPlain),
                    tag: sealed.suffix(Self.tagSize))
            else { return false }
            return (try? AES.GCM.open(
                box, using: key,
                authenticating: aad(header: header, assetID: id, chunkIndex: 0))) != nil
        }
        if opens(newKey) { return .alreadyCurrent }
        guard opens(oldKey) else { return .unreadable }

        let noncePrefix = try ProfileVaultCrypto.randomBytes(count: 8)
        let newHeader = makeHeader(plainByteCount: parsed.plainByteCount, noncePrefix: noncePrefix)
        let newAssetKey = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
            masterKey: newKey, purpose: "asset", salt: Data(id.utf8)))
        let temporary = file.deletingLastPathComponent()
            .appendingPathComponent(".\(id).\(UUID().uuidString).rekey")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        do {
            let output = try FileHandle(forWritingTo: temporary)
            defer { try? output.close() }
            try output.write(contentsOf: newHeader)
            var index: UInt32 = 0
            try decrypt(id: id, masterKey: oldKey) { chunk in
                let nonce = try AES.GCM.Nonce(data: noncePrefix + index.bigEndianData)
                let box = try AES.GCM.seal(
                    chunk, using: newAssetKey, nonce: nonce,
                    authenticating: aad(header: newHeader, assetID: id, chunkIndex: index))
                try output.write(contentsOf: box.ciphertext)
                try output.write(contentsOf: box.tag)
                index &+= 1
            }
            try output.synchronize()
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: temporary.path)
            // rename(2) replaces the destination atomically; moveItem refuses
            // an existing target.
            guard rename(temporary.path, file.path) == 0 else { throw AssetError.writeFailed }
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw error
        }
        return .rewritten
    }

    private static func assetFiles(under root: URL) throws -> [URL] {
        guard let enumerator = FileManager.default.enumerator(atPath: root.path) else { return [] }
        var result: [URL] = []
        while let relative = enumerator.nextObject() as? String {
            let url = root.appendingPathComponent(relative)
            if url.pathExtension == "tsasset" { result.append(url) }
        }
        return result.sorted { $0.path < $1.path }
    }

    private func contentID(
        fileURL: URL, masterKey: Data, limit: UInt64, cancel: () throws -> Void
    ) throws -> String {
        let key = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
            masterKey: masterKey, purpose: "asset-id"))
        var hmac = HMAC<SHA256>(key: key)
        let input = try FileHandle(forReadingFrom: fileURL)
        defer { try? input.close() }
        var remaining = limit
        while remaining > 0,
              let chunk = try input.read(upToCount: Int(min(UInt64(Self.chunkSize), remaining))),
              !chunk.isEmpty {
            try cancel()
            hmac.update(data: chunk)
            remaining -= UInt64(chunk.count)
            chunkObserver?()
        }
        return Data(hmac.finalize()).hexEncodedString
    }

    private func assetURL(id: String) -> URL {
        vault.assetsURL
            .appendingPathComponent(String(id.prefix(2)), isDirectory: true)
            .appendingPathComponent("\(id).tsasset")
    }

    private func decryptedCacheURL(profileID: String) throws -> URL {
        let base = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first!
            .appendingPathComponent("TurboSpark", isDirectory: true)
            .appendingPathComponent(profileID, isDirectory: true)
            .appendingPathComponent("decrypted", isDirectory: true)
        try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: base.path)
        return base
    }

    private func makeHeader(plainByteCount: UInt64, noncePrefix: Data) -> Data {
        var data = Self.magic
        data.append(UInt16(1).bigEndianData)
        data.append(UInt32(Self.chunkSize).bigEndianData)
        data.append(plainByteCount.bigEndianData)
        data.append(noncePrefix)
        return data
    }

    private func parseHeader(_ data: Data) -> (chunkSize: UInt32, plainByteCount: UInt64, noncePrefix: Data)? {
        guard data.prefix(Self.magic.count) == Self.magic else { return nil }
        var offset = Self.magic.count
        guard readUInt16(data, offset: &offset) == 1,
              let chunkSize = readUInt32(data, offset: &offset),
              chunkSize == UInt32(Self.chunkSize),
              let byteCount = readUInt64(data, offset: &offset),
              offset + 8 == data.count
        else { return nil }
        return (chunkSize, byteCount, data.subdata(in: offset..<(offset + 8)))
    }

    private func aad(header: Data, assetID: String, chunkIndex: UInt32) -> Data {
        header + Data(assetID.utf8) + chunkIndex.bigEndianData
    }

    private func readUInt16(_ data: Data, offset: inout Int) -> UInt16? {
        guard offset + 2 <= data.count else { return nil }
        defer { offset += 2 }
        return data[offset..<(offset + 2)].reduce(0) { ($0 << 8) | UInt16($1) }
    }

    private func readUInt32(_ data: Data, offset: inout Int) -> UInt32? {
        guard offset + 4 <= data.count else { return nil }
        defer { offset += 4 }
        return data[offset..<(offset + 4)].reduce(0) { ($0 << 8) | UInt32($1) }
    }

    private func readUInt64(_ data: Data, offset: inout Int) -> UInt64? {
        guard offset + 8 <= data.count else { return nil }
        defer { offset += 8 }
        return data[offset..<(offset + 8)].reduce(0) { ($0 << 8) | UInt64($1) }
    }
}

private extension FixedWidthInteger {
    var bigEndianData: Data {
        var value = bigEndian
        return withUnsafeBytes(of: &value) { Data($0) }
    }
}

private extension Data {
    var hexEncodedString: String {
        map { String(format: "%02x", $0) }.joined()
    }
}

private extension NSRecursiveLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}
