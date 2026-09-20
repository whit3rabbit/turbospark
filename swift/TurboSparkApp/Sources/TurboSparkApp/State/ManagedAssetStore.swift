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
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-asset-input-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: directory.path)
        defer { try? FileManager.default.removeItem(at: directory) }
        let temporary = directory.appendingPathComponent("input")
        try data.write(to: temporary, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: temporary.path)
        return try store(fileURL: temporary, fileName: fileName, mimeType: mimeType)
    }

    public func store(
        fileURL: URL,
        fileName: String? = nil,
        mimeType: String? = nil
    ) throws -> ManagedAssetDescriptor {
        try lock.withLock {
            guard let session = vault.session else { throw AssetError.locked }
            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            let byteCount = (attributes[.size] as? NSNumber)?.int64Value ?? 0
            let assetID = try contentID(fileURL: fileURL, masterKey: session.masterKey)
            let name = fileName ?? fileURL.lastPathComponent
            let descriptor = ManagedAssetDescriptor(
                id: assetID, fileName: name, mimeType: mimeType, byteCount: byteCount)
            let destination = assetURL(id: assetID)

            if FileManager.default.fileExists(atPath: destination.path) {
                try session.database.retainAsset(
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
                    masterKey: session.masterKey,
                    purpose: "asset",
                    salt: Data(assetID.utf8))
                var index: UInt32 = 0
                while let chunk = try input.read(upToCount: Self.chunkSize), !chunk.isEmpty {
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
                }
                try output.synchronize()
                try FileManager.default.moveItem(at: temporary, to: destination)
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o600], ofItemAtPath: destination.path)
                try session.database.retainAsset(
                    id: assetID, fileName: name, mimeType: mimeType, byteCount: byteCount)
                return descriptor
            } catch {
                try? FileManager.default.removeItem(at: temporary)
                throw error
            }
        }
    }

    public func descriptor(for reference: String) throws -> ManagedAssetDescriptor? {
        guard let id = Self.assetID(from: reference), let session = vault.session else { return nil }
        return try session.database.assetMetadata(id: id).map {
            ManagedAssetDescriptor(
                id: $0.id, fileName: $0.fileName, mimeType: $0.mimeType, byteCount: $0.byteCount)
        }
    }

    public func materializedURL(for reference: String) throws -> URL {
        try lock.withLock {
            guard let id = Self.assetID(from: reference),
                  let session = vault.session,
                  let metadata = try session.database.assetMetadata(id: id)
            else { throw AssetError.missing }
            let cache = try decryptedCacheURL(profileID: session.profileID)
            let safeName = ProfileBackup.sanitizedFileName(metadata.fileName, fallback: "asset")
            let destination = cache.appendingPathComponent("\(id)-\(safeName)")
            if FileManager.default.fileExists(atPath: destination.path) { return destination }
            try decrypt(id: id, to: destination, session: session)
            return destination
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
            guard let id = Self.assetID(from: reference),
                  let session = vault.session,
                  try session.database.assetMetadata(id: id) != nil
            else { throw AssetError.missing }
            try decrypt(id: id, session: session, consume: consume)
        }
    }

    public func release(reference: String) throws {
        try lock.withLock {
            guard let id = Self.assetID(from: reference), let session = vault.session else { return }
            if try session.database.releaseAsset(id: id) {
                try? FileManager.default.removeItem(at: assetURL(id: id))
                if let cache = try? decryptedCacheURL(profileID: session.profileID) {
                    let prefix = id + "-"
                    for item in (try? FileManager.default.contentsOfDirectory(
                        at: cache, includingPropertiesForKeys: nil)) ?? []
                    where item.lastPathComponent.hasPrefix(prefix) {
                        try? FileManager.default.removeItem(at: item)
                    }
                }
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
                guard let session = vault.session,
                      let cache = try? decryptedCacheURL(profileID: session.profileID)
                else { continue }
                let prefix = id + "-"
                for item in (try? FileManager.default.contentsOfDirectory(
                    at: cache, includingPropertiesForKeys: nil)) ?? []
                where item.lastPathComponent.hasPrefix(prefix) {
                    try? FileManager.default.removeItem(at: item)
                }
            }
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

    private func decrypt(id: String, to destination: URL, session: ProfileVaultSession) throws {
        let temporary = destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
        FileManager.default.createFile(atPath: temporary.path, contents: nil)
        do {
            let output = try FileHandle(forWritingTo: temporary)
            defer { try? output.close() }
            try decrypt(id: id, session: session) { chunk in
                try output.write(contentsOf: chunk)
            }
            try output.synchronize()
            try FileManager.default.moveItem(at: temporary, to: destination)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: destination.path)
        } catch {
            try? FileManager.default.removeItem(at: temporary)
            throw AssetError.malformed
        }
    }

    private func decrypt(
        id: String,
        session: ProfileVaultSession,
        consume: (Data) throws -> Void
    ) throws {
        let source = assetURL(id: id)
        guard FileManager.default.fileExists(atPath: source.path) else { throw AssetError.missing }
        let input = try FileHandle(forReadingFrom: source)
        defer { try? input.close() }
        let headerLength = Self.magic.count + 2 + 4 + 8 + 8
        guard let header = try input.read(upToCount: headerLength), header.count == headerLength,
              let parsed = parseHeader(header)
        else { throw AssetError.malformed }
        let assetKey = ProfileVaultCrypto.deriveKey(
            masterKey: session.masterKey, purpose: "asset", salt: Data(id.utf8))
        var remaining = parsed.plainByteCount
        var index: UInt32 = 0
        while remaining > 0 {
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

    private func contentID(fileURL: URL, masterKey: Data) throws -> String {
        let key = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
            masterKey: masterKey, purpose: "asset-id"))
        var hmac = HMAC<SHA256>(key: key)
        let input = try FileHandle(forReadingFrom: fileURL)
        defer { try? input.close() }
        while let chunk = try input.read(upToCount: Self.chunkSize), !chunk.isEmpty {
            hmac.update(data: chunk)
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
