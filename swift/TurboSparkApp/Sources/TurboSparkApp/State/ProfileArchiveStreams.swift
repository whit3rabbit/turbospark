import CryptoKit
import Foundation

/// Minimal streaming ZIP writer for profile exports. Entries use ZIP's
/// stored method so source files are copied in bounded chunks and never need
/// a plaintext staging directory. The writer intentionally rejects ZIP64
/// sized output rather than emitting a subtly incompatible archive.
final class ProfileZipStreamWriter {
    struct EntryDigest: Sendable {
        let path: String
        let byteCount: UInt64
        let sha256: String
    }

    enum StreamError: Error, LocalizedError {
        case invalidPath
        case archiveTooLarge
        case tooManyEntries

        var errorDescription: String? {
            switch self {
            case .invalidPath: return "An export entry has an unsafe path."
            case .archiveTooLarge: return "This export exceeds the standard ZIP size limit."
            case .tooManyEntries: return "This export has too many files for a standard ZIP."
            }
        }
    }

    private struct CentralEntry {
        let path: Data
        let crc32: UInt32
        let size: UInt32
        let offset: UInt32
        let time: UInt16
        let date: UInt16
    }

    private let sink: (Data) throws -> Void
    private var offset: UInt64 = 0
    private var entries: [CentralEntry] = []
    private var finished = false

    init(sink: @escaping (Data) throws -> Void) {
        self.sink = sink
    }

    @discardableResult
    func add(path: String, data: Data, modifiedAt: Date = Date()) throws -> EntryDigest {
        try add(path: path, modifiedAt: modifiedAt) { consume in
            try consume(data)
        }
    }

    @discardableResult
    func add(path: String, fileURL: URL, modifiedAt: Date? = nil) throws -> EntryDigest {
        let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
        let date = modifiedAt ?? attributes[.modificationDate] as? Date ?? Date()
        return try add(path: path, modifiedAt: date) { consume in
            let input = try FileHandle(forReadingFrom: fileURL)
            defer { try? input.close() }
            while let chunk = try input.read(upToCount: 1_048_576), !chunk.isEmpty {
                try consume(chunk)
            }
        }
    }

    @discardableResult
    func add(
        path: String,
        modifiedAt: Date = Date(),
        producer: (_ consume: (Data) throws -> Void) throws -> Void
    ) throws -> EntryDigest {
        guard !finished, Self.isSafe(path: path) else { throw StreamError.invalidPath }
        guard entries.count < Int(UInt16.max) else { throw StreamError.tooManyEntries }
        let name = Data(path.utf8)
        guard name.count <= Int(UInt16.max), offset <= UInt64(UInt32.max) else {
            throw StreamError.archiveTooLarge
        }
        let (dosTime, dosDate) = Self.dosDate(modifiedAt)
        let localOffset = UInt32(offset)
        var local = Data()
        local.appendLE(UInt32(0x0403_4b50))
        local.appendLE(UInt16(20))
        local.appendLE(UInt16(0x0808)) // UTF-8 and trailing data descriptor.
        local.appendLE(UInt16(0)) // Stored, no compression.
        local.appendLE(dosTime)
        local.appendLE(dosDate)
        local.appendLE(UInt32(0))
        local.appendLE(UInt32(0))
        local.appendLE(UInt32(0))
        local.appendLE(UInt16(name.count))
        local.appendLE(UInt16(0))
        local.append(name)
        try write(local)

        var crc = ProfileCRC32()
        var hash = SHA256()
        var size: UInt64 = 0
        try producer { chunk in
            guard size + UInt64(chunk.count) <= UInt64(UInt32.max) else {
                throw StreamError.archiveTooLarge
            }
            crc.update(chunk)
            hash.update(data: chunk)
            size += UInt64(chunk.count)
            try write(chunk)
        }
        let checksum = crc.finalized
        var descriptor = Data()
        descriptor.appendLE(UInt32(0x0807_4b50))
        descriptor.appendLE(checksum)
        descriptor.appendLE(UInt32(size))
        descriptor.appendLE(UInt32(size))
        try write(descriptor)
        entries.append(CentralEntry(
            path: name,
            crc32: checksum,
            size: UInt32(size),
            offset: localOffset,
            time: dosTime,
            date: dosDate))
        return EntryDigest(
            path: path,
            byteCount: size,
            sha256: Data(hash.finalize()).hexLowercase)
    }

    func finish() throws {
        guard !finished else { return }
        guard offset <= UInt64(UInt32.max) else { throw StreamError.archiveTooLarge }
        let centralOffset = UInt32(offset)
        for entry in entries {
            var record = Data()
            record.appendLE(UInt32(0x0201_4b50))
            record.appendLE(UInt16(0x0314)) // Unix creator, ZIP 2.0.
            record.appendLE(UInt16(20))
            record.appendLE(UInt16(0x0808))
            record.appendLE(UInt16(0))
            record.appendLE(entry.time)
            record.appendLE(entry.date)
            record.appendLE(entry.crc32)
            record.appendLE(entry.size)
            record.appendLE(entry.size)
            record.appendLE(UInt16(entry.path.count))
            record.appendLE(UInt16(0))
            record.appendLE(UInt16(0))
            record.appendLE(UInt16(0))
            record.appendLE(UInt16(0))
            record.appendLE(UInt32(0o100600 << 16))
            record.appendLE(entry.offset)
            record.append(entry.path)
            try write(record)
        }
        let centralSize = offset - UInt64(centralOffset)
        guard centralSize <= UInt64(UInt32.max) else { throw StreamError.archiveTooLarge }
        var end = Data()
        end.appendLE(UInt32(0x0605_4b50))
        end.appendLE(UInt16(0))
        end.appendLE(UInt16(0))
        end.appendLE(UInt16(entries.count))
        end.appendLE(UInt16(entries.count))
        end.appendLE(UInt32(centralSize))
        end.appendLE(centralOffset)
        end.appendLE(UInt16(0))
        try write(end)
        finished = true
    }

    private func write(_ data: Data) throws {
        guard offset + UInt64(data.count) <= UInt64(UInt32.max) else {
            throw StreamError.archiveTooLarge
        }
        try sink(data)
        offset += UInt64(data.count)
    }

    private static func isSafe(path: String) -> Bool {
        ProfileBackupImport.isSafeArchiveEntry(path)
            && !path.hasSuffix("/")
            && !path.split(separator: "/").contains(where: { $0.isEmpty || $0 == "." })
    }

    private static func dosDate(_ date: Date) -> (UInt16, UInt16) {
        let calendar = Calendar(identifier: .gregorian)
        let components = calendar.dateComponents(in: TimeZone.current, from: date)
        let year = min(max(components.year ?? 1980, 1980), 2107)
        let month = min(max(components.month ?? 1, 1), 12)
        let day = min(max(components.day ?? 1, 1), 31)
        let hour = min(max(components.hour ?? 0, 0), 23)
        let minute = min(max(components.minute ?? 0, 0), 59)
        let second = min(max(components.second ?? 0, 0), 59)
        return (
            UInt16((hour << 11) | (minute << 5) | (second / 2)),
            UInt16(((year - 1980) << 9) | (month << 5) | day))
    }
}

/// Authenticated chunk sink used by `.turbospark-profile`. Each record has
/// its own length and tag, followed by an authenticated zero-length terminal
/// record. This makes truncation and appended data unambiguous.
final class ProfileEncryptedChunkWriter {
    static let magic = Data("TSPRF001".utf8)
    static let version: UInt16 = 1
    static let chunkSize = 1_048_576

    struct Header: Sendable {
        let salt: Data
        let rounds: UInt32
        let noncePrefix: Data
        let wrappedMasterKey: Data
        let encoded: Data
    }

    private let handle: FileHandle
    private let key: SymmetricKey
    private let header: Header
    private var buffer = Data()
    private var index: UInt32 = 0
    private var finished = false

    init(destination: URL, passphrase: String, masterKey: Data) throws {
        let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
        let derived = try ProfileVaultCrypto.derivePassphraseKey(passphrase: passphrase, salt: salt)
        let wrapped = try ProfileVaultCrypto.wrapMasterKey(
            masterKey, with: derived, context: "TurboSpark profile backup key v1")
        let noncePrefix = try ProfileVaultCrypto.randomBytes(count: 8)
        var encoded = Self.magic
        encoded.appendBE(Self.version)
        encoded.appendBE(UInt32(Self.chunkSize))
        encoded.append(salt)
        encoded.appendBE(ProfileVaultCrypto.pbkdf2Rounds)
        encoded.append(noncePrefix)
        encoded.appendBE(UInt32(wrapped.count))
        encoded.append(wrapped)
        self.header = Header(
            salt: salt, rounds: ProfileVaultCrypto.pbkdf2Rounds,
            noncePrefix: noncePrefix, wrappedMasterKey: wrapped, encoded: encoded)
        self.key = SymmetricKey(data: derived)
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
        self.handle = try FileHandle(forWritingTo: destination)
        try handle.write(contentsOf: encoded)
    }

    deinit { try? handle.close() }

    func write(_ data: Data) throws {
        guard !finished else { return }
        buffer.append(data)
        while buffer.count >= Self.chunkSize {
            let chunk = buffer.prefix(Self.chunkSize)
            try seal(Data(chunk))
            buffer.removeFirst(Self.chunkSize)
        }
    }

    func finish() throws {
        guard !finished else { return }
        if !buffer.isEmpty { try seal(buffer); buffer.removeAll(keepingCapacity: false) }
        try seal(Data())
        try handle.synchronize()
        try handle.close()
        finished = true
    }

    private func seal(_ plain: Data) throws {
        let length = UInt32(plain.count)
        let lengthData = length.bigEndianBytes
        let nonce = try AES.GCM.Nonce(data: header.noncePrefix + index.bigEndianBytes)
        let box = try AES.GCM.seal(
            plain,
            using: key,
            nonce: nonce,
            authenticating: header.encoded + lengthData + index.bigEndianBytes)
        try handle.write(contentsOf: lengthData)
        try handle.write(contentsOf: box.ciphertext)
        try handle.write(contentsOf: box.tag)
        index &+= 1
    }

    static func decrypt(
        archive: URL,
        passphrase: String,
        destination: URL
    ) throws -> Data {
        let input = try FileHandle(forReadingFrom: archive)
        defer { try? input.close() }
        let fixedCount = magic.count + 2 + 4 + ProfileVaultCrypto.saltByteCount + 4 + 8 + 4
        guard let fixed = try input.read(upToCount: fixedCount), fixed.count == fixedCount else {
            throw ProfileVaultCrypto.CryptoError.malformedEnvelope
        }
        var cursor = 0
        guard fixed.prefix(magic.count) == magic else {
            throw ProfileVaultCrypto.CryptoError.malformedEnvelope
        }
        cursor += magic.count
        guard fixed.readBEUInt16(at: &cursor) == version,
              fixed.readBEUInt32(at: &cursor) == UInt32(chunkSize)
        else { throw ProfileVaultCrypto.CryptoError.malformedEnvelope }
        let salt = fixed.subdata(in: cursor..<(cursor + ProfileVaultCrypto.saltByteCount))
        cursor += ProfileVaultCrypto.saltByteCount
        guard let rounds = fixed.readBEUInt32(at: &cursor) else {
            throw ProfileVaultCrypto.CryptoError.malformedEnvelope
        }
        let noncePrefix = fixed.subdata(in: cursor..<(cursor + 8))
        cursor += 8
        guard let wrappedCount = fixed.readBEUInt32(at: &cursor), wrappedCount <= 4096,
              let wrapped = try input.read(upToCount: Int(wrappedCount)), wrapped.count == Int(wrappedCount)
        else { throw ProfileVaultCrypto.CryptoError.malformedEnvelope }
        let headerData = fixed + wrapped
        let derived = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: passphrase, salt: salt, rounds: rounds)
        let masterKey = try ProfileVaultCrypto.unwrapMasterKey(
            wrapped, with: derived, context: "TurboSpark profile backup key v1")
        let key = SymmetricKey(data: derived)
        FileManager.default.createFile(atPath: destination.path, contents: nil)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
        let output = try FileHandle(forWritingTo: destination)
        defer { try? output.close() }
        var index: UInt32 = 0
        var sawTerminal = false
        while true {
            guard let lengthData = try input.read(upToCount: 4), lengthData.count == 4 else {
                throw ProfileVaultCrypto.CryptoError.authenticationFailed
            }
            var lengthCursor = 0
            guard let length = lengthData.readBEUInt32(at: &lengthCursor), length <= UInt32(chunkSize),
                  let sealed = try input.read(upToCount: Int(length) + 16),
                  sealed.count == Int(length) + 16
            else { throw ProfileVaultCrypto.CryptoError.authenticationFailed }
            let nonce = try AES.GCM.Nonce(data: noncePrefix + index.bigEndianBytes)
            let box = try AES.GCM.SealedBox(
                nonce: nonce,
                ciphertext: sealed.prefix(Int(length)),
                tag: sealed.suffix(16))
            let plain: Data
            do {
                plain = try AES.GCM.open(
                    box, using: key,
                    authenticating: headerData + lengthData + index.bigEndianBytes)
            } catch {
                throw ProfileVaultCrypto.CryptoError.authenticationFailed
            }
            index &+= 1
            if length == 0 {
                sawTerminal = true
                break
            }
            try output.write(contentsOf: plain)
        }
        guard sawTerminal, (try input.read(upToCount: 1) ?? Data()).isEmpty else {
            throw ProfileVaultCrypto.CryptoError.authenticationFailed
        }
        try output.synchronize()
        return masterKey
    }
}

private struct ProfileCRC32 {
    private static let table: [UInt32] = (0..<256).map { value in
        var crc = UInt32(value)
        for _ in 0..<8 {
            crc = (crc & 1) == 1 ? (crc >> 1) ^ 0xedb8_8320 : crc >> 1
        }
        return crc
    }
    private var value: UInt32 = 0xffff_ffff
    mutating func update(_ data: Data) {
        for byte in data {
            value = Self.table[Int((value ^ UInt32(byte)) & 0xff)] ^ (value >> 8)
        }
    }
    var finalized: UInt32 { value ^ 0xffff_ffff }
}

private extension Data {
    mutating func appendLE<T: FixedWidthInteger>(_ value: T) {
        var copy = value.littleEndian
        append(Swift.withUnsafeBytes(of: &copy) { Data($0) })
    }
    mutating func appendBE<T: FixedWidthInteger>(_ value: T) {
        append(value.bigEndianBytes)
    }
    var hexLowercase: String { map { String(format: "%02x", $0) }.joined() }

    func readBEUInt16(at cursor: inout Int) -> UInt16? {
        guard cursor + 2 <= count else { return nil }
        defer { cursor += 2 }
        return self[cursor..<(cursor + 2)].reduce(0) { ($0 << 8) | UInt16($1) }
    }
    func readBEUInt32(at cursor: inout Int) -> UInt32? {
        guard cursor + 4 <= count else { return nil }
        defer { cursor += 4 }
        return self[cursor..<(cursor + 4)].reduce(0) { ($0 << 8) | UInt32($1) }
    }
}

private extension FixedWidthInteger {
    var bigEndianBytes: Data {
        var copy = bigEndian
        return withUnsafeBytes(of: &copy) { Data($0) }
    }
}
