import Compression
import Foundation

/// A bounded, in-process zip extractor for UNTRUSTED archives (ZP-05
/// `cdnZip`). System frameworks only (Compression for raw deflate), no
/// subprocess and no third-party library.
///
/// Why not `/usr/bin/unzip` or `ditto` (the ZP-08 and profile-backup paths):
/// both would write to disk themselves, and `unzip` happily creates the
/// symlinks and special files this extractor must refuse, treats an entry
/// name as a wildcard, and cannot tell the caller about a lying size table.
/// Here every decision is made from the central directory BEFORE a byte is
/// written, and every byte written is counted against a cap that does not
/// trust the archive's own numbers.
///
/// Policy (everything else is refused, nothing is skipped silently):
/// - stored (0) and deflate (8) members only; no encryption, no zip64, no
///   multi-disk archive;
/// - entry names that pass `ProfileBackupImport.isSafeArchiveEntry` (no
///   absolute path, `..`, backslash, control character), have no `.`
///   component, and are unique ignoring case (APFS folds case, so `A` and
///   `a` would overwrite each other);
/// - regular files and directories only: a Unix mode that says symlink,
///   device, fifo or socket is refused, so a plugin tree can never hold a
///   link that points out of the install;
/// - entry count, per-entry size, total expanded size and compression ratio
///   (declared AND actual) are capped, and compressed ranges may not overlap
///   (the overlapping-members bomb);
/// - the local header must agree with the central directory, and every
///   member's CRC-32 must match what was inflated.
///
/// Extraction never executes anything and never sets setuid/setgid/sticky
/// bits: a file is 0644, or 0755 when the archive marked it owner-executable.
enum ZipSafeExtractor {
    struct Limits: Equatable {
        var maxEntries = 2_000
        var maxEntryBytes = 16 * 1_024 * 1_024
        var maxTotalBytes = 64 * 1_024 * 1_024
        /// Expanded size over compressed size, per entry and overall.
        var maxRatio = 100
        var maxNameBytes = 512
        var maxPathDepth = 16
    }

    enum Failure: Error, Equatable, LocalizedError {
        case notAZip
        case unsupported(String)
        case unsafeEntry(String)
        case duplicateEntry(String)
        case tooManyEntries(Int)
        case entryTooLarge(String)
        case totalTooLarge
        case ratioExceeded(String)
        case corrupt(String)

        var errorDescription: String? {
            switch self {
            case .notAZip: return "The downloaded file is not a readable zip archive."
            case .unsupported(let why): return "The archive uses an unsupported feature: \(why)."
            case .unsafeEntry(let name): return "The archive holds an unsafe entry and was refused: \(name)."
            case .duplicateEntry(let name): return "The archive lists '\(name)' more than once."
            case .tooManyEntries(let n): return "The archive holds \(n) entries, over the limit."
            case .entryTooLarge(let name): return "The archive entry '\(name)' is over the size limit."
            case .totalTooLarge: return "The archive expands past the total size limit."
            case .ratioExceeded(let name): return "The archive entry '\(name)' expands suspiciously far (zip bomb guard)."
            case .corrupt(let why): return "The archive is corrupt: \(why)."
            }
        }
    }

    private struct Entry {
        let name: String
        let isDirectory: Bool
        let method: UInt16
        let crc: UInt32
        let compressedSize: Int
        let size: Int
        let localOffset: Int
        let executable: Bool
    }

    /// Extracts `archive` into `destination` (an existing, empty directory
    /// the caller owns, normally under a temp staging dir). On any failure
    /// the caller discards the directory; partial content may be present.
    static func extract(archive: URL, to destination: URL, limits: Limits = Limits()) throws {
        guard let data = try? Data(contentsOf: archive, options: .mappedIfSafe) else {
            throw Failure.notAZip
        }
        let entries = try parseDirectory(data, limits: limits)
        let fm = FileManager.default
        let root = destination.standardizedFileURL
        var written = 0
        let cdOffset = directoryOffset(data)
        for entry in entries {
            let target = root.appendingPathComponent(entry.name, isDirectory: entry.isDirectory)
                .standardizedFileURL
            // Belt and braces after the name policy: the resolved path must
            // still be inside the destination.
            guard target.path.hasPrefix(root.path + "/") else { throw Failure.unsafeEntry(entry.name) }
            if entry.isDirectory {
                try fm.createDirectory(
                    at: target, withIntermediateDirectories: true,
                    attributes: [.posixPermissions: 0o755])
                continue
            }
            let bytes = try inflate(entry, from: data, cdOffset: cdOffset)
            written += bytes.count
            guard written <= limits.maxTotalBytes else { throw Failure.totalTooLarge }
            try fm.createDirectory(
                at: target.deletingLastPathComponent(), withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o755])
            // withoutOverwriting: a name collision that slipped the checks
            // fails loudly instead of replacing a file.
            try bytes.write(to: target, options: .withoutOverwriting)
            try fm.setAttributes(
                [.posixPermissions: entry.executable ? 0o755 : 0o644], ofItemAtPath: target.path)
        }
    }

    // MARK: - Central directory

    private static func directoryOffset(_ data: Data) -> Int {
        guard let end = endRecord(data) else { return data.count }
        return Int(u32(data, end + 16))
    }

    private static func endRecord(_ data: Data) -> Int? {
        guard data.count >= 22 else { return nil }
        let floor = max(0, data.count - 22 - 65_535)
        var index = data.count - 22
        while index >= floor {
            if u32(data, index) == 0x0605_4B50 {
                // The record's own comment length must reach exactly to the
                // end of the file: a "second archive" appended after a
                // decoy end record is refused rather than guessed at.
                return index + 22 + Int(u16(data, index + 20)) == data.count ? index : nil
            }
            index -= 1
        }
        return nil
    }

    private static func parseDirectory(_ data: Data, limits: Limits) throws -> [Entry] {
        guard let end = endRecord(data) else { throw Failure.notAZip }
        let disk = u16(data, end + 4), cdDisk = u16(data, end + 6)
        let onDisk = u16(data, end + 8), count16 = u16(data, end + 10)
        let cdSize = u32(data, end + 12), cdOffset32 = u32(data, end + 16)
        guard disk == 0, cdDisk == 0, onDisk == count16 else {
            throw Failure.unsupported("multi-disk archive")
        }
        // 0xFFFF / 0xFFFFFFFF are the zip64 escape markers.
        guard count16 != 0xFFFF, cdSize != 0xFFFF_FFFF, cdOffset32 != 0xFFFF_FFFF else {
            throw Failure.unsupported("zip64")
        }
        let count = Int(count16)
        guard count > 0 else { throw Failure.corrupt("no entries") }
        guard count <= limits.maxEntries else { throw Failure.tooManyEntries(count) }
        let cdOffset = Int(cdOffset32)
        guard cdOffset + Int(cdSize) <= end else { throw Failure.corrupt("central directory out of range") }

        var entries: [Entry] = []
        var seenNames = Set<String>()
        var fileNames = Set<String>()
        var dirNames = Set<String>()
        var totalDeclared = 0
        var cursor = cdOffset
        for _ in 0..<count {
            guard cursor + 46 <= end, u32(data, cursor) == 0x0201_4B50 else {
                throw Failure.corrupt("bad central directory record")
            }
            let madeBy = u16(data, cursor + 4)
            let flags = u16(data, cursor + 8)
            let method = u16(data, cursor + 10)
            let crc = u32(data, cursor + 16)
            let compressed = u32(data, cursor + 20)
            let size = u32(data, cursor + 24)
            let nameLen = Int(u16(data, cursor + 28))
            let extraLen = Int(u16(data, cursor + 30))
            let commentLen = Int(u16(data, cursor + 32))
            let entryDisk = u16(data, cursor + 34)
            let external = u32(data, cursor + 38)
            let localOffset = u32(data, cursor + 42)
            let nameStart = cursor + 46
            guard nameStart + nameLen + extraLen + commentLen <= end, nameLen > 0 else {
                throw Failure.corrupt("central directory record runs past its end")
            }
            guard nameLen <= limits.maxNameBytes,
                let rawName = String(
                    data: data.subdata(in: nameStart..<(nameStart + nameLen)), encoding: .utf8)
            else { throw Failure.unsafeEntry("(name is not valid UTF-8 or is too long)") }
            cursor = nameStart + nameLen + extraLen + commentLen

            guard flags & 0x0001 == 0, flags & 0x0040 == 0 else { throw Failure.unsupported("encrypted entry") }
            guard entryDisk == 0 else { throw Failure.unsupported("multi-disk archive") }
            guard compressed != 0xFFFF_FFFF, size != 0xFFFF_FFFF, localOffset != 0xFFFF_FFFF else {
                throw Failure.unsupported("zip64")
            }
            guard method == 0 || method == 8 else { throw Failure.unsupported("compression method \(method)") }

            // A trailing slash marks a directory; the slash is not part of
            // the path policy below.
            let isDirectory = rawName.hasSuffix("/")
            let path = isDirectory ? String(rawName.dropLast()) : rawName
            guard ProfileBackupImport.isSafeArchiveEntry(path) else { throw Failure.unsafeEntry(rawName) }
            let components = path.split(separator: "/", omittingEmptySubsequences: false)
            guard !components.contains(where: { $0.isEmpty || $0 == "." || $0.utf8.count > 255 }),
                components.count <= limits.maxPathDepth
            else { throw Failure.unsafeEntry(rawName) }

            // Unix hosts (made-by high byte 3) carry st_mode in the high half
            // of the external attributes. Only regular files and directories
            // pass; a mode with no type bits (0) is treated by its trailing
            // slash. Non-Unix hosts have no symlink concept to refuse.
            var executable = false
            if madeBy >> 8 == 3 {
                let mode = external >> 16
                let type = mode & 0o170000
                switch type {
                case 0, 0o100000: break
                case 0o040000: guard isDirectory else { throw Failure.unsafeEntry(rawName) }
                default: throw Failure.unsafeEntry(rawName)
                }
                if type == 0o100000, isDirectory { throw Failure.unsafeEntry(rawName) }
                executable = mode & 0o100 != 0
            }

            let key = path.lowercased()
            guard seenNames.insert(key).inserted else { throw Failure.duplicateEntry(path) }
            // A file and a directory can not share a path prefix.
            for depth in 1..<max(components.count, 1) {
                let parent = components.prefix(depth).joined(separator: "/").lowercased()
                if fileNames.contains(parent) { throw Failure.unsafeEntry(rawName) }
                dirNames.insert(parent)
            }
            if isDirectory { dirNames.insert(key) } else {
                if dirNames.contains(key) { throw Failure.unsafeEntry(rawName) }
                fileNames.insert(key)
            }

            let compressedSize = Int(compressed), plainSize = Int(size)
            if !isDirectory {
                guard plainSize <= limits.maxEntryBytes else { throw Failure.entryTooLarge(path) }
                if method == 0 {
                    guard compressedSize == plainSize else { throw Failure.corrupt("stored size mismatch for \(path)") }
                } else if plainSize > 0 {
                    guard compressedSize > 0,
                        plainSize / compressedSize <= limits.maxRatio
                            || plainSize <= 1_024
                    else { throw Failure.ratioExceeded(path) }
                }
                totalDeclared += plainSize
                guard totalDeclared <= limits.maxTotalBytes else { throw Failure.totalTooLarge }
            } else if plainSize != 0 || compressedSize != 0 {
                throw Failure.unsafeEntry(rawName)
            }
            entries.append(Entry(
                name: path, isDirectory: isDirectory, method: method, crc: crc,
                compressedSize: compressedSize, size: plainSize,
                localOffset: Int(localOffset), executable: executable))
        }
        // Overall ratio, against the archive's real size.
        if totalDeclared > 1_024, totalDeclared / max(data.count, 1) > limits.maxRatio {
            throw Failure.ratioExceeded("(archive)")
        }
        try rejectOverlap(entries, data: data, cdOffset: cdOffset)
        return entries
    }

    /// Members whose data ranges overlap share compressed bytes: the classic
    /// bomb that expands a tiny archive far past its ratio. Each member's
    /// range is its local header plus data; they must be disjoint and must
    /// end before the central directory.
    private static func rejectOverlap(_ entries: [Entry], data: Data, cdOffset: Int) throws {
        var spans: [(Int, Int, String)] = []
        for entry in entries where !entry.isDirectory {
            let start = entry.localOffset
            guard start >= 0, start + 30 <= cdOffset, u32(data, start) == 0x0403_4B50 else {
                throw Failure.corrupt("bad local header for \(entry.name)")
            }
            let payloadStart = start + 30 + Int(u16(data, start + 26)) + Int(u16(data, start + 28))
            let payloadEnd = payloadStart + entry.compressedSize
            guard payloadEnd <= cdOffset else { throw Failure.corrupt("entry \(entry.name) runs past its data") }
            spans.append((start, payloadEnd, entry.name))
        }
        spans.sort { $0.0 < $1.0 }
        var previousEnd = 0
        for span in spans {
            guard span.0 >= previousEnd else { throw Failure.corrupt("entries overlap (\(span.2))") }
            previousEnd = span.1
        }
    }

    // MARK: - Member data

    private static func inflate(_ entry: Entry, from data: Data, cdOffset: Int) throws -> Data {
        let start = entry.localOffset
        guard start >= 0, start + 30 <= data.count, u32(data, start) == 0x0403_4B50 else {
            throw Failure.corrupt("bad local header for \(entry.name)")
        }
        let nameLen = Int(u16(data, start + 26)), extraLen = Int(u16(data, start + 28))
        let nameStart = start + 30
        guard nameStart + nameLen <= data.count,
            let localName = String(
                data: data.subdata(in: nameStart..<(nameStart + nameLen)), encoding: .utf8),
            localName == entry.name
        else { throw Failure.corrupt("local header disagrees with the directory for \(entry.name)") }
        let payloadStart = nameStart + nameLen + extraLen
        let payloadEnd = payloadStart + entry.compressedSize
        guard payloadEnd <= cdOffset, payloadEnd <= data.count else {
            throw Failure.corrupt("entry \(entry.name) runs past the archive")
        }
        let payload = [UInt8](data.subdata(in: payloadStart..<payloadEnd))
        var plain: [UInt8]
        switch entry.method {
        case 0:
            plain = payload
        default:
            if entry.size == 0 {
                plain = []
            } else {
                // One byte more than declared: a stream that inflates past
                // its declared size fills the extra byte and is refused,
                // whatever the size table claimed.
                var output = [UInt8](repeating: 0, count: entry.size + 1)
                let produced = compression_decode_buffer(
                    &output, output.count, payload, payload.count, nil, COMPRESSION_ZLIB)
                guard produced <= entry.size else { throw Failure.ratioExceeded(entry.name) }
                guard produced == entry.size else {
                    throw Failure.corrupt("entry \(entry.name) did not inflate to its declared size")
                }
                plain = Array(output.prefix(produced))
            }
        }
        guard plain.count == entry.size else { throw Failure.corrupt("size mismatch in \(entry.name)") }
        guard CRC32.checksum(plain) == entry.crc else { throw Failure.corrupt("CRC mismatch in \(entry.name)") }
        return Data(plain)
    }

    private static func u16(_ data: Data, _ i: Int) -> UInt16 {
        guard i >= 0, i + 2 <= data.count else { return 0 }
        let s = data.startIndex
        return UInt16(data[s + i]) | (UInt16(data[s + i + 1]) << 8)
    }

    private static func u32(_ data: Data, _ i: Int) -> UInt32 {
        guard i >= 0, i + 4 <= data.count else { return 0 }
        let s = data.startIndex
        return UInt32(data[s + i]) | (UInt32(data[s + i + 1]) << 8)
            | (UInt32(data[s + i + 2]) << 16) | (UInt32(data[s + i + 3]) << 24)
    }
}
