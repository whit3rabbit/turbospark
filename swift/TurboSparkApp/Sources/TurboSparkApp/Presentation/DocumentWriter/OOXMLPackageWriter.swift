import Compression
import Foundation

// ZP-09: the in-process OOXML package writer. It replaces the read-only
// `/usr/bin/unzip` wrapper (`OfficeArchive`) on the WRITE side only: the
// writers never spawn a process. System frameworks only (Compression for raw
// deflate), no third-party zip library.
//
// The reader half here exists for the postcheck. It is deliberately NOT
// `OfficeArchive`: `unzip` reads its entry argument as a wildcard, so
// `[Content_Types].xml` cannot be requested through `OfficePackage`, and a
// check that shares the render path proves little about the writer.

// MARK: - CRC-32

enum CRC32 {
    private static let table: [UInt32] = (0..<256).map { index -> UInt32 in
        var value = UInt32(index)
        for _ in 0..<8 { value = (value & 1) == 1 ? (0xEDB8_8320 ^ (value >> 1)) : (value >> 1) }
        return value
    }

    static func checksum(_ bytes: [UInt8]) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in bytes { crc = table[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8) }
        return crc ^ 0xFFFF_FFFF
    }
}

// MARK: - XML text

enum OOXMLText {
    /// Escapes `& < > " '` and DROPS characters XML 1.0 forbids (C0 controls
    /// other than tab, LF, CR, and U+FFFE/U+FFFF). A model can emit a NUL or
    /// vertical tab; written verbatim it makes the part malformed and Word
    /// refuses the whole file.
    static func escape(_ text: String) -> String {
        var out = ""
        out.reserveCapacity(text.utf8.count)
        for scalar in text.unicodeScalars {
            switch scalar.value {
            case 0x09, 0x0A, 0x0D: out.unicodeScalars.append(scalar)
            case 0x00...0x1F, 0xFFFE, 0xFFFF: continue
            case 0x26: out += "&amp;"
            case 0x3C: out += "&lt;"
            case 0x3E: out += "&gt;"
            case 0x22: out += "&quot;"
            case 0x27: out += "&apos;"
            default: out.unicodeScalars.append(scalar)
            }
        }
        return out
    }

    static let declaration = "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>\n"
}

// MARK: - Zip writer

enum ZipArchiveWriter {
    static let maximumEntries = 512
    static let maximumArchiveBytes = 64 * 1_024 * 1_024

    /// A fixed DOS timestamp (1980-01-01 00:00) keeps output deterministic.
    private static let dosTime: UInt16 = 0
    private static let dosDate: UInt16 = 0x0021

    private static func le16(_ value: UInt16) -> [UInt8] { [UInt8(value & 0xFF), UInt8(value >> 8)] }
    private static func le32(_ value: UInt32) -> [UInt8] {
        [UInt8(value & 0xFF), UInt8((value >> 8) & 0xFF), UInt8((value >> 16) & 0xFF), UInt8(value >> 24)]
    }

    /// Raw deflate through Compression (`COMPRESSION_ZLIB` is raw deflate on
    /// Apple platforms, which is what a zip member holds). nil when it does
    /// not shrink the data, and the entry is stored.
    private static func deflate(_ bytes: [UInt8]) -> [UInt8]? {
        guard bytes.count > 64 else { return nil }
        var output = [UInt8](repeating: 0, count: bytes.count)
        let written = compression_encode_buffer(
            &output, output.count, bytes, bytes.count, nil, COMPRESSION_ZLIB)
        guard written > 0, written < bytes.count else { return nil }
        return Array(output.prefix(written))
    }

    /// Builds a zip. Names must be unique, relative, ASCII-safe paths.
    static func build(_ entries: [(name: String, data: Data)]) throws -> Data {
        guard entries.count <= maximumEntries else {
            throw DocumentWriteError.overLimit("the package would hold \(entries.count) parts (maximum \(maximumEntries))")
        }
        var seen: Set<String> = []
        var archive: [UInt8] = []
        var central: [UInt8] = []
        for entry in entries {
            guard OfficeRenderModel.isContainedName(entry.name), seen.insert(entry.name).inserted,
                  entry.name.utf8.count <= 255 else {
                throw DocumentWriteError.invalidSpec("internal: bad or duplicate part name '\(entry.name)'")
            }
            let bytes = [UInt8](entry.data)
            let crc = CRC32.checksum(bytes)
            let compressed = deflate(bytes)
            let method: UInt16 = compressed == nil ? 0 : 8
            let payload = compressed ?? bytes
            let name = [UInt8](entry.name.utf8)
            let offset = UInt32(archive.count)

            archive += le32(0x0403_4B50) + le16(20) + le16(0x0800) + le16(method)
            archive += le16(dosTime) + le16(dosDate) + le32(crc)
            archive += le32(UInt32(payload.count)) + le32(UInt32(bytes.count))
            archive += le16(UInt16(name.count)) + le16(0) + name + payload

            central += le32(0x0201_4B50) + le16(20) + le16(20) + le16(0x0800) + le16(method)
            central += le16(dosTime) + le16(dosDate) + le32(crc)
            central += le32(UInt32(payload.count)) + le32(UInt32(bytes.count))
            central += le16(UInt16(name.count)) + le16(0) + le16(0) + le16(0) + le16(0)
            central += le32(0) + le32(offset) + name

            guard archive.count + central.count <= maximumArchiveBytes else {
                throw DocumentWriteError.overLimit("the package is over \(maximumArchiveBytes / (1_024 * 1_024)) MiB")
            }
        }
        let directoryOffset = UInt32(archive.count)
        var tail = le32(0x0605_4B50) + le16(0) + le16(0)
        tail += le16(UInt16(entries.count)) + le16(UInt16(entries.count))
        tail += le32(UInt32(central.count)) + le32(directoryOffset) + le16(0)
        return Data(archive + central + tail)
    }
}

// MARK: - Zip reader (postcheck only)

/// A bounded in-process zip reader. Stored and deflate members only, no
/// encryption, no zip64, no data descriptors: exactly what `ZipArchiveWriter`
/// emits, so anything else is a failed postcheck.
struct ZipArchiveReader {
    struct Member: Equatable {
        let name: String
        let method: UInt16
        let crc: UInt32
        let compressedSize: Int
        let uncompressedSize: Int
        let headerOffset: Int
    }

    enum ReadError: Error, Equatable {
        case notAZip
        case unsupported(String)
        case corrupt(String)
        case tooLarge
    }

    let members: [Member]
    private let bytes: [UInt8]
    private let maximumEntryBytes: Int

    init(data: Data, maximumEntries: Int = 5_000, maximumEntryBytes: Int = 32 * 1_024 * 1_024) throws {
        bytes = [UInt8](data)
        self.maximumEntryBytes = maximumEntryBytes
        guard bytes.count >= 22 else { throw ReadError.notAZip }
        // The end record sits in the last 64 KiB plus 22 bytes (comment).
        var end = -1
        var index = bytes.count - 22
        let floor = max(0, bytes.count - 22 - 65_535)
        while index >= floor {
            if Self.u32(bytes, index) == 0x0605_4B50 { end = index; break }
            index -= 1
        }
        guard end >= 0 else { throw ReadError.notAZip }
        let count = Int(Self.u16(bytes, end + 10))
        let directorySize = Int(Self.u32(bytes, end + 12))
        let directoryOffset = Int(Self.u32(bytes, end + 16))
        guard count <= maximumEntries, directoryOffset + directorySize <= end else {
            throw ReadError.corrupt("central directory out of range")
        }
        var members: [Member] = []
        var cursor = directoryOffset
        for _ in 0..<count {
            guard cursor + 46 <= bytes.count, Self.u32(bytes, cursor) == 0x0201_4B50 else {
                throw ReadError.corrupt("bad central directory record")
            }
            let flags = Self.u16(bytes, cursor + 8)
            guard flags & 0x0001 == 0 else { throw ReadError.unsupported("encrypted member") }
            guard flags & 0x0008 == 0 else { throw ReadError.unsupported("data descriptor") }
            let method = Self.u16(bytes, cursor + 10)
            let nameLength = Int(Self.u16(bytes, cursor + 28))
            let extraLength = Int(Self.u16(bytes, cursor + 30))
            let commentLength = Int(Self.u16(bytes, cursor + 32))
            let nameStart = cursor + 46
            guard nameStart + nameLength <= bytes.count,
                  let name = String(bytes: bytes[nameStart..<(nameStart + nameLength)], encoding: .utf8) else {
                throw ReadError.corrupt("bad member name")
            }
            members.append(Member(
                name: name, method: method, crc: Self.u32(bytes, cursor + 16),
                compressedSize: Int(Self.u32(bytes, cursor + 20)),
                uncompressedSize: Int(Self.u32(bytes, cursor + 24)),
                headerOffset: Int(Self.u32(bytes, cursor + 42))))
            cursor = nameStart + nameLength + extraLength + commentLength
        }
        self.members = members
    }

    func member(_ name: String) -> Member? { members.first { $0.name == name } }

    /// The member's bytes, checked against its recorded CRC and size.
    func read(_ member: Member) throws -> Data {
        guard member.uncompressedSize <= maximumEntryBytes else { throw ReadError.tooLarge }
        let header = member.headerOffset
        guard header >= 0, header + 30 <= bytes.count, Self.u32(bytes, header) == 0x0403_4B50 else {
            throw ReadError.corrupt("bad local header for \(member.name)")
        }
        let start = header + 30 + Int(Self.u16(bytes, header + 26)) + Int(Self.u16(bytes, header + 28))
        guard start >= 0, start + member.compressedSize <= bytes.count else {
            throw ReadError.corrupt("member \(member.name) runs past the archive")
        }
        let payload = Array(bytes[start..<(start + member.compressedSize)])
        let plain: [UInt8]
        switch member.method {
        case 0:
            plain = payload
        case 8:
            var output = [UInt8](repeating: 0, count: max(member.uncompressedSize, 1))
            let produced = compression_decode_buffer(
                &output, output.count, payload, payload.count, nil, COMPRESSION_ZLIB)
            guard produced == member.uncompressedSize else {
                throw ReadError.corrupt("member \(member.name) did not inflate to its recorded size")
            }
            plain = Array(output.prefix(produced))
        default:
            throw ReadError.unsupported("compression method \(member.method)")
        }
        guard plain.count == member.uncompressedSize else { throw ReadError.corrupt("size mismatch in \(member.name)") }
        guard CRC32.checksum(plain) == member.crc else { throw ReadError.corrupt("CRC mismatch in \(member.name)") }
        return Data(plain)
    }

    func read(_ name: String) throws -> Data {
        guard let found = member(name) else { throw ReadError.corrupt("missing part \(name)") }
        return try read(found)
    }

    private static func u16(_ b: [UInt8], _ i: Int) -> UInt16 {
        guard i >= 0, i + 2 <= b.count else { return 0 }
        return UInt16(b[i]) | (UInt16(b[i + 1]) << 8)
    }

    private static func u32(_ b: [UInt8], _ i: Int) -> UInt32 {
        guard i >= 0, i + 4 <= b.count else { return 0 }
        return UInt32(b[i]) | (UInt32(b[i + 1]) << 8) | (UInt32(b[i + 2]) << 16) | (UInt32(b[i + 3]) << 24)
    }
}

// MARK: - Package

/// Collects parts, generates `[Content_Types].xml`, and serializes the zip.
/// Every XML part declares its type through an Override; binary parts go
/// through Default extensions. There is no API to add an external
/// relationship: `relationships(_:)` only emits internal targets.
struct OOXMLPackage {
    enum ContentType {
        static let relationships = "application/vnd.openxmlformats-package.relationships+xml"
        static let docx = "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"
        static let docxStyles = "application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"
        static let docxNumbering = "application/vnd.openxmlformats-officedocument.wordprocessingml.numbering+xml"
        static let xlsx = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"
        static let xlsxSheet = "application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"
        static let xlsxStyles = "application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"
        static let xlsxShared = "application/vnd.openxmlformats-officedocument.spreadsheetml.sharedStrings+xml"
        static let pptx = "application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"
        static let pptxSlide = "application/vnd.openxmlformats-officedocument.presentationml.slide+xml"
        static let pptxMaster = "application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"
        static let pptxLayout = "application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"
        static let theme = "application/vnd.openxmlformats-officedocument.theme+xml"
        static let core = "application/vnd.openxmlformats-package.core-properties+xml"
    }

    enum RelationshipType {
        private static let base = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
        static let officeDocument = base + "officeDocument"
        static let styles = base + "styles"
        static let numbering = base + "numbering"
        static let image = base + "image"
        static let worksheet = base + "worksheet"
        static let sharedStrings = base + "sharedStrings"
        static let slide = base + "slide"
        static let slideMaster = base + "slideMaster"
        static let slideLayout = base + "slideLayout"
        static let theme = base + "theme"
    }

    struct Part {
        let name: String
        let data: Data
        /// nil: covered by a Default extension entry.
        let overrideType: String?
    }

    private(set) var parts: [Part] = []
    private var defaults: [String: String] = [
        "rels": ContentType.relationships, "xml": "application/xml",
    ]

    /// `type` nil means a Default extension covers the part (`.rels`).
    mutating func addXML(_ name: String, _ xml: String, type: String?) {
        parts.append(Part(name: name, data: Data((OOXMLText.declaration + xml).utf8), overrideType: type))
    }

    /// A `.rels` part. Internal targets only: `Target` is a relative path to a
    /// part this package holds, never a URL.
    mutating func addRelationships(for source: String, _ relationships: [(id: String, type: String, target: String)]) {
        var xml = "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
        for rel in relationships {
            xml += "<Relationship Id=\"\(OOXMLText.escape(rel.id))\" Type=\"\(rel.type)\" Target=\"\(OOXMLText.escape(rel.target))\"/>"
        }
        xml += "</Relationships>"
        let components = source.split(separator: "/", omittingEmptySubsequences: true)
        let directory = components.dropLast().joined(separator: "/")
        let leaf = components.last.map(String.init) ?? ""
        let name = (directory.isEmpty ? "" : directory + "/") + "_rels/" + leaf + ".rels"
        addXML(name, xml, type: nil)
    }

    mutating func addBinary(_ name: String, _ data: Data, extension ext: String, contentType: String) {
        defaults[ext] = contentType
        parts.append(Part(name: name, data: data, overrideType: nil))
    }

    func contentTypesXML() -> String {
        var xml = "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
        for (ext, type) in defaults.sorted(by: { $0.key < $1.key }) {
            xml += "<Default Extension=\"\(ext)\" ContentType=\"\(type)\"/>"
        }
        for part in parts {
            if let type = part.overrideType {
                xml += "<Override PartName=\"/\(OOXMLText.escape(part.name))\" ContentType=\"\(type)\"/>"
            }
        }
        return xml + "</Types>"
    }

    func serialize() throws -> Data {
        var entries: [(name: String, data: Data)] = [
            ("[Content_Types].xml", Data((OOXMLText.declaration + contentTypesXML()).utf8)),
        ]
        entries += parts.map { ($0.name, $0.data) }
        return try ZipArchiveWriter.build(entries)
    }
}
