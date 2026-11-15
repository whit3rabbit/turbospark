import Foundation

// Parse-to-render model for docx, xlsx and pptx (ZP-08).
//
// THE INPUT IS HOSTILE. A model can write any file and a user can attach
// anything, so every rule here assumes a crafted archive:
//
// - Reads go through `OfficeArchive` (system `unzip -p`, per-entry byte cap,
//   deadline). Nothing is ever extracted to disk, so zip-slip and symlink
//   entries are inert; the table of contents is still screened and an archive
//   naming `..` or an absolute path is refused outright.
// - Size, entry count, declared expansion and expansion ratio are checked
//   from the table of contents before any entry is decompressed.
// - XML is parsed with external entities off and any DOCTYPE or ENTITY
//   declaration refused, so there is no XXE and no entity expansion.
// - Relationships resolve only to entries inside the archive. A relationship
//   marked external is recorded and NEVER followed: there is no network
//   access from this layer.
// - Macros are never executed or even read (`vbaProject.bin` is only noted).
// - Every unbounded list has a cap with a truncation flag, never a silent cut.
//
// Any failure is a typed `OfficeRenderError`. The view turns it into a
// bounded error card plus the system QuickLook preview, so a bad file costs
// the user fidelity and never the preview.

/// Numbers for every bound in this layer. One place, so a test can shrink one
/// and prove the guard it feeds.
struct OfficeRenderLimits: Sendable {
    static let `default` = OfficeRenderLimits()

    /// In-app rendering ceiling for the file on disk. Above it the pane
    /// hands off to the system preview and external opener.
    var maximumFileBytes = 50 * 1_024 * 1_024
    var maximumEntries = 5_000
    /// Sum of the sizes the archive CLAIMS to expand to.
    var maximumDeclaredBytes = 256 * 1_024 * 1_024
    /// Declared expansion over file size, checked once the claim passes
    /// `ratioFloorBytes` so a tiny honest file is never refused on ratio.
    var maximumRatio = 200
    var ratioFloorBytes = 8 * 1_024 * 1_024
    var maximumPartBytes = 16 * 1_024 * 1_024
    var maximumTotalBytes = 96 * 1_024 * 1_024
    var timeout: TimeInterval = 20

    var maximumXMLNodes = 400_000
    var maximumXMLDepth = 64
    var maximumNodeTextCharacters = 100_000

    var maximumDocxBlocks = 4_000
    var maximumDocxTableCells = 10_000
    var maximumDocxImages = 16
    var maximumImageBytes = 8 * 1_024 * 1_024

    var maximumSheets = 32
    var maximumSheetRows = 1_000
    var maximumSheetColumns = 64
    var maximumSheetCells = 30_000
    var maximumSharedStrings = 200_000
    var maximumCellCharacters = 4_000

    var maximumSlides = 200
    var maximumShapesPerSlide = 300
    var maximumTableCells = 2_000

    var archiveLimits: DocumentTextExtractor.Limits {
        var limits = DocumentTextExtractor.Limits()
        limits.maximumEntryBytes = maximumPartBytes
        limits.maximumSelectedBytes = maximumTotalBytes
        limits.timeout = timeout
        return limits
    }
}

/// Why a document was not rendered. The view shows a localized body per
/// `Category` and the part name verbatim; nothing here carries file content.
enum OfficeRenderError: Error, Equatable {
    case unsupportedExtension
    /// File, entry count, declared expansion, ratio, or a part over its cap.
    case tooLarge
    /// A table-of-contents entry naming a parent directory or absolute path,
    /// or a symlink standing where the file should be.
    case unsafeArchive
    case notAnArchive
    /// OLE compound header: a password-protected OOXML file.
    case encrypted
    case malformedXML(part: String)
    /// A DOCTYPE or ENTITY declaration. Never resolved, never tolerated.
    case forbiddenDeclaration(part: String)
    case missingPart(String)
    case timedOut
    case unreadable

    enum Category: Equatable {
        case overLimits
        case damaged
    }

    var category: Category {
        switch self {
        case .tooLarge, .timedOut: return .overLimits
        default: return .damaged
        }
    }

    /// The archive part the failure names, for the verbatim detail line.
    var part: String? {
        switch self {
        case .malformedXML(let part), .forbiddenDeclaration(let part), .missingPart(let part):
            return String(part.prefix(120))
        default:
            return nil
        }
    }
}

enum OfficeFormat: Equatable {
    case docx, xlsx, pptx

    /// The extensions routed to the native viewers. Macro-enabled variants
    /// are included and carry a notice; legacy binary `.doc/.xls/.ppt` are
    /// not OOXML and stay on the system preview.
    static let extensions: Set<String> = ["docx", "docm", "xlsx", "xlsm", "pptx", "pptm"]

    static func forExtension(_ ext: String) -> OfficeFormat? {
        switch ext.lowercased() {
        case "docx", "docm": return .docx
        case "xlsx", "xlsm": return .xlsx
        case "pptx", "pptm": return .pptx
        default: return nil
        }
    }
}

/// What a viewer footer needs beyond the content.
struct OfficeRenderNotes: Equatable {
    /// A `vbaProject.bin` exists. Never read, never run.
    var hasMacros = false
    /// A cap cut the content.
    var truncated = false
    var unrenderedCharts = 0
    /// Relationships marked external were seen and left unfollowed.
    var externalReferences = 0
}

enum OfficeDocument {
    case docx(DocxDocument)
    case xlsx(XlsxDocument)
    case pptx(PptxDocument)

    var notes: OfficeRenderNotes {
        switch self {
        case .docx(let document): return document.notes
        case .xlsx(let document): return document.notes
        case .pptx(let document): return document.notes
        }
    }
}

// MARK: - Loader

enum OfficeRenderModel {
    /// Parses `url` into a render model. Synchronous and bounded; call it off
    /// the main actor. Read-only by construction: the only things touched are
    /// a read of the source and a private temp copy `OfficeArchive` owns.
    static func load(
        url: URL,
        limits: OfficeRenderLimits = .default
    ) throws -> OfficeDocument {
        guard let format = OfficeFormat.forExtension(url.pathExtension) else {
            throw OfficeRenderError.unsupportedExtension
        }
        try screenSource(url, limits: limits)

        let archive: OfficeArchive
        do {
            archive = try OfficeArchive(url: url, limits: limits.archiveLimits)
        } catch let error as DocumentTextExtractionError {
            switch error {
            case .documentTooLarge: throw OfficeRenderError.tooLarge
            case .extractionTimedOut: throw OfficeRenderError.timedOut
            default: throw OfficeRenderError.notAnArchive
            }
        }
        let fileSize = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int) ?? 0
        try screenArchive(archive, fileSize: fileSize, limits: limits)

        let package = OfficePackage(
            archive: archive,
            limits: limits,
            deadline: Date().addingTimeInterval(limits.timeout))
        var document: OfficeDocument
        switch format {
        case .docx: document = .docx(try DocxDocument.parse(package))
        case .xlsx: document = .xlsx(try XlsxDocument.parse(package))
        case .pptx: document = .pptx(try PptxDocument.parse(package))
        }
        // Macros are only NOTED from the table of contents.
        let hasMacros = archive.entries.contains { $0.lowercased().hasSuffix("vbaproject.bin") }
        if hasMacros {
            switch document {
            case .docx(var value): value.notes.hasMacros = true; document = .docx(value)
            case .xlsx(var value): value.notes.hasMacros = true; document = .xlsx(value)
            case .pptx(var value): value.notes.hasMacros = true; document = .pptx(value)
            }
        }
        return document
    }

    /// The source must be a regular file (not a symlink standing in for one),
    /// within the size ceiling, and start with the zip signature. An OLE
    /// header is how Office stores a password-protected OOXML file.
    static func screenSource(_ url: URL, limits: OfficeRenderLimits) throws {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { throw OfficeRenderError.unreadable }
        let kind = info.st_mode & S_IFMT
        if kind == S_IFLNK { throw OfficeRenderError.unsafeArchive }
        guard kind == S_IFREG else { throw OfficeRenderError.unreadable }
        guard Int64(info.st_size) <= Int64(limits.maximumFileBytes) else {
            throw OfficeRenderError.tooLarge
        }
        guard let handle = try? FileHandle(forReadingFrom: url) else {
            throw OfficeRenderError.unreadable
        }
        defer { try? handle.close() }
        let head = (try? handle.read(upToCount: 8)) ?? Data()
        if head.starts(with: [0xD0, 0xCF, 0x11, 0xE0]) { throw OfficeRenderError.encrypted }
        guard head.starts(with: [0x50, 0x4B, 0x03, 0x04]) else {
            throw OfficeRenderError.notAnArchive
        }
    }

    static func screenArchive(
        _ archive: OfficeArchive,
        fileSize: Int,
        limits: OfficeRenderLimits
    ) throws {
        guard archive.entries.count <= limits.maximumEntries else {
            throw OfficeRenderError.tooLarge
        }
        for name in archive.entries where !isContainedName(name) {
            throw OfficeRenderError.unsafeArchive
        }
        var declared = 0
        for size in archive.entrySizes.values {
            let (sum, overflow) = declared.addingReportingOverflow(size)
            if overflow { throw OfficeRenderError.tooLarge }
            declared = sum
        }
        guard declared <= limits.maximumDeclaredBytes else { throw OfficeRenderError.tooLarge }
        if declared > limits.ratioFloorBytes, declared / max(1, fileSize) > limits.maximumRatio {
            throw OfficeRenderError.tooLarge
        }
    }

    /// A table-of-contents name that stays inside the archive: relative, no
    /// parent-directory component, no backslash, no NUL.
    static func isContainedName(_ name: String) -> Bool {
        if name.isEmpty || name.hasPrefix("/") || name.contains("\\") || name.contains("\u{0}") {
            return false
        }
        return !name.split(separator: "/", omittingEmptySubsequences: false).contains("..")
    }
}

// MARK: - Package access

/// One OOXML package open for reading: parts as XML trees, relationships
/// resolved inside the archive only.
struct OfficePackage {
    let archive: OfficeArchive
    let limits: OfficeRenderLimits
    let deadline: Date

    struct Relationship {
        let id: String
        /// Last path component of the relationship type, e.g. `image`.
        let kind: String
        /// Resolved archive entry, nil when external or outside the archive.
        let entry: String?
        let isExternal: Bool
    }

    func checkDeadline() throws {
        if Date() > deadline { throw OfficeRenderError.timedOut }
    }

    func has(_ entry: String) -> Bool { archive.entries.contains(entry) && Self.isReadable(entry) }

    /// `unzip` reads its entry argument as a wildcard pattern. A real entry
    /// named with a metacharacter could match several entries and concatenate
    /// them, so such names are never requested.
    static func isReadable(_ entry: String) -> Bool {
        !entry.contains(where: { "*?[]".contains($0) })
    }

    func data(_ entry: String) throws -> Data {
        try checkDeadline()
        guard has(entry) else { throw OfficeRenderError.missingPart(entry) }
        do {
            return try archive.data(for: entry)
        } catch let error as DocumentTextExtractionError {
            switch error {
            case .documentTooLarge: throw OfficeRenderError.tooLarge
            case .extractionTimedOut: throw OfficeRenderError.timedOut
            default: throw OfficeRenderError.unreadable
            }
        }
    }

    func xml(_ entry: String) throws -> XMLNode {
        try OfficeXMLTree.parse(try data(entry), part: entry, limits: limits)
    }

    func optionalXML(_ entry: String) throws -> XMLNode? {
        has(entry) ? try xml(entry) : nil
    }

    /// Relationships of `part`, keyed by id. A missing `.rels` is empty.
    func relationships(of part: String) throws -> [String: Relationship] {
        let components = part.split(separator: "/", omittingEmptySubsequences: false)
        let name = components.last.map(String.init) ?? part
        let directory = components.dropLast().joined(separator: "/")
        let relsEntry = (directory.isEmpty ? "" : directory + "/") + "_rels/" + name + ".rels"
        guard let root = try optionalXML(relsEntry) else { return [:] }
        var result: [String: Relationship] = [:]
        for node in root.children where node.name == "Relationship" {
            guard let id = node.attrs["Id"], let target = node.attrs["Target"] else { continue }
            let kind = (node.attrs["Type"] ?? "").split(separator: "/").last.map(String.init) ?? ""
            let external = (node.attrs["TargetMode"] ?? "").lowercased() == "external"
            let entry = external ? nil : resolve(target, relativeTo: part)
            result[id] = Relationship(id: id, kind: kind, entry: entry, isExternal: external)
        }
        return result
    }

    /// The main document part named by the package's root relationships
    /// (`relationships(of: "")` reads `_rels/.rels`),
    /// falling back to the conventional path.
    func mainPart(fallback: String) throws -> String {
        let rels = try relationships(of: "")
        if let entry = rels.values.first(where: { $0.kind == "officeDocument" })?.entry,
           has(entry) {
            return entry
        }
        return fallback
    }

    /// Resolves a relationship target against its source part, staying inside
    /// the archive. Returns nil for an escape, an absent entry, or a name
    /// `unzip` would read as a pattern.
    func resolve(_ target: String, relativeTo part: String) -> String? {
        var parts: [Substring]
        if target.hasPrefix("/") {
            parts = []
        } else {
            parts = part.split(separator: "/", omittingEmptySubsequences: true).dropLast().map { $0 }
        }
        for piece in target.split(separator: "/", omittingEmptySubsequences: true) {
            if piece == "." { continue }
            if piece == ".." {
                guard !parts.isEmpty else { return nil }
                parts.removeLast()
            } else {
                parts.append(piece)
            }
        }
        let entry = parts.joined(separator: "/")
        return has(entry) ? entry : nil
    }
}

// MARK: - XML tree

/// A parsed XML element. Names and attribute keys are LOCAL (prefix stripped).
final class XMLNode {
    let name: String
    var attrs: [String: String]
    var children: [XMLNode] = []
    var text = ""

    init(name: String, attrs: [String: String]) {
        self.name = name
        self.attrs = attrs
    }

    func child(_ name: String) -> XMLNode? { children.first { $0.name == name } }
    func all(_ name: String) -> [XMLNode] { children.filter { $0.name == name } }

    /// Boolean toggle property: present with no value or a truthy value.
    func toggle(_ name: String) -> Bool? {
        guard let node = child(name) else { return nil }
        guard let value = node.attrs["val"]?.lowercased() else { return true }
        return !["0", "false", "off", "none"].contains(value)
    }
}

/// Bounded tree builder over `XMLParser` with external entities off.
enum OfficeXMLTree {
    static func parse(_ data: Data, part: String, limits: OfficeRenderLimits) throws -> XMLNode {
        // A DOCTYPE or ENTITY declaration has no place in OOXML. Refusing it
        // outright is simpler and safer than configuring around it. UTF-16
        // would hide the marker from a byte scan, and OOXML parts are UTF-8,
        // so a NUL byte or UTF-16 BOM near the start is refused too.
        if data.prefix(4).starts(with: [0xFF, 0xFE]) || data.prefix(4).starts(with: [0xFE, 0xFF])
            || data.prefix(64).contains(0) {
            throw OfficeRenderError.malformedXML(part: part)
        }
        let doctype = Data("<!DOCTYPE".utf8)
        let entity = Data("<!ENTITY".utf8)
        if data.range(of: doctype) != nil || data.range(of: entity) != nil {
            throw OfficeRenderError.forbiddenDeclaration(part: part)
        }
        let builder = Builder(limits: limits)
        let parser = XMLParser(data: data)
        parser.shouldResolveExternalEntities = false
        parser.shouldProcessNamespaces = false
        parser.delegate = builder
        let ok = parser.parse()
        if builder.exceededBounds { throw OfficeRenderError.tooLarge }
        guard ok, let root = builder.root else {
            throw OfficeRenderError.malformedXML(part: part)
        }
        return root
    }

    private final class Builder: NSObject, XMLParserDelegate {
        let limits: OfficeRenderLimits
        var root: XMLNode?
        var stack: [XMLNode] = []
        var count = 0
        var exceededBounds = false

        init(limits: OfficeRenderLimits) { self.limits = limits }

        func parser(
            _ parser: XMLParser,
            didStartElement elementName: String,
            namespaceURI: String?,
            qualifiedName qName: String?,
            attributes attributeDict: [String: String] = [:]
        ) {
            count += 1
            if count > limits.maximumXMLNodes || stack.count >= limits.maximumXMLDepth {
                exceededBounds = true
                parser.abortParsing()
                return
            }
            var attrs: [String: String] = [:]
            for (key, value) in attributeDict { attrs[localName(key)] = value }
            let node = XMLNode(name: localName(elementName), attrs: attrs)
            if let parent = stack.last {
                parent.children.append(node)
            } else if root == nil {
                root = node
            }
            stack.append(node)
        }

        func parser(
            _ parser: XMLParser,
            didEndElement elementName: String,
            namespaceURI: String?,
            qualifiedName qName: String?
        ) {
            _ = stack.popLast()
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            guard let node = stack.last,
                  node.text.count < limits.maximumNodeTextCharacters else { return }
            node.text += string
        }
    }
}

// MARK: - Color

/// Hex colors for the render models, `RRGGBB` uppercase.
enum OfficeColor {
    /// Office's default theme palette, in spreadsheetML theme-index order
    /// (lt1, dk1, lt2, dk2, accent1 to 6, hlink, folHlink). `theme1.xml` is
    /// NOT parsed (a documented limit), so a file with a custom theme shows
    /// these defaults where it names a theme color with no literal value.
    static let defaultTheme = [
        "FFFFFF", "000000", "E7E6E6", "44546A",
        "4472C4", "ED7D31", "A5A5A5", "FFC000", "5B9BD5", "70AD47",
        "0563C1", "954F72",
    ]

    /// The first sixteen legacy indexed colors; the rest, and the system
    /// foreground and background slots, have no fallback here.
    static let indexed = [
        "000000", "FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF",
        "000000", "FFFFFF", "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF",
    ]

    static func hex(_ value: String?) -> String? {
        guard var value = value?.trimmingCharacters(in: .whitespaces) else { return nil }
        if value.count == 8 { value = String(value.suffix(6)) }
        guard value.count == 6, value.allSatisfy(\.isHexDigit) else { return nil }
        return value.uppercased()
    }

    /// SpreadsheetML `<color>` / `<fgColor>`: rgb, then theme with tint, then
    /// indexed. nil means "no usable color", and the caller uses its default.
    static func resolve(_ node: XMLNode?) -> String? {
        guard let node else { return nil }
        if let rgb = hex(node.attrs["rgb"]) { return rgb }
        if let theme = node.attrs["theme"].flatMap(Int.init), defaultTheme.indices.contains(theme),
           let base = hex(defaultTheme[theme]) {
            return tint(base, node.attrs["tint"].flatMap(Double.init) ?? 0)
        }
        if let index = node.attrs["indexed"].flatMap(Int.init), indexed.indices.contains(index) {
            return indexed[index]
        }
        return nil
    }

    static func tint(_ hex: String, _ amount: Double) -> String {
        guard amount != 0, let value = Int(hex, radix: 16) else { return hex }
        func channel(_ shift: Int) -> Int {
            let component = Double((value >> shift) & 0xFF)
            let adjusted = amount > 0
                ? component + (255 - component) * amount
                : component * (1 + amount)
            return max(0, min(255, Int(adjusted.rounded())))
        }
        return String(format: "%02X%02X%02X", channel(16), channel(8), channel(0))
    }
}
