import Foundation
import PDFKit

/// What the postcheck holds a written file against: counts taken from the
/// spec, not from the file, so a writer that drops content fails.
struct DocumentExpectations: Equatable, Sendable {
    var format: DocumentFormat
    var sheetCount = 0
    var slideCount = 0
    var tableCount = 0
    var imageCount = 0
    var headingCount = 0
    /// pdf only, points.
    var pageSize: (width: Double, height: Double)?
    /// Whether the spec carried any visible text at all.
    var expectsText = true

    static func == (lhs: DocumentExpectations, rhs: DocumentExpectations) -> Bool {
        lhs.format == rhs.format && lhs.sheetCount == rhs.sheetCount && lhs.slideCount == rhs.slideCount
            && lhs.tableCount == rhs.tableCount && lhs.imageCount == rhs.imageCount
            && lhs.headingCount == rhs.headingCount && lhs.expectsText == rhs.expectsText
            && lhs.pageSize?.width == rhs.pageSize?.width && lhs.pageSize?.height == rhs.pageSize?.height
    }

    init(spec: DocumentSpec, format: DocumentFormat) {
        self.format = format
        switch format {
        case .docx, .pdf:
            for block in spec.blocks {
                switch block {
                case .table: tableCount += 1
                case .image: imageCount += 1
                case .heading: headingCount += 1
                default: break
                }
            }
            if format == .pdf { pageSize = spec.pageSize.size }
        case .xlsx:
            sheetCount = spec.sheets.count
        case .pptx:
            slideCount = spec.slides.count
        }
        switch format {
        case .docx, .pdf:
            expectsText = spec.blocks.contains {
                switch $0 {
                case .heading, .paragraph, .list, .table: return true
                case .image, .pageBreak: return false
                }
            }
        case .xlsx:
            expectsText = spec.sheets.contains { $0.rows.contains { $0.contains { $0.value != .empty } } }
        case .pptx:
            expectsText = true
        }
    }
}

struct DocumentPostcheckReport: Equatable, Sendable {
    /// Named checks that passed, in the order they ran.
    var passed: [String] = []
    var characterCount = 0
    var partCount = 0
    var pageCount = 0
}

/// ZP-09: the mandatory tool-side postcheck. It runs INSIDE `document_write`
/// on a temporary file, before the file reaches the project, so a failure
/// leaves nothing behind and is reported once, never retried.
///
/// Four named checks, each independent of how the writer built the file:
///  - `archive`: our own zip reader parses the central directory and every
///    member's CRC and size match (not `unzip`, not the ZP-08 loader).
///  - `content-types`: `[Content_Types].xml` exists and covers every part and
///    the main part's type. The ZP-08 loader never reads it, so this is the
///    one check that catches a package Word would call corrupt.
///  - `relationships`: every relationship target exists inside the package and
///    none is marked external.
///  - `render`: the ZP-08 loader opens the file and the counts and text match
///    the spec.
/// A pdf runs `pdf` (PDFKit opens it, page count, page size, extractable text).
/// This does NOT prove Word, Excel or PowerPoint opens the file; no Office
/// application is available to this layer.
enum DocumentPostcheck {
    static func run(
        fileURL: URL, expectations: DocumentExpectations
    ) throws -> DocumentPostcheckReport {
        switch expectations.format {
        case .pdf: return try runPdf(fileURL: fileURL, expectations: expectations)
        default: return try runOOXML(fileURL: fileURL, expectations: expectations)
        }
    }

    private static func fail(_ check: String, _ detail: String) -> DocumentWriteError {
        .postcheckFailed(check: check, detail: detail)
    }

    // MARK: OOXML

    private static func runOOXML(
        fileURL: URL, expectations: DocumentExpectations
    ) throws -> DocumentPostcheckReport {
        var report = DocumentPostcheckReport()
        let data: Data
        do { data = try Data(contentsOf: fileURL) } catch { throw fail("archive", "the file could not be read back") }

        // archive
        let zip: ZipArchiveReader
        do {
            zip = try ZipArchiveReader(data: data)
            var names: Set<String> = []
            for member in zip.members {
                guard OfficeRenderModel.isContainedName(member.name), names.insert(member.name).inserted else {
                    throw fail("archive", "unsafe or duplicate part name '\(member.name.prefix(80))'")
                }
                _ = try zip.read(member)
            }
        } catch let error as DocumentWriteError {
            throw error
        } catch {
            throw fail("archive", "\(error)")
        }
        report.partCount = zip.members.count
        report.passed.append("archive")

        // content-types
        let contentTypes: XMLNode
        do {
            contentTypes = try OfficeXMLTree.parse(
                try zip.read("[Content_Types].xml"), part: "[Content_Types].xml", limits: .default)
        } catch {
            throw fail("content-types", "[Content_Types].xml is missing or not well-formed XML")
        }
        var defaults: [String: String] = [:]
        var overrides: [String: String] = [:]
        for node in contentTypes.children {
            if node.name == "Default", let ext = node.attrs["Extension"], let type = node.attrs["ContentType"] {
                defaults[ext.lowercased()] = type
            } else if node.name == "Override", let part = node.attrs["PartName"], let type = node.attrs["ContentType"] {
                overrides[part] = type
            }
        }
        for member in zip.members where member.name != "[Content_Types].xml" {
            // Not `pathExtension`: it reads `.rels` (the leaf of `_rels/.rels`) as no extension.
            let leaf = member.name.split(separator: "/").last.map(String.init) ?? member.name
            let ext = leaf.split(separator: ".", omittingEmptySubsequences: false).dropFirst().last
                .map { $0.lowercased() } ?? ""
            guard overrides["/" + member.name] != nil || defaults[ext] != nil else {
                throw fail("content-types", "part '\(member.name.prefix(80))' has no content type")
            }
        }
        let mainPart = try mainPartName(zip: zip, expectations: expectations)
        let expectedMain = expectedMainType(expectations.format)
        guard overrides["/" + mainPart] == expectedMain else {
            throw fail("content-types", "main part '\(mainPart)' is not declared as \(expectedMain)")
        }
        report.passed.append("content-types")

        // relationships
        try checkRelationships(zip: zip)
        report.passed.append("relationships")

        // render
        let document: OfficeDocument
        do {
            document = try OfficeRenderModel.load(url: fileURL)
        } catch {
            throw fail("render", "the ZP-08 loader refused the file (\(error))")
        }
        report.characterCount = try checkRendered(document, expectations: expectations)
        report.passed.append("render")
        return report
    }

    private static func expectedMainType(_ format: DocumentFormat) -> String {
        switch format {
        case .docx: return OOXMLPackage.ContentType.docx
        case .xlsx: return OOXMLPackage.ContentType.xlsx
        case .pptx: return OOXMLPackage.ContentType.pptx
        case .pdf: return ""
        }
    }

    private static func mainPartName(zip: ZipArchiveReader, expectations: DocumentExpectations) throws -> String {
        let rels: XMLNode
        do {
            rels = try OfficeXMLTree.parse(try zip.read("_rels/.rels"), part: "_rels/.rels", limits: .default)
        } catch {
            throw fail("relationships", "_rels/.rels is missing or not well-formed XML")
        }
        guard let target = rels.children.first(where: { ($0.attrs["Type"] ?? "").hasSuffix("/officeDocument") })?
            .attrs["Target"] else {
            throw fail("relationships", "_rels/.rels has no officeDocument relationship")
        }
        return target.hasPrefix("/") ? String(target.dropFirst()) : target
    }

    private static func checkRelationships(zip: ZipArchiveReader) throws {
        let names = Set(zip.members.map(\.name))
        for member in zip.members where member.name.hasSuffix(".rels") {
            let root: XMLNode
            do {
                root = try OfficeXMLTree.parse(try zip.read(member), part: member.name, limits: .default)
            } catch {
                throw fail("relationships", "\(member.name) is not well-formed XML")
            }
            // `a/_rels/b.xml.rels` belongs to `a/b.xml`; `_rels/.rels` to the package root.
            let parts = member.name.split(separator: "/", omittingEmptySubsequences: true).map(String.init)
            let sourceDirectory = parts.count >= 2 ? Array(parts.dropLast(2)) : []
            for node in root.children where node.name == "Relationship" {
                if (node.attrs["TargetMode"] ?? "").lowercased() == "external" {
                    throw fail("relationships", "\(member.name) holds an external relationship")
                }
                guard let target = node.attrs["Target"], !target.isEmpty else {
                    throw fail("relationships", "\(member.name) has a relationship with no target")
                }
                var path: [String] = target.hasPrefix("/") ? [] : sourceDirectory
                for piece in target.split(separator: "/", omittingEmptySubsequences: true) {
                    if piece == "." { continue }
                    if piece == ".." {
                        guard !path.isEmpty else { throw fail("relationships", "\(member.name) escapes the package") }
                        path.removeLast()
                    } else {
                        path.append(String(piece))
                    }
                }
                guard names.contains(path.joined(separator: "/")) else {
                    throw fail("relationships", "\(member.name) points at missing part '\(path.joined(separator: "/").prefix(80))'")
                }
            }
        }
    }

    /// Returns the visible character count.
    private static func checkRendered(_ document: OfficeDocument, expectations: DocumentExpectations) throws -> Int {
        var characters = 0
        switch document {
        case .docx(let docx):
            var tables = 0
            var headings = 0
            for block in docx.blocks {
                switch block {
                case .paragraph(let paragraph):
                    characters += paragraph.plainText.count
                    if paragraph.headingLevel != nil { headings += 1 }
                case .table(let table):
                    tables += 1
                    characters += table.rows.flatMap { $0 }.flatMap(\.paragraphs).map { $0.plainText.count }.reduce(0, +)
                }
            }
            guard tables == expectations.tableCount else {
                throw fail("render", "expected \(expectations.tableCount) tables, found \(tables)")
            }
            guard headings == expectations.headingCount else {
                throw fail("render", "expected \(expectations.headingCount) headings, found \(headings)")
            }
            guard docx.images.count >= min(expectations.imageCount, 1) else {
                throw fail("render", "expected images, found none")
            }
            guard !docx.notes.truncated else { throw fail("render", "the file exceeds the viewer's block limits") }
        case .xlsx(let xlsx):
            guard xlsx.sheets.count == expectations.sheetCount else {
                throw fail("render", "expected \(expectations.sheetCount) sheets, found \(xlsx.sheets.count)")
            }
            for sheet in xlsx.sheets {
                characters += sheet.rows.flatMap(\.cells).map { $0.text.count }.reduce(0, +)
            }
            guard !xlsx.notes.truncated else { throw fail("render", "the file exceeds the viewer's sheet limits") }
        case .pptx(let pptx):
            guard pptx.slides.count == expectations.slideCount else {
                throw fail("render", "expected \(expectations.slideCount) slides, found \(pptx.slides.count)")
            }
            for slide in pptx.slides {
                for shape in slide.shapes {
                    characters += shape.paragraphs.flatMap(\.runs).map { $0.text.count }.reduce(0, +)
                }
            }
        }
        if expectations.expectsText, characters == 0 {
            throw fail("render", "the file has no readable text")
        }
        return characters
    }

    // MARK: PDF

    private static func runPdf(
        fileURL: URL, expectations: DocumentExpectations
    ) throws -> DocumentPostcheckReport {
        var report = DocumentPostcheckReport()
        guard let head = try? FileHandle(forReadingFrom: fileURL), let prefix = try? head.read(upToCount: 5),
              prefix == Data("%PDF-".utf8) else {
            throw fail("pdf", "the file does not start with %PDF-")
        }
        try? head.close()
        guard let document = PDFDocument(url: fileURL), document.pageCount > 0 else {
            throw fail("pdf", "PDFKit could not open the file or it has no pages")
        }
        report.pageCount = document.pageCount
        if let expected = expectations.pageSize {
            for index in 0..<document.pageCount {
                guard let bounds = document.page(at: index)?.bounds(for: .mediaBox),
                      abs(Double(bounds.width) - expected.width) < 1, abs(Double(bounds.height) - expected.height) < 1 else {
                    throw fail("pdf", "page \(index + 1) is not \(Int(expected.width))x\(Int(expected.height)) points")
                }
            }
        }
        let text = document.string ?? ""
        report.characterCount = text.trimmingCharacters(in: .whitespacesAndNewlines).count
        if expectations.expectsText, report.characterCount == 0 {
            throw fail("pdf", "no extractable text")
        }
        report.passed.append("pdf")
        return report
    }
}
