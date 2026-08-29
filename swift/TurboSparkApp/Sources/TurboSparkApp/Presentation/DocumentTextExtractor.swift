import Foundation
import PDFKit
import UniformTypeIdentifiers

/// Extracted plain text and metadata from a document attachment.
public struct ExtractedPromptDocument: Equatable, Sendable {
    /// The original file name of the document.
    public let fileName: String
    /// Display label representing the document format (e.g. PDF, Word, Excel).
    public let formatLabel: String
    /// The extracted and normalized text content.
    public let text: String
    /// Whether the extracted text exceeded the maximum character limit and was truncated.
    public let wasTruncated: Bool

    /// Creates an extracted prompt document.
    public init(
        fileName: String,
        formatLabel: String,
        text: String,
        wasTruncated: Bool
    ) {
        self.fileName = fileName
        self.formatLabel = formatLabel
        self.text = text
        self.wasTruncated = wasTruncated
    }
}

/// Errors that can occur during document text extraction.
public enum DocumentTextExtractionError: LocalizedError, Equatable, Sendable {
    /// The document file format is unsupported.
    case unsupportedFormat(String)
    /// The document file could not be read.
    case unreadableFile(String)
    /// The document archive is invalid or missing expected structural entries.
    case invalidArchive(String)
    /// The document exceeds safe extraction size limits.
    case documentTooLarge(String)
    /// Text extraction exceeded the timeout limit.
    case extractionTimedOut(String)
    /// No selectable text was found in the document.
    case noExtractableText(String)

    /// Localized human-readable error description.
    public var errorDescription: String? {
        switch self {
        case .unsupportedFormat(let file):
            return "\(file) is not a supported PDF, Word, PowerPoint, or Excel file."
        case .unreadableFile(let file):
            return "\(file) could not be read."
        case .invalidArchive(let file):
            return "\(file) is not a valid Office document."
        case .documentTooLarge(let file):
            return "\(file) is too large to extract safely."
        case .extractionTimedOut(let file):
            return "Reading \(file) took too long and was stopped."
        case .noExtractableText(let file):
            return "No selectable text was found in \(file). Scanned PDFs require OCR."
        }
    }
}

/// Extracts plain text content from documents and source code files.
public enum DocumentTextExtractor {
    /// Maximum character count retained from any single document.
    public static let maximumExtractedCharacters = 240_000

    /// Safety limits for archive unpacking and extraction.
    struct Limits: Sendable {
        static let `default` = Limits()
        var maximumEntryBytes = 32 * 1_024 * 1_024
        var maximumSelectedBytes = 64 * 1_024 * 1_024
        var timeout: TimeInterval = 30
    }

    /// Uniform Type Identifiers for all supported document and text formats.
    public static var supportedContentTypes: [UTType] {
        [UTType.pdf, UTType.plainText] + ["docx", "pptx", "xlsx", "txt", "md", "json", "swift", "py", "rs", "c", "cpp", "h", "js", "ts", "html", "css"].compactMap {
            UTType(filenameExtension: $0)
        }
    }

    /// Extracts text from a document URL using default safety limits.
    public static func extract(from url: URL) throws -> ExtractedPromptDocument {
        try extract(from: url, limits: .default)
    }

    /// Extracts text from a document URL with custom limits.
    static func extract(
        from url: URL,
        limits: Limits
    ) throws -> ExtractedPromptDocument {
        let didAccessSecurityScope = url.startAccessingSecurityScopedResource()
        defer {
            if didAccessSecurityScope {
                url.stopAccessingSecurityScopedResource()
            }
        }

        let fileName = url.lastPathComponent
        let extensionName = url.pathExtension.lowercased()
        let extracted: (label: String, text: String)

        switch extensionName {
        case "pdf":
            extracted = ("PDF", try extractPDF(at: url))
        case "docx":
            extracted = ("Word", try extractDOCX(at: url, limits: limits))
        case "pptx":
            extracted = ("PowerPoint", try extractPPTX(at: url, limits: limits))
        case "xlsx":
            extracted = ("Excel", try extractXLSX(at: url, limits: limits))
        default:
            // Text or code files
            if let text = try? String(contentsOf: url, encoding: .utf8) {
                extracted = (extensionName.uppercased().isEmpty ? "Text" : extensionName.uppercased(), text)
            } else if let text = try? String(contentsOf: url, encoding: .ascii) {
                extracted = (extensionName.uppercased().isEmpty ? "Text" : extensionName.uppercased(), text)
            } else {
                throw DocumentTextExtractionError.unsupportedFormat(fileName)
            }
        }

        let normalized = normalize(extracted.text)
        guard !normalized.isEmpty else {
            throw DocumentTextExtractionError.noExtractableText(fileName)
        }
        let text = String(normalized.prefix(maximumExtractedCharacters))
        return ExtractedPromptDocument(
            fileName: fileName,
            formatLabel: extracted.label,
            text: text,
            wasTruncated: text.count < normalized.count)
    }

    private static func extractPDF(at url: URL) throws -> String {
        guard let document = PDFDocument(url: url) else {
            throw DocumentTextExtractionError.unreadableFile(url.lastPathComponent)
        }

        var pages: [String] = []
        var extractedCharacterCount = 0
        pages.reserveCapacity(document.pageCount)
        for index in 0..<document.pageCount {
            guard let text = document.page(at: index)?.string,
                  !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                continue
            }
            pages.append("[Page \(index + 1)]\n\(text)")
            extractedCharacterCount += text.count
            if extractedCharacterCount > maximumExtractedCharacters * 2 {
                break
            }
        }
        return pages.joined(separator: "\n\n")
    }

    private static func extractDOCX(at url: URL, limits: Limits) throws -> String {
        let archive = try OfficeArchive(url: url, limits: limits)
        let entries = archive.entries.filter { entry in
            entry == "word/document.xml"
                || (entry.hasPrefix("word/header") && entry.hasSuffix(".xml"))
                || (entry.hasPrefix("word/footer") && entry.hasSuffix(".xml"))
                || ["word/footnotes.xml", "word/endnotes.xml", "word/comments.xml"].contains(entry)
        }.sorted { left, right in
            docxPriority(left) < docxPriority(right)
        }
        guard entries.contains("word/document.xml") else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try entries.map { entry in
            let parser = FlowingTextXMLParser()
            let text = try parser.parse(archive.data(for: entry))
            guard entry != "word/document.xml" else { return text }
            let section = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            return "[\(section.capitalized)]\n\(text)"
        }.joined(separator: "\n\n")
    }

    private static func extractPPTX(at url: URL, limits: Limits) throws -> String {
        let archive = try OfficeArchive(url: url, limits: limits)
        let slides = archive.entries.compactMap { entry -> (Int, String)? in
            guard entry.hasPrefix("ppt/slides/slide"),
                  entry.hasSuffix(".xml"),
                  !entry.contains("/_rels/") else {
                return nil
            }
            let name = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            guard let number = Int(name.dropFirst("slide".count)) else { return nil }
            return (number, entry)
        }.sorted { $0.0 < $1.0 }
        guard !slides.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try slides.map { number, entry in
            let text = try FlowingTextXMLParser().parse(archive.data(for: entry))
            return "[Slide \(number)]\n\(text)"
        }.joined(separator: "\n\n")
    }

    private static func extractXLSX(at url: URL, limits: Limits) throws -> String {
        let archive = try OfficeArchive(url: url, limits: limits)
        let sharedStrings: [String]
        if archive.entries.contains("xl/sharedStrings.xml") {
            sharedStrings = try SharedStringsXMLParser().parse(
                archive.data(for: "xl/sharedStrings.xml"))
        } else {
            sharedStrings = []
        }

        let sheets = try workbookSheets(in: archive)
        guard !sheets.isEmpty else {
            throw DocumentTextExtractionError.invalidArchive(url.lastPathComponent)
        }

        return try sheets.map { sheet in
            let rows = try WorksheetXMLParser(sharedStrings: sharedStrings).parse(
                archive.data(for: sheet.entry))
            return "[Sheet: \(sheet.name)]\n\(rows)"
        }.joined(separator: "\n\n")
    }

    private static func workbookSheets(in archive: OfficeArchive) throws -> [WorkbookSheet] {
        if archive.entries.contains("xl/workbook.xml"),
           archive.entries.contains("xl/_rels/workbook.xml.rels") {
            let sheetReferences = try WorkbookXMLParser().parse(
                archive.data(for: "xl/workbook.xml"))
            let relationships = try RelationshipsXMLParser().parse(
                archive.data(for: "xl/_rels/workbook.xml.rels"))
            let resolved = sheetReferences.compactMap { sheet -> WorkbookSheet? in
                guard let target = relationships[sheet.relationshipID] else { return nil }
                let entry = normalizedWorkbookTarget(target)
                guard archive.entries.contains(entry) else { return nil }
                return WorkbookSheet(name: sheet.name, entry: entry)
            }
            if !resolved.isEmpty {
                return resolved
            }
        }

        return archive.entries.compactMap { entry -> (Int, String)? in
            guard entry.hasPrefix("xl/worksheets/sheet"),
                  entry.hasSuffix(".xml"),
                  !entry.contains("/_rels/") else {
                return nil
            }
            let name = URL(fileURLWithPath: entry).deletingPathExtension().lastPathComponent
            guard let number = Int(name.dropFirst("sheet".count)) else { return nil }
            return (number, entry)
        }
        .sorted { $0.0 < $1.0 }
        .map { WorkbookSheet(name: "Sheet \($0.0)", entry: $0.1) }
    }

    private static func normalizedWorkbookTarget(_ target: String) -> String {
        let withoutLeadingSlash = target.hasPrefix("/") ? String(target.dropFirst()) : target
        if withoutLeadingSlash.hasPrefix("xl/") {
            return withoutLeadingSlash
        }
        let path = "xl/" + withoutLeadingSlash
        return (path as NSString).standardizingPath
    }

    private static func docxPriority(_ entry: String) -> (Int, String) {
        if entry == "word/document.xml" { return (0, entry) }
        if entry.hasPrefix("word/header") { return (1, entry) }
        if entry.hasPrefix("word/footer") { return (2, entry) }
        if entry == "word/footnotes.xml" { return (3, entry) }
        if entry == "word/endnotes.xml" { return (4, entry) }
        return (5, entry)
    }

    private static func normalize(_ source: String) -> String {
        let source = source
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\u{0}", with: "")
        var lines: [String] = []
        lines.reserveCapacity(source.count / 40)
        var previousLineWasEmpty = false

        for rawLine in source.split(separator: "\n", omittingEmptySubsequences: false) {
            let line = rawLine.trimmingCharacters(in: .whitespaces)
            if line.isEmpty {
                if !previousLineWasEmpty {
                    lines.append("")
                }
                previousLineWasEmpty = true
            } else {
                lines.append(line)
            }
        }
        return lines.joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
