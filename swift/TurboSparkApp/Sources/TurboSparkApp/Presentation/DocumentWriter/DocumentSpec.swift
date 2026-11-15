import Foundation

// ZP-09: the bounded JSON spec the `document_write` tool accepts, and the
// limits every writer enforces. The spec is model output and therefore
// hostile input: every count and string length is capped here, before any
// writer builds a part, and the caps sit inside `OfficeRenderLimits` so a
// file this layer writes always opens in the ZP-08 viewers.

enum DocumentFormat: String, Equatable, Sendable {
    case docx, xlsx, pptx, pdf

    /// Macro-enabled variants (docm, xlsm, pptm) are deliberately absent: the
    /// writers never emit a vbaProject part.
    static func forExtension(_ ext: String) -> DocumentFormat? {
        DocumentFormat(rawValue: ext.lowercased())
    }
}

enum DocumentWriteError: Error, LocalizedError, Equatable {
    case invalidSpec(String)
    case overLimit(String)
    case unsupported(String)
    case postcheckFailed(check: String, detail: String)

    var errorDescription: String? {
        switch self {
        case .invalidSpec(let message): return "Invalid document spec: \(message)"
        case .overLimit(let message): return "Document spec over a limit: \(message)"
        case .unsupported(let message): return "Not supported by the native writer: \(message)"
        case .postcheckFailed(let check, let detail):
            return "Postcheck '\(check)' failed: \(detail). Nothing was written to the project."
        }
    }
}

/// Caps on the spec and the output. Chosen below the ZP-08 render limits
/// (`OfficeRenderLimits`) so the postcheck's own load never trips a cap the
/// writer allowed.
struct DocumentWriteLimits: Sendable {
    static let `default` = DocumentWriteLimits()

    var maximumSpecBytes = 2 * 1_024 * 1_024
    var maximumOutputBytes = 20 * 1_024 * 1_024
    var maximumBlocks = 2_000
    var maximumParagraphCharacters = 20_000
    var maximumRunsPerParagraph = 200
    var maximumListItems = 500
    var maximumTableRows = 500
    var maximumTableColumns = 16
    var maximumTableCells = 5_000
    var maximumCellCharacters = 2_000
    var maximumImages = 16
    var maximumImageBytes = 8 * 1_024 * 1_024
    var maximumImagePixels = 40_000_000
    var maximumSheets = 16
    var maximumSheetRows = 1_000
    var maximumSheetColumns = 64
    var maximumWorkbookCells = 30_000
    var maximumSheetCellCharacters = 4_000
    var maximumSlides = 100
    var maximumSlideItems = 16
    var maximumSlideItemCharacters = 1_000
    var maximumPdfPages = 200
    var maximumTitleCharacters = 200
}

// MARK: - Inline text

/// A styled run. Shared by docx and pdf paragraphs.
struct DocumentRun: Decodable, Equatable, Sendable {
    var text: String
    var bold = false
    var italic = false
    var underline = false
    var strike = false
    /// `RRGGBB`, validated by `DocumentSpecValidation`.
    var color: String?
    /// Points.
    var size: Double?

    enum CodingKeys: String, CodingKey { case text, bold, italic, underline, strike, color, size }

    init(text: String) { self.text = text }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try c.decode(String.self, forKey: .text)
        bold = try c.decodeIfPresent(Bool.self, forKey: .bold) ?? false
        italic = try c.decodeIfPresent(Bool.self, forKey: .italic) ?? false
        underline = try c.decodeIfPresent(Bool.self, forKey: .underline) ?? false
        strike = try c.decodeIfPresent(Bool.self, forKey: .strike) ?? false
        color = try c.decodeIfPresent(String.self, forKey: .color)
        size = try c.decodeIfPresent(Double.self, forKey: .size)
    }
}

enum DocumentAlignment: String, Decodable, Equatable, Sendable {
    case left, center, right, justify
}

// MARK: - Blocks (docx, pdf)

struct DocumentImageRef: Equatable, Sendable {
    var path: String
    /// Display width in points; nil picks a default that fits the page.
    var widthPoints: Double?
    var alt: String
}

enum DocumentBlock: Equatable, Sendable {
    case heading(level: Int, runs: [DocumentRun])
    case paragraph(runs: [DocumentRun], alignment: DocumentAlignment)
    case list(ordered: Bool, items: [[DocumentRun]])
    case table(header: [String], rows: [[String]])
    case image(DocumentImageRef)
    case pageBreak
}

extension DocumentBlock: Decodable {
    private enum CodingKeys: String, CodingKey {
        case type, level, text, runs, align, ordered, items, header, rows, path, width, widthPt = "width_pt", alt
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decode(String.self, forKey: .type).lowercased()

        func runs() throws -> [DocumentRun] {
            if let list = try c.decodeIfPresent([DocumentRun].self, forKey: .runs), !list.isEmpty {
                return list
            }
            guard let text = try c.decodeIfPresent(String.self, forKey: .text) else {
                throw DocumentWriteError.invalidSpec("a '\(type)' block needs 'text' or 'runs'")
            }
            return [DocumentRun(text: text)]
        }

        switch type {
        case "heading":
            let level = try c.decodeIfPresent(Int.self, forKey: .level) ?? 1
            guard (1...6).contains(level) else {
                throw DocumentWriteError.invalidSpec("heading level must be 1 to 6, got \(level)")
            }
            self = .heading(level: level, runs: try runs())
        case "paragraph", "text":
            let align = try c.decodeIfPresent(String.self, forKey: .align).map { $0.lowercased() }
            guard let alignment = align.map({ DocumentAlignment(rawValue: $0) }) ?? .some(.left) else {
                throw DocumentWriteError.invalidSpec("align must be left, center, right or justify")
            }
            self = .paragraph(runs: try runs(), alignment: alignment)
        case "list":
            let ordered = try c.decodeIfPresent(Bool.self, forKey: .ordered) ?? false
            let items = try c.decode([ListItem].self, forKey: .items)
            self = .list(ordered: ordered, items: items.map(\.runs))
        case "table":
            let header = try c.decodeIfPresent([String].self, forKey: .header) ?? []
            let rows = try c.decode([[String]].self, forKey: .rows)
            self = .table(header: header, rows: rows)
        case "image":
            let path = try c.decode(String.self, forKey: .path)
            let width = try c.decodeIfPresent(Double.self, forKey: .widthPt)
                ?? c.decodeIfPresent(Double.self, forKey: .width)
            let alt = try c.decodeIfPresent(String.self, forKey: .alt) ?? ""
            self = .image(DocumentImageRef(path: path, widthPoints: width, alt: alt))
        case "page_break", "pagebreak":
            self = .pageBreak
        default:
            throw DocumentWriteError.invalidSpec(
                "unknown block type '\(type)' (use heading, paragraph, list, table, image or page_break)")
        }
    }

    /// A list item is a plain string or `{ "text": ... }` / `{ "runs": [...] }`.
    private struct ListItem: Decodable {
        var runs: [DocumentRun]

        init(from decoder: Decoder) throws {
            if let single = try? decoder.singleValueContainer().decode(String.self) {
                runs = [DocumentRun(text: single)]
                return
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            if let list = try c.decodeIfPresent([DocumentRun].self, forKey: .runs), !list.isEmpty {
                runs = list
            } else {
                runs = [DocumentRun(text: try c.decode(String.self, forKey: .text))]
            }
        }

        private enum CodingKeys: String, CodingKey { case text, runs }
    }
}

// MARK: - Sheets (xlsx)

enum SheetCellValue: Equatable, Sendable {
    case text(String)
    case number(Double)
    case bool(Bool)
    /// Days since 1899-12-30, the Excel serial.
    case date(Double)
    case empty
}

struct SheetCell: Equatable, Sendable {
    var value: SheetCellValue
    var bold = false
    var italic = false
    var color: String?
    var fill: String?
    var numberFormat: SheetNumberFormat = .general
    var alignment: DocumentAlignment?
}

/// Named formats only. A free-form format code would be one more string the
/// model controls inside styles.xml for no benefit.
enum SheetNumberFormat: String, Decodable, Equatable, Sendable, CaseIterable {
    case general, integer, decimal, thousands, thousandsDecimal = "thousands_decimal"
    case percent, percentDecimal = "percent_decimal", currency, date

    var code: String? {
        switch self {
        case .general: return nil
        case .integer: return "0"
        case .decimal: return "0.00"
        case .thousands: return "#,##0"
        case .thousandsDecimal: return "#,##0.00"
        case .percent: return "0%"
        case .percentDecimal: return "0.00%"
        case .currency: return "$#,##0.00"
        case .date: return "yyyy-mm-dd"
        }
    }
}

extension SheetCell: Decodable {
    private enum CodingKeys: String, CodingKey { case v, value, type, bold, italic, color, fill, format, align }

    init(from decoder: Decoder) throws {
        let single = try? decoder.singleValueContainer()
        if let single {
            if single.decodeNil() { value = .empty; return }
            if let bool = try? single.decode(Bool.self) { value = .bool(bool); return }
            if let number = try? single.decode(Double.self) {
                guard number.isFinite else { throw DocumentWriteError.invalidSpec("a cell number is not finite") }
                value = .number(number); return
            }
            if let text = try? single.decode(String.self) { value = .text(text); return }
        }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let type = try c.decodeIfPresent(String.self, forKey: .type)?.lowercased()
        let raw: SheetCellValue
        let key: CodingKeys = c.contains(.v) ? .v : .value
        if try c.decodeNil(forKey: key) || !c.contains(key) {
            raw = .empty
        } else if let bool = try? c.decode(Bool.self, forKey: key) {
            raw = .bool(bool)
        } else if let number = try? c.decode(Double.self, forKey: key) {
            guard number.isFinite else { throw DocumentWriteError.invalidSpec("a cell number is not finite") }
            raw = .number(number)
        } else {
            raw = .text(try c.decode(String.self, forKey: key))
        }
        if type == "date", case .text(let text) = raw {
            guard let serial = SheetCell.excelSerial(isoDate: text) else {
                throw DocumentWriteError.invalidSpec("a date cell must be YYYY-MM-DD, got '\(text.prefix(40))'")
            }
            value = .date(serial)
        } else if type == "text", case .number(let number) = raw {
            value = .text(String(number))
        } else {
            value = raw
        }
        bold = try c.decodeIfPresent(Bool.self, forKey: .bold) ?? false
        italic = try c.decodeIfPresent(Bool.self, forKey: .italic) ?? false
        color = try c.decodeIfPresent(String.self, forKey: .color)
        fill = try c.decodeIfPresent(String.self, forKey: .fill)
        if let name = try c.decodeIfPresent(String.self, forKey: .format) {
            guard let format = SheetNumberFormat(rawValue: name.lowercased()) else {
                let known = SheetNumberFormat.allCases.map(\.rawValue).joined(separator: ", ")
                throw DocumentWriteError.invalidSpec("unknown number format '\(name.prefix(40))' (use \(known))")
            }
            numberFormat = format
        } else if case .date = value {
            numberFormat = .date
        }
        if let align = try c.decodeIfPresent(String.self, forKey: .align) {
            guard let parsed = DocumentAlignment(rawValue: align.lowercased()) else {
                throw DocumentWriteError.invalidSpec("align must be left, center, right or justify")
            }
            alignment = parsed
        }
    }

    /// `2026-01-31` to the 1900-system serial, nil when it is not a real date.
    static func excelSerial(isoDate text: String) -> Double? {
        let parts = text.split(separator: "-")
        guard parts.count == 3, parts[0].count == 4,
              let year = Int(parts[0]), let month = Int(parts[1]), let day = Int(parts[2]),
              (1900...9999).contains(year) else { return nil }
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC") ?? .gmt
        var components = DateComponents()
        components.year = year; components.month = month; components.day = day
        guard let date = calendar.date(from: components) else { return nil }
        // A rolled-over date (2026-02-31) comes back with a different day.
        let back = calendar.dateComponents([.year, .month, .day], from: date)
        guard back.year == year, back.month == month, back.day == day else { return nil }
        var origin = DateComponents()
        origin.year = 1899; origin.month = 12; origin.day = 30
        guard let start = calendar.date(from: origin) else { return nil }
        let days = calendar.dateComponents([.day], from: start, to: date).day ?? 0
        return Double(days)
    }
}

struct SheetSpec: Decodable, Equatable, Sendable {
    var name: String
    /// Column widths in characters, by 0-based column; shorter than the data is fine.
    var columnWidths: [Double] = []
    var freezeRows = 0
    var rows: [[SheetCell]]

    private enum CodingKeys: String, CodingKey { case name, columns, freezeRows = "freeze_rows", rows }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        name = try c.decode(String.self, forKey: .name)
        columnWidths = try c.decodeIfPresent([Double].self, forKey: .columns) ?? []
        freezeRows = try c.decodeIfPresent(Int.self, forKey: .freezeRows) ?? 0
        rows = try c.decode([[SheetCell]].self, forKey: .rows)
    }
}

// MARK: - Slides (pptx)

struct SlideSpec: Decodable, Equatable, Sendable {
    enum Layout: String, Decodable, Sendable { case title, content }
    struct Bullet: Equatable, Sendable { var text: String; var level: Int }

    var layout: Layout = .content
    var title: String
    var subtitle: String?
    var bullets: [Bullet] = []
    var paragraphs: [String] = []

    private enum CodingKeys: String, CodingKey { case layout, title, subtitle, bullets, paragraphs }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let raw = try c.decodeIfPresent(String.self, forKey: .layout) {
            guard let parsed = Layout(rawValue: raw.lowercased()) else {
                throw DocumentWriteError.invalidSpec("slide layout must be title or content")
            }
            layout = parsed
        }
        title = try c.decode(String.self, forKey: .title)
        subtitle = try c.decodeIfPresent(String.self, forKey: .subtitle)
        paragraphs = try c.decodeIfPresent([String].self, forKey: .paragraphs) ?? []
        bullets = (try c.decodeIfPresent([BulletInput].self, forKey: .bullets) ?? []).map(\.bullet)
    }

    private struct BulletInput: Decodable {
        var bullet: Bullet
        init(from decoder: Decoder) throws {
            if let single = try? decoder.singleValueContainer().decode(String.self) {
                bullet = Bullet(text: single, level: 0)
                return
            }
            let c = try decoder.container(keyedBy: CodingKeys.self)
            bullet = Bullet(
                text: try c.decode(String.self, forKey: .text),
                level: try c.decodeIfPresent(Int.self, forKey: .level) ?? 0)
        }
        private enum CodingKeys: String, CodingKey { case text, level }
    }
}

// MARK: - Whole spec

enum DocumentPageSize: String, Decodable, Sendable {
    case letter, a4

    /// Points.
    var size: (width: Double, height: Double) {
        switch self {
        case .letter: return (612, 792)
        case .a4: return (595.28, 841.89)
        }
    }
}

struct DocumentSpec: Decodable, Equatable, Sendable {
    var title: String?
    var pageSize: DocumentPageSize = .letter
    var blocks: [DocumentBlock] = []
    var sheets: [SheetSpec] = []
    var slides: [SlideSpec] = []

    private enum CodingKeys: String, CodingKey { case title, page, blocks, sheets, slides }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        title = try c.decodeIfPresent(String.self, forKey: .title)
        if let raw = try c.decodeIfPresent(String.self, forKey: .page) {
            guard let size = DocumentPageSize(rawValue: raw.lowercased()) else {
                throw DocumentWriteError.invalidSpec("page must be letter or a4")
            }
            pageSize = size
        }
        blocks = try c.decodeIfPresent([DocumentBlock].self, forKey: .blocks) ?? []
        sheets = try c.decodeIfPresent([SheetSpec].self, forKey: .sheets) ?? []
        slides = try c.decodeIfPresent([SlideSpec].self, forKey: .slides) ?? []
    }

    static func decode(_ json: String, limits: DocumentWriteLimits = .default) throws -> DocumentSpec {
        let data = Data(json.utf8)
        guard data.count <= limits.maximumSpecBytes else {
            throw DocumentWriteError.overLimit("the spec is \(data.count) bytes (maximum \(limits.maximumSpecBytes))")
        }
        do {
            return try JSONDecoder().decode(DocumentSpec.self, from: data)
        } catch let error as DocumentWriteError {
            throw error
        } catch let error as DecodingError {
            throw DocumentWriteError.invalidSpec(Self.describe(error))
        }
    }

    private static func describe(_ error: DecodingError) -> String {
        func path(_ context: DecodingError.Context) -> String {
            context.codingPath.map { $0.intValue.map { "[\($0)]" } ?? ".\($0.stringValue)" }.joined()
        }
        switch error {
        case .keyNotFound(let key, let context):
            return "missing '\(key.stringValue)' at \(path(context))"
        case .typeMismatch(_, let context), .valueNotFound(_, let context), .dataCorrupted(let context):
            return "\(context.debugDescription) at \(path(context))"
        @unknown default:
            return "the spec is not valid JSON of the documented shape"
        }
    }
}
