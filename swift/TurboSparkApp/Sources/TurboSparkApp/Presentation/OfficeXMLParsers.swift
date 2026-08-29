import Foundation

/// Fast streaming SAX XML parser that extracts linear text from Word (.docx)
/// and PowerPoint (.pptx) XML document parts.
///
/// Handles paragraph boundaries (`<w:p>`, `<a:p>`), text runs (`<w:t>`, `<a:t>`),
/// explicit line breaks (`<w:br>`), and tabs (`<w:tab>`).
final class FlowingTextXMLParser: NSObject, XMLParserDelegate {
    private var output = ""
    private var capturedText: String?

    /// Parses XML data and returns extracted plain text.
    ///
    /// - Parameter data: Raw XML bytes from a document part.
    /// - Returns: Reconstructed text with whitespace and line breaks preserved.
    func parse(_ data: Data) throws -> String {
        output = ""
        capturedText = nil
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Office document")
        }
        return output
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        switch localName(elementName) {
        case "t":
            capturedText = ""
        case "tab":
            output += "\t"
        case "br":
            output += "\n"
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedText != nil else { return }
        capturedText! += string
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        switch localName(elementName) {
        case "t":
            output += capturedText ?? ""
            capturedText = nil
        case "p":
            output += "\n"
        default:
            break
        }
    }
}

/// SAX parser for Excel `xl/sharedStrings.xml` tables.
///
/// OpenXML Excel stores repeated string literals in a global shared string table (`<sst>`).
/// Each entry (`<si>`) contains one or more text runs (`<t>`).
final class SharedStringsXMLParser: NSObject, XMLParserDelegate {
    private var strings: [String] = []
    private var currentString: String?
    private var capturedText: String?

    /// Parses shared strings XML into an indexed array of strings.
    ///
    /// - Parameter data: Raw XML bytes for `xl/sharedStrings.xml`.
    /// - Returns: Array of strings indexed by 0-based integer IDs.
    func parse(_ data: Data) throws -> [String] {
        strings = []
        currentString = nil
        capturedText = nil
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel document")
        }
        return strings
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        switch localName(elementName) {
        case "si":
            currentString = ""
        case "t":
            capturedText = ""
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedText != nil else { return }
        capturedText! += string
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        switch localName(elementName) {
        case "t":
            currentString? += capturedText ?? ""
            capturedText = nil
        case "si":
            strings.append(currentString ?? "")
            currentString = nil
        default:
            break
        }
    }
}

/// SAX parser for Excel worksheet parts (`xl/worksheets/sheet*.xml`).
///
/// Decodes cell values (`<c>`), formulas (`<f>`), inline strings (`<inlineStr>`),
/// shared string lookups (`s`), and boolean flags (`b`), outputting tab-separated rows.
final class WorksheetXMLParser: NSObject, XMLParserDelegate {
    private struct Cell {
        var reference = ""
        var type: String?
        var formula = ""
        var value = ""
        var inlineText = ""
    }

    private let sharedStrings: [String]
    private var output = ""
    private var rowCells: [String] = []
    private var cell: Cell?
    private var capturedElement: String?
    private var capturedText = ""

    /// Initializes a worksheet parser with a preloaded shared string table.
    ///
    /// - Parameter sharedStrings: Array of shared strings indexed by cell values.
    init(sharedStrings: [String]) {
        self.sharedStrings = sharedStrings
    }

    /// Parses worksheet XML data into tab-delimited row text.
    ///
    /// - Parameter data: Raw XML bytes for a worksheet part.
    /// - Returns: Extracted table text formatted as lines of tab-separated cells.
    func parse(_ data: Data) throws -> String {
        output = ""
        rowCells = []
        cell = nil
        capturedElement = nil
        capturedText = ""
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel worksheet")
        }
        return output
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        switch localName(elementName) {
        case "row":
            rowCells = []
        case "c":
            cell = Cell(
                reference: attribute(attributeDict, localName: "r") ?? "",
                type: attribute(attributeDict, localName: "t"))
        case "v", "f", "t":
            guard cell != nil else { return }
            capturedElement = localName(elementName)
            capturedText = ""
        default:
            break
        }
    }

    func parser(_ parser: XMLParser, foundCharacters string: String) {
        guard capturedElement != nil else { return }
        capturedText += string
    }

    func parser(
        _ parser: XMLParser,
        didEndElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?
    ) {
        let name = localName(elementName)
        if name == capturedElement {
            switch name {
            case "v": cell?.value = capturedText
            case "f": cell?.formula = capturedText
            case "t": cell?.inlineText += capturedText
            default: break
            }
            capturedElement = nil
            capturedText = ""
        }

        switch name {
        case "c":
            if let rendered = render(cell) {
                rowCells.append(rendered)
            }
            cell = nil
        case "row":
            if !rowCells.isEmpty {
                output += rowCells.joined(separator: "\t") + "\n"
            }
        default:
            break
        }
    }

    private func render(_ cell: Cell?) -> String? {
        guard let cell else { return nil }
        let value: String
        switch cell.type {
        case "s":
            if let index = Int(cell.value), sharedStrings.indices.contains(index) {
                value = sharedStrings[index]
            } else {
                value = cell.value
            }
        case "inlineStr":
            value = cell.inlineText
        case "b":
            value = cell.value == "1" ? "TRUE" : "FALSE"
        default:
            value = cell.value.isEmpty ? cell.inlineText : cell.value
        }

        let renderedValue: String
        if !cell.formula.isEmpty {
            renderedValue = value.isEmpty
                ? "=\(cell.formula)"
                : "=\(cell.formula) -> \(value)"
        } else {
            renderedValue = value
        }
        guard !renderedValue.isEmpty else { return nil }
        return cell.reference.isEmpty
            ? renderedValue
            : "\(cell.reference): \(renderedValue)"
    }
}

/// Description of an Excel sheet pairing its user-facing name and archive entry path.
struct WorkbookSheet {
    let name: String
    let entry: String
}

/// SAX parser for Excel `xl/workbook.xml` structure.
///
/// Extracts sheet declarations and their corresponding relationship IDs (`r:id`).
final class WorkbookXMLParser: NSObject, XMLParserDelegate {
    /// Reference to a sheet defined in `xl/workbook.xml`.
    struct SheetReference {
        let name: String
        let relationshipID: String
    }

    private var sheets: [SheetReference] = []

    /// Parses workbook XML to list sheets and their relationship IDs.
    ///
    /// - Parameter data: Raw XML bytes for `xl/workbook.xml`.
    /// - Returns: List of sheet references with names and relationship IDs.
    func parse(_ data: Data) throws -> [SheetReference] {
        sheets = []
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel workbook")
        }
        return sheets
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        guard localName(elementName) == "sheet",
            let name = attribute(attributeDict, localName: "name"),
            let relationshipID = attribute(attributeDict, localName: "id") else {
            return
        }
        sheets.append(SheetReference(name: name, relationshipID: relationshipID))
    }
}

/// SAX parser for OpenXML `.rels` relationship maps.
///
/// Maps relationship identifiers (`Id="rId1"`) to target relative paths (`Target="worksheets/sheet1.xml"`).
final class RelationshipsXMLParser: NSObject, XMLParserDelegate {
    private var relationships: [String: String] = [:]

    /// Parses relationships XML data.
    ///
    /// - Parameter data: Raw XML bytes for a `.rels` part.
    /// - Returns: Dictionary mapping relationship ID to target relative path.
    func parse(_ data: Data) throws -> [String: String] {
        relationships = [:]
        let parser = XMLParser(data: data)
        parser.delegate = self
        guard parser.parse() else {
            throw parser.parserError
                ?? DocumentTextExtractionError.invalidArchive("Excel relationships")
        }
        return relationships
    }

    func parser(
        _ parser: XMLParser,
        didStartElement elementName: String,
        namespaceURI: String?,
        qualifiedName qName: String?,
        attributes attributeDict: [String: String] = [:]
    ) {
        guard localName(elementName) == "Relationship",
            let id = attribute(attributeDict, localName: "Id"),
            let target = attribute(attributeDict, localName: "Target") else {
            return
        }
        relationships[id] = target
    }
}

/// Extracts the local element or attribute name by stripping XML namespace prefixes.
///
/// - Parameter qualifiedName: Fully qualified tag name (e.g. `w:t` or `a:p`).
/// - Returns: Local tag name (e.g. `t` or `p`).
func localName(_ qualifiedName: String) -> String {
    qualifiedName.split(separator: ":").last.map(String.init) ?? qualifiedName
}

/// Searches an XML attribute dictionary for a key matching a local name regardless of namespace prefix.
///
/// - Parameters:
///   - attributes: Dictionary of XML attribute key-value pairs.
///   - name: Unprefixed local attribute name to find.
/// - Returns: Attribute string value if found, or nil.
func attribute(_ attributes: [String: String], localName name: String) -> String? {
    attributes.first { localName($0.key) == name }?.value
}
