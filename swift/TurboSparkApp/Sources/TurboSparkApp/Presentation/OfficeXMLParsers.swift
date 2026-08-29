import Foundation

final class FlowingTextXMLParser: NSObject, XMLParserDelegate {
    private var output = ""
    private var capturedText: String?

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

final class SharedStringsXMLParser: NSObject, XMLParserDelegate {
    private var strings: [String] = []
    private var currentString: String?
    private var capturedText: String?

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

    init(sharedStrings: [String]) {
        self.sharedStrings = sharedStrings
    }

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
                : "=\(cell.formula) → \(value)"
        } else {
            renderedValue = value
        }
        guard !renderedValue.isEmpty else { return nil }
        return cell.reference.isEmpty
            ? renderedValue
            : "\(cell.reference): \(renderedValue)"
    }
}

struct WorkbookSheet {
    let name: String
    let entry: String
}

final class WorkbookXMLParser: NSObject, XMLParserDelegate {
    struct SheetReference {
        let name: String
        let relationshipID: String
    }

    private var sheets: [SheetReference] = []

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

final class RelationshipsXMLParser: NSObject, XMLParserDelegate {
    private var relationships: [String: String] = [:]

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

func localName(_ qualifiedName: String) -> String {
    qualifiedName.split(separator: ":").last.map(String.init) ?? qualifiedName
}

func attribute(_ attributes: [String: String], localName name: String) -> String? {
    attributes.first { localName($0.key) == name }?.value
}
