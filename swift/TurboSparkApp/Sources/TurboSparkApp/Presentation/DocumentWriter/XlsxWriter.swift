import Foundation

/// ZP-09: a pure function from a validated spec to xlsx bytes. Cells only:
/// text, numbers, booleans, ISO dates, bold/italic, text and fill colors,
/// named number formats, alignment, column widths, frozen header rows.
///
/// **No formulas, charts, merged ranges, data validation, conditional formats
/// or macros in v1.** A text cell is always written as a string (shared
/// string), never parsed, so a value that starts with `=`, `+`, `-` or `@`
/// cannot become a formula. That is the whole formula-injection defense, and
/// it is why formulas are a v1 non-goal: the agent computes totals itself.
enum XlsxWriter {
    private static let mainNS = "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\""
    private static let relNS = "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""

    /// One distinct combination of cell formatting, mapped to a `cellXfs` index.
    private struct StyleKey: Hashable {
        var bold: Bool
        var italic: Bool
        var color: String?
        var fill: String?
        var format: SheetNumberFormat
        var alignment: DocumentAlignment?

        var isDefault: Bool {
            !bold && !italic && color == nil && fill == nil && format == .general && alignment == nil
        }
    }

    private struct Registry {
        var keys: [StyleKey] = [StyleKey(bold: false, italic: false, color: nil, fill: nil, format: .general, alignment: nil)]
        var index: [StyleKey: Int] = [:]

        init() { index[keys[0]] = 0 }

        mutating func id(for key: StyleKey) -> Int {
            if let existing = index[key] { return existing }
            keys.append(key)
            index[key] = keys.count - 1
            return keys.count - 1
        }
    }

    private struct SharedStrings {
        var values: [String] = []
        var index: [String: Int] = [:]
        var references = 0

        mutating func id(for text: String) -> Int {
            references += 1
            if let existing = index[text] { return existing }
            values.append(text)
            index[text] = values.count - 1
            return values.count - 1
        }
    }

    static func write(spec: DocumentSpec, limits: DocumentWriteLimits = .default) throws -> Data {
        var registry = Registry()
        var strings = SharedStrings()
        var package = OOXMLPackage()

        var sheetXML: [String] = []
        for (position, sheet) in spec.sheets.enumerated() {
            sheetXML.append(worksheetXML(sheet, selected: position == 0, registry: &registry, strings: &strings))
        }

        package.addRelationships(for: "", [("rId1", OOXMLPackage.RelationshipType.officeDocument, "xl/workbook.xml")])
        var workbook = "<workbook \(mainNS) \(relNS)><bookViews><workbookView/></bookViews><sheets>"
        var relationships: [(id: String, type: String, target: String)] = []
        for (position, sheet) in spec.sheets.enumerated() {
            let id = "rId\(position + 1)"
            workbook += "<sheet name=\"\(OOXMLText.escape(sheet.name))\" sheetId=\"\(position + 1)\" r:id=\"\(id)\"/>"
            relationships.append((id, OOXMLPackage.RelationshipType.worksheet, "worksheets/sheet\(position + 1).xml"))
            package.addXML("xl/worksheets/sheet\(position + 1).xml", sheetXML[position],
                           type: OOXMLPackage.ContentType.xlsxSheet)
        }
        workbook += "</sheets></workbook>"
        relationships.append(("rId\(spec.sheets.count + 1)", OOXMLPackage.RelationshipType.styles, "styles.xml"))
        relationships.append(("rId\(spec.sheets.count + 2)", OOXMLPackage.RelationshipType.sharedStrings, "sharedStrings.xml"))
        package.addXML("xl/workbook.xml", workbook, type: OOXMLPackage.ContentType.xlsx)
        package.addRelationships(for: "xl/workbook.xml", relationships)
        package.addXML("xl/styles.xml", stylesXML(registry), type: OOXMLPackage.ContentType.xlsxStyles)
        package.addXML("xl/sharedStrings.xml", sharedStringsXML(strings), type: OOXMLPackage.ContentType.xlsxShared)

        let output = try package.serialize()
        guard output.count <= limits.maximumOutputBytes else {
            throw DocumentWriteError.overLimit("the xlsx is \(output.count) bytes (maximum \(limits.maximumOutputBytes))")
        }
        return output
    }

    // MARK: Worksheet

    /// `0` to `A`, `26` to `AA`.
    static func columnName(_ index: Int) -> String {
        var number = index + 1
        var name = ""
        while number > 0 {
            let remainder = (number - 1) % 26
            name = String(UnicodeScalar(65 + remainder)!) + name
            number = (number - 1) / 26
        }
        return name
    }

    private static func numberText(_ value: Double) -> String {
        if value == value.rounded(), abs(value) < 1e15 { return String(Int64(value)) }
        return String(format: "%.15g", value)
    }

    private static func worksheetXML(
        _ sheet: SheetSpec, selected: Bool, registry: inout Registry, strings: inout SharedStrings
    ) -> String {
        var xml = "<worksheet \(mainNS)><sheetViews><sheetView workbookViewId=\"0\"\(selected ? " tabSelected=\"1\"" : "")>"
        if sheet.freezeRows > 0 {
            xml += "<pane ySplit=\"\(sheet.freezeRows)\" topLeftCell=\"A\(sheet.freezeRows + 1)\" "
                + "activePane=\"bottomLeft\" state=\"frozen\"/>"
        }
        xml += "</sheetView></sheetViews><sheetFormatPr defaultRowHeight=\"15\"/>"
        if !sheet.columnWidths.isEmpty {
            xml += "<cols>"
            for (index, width) in sheet.columnWidths.enumerated() {
                xml += "<col min=\"\(index + 1)\" max=\"\(index + 1)\" width=\"\(numberText(width))\" customWidth=\"1\"/>"
            }
            xml += "</cols>"
        }
        xml += "<sheetData>"
        for (rowIndex, row) in sheet.rows.enumerated() {
            var cellsXML = ""
            for (columnIndex, cell) in row.enumerated() {
                let key = StyleKey(
                    bold: cell.bold, italic: cell.italic, color: cell.color?.uppercased(),
                    fill: cell.fill?.uppercased(), format: cell.numberFormat, alignment: cell.alignment)
                let style = registry.id(for: key)
                let reference = "\(columnName(columnIndex))\(rowIndex + 1)"
                let styleAttribute = style == 0 ? "" : " s=\"\(style)\""
                switch cell.value {
                case .empty:
                    if style != 0 { cellsXML += "<c r=\"\(reference)\"\(styleAttribute)/>" }
                case .text(let text):
                    // Always a shared string: never `t="str"` or a formula.
                    cellsXML += "<c r=\"\(reference)\"\(styleAttribute) t=\"s\"><v>\(strings.id(for: text))</v></c>"
                case .number(let number), .date(let number):
                    cellsXML += "<c r=\"\(reference)\"\(styleAttribute)><v>\(numberText(number))</v></c>"
                case .bool(let flag):
                    cellsXML += "<c r=\"\(reference)\"\(styleAttribute) t=\"b\"><v>\(flag ? 1 : 0)</v></c>"
                }
            }
            if !cellsXML.isEmpty { xml += "<row r=\"\(rowIndex + 1)\">\(cellsXML)</row>" }
        }
        return xml + "</sheetData></worksheet>"
    }

    private static func sharedStringsXML(_ strings: SharedStrings) -> String {
        var xml = "<sst \(mainNS) count=\"\(strings.references)\" uniqueCount=\"\(strings.values.count)\">"
        for value in strings.values {
            xml += "<si><t xml:space=\"preserve\">\(OOXMLText.escape(value))</t></si>"
        }
        return xml + "</sst>"
    }

    // MARK: Styles

    private static func stylesXML(_ registry: Registry) -> String {
        // Custom number formats get ids from 164.
        var formatIDs: [SheetNumberFormat: Int] = [:]
        for key in registry.keys where key.format.code != nil && formatIDs[key.format] == nil {
            formatIDs[key.format] = 164 + formatIDs.count
        }
        var fontKeys: [[String]] = [[]]          // 0 = default
        var fillKeys: [String?] = [nil, nil]     // 0 none, 1 gray125 (required by Excel)
        var xfs = ""
        for key in registry.keys {
            var fontParts: [String] = []
            if key.bold { fontParts.append("b") }
            if key.italic { fontParts.append("i") }
            if let color = key.color { fontParts.append("c" + color) }
            var fontID = 0
            if !fontParts.isEmpty {
                if let existing = fontKeys.firstIndex(of: fontParts) { fontID = existing } else {
                    fontKeys.append(fontParts); fontID = fontKeys.count - 1
                }
            }
            var fillID = 0
            if let fill = key.fill {
                if let existing = fillKeys.firstIndex(where: { $0 == fill }) { fillID = existing } else {
                    fillKeys.append(fill); fillID = fillKeys.count - 1
                }
            }
            let numberID = key.format.code == nil ? 0 : (formatIDs[key.format] ?? 0)
            xfs += "<xf numFmtId=\"\(numberID)\" fontId=\"\(fontID)\" fillId=\"\(fillID)\" borderId=\"0\" xfId=\"0\""
            if numberID != 0 { xfs += " applyNumberFormat=\"1\"" }
            if fontID != 0 { xfs += " applyFont=\"1\"" }
            if fillID != 0 { xfs += " applyFill=\"1\"" }
            if let alignment = key.alignment {
                let value = alignment == .justify ? "justify" : alignment.rawValue
                xfs += " applyAlignment=\"1\"><alignment horizontal=\"\(value)\"/></xf>"
            } else {
                xfs += "/>"
            }
        }

        var xml = "<styleSheet \(mainNS)>"
        if !formatIDs.isEmpty {
            xml += "<numFmts count=\"\(formatIDs.count)\">"
            for (format, id) in formatIDs.sorted(by: { $0.value < $1.value }) {
                xml += "<numFmt numFmtId=\"\(id)\" formatCode=\"\(OOXMLText.escape(format.code ?? ""))\"/>"
            }
            xml += "</numFmts>"
        }
        xml += "<fonts count=\"\(fontKeys.count)\">"
        for parts in fontKeys {
            xml += "<font>"
            if parts.contains("b") { xml += "<b/>" }
            if parts.contains("i") { xml += "<i/>" }
            xml += "<sz val=\"11\"/>"
            if let color = parts.first(where: { $0.hasPrefix("c") }) {
                xml += "<color rgb=\"FF\(color.dropFirst())\"/>"
            }
            xml += "<name val=\"Calibri\"/><family val=\"2\"/></font>"
        }
        xml += "</fonts><fills count=\"\(fillKeys.count)\">"
        xml += "<fill><patternFill patternType=\"none\"/></fill><fill><patternFill patternType=\"gray125\"/></fill>"
        for fill in fillKeys.dropFirst(2) {
            xml += "<fill><patternFill patternType=\"solid\"><fgColor rgb=\"FF\(fill ?? "FFFFFF")\"/>"
                + "<bgColor indexed=\"64\"/></patternFill></fill>"
        }
        xml += "</fills><borders count=\"1\"><border><left/><right/><top/><bottom/><diagonal/></border></borders>"
        xml += "<cellStyleXfs count=\"1\"><xf numFmtId=\"0\" fontId=\"0\" fillId=\"0\" borderId=\"0\"/></cellStyleXfs>"
        xml += "<cellXfs count=\"\(registry.keys.count)\">\(xfs)</cellXfs>"
        xml += "<cellStyles count=\"1\"><cellStyle name=\"Normal\" xfId=\"0\" builtinId=\"0\"/></cellStyles></styleSheet>"
        return xml
    }
}
