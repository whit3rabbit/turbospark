import Foundation
import XCTest

@testable import TurboSparkApp

/// ZP-08: the render model for docx, xlsx and pptx, treated as hostile input.
/// Every fixture is generated here (a stored-zip writer, so a test can name an
/// entry `../x` or declare a lying size, which the `zip` tool refuses).
final class OfficeRenderModelTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("OfficeRender-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - Fixture writer

    private static let crcTable: [UInt32] = (0..<256).map { index -> UInt32 in
        var value = UInt32(index)
        for _ in 0..<8 { value = value & 1 == 1 ? 0xEDB88320 ^ (value >> 1) : value >> 1 }
        return value
    }

    private func crc32(_ data: Data) -> UInt32 {
        var crc: UInt32 = 0xFFFF_FFFF
        for byte in data { crc = Self.crcTable[Int((crc ^ UInt32(byte)) & 0xFF)] ^ (crc >> 8) }
        return crc ^ 0xFFFF_FFFF
    }

    private func le(_ value: UInt32, _ bytes: Int) -> [UInt8] {
        (0..<bytes).map { UInt8((value >> (8 * UInt32($0))) & 0xFF) }
    }

    /// A zip of stored (uncompressed) entries, in order.
    private func makeArchive(_ entries: [(String, Data)], named name: String) throws -> URL {
        var body = Data()
        var central = Data()
        for (entryName, data) in entries {
            let nameBytes = Array(entryName.utf8)
            let crc = crc32(data)
            let offset = UInt32(body.count)
            var local: [UInt8] = le(0x04034B50, 4) + le(20, 2) + le(0, 2) + le(0, 2) + le(0, 2) + le(0x21, 2)
            local += le(crc, 4) + le(UInt32(data.count), 4) + le(UInt32(data.count), 4)
            local += le(UInt32(nameBytes.count), 2) + le(0, 2) + nameBytes
            body.append(contentsOf: local)
            body.append(data)
            var entry: [UInt8] = le(0x02014B50, 4) + le(20, 2) + le(20, 2) + le(0, 2) + le(0, 2) + le(0, 2) + le(0x21, 2)
            entry += le(crc, 4) + le(UInt32(data.count), 4) + le(UInt32(data.count), 4)
            entry += le(UInt32(nameBytes.count), 2) + le(0, 2) + le(0, 2) + le(0, 2) + le(0, 2) + le(0, 4)
            entry += le(offset, 4) + nameBytes
            central.append(contentsOf: entry)
        }
        var end: [UInt8] = le(0x06054B50, 4) + le(0, 2) + le(0, 2)
        end += le(UInt32(entries.count), 2) + le(UInt32(entries.count), 2)
        end += le(UInt32(central.count), 4) + le(UInt32(body.count), 4) + le(0, 2)
        let url = root.appendingPathComponent(name)
        try (body + central + Data(end)).write(to: url)
        return url
    }

    private func makeDocument(_ files: [String: String], named name: String) throws -> URL {
        try makeArchive(files.sorted { $0.key < $1.key }.map { ($0.key, Data($0.value.utf8)) }, named: name)
    }

    private let wNS = "xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""
    private let relNS = "xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\""

    private func rels(_ items: [(id: String, type: String, target: String, external: Bool)]) -> String {
        let body = items.map {
            "<Relationship Id=\"\($0.id)\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/\($0.type)\" Target=\"\($0.target)\"\($0.external ? " TargetMode=\"External\"" : "")/>"
        }.joined()
        return "<Relationships \(relNS)>\(body)</Relationships>"
    }

    private func docx(body: String, extra: [String: String] = [:], name: String = "t.docx") throws -> URL {
        var files = [
            "word/document.xml": "<w:document \(wNS)><w:body>\(body)</w:body></w:document>",
        ]
        for (key, value) in extra { files[key] = value }
        return try makeDocument(files, named: name)
    }

    private func load(_ url: URL, limits: OfficeRenderLimits = .default) throws -> OfficeDocument {
        try OfficeRenderModel.load(url: url, limits: limits)
    }

    private func loadError(_ url: URL, limits: OfficeRenderLimits = .default) -> OfficeRenderError? {
        do { _ = try load(url, limits: limits); return nil } catch { return error as? OfficeRenderError }
    }

    private func requireDocx(_ url: URL, limits: OfficeRenderLimits = .default) throws -> DocxDocument {
        guard case .docx(let value) = try load(url, limits: limits) else {
            XCTFail("expected docx"); throw OfficeRenderError.unreadable
        }
        return value
    }

    // MARK: - docx

    func testDocxRunsHeadingsListsAndTable() throws {
        let styles = """
        <w:styles \(wNS)>
          <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>
          <w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Heading1"/></w:style>
        </w:styles>
        """
        let numbering = """
        <w:numbering \(wNS)>
          <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
          <w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>
          <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
          <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
        </w:numbering>
        """
        let body = """
        <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Report</w:t></w:r></w:p>
        <w:p><w:r><w:rPr><w:b/><w:i/><w:u w:val="single"/><w:color w:val="C00000" w:themeColor="accent2"/></w:rPr><w:t>Bold</w:t></w:r><w:r><w:t xml:space="preserve"> plain</w:t></w:r></w:p>
        <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:r><w:t>Sub</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>bullet</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t>one</w:t></w:r></w:p>
        <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t>two</w:t></w:r></w:p>
        <w:tbl><w:tr><w:tc><w:tcPr><w:shd w:fill="FFFF00"/></w:tcPr><w:p><w:r><w:t>A1</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>B1</w:t></w:r></w:p></w:tc></w:tr></w:tbl>
        """
        let url = try docx(body: body, extra: ["word/styles.xml": styles, "word/numbering.xml": numbering])
        let document = try requireDocx(url)

        XCTAssertEqual(document.outline.map(\.title), ["Report", "Sub"])
        XCTAssertEqual(document.outline.map(\.level), [1, 2])
        guard case .paragraph(let heading) = document.blocks[0] else { return XCTFail() }
        XCTAssertEqual(heading.headingLevel, 1)
        XCTAssertTrue(heading.runs[0].bold, "style run properties apply")
        XCTAssertEqual(heading.runs[0].size, 16)
        guard case .paragraph(let styled) = document.blocks[1] else { return XCTFail() }
        XCTAssertEqual(styled.runs.count, 2)
        XCTAssertTrue(styled.runs[0].bold && styled.runs[0].italic && styled.runs[0].underline)
        XCTAssertEqual(styled.runs[0].colorHex, "C00000", "the literal Word writes beside a theme color is the fallback")
        XCTAssertFalse(styled.runs[1].bold)
        guard case .paragraph(let sub) = document.blocks[2] else { return XCTFail() }
        XCTAssertEqual(sub.headingLevel, 2)
        XCTAssertTrue(sub.runs[0].bold, "basedOn chain folds in the base style")
        guard case .paragraph(let bullet) = document.blocks[3],
              case .paragraph(let one) = document.blocks[4],
              case .paragraph(let two) = document.blocks[5] else { return XCTFail() }
        XCTAssertEqual([bullet.marker, one.marker, two.marker], ["\u{2022}", "1.", "2."])
        guard case .table(let table) = document.blocks[6] else { return XCTFail() }
        XCTAssertEqual(table.rows.count, 1)
        XCTAssertEqual(table.rows[0].map { $0.paragraphs[0].plainText }, ["A1", "B1"])
        XCTAssertEqual(table.rows[0][0].shadingHex, "FFFF00")
    }

    func testDocxEmbeddedImageResolvedInsideArchiveOnly() throws {
        let png = Data([0x89, 0x50, 0x4E, 0x47])
        let drawing = { (id: String) in
            "<w:p><w:r><w:drawing><a:graphic xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\"><a:graphicData><pic:pic xmlns:pic=\"p\"><pic:blipFill><a:blip r:embed=\"\(id)\"/></pic:blipFill></pic:pic></a:graphicData></a:graphic></w:drawing></w:r></w:p>"
        }
        let url = try makeArchive([
            ("word/document.xml", Data("<w:document \(wNS)><w:body>\(drawing("rId1"))\(drawing("rId2"))\(drawing("rId3"))</w:body></w:document>".utf8)),
            ("word/_rels/document.xml.rels", Data(rels([
                ("rId1", "image", "media/a.png", false),
                ("rId2", "image", "../../../../etc/hosts", false),
                ("rId3", "image", "https://example.invalid/x.png", true),
            ]).utf8)),
            ("word/media/a.png", png),
        ], named: "img.docx")
        let document = try requireDocx(url)
        XCTAssertEqual(document.images, [png])
        XCTAssertEqual(document.notes.externalReferences, 1, "external relationships are counted, never followed")
        guard case .paragraph(let first) = document.blocks[0], case .paragraph(let second) = document.blocks[1],
              case .paragraph(let third) = document.blocks[2] else { return XCTFail() }
        XCTAssertEqual(first.imageIndices, [0])
        XCTAssertTrue(second.imageIndices.isEmpty, "a target that climbs out of the archive resolves to nothing")
        XCTAssertTrue(third.imageIndices.isEmpty)
    }

    func testRelationshipTargetNamingAWildcardEntryIsNeverRead() throws {
        // unzip reads its entry argument as a pattern; a real entry named
        // `media/*.png` would otherwise concatenate every png.
        let drawing = "<w:p><w:r><w:drawing><a:blip xmlns:a=\"a\" r:embed=\"rId1\"/></w:drawing></w:r></w:p>"
        let url = try makeArchive([
            ("word/document.xml", Data("<w:document \(wNS)><w:body>\(drawing)</w:body></w:document>".utf8)),
            ("word/_rels/document.xml.rels", Data(rels([("rId1", "image", "media/*.png", false)]).utf8)),
            ("word/media/*.png", Data([1, 2, 3])),
            ("word/media/other.png", Data([9, 9, 9])),
        ], named: "wild.docx")
        XCTAssertTrue(try requireDocx(url).images.isEmpty)
    }

    func testDocxBlockCapSetsTruncatedFlag() throws {
        let paragraphs = (0..<10).map { "<w:p><w:r><w:t>p\($0)</w:t></w:r></w:p>" }.joined()
        var limits = OfficeRenderLimits.default
        limits.maximumDocxBlocks = 4
        let document = try requireDocx(try docx(body: paragraphs), limits: limits)
        XCTAssertEqual(document.blocks.count, 4)
        XCTAssertTrue(document.notes.truncated)
    }

    func testDocxTableCellCapIsBounded() throws {
        let cells = (0..<30).map { "<w:tc><w:p><w:r><w:t>c\($0)</w:t></w:r></w:p></w:tc>" }.joined()
        var limits = OfficeRenderLimits.default
        limits.maximumDocxTableCells = 5
        let document = try requireDocx(try docx(body: "<w:tbl><w:tr>\(cells)</w:tr></w:tbl>"), limits: limits)
        guard case .table(let table) = document.blocks[0] else { return XCTFail() }
        XCTAssertLessThanOrEqual(table.rows.flatMap { $0 }.count, 5)
        XCTAssertTrue(document.notes.truncated)
    }

    func testMacroEnabledWordFileIsNotedAndNeverRead() throws {
        let url = try docx(body: "<w:p><w:r><w:t>hi</w:t></w:r></w:p>",
                           extra: ["word/vbaProject.bin": "BINARY"], name: "m.docm")
        let document = try requireDocx(url)
        XCTAssertTrue(document.notes.hasMacros)
    }

    // MARK: - xlsx

    private let sNS = "xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""

    private func workbook(sheets: [(name: String, state: String)], extra: [String: String] = [:]) throws -> URL {
        var files: [String: String] = [
            "xl/workbook.xml": "<workbook \(sNS)><workbookPr/><sheets>"
                + sheets.enumerated().map { "<sheet name=\"\($1.name)\" sheetId=\"\($0 + 1)\" state=\"\($1.state)\" r:id=\"rId\($0 + 1)\"/>" }.joined()
                + "</sheets></workbook>",
            "xl/_rels/workbook.xml.rels": rels(sheets.indices.map { ("rId\($0 + 1)", "worksheet", "worksheets/sheet\($0 + 1).xml", false) }),
        ]
        for (key, value) in extra { files[key] = value }
        return try makeDocument(files, named: "w.xlsx")
    }

    private func requireXlsx(_ url: URL, limits: OfficeRenderLimits = .default) throws -> XlsxDocument {
        guard case .xlsx(let value) = try load(url, limits: limits) else {
            XCTFail("expected xlsx"); throw OfficeRenderError.unreadable
        }
        return value
    }

    func testXlsxSheetsStylesNumberFormatsFrozenPaneCommentsAndHidden() throws {
        let sheet1 = """
        <worksheet \(sNS)>
          <sheetViews><sheetView><pane xSplit="1" ySplit="1" state="frozen"/></sheetView></sheetViews>
          <cols><col min="1" max="1" width="20"/></cols>
          <sheetData>
            <row r="1"><c r="A1" t="s" s="1"><v>0</v></c><c r="B1" t="s"><v>1</v></c></row>
            <row r="2"><c r="A2"><v>1234.5</v></c><c r="B2" s="2"><v>0.256</v></c><c r="C2" s="3"><v>45000</v></c><c r="D2" t="b"><v>1</v></c></row>
            <row r="3"><c r="A3" s="4"><v>3</v></c><c r="B3"><f>A3*2</f><v>6</v></c></row>
          </sheetData>
          <legacyDrawing r:id="rId1"/>
        </worksheet>
        """
        let styles = """
        <styleSheet \(sNS)>
          <numFmts count="1"><numFmt numFmtId="164" formatCode="&quot;$&quot;#,##0.00"/></numFmts>
          <fonts count="2"><font><sz val="11"/></font><font><b/><color theme="4" tint="0.4"/></font></fonts>
          <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFFFFF00"/></patternFill></fill></fills>
          <cellXfs count="5">
            <xf numFmtId="0" fontId="0" fillId="0"/>
            <xf numFmtId="0" fontId="1" fillId="2"/>
            <xf numFmtId="10" fontId="0" fillId="0"/>
            <xf numFmtId="14" fontId="0" fillId="0"/>
            <xf numFmtId="164" fontId="0" fillId="0"/>
          </cellXfs>
        </styleSheet>
        """
        let comments = "<comments \(sNS)><commentList><comment ref=\"A1\"><text><r><t>check this</t></r></text></comment></commentList></comments>"
        let url = try workbook(
            sheets: [("Data", "visible"), ("Second", "visible"), ("Secret", "hidden")],
            extra: [
                "xl/sharedStrings.xml": "<sst \(sNS)><si><t>Name</t></si><si><r><t>Va</t></r><r><t>lue</t></r></si></sst>",
                "xl/styles.xml": styles,
                "xl/worksheets/sheet1.xml": sheet1,
                "xl/worksheets/sheet2.xml": "<worksheet \(sNS)><sheetData><row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>second</t></is></c></row></sheetData></worksheet>",
                "xl/worksheets/sheet3.xml": "<worksheet \(sNS)><sheetData/></worksheet>",
                "xl/worksheets/_rels/sheet1.xml.rels": rels([("rId1", "comments", "../comments1.xml", false)]),
                "xl/comments1.xml": comments,
            ])
        let document = try requireXlsx(url)

        XCTAssertEqual(document.sheets.map(\.name), ["Data", "Second"])
        XCTAssertEqual(document.hiddenSheetCount, 1)
        let data = document.sheets[0]
        XCTAssertEqual(data.frozenRows, 1)
        XCTAssertEqual(data.frozenColumns, 1)
        XCTAssertEqual(data.columnWidths[0], 20)
        let header = data.rows[0].cells
        XCTAssertEqual(header[0].text, "Name")
        XCTAssertTrue(header[0].bold)
        XCTAssertEqual(header[0].fillHex, "FFFF00")
        XCTAssertEqual(header[0].colorHex, OfficeColor.tint("4472C4", 0.4), "theme color falls back to the default palette with tint")
        XCTAssertEqual(header[0].comment, "check this")
        XCTAssertEqual(header[1].text, "Value", "rich-text runs concatenate")
        let numbers = data.rows[1].cells
        XCTAssertEqual(numbers.map(\.text), ["1234.5", "25.60%", "3/15/2023", "TRUE"])
        XCTAssertTrue(numbers[0].isNumeric)
        XCTAssertEqual(data.rows[2].cells[0].text, "$3.00")
        XCTAssertEqual(data.rows[2].cells[1].text, "6", "a formula shows its cached value")
        XCTAssertEqual(document.sheets[1].rows[0].cells[0].text, "second")
        XCTAssertFalse(document.notes.hasMacros)
    }

    func testXlsmReportsMacrosWithoutReadingThem() throws {
        let url = try workbook(
            sheets: [("S", "visible")],
            extra: ["xl/worksheets/sheet1.xml": "<worksheet \(sNS)><sheetData/></worksheet>", "xl/vbaProject.bin": "MACRO"])
        XCTAssertTrue(try requireXlsx(url).notes.hasMacros)
    }

    func testXlsxRowColumnAndCellCapsFlagTruncation() throws {
        let rows = (1...10).map { row in
            "<row r=\"\(row)\">" + (1...10).map { "<c r=\"\(["A","B","C","D","E","F","G","H","I","J"][$0 - 1])\(row)\"><v>\($0)</v></c>" }.joined() + "</row>"
        }.joined()
        var limits = OfficeRenderLimits.default
        limits.maximumSheetRows = 3
        limits.maximumSheetColumns = 4
        let url = try workbook(sheets: [("S", "visible")],
                               extra: ["xl/worksheets/sheet1.xml": "<worksheet \(sNS)><sheetData>\(rows)</sheetData></worksheet>"])
        let document = try requireXlsx(url, limits: limits)
        XCTAssertEqual(document.sheets[0].rows.count, 3)
        XCTAssertLessThanOrEqual(document.sheets[0].columnCount, 4)
        XCTAssertTrue(document.sheets[0].truncated)
        XCTAssertTrue(document.notes.truncated)
    }

    func testHostileColumnRangeAndSplitDoNotTrapOrBalloon() throws {
        let sheet = "<worksheet \(sNS)><sheetViews><sheetView><pane xSplit=\"1e300\" ySplit=\"-5\" state=\"frozen\"/></sheetView></sheetViews><cols><col min=\"1\" max=\"1000000000\" width=\"9\"/></cols><sheetData><row r=\"1\"><c r=\"A1\"><v>1</v></c><c r=\"A1\"><v>2</v></c></row></sheetData></worksheet>"
        let url = try workbook(sheets: [("S", "visible")], extra: ["xl/worksheets/sheet1.xml": sheet])
        let document = try requireXlsx(url)
        XCTAssertLessThanOrEqual(document.sheets[0].columnWidths.count, OfficeRenderLimits.default.maximumSheetColumns)
        XCTAssertEqual(document.sheets[0].frozenRows, 0)
        XCTAssertEqual(document.sheets[0].frozenColumns, 20)
    }

    func testNumberFormatTable() {
        let cases: [(String, String, String)] = [
            ("1234.5", "General", "1234.5"),
            ("3", "0.00", "3.00"),
            ("1234567.891", "#,##0.00", "1,234,567.89"),
            ("0.5", "0%", "50%"),
            ("-4", "#,##0;(#,##0)", "(4)"),
            ("-4", "0", "-4"),
            ("1234.5", "\"$\"#,##0.00", "$1,234.50"),
            ("1234.5", "[$EUR-407] #,##0.0", "EUR 1,234.5"),
            ("45000", "yyyy-mm-dd", "2023-03-15"),
            ("45000.75", "m/d/yyyy h:mm", "3/15/2023 18:00"),
            ("45000.75", "h:mm AM/PM", "6:00 PM"),
            ("45000", "dddd, mmmm d", "Wednesday, March 15"),
            ("61", "yyyy-mm-dd", "1900-03-01"),
            ("59", "yyyy-mm-dd", "1900-02-28"),
            ("abc", "0.00", "abc"),
            ("1e300", "0.00E+00", "1.000000E+300"),
        ]
        for (raw, code, expected) in cases {
            XCTAssertEqual(OfficeNumberFormat.format(raw, code: code, date1904: false), expected, "\(raw) as \(code)")
        }
        XCTAssertEqual(OfficeNumberFormat.format("0", code: "yyyy-mm-dd", date1904: true), "1904-01-01")
    }

    // MARK: - pptx

    private let pNS = "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\""

    private func slideXML(_ tree: String) -> String {
        "<p:sld \(pNS)><p:cSld><p:spTree>\(tree)</p:spTree></p:cSld></p:sld>"
    }

    private func textShape(_ text: String, x: Int = 100, y: Int = 200, placeholder: String? = nil, geometry: Bool = true, fill: String? = nil) -> String {
        let ph = placeholder.map { "<p:nvPr><p:ph type=\"\($0)\"/></p:nvPr>" } ?? "<p:nvPr/>"
        let xfrm = geometry ? "<a:xfrm><a:off x=\"\(x)\" y=\"\(y)\"/><a:ext cx=\"3000000\" cy=\"1000000\"/></a:xfrm>" : ""
        let fillXML = fill.map { "<a:solidFill><a:srgbClr val=\"\($0)\"/></a:solidFill>" } ?? ""
        return "<p:sp><p:nvSpPr><p:cNvPr id=\"1\" name=\"s\"/><p:cNvSpPr/>\(ph)</p:nvSpPr><p:spPr>\(xfrm)<a:prstGeom prst=\"rect\"/>\(fillXML)</p:spPr><p:txBody><a:p><a:r><a:rPr sz=\"2400\" b=\"1\"><a:solidFill><a:schemeClr val=\"accent1\"/></a:solidFill></a:rPr><a:t>\(text)</a:t></a:r></a:p></p:txBody></p:sp>"
    }

    private func presentation(slides: [String], extra: [String: String] = [:], name: String = "p.pptx") throws -> URL {
        var files: [String: String] = [
            "ppt/presentation.xml": "<p:presentation \(pNS)><p:sldSz cx=\"9144000\" cy=\"6858000\"/><p:sldIdLst>"
                + slides.indices.map { "<p:sldId id=\"\(256 + $0)\" r:id=\"rId\($0 + 1)\"/>" }.joined() + "</p:sldIdLst></p:presentation>",
            "ppt/_rels/presentation.xml.rels": rels(slides.indices.map { ("rId\($0 + 1)", "slide", "slides/slide\($0 + 1).xml", false) }),
        ]
        for (index, slide) in slides.enumerated() { files["ppt/slides/slide\(index + 1).xml"] = slide }
        for (key, value) in extra { files[key] = value }
        return try makeDocument(files, named: name)
    }

    private func requirePptx(_ url: URL, limits: OfficeRenderLimits = .default) throws -> PptxDocument {
        guard case .pptx(let value) = try load(url, limits: limits) else {
            XCTFail("expected pptx"); throw OfficeRenderError.unreadable
        }
        return value
    }

    func testPptxSlidesTextFramesShapesGroupsAndPlaceholderFallback() throws {
        let group = "<p:grpSp><p:nvGrpSpPr/><p:grpSpPr><a:xfrm><a:off x=\"1000\" y=\"1000\"/><a:ext cx=\"2000\" cy=\"2000\"/><a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"1000\" cy=\"1000\"/></a:xfrm></p:grpSpPr>"
            + "<p:sp><p:nvSpPr><p:cNvPr id=\"3\" name=\"e\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr><p:spPr><a:xfrm><a:off x=\"100\" y=\"100\"/><a:ext cx=\"500\" cy=\"500\"/></a:xfrm><a:prstGeom prst=\"ellipse\"/><a:solidFill><a:srgbClr val=\"00FF00\"/></a:solidFill></p:spPr></p:sp></p:grpSp>"
        let url = try presentation(slides: [
            slideXML(textShape("Title", placeholder: "title", geometry: false) + textShape("Box", fill: "FF0000") + group),
            slideXML(textShape("Two")),
        ])
        let document = try requirePptx(url)
        XCTAssertEqual(document.slides.count, 2)
        let shapes = document.slides[0].shapes
        XCTAssertEqual(shapes.count, 3)
        XCTAssertEqual(shapes[0].paragraphs[0].plainText, "Title")
        XCTAssertEqual(shapes[0].y, document.height * 0.05, "a placeholder with no geometry takes the fixed title box")
        XCTAssertEqual(shapes[0].paragraphs[0].runs[0].size, 24)
        XCTAssertTrue(shapes[0].paragraphs[0].runs[0].bold)
        XCTAssertEqual(shapes[0].paragraphs[0].runs[0].colorHex, "4472C4", "scheme colors use the default palette")
        XCTAssertEqual(shapes[1].kind, .rectangle)
        XCTAssertEqual(shapes[1].fillHex, "FF0000")
        XCTAssertEqual(shapes[2].kind, .ellipse)
        XCTAssertEqual(shapes[2].x, 1000 + 100 * 2, "group transform maps child space")
        XCTAssertEqual(shapes[2].width, 1000)
        XCTAssertEqual(document.slides[1].shapes[0].paragraphs[0].plainText, "Two")
    }

    func testPptxSlideAndShapeCaps() throws {
        let many = (0..<10).map { textShape("s\($0)") }.joined()
        var limits = OfficeRenderLimits.default
        limits.maximumSlides = 1
        limits.maximumShapesPerSlide = 3
        let url = try presentation(slides: [slideXML(many), slideXML(many)])
        let document = try requirePptx(url, limits: limits)
        XCTAssertEqual(document.slides.count, 1)
        XCTAssertEqual(document.slides[0].shapes.count, 3)
        XCTAssertTrue(document.notes.truncated)
    }

    // MARK: - hostile input

    func testReadOnlyInvariantSourceAndNeighborsUntouched() throws {
        let url = try docx(body: "<w:p><w:r><w:t>hello</w:t></w:r></w:p>")
        let before = try FileManager.default.attributesOfItem(atPath: url.path)
        let bytes = try Data(contentsOf: url)
        let listing = try FileManager.default.contentsOfDirectory(atPath: root.path).sorted()
        _ = try load(url)
        let after = try FileManager.default.attributesOfItem(atPath: url.path)
        XCTAssertEqual(before[.modificationDate] as? Date, after[.modificationDate] as? Date)
        XCTAssertEqual(before[.size] as? Int, after[.size] as? Int)
        XCTAssertEqual(try Data(contentsOf: url), bytes, "content, not just size and mtime")
        XCTAssertEqual(try FileManager.default.contentsOfDirectory(atPath: root.path).sorted(), listing,
                       "the render pass creates nothing beside the source")
    }

    func testZipSlipNameIsRefused() throws {
        let url = try makeArchive([
            ("word/document.xml", Data("<w:document \(wNS)><w:body/></w:document>".utf8)),
            ("../evil.xml", Data("x".utf8)),
        ], named: "slip.docx")
        XCTAssertEqual(loadError(url), .unsafeArchive)
    }

    func testAbsoluteEntryNameIsRefused() throws {
        let url = try makeArchive([
            ("word/document.xml", Data("<w:document \(wNS)><w:body/></w:document>".utf8)),
            ("/etc/passwd", Data("x".utf8)),
        ], named: "abs.docx")
        XCTAssertEqual(loadError(url), .unsafeArchive)
    }

    func testSymlinkSourceIsRefused() throws {
        let real = try docx(body: "<w:p><w:r><w:t>hi</w:t></w:r></w:p>", name: "real.docx")
        let link = root.appendingPathComponent("link.docx")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        XCTAssertEqual(loadError(link), .unsafeArchive)
    }

    func testExpansionRatioIsRefusedBeforeAnyEntryIsRead() throws {
        // A real deflate bomb: 12 MB of zeros compresses to a few KB. It is
        // under the 16 MB per-part cap on purpose, so only the table-of-contents
        // ratio check can refuse it; the byte cap is a second line of defense.
        let staging = root.appendingPathComponent("bomb", isDirectory: true)
        try FileManager.default.createDirectory(at: staging.appendingPathComponent("word"), withIntermediateDirectories: true)
        try Data(count: 12 * 1_024 * 1_024).write(to: staging.appendingPathComponent("word/document.xml"))
        let archive = root.appendingPathComponent("bomb.docx")
        let zip = Process()
        zip.executableURL = URL(fileURLWithPath: "/usr/bin/zip")
        zip.currentDirectoryURL = staging
        zip.arguments = ["-q", "-r", archive.path, "word"]
        try zip.run()
        zip.waitUntilExit()
        XCTAssertEqual(zip.terminationStatus, 0)
        let size = (try FileManager.default.attributesOfItem(atPath: archive.path)[.size] as? Int) ?? 0
        XCTAssertLessThan(size, 200_000, "fixture really is a bomb")
        XCTAssertEqual(loadError(archive), .tooLarge)
    }

    func testDeclaredSizeAndEntryCountAndFileSizeCaps() throws {
        let url = try docx(body: "<w:p><w:r><w:t>hi</w:t></w:r></w:p>", extra: ["a.txt": "x", "b.txt": "y"])
        var limits = OfficeRenderLimits.default
        limits.maximumEntries = 2
        XCTAssertEqual(loadError(url, limits: limits), .tooLarge)
        limits = .default
        limits.maximumDeclaredBytes = 10
        XCTAssertEqual(loadError(url, limits: limits), .tooLarge)
        limits = .default
        limits.maximumFileBytes = 10
        XCTAssertEqual(loadError(url, limits: limits), .tooLarge)
    }

    func testXMLNodeAndDepthBoundsFailClosed() throws {
        let deep = String(repeating: "<w:x>", count: 200) + String(repeating: "</w:x>", count: 200)
        XCTAssertEqual(loadError(try docx(body: deep, name: "deep.docx")), .tooLarge)
        var limits = OfficeRenderLimits.default
        limits.maximumXMLNodes = 5
        let wide = (0..<20).map { "<w:p><w:r><w:t>\($0)</w:t></w:r></w:p>" }.joined()
        XCTAssertEqual(loadError(try docx(body: wide, name: "wide.docx"), limits: limits), .tooLarge)
    }

    func testEntityDeclarationsAreRefusedAndNothingIsResolved() throws {
        let secret = root.appendingPathComponent("secret.txt")
        try "TOP-SECRET".write(to: secret, atomically: true, encoding: .utf8)
        let xml = "<?xml version=\"1.0\"?><!DOCTYPE d [<!ENTITY x SYSTEM \"\(secret.absoluteString)\">]><w:document \(wNS)><w:body><w:p><w:r><w:t>&x;</w:t></w:r></w:p></w:body></w:document>"
        let url = try makeDocument(["word/document.xml": xml], named: "xxe.docx")
        XCTAssertEqual(loadError(url), .forbiddenDeclaration(part: "word/document.xml"))

        let bomb = "<?xml version=\"1.0\"?><!ENTITY a \"aaaa\"><w:document \(wNS)/>"
        XCTAssertEqual(loadError(try makeDocument(["word/document.xml": bomb], named: "lol.docx")),
                       .forbiddenDeclaration(part: "word/document.xml"))
    }

    func testUTF16PartIsRefusedSoAMarkerCannotHideFromTheScan() throws {
        let text = "<?xml version=\"1.0\"?><!DOCTYPE d><w:document \(wNS)/>"
        let url = try makeArchive([("word/document.xml", text.data(using: .utf16)!)], named: "u16.docx")
        XCTAssertEqual(loadError(url), .malformedXML(part: "word/document.xml"))
    }

    func testMalformedXMLGivesATypedBoundedErrorNotACrash() throws {
        // ZP-17 CUJ 2: the view renders this as a bounded card.
        let url = try makeDocument(["word/document.xml": "<w:document \(wNS)><w:body><w:p>"], named: "bad.docx")
        let error = try XCTUnwrap(loadError(url))
        XCTAssertEqual(error, .malformedXML(part: "word/document.xml"))
        XCTAssertEqual(error.category, .damaged)
        XCTAssertEqual(error.part, "word/document.xml")
        let long = OfficeRenderError.malformedXML(part: String(repeating: "p", count: 500))
        XCTAssertEqual(long.part?.count, 120, "the named part is capped before it reaches the card")
    }

    func testMissingMainPartAndNotAnArchiveAndEncrypted() throws {
        XCTAssertEqual(loadError(try makeDocument(["x.txt": "x"], named: "none.docx")), .missingPart("word/document.xml"))
        let junk = root.appendingPathComponent("junk.docx")
        try Data("not a zip at all".utf8).write(to: junk)
        XCTAssertEqual(loadError(junk), .notAnArchive)
        let ole = root.appendingPathComponent("enc.xlsx")
        try Data([0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1] + [UInt8](repeating: 0, count: 64)).write(to: ole)
        XCTAssertEqual(loadError(ole), .encrypted)
        XCTAssertEqual(OfficeRenderError.encrypted.category, .damaged)
        XCTAssertEqual(OfficeRenderError.tooLarge.category, .overLimits)
    }

    func testLegacyFormatsStayOnTheSystemPreview() {
        XCTAssertNil(OfficeFormat.forExtension("doc"))
        XCTAssertNil(OfficeFormat.forExtension("xls"))
        XCTAssertNil(OfficeFormat.forExtension("ppt"))
        XCTAssertEqual(OfficeFormat.forExtension("DOCM"), .docx)
        XCTAssertEqual(loadError(root.appendingPathComponent("a.doc")), .unsupportedExtension)
    }

    // MARK: - routing

    func testArtifactRenderKindRoutesOfficeAndFallsBackForUnknownValues() throws {
        XCTAssertEqual(AppArtifactRenderKind.forFileName("a.docx"), .docx)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("a.XLSM"), .xlsx)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("a.pptx"), .pptx)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("a.doc"), .opaque)
        XCTAssertEqual(AppArtifactRenderKind.forFileName("a.zip"), .opaque)
        let future = try JSONDecoder().decode(AppArtifactRenderKind.self, from: Data("\"hologram\"".utf8))
        XCTAssertEqual(future, .opaque)
        XCTAssertEqual(try JSONDecoder().decode(AppArtifactRenderKind.self, from: Data("\"docx\"".utf8)), .docx)
    }
}
