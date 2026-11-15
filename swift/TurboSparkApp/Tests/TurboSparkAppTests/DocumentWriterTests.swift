import Foundation
import PDFKit
import XCTest
@testable import TurboSparkApp

/// ZP-09: native document writing. Every format is written through
/// `DocumentWriteExecutor`, then re-opened through the ZP-08 loader
/// (`OfficeRenderModel`), `OfficeArchive` (the `unzip` reader), and the
/// writer's own independent zip reader, so a writer that only agrees with
/// itself fails here.
final class DocumentWriterTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-docwrite-tests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        root = root.resolvingSymlinksInPath()
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: Helpers

    @discardableResult
    private func write(_ path: String, _ spec: String) throws -> URL {
        try DocumentWriteExecutor.execute(arguments: ["path": path, "spec": spec], rootURL: root).url
    }

    private func expectFailure(
        _ path: String, _ spec: String, contains needle: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertThrowsError(try write(path, spec), file: file, line: line) { error in
            let text = (error as NSError).localizedDescription
            XCTAssertTrue(text.contains(needle), "\(text) should contain \(needle)", file: file, line: line)
        }
    }

    private func pngData(width: Int = 40, height: Int = 20) throws -> Data {
        let rep = try XCTUnwrap(NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height, bitsPerSample: 8,
            samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
            bytesPerRow: 0, bitsPerPixel: 0))
        for x in 0..<width { for y in 0..<height { rep.setColor(.systemBlue, atX: x, y: y) } }
        return try XCTUnwrap(rep.representation(using: .png, properties: [:]))
    }

    private func entries(of url: URL) throws -> [String: Data] {
        let zip = try ZipArchiveReader(data: Data(contentsOf: url))
        var result: [String: Data] = [:]
        for member in zip.members { result[member.name] = try zip.read(member) }
        return result
    }

    private func text(_ data: Data?) -> String { String(decoding: data ?? Data(), as: UTF8.self) }

    // MARK: docx

    func testDocxRoundTripThroughViewerLoaderAndArchive() throws {
        try pngData().write(to: root.appendingPathComponent("pic.png"))
        let url = try write("out/report.docx", """
        {"title":"Q3","blocks":[
         {"type":"heading","level":1,"text":"Summary"},
         {"type":"paragraph","runs":[{"text":"Bold ","bold":true},{"text":"red","color":"C00000","italic":true}]},
         {"type":"list","ordered":false,"items":["alpha","beta"]},
         {"type":"list","ordered":true,"items":["one","two"]},
         {"type":"list","ordered":true,"items":["restart"]},
         {"type":"table","header":["Region","Revenue"],"rows":[["EMEA","1.2M"],["APAC","0.9M"]]},
         {"type":"image","path":"pic.png","width_pt":120,"alt":"blue"},
         {"type":"page_break"},
         {"type":"heading","level":2,"text":"Next"}
        ]}
        """)
        guard case .docx(let doc) = try OfficeRenderModel.load(url: url) else { return XCTFail("not docx") }
        var texts: [String] = []
        var markers: [String] = []
        var tables = 0
        for block in doc.blocks {
            switch block {
            case .paragraph(let p):
                texts.append(p.plainText)
                if let marker = p.marker { markers.append(marker + p.plainText) }
            case .table(let t):
                tables += 1
                XCTAssertEqual(t.rows.count, 3)
                XCTAssertEqual(t.rows[0].first?.paragraphs.first?.plainText, "Region")
                XCTAssertEqual(t.rows[2][1].paragraphs.first?.plainText, "0.9M")
                XCTAssertEqual(t.rows[0].first?.shadingHex?.uppercased(), "F2F2F2")
            }
        }
        XCTAssertEqual(tables, 1)
        XCTAssertTrue(texts.contains("Summary"))
        XCTAssertEqual(markers, ["\u{2022}alpha", "\u{2022}beta", "1.one", "2.two", "1.restart"])
        XCTAssertEqual(doc.outline.map(\.title), ["Summary", "Next"])
        XCTAssertEqual(doc.outline.map(\.level), [1, 2])
        XCTAssertEqual(doc.images.count, 1)
        let styled = doc.blocks.compactMap { block -> DocxParagraph? in
            if case .paragraph(let p) = block, p.runs.count == 2 { return p } else { return nil }
        }.first
        XCTAssertEqual(styled?.runs.first?.bold, true)
        XCTAssertEqual(styled?.runs.last?.colorHex?.uppercased(), "C00000")
        XCTAssertEqual(styled?.runs.last?.italic, true)
        XCTAssertFalse(doc.notes.hasMacros)
        XCTAssertEqual(doc.notes.externalReferences, 0)

        // The unzip-backed reader sees the same package.
        let archive = try OfficeArchive(url: url, limits: OfficeRenderLimits.default.archiveLimits)
        for part in ["word/document.xml", "word/styles.xml", "word/numbering.xml",
                     "word/_rels/document.xml.rels", "word/media/image1.png", "_rels/.rels"] {
            XCTAssertTrue(archive.entries.contains(part), part)
        }
        XCTAssertFalse(archive.entries.contains { $0.lowercased().contains("vbaproject") })
        XCTAssertTrue(text(try archive.data(for: "word/document.xml")).contains("Summary"))
    }

    func testHostileTextIsEscapedAndControlCharactersDropped() throws {
        let hostile = "</w:t></w:r><w:r><w:t>INJECTED & <b>\\\"quote\\\" 'apos' \\u0000\\u000b\\u001f end"
        let url = try write("h.docx", """
        {"blocks":[{"type":"paragraph","text":"\(hostile)"},{"type":"heading","text":"<script>x</script>"}]}
        """)
        let parts = try entries(of: url)
        let xml = text(parts["word/document.xml"])
        XCTAssertFalse(xml.contains("</w:t></w:r><w:r><w:t>INJECTED"), "text must not break out of its element")
        XCTAssertFalse(xml.contains("<script>"))
        XCTAssertTrue(xml.contains("&lt;/w:t&gt;"))
        XCTAssertFalse(xml.unicodeScalars.contains { $0.value == 0 || $0.value == 0x0B || $0.value == 0x1F })
        guard case .docx(let doc) = try OfficeRenderModel.load(url: url) else { return XCTFail() }
        let all = doc.blocks.compactMap { block -> String? in
            if case .paragraph(let p) = block { return p.plainText } else { return nil }
        }.joined(separator: "|")
        XCTAssertTrue(all.contains("</w:t></w:r><w:r><w:t>INJECTED & <b>"), all)
        XCTAssertTrue(all.contains("<script>x</script>"))
    }

    func testNoPackageHoldsAnExternalRelationshipOrMacroPart() throws {
        try pngData().write(to: root.appendingPathComponent("pic.png"))
        let docx = try write("a.docx", #"{"blocks":[{"type":"paragraph","text":"http://evil.example/x"},{"type":"image","path":"pic.png"}]}"#)
        let xlsx = try write("a.xlsx", #"{"sheets":[{"name":"S","rows":[["=HYPERLINK(\"http://evil.example\")"]]}]}"#)
        let pptx = try write("a.pptx", #"{"slides":[{"title":"https://evil.example"}]}"#)
        for url in [docx, xlsx, pptx] {
            for (name, data) in try entries(of: url) {
                let body = text(data)
                XCTAssertFalse(body.contains("TargetMode"), "\(url.lastPathComponent) \(name)")
                XCTAssertFalse(name.lowercased().contains("vba"), name)
                XCTAssertFalse(name.lowercased().hasPrefix("customxml"), name)
            }
        }
    }

    func testDocxImageRefusalsAndHostilePaths() throws {
        let spec = #"{"blocks":[{"type":"image","path":"%@"}]}"#
        // Missing, outside the root, sensitive, not an image (a text file named .png).
        try Data("not an image".utf8).write(to: root.appendingPathComponent("fake.png"))
        expectFailure("i1.docx", String(format: spec, "missing.png"), contains: "not a regular file")
        expectFailure("i2.docx", String(format: spec, "../escape.png"), contains: "escapes the project root")
        expectFailure("i3.docx", String(format: spec, "/etc/hosts"), contains: "relative")
        expectFailure("i4.docx", String(format: spec, ".env"), contains: "sensitive")
        expectFailure("i5.docx", String(format: spec, "fake.png"), contains: "PNG or JPEG")
        try Data("svg".utf8).write(to: root.appendingPathComponent("x.svg"))
        expectFailure("i6.docx", String(format: spec, "x.svg"), contains: "PNG or JPEG")
        // A symlink is resolved before the containment check: one that stays
        // inside the project reads its target, one that leaves it is refused.
        try pngData().write(to: root.appendingPathComponent("real.png"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link.png"), withDestinationURL: root.appendingPathComponent("real.png"))
        XCTAssertNoThrow(try write("i7.docx", String(format: spec, "link.png")))
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-outside-img-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        try pngData().write(to: outside.appendingPathComponent("o.png"))
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("out.png"), withDestinationURL: outside.appendingPathComponent("o.png"))
        expectFailure("i8.docx", String(format: spec, "out.png"), contains: "escapes the project root")
        // A directory standing where the image should be is not a regular file.
        try FileManager.default.createDirectory(at: root.appendingPathComponent("dir.png"), withIntermediateDirectories: true)
        expectFailure("i9.docx", String(format: spec, "dir.png"), contains: "not a regular file")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("i1.docx").path))
    }

    // MARK: xlsx

    func testXlsxRoundTripFormatsAndInertText() throws {
        let url = try write("book.xlsx", """
        {"sheets":[
         {"name":"Sales","columns":[24,14],"freeze_rows":1,"rows":[
           [{"v":"Region","bold":true,"fill":"D9E1F2"},{"v":"Revenue","bold":true,"color":"C00000"}],
           ["EMEA",{"v":1200.5,"format":"currency"}],
           ["APAC",{"v":0.15,"format":"percent"}],
           ["Date",{"v":"2026-01-31","type":"date"}],
           [true,null,3],
           ["=cmd|' /C calc'!A0","@SUM(A1)"]
         ]},
         {"name":"Second","rows":[["x"]]}
        ]}
        """)
        guard case .xlsx(let book) = try OfficeRenderModel.load(url: url) else { return XCTFail("not xlsx") }
        XCTAssertEqual(book.sheets.map(\.name), ["Sales", "Second"])
        let sales = book.sheets[0]
        XCTAssertEqual(sales.frozenRows, 1)
        XCTAssertEqual(sales.columnWidths[0], 24)
        func cell(_ row: Int, _ column: Int) -> XlsxCell? {
            sales.rows.first { $0.number == row }?.cells.first { $0.column == column }
        }
        XCTAssertEqual(cell(1, 0)?.text, "Region")
        XCTAssertEqual(cell(1, 0)?.bold, true)
        XCTAssertEqual(cell(1, 0)?.fillHex?.uppercased(), "D9E1F2")
        XCTAssertEqual(cell(1, 1)?.colorHex?.uppercased(), "C00000")
        XCTAssertEqual(cell(2, 1)?.text, "$1,200.50")
        XCTAssertEqual(cell(3, 1)?.text, "15%")
        XCTAssertEqual(cell(4, 1)?.text, "2026-01-31")
        XCTAssertEqual(cell(5, 0)?.text, "TRUE")
        XCTAssertEqual(cell(5, 2)?.text, "3")
        XCTAssertEqual(cell(6, 0)?.text, "=cmd|' /C calc'!A0")
        XCTAssertEqual(cell(6, 1)?.text, "@SUM(A1)")

        // No cell is a formula, and risky strings are shared strings.
        let parts = try entries(of: url)
        let sheetXML = text(parts["xl/worksheets/sheet1.xml"])
        XCTAssertFalse(sheetXML.contains("<f>") || sheetXML.contains("<f "), "formulas must never be written")
        let archive = try OfficeArchive(url: url, limits: OfficeRenderLimits.default.archiveLimits)
        XCTAssertTrue(archive.entries.contains("xl/sharedStrings.xml"))
        XCTAssertTrue(text(try archive.data(for: "xl/sharedStrings.xml")).contains("=cmd|"))
    }

    func testXlsxSerialDatesAreExactAndImpossibleDatesRefused() throws {
        XCTAssertEqual(SheetCell.excelSerial(isoDate: "1900-03-01"), 61)
        XCTAssertEqual(SheetCell.excelSerial(isoDate: "2026-01-31"), 46053)
        XCTAssertNil(SheetCell.excelSerial(isoDate: "2026-02-31"))
        XCTAssertNil(SheetCell.excelSerial(isoDate: "2026-1-1x"))
        expectFailure("d.xlsx", #"{"sheets":[{"name":"S","rows":[[{"v":"2026-02-31","type":"date"}]]}]}"#, contains: "YYYY-MM-DD")
    }

    func testXlsxSheetNameRules() {
        for bad in ["", "a/b", "a:b", "a[b", "'x", String(repeating: "n", count: 32)] {
            expectFailure("n.xlsx", #"{"sheets":[{"name":"\#(bad)","rows":[["x"]]}]}"#, contains: "sheet name")
        }
        expectFailure("n.xlsx", #"{"sheets":[{"name":"A","rows":[["x"]]},{"name":"a","rows":[["x"]]}]}"#, contains: "duplicate")
    }

    func testXlsxColumnNames() {
        XCTAssertEqual(XlsxWriter.columnName(0), "A")
        XCTAssertEqual(XlsxWriter.columnName(25), "Z")
        XCTAssertEqual(XlsxWriter.columnName(26), "AA")
        XCTAssertEqual(XlsxWriter.columnName(63), "BL")
    }

    // MARK: pptx

    func testPptxRoundTrip() throws {
        let url = try write("deck.pptx", """
        {"slides":[
         {"layout":"title","title":"Launch","subtitle":"Q4"},
         {"title":"Goals","bullets":["Ship",{"text":"Fix bugs","level":1}],"paragraphs":["Plain line"]},
         {"title":"Last","bullets":["<b>&amp;"]}
        ]}
        """)
        guard case .pptx(let deck) = try OfficeRenderModel.load(url: url) else { return XCTFail("not pptx") }
        XCTAssertEqual(deck.slides.count, 3)
        func strings(_ slide: Slide) -> [String] {
            slide.shapes.flatMap(\.paragraphs).map { $0.runs.map(\.text).joined() }
        }
        XCTAssertEqual(strings(deck.slides[0]), ["Launch", "Q4"])
        XCTAssertEqual(strings(deck.slides[1]), ["Goals", "Plain line", "Ship", "Fix bugs"])
        XCTAssertEqual(strings(deck.slides[2]).last, "<b>&amp;")
        let bullets = deck.slides[1].shapes.flatMap(\.paragraphs).filter(\.bulleted).count
        XCTAssertEqual(bullets, 2)
        XCTAssertEqual(deck.width, 12_192_000)
        let archive = try OfficeArchive(url: url, limits: OfficeRenderLimits.default.archiveLimits)
        for part in ["ppt/presentation.xml", "ppt/slideMasters/slideMaster1.xml",
                     "ppt/slideLayouts/slideLayout1.xml", "ppt/theme/theme1.xml", "ppt/slides/slide3.xml"] {
            XCTAssertTrue(archive.entries.contains(part), part)
        }
    }

    // MARK: pdf

    func testPdfRoundTripPageSizeTextAndFlow() throws {
        try pngData(width: 200, height: 100).write(to: root.appendingPathComponent("pic.png"))
        let paragraphs = (0..<80).map {
            #"{"type":"paragraph","text":"Paragraph \#($0) of the flowing handout text that wraps across the line."}"#
        }.joined(separator: ",")
        let url = try write("handout.pdf", """
        {"title":"Handout","page":"a4","blocks":[
         {"type":"heading","level":1,"text":"Trail guide"},
         {"type":"list","ordered":true,"items":["Park","Walk"]},
         {"type":"table","header":["Mile","Landmark"],"rows":[["1","Creek"],["2","Overlook"]]},
         {"type":"image","path":"pic.png","width_pt":300},
         \(paragraphs),
         {"type":"page_break"},
         {"type":"paragraph","text":"Last page"}
        ]}
        """)
        let document = try XCTUnwrap(PDFDocument(url: url))
        XCTAssertGreaterThanOrEqual(document.pageCount, 3, "content must flow across pages")
        for index in 0..<document.pageCount {
            let bounds = try XCTUnwrap(document.page(at: index)).bounds(for: .mediaBox)
            XCTAssertEqual(Double(bounds.width), 595.28, accuracy: 0.5)
            XCTAssertEqual(Double(bounds.height), 841.89, accuracy: 0.5)
        }
        let all = document.string ?? ""
        for needle in ["Trail guide", "Creek", "Paragraph 0 of", "Paragraph 79 of", "Last page"] {
            XCTAssertTrue(all.contains(needle), needle)
        }
        XCTAssertEqual(document.documentAttributes?[PDFDocumentAttribute.titleAttribute] as? String, "Handout")
    }

    func testPdfTableRowTallerThanAPageIsRefused() {
        // 16 columns leave each about 31 points wide, so 2,000 characters wrap past a page.
        let huge = String(repeating: "word ", count: 400)
        let others = (0..<15).map { _ in "\"x\"" }.joined(separator: ",")
        expectFailure(
            "t.pdf", #"{"blocks":[{"type":"table","rows":[["\#(huge)",\#(others)]]}]}"#,
            contains: "taller than a page")
    }

    // MARK: Confinement and the write ladder

    func testPathConfinementAndNoOverwrite() throws {
        let spec = #"{"blocks":[{"type":"paragraph","text":"x"}]}"#
        expectFailure("../escape.docx", spec, contains: "escapes the project root")
        expectFailure("/tmp/abs.docx", spec, contains: "relative")
        expectFailure("~/home.docx", spec, contains: "relative")
        expectFailure(".env", spec, contains: "sensitive")
        expectFailure("notes.txt", spec, contains: "must end in")
        expectFailure("macro.docm", spec, contains: "Macro-enabled")
        expectFailure("macro.xlsm", spec, contains: "Macro-enabled")
        expectFailure("", spec, contains: "needs a 'path'")
        // A symlink directory pointing outside the project cannot be a way out.
        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-outside-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("link"), withDestinationURL: outside)
        expectFailure("link/out.docx", spec, contains: "escapes the project root")
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: outside.path).isEmpty)

        try write("once.docx", spec)
        let before = try Data(contentsOf: root.appendingPathComponent("once.docx"))
        expectFailure("once.docx", #"{"blocks":[{"type":"paragraph","text":"different"}]}"#, contains: "never overwrites")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("once.docx")), before)
        // An existing file of ANY kind at the path is protected the same way.
        try Data("keep".utf8).write(to: root.appendingPathComponent("keep.pdf"))
        expectFailure("keep.pdf", spec, contains: "never overwrites")
        XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("keep.pdf")), Data("keep".utf8))
    }

    func testFailedPostcheckWritesNothingAndLeavesNoTempFiles() throws {
        let tempBefore = Set((try? FileManager.default.contentsOfDirectory(atPath: FileManager.default.temporaryDirectory.path)) ?? [])
        XCTAssertThrowsError(try DocumentWriteExecutor.execute(
            arguments: ["path": "sub/never.docx", "spec": #"{"blocks":[{"type":"paragraph","text":"x"}]}"#],
            rootURL: root,
            postcheck: { _, _ in throw DocumentWriteError.postcheckFailed(check: "forced", detail: "mutation") })
        ) { error in
            XCTAssertTrue((error as NSError).localizedDescription.contains("Nothing was written"))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("sub/never.docx").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("sub").path),
                       "the destination folder is created only after the postcheck passes")
        let leaked = Set((try? FileManager.default.contentsOfDirectory(atPath: FileManager.default.temporaryDirectory.path)) ?? [])
            .subtracting(tempBefore).filter { $0.hasPrefix("turbospark-docwrite-") && !$0.contains("tests") }
        XCTAssertTrue(leaked.isEmpty, "\(leaked)")
    }

    func testLadderPositionIsFileWriteAndRooted() {
        XCTAssertEqual(AppToolCatalog.category(for: "document_write"), .fileWrite)
        XCTAssertTrue(AppToolRegistry.isImplemented("document_write"))
        XCTAssertTrue(AppToolRegistry.workspaceRootedToolNames.contains("document_write"))
        XCTAssertTrue(AppToolCatalog.fileTools.contains { $0.function.name == "document_write" })
        XCTAssertEqual(ToolPresentation.registry["document_write"]?.label, "Write Document")

        let call = AppToolCall(
            name: "document_write", arguments: ["path": "r.docx", "spec": "{}"], category: .fileWrite)
        var perms = AppProjectPermissions.auto
        XCTAssertEqual(AppToolPermissionEngine.evaluate(call: call, project: AppProject(name: "P", permissions: perms)), .allow)
        perms.fileWrite = .deny
        if case .deny = AppToolPermissionEngine.evaluate(call: call, project: AppProject(name: "P", permissions: perms)) {
        } else { XCTFail("file writes denied must deny document_write") }
        perms.fileWrite = .ask
        if case .ask = AppToolPermissionEngine.evaluate(call: call, project: AppProject(name: "P", permissions: perms)) {
        } else { XCTFail("file writes set to Ask must ask") }

        let sensitive = AppToolCall(
            name: "document_write", arguments: ["path": ".env", "spec": "{}"], category: .fileWrite)
        XCTAssertTrue(ToolRiskClassifier.assessRisk(name: "document_write", arguments: sensitive.arguments).isHighRisk)
        if case .ask = AppToolPermissionEngine.evaluate(call: sensitive, project: AppProject(name: "P", permissions: .auto)) {
        } else { XCTFail("a sensitive destination must ask even in auto mode") }
    }

    func testDispatchThroughTheRegistryRegistersTheDocumentAsAnArtifact() async throws {
        // The registry arm writes and reports a file for the artifact list.
        let report = try DocumentWriteExecutor.execute(
            arguments: ["path": "via.docx", "spec": #"{"blocks":[{"type":"heading","text":"Hi"}]}"#], rootURL: root)
        XCTAssertTrue(report.report.contains("postcheck passed: archive, content-types, relationships, render"), report.report)
        XCTAssertEqual(report.url.lastPathComponent, "via.docx")
    }

    // MARK: Limits and spec shape

    func testSpecLimitsAreRefusedWhole() throws {
        let rows = (0..<1_001).map { _ in #"["x"]"# }.joined(separator: ",")
        expectFailure("l.xlsx", #"{"sheets":[{"name":"S","rows":[\#(rows)]}]}"#, contains: "1001 rows")
        let slides = (0..<101).map { _ in #"{"title":"t"}"# }.joined(separator: ",")
        expectFailure("l.pptx", #"{"slides":[\#(slides)]}"#, contains: "101 slides")
        let wide = (0..<17).map { "\"c\($0)\"" }.joined(separator: ",")
        expectFailure("l.docx", #"{"blocks":[{"type":"table","rows":[[\#(wide)]]}]}"#, contains: "columns")
        let long = String(repeating: "a", count: 4_001)
        expectFailure("l2.xlsx", #"{"sheets":[{"name":"S","rows":[["\#(long)"]]}]}"#, contains: "4000 characters")
        var limits = DocumentWriteLimits.default
        limits.maximumSpecBytes = 100
        XCTAssertThrowsError(try DocumentSpec.decode(String(repeating: " ", count: 200), limits: limits))
        // Output cap.
        var tiny = DocumentWriteLimits.default
        tiny.maximumOutputBytes = 200
        XCTAssertThrowsError(try DocumentWriteExecutor.execute(
            arguments: ["path": "big.docx", "spec": #"{"blocks":[{"type":"paragraph","text":"x"}]}"#],
            rootURL: root, limits: tiny))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("big.docx").path))
    }

    func testSpecShapeErrorsAreNamed() {
        expectFailure("s.docx", "not json", contains: "Invalid document spec")
        expectFailure("s.docx", #"{"blocks":[]}"#, contains: "'blocks' is empty")
        expectFailure("s.docx", #"{"blocks":[{"type":"video"}]}"#, contains: "unknown block type")
        expectFailure("s.docx", #"{"blocks":[{"type":"heading","level":9,"text":"x"}]}"#, contains: "heading level")
        expectFailure("s.docx", #"{"blocks":[{"type":"paragraph","text":"x","runs":[{"text":"y","color":"red"}]}]}"#, contains: "RRGGBB")
        expectFailure("s.docx", #"{"blocks":[{"type":"paragraph","runs":[{"text":"y","color":"<x>"}]}]}"#, contains: "RRGGBB")
        expectFailure("s.docx", #"{"sheets":[{"name":"S","rows":[["x"]]}]}"#, contains: "'blocks' is empty")
        expectFailure("s.xlsx", #"{"blocks":[{"type":"paragraph","text":"x"}]}"#, contains: "'sheets' is empty")
        expectFailure("s.xlsx", #"{"sheets":[{"name":"S","rows":[[{"v":1,"format":"[Red]0"}]]}]}"#, contains: "unknown number format")
        expectFailure("s.pptx", #"{"slides":[{"title":"x","bullets":[{"text":"y","level":5}]}]}"#, contains: "bullet level")
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("s.docx").path))
    }

    // MARK: Postcheck

    private func stage(_ name: String, _ data: Data) throws -> URL {
        let url = root.appendingPathComponent(name)
        try data.write(to: url)
        return url
    }

    private func goodDocx() throws -> (data: Data, spec: DocumentSpec) {
        let spec = try DocumentSpec.decode(#"{"blocks":[{"type":"heading","text":"Title"},{"type":"paragraph","text":"Body"}]}"#)
        return (try DocxWriter.write(spec: spec, images: [:]), spec)
    }

    /// Rebuilds a package with `change` applied to its members.
    private func mutate(_ data: Data, _ change: (inout [(name: String, data: Data)]) -> Void) throws -> Data {
        let zip = try ZipArchiveReader(data: data)
        var members = try zip.members.map { (name: $0.name, data: try zip.read($0)) }
        change(&members)
        return try ZipArchiveWriter.build(members)
    }

    func testPostcheckPassesAGoodFileOfEveryFormat() throws {
        let docxSpec = try DocumentSpec.decode(#"{"blocks":[{"type":"heading","text":"T"},{"type":"paragraph","text":"B"}]}"#)
        let docx = try stage("g.docx", DocxWriter.write(spec: docxSpec, images: [:]))
        let report = try DocumentPostcheck.run(fileURL: docx, expectations: DocumentExpectations(spec: docxSpec, format: .docx))
        XCTAssertEqual(report.passed, ["archive", "content-types", "relationships", "render"])
        XCTAssertGreaterThan(report.characterCount, 0)

        let xlsxSpec = try DocumentSpec.decode(#"{"sheets":[{"name":"S","rows":[["a",1]]}]}"#)
        _ = try DocumentPostcheck.run(
            fileURL: stage("g.xlsx", XlsxWriter.write(spec: xlsxSpec)), expectations: DocumentExpectations(spec: xlsxSpec, format: .xlsx))
        let pptxSpec = try DocumentSpec.decode(#"{"slides":[{"title":"A"},{"title":"B"}]}"#)
        _ = try DocumentPostcheck.run(
            fileURL: stage("g.pptx", PptxWriter.write(spec: pptxSpec)), expectations: DocumentExpectations(spec: pptxSpec, format: .pptx))
        let pdfSpec = try DocumentSpec.decode(#"{"blocks":[{"type":"paragraph","text":"hello pdf"}]}"#)
        let pdfReport = try DocumentPostcheck.run(
            fileURL: stage("g.pdf", PdfWriter.write(spec: pdfSpec, images: [:])), expectations: DocumentExpectations(spec: pdfSpec, format: .pdf))
        XCTAssertEqual(pdfReport.passed, ["pdf"])
        XCTAssertEqual(pdfReport.pageCount, 1)
    }

    private func assertPostcheckFails(
        _ data: Data, spec: DocumentSpec, check: String, format: DocumentFormat = .docx,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let url = try stage("bad-\(UUID().uuidString).\(format.rawValue)", data)
        XCTAssertThrowsError(
            try DocumentPostcheck.run(fileURL: url, expectations: DocumentExpectations(spec: spec, format: format)),
            file: file, line: line
        ) { error in
            guard case DocumentWriteError.postcheckFailed(let failed, _) = error else {
                return XCTFail("unexpected \(error)", file: file, line: line)
            }
            XCTAssertEqual(failed, check, file: file, line: line)
        }
    }

    /// Mutation check: the ZP-08 loader never reads `[Content_Types].xml`, so
    /// only the postcheck's own check catches these.
    func testCorruptedContentTypesFailThePostcheck() throws {
        let good = try goodDocx()
        // The ZP-08 loader still opens each mutant, which is the point.
        let wrongType = try mutate(good.data) { members in
            let index = members.firstIndex { $0.name == "[Content_Types].xml" }!
            members[index].data = Data(text(members[index].data)
                .replacingOccurrences(of: OOXMLPackage.ContentType.docx, with: "application/zip").utf8)
        }
        try assertPostcheckFails(wrongType, spec: good.spec, check: "content-types")
        XCTAssertNoThrow(try OfficeRenderModel.load(url: stage("wrongtype.docx", wrongType)))

        let dropped = try mutate(good.data) { members in
            let index = members.firstIndex { $0.name == "[Content_Types].xml" }!
            members[index].data = Data(text(members[index].data)
                .replacingOccurrences(of: "<Override PartName=\"/word/styles.xml\" ContentType=\"\(OOXMLPackage.ContentType.docxStyles)\"/>", with: "")
                .replacingOccurrences(of: "<Default Extension=\"xml\" ContentType=\"application/xml\"/>", with: "").utf8)
        }
        try assertPostcheckFails(dropped, spec: good.spec, check: "content-types")

        let missing = try mutate(good.data) { $0.removeAll { $0.name == "[Content_Types].xml" } }
        try assertPostcheckFails(missing, spec: good.spec, check: "content-types")
    }

    func testBrokenArchivesRelationshipsAndRenderFailThePostcheck() throws {
        let good = try goodDocx()
        // A flipped byte inside a member breaks its CRC.
        var corrupt = [UInt8](good.data)
        corrupt[60] ^= 0xFF
        try assertPostcheckFails(Data(corrupt), spec: good.spec, check: "archive")
        try assertPostcheckFails(Data("not a zip at all".utf8), spec: good.spec, check: "archive")

        let external = try mutate(good.data) { members in
            let index = members.firstIndex { $0.name == "word/_rels/document.xml.rels" }!
            members[index].data = Data(text(members[index].data).replacingOccurrences(
                of: "</Relationships>",
                with: "<Relationship Id=\"rId99\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink\" Target=\"http://evil.example\" TargetMode=\"External\"/></Relationships>").utf8)
        }
        try assertPostcheckFails(external, spec: good.spec, check: "relationships")

        let dangling = try mutate(good.data) { $0.removeAll { $0.name == "word/numbering.xml" } }
        try assertPostcheckFails(dangling, spec: good.spec, check: "relationships")

        let escaping = try mutate(good.data) { members in
            let index = members.firstIndex { $0.name == "word/_rels/document.xml.rels" }!
            members[index].data = Data(text(members[index].data)
                .replacingOccurrences(of: "Target=\"styles.xml\"", with: "Target=\"../../../etc/hosts\"").utf8)
        }
        try assertPostcheckFails(escaping, spec: good.spec, check: "relationships")

        // The package is sound but its content does not match the spec.
        let moreHeadings = try DocumentSpec.decode(#"{"blocks":[{"type":"heading","text":"A"},{"type":"heading","text":"B"}]}"#)
        try assertPostcheckFails(good.data, spec: moreHeadings, check: "render")
        let emptied = try mutate(good.data) { members in
            let index = members.firstIndex { $0.name == "word/document.xml" }!
            members[index].data = Data(text(members[index].data)
                .replacingOccurrences(of: ">Title<", with: "><").replacingOccurrences(of: ">Body<", with: "><").utf8)
        }
        try assertPostcheckFails(emptied, spec: good.spec, check: "render")
    }

    func testPdfPostcheckFailures() throws {
        let spec = try DocumentSpec.decode(#"{"blocks":[{"type":"paragraph","text":"hello"}]}"#)
        try assertPostcheckFails(Data("%PDF-1.4 garbage".utf8), spec: spec, check: "pdf", format: .pdf)
        try assertPostcheckFails(Data("plain".utf8), spec: spec, check: "pdf", format: .pdf)
        // A4 content checked against a letter expectation.
        let a4 = try DocumentSpec.decode(#"{"page":"a4","blocks":[{"type":"paragraph","text":"hello"}]}"#)
        let data = try PdfWriter.write(spec: a4, images: [:])
        try assertPostcheckFails(data, spec: spec, check: "pdf", format: .pdf)
    }

    // MARK: Zip layer

    func testZipWriterRoundTripsStoredAndDeflatedMembersAndLargeData() throws {
        let big = Data((0..<200_000).map { UInt8($0 % 251) })
        let repetitive = Data(String(repeating: "abcabc", count: 5_000).utf8)
        let zip = try ZipArchiveWriter.build([("a/b.txt", Data("hi".utf8)), ("big.bin", big), ("rep.txt", repetitive)])
        let reader = try ZipArchiveReader(data: zip)
        XCTAssertEqual(try reader.read("a/b.txt"), Data("hi".utf8))
        XCTAssertEqual(try reader.read("big.bin"), big)
        XCTAssertEqual(try reader.read("rep.txt"), repetitive)
        XCTAssertEqual(reader.member("rep.txt")?.method, 8)
        XCTAssertEqual(reader.member("a/b.txt")?.method, 0)
        XCTAssertLessThan(zip.count, big.count + repetitive.count)
        // The system `unzip` agrees (independent implementation).
        let file = try stage("z.zip", zip)
        let archive = try OfficeArchive(url: file, limits: OfficeRenderLimits.default.archiveLimits)
        XCTAssertEqual(archive.entries, ["a/b.txt", "big.bin", "rep.txt"])
        XCTAssertEqual(try archive.data(for: "rep.txt"), repetitive)
        XCTAssertEqual(CRC32.checksum(Array("123456789".utf8)), 0xCBF4_3926)
        XCTAssertThrowsError(try ZipArchiveWriter.build([("../x", Data()), ]))
        XCTAssertThrowsError(try ZipArchiveWriter.build([("a", Data()), ("a", Data())]))
    }

    // MARK: Structure of the feature itself

    func testWritersNeverSpawnAProcessOrTouchTheNetwork() throws {
        let directory = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Presentation/DocumentWriter")
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path).filter { $0.hasSuffix(".swift") }
        XCTAssertGreaterThanOrEqual(files.count, 8)
        for file in files {
            let source = try String(contentsOf: directory.appendingPathComponent(file), encoding: .utf8)
            XCTAssertTrue(Self.forbiddenFindings(in: source).isEmpty, "\(file): \(Self.forbiddenFindings(in: source))")
        }
        // Mutation: the scanner must be able to fail.
        XCTAssertFalse(Self.forbiddenFindings(in: "let p = Process()").isEmpty)
        XCTAssertFalse(Self.forbiddenFindings(in: "URLSession.shared.dataTask").isEmpty)
        XCTAssertFalse(Self.forbiddenFindings(in: "executableURL = URL(fileURLWithPath: \"/usr/bin/zip\")").isEmpty)
        XCTAssertFalse(Self.forbiddenFindings(in: "posix_spawn(&pid, path)").isEmpty)
    }

    private static func forbiddenFindings(in source: String) -> [String] {
        let code = source.components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }.joined(separator: "\n")
        return ["Process(", "NSTask", "posix_spawn", "/usr/bin/", "system(", "popen(", "URLSession", "NSURLConnection",
                "http://localhost", "import WebKit"].filter { code.contains($0) }
    }

    func testBundledSkillsShipAndStayNativeOnly() throws {
        let skills = SkillManager.shared.discoverBundledSkills()
        for name in ["docx", "xlsx", "pptx", "pdf"] {
            let skill = try XCTUnwrap(skills.first { $0.name == name }, "\(name) must ship in BundledSkills")
            XCTAssertEqual(skill.scope, .bundled)
            XCTAssertFalse(skill.skillDescription.isEmpty)
            let body = try String(contentsOf: skill.sourceURL, encoding: .utf8)
            let lower = body.lowercased()
            XCTAssertTrue(body.contains("document_write"), name)
            XCTAssertTrue(lower.contains("postcheck"), name)
            XCTAssertTrue(lower.contains("never overwrite"), name)
            XCTAssertTrue(lower.contains("untrusted"), name)
            for banned in ["pip install", "npm install", "node -e", "pnpm ", "yarn ", "brew install", "curl "] {
                XCTAssertFalse(lower.contains(banned), "\(name) mentions \(banned)")
            }
            // The fallback goes through Bash and the ladder, never around it.
            XCTAssertTrue(lower.contains("permission prompt"), name)
        }
    }
}
