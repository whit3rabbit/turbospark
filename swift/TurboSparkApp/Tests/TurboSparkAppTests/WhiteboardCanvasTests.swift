import AppKit
import PencilKit
import XCTest
@testable import TurboSparkApp

/// ZP-10: strokes and erasing, history, the canvas view's pointer handling,
/// the PNG export, the right-column slot and the composer handoff.
@MainActor
final class WhiteboardCanvasTests: XCTestCase {
    private var directory: URL!

    override func setUp() {
        super.setUp()
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("whiteboard-canvas-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
        super.tearDown()
    }

    private func line(_ a: CGPoint, _ b: CGPoint, color: NSColor = .black, width: CGFloat = 4) -> PKStroke {
        WhiteboardStrokes.stroke(through: [a, b], color: color, width: width)!
    }

    // MARK: - Stroke geometry

    func testStrokeDropsNearDuplicatesAndAClickBecomesADot() throws {
        let still = WhiteboardStrokes.stroke(
            through: [CGPoint(x: 5, y: 5), CGPoint(x: 5.1, y: 5), CGPoint(x: 5.2, y: 5)],
            color: .black, width: 3)
        XCTAssertEqual(try XCTUnwrap(still).path.count, 2, "one kept point plus the dot's second point")
        XCTAssertNil(WhiteboardStrokes.stroke(through: [], color: .black, width: 3))
    }

    func testEraserHitsOnlyStrokesItTouches() {
        var drawing = PKDrawing()
        drawing.strokes = [
            line(CGPoint(x: 0, y: 50), CGPoint(x: 100, y: 50)),
            line(CGPoint(x: 0, y: 150), CGPoint(x: 100, y: 150)),
        ]
        // On the first stroke's span, between its two control points.
        let erased = WhiteboardStrokes.erasing(drawing, at: [CGPoint(x: 50, y: 53)])
        XCTAssertEqual(erased.strokes.count, 1)
        XCTAssertEqual(erased.strokes[0].path[0].location.y, 150, accuracy: 0.01)
        // Miss: the very same value comes back.
        XCTAssertEqual(WhiteboardStrokes.erasing(drawing, at: [CGPoint(x: 50, y: 100)]), drawing)
    }

    // MARK: - History and tools

    func testUndoRedoClearAndRedoInvalidation() {
        let controller = WhiteboardCanvasController()
        controller.addStroke(line(CGPoint(x: 0, y: 10), CGPoint(x: 50, y: 10)))
        controller.addStroke(line(CGPoint(x: 0, y: 30), CGPoint(x: 50, y: 30)))
        XCTAssertEqual(controller.strokeCount, 2)
        XCTAssertTrue(controller.canUndo)
        XCTAssertFalse(controller.canRedo)

        controller.undo()
        XCTAssertEqual(controller.strokeCount, 1)
        XCTAssertTrue(controller.canRedo)
        controller.redo()
        XCTAssertEqual(controller.strokeCount, 2)

        controller.clear()
        XCTAssertEqual(controller.strokeCount, 0)
        controller.undo()
        XCTAssertEqual(controller.strokeCount, 2, "clear is one undoable step")

        controller.undo()
        controller.addStroke(line(CGPoint(x: 0, y: 70), CGPoint(x: 50, y: 70)))
        XCTAssertFalse(controller.canRedo, "a new stroke discards the redo branch")
    }

    func testHistoryDepthIsBounded() {
        var history = WhiteboardHistory(base: Data([0]))
        for i in 1...(WhiteboardHistory.maxUndoDepth + 20) { history.record(Data([UInt8(i % 250 + 1), UInt8(i / 250)])) }
        var steps = 0
        while history.undo() != nil { steps += 1 }
        XCTAssertEqual(steps, WhiteboardHistory.maxUndoDepth)
    }

    func testToolSettingsReachTheCanvas() {
        let controller = WhiteboardCanvasController()
        controller.swatch = .blue
        controller.width = 9
        controller.tool = .eraser
        XCTAssertEqual(controller.canvas.inkColor, WhiteboardSwatch.blue.color)
        XCTAssertEqual(controller.canvas.inkWidth, 9)
        XCTAssertEqual(controller.canvas.tool, .eraser)
        XCTAssertEqual(WhiteboardSwatch.allCases.count, 6)
        XCTAssertEqual(Set(WhiteboardSwatch.allCases.map { $0.color.usingColorSpace(.sRGB)!.hexKey }).count, 6)
    }

    // MARK: - Pointer handling

    private func window(for view: NSView) -> NSWindow {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300), styleMask: [.titled],
            backing: .buffered, defer: true)
        window.contentView = view
        return window
    }

    private func event(_ type: NSEvent.EventType, at viewPoint: CGPoint, in view: NSView) -> NSEvent {
        let windowPoint = view.convert(viewPoint, to: nil)
        return NSEvent.mouseEvent(
            with: type, location: windowPoint, modifierFlags: [], timestamp: 0,
            windowNumber: view.window?.windowNumber ?? 0, context: nil, eventNumber: 0,
            clickCount: 1, pressure: 1)!
    }

    func testMouseDragDrawsOneStrokeAsOneHistoryStep() {
        let controller = WhiteboardCanvasController()
        let view = controller.canvas
        _ = window(for: view)
        view.mouseDown(with: event(.leftMouseDown, at: CGPoint(x: 50, y: 50), in: view))
        view.mouseDragged(with: event(.leftMouseDragged, at: CGPoint(x: 100, y: 60), in: view))
        view.mouseDragged(with: event(.leftMouseDragged, at: CGPoint(x: 150, y: 70), in: view))
        XCTAssertEqual(controller.strokeCount, 0, "nothing is committed mid-gesture")
        view.mouseUp(with: event(.leftMouseUp, at: CGPoint(x: 200, y: 80), in: view))
        XCTAssertEqual(controller.strokeCount, 1)
        let first = view.drawing.strokes[0].path[0].location
        XCTAssertEqual(first.x, 50, accuracy: 0.5)
        XCTAssertEqual(first.y, 50, accuracy: 0.5, "view and drawing share one top-left space")
        controller.undo()
        XCTAssertEqual(controller.strokeCount, 0)
    }

    func testEraserDragRemovesStrokesAndIsOneStep() {
        let controller = WhiteboardCanvasController()
        let view = controller.canvas
        _ = window(for: view)
        controller.addStroke(line(CGPoint(x: 20, y: 100), CGPoint(x: 200, y: 100)))
        controller.addStroke(line(CGPoint(x: 20, y: 140), CGPoint(x: 200, y: 140)))
        controller.tool = .eraser
        view.mouseDown(with: event(.leftMouseDown, at: CGPoint(x: 100, y: 60), in: view))
        view.mouseDragged(with: event(.leftMouseDragged, at: CGPoint(x: 100, y: 160), in: view))
        view.mouseUp(with: event(.leftMouseUp, at: CGPoint(x: 100, y: 160), in: view))
        XCTAssertEqual(controller.strokeCount, 0, "a fast drag across both strokes still hits both")
        controller.undo()
        XCTAssertEqual(controller.strokeCount, 2, "one gesture, one undo step")
    }

    // MARK: - Persistence through the controller

    func testChangesQueueInTheStoreAndSwitchingBoardsFlushes() throws {
        let store = WhiteboardStore(directory: directory, debounce: .seconds(30))
        let a = store.createBoard(workspaceKey: "w")
        let b = store.createBoard(workspaceKey: "w")
        let controller = WhiteboardCanvasController()
        controller.open(boardID: a.id, store: store)
        controller.addStroke(line(CGPoint(x: 0, y: 10), CGPoint(x: 50, y: 10)))
        XCTAssertTrue(store.hasPendingWrites)

        controller.open(boardID: b.id, store: store)
        XCTAssertFalse(store.hasPendingWrites, "leaving a board flushes it")
        XCTAssertEqual(controller.strokeCount, 0, "board B is blank")
        XCTAssertFalse(controller.canUndo, "history does not leak across boards")

        controller.open(boardID: a.id, store: store)
        XCTAssertEqual(controller.strokeCount, 1)
    }

    // MARK: - Export

    private func decode(_ png: Data) throws -> (width: Int, height: Int, rgba: [UInt8]) {
        let rep = try XCTUnwrap(NSBitmapImageRep(data: png))
        let cg = try XCTUnwrap(rep.cgImage)
        var buffer = [UInt8](repeating: 0, count: cg.width * cg.height * 4)
        buffer.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress, width: cg.width, height: cg.height, bitsPerComponent: 8,
                bytesPerRow: cg.width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: cg.width, height: cg.height))
        }
        return (cg.width, cg.height, buffer)
    }

    func testEmptyBoardRefusesToExport() {
        XCTAssertThrowsError(try WhiteboardExporter.png(for: PKDrawing())) {
            XCTAssertEqual($0 as? WhiteboardExporter.ExportError, .empty)
        }
    }

    func testExportIsOpaqueNaturalSizeAndUpright() throws {
        var drawing = PKDrawing()
        // A red bar near the TOP of a 200 x 100 region.
        drawing.strokes = [
            line(CGPoint(x: 100, y: 100), CGPoint(x: 300, y: 110), color: WhiteboardSwatch.red.color, width: 6),
            line(CGPoint(x: 100, y: 190), CGPoint(x: 300, y: 200), color: .white, width: 1),
        ]
        let rect = WhiteboardExporter.exportRect(for: drawing)
        let scale = WhiteboardExporter.scale(for: rect)
        XCTAssertEqual(scale, WhiteboardExporter.preferredScale)
        let expected = WhiteboardExporter.pixelSize(for: rect, scale: scale)
        let png = try WhiteboardExporter.png(for: drawing)
        let out = try decode(png)
        XCTAssertEqual(out.width, expected.width)
        XCTAssertEqual(out.height, expected.height)
        XCTAssertGreaterThan(out.width, Int(200 * scale), "bounds plus a margin on each side")

        func px(_ x: Int, _ y: Int) -> (r: Int, g: Int, b: Int, a: Int) {
            let i = (y * out.width + x) * 4
            return (Int(out.rgba[i]), Int(out.rgba[i + 1]), Int(out.rgba[i + 2]), Int(out.rgba[i + 3]))
        }
        let corner = px(1, 1)
        XCTAssertEqual([corner.r, corner.g, corner.b, corner.a], [255, 255, 255, 255], "opaque white paper")

        // Red ink lives in the top half only (strokes 1 sits at the top of the
        // drawing), so a flipped render would put it in the bottom half.
        func redRows(_ range: Range<Int>) -> Int {
            range.filter { y in
                (0..<out.width).contains { x in let p = px(x, y); return p.r > 180 && p.g < 90 && p.b < 90 }
            }.count
        }
        XCTAssertGreaterThan(redRows(0..<(out.height / 2)), 0)
        XCTAssertEqual(redRows((out.height / 2)..<out.height), 0)
    }

    func testExportCapsPixelCountByLoweringTheScale() throws {
        var drawing = PKDrawing()
        drawing.strokes = [line(CGPoint(x: 0, y: 0), CGPoint(x: 1000, y: 800))]
        let cap = 50_000
        let image = try WhiteboardExporter.render(drawing, maxPixels: cap)
        XCTAssertLessThanOrEqual(image.width * image.height, Int(Double(cap) * 1.05))
        XCTAssertGreaterThan(image.width * image.height, cap / 2, "scaled down, not cropped")
        let uncapped = try WhiteboardExporter.render(drawing)
        XCTAssertGreaterThan(uncapped.width * uncapped.height, image.width * image.height)
    }

    func testExportFileNameNeverCarriesAPath() {
        XCTAssertEqual(WhiteboardExporter.fileName(forBoardTitle: "Board 1"), "Board 1.png")
        XCTAssertEqual(WhiteboardExporter.fileName(forBoardTitle: "a/b:c"), "a-b-c.png")
        XCTAssertEqual(WhiteboardExporter.fileName(forBoardTitle: "  "), "whiteboard.png")
    }

    // MARK: - Right column

    func testWhiteboardOwnsTheColumnBetweenTheBrowserAndTheInspector() {
        func resolve(browser: Bool, whiteboard: Bool, inspector: Bool, preview: UUID? = nil) -> AppRightColumnClaimant {
            AppRightColumnClaimant.resolve(
                openArtifactID: nil, previewAttachmentID: preview, isInspectorVisible: inspector,
                showBrowserPane: browser, showWhiteboardPane: whiteboard)
        }
        XCTAssertEqual(resolve(browser: false, whiteboard: true, inspector: true), .whiteboard)
        XCTAssertEqual(resolve(browser: true, whiteboard: true, inspector: true), .browser)
        let id = UUID()
        XCTAssertEqual(resolve(browser: false, whiteboard: true, inspector: true, preview: id), .filePreview(id))
        XCTAssertEqual(resolve(browser: false, whiteboard: false, inspector: true), .inspector)
        XCTAssertTrue(AppRightColumnClaimant.whiteboard.isWhiteboardPane)
        XCTAssertFalse(AppRightColumnClaimant.whiteboard.isPreviewPane)
        XCTAssertTrue(AppRightColumnClaimant.whiteboard.isVisible)
    }

    func testWhiteboardWidthFeedsTheWindowFloor() {
        let board = AppChromeLayout.minimumWindowWidth(isPanelVisible: true, rightColumn: .whiteboard)
        let none = AppChromeLayout.minimumWindowWidth(isPanelVisible: true, rightColumn: .none)
        XCTAssertEqual(board - none, AppChromeLayout.whiteboardPanelWidth + AppChromeLayout.dividerWidth)
        XCTAssertNotEqual(AppChromeLayout.whiteboardPanelWidth, AppChromeLayout.inspectorWidth,
            "a distinct width, so the arm cannot pass by reusing the inspector's")
    }

    func testOpeningTheWhiteboardClosesTheBrowserSlotAndTheReverse() {
        let model = AppModel()
        model.setWhiteboardPaneOpen(true)
        XCTAssertTrue(model.whiteboardPaneIsOpen)
        model.browserPaneIsOpen = true
        model.setWhiteboardPaneOpen(true)
        XCTAssertFalse(model.browserPaneIsOpen, "one slot")
        model.setWhiteboardPaneOpen(false)
        XCTAssertFalse(model.whiteboardPaneIsOpen)
    }

    // MARK: - Composer handoff

    func testExportAttachesAnImageToTheFocusedChatAndLeavesTheBoardAlone() throws {
        let model = AppModel()
        let store = WhiteboardStore(directory: directory, debounce: .seconds(30))
        let board = store.createBoard(workspaceKey: "w")
        let controller = WhiteboardCanvasController()
        controller.open(boardID: board.id, store: store)
        controller.addStroke(line(CGPoint(x: 10, y: 10), CGPoint(x: 120, y: 60)))
        let before = controller.canvas.drawing

        let png = try controller.exportPNG()
        let chatID = model.selectedChatID
        model.attachWhiteboardExport(png, boardTitle: board.title)

        let chat = try XCTUnwrap(model.chats.first { $0.id == chatID })
        let attachment = try XCTUnwrap(chat.draftAttachments.last)
        XCTAssertEqual(attachment.fileName, "\(board.title).png")
        XCTAssertEqual(attachment.formatLabel, "Image")
        XCTAssertNotNil(attachment.sourcePath)
        XCTAssertEqual(attachment.sourceByteSize, png.count)
        XCTAssertTrue(attachment.isSendableImage)
        XCTAssertEqual(controller.canvas.drawing, before, "export does not touch the board")
        // No vision model is loaded here: the attach still happens, and the
        // toast says why a send would be refused instead of claiming success.
        XCTAssertEqual(model.activeToast?.style, .warning)
    }
}

private extension NSColor {
    var hexKey: String {
        String(format: "%.3f-%.3f-%.3f", redComponent, greenComponent, blueComponent)
    }
}
