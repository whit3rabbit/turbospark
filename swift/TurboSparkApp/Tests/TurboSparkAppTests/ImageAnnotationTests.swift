import AppKit
import CryptoKit
import SwiftUI
import XCTest
@testable import TurboSparkApp

/// ZP-11: coordinate mapping, history semantics, flatten fidelity and source
/// immutability for the image preview annotation layer.
@MainActor
final class ImageAnnotationTests: XCTestCase {
    // MARK: - Helpers

    private func solidImage(width: Int, height: Int, gray: UInt8 = 200) -> CGImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let g = CGFloat(gray) / 255
        ctx.setFillColor(CGColor(srgbRed: g, green: g, blue: g, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return ctx.makeImage()!
    }

    /// RGBA bytes, row 0 = top row, in a fixed layout so two renders compare.
    private func pixels(_ image: CGImage) -> [UInt8] {
        let width = image.width, height = image.height
        var buffer = [UInt8](repeating: 0, count: width * height * 4)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        buffer.withUnsafeMutableBytes { raw in
            let ctx = CGContext(
                data: raw.baseAddress, width: width, height: height, bitsPerComponent: 8,
                bytesPerRow: width * 4, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
        return buffer
    }

    private func pixel(_ buffer: [UInt8], width: Int, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8) {
        let i = (y * width + x) * 4
        return (buffer[i], buffer[i + 1], buffer[i + 2])
    }

    private func isInk(_ p: (r: UInt8, g: UInt8, b: UInt8)) -> Bool {
        p.r > 200 && p.g < 80 && p.b < 80
    }

    private func decodePNG(_ data: Data) throws -> CGImage {
        let source = try XCTUnwrap(CGImageSourceCreateWithData(data as CFData, nil))
        return try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
    }

    private func box(_ x: CGFloat, _ y: CGFloat, _ w: CGFloat, _ h: CGFloat) -> ImageAnnotation {
        ImageAnnotation(kind: .box(CGRect(x: x, y: y, width: w, height: h)))
    }

    // MARK: - Coordinate mapping (CUJ 2)

    func testViewImageMappingRoundTripsAtEveryZoom() {
        var viewport = AnnotationViewport(imageSize: CGSize(width: 2000, height: 1000))
        viewport.fitScale = 0.4
        for zoom: CGFloat in [1, 1.25, 2, 4, 8] {
            viewport.setZoom(zoom)
            XCTAssertEqual(viewport.scale, 0.4 * zoom, accuracy: 1e-9)
            let imagePoint = CGPoint(x: 733, y: 281)
            let back = viewport.imagePoint(fromView: viewport.viewPoint(fromImage: imagePoint))
            XCTAssertEqual(back.x, imagePoint.x, accuracy: 1e-6, "zoom \(zoom)")
            XCTAssertEqual(back.y, imagePoint.y, accuracy: 1e-6, "zoom \(zoom)")
        }
    }

    func testZoomIsClampedToRange() {
        var viewport = AnnotationViewport(imageSize: CGSize(width: 10, height: 10))
        viewport.setZoom(400)
        XCTAssertEqual(viewport.zoom, AnnotationViewport.zoomRange.upperBound)
        viewport.setZoom(0.01)
        XCTAssertEqual(viewport.zoom, AnnotationViewport.zoomRange.lowerBound)
    }

    func testAnnotationHitTestTracksTheImageAtEveryZoom() {
        // A box outline at image x = 500. At each zoom the view point that
        // lands on that outline moves with the scale, and a point on the
        // OLD view position must stop hitting once the zoom changed.
        var session = ImageAnnotationSession(imageSize: CGSize(width: 2000, height: 1000))
        session.viewport.fitScale = 0.25
        session.history.add(box(500, 200, 600, 400))
        for zoom: CGFloat in [1, 2, 4, 8] {
            session.viewport.setZoom(zoom)
            let onOutline = session.viewport.viewPoint(fromImage: CGPoint(x: 500, y: 400))
            XCTAssertNotNil(session.annotation(atView: onOutline), "outline at zoom \(zoom)")
            let inside = session.viewport.viewPoint(fromImage: CGPoint(x: 800, y: 400))
            XCTAssertNil(session.annotation(atView: inside), "interior is not a hit at zoom \(zoom)")
        }
        session.viewport.setZoom(1)
        let atOne = session.viewport.viewPoint(fromImage: CGPoint(x: 500, y: 400))
        session.viewport.setZoom(4)
        XCTAssertNil(session.annotation(atView: atOne))
    }

    func testHitTestingCoversArrowAndText() {
        let style = AnnotationStyle(imageSize: CGSize(width: 1000, height: 1000))
        let arrow = ImageAnnotation(kind: .arrow(from: CGPoint(x: 100, y: 100), to: CGPoint(x: 300, y: 100)))
        XCTAssertTrue(arrow.hits(CGPoint(x: 200, y: 102), tolerance: 4, style: style))
        XCTAssertFalse(arrow.hits(CGPoint(x: 200, y: 160), tolerance: 4, style: style))
        let text = ImageAnnotation(kind: .text(origin: CGPoint(x: 50, y: 50), string: "here"))
        let rect = text.bounds(style: style)
        XCTAssertGreaterThan(rect.width, 10)
        XCTAssertTrue(text.hits(CGPoint(x: rect.midX, y: rect.midY), tolerance: 0, style: style))
        XCTAssertFalse(text.hits(CGPoint(x: rect.maxX + 40, y: rect.midY), tolerance: 2, style: style))
    }

    // MARK: - History (CUJ 2)

    func testUndoRemovesLastTwoOperationsWithoutTouchingPanZoom() {
        var session = ImageAnnotationSession(imageSize: CGSize(width: 800, height: 600))
        session.history.add(box(10, 10, 50, 50))
        session.history.add(box(100, 100, 50, 50))
        session.history.add(ImageAnnotation(kind: .arrow(from: .zero, to: CGPoint(x: 90, y: 90))))
        session.viewport.fitScale = 0.5
        session.viewport.setZoom(4)
        let viewportBefore = session.viewport

        session.undo()
        session.undo()

        XCTAssertEqual(session.history.annotations.count, 1)
        XCTAssertEqual(session.viewport, viewportBefore, "undo must not touch zoom")
        XCTAssertTrue(session.history.canRedo)
    }

    func testRedoReplaysAndANewEditClearsIt() {
        var history = AnnotationHistory()
        let a = box(0, 0, 20, 20), b = box(30, 30, 20, 20)
        history.add(a)
        history.add(b)
        history.undo()
        XCTAssertEqual(history.annotations, [a])
        history.redo()
        XCTAssertEqual(history.annotations, [a, b])
        history.undo()
        history.add(box(60, 60, 20, 20))
        XCTAssertFalse(history.canRedo)
        XCTAssertFalse(history.redo())
    }

    func testMoveAndDeleteAreOneHistoryEntryEach() {
        var history = AnnotationHistory()
        let original = box(10, 10, 40, 40)
        history.add(original)
        history.replace(original.translated(by: CGSize(width: 25, height: 5)))
        history.remove(id: original.id)
        XCTAssertTrue(history.annotations.isEmpty)
        history.undo()
        XCTAssertEqual(history.annotations.count, 1)
        history.undo()
        XCTAssertEqual(history.annotations, [original])
    }

    func testStaleReplaceCannotResurrectADeletedMark() {
        var history = AnnotationHistory()
        let a = box(0, 0, 20, 20)
        history.add(a)
        history.remove(id: a.id)
        history.replace(a.translated(by: CGSize(width: 5, height: 5)))
        XCTAssertTrue(history.annotations.isEmpty)
    }

    func testUndoDropsASelectionThatNoLongerExists() {
        var session = ImageAnnotationSession(imageSize: CGSize(width: 100, height: 100))
        let a = box(0, 0, 20, 20)
        session.history.add(a)
        session.selectedID = a.id
        session.undo()
        XCTAssertNil(session.selectedID)
    }

    func testHistoryDepthIsBounded() {
        var history = AnnotationHistory()
        for i in 0..<(AnnotationHistory.maxUndoDepth + 20) {
            history.add(box(CGFloat(i), 0, 10, 10))
        }
        var undone = 0
        while history.undo() { undone += 1 }
        XCTAssertEqual(undone, AnnotationHistory.maxUndoDepth)
    }

    // MARK: - Flatten (CUJ 1)

    func testFlattenKeepsNaturalPixelSize() throws {
        let base = solidImage(width: 321, height: 123)
        let data = try AnnotationExporter.flatten(base, annotations: [box(20, 20, 80, 40)])
        let decoded = try decodePNG(data)
        XCTAssertEqual(decoded.width, 321)
        XCTAssertEqual(decoded.height, 123)
    }

    func testFlattenIsDeterministicAtThePixelLevel() throws {
        let base = solidImage(width: 200, height: 120)
        let marks = [
            box(10, 10, 100, 60),
            ImageAnnotation(kind: .arrow(from: CGPoint(x: 20, y: 100), to: CGPoint(x: 150, y: 90))),
            ImageAnnotation(kind: .text(origin: CGPoint(x: 120, y: 20), string: "bug")),
        ]
        let first = pixels(try decodePNG(try AnnotationExporter.flatten(base, annotations: marks)))
        let second = pixels(try decodePNG(try AnnotationExporter.flatten(base, annotations: marks)))
        XCTAssertEqual(first, second)
    }

    func testFlattenBurnsTheBoxAtImagePixelsWithTopLeftOrigin() throws {
        let base = solidImage(width: 400, height: 300)
        // Box hugging the top-left corner; stroke is 2 px for this size.
        let data = try AnnotationExporter.flatten(base, annotations: [box(40, 30, 120, 90)])
        let out = pixels(try decodePNG(data))
        // Left edge of the box, vertically centered, is ink.
        XCTAssertTrue(isInk(pixel(out, width: 400, x: 40, y: 75)))
        // Top edge is near y = 30, NOT near the bottom (y = 270).
        XCTAssertTrue(isInk(pixel(out, width: 400, x: 100, y: 30)))
        XCTAssertFalse(isInk(pixel(out, width: 400, x: 100, y: 270)))
        // The interior and the far corner keep the source gray.
        XCTAssertEqual(pixel(out, width: 400, x: 100, y: 75).r, 200)
        XCTAssertEqual(pixel(out, width: 400, x: 390, y: 290).r, 200)
    }

    func testFlattenWithNoAnnotationsMatchesTheSourcePixels() throws {
        let base = solidImage(width: 64, height: 48, gray: 90)
        let out = pixels(try decodePNG(try AnnotationExporter.flatten(base, annotations: [])))
        XCTAssertEqual(out, pixels(base))
    }

    func testOversizeImageIsRefusedBeforeRasterizing() {
        let base = solidImage(width: 100, height: 100)
        XCTAssertThrowsError(try AnnotationExporter.render(base, annotations: [], maxPixels: 5_000)) {
            XCTAssertEqual($0 as? AnnotationExporter.ExportError, .tooLarge)
        }
    }

    func testAnnotatedFileNameIsAlwaysPNG() {
        XCTAssertEqual(AnnotationExporter.annotatedFileName(from: "shot.jpeg"), "shot-annotated.png")
        XCTAssertEqual(AnnotationExporter.annotatedFileName(from: "a.b.png"), "a.b-annotated.png")
        XCTAssertEqual(AnnotationExporter.annotatedFileName(from: ""), "image-annotated.png")
    }

    func testSourceFileBytesAreUnchangedByAnnotatingAndExporting() throws {
        let base = solidImage(width: 120, height: 80)
        let source = try AnnotationExporter.flatten(base, annotations: [])
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("zp11-\(UUID().uuidString).png")
        try source.write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        let hashBefore = SHA256.hash(data: try Data(contentsOf: url))
        let modifiedBefore = try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date

        let image = try XCTUnwrap(NSImage(contentsOf: url))
        let cg = try XCTUnwrap(AnnotationExporter.sourceCGImage(of: image))
        _ = try AnnotationExporter.flatten(cg, annotations: [box(5, 5, 50, 30)])

        XCTAssertEqual(SHA256.hash(data: try Data(contentsOf: url)), hashBefore)
        XCTAssertEqual(
            try FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date,
            modifiedBefore)
    }

    func testSourcePixelSizeComesFromTheBitmapNotThePointSize() throws {
        // A 2x image: 200x100 pixels declared as 100x50 points.
        let base = solidImage(width: 200, height: 100)
        let rep = NSBitmapImageRep(cgImage: base)
        rep.size = NSSize(width: 100, height: 50)
        let image = NSImage(size: rep.size)
        image.addRepresentation(rep)
        let cg = try XCTUnwrap(AnnotationExporter.sourceCGImage(of: image))
        XCTAssertEqual(AnnotationExporter.pixelSize(of: cg), CGSize(width: 200, height: 100))
    }

    // MARK: - Overlay and exporter share one orientation

    func testOverlayDrawsInTheSameOrientationAsTheExporter() throws {
        var session = ImageAnnotationSession(imageSize: CGSize(width: 200, height: 100))
        session.viewport.fitScale = 1
        session.history.add(box(10, 10, 60, 30))
        let view = AnnotationLayerView(session: .constant(session))
            .environment(\.appTheme, .fallback)
        let renderer = ImageRenderer(content: view)
        renderer.scale = 1
        let cg = try XCTUnwrap(renderer.cgImage)
        let out = pixels(cg)
        let width = cg.width
        // Box's left edge near the TOP of the view, nothing near the bottom.
        XCTAssertTrue(isInk(pixel(out, width: width, x: 10, y: 25)), "box left edge at top")
        XCTAssertFalse(isInk(pixel(out, width: width, x: 10, y: 75)), "nothing mirrored at bottom")
    }
}
