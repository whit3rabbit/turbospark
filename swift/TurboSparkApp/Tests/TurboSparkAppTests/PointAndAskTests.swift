import AppKit
import XCTest
@testable import TurboSparkApp

/// Point-and-ask: gesture to image-pixel region mapping, the drafted question,
/// and the vision gate on the AppModel handoff.
@MainActor
final class PointAndAskTests: XCTestCase {
    /// A 400 x 300 px image shown on a 2x display at natural size: one view
    /// point covers two image pixels (fit scale 0.5).
    private func retinaViewport(zoom: CGFloat = 1) -> AnnotationViewport {
        var viewport = AnnotationViewport(imageSize: CGSize(width: 400, height: 300))
        viewport.fitScale = 0.5
        viewport.setZoom(zoom)
        return viewport
    }

    // MARK: - Region mapping

    func testDragMapsViewPointsToImagePixelsAtRetinaScale() throws {
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 10, y: 20), to: CGPoint(x: 60, y: 80), viewport: retinaViewport()))
        XCTAssertEqual(rect, CGRect(x: 20, y: 40, width: 100, height: 120))
    }

    func testDragFollowsZoom() throws {
        // zoom 2 over fit 0.5 is one point per pixel.
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 10, y: 20), to: CGPoint(x: 60, y: 80),
            viewport: retinaViewport(zoom: 2)))
        XCTAssertEqual(rect, CGRect(x: 10, y: 20, width: 50, height: 60))
    }

    func testReversedDragIsNormalized() throws {
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 60, y: 80), to: CGPoint(x: 10, y: 20), viewport: retinaViewport()))
        XCTAssertEqual(rect, CGRect(x: 20, y: 40, width: 100, height: 120))
    }

    func testDragPastTheImageEdgeIsClamped() throws {
        // The layer can report points outside the image (a drag that leaves
        // it); the region never extends past the pixels that exist.
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: -30, y: -10), to: CGPoint(x: 500, y: 90), viewport: retinaViewport()))
        XCTAssertEqual(rect, CGRect(x: 0, y: 0, width: 400, height: 180))
    }

    func testClickBecomesABoxCenteredOnThePoint() throws {
        // 8% of the 300 px short side is 24 px, the minimum side as well.
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 100, y: 75), to: CGPoint(x: 100, y: 75), viewport: retinaViewport()))
        XCTAssertEqual(rect, CGRect(x: 188, y: 138, width: 24, height: 24))
    }

    func testClickWobbleUnderTheSlopIsStillAClick() throws {
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 100, y: 75), to: CGPoint(x: 101.5, y: 76), viewport: retinaViewport()))
        XCTAssertEqual(rect.width, 24)
        XCTAssertEqual(rect.height, 24)
    }

    func testClickBoxScalesWithLargeImages() throws {
        var viewport = AnnotationViewport(imageSize: CGSize(width: 4000, height: 3000))
        viewport.fitScale = 0.1
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 200, y: 150), to: CGPoint(x: 200, y: 150), viewport: viewport))
        XCTAssertEqual(rect, CGRect(x: 1880, y: 1380, width: 240, height: 240))
    }

    func testClickNearACornerIsShiftedInsideAtFullSize() throws {
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 0, y: 0), to: CGPoint(x: 0, y: 0), viewport: retinaViewport()))
        XCTAssertEqual(rect, CGRect(x: 0, y: 0, width: 24, height: 24))
        let far = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 200, y: 150), to: CGPoint(x: 200, y: 150), viewport: retinaViewport()))
        XCTAssertEqual(far, CGRect(x: 376, y: 276, width: 24, height: 24))
    }

    func testClickOnAnImageSmallerThanTheBoxCoversTheImage() throws {
        var viewport = AnnotationViewport(imageSize: CGSize(width: 16, height: 10))
        viewport.fitScale = 1
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 8, y: 5), to: CGPoint(x: 8, y: 5), viewport: viewport))
        XCTAssertEqual(rect, CGRect(x: 0, y: 0, width: 16, height: 10))
    }

    func testThinDragKeepsAVisibleHeight() throws {
        // A long horizontal drag one pixel tall is a deliberate region, not a
        // click, but a 1 px box would vanish under the stroke.
        let rect = try XCTUnwrap(PointAndAsk.region(
            fromView: CGPoint(x: 10, y: 50), to: CGPoint(x: 90, y: 50.5), viewport: retinaViewport()))
        XCTAssertEqual(rect.minX, 20)
        XCTAssertEqual(rect.width, 160)
        XCTAssertGreaterThanOrEqual(rect.height, ImageAnnotation.minimumExtent)
        XCTAssertTrue(CGRect(x: 0, y: 0, width: 400, height: 300).contains(rect))
    }

    func testUnknownImageSizeOrNonFinitePointsGiveNil() {
        var empty = AnnotationViewport(imageSize: .zero)
        empty.fitScale = 1
        XCTAssertNil(PointAndAsk.region(fromView: .zero, to: CGPoint(x: 5, y: 5), viewport: empty))
        XCTAssertNil(PointAndAsk.region(
            fromView: CGPoint(x: CGFloat.nan, y: 0), to: CGPoint(x: 5, y: 5), viewport: retinaViewport()))
    }

    // MARK: - Drafted question

    func testRegionQuestionNamesThePixelRangesAndImageSize() {
        let question = PointAndAsk.question(
            region: CGRect(x: 20, y: 40, width: 100, height: 120),
            imageSize: CGSize(width: 400, height: 300))
        XCTAssertTrue(question.contains("20-120"), question)
        XCTAssertTrue(question.contains("40-160"), question)
        XCTAssertTrue(question.contains("400x300"), question)
        XCTAssertFalse(question.contains("%"), "no unfilled specifier: \(question)")
    }

    func testDraftAppendsTheQuestionAfterExistingText() {
        XCTAssertEqual(PointAndAsk.draft("", appending: "Q?"), "Q?")
        XCTAssertEqual(PointAndAsk.draft("  \n", appending: "Q?"), "Q?")
        XCTAssertEqual(PointAndAsk.draft("Context  \n", appending: "Q?"), "Context\n\nQ?")
    }

    // MARK: - AppModel handoff

    func testAskWithoutAVisionModelAttachesNothingAndKeepsTheDraft() {
        let model = AppModel()
        model.writePromptTextDirectly("Draft")
        let attachmentsBefore = model.selectedChat.draftAttachments.count

        model.askAboutImage(Data([0x89, 0x50]), sourceName: "shot.png", question: "Q?")

        XCTAssertEqual(model.promptText, "Draft")
        XCTAssertEqual(model.selectedChat.draftAttachments.count, attachmentsBefore)
        XCTAssertEqual(model.activeToast?.style, .warning)
    }

    func testAskAttachesTheMarkedCopyAndDraftsWithoutSending() throws {
        let model = AppModel()
        model.writePromptTextDirectly("Draft")
        let messagesBefore = model.selectedChat.messages.count
        let png = try AnnotationExporter.flatten(solidImage(), annotations: [
            ImageAnnotation(kind: .box(CGRect(x: 2, y: 2, width: 8, height: 8)))
        ])

        model.askAboutImage(png, sourceName: "shot.jpeg", question: "Q?", visionActive: true)

        let attachment = try XCTUnwrap(model.selectedChat.draftAttachments.last)
        XCTAssertEqual(attachment.fileName, "shot-annotated.png")
        XCTAssertEqual(attachment.sourceByteSize, png.count)
        XCTAssertEqual(model.promptText, "Draft\n\nQ?")
        XCTAssertEqual(model.selectedChat.messages.count, messagesBefore, "never sends")
        XCTAssertFalse(model.isRunning)
    }

    private func solidImage() -> CGImage {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(
            data: nil, width: 16, height: 16, bitsPerComponent: 8, bytesPerRow: 0,
            space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        ctx.setFillColor(CGColor(srgbRed: 0.5, green: 0.5, blue: 0.5, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: 16, height: 16))
        return ctx.makeImage()!
    }
}
