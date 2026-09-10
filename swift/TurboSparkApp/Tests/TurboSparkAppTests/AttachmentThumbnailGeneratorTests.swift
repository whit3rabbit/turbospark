import AppKit
import XCTest

@testable import TurboSparkApp

/// The one case in this area that drives the REAL QuickLook generator.
///
/// `AttachmentPreviewTests` is pure values and cannot see any of this: whether
/// the generator answers at all, what it answers for a file that is not there,
/// or whether the cache is consulted. Each of those is a contract the view
/// leans on, and the second one was WRONG on the first run -- with
/// `representationTypes: .all` the generator never fails, substituting a
/// generic document icon for a missing file, so the "keep the themed symbol"
/// fallback could never fire.
final class AttachmentThumbnailGeneratorTests: XCTestCase {

    private func writePNG(to url: URL) throws {
        let image = NSImage(size: NSSize(width: 120, height: 80))
        image.lockFocus()
        NSColor.systemTeal.drawSwatch(in: NSRect(x: 0, y: 0, width: 120, height: 80))
        image.unlockFocus()
        let cg = try XCTUnwrap(image.cgImage(forProposedRect: nil, context: nil, hints: nil))
        let data = try XCTUnwrap(
            NSBitmapImageRep(cgImage: cg).representation(using: .png, properties: [:]))
        try data.write(to: url)
    }

    @MainActor
    func testTheGeneratorRendersARealImageAndCachesIt() async throws {
        let dir = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-thumb-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let png = dir.appendingPathComponent("swatch.png")
        try writePNG(to: png)

        let store = AttachmentThumbnailStore.shared
        let rendered = await store.thumbnail(for: png, pixelSize: 28)
        let first = try XCTUnwrap(rendered, "the generator produced nothing for a real PNG")

        // Aspect-fitted rather than squashed into the square it was asked for:
        // a 120x80 source must come back wider than it is tall.
        XCTAssertGreaterThan(first.size.width, first.size.height)
        XCTAssertLessThanOrEqual(first.size.width, 28)

        // The SAME object, not merely an equal one. A cache that re-renders is
        // not a cache, and nothing else in the suite can tell the difference.
        let second = await store.thumbnail(for: png, pixelSize: 28)
        XCTAssertTrue(first === second, "the second call should have hit the cache")
    }

    /// **nil has to mean nil.** The caller's fallback is a themed SF Symbol
    /// that tracks the accent color and the type scale, so a generic grey page
    /// glyph in its place is a regression rather than a graceful degradation.
    @MainActor
    func testAFileThatIsNotThereProducesNoThumbnail() async throws {
        let missing = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ts-absent-\(UUID().uuidString).png")
        let result = await AttachmentThumbnailStore.shared.thumbnail(
            for: missing, pixelSize: 28)
        XCTAssertNil(result)
    }
}
