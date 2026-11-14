import AppKit
import XCTest
@testable import TurboSparkApp

final class DevIconBadgerTests: XCTestCase {
    /// An opaque, solid red square: any other pixel colour came from the ribbon.
    private func solidIcon(_ side: Int, alpha: CGFloat = 1) -> NSImage {
        let rep = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: side, pixelsHigh: side,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
            colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: rep)
        NSColor(srgbRed: 1, green: 0, blue: 0, alpha: alpha).setFill()
        NSRect(x: 0, y: 0, width: side, height: side).fill()
        NSGraphicsContext.restoreGraphicsState()
        let image = NSImage(size: NSSize(width: side, height: side))
        image.addRepresentation(rep)
        return image
    }

    private func rep(_ image: NSImage) -> NSBitmapImageRep {
        NSBitmapImageRep(data: image.tiffRepresentation!)!
    }

    func testPolicyBadgesUnlessReleaseMarkerIsTrue() {
        XCTAssertTrue(DevIconBadger.shouldBadge(releaseMarker: nil))
        XCTAssertTrue(DevIconBadger.shouldBadge(releaseMarker: false))
        XCTAssertTrue(DevIconBadger.shouldBadge(releaseMarker: "true"))
        XCTAssertFalse(DevIconBadger.shouldBadge(releaseMarker: true))
    }

    func testBadgedKeepsDimensionsAndDrawsRibbon() throws {
        let source = solidIcon(128)
        let badged = try XCTUnwrap(DevIconBadger.badged(source))
        let out = rep(badged)
        XCTAssertEqual(out.pixelsWide, 128)
        XCTAssertEqual(out.pixelsHigh, 128)
        // Top-left stays the source colour; the ribbon centre (78%, 22% from
        // the bottom, so 78% down from the top edge) is blue.
        let corner = try XCTUnwrap(out.colorAt(x: 4, y: 4)?.usingColorSpace(.sRGB))
        XCTAssertGreaterThan(corner.redComponent, 0.9)
        let band = try XCTUnwrap(out.colorAt(x: 100, y: 100 - 8)?.usingColorSpace(.sRGB))
        XCTAssertGreaterThan(band.blueComponent, 0.5, "ribbon pixel expected")
        XCTAssertLessThan(band.redComponent, 0.5)
    }

    func testRibbonStaysInsideTransparentIconShape() throws {
        let badged = try XCTUnwrap(DevIconBadger.badged(solidIcon(128, alpha: 0)))
        let band = try XCTUnwrap(rep(badged).colorAt(x: 100, y: 92))
        XCTAssertEqual(band.alphaComponent, 0, accuracy: 0.01)
    }

    @MainActor
    func testFallbackKeepsOriginalWhenCompositingFails() {
        let empty = NSImage(size: .zero)
        XCTAssertNil(DevIconBadger.badged(empty))
        XCTAssertTrue(DevIconBadger.resolvedIcon(empty) === empty)
    }

    @MainActor
    func testReleaseBundleKeepsOriginalIcon() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ts_badge_\(UUID().uuidString)/Fake.app/Contents", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir.deletingLastPathComponent().deletingLastPathComponent()) }
        let plist: [String: Any] = ["CFBundleIdentifier": "test.badge", "TSReleasePackage": true]
        try (plist as NSDictionary).write(to: dir.appendingPathComponent("Info.plist"))
        let bundle = try XCTUnwrap(Bundle(url: dir.deletingLastPathComponent()))
        let source = solidIcon(64)
        XCTAssertTrue(DevIconBadger.resolvedIcon(source, bundle: bundle) === source)
        XCTAssertFalse(DevIconBadger.resolvedIcon(source, bundle: .main) === source)
    }
}
