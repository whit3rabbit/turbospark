import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import XCTest

@testable import TurboSparkApp

/// The files are generated here (not committed fixtures) so each format and
/// orientation case is exactly what the test says it is.
final class EngineImageTranscoderTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("EngineImageTranscoderTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
    }

    /// A solid-colour image `width` x `height`, written as `type` with an
    /// optional EXIF orientation tag.
    private func write(_ type: UTType, name: String, width: Int, height: Int, orientation: Int? = nil)
        throws -> String
    {
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue))
        context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.8, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let image = try XCTUnwrap(context.makeImage())
        let url = directory.appendingPathComponent(name)
        let destination = try XCTUnwrap(
            CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil))
        var properties: [CFString: Any] = [:]
        if let orientation { properties[kCGImagePropertyOrientation] = orientation }
        CGImageDestinationAddImage(destination, image, properties as CFDictionary)
        XCTAssertTrue(CGImageDestinationFinalize(destination))
        return url.path
    }

    private func size(of path: String) throws -> (Int, Int) {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil))
        let p = try XCTUnwrap(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        return (p[kCGImagePropertyPixelWidth] as? Int ?? 0, p[kCGImagePropertyPixelHeight] as? Int ?? 0)
    }

    private func typeOf(_ path: String) -> String? {
        CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil)
            .flatMap { CGImageSourceGetType($0) as String? }
    }

    func testPNGIsPassedThroughUntouched() throws {
        let png = try write(.png, name: "a.png", width: 8, height: 4)
        XCTAssertEqual(EngineImageTranscoder.enginePath(for: png, cacheDirectory: directory), png)
    }

    func testPlainJPEGIsPassedThroughUntouched() throws {
        let jpg = try write(.jpeg, name: "a.jpg", width: 8, height: 4)
        XCTAssertEqual(EngineImageTranscoder.enginePath(for: jpg, cacheDirectory: directory), jpg)
    }

    func testUnsupportedFormatsBecomeAPNGOfTheSameSize() throws {
        for (type, name) in [(UTType.tiff, "a.tiff"), (.bmp, "a.bmp"), (.gif, "a.gif")] {
            let source = try write(type, name: name, width: 12, height: 6)
            let cache = directory.appendingPathComponent("cache-\(name)")
            let out = EngineImageTranscoder.enginePath(for: source, cacheDirectory: cache)
            XCTAssertNotEqual(out, source, name)
            XCTAssertEqual(typeOf(out), UTType.png.identifier, name)
            let (w, h) = try size(of: out)
            XCTAssertEqual([w, h], [12, 6], "\(name) must not be resized")
        }
    }

    func testOrientedJPEGIsRotatedIntoARealPNG() throws {
        // Orientation 6 = rotate 90 degrees; a 10x4 stored image displays 4x10.
        let jpg = try write(.jpeg, name: "phone.jpg", width: 10, height: 4, orientation: 6)
        let out = EngineImageTranscoder.enginePath(for: jpg, cacheDirectory: directory)
        XCTAssertNotEqual(out, jpg)
        XCTAssertEqual(typeOf(out), UTType.png.identifier)
        let (w, h) = try size(of: out)
        XCTAssertEqual([w, h], [4, 10], "the engine ignores EXIF, so the pixels must be turned")
    }

    func testResultIsCachedAndStable() throws {
        let source = try write(.tiff, name: "c.tiff", width: 6, height: 6)
        let first = EngineImageTranscoder.enginePath(for: source, cacheDirectory: directory)
        let before = try FileManager.default.attributesOfItem(atPath: first)[.modificationDate] as? Date
        let second = EngineImageTranscoder.enginePath(for: source, cacheDirectory: directory)
        XCTAssertEqual(first, second)
        let after = try FileManager.default.attributesOfItem(atPath: second)[.modificationDate] as? Date
        XCTAssertEqual(before, after, "the second call must reuse the cached file")
    }

    func testUnreadableAndRemotePathsAreReturnedAsIs() {
        XCTAssertEqual(EngineImageTranscoder.enginePath(for: ""), "")
        XCTAssertEqual(
            EngineImageTranscoder.enginePath(for: "https://example.com/a.webp"),
            "https://example.com/a.webp")
        let missing = directory.appendingPathComponent("nope.heic").path
        XCTAssertEqual(
            EngineImageTranscoder.enginePath(for: missing, cacheDirectory: directory), missing)
    }

    func testDecisionTable() {
        XCTAssertFalse(EngineImageTranscoder.needsTranscode(type: UTType.png.identifier, orientation: 6))
        XCTAssertFalse(EngineImageTranscoder.needsTranscode(type: UTType.jpeg.identifier, orientation: 1))
        XCTAssertTrue(EngineImageTranscoder.needsTranscode(type: UTType.jpeg.identifier, orientation: 3))
        XCTAssertTrue(EngineImageTranscoder.needsTranscode(type: UTType.heic.identifier, orientation: 1))
        XCTAssertTrue(EngineImageTranscoder.needsTranscode(type: UTType.webP.identifier, orientation: 1))
    }
}
