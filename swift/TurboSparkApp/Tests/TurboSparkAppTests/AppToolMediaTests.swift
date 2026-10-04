import CoreGraphics
import Foundation
import ImageIO
import TurboSpark
import UniformTypeIdentifiers
import XCTest
@testable import TurboSparkApp

enum AppToolMediaPNGFixture {
    private enum FixtureError: Error { case context, image, destination, encode }

    static func make(width: Int, height: Int, noisy: Bool) throws -> Data {
        guard let context = CGContext(
            data: nil,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: width * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let pixels = context.data?.assumingMemoryBound(to: UInt8.self)
        else {
            throw FixtureError.context
        }
        var state: UInt32 = 0x49a2_7bd1
        for offset in stride(from: 0, to: width * height * 4, by: 4) {
            if noisy {
                state = state &* 1_664_525 &+ 1_013_904_223
                pixels[offset] = UInt8(truncatingIfNeeded: state >> 8)
                state = state &* 1_664_525 &+ 1_013_904_223
                pixels[offset + 1] = UInt8(truncatingIfNeeded: state >> 8)
                state = state &* 1_664_525 &+ 1_013_904_223
                pixels[offset + 2] = UInt8(truncatingIfNeeded: state >> 8)
            } else {
                pixels[offset] = 42
                pixels[offset + 1] = 105
                pixels[offset + 2] = 180
            }
            pixels[offset + 3] = 255
        }
        guard let image = context.makeImage() else { throw FixtureError.image }
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData,
            UTType.png.identifier as CFString,
            1,
            nil)
        else {
            throw FixtureError.destination
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw FixtureError.encode }
        return data as Data
    }
}

final class AppToolMediaTests: XCTestCase {
    func testMediaCapabilityEnforcesApplicationCeilingAndProviderLimit() {
        let capped = AppToolMediaCapability(
            supportsImageBearingToolResults: true,
            maximumPixelCount: Int.max,
            maximumEncodedByteCount: Int.max)
        XCTAssertEqual(capped.maximumPixelCount, AppToolMediaPolicy.applicationMaximumPixelCount)
        XCTAssertEqual(capped.maximumEncodedByteCount, 8 * 1_024 * 1_024)

        let smaller = AppToolMediaCapability(
            supportsImageBearingToolResults: true,
            maximumPixelCount: 75_000,
            maximumEncodedByteCount: 73_000)
        XCTAssertEqual(smaller.maximumPixelCount, 75_000)
        XCTAssertEqual(smaller.maximumEncodedByteCount, 73_000)
    }

    func testMissingProviderLimitsUseSharedApplicationCeilings() {
        let capability = AppToolMediaCapability(supportsImageBearingToolResults: true)
        XCTAssertEqual(capability.maximumPixelCount, AppToolMediaPolicy.applicationMaximumPixelCount)
        XCTAssertEqual(
            capability.maximumEncodedByteCount,
            AppToolMediaPolicy.applicationMaximumEncodedByteCount)
    }

    func testScreenshotPolicyKeepsSmallPNGAndDownscalesToProviderLimit() throws {
        let smallPNG = try AppToolMediaPNGFixture.make(width: 160, height: 90, noisy: false)
        guard case .prepared(let kept) = AppToolMediaPolicy.prepareScreenshotPNG(
            smallPNG,
            capability: .imageBearingToolResults()) else {
            return XCTFail("A valid small PNG should be kept.")
        }
        XCTAssertEqual(kept.disposition, .kept)
        XCTAssertEqual(kept.pixelWidth, 160)
        XCTAssertEqual(kept.pixelHeight, 90)
        XCTAssertEqual(kept.data, smallPNG)

        let largePNG = try AppToolMediaPNGFixture.make(width: 512, height: 512, noisy: true)
        let byteLimit = 128 * 1_024
        XCTAssertGreaterThan(largePNG.count, byteLimit)
        guard case .prepared(let downscaled) = AppToolMediaPolicy.prepareScreenshotPNG(
            largePNG,
            capability: .imageBearingToolResults(maximumEncodedByteCount: byteLimit)) else {
            return XCTFail("A resizeable PNG should fit after downscaling.")
        }
        XCTAssertEqual(downscaled.disposition, .downscaled)
        XCTAssertLessThan(downscaled.pixelWidth, 512)
        XCTAssertLessThan(downscaled.pixelHeight, 512)
        XCTAssertLessThanOrEqual(downscaled.data.count, byteLimit)
    }

    func testScreenshotPolicyDownscalesToProviderPixelLimit() throws {
        let source = try AppToolMediaPNGFixture.make(width: 512, height: 512, noisy: false)
        guard case .prepared(let downscaled) = AppToolMediaPolicy.prepareScreenshotPNG(
            source,
            capability: .imageBearingToolResults(maximumPixelCount: 40_000)) else {
            return XCTFail("A resizeable PNG should fit the provider pixel limit.")
        }
        XCTAssertEqual(downscaled.disposition, .downscaled)
        XCTAssertLessThanOrEqual(downscaled.pixelWidth * downscaled.pixelHeight, 40_000)
        XCTAssertLessThanOrEqual(downscaled.data.count, AppToolMediaPolicy.applicationMaximumEncodedByteCount)
    }

    func testRefusalHasDispositionAndNoMediaReference() {
        let refused = AppToolMediaPolicy.storeScreenshot(
            pngData: nil,
            capability: .imageBearingToolResults())
        XCTAssertEqual(refused.disposition, .refused)
        XCTAssertNil(refused.reference)
    }

    func testScreenshotPolicyRefusesMissingInvalidAndUnsupportedImages() {
        XCTAssertEqual(
            AppToolMediaPolicy.prepareScreenshotPNG(
                nil,
                capability: .imageBearingToolResults()),
            .refused)
        XCTAssertEqual(
            AppToolMediaPolicy.prepareScreenshotPNG(
                Data("not a PNG".utf8),
                capability: .imageBearingToolResults()),
            .refused)
        XCTAssertEqual(
            AppToolMediaPolicy.prepareScreenshotPNG(
                Data([1, 2, 3]),
                capability: .textOnly),
            .refused)
    }

    func testHistoryAdapterOnlyMaterializesWithinProviderCapability() {
        let reference = AppToolMediaReference(
            assetReference: "turbospark-asset:" + String(repeating: "a", count: 64),
            mimeType: "image/png",
            byteCount: 4_096,
            pixelWidth: 320,
            pixelHeight: 200)
        var materializeCalls = 0
        let resolve: (String) throws -> URL? = { _ in
            materializeCalls += 1
            return URL(fileURLWithPath: "/tmp/browser-screenshot.png")
        }

        let supported = AppToolMediaHistoryAdapter.project(
            [reference], capability: .imageBearingToolResults(), materialize: resolve)
        XCTAssertEqual(supported.images, [.path("/tmp/browser-screenshot.png")])
        XCTAssertEqual(supported.omittedReferenceCount, 0)
        XCTAssertEqual(materializeCalls, 1)

        materializeCalls = 0
        let unsupported = AppToolMediaHistoryAdapter.project(
            [reference], capability: .textOnly, materialize: resolve)
        XCTAssertEqual(unsupported.images, [])
        XCTAssertEqual(unsupported.omittedReferenceCount, 1)
        XCTAssertEqual(materializeCalls, 0)

        let providerLimit = AppToolMediaHistoryAdapter.project(
            [reference],
            capability: .imageBearingToolResults(maximumEncodedByteCount: 1_024),
            materialize: resolve)
        XCTAssertEqual(providerLimit.images, [])
        XCTAssertEqual(providerLimit.omittedReferenceCount, 1)
        XCTAssertEqual(materializeCalls, 0)

        let providerPixelLimit = AppToolMediaHistoryAdapter.project(
            [reference],
            capability: .imageBearingToolResults(maximumPixelCount: 30_000),
            materialize: resolve)
        XCTAssertEqual(providerPixelLimit.images, [])
        XCTAssertEqual(providerPixelLimit.omittedReferenceCount, 1)
        XCTAssertEqual(materializeCalls, 0)

        let overSharedPixelLimit = AppToolMediaReference(
            assetReference: "turbospark-asset:" + String(repeating: "c", count: 64),
            mimeType: "image/png",
            byteCount: 4_096,
            pixelWidth: 5_000,
            pixelHeight: 5_000)
        let sharedPixelLimit = AppToolMediaHistoryAdapter.project(
            [overSharedPixelLimit], capability: .imageBearingToolResults(), materialize: resolve)
        XCTAssertEqual(sharedPixelLimit.images, [])
        XCTAssertEqual(sharedPixelLimit.omittedReferenceCount, 1)
        XCTAssertEqual(materializeCalls, 0)
    }

    func testToolResultDecodesOldArchiveAndRoundTripsMediaFields() throws {
        let callID = UUID()
        let oldArchive = """
        {"callID":"\(callID.uuidString)","output":"captured","isError":false,"durationSeconds":0.2}
        """
        let oldResult = try JSONDecoder().decode(AppToolResult.self, from: Data(oldArchive.utf8))
        XCTAssertNil(oldResult.mediaReferences)
        XCTAssertNil(oldResult.mediaDisposition)

        let reference = AppToolMediaReference(
            assetReference: "turbospark-asset:" + String(repeating: "b", count: 64),
            mimeType: "image/png",
            byteCount: 812,
            pixelWidth: 80,
            pixelHeight: 60)
        let result = AppToolResult(
            callID: callID,
            output: "captured",
            mediaReferences: [reference],
            mediaDisposition: .downscaled)
        let encoded = try JSONEncoder().encode(result)
        let decoded = try JSONDecoder().decode(AppToolResult.self, from: encoded)
        XCTAssertEqual(decoded.mediaReferences, [reference])
        XCTAssertEqual(decoded.mediaDisposition, .downscaled)
    }

    func testBrowserControlEncodingOmitsTransientScreenshotBytes() throws {
        let request = BrowserControlRequest(command: .screenshot(fullPage: false))
        let bytes = Data("private screenshot pixels".utf8)
        let result = BrowserControlResult(
            request: request,
            value: .screenshot(BrowserScreenshotResult(
                captureID: UUID(), width: 320, height: 200, fullPage: false)),
            durationMilliseconds: 10,
            screenshotPNGData: bytes)

        let encoded = try JSONEncoder().encode(result)
        XCTAssertNil(String(data: encoded, encoding: .utf8)?.range(of: "private screenshot pixels"))
        let decoded = try JSONDecoder().decode(BrowserControlResult.self, from: encoded)
        XCTAssertNil(decoded.screenshotPNGData)
    }

}
