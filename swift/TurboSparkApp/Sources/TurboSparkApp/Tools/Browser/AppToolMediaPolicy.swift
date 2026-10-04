import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Image-bearing tool-result support and the provider's resolved media limits.
public struct AppToolMediaCapability: Equatable, Sendable {
    public let supportsImageBearingToolResults: Bool
    public let maximumPixelCount: Int
    public let maximumEncodedByteCount: Int

    public init(
        supportsImageBearingToolResults: Bool,
        maximumPixelCount: Int? = nil,
        maximumEncodedByteCount: Int? = nil
    ) {
        self.supportsImageBearingToolResults = supportsImageBearingToolResults
        self.maximumPixelCount = Self.resolvedLimit(
            maximumPixelCount,
            applicationLimit: AppToolMediaPolicy.applicationMaximumPixelCount)
        self.maximumEncodedByteCount = min(
            max(1, maximumEncodedByteCount ?? AppToolMediaPolicy.applicationMaximumEncodedByteCount),
            AppToolMediaPolicy.applicationMaximumEncodedByteCount)
    }

    public static let textOnly = AppToolMediaCapability(supportsImageBearingToolResults: false)

    public static func imageBearingToolResults(
        maximumPixelCount: Int? = nil,
        maximumEncodedByteCount: Int? = nil
    ) -> AppToolMediaCapability {
        AppToolMediaCapability(
            supportsImageBearingToolResults: true,
            maximumPixelCount: maximumPixelCount,
            maximumEncodedByteCount: maximumEncodedByteCount)
    }

    private static func resolvedLimit(_ providerLimit: Int?, applicationLimit: Int) -> Int {
        guard let providerLimit else { return applicationLimit }
        return min(max(1, providerLimit), applicationLimit)
    }
}

struct PreparedAppToolImage: Equatable, Sendable {
    let data: Data
    let pixelWidth: Int
    let pixelHeight: Int
    let disposition: AppToolMediaDisposition
}

enum AppToolMediaPreparation: Equatable, Sendable {
    case prepared(PreparedAppToolImage)
    case refused
}

public struct AppToolMediaProcessingOutcome: Equatable, Sendable {
    public let reference: AppToolMediaReference?
    public let disposition: AppToolMediaDisposition

    public init(reference: AppToolMediaReference?, disposition: AppToolMediaDisposition) {
        self.reference = reference
        self.disposition = disposition
    }
}

/// Validates, downsizes, and stores screenshots before a tool result is persisted.
public enum AppToolMediaPolicy {
    public static let applicationMaximumPixelCount = 24_000_000
    public static let applicationMaximumEncodedByteCount = 8 * 1_024 * 1_024
    public static let applicationMaximumPixelDimension = 4_096

    static func prepareScreenshotPNG(
        _ data: Data?,
        capability: AppToolMediaCapability
    ) -> AppToolMediaPreparation {
        guard capability.supportsImageBearingToolResults,
              capability.maximumEncodedByteCount > 0,
              let data, !data.isEmpty,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let type = CGImageSourceGetType(source),
              type as String == UTType.png.identifier,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let widthValue = properties[kCGImagePropertyPixelWidth] as? NSNumber,
              let heightValue = properties[kCGImagePropertyPixelHeight] as? NSNumber
        else {
            return .refused
        }

        let width = widthValue.intValue
        let height = heightValue.intValue
        guard width > 0, height > 0 else { return .refused }
        let exceedsPixelLimit = exceedsPixelLimits(
            width: width,
            height: height,
            maximumPixelCount: capability.maximumPixelCount)
        if data.count <= capability.maximumEncodedByteCount, !exceedsPixelLimit {
            return .prepared(PreparedAppToolImage(
                data: data,
                pixelWidth: width,
                pixelHeight: height,
                disposition: .kept))
        }

        var maximumSide = min(max(width, height), applicationMaximumPixelDimension)
        for _ in 0..<32 {
            guard let thumbnail = createPNGThumbnail(source: source, maximumSide: maximumSide),
                  let encoded = encodePNG(thumbnail)
            else {
                return .refused
            }
            let exceedsPixelLimit = exceedsPixelLimits(
                width: thumbnail.width,
                height: thumbnail.height,
                maximumPixelCount: capability.maximumPixelCount)
            if !exceedsPixelLimit, encoded.count <= capability.maximumEncodedByteCount {
                return .prepared(PreparedAppToolImage(
                    data: encoded,
                    pixelWidth: thumbnail.width,
                    pixelHeight: thumbnail.height,
                    disposition: .downscaled))
            }
            guard maximumSide > 1 else { break }
            maximumSide = max(1, maximumSide * 4 / 5)
        }
        return .refused
    }

    public static func storeScreenshot(
        pngData: Data?,
        capability: AppToolMediaCapability,
        assetStore: ManagedAssetStore = .shared
    ) -> AppToolMediaProcessingOutcome {
        guard case .prepared(let image) = prepareScreenshotPNG(pngData, capability: capability) else {
            return AppToolMediaProcessingOutcome(reference: nil, disposition: .refused)
        }
        do {
            let asset = try assetStore.store(
                data: image.data,
                fileName: "browser-screenshot.png",
                mimeType: UTType.png.preferredMIMEType ?? "image/png")
            let reference = AppToolMediaReference(
                assetReference: asset.storedReference,
                mimeType: UTType.png.preferredMIMEType ?? "image/png",
                byteCount: asset.byteCount,
                pixelWidth: image.pixelWidth,
                pixelHeight: image.pixelHeight)
            return AppToolMediaProcessingOutcome(reference: reference, disposition: image.disposition)
        } catch {
            return AppToolMediaProcessingOutcome(reference: nil, disposition: .refused)
        }
    }

    private static func createPNGThumbnail(source: CGImageSource, maximumSide: Int) -> CGImage? {
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumSide,
            kCGImageSourceShouldCacheImmediately: true,
        ]
        return CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
    }

    private static func exceedsPixelLimits(width: Int, height: Int, maximumPixelCount: Int) -> Bool {
        guard width > 0, height > 0 else { return true }
        return width > applicationMaximumPixelDimension
            || height > applicationMaximumPixelDimension
            || width > maximumPixelCount / height
    }

    private static func encodePNG(_ image: CGImage) -> Data? {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data as CFMutableData,
            UTType.png.identifier as CFString,
            1,
            nil)
        else {
            return nil
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { return nil }
        return data as Data
    }
}
