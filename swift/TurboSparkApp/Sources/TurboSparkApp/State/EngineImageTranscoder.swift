import CryptoKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Makes an attachment readable by the engine's image decoder.
///
/// `crates/vision-io` enables the `jpeg` and `png` decoders only, and applies
/// no EXIF orientation. The attachment picker accepts more than that (gif,
/// heic, tiff, bmp, webp), and a phone JPEG is routinely stored sideways with
/// an orientation tag. Handing those to the engine as-is either fails the turn
/// with a decode error or sends the model a rotated picture, which it answers
/// fluently and wrongly.
///
/// So the engine is given a PNG for anything it cannot read faithfully. The
/// stored attachment is never modified; the PNG lives in a cache keyed by the
/// source's path, size and modification time.
enum EngineImageTranscoder {
    /// The cache directory under the profile's store root.
    static var defaultCacheDirectory: URL { AppStorageRoot.subdirectory("EngineImageCache") }

    /// The path to hand the engine for `path`: the same path when it is
    /// already faithful, a cached PNG when not, and the original path again if
    /// anything fails, so the engine reports its own (named) decode error
    /// rather than this layer inventing one.
    static func enginePath(for path: String, cacheDirectory: URL? = nil) -> String {
        // Remote URLs and empty (unresolvable) paths are not ours to rewrite.
        guard path.hasPrefix("/"),
            let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
            let type = CGImageSourceGetType(source) as String?
        else { return path }

        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let orientation = (properties?[kCGImagePropertyOrientation] as? Int) ?? 1
        guard needsTranscode(type: type, orientation: orientation) else { return path }

        let cache = cacheDirectory ?? defaultCacheDirectory
        let output = cache.appendingPathComponent(cacheName(for: path) + ".png")
        if FileManager.default.fileExists(atPath: output.path) { return output.path }

        let width = (properties?[kCGImagePropertyPixelWidth] as? Int) ?? 0
        let height = (properties?[kCGImagePropertyPixelHeight] as? Int) ?? 0
        // Never upscale: the cap is the source's own long edge.
        let longEdge = max(width, height) > 0 ? max(width, height) : 8192
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: longEdge,
        ]
        guard let image = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary)
        else { return path }

        do {
            try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
            // Written beside the destination and moved in, so a concurrent
            // reader never sees a half-written PNG.
            let staging = cache.appendingPathComponent(".\(UUID().uuidString).png")
            guard
                let destination = CGImageDestinationCreateWithURL(
                    staging as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { return path }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else {
                try? FileManager.default.removeItem(at: staging)
                return path
            }
            if FileManager.default.fileExists(atPath: output.path) {
                try? FileManager.default.removeItem(at: staging)
            } else {
                try FileManager.default.moveItem(at: staging, to: output)
            }
            return output.path
        } catch {
            return path
        }
    }

    /// PNG is read as-is (the engine's decoder ignores a PNG's rare
    /// orientation chunk; transcoding would not change that). JPEG is read
    /// as-is unless it carries a non-identity orientation. Everything else is
    /// converted.
    static func needsTranscode(type: String, orientation: Int) -> Bool {
        switch type {
        case UTType.png.identifier: return false
        case UTType.jpeg.identifier: return orientation > 1
        default: return true
        }
    }

    private static func cacheName(for path: String) -> String {
        let attributes = try? FileManager.default.attributesOfItem(atPath: path)
        let size = (attributes?[.size] as? NSNumber)?.int64Value ?? 0
        let modified = (attributes?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        let key = "\(path)|\(size)|\(modified)"
        return SHA256.hash(data: Data(key.utf8)).prefix(16).map { String(format: "%02x", $0) }
            .joined()
    }
}
