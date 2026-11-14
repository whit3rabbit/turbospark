import Foundation
import ImageIO

/// An image already read, sniffed and bounded by the executor. The writers
/// take these instead of paths, so a writer never touches the filesystem.
struct DocumentImage: Equatable, Sendable {
    enum Kind: String, Sendable {
        case png, jpeg

        var fileExtension: String { self == .png ? "png" : "jpg" }
        var contentType: String { self == .png ? "image/png" : "image/jpeg" }
    }

    let data: Data
    let kind: Kind
    let pixelWidth: Int
    let pixelHeight: Int

    /// PNG and JPEG only, by magic bytes (never the extension). Other raster
    /// kinds and SVG (XML that can carry script) are refused.
    static func make(
        data: Data, limits: DocumentWriteLimits = .default
    ) throws -> DocumentImage {
        guard data.count <= limits.maximumImageBytes else {
            throw DocumentWriteError.overLimit("an image is over \(limits.maximumImageBytes / (1_024 * 1_024)) MiB")
        }
        let head = [UInt8](data.prefix(8))
        let kind: Kind
        if head.count >= 8, head == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
            kind = .png
        } else if head.count >= 3, Array(head[0..<3]) == [0xFF, 0xD8, 0xFF] {
            kind = .jpeg
        } else {
            throw DocumentWriteError.unsupported("images must be PNG or JPEG (checked by content, not extension)")
        }
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0 else {
            throw DocumentWriteError.invalidSpec("an image could not be decoded")
        }
        guard width * height <= limits.maximumImagePixels else {
            throw DocumentWriteError.overLimit("an image is \(width)x\(height) pixels (maximum \(limits.maximumImagePixels) pixels)")
        }
        return DocumentImage(data: data, kind: kind, pixelWidth: width, pixelHeight: height)
    }

    /// Display size in points: `requested` width (or the natural size at
    /// 96 dpi) clamped to `maximumWidth`, height from the aspect ratio.
    func displaySize(requestedWidth: Double?, maximumWidth: Double) -> (width: Double, height: Double) {
        let natural = Double(pixelWidth) * 0.75
        let width = min(max(requestedWidth ?? natural, 16), maximumWidth)
        return (width, width * Double(pixelHeight) / Double(pixelWidth))
    }
}
