import AppKit
import ImageIO
import SwiftUI

/// The fixed "paper" an Office document renders on. A document carries its
/// own colors (run colors, cell fills, shape fills) chosen against a white
/// page, so drawing it on the app's themed surface would put dark text on a
/// dark background in dark mode. Same reasoning as `WhiteboardPaper`: the
/// page is fixed white, and only the viewer chrome follows the theme.
///
/// Lives in `Theme/` because the document colors are literals by nature (they
/// come from the file), and the design ratchet scans everything else for them.
enum OfficePaper {
    static let background = Color(.sRGB, red: 1, green: 1, blue: 1)
    static let ink = Color(.sRGB, red: 0.11, green: 0.11, blue: 0.12)
    static let gridLine = Color(.sRGB, red: 0.82, green: 0.82, blue: 0.84)
    static let headerFill = Color(.sRGB, red: 0.95, green: 0.95, blue: 0.96)
    static let noteMarker = Color(.sRGB, red: 0.86, green: 0.2, blue: 0.2)
    static let placeholderFill = Color(.sRGB, red: 0.93, green: 0.93, blue: 0.95)

    /// `RRGGBB` (already validated by `OfficeColor.hex`) to a color.
    static func color(hex: String?) -> Color? {
        guard let hex, hex.count == 6, let value = Int(hex, radix: 16) else { return nil }
        return Color(
            .sRGB,
            red: Double((value >> 16) & 0xFF) / 255,
            green: Double((value >> 8) & 0xFF) / 255,
            blue: Double(value & 0xFF) / 255)
    }

    /// Decodes untrusted image bytes with a ceiling on the DECODED size. A
    /// few hundred bytes of PNG can claim a 60000 by 60000 canvas, and a
    /// plain image load would try to allocate it. The thumbnail path
    /// downsamples without a full-size decode, and a claim over the pixel
    /// budget is refused before any decode.
    static func decodeImage(_ data: Data, maximumPixels: Int = 40_000_000, maximumSide: Int = 2_048) -> NSImage? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              CGImageSourceGetCount(source) > 0,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              let width = properties[kCGImagePropertyPixelWidth] as? Int,
              let height = properties[kCGImagePropertyPixelHeight] as? Int,
              width > 0, height > 0, width <= 100_000, height <= 100_000,
              width * height <= maximumPixels else { return nil }
        let options: [CFString: Any] = [
            kCGImageSourceCreateThumbnailFromImageAlways: true,
            kCGImageSourceCreateThumbnailWithTransform: true,
            kCGImageSourceThumbnailMaxPixelSize: maximumSide,
        ]
        guard let cgImage = CGImageSourceCreateThumbnailAtIndex(source, 0, options as CFDictionary) else {
            return nil
        }
        return NSImage(cgImage: cgImage, size: NSSize(width: cgImage.width, height: cgImage.height))
    }
}
