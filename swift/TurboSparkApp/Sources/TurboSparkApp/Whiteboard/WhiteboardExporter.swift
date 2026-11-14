import AppKit
import CoreGraphics
import Foundation
import PencilKit

/// Renders a drawing to an opaque PNG.
///
/// "Natural size" is the drawing's own bounds plus a margin, at 2x: not the
/// size of the pane, so the picture does not depend on how wide the window
/// was. The background is opaque white (a transparent export reads as black
/// ink on black in a dark-mode viewer, and the model gets no alpha channel to
/// misread). The pixel count is capped by lowering the scale, never by
/// cropping.
enum WhiteboardExporter {
    static let margin: CGFloat = 24
    static let preferredScale: CGFloat = 2
    /// One RGBA bitmap at this size is 64 MB; a sketch needs nothing larger.
    static let maxPixelCount = 16_000_000

    enum ExportError: Error, LocalizedError, Equatable {
        case empty
        case renderFailed

        var errorDescription: String? {
            switch self {
            case .empty:
                return AppStrings.localized("The board is empty. Draw something before exporting.")
            case .renderFailed:
                return AppStrings.localized("The board could not be rendered to an image.")
            }
        }
    }

    /// The point rectangle exported for `drawing`: bounds plus margin, with the
    /// origin kept so strokes in negative space are not clipped.
    static func exportRect(for drawing: PKDrawing) -> CGRect {
        drawing.bounds.insetBy(dx: -margin, dy: -margin).integral
    }

    /// The scale actually used: `preferredScale`, lowered so the bitmap stays
    /// within `maxPixels`.
    static func scale(for rect: CGRect, maxPixels: Int = maxPixelCount) -> CGFloat {
        let points = max(rect.width * rect.height, 1)
        let limit = (CGFloat(maxPixels) / points).squareRoot()
        return min(preferredScale, limit)
    }

    static func pixelSize(for rect: CGRect, scale: CGFloat) -> (width: Int, height: Int) {
        (max(Int((rect.width * scale).rounded(.up)), 1), max(Int((rect.height * scale).rounded(.up)), 1))
    }

    static func render(_ drawing: PKDrawing, maxPixels: Int = maxPixelCount) throws -> CGImage {
        guard !drawing.strokes.isEmpty else { throw ExportError.empty }
        let rect = exportRect(for: drawing)
        let scale = scale(for: rect, maxPixels: maxPixels)
        let size = pixelSize(for: rect, scale: scale)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: size.width, height: size.height, bitsPerComponent: 8,
                bytesPerRow: 0, space: space,
                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ExportError.renderFailed }
        ctx.setFillColor(WhiteboardPaper.color.cgColor)
        ctx.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        // Rendered in the light appearance so any dynamic ink resolves to the
        // same colors the white paper showed on screen.
        var inkImage: CGImage?
        NSAppearance(named: .aqua)?.performAsCurrentDrawingAppearance {
            let image = drawing.image(from: rect, scale: scale)
            var proposed = CGRect(origin: .zero, size: image.size)
            inkImage = image.cgImage(forProposedRect: &proposed, context: nil, hints: nil)
        }
        guard let inkImage else { throw ExportError.renderFailed }
        ctx.interpolationQuality = .high
        ctx.draw(inkImage, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        guard let output = ctx.makeImage() else { throw ExportError.renderFailed }
        return output
    }

    static func png(for drawing: PKDrawing, maxPixels: Int = maxPixelCount) throws -> Data {
        let image = try render(drawing, maxPixels: maxPixels)
        do {
            return try AnnotationExporter.pngData(from: image)
        } catch {
            throw ExportError.renderFailed
        }
    }

    /// `Board 1` becomes `Board 1.png`; path separators cannot reach the
    /// asset store's file name.
    static func fileName(forBoardTitle title: String) -> String {
        let cleaned = title
            .replacingOccurrences(of: "/", with: "-")
            .replacingOccurrences(of: ":", with: "-")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        return "\(cleaned.isEmpty ? "whiteboard" : cleaned).png"
    }
}
