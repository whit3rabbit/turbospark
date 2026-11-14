import AppKit
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// The ONE draw path for annotations. The live overlay replays it at view
/// resolution (after scaling the context by the viewport) and the exporter
/// replays it at natural size, so what the user sees is what the model gets.
///
/// Contract: `ctx` has a TOP-LEFT origin (y grows downward) and one unit is
/// one image pixel.
enum AnnotationRenderer {
    /// Fixed high-visibility ink. It is burned into the user's pixels, so it
    /// must read on arbitrary images and cannot follow the app theme.
    static let ink = CGColor(srgbRed: 0.90, green: 0.12, blue: 0.14, alpha: 1)
    private static let labelBackground = CGColor(srgbRed: 1, green: 1, blue: 1, alpha: 0.88)

    static func font(for style: AnnotationStyle) -> NSFont {
        NSFont.systemFont(ofSize: style.fontSize, weight: .semibold)
    }

    private static func attributes(style: AnnotationStyle) -> [NSAttributedString.Key: Any] {
        [.font: font(for: style), .foregroundColor: NSColor(cgColor: ink) ?? NSColor.red]
    }

    /// The label's rectangle: text size plus padding, anchored at `origin`.
    static func textRect(origin: CGPoint, string: String, style: AnnotationStyle) -> CGRect {
        let size = (string as NSString).size(withAttributes: attributes(style: style))
        let pad = style.textPadding
        return CGRect(
            x: origin.x, y: origin.y,
            width: ceil(size.width) + pad * 2, height: ceil(size.height) + pad * 2)
    }

    /// Draws one annotation. `selection`, when set, adds the dashed selection
    /// ring in that color (overlay only; the exporter never passes it).
    static func draw(
        _ annotation: ImageAnnotation,
        in ctx: CGContext,
        style: AnnotationStyle,
        selection: CGColor? = nil
    ) {
        ctx.saveGState()
        defer { ctx.restoreGState() }
        ctx.setStrokeColor(ink)
        ctx.setFillColor(ink)
        ctx.setLineWidth(style.strokeWidth)
        ctx.setLineCap(.round)
        ctx.setLineJoin(.round)

        switch annotation.kind {
        case .box(let rect):
            ctx.stroke(rect)
        case .arrow(let from, let to):
            let angle = atan2(to.y - from.y, to.x - from.x)
            let head = style.arrowHeadLength
            let spread = CGFloat.pi / 7
            // The shaft stops at the head's base so a thick stroke does not
            // poke past the tip.
            let base = CGPoint(
                x: to.x - cos(angle) * head * 0.6, y: to.y - sin(angle) * head * 0.6)
            ctx.move(to: from)
            ctx.addLine(to: base)
            ctx.strokePath()
            ctx.move(to: to)
            ctx.addLine(to: CGPoint(
                x: to.x - cos(angle - spread) * head, y: to.y - sin(angle - spread) * head))
            ctx.addLine(to: CGPoint(
                x: to.x - cos(angle + spread) * head, y: to.y - sin(angle + spread) * head))
            ctx.closePath()
            ctx.fillPath()
        case .text(let origin, let string):
            let rect = textRect(origin: origin, string: string, style: style)
            ctx.setFillColor(labelBackground)
            ctx.addPath(CGPath(
                roundedRect: rect, cornerWidth: style.textPadding, cornerHeight: style.textPadding,
                transform: nil))
            ctx.fillPath()
            // AppKit text needs a graphics context; `flipped: true` matches
            // the top-left contract above so glyphs come out upright.
            let graphics = NSGraphicsContext(cgContext: ctx, flipped: true)
            NSGraphicsContext.saveGraphicsState()
            NSGraphicsContext.current = graphics
            (string as NSString).draw(
                at: CGPoint(x: rect.minX + style.textPadding, y: rect.minY + style.textPadding),
                withAttributes: attributes(style: style))
            NSGraphicsContext.restoreGraphicsState()
        }

        if let selection {
            let ring = annotation.bounds(style: style).insetBy(
                dx: -style.strokeWidth, dy: -style.strokeWidth)
            ctx.setStrokeColor(selection)
            ctx.setLineWidth(max(1, style.strokeWidth * 0.6))
            let dash = style.strokeWidth * 3
            ctx.setLineDash(phase: 0, lengths: [dash, dash])
            ctx.stroke(ring)
        }
    }
}

/// Flattens a base image plus annotations into a natural-size PNG.
///
/// Reads the source image only; it never opens a file for writing, so the
/// original bytes cannot change.
enum AnnotationExporter {
    /// Largest image the exporter will rasterize (width * height). A single
    /// RGBA bitmap at this size is 256 MB, so a bigger source is refused with
    /// a message rather than risking the process.
    static let maxPixelCount = 64_000_000

    enum ExportError: Error, LocalizedError, Equatable {
        case noBitmap
        case tooLarge
        case encodeFailed

        var errorDescription: String? {
            switch self {
            case .noBitmap:
                return AppStrings.localized("The image could not be read as pixels.")
            case .tooLarge:
                return AppStrings.localized("The image is too large to export with annotations.")
            case .encodeFailed:
                return AppStrings.localized("The annotated image could not be encoded.")
            }
        }
    }

    /// The source's pixel grid. `NSImage.size` is in points and is half the
    /// pixel size on a 2x screenshot, so the largest bitmap rep decides.
    static func sourceCGImage(of image: NSImage) -> CGImage? {
        let reps = image.representations.compactMap { $0 as? NSBitmapImageRep }
        if let best = reps.max(by: { $0.pixelsWide * $0.pixelsHigh < $1.pixelsWide * $1.pixelsHigh }),
            let cg = best.cgImage
        {
            return cg
        }
        var rect = CGRect(origin: .zero, size: image.size)
        return image.cgImage(forProposedRect: &rect, context: nil, hints: nil)
    }

    static func pixelSize(of cgImage: CGImage) -> CGSize {
        CGSize(width: cgImage.width, height: cgImage.height)
    }

    /// Draws the base image and every annotation into an RGBA bitmap at the
    /// source's pixel size.
    static func render(
        _ base: CGImage, annotations: [ImageAnnotation], maxPixels: Int = AnnotationExporter.maxPixelCount
    ) throws -> CGImage {
        let width = base.width
        let height = base.height
        guard width > 0, height > 0 else { throw ExportError.noBitmap }
        guard width * height <= maxPixels else { throw ExportError.tooLarge }
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
            let ctx = CGContext(
                data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
                space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw ExportError.noBitmap }
        ctx.interpolationQuality = .high
        ctx.draw(base, in: CGRect(x: 0, y: 0, width: width, height: height))
        // Flip to the renderer's top-left contract after the (bottom-left)
        // base draw so the picture is not upside down.
        ctx.translateBy(x: 0, y: CGFloat(height))
        ctx.scaleBy(x: 1, y: -1)
        let style = AnnotationStyle(imageSize: CGSize(width: width, height: height))
        for annotation in annotations {
            AnnotationRenderer.draw(annotation, in: ctx, style: style)
        }
        guard let output = ctx.makeImage() else { throw ExportError.noBitmap }
        return output
    }

    /// The flattened image as PNG data.
    static func flatten(_ base: CGImage, annotations: [ImageAnnotation]) throws -> Data {
        try pngData(from: render(base, annotations: annotations))
    }

    /// PNG-encodes a finished bitmap. Shared with the whiteboard export
    /// (ZP-10) so both produce PNGs the same way.
    static func pngData(from image: CGImage) throws -> Data {
        let data = NSMutableData()
        guard let destination = CGImageDestinationCreateWithData(
            data, UTType.png.identifier as CFString, 1, nil)
        else { throw ExportError.encodeFailed }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw ExportError.encodeFailed }
        return data as Data
    }

    /// `shot.jpeg` becomes `shot-annotated.png`: the export is always PNG.
    static func annotatedFileName(from fileName: String) -> String {
        let stem = (fileName as NSString).deletingPathExtension
        let base = stem.isEmpty ? "image" : stem
        return "\(base)-annotated.png"
    }
}

extension AppModel {
    /// Stores a PNG in the managed asset store and appends it to the focused
    /// chat's draft as an image attachment (by asset reference, empty
    /// extracted text), the same way an imported picture is carried. Returns
    /// whether it attached; a failure has already toasted.
    @discardableResult
    func attachPNGToPrompt(
        _ png: Data, fileName: String, failureMessage: (String) -> String
    ) -> Bool {
        // `addPromptAttachment` refuses mid-turn without throwing, so check
        // first rather than toast a success that did not happen.
        guard !generating, !submitting else {
            showToast(AppStrings.localized("Cannot attach files while generating."), style: .warning)
            return false
        }
        do {
            let asset = try ManagedAssetStore.shared.store(
                data: png, fileName: fileName, mimeType: "image/png")
            let attachment = AppPromptAttachment(
                fileName: fileName,
                formatLabel: "Image",
                extractedText: "",
                wasTruncatedDuringExtraction: false,
                sourcePath: asset.storedReference,
                sourceByteSize: Int(asset.byteCount))
            addPromptAttachment(attachment, toChatID: nil)
            return true
        } catch {
            showToast(failureMessage(error.localizedDescription), style: .error)
            return false
        }
    }

    /// Attaches a flattened annotation export to the prompt.
    func attachAnnotatedImage(_ png: Data, sourceName: String) {
        let name = AnnotationExporter.annotatedFileName(from: sourceName)
        let attached = attachPNGToPrompt(png, fileName: name) { detail in
            AppStrings.localized("Could not attach the annotated image: \(detail)")
        }
        if attached {
            showToast(AppStrings.localized("Annotated image attached to the prompt."), style: .success)
        }
    }
}
