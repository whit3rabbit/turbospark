import CoreGraphics
import Foundation

/// Point-and-ask on the image preview: the user clicks a spot or drags a
/// region, the app burns a box onto a COPY of the image (the same
/// `AnnotationExporter` path as Send to chat), attaches it, and drafts a
/// question in the composer. Nothing is sent.
///
/// The region is in image pixels with a top-left origin, like every other
/// annotation. The annotation layer is laid out at the image's exact on-screen
/// rect (no letterbox margin inside it), so the only view-to-pixel step is the
/// viewport's scale, which already folds in the display's backing scale.
enum PointAndAsk {
    /// A click (no real drag) marks a square this fraction of the shorter
    /// image side, so the box reads at export size on any resolution.
    static let clickBoxFraction: CGFloat = 0.08
    /// Smallest side of a click box, in image pixels.
    static let minimumClickBoxSide: CGFloat = 24
    /// Pointer travel under this many VIEW points is a click, not a drag: the
    /// threshold has to be in points, since at a small fit scale a few pixels
    /// of image are less than one point of hand wobble.
    static let clickSlopPoints: CGFloat = 4

    /// The image-pixel region for a gesture reported in the annotation
    /// layer's view points. Nil when the image size or a point is unusable.
    static func region(
        fromView start: CGPoint, to end: CGPoint, viewport: AnnotationViewport
    ) -> CGRect? {
        region(
            fromImage: viewport.imagePoint(fromView: start),
            to: viewport.imagePoint(fromView: end),
            imageSize: viewport.imageSize,
            clickTolerance: clickSlopPoints / viewport.scale)
    }

    /// The region for two image-pixel points: normalized, clamped to the
    /// image, a click widened to a visible box, rounded out to whole pixels.
    static func region(
        fromImage a: CGPoint, to b: CGPoint, imageSize: CGSize, clickTolerance: CGFloat
    ) -> CGRect? {
        guard imageSize.width.isFinite, imageSize.height.isFinite,
            imageSize.width >= 1, imageSize.height >= 1,
            a.x.isFinite, a.y.isFinite, b.x.isFinite, b.y.isFinite
        else { return nil }
        let p = clamp(a, to: imageSize)
        let q = clamp(b, to: imageSize)
        let travel = max(abs(p.x - q.x), abs(p.y - q.y))
        if travel < clickTolerance {
            let side = max(minimumClickBoxSide, min(imageSize.width, imageSize.height) * clickBoxFraction)
            return fitted(
                CGRect(x: p.x, y: p.y, width: 0, height: 0),
                minWidth: side, minHeight: side, imageSize: imageSize)
        }
        let rect = CGRect(
            x: min(p.x, q.x), y: min(p.y, q.y), width: abs(p.x - q.x), height: abs(p.y - q.y))
        return fitted(
            rect, minWidth: ImageAnnotation.minimumExtent, minHeight: ImageAnnotation.minimumExtent,
            imageSize: imageSize)
    }

    static func clamp(_ point: CGPoint, to size: CGSize) -> CGPoint {
        CGPoint(x: min(max(point.x, 0), size.width), y: min(max(point.y, 0), size.height))
    }

    /// Grows `rect` about its center to the minimum sides (capped at the
    /// image), then shifts it inside the image rather than cropping it, so a
    /// click at an edge still marks a full-size box.
    private static func fitted(
        _ rect: CGRect, minWidth: CGFloat, minHeight: CGFloat, imageSize: CGSize
    ) -> CGRect {
        let width = min(max(rect.width, minWidth), imageSize.width)
        let height = min(max(rect.height, minHeight), imageSize.height)
        var x = rect.midX - width / 2
        var y = rect.midY - height / 2
        x = min(max(x, 0), imageSize.width - width)
        y = min(max(y, 0), imageSize.height - height)
        let integral = CGRect(x: x, y: y, width: width, height: height).integral
        return integral.intersection(CGRect(origin: .zero, size: imageSize))
    }

    // MARK: - Drafted question

    /// The question drafted for a marked region. It is a seeded composer
    /// draft the user edits before sending, so it is localized (DESIGN.md,
    /// section 6). The pixel ranges help a model whose vision preprocessing
    /// resizes the image; the red box is what it actually looks at.
    static func question(region: CGRect, imageSize: CGSize) -> String {
        let xRange = "\(Int(region.minX))-\(Int(region.maxX))"
        let yRange = "\(Int(region.minY))-\(Int(region.maxY))"
        let size = "\(Int(imageSize.width))x\(Int(imageSize.height))"
        return AppStrings.localized(
            "In the attached image, what is inside the red box at x \(xRange), y \(yRange) (image size \(size))?")
    }

    /// The non-pointer alternative: a question about the whole image.
    static func wholeImageQuestion() -> String {
        AppStrings.localized("What is in the attached image?")
    }

    /// The draft with the question after any text already there.
    static func draft(_ draft: String, appending question: String) -> String {
        var head = draft
        while let last = head.last, last.isWhitespace { head.removeLast() }
        return head.isEmpty ? question : head + "\n\n" + question
    }
}

extension AppModel {
    /// Attaches a marked copy of an image to the focused chat's draft and
    /// appends `question` to the composer. Never sends: the user edits and
    /// presses Send.
    ///
    /// The preview offers Ask only while `visionIsActive`, but a model can
    /// unload between the render and the press, so this re-checks. Asking a
    /// text-only model about a picture would draft a question it refuses.
    func askAboutImage(_ png: Data, sourceName: String, question: String) {
        askAboutImage(png, sourceName: sourceName, question: question, visionActive: visionIsActive)
    }

    /// Seam for tests: no vision session can be loaded in a unit test.
    func askAboutImage(_ png: Data, sourceName: String, question: String, visionActive: Bool) {
        guard visionActive else {
            let reason = visionRefusalReason
                ?? AppStrings.localized("the loaded model has no active vision tower")
            showToast(AppStrings.localized("Cannot send this image: \(reason)"), style: .warning)
            return
        }
        let name = AnnotationExporter.annotatedFileName(from: sourceName)
        let attached = attachPNGToPrompt(png, fileName: name) { detail in
            AppStrings.localized("Could not attach the annotated image: \(detail)")
        }
        guard attached else { return }
        // Programmatic write, so the drafted text is not turned into a paste
        // attachment chip.
        writePromptTextDirectly(PointAndAsk.draft(promptText, appending: question))
        NotificationCenter.default.post(name: .focusPrompt, object: nil)
        showToast(
            AppStrings.localized("Image attached. Edit the question in the composer, then send."),
            style: .success)
    }
}
