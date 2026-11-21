import AppKit
import CoreGraphics
import Foundation

// The data model behind the image preview's annotation layer (ZP-11).
//
// Everything here is in IMAGE PIXELS with a top-left origin: stroke widths,
// font sizes, endpoints and hit tolerances. The view's zoom is a pure
// transform (`AnnotationViewport`) applied at draw and hit-test time, so a
// zoom never resamples or moves an annotation, and the exporter can replay the
// same list at natural size. Nothing here touches the source file.

/// The tools in the palette. `select` picks, moves and deletes. `ask` marks a
/// region and hands it to point-and-ask (`PointAndAsk`); the palette shows it
/// only when the loaded model takes images.
enum AnnotationTool: String, CaseIterable, Identifiable, Hashable {
    case select
    case box
    case arrow
    case text
    case ask

    var id: String { rawValue }
}

/// Stroke and type sizes derived from the image, so a 4000 px screenshot and a
/// 400 px icon both get marks that read at the size they are exported.
struct AnnotationStyle: Equatable {
    let strokeWidth: CGFloat
    let fontSize: CGFloat

    init(imageSize: CGSize) {
        let longest = max(imageSize.width, imageSize.height, 1)
        strokeWidth = max(2, (longest / 250).rounded())
        fontSize = max(14, (longest / 45).rounded())
    }

    var arrowHeadLength: CGFloat { strokeWidth * 5 }
    var textPadding: CGFloat { fontSize * 0.3 }
}

/// One mark on the image. Value type, so a history entry is a snapshot.
struct ImageAnnotation: Identifiable, Equatable {
    enum Kind: Equatable {
        case box(CGRect)
        case arrow(from: CGPoint, to: CGPoint)
        /// `origin` is the top-left of the label's box.
        case text(origin: CGPoint, string: String)
    }

    let id: UUID
    var kind: Kind

    init(id: UUID = UUID(), kind: Kind) {
        self.id = id
        self.kind = kind
    }

    /// Smallest drag worth keeping, in image pixels: a click that wobbles a
    /// pixel is not a box.
    static let minimumExtent: CGFloat = 4

    func translated(by delta: CGSize) -> ImageAnnotation {
        var copy = self
        switch kind {
        case .box(let rect):
            copy.kind = .box(rect.offsetBy(dx: delta.width, dy: delta.height))
        case .arrow(let from, let to):
            copy.kind = .arrow(
                from: CGPoint(x: from.x + delta.width, y: from.y + delta.height),
                to: CGPoint(x: to.x + delta.width, y: to.y + delta.height))
        case .text(let origin, let string):
            copy.kind = .text(
                origin: CGPoint(x: origin.x + delta.width, y: origin.y + delta.height),
                string: string)
        }
        return copy
    }

    /// Bounding box in image pixels, including the stroke.
    func bounds(style: AnnotationStyle) -> CGRect {
        switch kind {
        case .box(let rect):
            return rect.insetBy(dx: -style.strokeWidth / 2, dy: -style.strokeWidth / 2)
        case .arrow(let from, let to):
            let rect = CGRect(
                x: min(from.x, to.x), y: min(from.y, to.y),
                width: abs(from.x - to.x), height: abs(from.y - to.y))
            return rect.insetBy(dx: -style.arrowHeadLength, dy: -style.arrowHeadLength)
        case .text(let origin, let string):
            return AnnotationRenderer.textRect(origin: origin, string: string, style: style)
        }
    }

    /// Whether `point` (image pixels) lands on the mark. A box is hit on its
    /// outline only, so a large box never hides what sits inside it.
    func hits(_ point: CGPoint, tolerance: CGFloat, style: AnnotationStyle) -> Bool {
        let slack = tolerance + style.strokeWidth / 2
        switch kind {
        case .box(let rect):
            let outer = rect.insetBy(dx: -slack, dy: -slack)
            let inner = rect.insetBy(dx: slack, dy: slack)
            // `insetBy` past the center yields the null rect, which contains
            // nothing: a thin box is then all outline.
            return outer.contains(point) && !inner.contains(point)
        case .arrow(let from, let to):
            return Self.distance(from: point, toSegment: from, to) <= slack
        case .text:
            return bounds(style: style).insetBy(dx: -tolerance, dy: -tolerance).contains(point)
        }
    }

    static func distance(from p: CGPoint, toSegment a: CGPoint, _ b: CGPoint) -> CGFloat {
        let dx = b.x - a.x
        let dy = b.y - a.y
        let lengthSquared = dx * dx + dy * dy
        guard lengthSquared > 0 else { return hypot(p.x - a.x, p.y - a.y) }
        let t = max(0, min(1, ((p.x - a.x) * dx + (p.y - a.y) * dy) / lengthSquared))
        return hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy))
    }
}

/// The annotation list with undo and redo, owned by the open preview.
///
/// One history entry per COMPLETED gesture (a drawn box, a finished move, a
/// delete, a placed label). History covers annotations only: pan and zoom are
/// view state in `AnnotationViewport` and undo never touches them.
struct AnnotationHistory: Equatable {
    private(set) var annotations: [ImageAnnotation] = []
    private var undoStack: [[ImageAnnotation]] = []
    private var redoStack: [[ImageAnnotation]] = []

    /// Bounds memory on a long session; the oldest entries fall off.
    static let maxUndoDepth = 200

    var canUndo: Bool { !undoStack.isEmpty }
    var canRedo: Bool { !redoStack.isEmpty }

    mutating func add(_ annotation: ImageAnnotation) {
        record()
        annotations.append(annotation)
    }

    /// Replaces the annotation with the same id (a finished move). A missing id
    /// is a no-op so a stale gesture cannot resurrect a deleted mark.
    mutating func replace(_ annotation: ImageAnnotation) {
        guard let index = annotations.firstIndex(where: { $0.id == annotation.id }),
            annotations[index] != annotation
        else { return }
        record()
        annotations[index] = annotation
    }

    mutating func remove(id: UUID) {
        guard annotations.contains(where: { $0.id == id }) else { return }
        record()
        annotations.removeAll { $0.id == id }
    }

    @discardableResult
    mutating func undo() -> Bool {
        guard let previous = undoStack.popLast() else { return false }
        redoStack.append(annotations)
        annotations = previous
        return true
    }

    @discardableResult
    mutating func redo() -> Bool {
        guard let next = redoStack.popLast() else { return false }
        undoStack.append(annotations)
        annotations = next
        return true
    }

    private mutating func record() {
        undoStack.append(annotations)
        if undoStack.count > Self.maxUndoDepth { undoStack.removeFirst() }
        redoStack.removeAll()
    }
}

/// The pan and zoom transform between the view (points) and the image
/// (pixels). `scale` is points per image pixel; annotations never change, only
/// this does.
struct AnnotationViewport: Equatable {
    /// Source pixel dimensions.
    var imageSize: CGSize
    /// Points per pixel that fits the whole image in the available area.
    var fitScale: CGFloat = 1
    /// Multiplier over the fit scale.
    var zoom: CGFloat = 1

    static let zoomRange: ClosedRange<CGFloat> = 1...8
    static let zoomStep: CGFloat = 1.25

    var scale: CGFloat { max(fitScale * zoom, .leastNonzeroMagnitude) }
    var contentSize: CGSize {
        CGSize(width: imageSize.width * scale, height: imageSize.height * scale)
    }

    func imagePoint(fromView point: CGPoint) -> CGPoint {
        CGPoint(x: point.x / scale, y: point.y / scale)
    }

    func viewPoint(fromImage point: CGPoint) -> CGPoint {
        CGPoint(x: point.x * scale, y: point.y * scale)
    }

    mutating func setZoom(_ value: CGFloat) {
        zoom = min(max(value, Self.zoomRange.lowerBound), Self.zoomRange.upperBound)
    }

    mutating func zoomIn() { setZoom(zoom * Self.zoomStep) }
    mutating func zoomOut() { setZoom(zoom / Self.zoomStep) }
    mutating func resetZoom() { setZoom(1) }

    /// Sets the fit scale for a container, never upscaling past natural size
    /// on a 1x display (a small icon stays small instead of blurring).
    mutating func fit(into container: CGSize, maxScale: CGFloat) {
        guard imageSize.width > 0, imageSize.height > 0,
            container.width > 0, container.height > 0
        else { return }
        let fitted = min(container.width / imageSize.width, container.height / imageSize.height)
        fitScale = min(fitted, maxScale)
    }
}

/// Everything one open preview owns. Value type held in `@State`, so closing
/// the preview discards it (annotations are session-only by design).
struct ImageAnnotationSession: Equatable {
    var viewport: AnnotationViewport
    var history = AnnotationHistory()
    var tool: AnnotationTool = .select
    var selectedID: UUID?

    init(imageSize: CGSize) {
        viewport = AnnotationViewport(imageSize: imageSize)
    }

    mutating func undo() {
        history.undo()
        dropStaleSelection()
    }

    mutating func redo() {
        history.redo()
        dropStaleSelection()
    }

    mutating func deleteSelection() {
        guard let selectedID else { return }
        history.remove(id: selectedID)
        self.selectedID = nil
    }

    private mutating func dropStaleSelection() {
        if let selectedID, !history.annotations.contains(where: { $0.id == selectedID }) {
            self.selectedID = nil
        }
    }

    /// The topmost annotation under a view-space point, or nil.
    func annotation(atView point: CGPoint, slopPoints: CGFloat = 6) -> ImageAnnotation? {
        let style = AnnotationStyle(imageSize: viewport.imageSize)
        let imagePoint = viewport.imagePoint(fromView: point)
        let tolerance = slopPoints / viewport.scale
        return history.annotations.last { $0.hits(imagePoint, tolerance: tolerance, style: style) }
    }
}
