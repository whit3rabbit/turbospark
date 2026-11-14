import AppKit
import PencilKit

// FEASIBILITY FINDING (ZP-10 spike, Xcode 27 / macOS 27 SDK, deployment 14):
// `PKCanvasView` and `PKToolPicker` do NOT exist for native AppKit: they are
// iOS and Mac Catalyst only (the macOS SDK's PencilKit headers stop at the
// data model, `PKDrawing`, `PKStroke`, `PKInk`, and `PKDrawing.image`). So the
// draft's "PKCanvasView wrapper" cannot be built. What survives is the part
// the draft cared about: `PKDrawing` is still the persisted format and its
// renderer still draws every stroke (smoothing and ink included). This file
// supplies only what PencilKit leaves out on macOS: pointer capture, a live
// preview, and a stroke eraser.

/// Turns pointer samples into a `PKStroke`, and erases strokes by position.
/// Pure geometry so tests need no window or events.
enum WhiteboardStrokes {
    /// Samples closer than this to the previous one are dropped: a mouse
    /// reports many near-identical points while held still.
    static let minimumSpacing: CGFloat = 0.75
    /// Eraser reach in points, added to half the stroke's own width.
    static let eraserRadius: CGFloat = 8

    /// A pen stroke through `points` (view coordinates), or nil for no input.
    /// A single click becomes a dot: PencilKit needs two control points to
    /// draw anything.
    static func stroke(
        through points: [CGPoint], color: NSColor, width: CGFloat, start: Date = Date()
    ) -> PKStroke? {
        var kept: [CGPoint] = []
        for point in points {
            if let last = kept.last, hypot(point.x - last.x, point.y - last.y) < minimumSpacing { continue }
            kept.append(point)
        }
        guard var first = kept.first else { return nil }
        if kept.count == 1 {
            first.x += 0.01
            kept.append(first)
        }
        let size = CGSize(width: width, height: width)
        let control = kept.enumerated().map { index, point in
            PKStrokePoint(
                location: point, timeOffset: TimeInterval(index) * 0.01, size: size,
                opacity: 1, force: 1, azimuth: 0, altitude: .pi / 2)
        }
        let path = PKStrokePath(controlPoints: control, creationDate: start)
        return PKStroke(ink: PKInk(.pen, color: color), path: path)
    }

    /// Whether the circle at `center` touches `stroke`.
    static func hits(_ stroke: PKStroke, at center: CGPoint, radius: CGFloat) -> Bool {
        let points = (0..<stroke.path.count).map { stroke.path[$0] }
        guard let first = points.first else { return false }
        let reach = radius + (first.size.width + first.size.height) / 4
        if points.count == 1 { return hypot(center.x - first.location.x, center.y - first.location.y) <= reach }
        for index in 1..<points.count
        where ImageAnnotation.distance(from: center, toSegment: points[index - 1].location, points[index].location) <= reach {
            return true
        }
        return false
    }

    /// `drawing` without the strokes touched by any of `centers`. Returns the
    /// same value when nothing was hit.
    static func erasing(_ drawing: PKDrawing, at centers: [CGPoint], radius: CGFloat = eraserRadius) -> PKDrawing {
        guard !centers.isEmpty else { return drawing }
        var result = drawing
        result.strokes = drawing.strokes.filter { stroke in
            !centers.contains { hits(stroke, at: $0, radius: radius) }
        }
        return result.strokes.count == drawing.strokes.count ? drawing : result
    }
}

/// The drawing surface. Top-left origin, one point is one drawing unit, so a
/// stroke keeps its place when the pane resizes (strokes past the edge are
/// kept, only not shown).
final class WhiteboardCanvasNSView: NSView {
    var drawing = PKDrawing() {
        didSet { cachedInk = nil; needsDisplay = true }
    }
    var tool: WhiteboardTool = .pen
    var inkColor: NSColor = .black
    var inkWidth: CGFloat = 4

    /// A gesture ended and changed the drawing (a stroke added, or strokes
    /// erased). Once per gesture, so history gets one step each.
    var onDrawingCommitted: (() -> Void)?

    private var livePoints: [CGPoint] = []
    private var gestureStart: PKDrawing?
    private var eraserPath: [CGPoint] = []
    private var cachedInk: NSImage?
    private var cachedBounds: CGRect = .zero

    override var isFlipped: Bool { true }
    override var acceptsFirstResponder: Bool { true }
    override func acceptsFirstMouse(for event: NSEvent?) -> Bool { true }
    override var isOpaque: Bool { true }

    // MARK: - Drawing

    override func draw(_ dirtyRect: NSRect) {
        WhiteboardPaper.color.setFill()
        bounds.fill()
        if !drawing.strokes.isEmpty {
            if cachedInk == nil || cachedBounds != bounds {
                cachedBounds = bounds
                // Light appearance so the ink resolves as it does in export.
                NSAppearance(named: .aqua)?.performAsCurrentDrawingAppearance {
                    cachedInk = drawing.image(from: bounds, scale: window?.backingScaleFactor ?? 2)
                }
            }
            cachedInk?.draw(
                in: bounds, from: .zero, operation: .sourceOver, fraction: 1,
                respectFlipped: true, hints: nil)
        }
        if tool == .pen, let first = livePoints.first {
            inkColor.setStroke()
            let path = NSBezierPath()
            path.lineWidth = inkWidth
            path.lineCapStyle = .round
            path.lineJoinStyle = .round
            path.move(to: first)
            for point in livePoints.dropFirst() { path.line(to: point) }
            if livePoints.count == 1 { path.line(to: CGPoint(x: first.x + 0.01, y: first.y)) }
            path.stroke()
        }
    }

    // MARK: - Pointer

    private func location(of event: NSEvent) -> CGPoint {
        convert(event.locationInWindow, from: nil)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(self)
        gestureStart = drawing
        let point = location(of: event)
        switch tool {
        case .pen:
            livePoints = [point]
            needsDisplay = true
        case .eraser:
            eraserPath = [point]
            drawing = WhiteboardStrokes.erasing(drawing, at: [point])
        }
    }

    override func mouseDragged(with event: NSEvent) {
        let point = location(of: event)
        switch tool {
        case .pen:
            livePoints.append(point)
            needsDisplay = true
        case .eraser:
            // Sample along the move so a fast drag cannot skip over a stroke.
            let last = eraserPath.last ?? point
            let distance = hypot(point.x - last.x, point.y - last.y)
            let steps = max(Int(distance / 2), 1)
            let samples = (1...steps).map { step -> CGPoint in
                let t = CGFloat(step) / CGFloat(steps)
                return CGPoint(x: last.x + (point.x - last.x) * t, y: last.y + (point.y - last.y) * t)
            }
            eraserPath.append(point)
            drawing = WhiteboardStrokes.erasing(drawing, at: samples)
        }
    }

    override func mouseUp(with event: NSEvent) {
        defer {
            livePoints = []
            eraserPath = []
            gestureStart = nil
            needsDisplay = true
        }
        switch tool {
        case .pen:
            livePoints.append(location(of: event))
            if let stroke = WhiteboardStrokes.stroke(through: livePoints, color: inkColor, width: inkWidth) {
                var next = drawing
                next.strokes.append(stroke)
                livePoints = []
                drawing = next
                onDrawingCommitted?()
            }
        case .eraser:
            if drawing.strokes.count != gestureStart?.strokes.count { onDrawingCommitted?() }
        }
    }
}
