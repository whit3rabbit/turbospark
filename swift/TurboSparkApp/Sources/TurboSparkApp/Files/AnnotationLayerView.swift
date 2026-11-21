import AppKit
import SwiftUI

/// The annotation overlay drawn over the image (ZP-11).
///
/// Sized to the image's on-screen rect. Every gesture point is converted to
/// image pixels through `AnnotationViewport` before it touches the model, and
/// drawing replays `AnnotationRenderer` under the same transform, so marks
/// track the image at every zoom level.
@MainActor
struct AnnotationLayerView: View {
    @Binding var session: ImageAnnotationSession
    /// Receives the image-pixel region of a finished `.ask` gesture. The
    /// layer only reports it; the preview decides what to attach.
    var onAskRegion: ((CGRect) -> Void)?
    @Environment(\.appTheme) private var theme

    /// A box or arrow being dragged out; committed on release.
    @State private var draft: ImageAnnotation?
    /// The annotation being moved, and where it sits mid-drag.
    @State private var movingOriginal: ImageAnnotation?
    @State private var movingDraft: ImageAnnotation?
    @State private var gestureBegan = false
    /// Image-pixel anchor of a label awaiting its text.
    @State private var pendingTextAnchor: CGPoint?
    @State private var pendingText = ""
    @FocusState private var textFocused: Bool

    private var viewport: AnnotationViewport { session.viewport }
    private var style: AnnotationStyle { AnnotationStyle(imageSize: viewport.imageSize) }

    var body: some View {
        ZStack(alignment: .topLeading) {
            canvas
            Color.clear
                .contentShape(Rectangle())
                .gesture(dragGesture)
            if let anchor = pendingTextAnchor {
                textEntry(at: viewport.viewPoint(fromImage: anchor))
            }
        }
        .frame(width: viewport.contentSize.width, height: viewport.contentSize.height)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Annotations", bundle: .module))
        .accessibilityChildren {
            ForEach(session.history.annotations) { annotation in
                Button {
                    session.selectedID = annotation.id
                } label: {
                    Text(Self.spokenLabel(annotation))
                }
                .accessibilityAddTraits(session.selectedID == annotation.id ? .isSelected : [])
            }
        }
    }

    // MARK: - Drawing

    private var canvas: some View {
        Canvas { context, _ in
            let scale = viewport.scale
            let selection = NSColor(theme.accent).cgColor
            context.withCGContext { cg in
                cg.scaleBy(x: scale, y: scale)
                for annotation in visibleAnnotations {
                    AnnotationRenderer.draw(
                        annotation, in: cg, style: style,
                        selection: annotation.id == session.selectedID ? selection : nil)
                }
                if let draft {
                    AnnotationRenderer.draw(draft, in: cg, style: style)
                }
            }
        }
        .allowsHitTesting(false)
    }

    /// The history, with the annotation mid-move shown at its dragged place.
    private var visibleAnnotations: [ImageAnnotation] {
        session.history.annotations.map { annotation in
            if let movingDraft, movingDraft.id == annotation.id { return movingDraft }
            return annotation
        }
    }

    // MARK: - Gestures

    private var dragGesture: some Gesture {
        DragGesture(minimumDistance: 0, coordinateSpace: .local)
            .onChanged { value in
                if !gestureBegan {
                    gestureBegan = true
                    began(at: value.startLocation)
                }
                changed(start: value.startLocation, current: value.location)
            }
            .onEnded { value in
                ended(start: value.startLocation, end: value.location)
                gestureBegan = false
            }
    }

    private func began(at point: CGPoint) {
        guard pendingTextAnchor == nil else { return }
        if session.tool == .select {
            let hit = session.annotation(atView: point)
            session.selectedID = hit?.id
            movingOriginal = hit
        }
    }

    private func changed(start: CGPoint, current: CGPoint) {
        guard pendingTextAnchor == nil else { return }
        let a = clamped(viewport.imagePoint(fromView: start))
        let b = clamped(viewport.imagePoint(fromView: current))
        switch session.tool {
        case .select:
            guard let original = movingOriginal else { return }
            movingDraft = original.translated(by: CGSize(width: b.x - a.x, height: b.y - a.y))
        case .box:
            draft = ImageAnnotation(kind: .box(Self.rect(a, b)))
        case .ask:
            // Preview exactly the box that will be burned in, through the
            // same function, so a press already shows the widened click box.
            draft = PointAndAsk.region(fromView: start, to: current, viewport: viewport)
                .map { ImageAnnotation(kind: .box($0)) }
        case .arrow:
            draft = ImageAnnotation(kind: .arrow(from: a, to: b))
        case .text:
            break
        }
    }

    private func ended(start: CGPoint, end: CGPoint) {
        defer {
            draft = nil
            movingOriginal = nil
            movingDraft = nil
        }
        guard pendingTextAnchor == nil else { return }
        switch session.tool {
        case .select:
            if let movingDraft { session.history.replace(movingDraft) }
        case .box, .arrow:
            guard let draft, Self.extent(of: draft) >= ImageAnnotation.minimumExtent else { return }
            session.history.add(draft)
            session.selectedID = draft.id
        case .ask:
            guard let onAskRegion,
                let region = PointAndAsk.region(fromView: start, to: end, viewport: viewport)
            else { return }
            onAskRegion(region)
        case .text:
            pendingText = ""
            pendingTextAnchor = clamped(viewport.imagePoint(fromView: start))
            textFocused = true
        }
    }

    private func clamped(_ point: CGPoint) -> CGPoint {
        CGPoint(
            x: min(max(point.x, 0), viewport.imageSize.width),
            y: min(max(point.y, 0), viewport.imageSize.height))
    }

    static func rect(_ a: CGPoint, _ b: CGPoint) -> CGRect {
        CGRect(x: min(a.x, b.x), y: min(a.y, b.y), width: abs(a.x - b.x), height: abs(a.y - b.y))
    }

    static func extent(of annotation: ImageAnnotation) -> CGFloat {
        switch annotation.kind {
        case .box(let rect): return min(rect.width, rect.height)
        case .arrow(let from, let to): return hypot(to.x - from.x, to.y - from.y)
        case .text: return .infinity
        }
    }

    // MARK: - Text entry

    private func textEntry(at point: CGPoint) -> some View {
        TextField(
            text: $pendingText,
            prompt: Text("Label text", bundle: .module)
        ) {
            Text("Label text", bundle: .module)
        }
        .textFieldStyle(.roundedBorder)
        .themedFont(.small)
        .frame(minWidth: 140, idealWidth: 200, maxWidth: 280)
        .focused($textFocused)
        .onSubmit(commitText)
        .onExitCommand { pendingTextAnchor = nil }
        .offset(x: point.x, y: point.y)
    }

    private func commitText() {
        let trimmed = pendingText.trimmingCharacters(in: .whitespacesAndNewlines)
        defer { pendingTextAnchor = nil }
        guard let anchor = pendingTextAnchor, !trimmed.isEmpty else { return }
        let annotation = ImageAnnotation(kind: .text(origin: anchor, string: trimmed))
        session.history.add(annotation)
        session.selectedID = annotation.id
    }

    // MARK: - VoiceOver

    /// "Box at 120, 80" style description: kind plus the position in image
    /// pixels, so a screen-reader user can tell the marks apart.
    static func spokenLabel(_ annotation: ImageAnnotation) -> String {
        switch annotation.kind {
        case .box(let rect):
            return AppStrings.localized(
                "Box at \(Int(rect.minX)), \(Int(rect.minY))")
        case .arrow(let from, _):
            return AppStrings.localized(
                "Arrow at \(Int(from.x)), \(Int(from.y))")
        case .text(let origin, let string):
            return AppStrings.localized(
                "Text label \(string) at \(Int(origin.x)), \(Int(origin.y))")
        }
    }
}
