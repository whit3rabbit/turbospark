import AppKit
import SwiftUI

/// Pan-and-zoom image with an annotation layer (ZP-11).
///
/// Takes a loaded `NSImage` rather than a URL so the caller keeps its own
/// missing-file message and accessibility label. The original bytes are never
/// written: annotations live in `ImageAnnotationSession` (view state, gone when
/// the preview closes) and reach the model only as a flattened PNG handed to
/// `onSendToChat`. A caller with no composer to attach to omits the callback
/// and the Send action does not appear.
///
/// Point-and-ask (`onAsk`) is offered only when the caller passes it, and
/// callers pass it only while the loaded model takes images
/// (`AppModel.visionIsActive`). It attaches a marked copy and drafts a
/// question; the user still presses Send.
///
/// Pan is the scroll view, zoom is `AnnotationViewport.zoom` (buttons or
/// pinch), and the image is always laid out at natural pixel size times
/// `scale`, so there is no resampling to lose quality at any zoom.
@MainActor
struct ImagePreviewView: View {
    let image: NSImage
    var onSendToChat: ((Data) -> Void)?
    /// Receives the flattened PNG (with the asked region boxed) and the
    /// drafted question. Nil hides every Ask control.
    var onAsk: ((Data, String) -> Void)?

    @Environment(\.appTheme) private var theme
    @State private var session: ImageAnnotationSession
    @State private var pinchStartZoom: CGFloat?
    @State private var exportError: String?
    private let cgImage: CGImage?

    init(
        image: NSImage, onAsk: ((Data, String) -> Void)? = nil,
        onSendToChat: ((Data) -> Void)? = nil
    ) {
        self.image = image
        self.onAsk = onAsk
        self.onSendToChat = onSendToChat
        let source = AnnotationExporter.sourceCGImage(of: image)
        cgImage = source
        let size = source.map(AnnotationExporter.pixelSize(of:)) ?? image.size
        _session = State(initialValue: ImageAnnotationSession(imageSize: size))
    }

    /// Points per pixel at which a pixel is one device pixel, the ceiling for
    /// the fit scale so a small image is not blown up and blurred.
    private var naturalScale: CGFloat {
        1 / max(NSScreen.main?.backingScaleFactor ?? 2, 1)
    }

    var body: some View {
        VStack(spacing: 0) {
            toolbar
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            if let exportError {
                Label {
                    Text(exportError).themedFont(.small)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .foregroundStyle(.appWarning)
                .padding(.horizontal, 12)
                .padding(.bottom, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            if session.tool == .ask, onAsk != nil {
                Label {
                    Text("Click a spot or drag a region to ask about it.", bundle: .module)
                        .themedFont(.small)
                } icon: {
                    Image(systemName: "questionmark.bubble")
                }
                .foregroundStyle(theme.secondaryText)
                .padding(.horizontal, 12)
                .padding(.bottom, 6)
                .frame(maxWidth: .infinity, alignment: .leading)
            }
            Divider()
            GeometryReader { proxy in
                canvasArea(container: proxy.size)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        // The model can unload while the preview is open; Ask mode must not
        // outlive the gate that offered it.
        .onChange(of: onAsk == nil) {
            if onAsk == nil, session.tool == .ask { session.tool = .select }
        }
    }

    // MARK: - Image area

    private func canvasArea(container: CGSize) -> some View {
        let inset: CGFloat = 12
        let available = CGSize(
            width: max(container.width - inset * 2, 1),
            height: max(container.height - inset * 2, 1))
        // Binding that keeps the fit scale current without a side-effecting
        // body: the fit follows the container on every layout.
        var fitted = session.viewport
        fitted.fit(into: available, maxScale: naturalScale)
        let content = fitted.contentSize
        return ScrollView([.horizontal, .vertical]) {
            ZStack(alignment: .topLeading) {
                Image(nsImage: image)
                    .resizable()
                    .interpolation(.high)
                    .frame(width: content.width, height: content.height)
                    .accessibilityHidden(true)
                AnnotationLayerView(
                    session: $session,
                    onAskRegion: onAsk == nil ? nil : { ask(region: $0) })
            }
            .frame(width: content.width, height: content.height)
            .frame(
                minWidth: container.width - inset * 2,
                minHeight: container.height - inset * 2)
            .padding(inset)
        }
        .onAppear { session.viewport.fitScale = fitted.fitScale }
        .onChange(of: fitted.fitScale) { session.viewport.fitScale = fitted.fitScale }
        .gesture(
            MagnifyGesture()
                .onChanged { value in
                    let start = pinchStartZoom ?? session.viewport.zoom
                    pinchStartZoom = start
                    session.viewport.setZoom(start * value.magnification)
                }
                .onEnded { _ in pinchStartZoom = nil }
        )
        .focusable()
        .focusEffectDisabled()
        .onDeleteCommand { session.deleteSelection() }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        AppAdaptiveHeader {
            AppPillPicker(
                selection: $session.tool,
                items: [
                    AppPillItem(value: .select, label: Text("Select", bundle: .module), systemImage: "cursorarrow"),
                    AppPillItem(value: .box, label: Text("Box", bundle: .module), systemImage: "rectangle"),
                    AppPillItem(value: .arrow, label: Text("Arrow", bundle: .module), systemImage: "arrow.up.right"),
                    AppPillItem(value: .text, label: Text("Text", bundle: .module), systemImage: "textformat"),
                ],
                label: Text("Annotation tool", bundle: .module))
        } actions: {
            actionButtons
        }
    }

    @ViewBuilder
    private var actionButtons: some View {
        AppIconButton("arrow.uturn.backward", label: Text("Undo", bundle: .module)) {
            session.undo()
        }
        .disabled(!session.history.canUndo)
        AppIconButton("arrow.uturn.forward", label: Text("Redo", bundle: .module)) {
            session.redo()
        }
        .disabled(!session.history.canRedo)
        // Deletion is reversible through the Undo action beside it, so this
        // carries no destructive role: no red styling, no confirmation.
        AppIconButton("trash", label: Text("Delete annotation", bundle: .module)) {
            session.deleteSelection()
        }
        .disabled(session.selectedID == nil)
        AppIconButton("minus.magnifyingglass", label: Text("Zoom out", bundle: .module)) {
            session.viewport.zoomOut()
        }
        .disabled(session.viewport.zoom <= AnnotationViewport.zoomRange.lowerBound)
        AppIconButton("plus.magnifyingglass", label: Text("Zoom in", bundle: .module)) {
            session.viewport.zoomIn()
        }
        .disabled(session.viewport.zoom >= AnnotationViewport.zoomRange.upperBound)
        AppIconButton("arrow.up.left.and.down.right.magnifyingglass", label: Text("Fit to window", bundle: .module)) {
            session.viewport.resetZoom()
        }
        .disabled(session.viewport.zoom == 1)
        if onAsk != nil {
            askControls
        }
        if onSendToChat != nil {
            sendButton
        }
    }

    private var sendButton: some View {
        let hasMarks = !session.history.annotations.isEmpty
        return Button {
            send()
        } label: {
            Label { Text("Send to chat", bundle: .module) } icon: { Image(systemName: "paperplane") }
        }
        .appAction(.primary, size: .small)
        .disabled(!hasMarks)
        .help(hasMarks
            ? Text("Attach the annotated image to the prompt", bundle: .module)
            : Text("Add an annotation first", bundle: .module))
    }

    private func send() {
        guard let onSendToChat, let png = flattened(session.history.annotations) else { return }
        onSendToChat(png)
    }

    /// The image plus `annotations` as PNG, or nil after showing and
    /// announcing why it failed.
    private func flattened(_ annotations: [ImageAnnotation]) -> Data? {
        guard let cgImage else { return nil }
        do {
            let png = try AnnotationExporter.flatten(cgImage, annotations: annotations)
            exportError = nil
            return png
        } catch {
            let message = error.localizedDescription
            exportError = message
            AccessibilityNotification.Announcement(message).post()
            return nil
        }
    }

    // MARK: - Point-and-ask

    /// The pointer path (Ask mode) and the non-pointer path (Ask about image,
    /// reachable by keyboard and VoiceOver) side by side.
    @ViewBuilder
    private var askControls: some View {
        let asking = session.tool == .ask
        AppChip(Text("Ask", bundle: .module), systemImage: "questionmark.bubble", isOn: asking) {
            session.tool = asking ? .select : .ask
        }
        .accessibilityAddTraits(asking ? .isSelected : [])
        .accessibilityHint(Text("Click a spot or drag a region to ask about it.", bundle: .module))
        AppIconButton("text.bubble", label: Text("Ask about image", bundle: .module)) {
            ask(region: nil)
        }
    }

    /// Attaches a copy with every existing mark plus, for a region, a box
    /// around it, and hands the drafted question to the caller. The box joins
    /// the history so what stays on screen is what the model was sent, and
    /// Undo removes it.
    private func ask(region: CGRect?) {
        guard let onAsk else { return }
        var marks = session.history.annotations
        var box: ImageAnnotation?
        let question: String
        if let region {
            let mark = ImageAnnotation(kind: .box(region))
            marks.append(mark)
            box = mark
            question = PointAndAsk.question(region: region, imageSize: session.viewport.imageSize)
        } else {
            question = PointAndAsk.wholeImageQuestion()
        }
        guard let png = flattened(marks) else { return }
        if let box {
            session.history.add(box)
            session.selectedID = box.id
            // One ask per gesture: staying in Ask mode would attach another
            // copy on the next stray click.
            session.tool = .select
        }
        onAsk(png, question)
    }
}
