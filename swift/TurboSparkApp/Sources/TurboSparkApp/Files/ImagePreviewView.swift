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
/// Pan is the scroll view, zoom is `AnnotationViewport.zoom` (buttons or
/// pinch), and the image is always laid out at natural pixel size times
/// `scale`, so there is no resampling to lose quality at any zoom.
@MainActor
struct ImagePreviewView: View {
    let image: NSImage
    var onSendToChat: ((Data) -> Void)?

    @Environment(\.appTheme) private var theme
    @State private var session: ImageAnnotationSession
    @State private var pinchStartZoom: CGFloat?
    @State private var exportError: String?
    private let cgImage: CGImage?

    init(image: NSImage, onSendToChat: ((Data) -> Void)? = nil) {
        self.image = image
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
            Divider()
            GeometryReader { proxy in
                canvasArea(container: proxy.size)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                AnnotationLayerView(session: $session)
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
        AppIconButton("trash", label: Text("Delete annotation", bundle: .module), role: .destructive) {
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
        guard let cgImage, let onSendToChat else { return }
        do {
            let png = try AnnotationExporter.flatten(
                cgImage, annotations: session.history.annotations)
            exportError = nil
            onSendToChat(png)
        } catch {
            let message = error.localizedDescription
            exportError = message
            AccessibilityNotification.Announcement(message).post()
        }
    }
}
