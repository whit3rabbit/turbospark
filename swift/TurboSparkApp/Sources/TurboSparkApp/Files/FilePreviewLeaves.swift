import AppKit
import PDFKit
import QuickLookUI
import SwiftUI

// The leaf renderers every preview surface shares. They live here rather than
// in one of the two panels because both panels reach for all three: putting
// them in `FilePreviewView` (where PDFDocumentView started) made the artifact
// panel import a file it has nothing else to do with.
//
// Each one renders ONE already-resolved URL or image and owns no fallback.
// Deciding what to show when the file is gone is the caller's job, and the two
// callers word it differently on purpose ("no longer at its original path" for
// an attachment the user chose, "no longer at its written path" for an artifact
// the model wrote).

/// Scrollable, aspect-fitting image at full resolution.
///
/// Takes a loaded `NSImage` rather than a URL so the caller keeps its own
/// missing-file message and accessibility label.
@MainActor
struct ImagePreviewView: View {
    let image: NSImage

    var body: some View {
        ScrollView([.horizontal, .vertical]) {
            Image(nsImage: image)
                .resizable()
                .scaledToFit()
                .padding(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

/// PDFKit page view, shared with the artifact panel. `autoScales` is what
/// makes the document fit the narrow preview column rather than opening at
/// 100% and needing a scroll to see.
struct PDFDocumentView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> PDFView {
        let view = PDFView()
        view.autoScales = true
        view.displayMode = .singlePageContinuous
        view.displayDirection = .vertical
        view.backgroundColor = .clear
        view.document = PDFDocument(url: url)
        return view
    }

    func updateNSView(_ view: PDFView, context: Context) {
        if view.document?.documentURL != url {
            view.document = PDFDocument(url: url)
        }
    }
}

/// System QuickLook, for the formats neither `NSImage` nor PDFKit renders.
///
/// The population is mostly ARTIFACTS rather than attachments: the attachment
/// picker is restricted to `attachmentContentTypes` and `AttachmentImporter`
/// drops a file whose extraction throws, so an attachment is always an image or
/// a document with text. A model can write anything.
///
/// Wrapped in a plain container `NSView` because `QLPreviewView.init(frame:style:)`
/// is FAILABLE and `makeNSView` is not: returning the container lets a nil
/// preview degrade to an empty area instead of forcing an unwrap that a
/// missing QuickLook generator would trap on.
struct QuickLookPreviewView: NSViewRepresentable {
    let url: URL

    func makeNSView(context: Context) -> NSView {
        let container = NSView()
        guard let preview = QLPreviewView(frame: .zero, style: .normal) else {
            return container
        }
        preview.autostarts = true
        preview.previewItem = url as NSURL
        preview.translatesAutoresizingMaskIntoConstraints = false
        container.addSubview(preview)
        NSLayoutConstraint.activate([
            preview.leadingAnchor.constraint(equalTo: container.leadingAnchor),
            preview.trailingAnchor.constraint(equalTo: container.trailingAnchor),
            preview.topAnchor.constraint(equalTo: container.topAnchor),
            preview.bottomAnchor.constraint(equalTo: container.bottomAnchor),
        ])
        return container
    }

    func updateNSView(_ nsView: NSView, context: Context) {
        guard let preview = nsView.subviews.first as? QLPreviewView else { return }
        if preview.previewItem?.previewItemURL != url {
            preview.previewItem = url as NSURL
        }
    }

    // NOT tidiness. `QLPreviewView` holds a preview session (and on some types
    // a helper process) that outlives the SwiftUI view unless it is closed.
    static func dismantleNSView(_ nsView: NSView, coordinator: ()) {
        (nsView.subviews.first as? QLPreviewView)?.close()
    }
}
