import AppKit
import PDFKit
import SwiftUI

/// Right-hand preview of one attachment: the real document where the format
/// can be rendered, and the extracted text otherwise.
///
/// The extracted text is what the MODEL sees, so it is never merely a fallback
/// -- the toggle is always available, and on a PDF it is the only way to check
/// what was actually pulled out of the file.
struct FilePreviewView: View {
    @ObservedObject var model: AppModel
    let attachment: AppPromptAttachment

    @State private var showsExtractedText = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .background(TurboSparkTheme.barBackgroundColor)
        .onChange(of: attachment.id) { showsExtractedText = false }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: attachment.symbolName)
                .font(.system(size: 13))
                .foregroundStyle(TurboSparkTheme.accentColor)
                .help("\(attachment.formatLabel) file")
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.fileName)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            Button {
                model.dismissPreview()
            } label: {
                Image(systemName: "xmark")
                    .font(.system(size: 10, weight: .semibold))
                    .frame(width: 22, height: 22)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .foregroundStyle(.secondary)
            .help("Close preview")
            .accessibilityLabel("Close preview")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var subtitle: String {
        var parts = [attachment.formatLabel]
        if let size = MetricFormat.fileSize(attachment.sourceByteSize) {
            parts.append(size)
        }
        parts.append("\(attachment.characterCount.formatted(.number.notation(.compactName))) chars")
        if attachment.wasTruncatedDuringExtraction {
            parts.append("truncated")
        }
        return parts.joined(separator: " • ")
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if showsExtractedText {
            extractedTextView
        } else {
            switch attachment.previewKind {
            case .pdf:
                if let url = attachment.sourceURL {
                    PDFDocumentView(url: url)
                } else {
                    extractedTextView
                }
            case .image:
                imageView
            case .text:
                extractedTextView
            }
        }
    }

    private var extractedTextView: some View {
        ScrollView {
            Text(attachment.extractedText.isEmpty
                 ? "No text was extracted from this document."
                 : attachment.extractedText)
                .font(.system(size: 11, design: .monospaced))
                .textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityLabel("Extracted text of \(attachment.fileName)")
    }

    @ViewBuilder
    private var imageView: some View {
        if let url = attachment.sourceURL, let image = NSImage(contentsOf: url) {
            ScrollView([.horizontal, .vertical]) {
                Image(nsImage: image)
                    .resizable()
                    .scaledToFit()
                    .padding(12)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .accessibilityLabel("Image preview of \(attachment.fileName)")
        } else {
            unavailableView
        }
    }

    private var unavailableView: some View {
        VStack(spacing: 8) {
            Image(systemName: "eye.slash")
                .font(.system(size: 24))
                .foregroundStyle(.quaternary)
            Text("Preview unavailable")
                .font(.callout.weight(.medium))
            Text("The source file is no longer at its original path.")
                .font(.caption)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 6) {
            if attachment.previewKind != .text {
                Toggle(isOn: $showsExtractedText) {
                    Text("Model text")
                        .font(.system(size: 10, weight: .medium))
                }
                .toggleStyle(.button)
                .controlSize(.mini)
                .help("Show the text the model actually receives instead of the rendered document")
            }

            Spacer(minLength: 0)

            if attachment.sourceExists {
                footerButton(
                    "Reveal in Finder",
                    systemImage: "folder",
                    hint: "Opens Finder showing \(attachment.fileName)",
                    traits: .isLink
                ) {
                    model.revealAttachmentInFinder(attachment)
                }
                footerButton(
                    "Open with default app",
                    systemImage: "arrow.up.forward.app",
                    hint: "Opens \(attachment.fileName) in its default viewer",
                    traits: .isLink
                ) {
                    model.openAttachmentExternally(attachment)
                }
            }

            if let reference = model.attachmentReference(id: attachment.id) {
                footerButton(
                    "Remove",
                    systemImage: "trash",
                    role: .destructive,
                    hint: "Detaches this document from the chat"
                ) {
                    model.removeAttachment(reference: reference)
                }
                .disabled(model.isRunning)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
    }

    private func footerButton(
        _ title: String,
        systemImage: String,
        role: ButtonRole? = nil,
        hint: String? = nil,
        traits: AccessibilityTraits = [],
        action: @escaping () -> Void
    ) -> some View {
        Button(role: role, action: action) {
            Label(title, systemImage: systemImage)
                .font(.system(size: 10, weight: .medium))
                .labelStyle(.iconOnly)
                .frame(width: 24, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(role == .destructive ? Color.red.opacity(0.85) : Color.secondary)
        .help(title)
        .accessibilityLabel("\(title) \(attachment.fileName)")
        .accessibilityHint(hint ?? title)
        .accessibilityAddTraits(traits)
    }
}

/// PDFKit page view. `autoScales` is what makes the document fit the narrow
/// preview column rather than opening at 100% and needing a scroll to see.
private struct PDFDocumentView: NSViewRepresentable {
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
