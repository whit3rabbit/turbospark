import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Right-hand preview of one attachment: the real document where the format
/// can be rendered, and the extracted text otherwise.
///
/// The extracted text is what the MODEL sees, so it is never merely a fallback
/// -- the toggle is always available, and on a PDF it is the only way to check
/// what was actually pulled out of the file.
@MainActor
struct FilePreviewView: View {
    @ObservedObject var model: AppModel
    let attachment: AppPromptAttachment

    @State private var showsExtractedText = false
    @State private var isMaximized = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .background(TurboSparkTheme.barBackgroundColor)
        .sheet(isPresented: $isMaximized) { maximizedSheet }
        .onChange(of: attachment.id) { showsExtractedText = false }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: attachment.symbolName)
                .themedFont(.callout)
                .foregroundStyle(.appAccent)
                .help("\(attachment.formatLabel) file")
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.fileName)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Text(subtitle)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            headerButton(
                "arrow.up.left.and.arrow.down.right",
                help: "Open a larger preview",
                label: "Open a larger preview")
            {
                isMaximized = true
            }

            headerButton("xmark", help: "Close preview", label: "Close preview") {
                model.dismissPreview()
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private func headerButton(
        _ systemImage: String,
        help: String,
        label: String,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Image(systemName: systemImage)
                .themedFont(.tiny, weight: .semibold)
                .frame(width: 22, height: 22)
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .foregroundStyle(.appSecondary)
        .help(help)
        .accessibilityLabel(label)
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
                textOrQuickLookView
            }
        }
    }

    private var extractedTextView: some View {
        ScrollView {
            Text(attachment.extractedText.isEmpty
                 ? "No text was extracted from this document."
                 : attachment.extractedText)
                .themedCode(.callout)
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
            ImagePreviewView(image: image)
                .accessibilityLabel("Image preview of \(attachment.fileName)")
        } else {
            unavailableView
        }
    }

    /// An extraction that produced nothing is not proof the file is unviewable:
    /// a scanned PDF and some office files extract empty and QuickLook renders
    /// both. Preferring it here hides nothing, because the Model-text toggle
    /// still reaches `extractedTextView` (see `hasRenderedAlternative`).
    @ViewBuilder
    private var textOrQuickLookView: some View {
        if let url = quickLookURL {
            QuickLookPreviewView(url: url)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .accessibilityLabel("Quick Look preview of \(attachment.fileName)")
        } else {
            extractedTextView
        }
    }

    /// The source to hand QuickLook, or nil when the extracted text is the
    /// better answer. Nil whenever there IS text, so a readable document is
    /// never replaced by a picture of itself.
    private var quickLookURL: URL? {
        guard attachment.extractedText.isEmpty, attachment.sourceExists else { return nil }
        return attachment.sourceURL
    }

    /// Whether `content` shows anything other than the extracted text, and so
    /// whether the Model-text toggle has work to do.
    ///
    /// NOT `previewKind != .text`, which was the test until the QuickLook arm
    /// landed: a `.text` attachment rendering through QuickLook would have lost
    /// its only route back to what the model actually receives.
    private var hasRenderedAlternative: Bool {
        switch attachment.previewKind {
        case .pdf, .image: return true
        case .text: return quickLookURL != nil
        }
    }

    private var unavailableView: some View {
        VStack(spacing: 8) {
            Image(systemName: "eye.slash")
                .themedFont(.title2)
                .foregroundStyle(.quaternary)
            Text("Preview unavailable", bundle: .module)
                .themedFont(.base, weight: .medium)
            Text("The source file is no longer at its original path.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Footer

    private var footer: some View {
        HStack(spacing: 6) {
            if hasRenderedAlternative {
                Toggle(isOn: $showsExtractedText) {
                    Text("Model text", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
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

    // MARK: - Maximized sheet

    /// Same shape and size as the artifact panel's maximize sheet, deliberately.
    /// It reuses `content`, so `showsExtractedText` carries into the sheet and
    /// the Model-text toggle survives the transition in both directions.
    private var maximizedSheet: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Image(systemName: attachment.symbolName)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Text(attachment.fileName)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                Spacer(minLength: 4)
                headerButton("xmark", help: "Close", label: "Close maximized preview") {
                    isMaximized = false
                }
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            content
        }
        .frame(width: 1100, height: 800)
        .background(TurboSparkTheme.barBackgroundColor)
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
                .themedFont(.tiny, weight: .medium)
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
