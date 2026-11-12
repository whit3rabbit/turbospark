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
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let attachment: AppPromptAttachment

    @State private var showsExtractedText = false
    @State private var isMaximized = false
    /// The full extracted text behind a preview gate. The preview renders
    /// the head+tail window from `MessageContentPreview` until this flips;
    /// a 240k-character extraction (the extractor's own ceiling) must not
    /// lay out in full just because the pane opened.
    @State private var showsFullText = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            content
            Divider()
            footer
        }
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
        .sheet(isPresented: $isMaximized) { maximizedSheet }
        .onChange(of: attachment.id) { showsExtractedText = false }
        .onChange(of: attachment.id) { showsFullText = false }
    }

    // MARK: - Header

    private var header: some View {
        HStack(spacing: 8) {
            Image(systemName: attachment.symbolName)
                .themedFont(.callout)
                .foregroundStyle(.appAccent)
                .help(Text("\(attachment.formatLabel) file", bundle: .module))
                .accessibilityHidden(true)

            VStack(alignment: .leading, spacing: 1) {
                Text(attachment.fileName)
                    .themedFont(.small, weight: .semibold)
                    .lineLimit(1)
                    .truncationMode(.middle)
                subtitle
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 4)

            HStack(spacing: 2) {
                AppIconButton(
                    "arrow.up.left.and.arrow.down.right",
                    label: Text("Open a larger preview", bundle: .module)
                ) {
                    isMaximized = true
                }
                AppIconButton("xmark", label: Text("Close preview", bundle: .module)) {
                    model.dismissPreview()
                }
            }
            .appAction(.quiet, size: .small)
            .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    /// Format, size and character count. The parts that carry no words
    /// (format label, byte size) stay verbatim; the count and the truncation
    /// note are catalog strings so they follow the in-app language.
    private var subtitle: Text {
        var parts = [attachment.formatLabel]
        if let size = MetricFormat.fileSize(attachment.sourceByteSize) {
            parts.append(size)
        }
        let separator = Text(verbatim: " \u{2022} ")
        let count = attachment.characterCount.formatted(.number.notation(.compactName))
        var line = Text(verbatim: parts.joined(separator: " \u{2022} "))
            + separator + Text("\(count) characters", bundle: .module)
        if attachment.wasTruncatedDuringExtraction {
            line = line + separator + Text("truncated", bundle: .module)
        }
        return line
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
        let preview = showsFullText ? nil : MessageContentPreview.make(attachment.extractedText)
        return ScrollView {
            VStack(alignment: .leading, spacing: 8) {
                displayText(preview: preview)
                    .themedCode(.callout)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                if preview != nil {
                    Button {
                        showsFullText = true
                    } label: {
                        Text(
                            "View all (+\(preview?.hiddenCharacterCount ?? 0) characters)",
                            bundle: .module)
                            .themedFont(.small, weight: .medium)
                    }
                    .buttonStyle(.plain)
                    .foregroundStyle(.appAccent)
                }
            }
            .padding(12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .accessibilityLabel(Text("Extracted text of \(attachment.fileName)", bundle: .module))
    }

    /// The text to show: the preview window while gated, the empty-extraction
    /// notice, or the whole extraction once expanded.
    private func displayText(preview: MessageContentPreview.Preview?) -> Text {
        if let preview { return Text(preview.visible) }
        if attachment.extractedText.isEmpty {
            return Text("No text was extracted from this document.", bundle: .module)
        }
        return Text(attachment.extractedText)
    }

    @ViewBuilder
    private var imageView: some View {
        if let url = attachment.sourceURL, let image = NSImage(contentsOf: url) {
            ImagePreviewView(image: image)
                .accessibilityLabel(Text("Image preview of \(attachment.fileName)", bundle: .module))
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
                .accessibilityLabel(Text("Quick Look preview of \(attachment.fileName)", bundle: .module))
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
                .help(Text("Show the text the model actually receives instead of the rendered document", bundle: .module))
            }

            Spacer(minLength: 0)

            HStack(spacing: 2) {
                if attachment.sourceExists {
                    AppIconButton("folder", label: Text("Reveal in Finder", bundle: .module)) {
                        model.revealAttachmentInFinder(attachment)
                    }
                    .accessibilityHint(Text("Opens Finder showing \(attachment.fileName)", bundle: .module))
                    .accessibilityAddTraits(.isLink)
                    AppIconButton("arrow.up.forward.app", label: Text("Open with default app", bundle: .module)) {
                        model.openAttachmentExternally(attachment)
                    }
                    .accessibilityHint(Text("Opens \(attachment.fileName) in its default viewer", bundle: .module))
                    .accessibilityAddTraits(.isLink)
                }

                if let reference = model.attachmentReference(id: attachment.id) {
                    AppIconButton("trash", label: Text("Remove", bundle: .module), role: .destructive) {
                        model.removeAttachment(reference: reference)
                    }
                    .foregroundStyle(.appDanger)
                    .accessibilityHint(Text("Detaches this document from the chat", bundle: .module))
                    .disabled(model.isRunning)
                }
            }
            .appAction(.quiet, size: .small)
            .foregroundStyle(.appSecondary)
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
                AppIconButton("xmark", label: Text("Close preview", bundle: .module)) {
                    isMaximized = false
                }
                .appAction(.quiet, size: .small)
                .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            Divider()
            content
        }
        .frame(width: 1100, height: 800)
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
    }
}
