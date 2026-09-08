import AppKit
import MarkdownUI
import SwiftUI

/// Renders markdown-formatted chat message content using MarkdownUI.
public struct ChatMessageMarkdownView: View {
    public let text: String
    /// Called when an ```html fence's Preview button is clicked. Nil (the
    /// default) hides the button, which is what every non-transcript usage
    /// wants: only the chat rows can open the panel.
    public var onPreviewHTML: ((String) -> Void)?
    @Environment(\.appTheme) private var theme

    public init(_ text: String, onPreviewHTML: ((String) -> Void)? = nil) {
        self.text = text
        self.onPreviewHTML = onPreviewHTML
    }

    public var body: some View {
        Markdown(text)
            .markdownTheme(.turboSpark(ui: theme.uiFontDescriptor, code: theme.codeFontDescriptor, onPreviewHTML: onPreviewHTML))
            .id("\(theme.uiFontDescriptor.family)-\(theme.uiFontDescriptor.size)-\(theme.uiFontDescriptor.weight)-\(theme.codeFontDescriptor.family)-\(theme.codeFontDescriptor.size)-\(theme.codeFontDescriptor.weight)")
            .foregroundStyle(theme.foreground)
            .textSelection(.enabled)
    }
}

extension Theme {
    /// Custom MarkdownUI theme tailored for TurboSpark chat transcripts.
    ///
    /// A FUNCTION rather than the `static let` this was, because a constant
    /// cannot read live settings: the transcript is the surface the font and
    /// foreground preferences most obviously describe, and it was the one
    /// place guaranteed not to follow them.
    ///
    /// The base theme carries no text style of its own, so it fell back to the
    /// ambient SwiftUI `.body` font (~13pt on macOS) regardless of the
    /// `dynamicTypeSize` scale set elsewhere in the app -- that scale moves the
    /// baseline by about a point, too small a shift to read as "bigger". The
    /// size now comes from the user's UI font setting. Everything under it
    /// (`.code`'s `.em(0.88)`, etc.) stays relative, so it scales along with
    /// the base rather than needing its own bump.
    public static func turboSpark(
        ui: AppFontDescriptor,
        code: AppFontDescriptor,
        onPreviewHTML: ((String) -> Void)? = nil
    ) -> Theme {
        Theme()
        .text {
            FontFamily(ui.markdownFamily)
            FontSize(ui.size)
            FontWeight(ui.weight)
        }
        .code {
            FontFamily(code.markdownFamily)
            FontSize(.em(0.88))
            BackgroundColor(Color.primary.opacity(0.06))
        }
        .link {
            ForegroundColor(Color.accentColor)
            UnderlineStyle(.single)
        }
        .heading1 { configuration in
            configuration.label
                .relativeLineSpacing(.em(0.2))
                .markdownMargin(top: 16, bottom: 8)
                .markdownTextStyle {
                    FontFamily(ui.markdownFamily)
                    FontSize(.em(1.5))
                    FontWeight(.bold)
                }
        }
        .heading2 { configuration in
            configuration.label
                .relativeLineSpacing(.em(0.2))
                .markdownMargin(top: 14, bottom: 6)
                .markdownTextStyle {
                    FontFamily(ui.markdownFamily)
                    FontSize(.em(1.3))
                    FontWeight(.bold)
                }
        }
        .heading3 { configuration in
            configuration.label
                .relativeLineSpacing(.em(0.2))
                .markdownMargin(top: 12, bottom: 4)
                .markdownTextStyle {
                    FontFamily(ui.markdownFamily)
                    FontSize(.em(1.15))
                    FontWeight(.semibold)
                }
        }
        .heading4 { configuration in
            configuration.label
                .relativeLineSpacing(.em(0.2))
                .markdownMargin(top: 10, bottom: 4)
                .markdownTextStyle {
                    FontFamily(ui.markdownFamily)
                    FontSize(.em(1.05))
                    FontWeight(.semibold)
                }
        }
        .codeBlock { configuration in
            CodeBlockContainer(
                language: configuration.language,
                code: configuration.content,
                onPreviewHTML: onPreviewHTML
            ) {
                configuration.label
                    .relativeLineSpacing(.em(0.225))
                    .markdownTextStyle {
                        FontFamily(code.markdownFamily)
                        FontSize(.em(0.88))
                        FontWeight(code.weight)
                    }
            }
        }
        .blockquote { configuration in
            HStack(spacing: 0) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(Color.accentColor.opacity(0.7))
                    .relativeFrame(width: .em(0.25))
                configuration.label
                    .markdownTextStyle {
                        ForegroundColor(Color.secondary)
                    }
                    .relativePadding(.horizontal, length: .em(0.8))
            }
            .fixedSize(horizontal: false, vertical: true)
        }
        .table { configuration in
            configuration.label
                .fixedSize(horizontal: false, vertical: true)
                .markdownTableBorderStyle(.init(color: Color(nsColor: .separatorColor).opacity(0.5)))
                .markdownTableBackgroundStyle(
                    .alternatingRows(Color.clear, Color.primary.opacity(0.03))
                )
                .markdownMargin(top: 8, bottom: 12)
        }
        .tableCell { configuration in
            configuration.label
                .markdownTextStyle {
                    if configuration.row == 0 {
                        FontWeight(.semibold)
                    }
                    BackgroundColor(nil)
                }
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, 6)
                .padding(.horizontal, 10)
                .relativeLineSpacing(.em(0.2))
        }
    }
}

extension AppFontDescriptor {
    /// This descriptor as MarkdownUI's own family type.
    ///
    /// MarkdownUI cannot take a built `Font`, which is why `AppFontDescriptor`
    /// keeps the parts. `FontFamilyVariant(.monospaced)` is deliberately NOT
    /// used for code any more: it asks for a fixed-width face of whatever
    /// family is current, which silently ignores the user's code-font choice.
    var markdownFamily: FontProperties.Family {
        if let name = customFamilyName {
            return .custom(name)
        }
        return .system(isCode ? .monospaced : .default)
    }
}

/// Container view for fenced code blocks featuring a language tag, copy and
/// (for html) a sandboxed-preview action.
private struct CodeBlockContainer<Content: View>: View {
    let language: String?
    let code: String
    var onPreviewHTML: ((String) -> Void)?
    @ViewBuilder let content: () -> Content
    @Environment(\.appTheme) private var theme

    @State private var isCopied = false

    /// Which language tags the Preview button offers. Only a complete HTML
    /// document is worth a panel; a fragment renders, which is acceptable,
    /// but an svg fence is deliberately NOT here: it is inline artwork in
    /// the prose, not a page.
    private var isPreviewableHTML: Bool {
        guard onPreviewHTML != nil else { return false }
        let tag = language?.lowercased()
        return tag == "html" || tag == "htm"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                if let language, !language.isEmpty {
                    Text(language.lowercased())
                        .themedCode(.tiny, weight: .semibold)
                        .foregroundStyle(.secondary)
                } else {
                    Text("code", bundle: .module)
                        .themedCode(.tiny, weight: .semibold)
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                if isPreviewableHTML {
                    Button {
                        onPreviewHTML?(code)
                    } label: {
                        HStack(spacing: 4) {
                            Image(systemName: "eye")
                                .accessibilityHidden(true)
                            Text("Preview", bundle: .module)
                        }
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(Color.secondary)
                    }
                    .buttonStyle(.plain)
                    .help("Open this HTML in the sandboxed preview panel")
                    .accessibilityLabel("Preview \(language ?? "html") block")
                    .accessibilityHint("Opens the block in the sandboxed preview panel")
                }
                Button {
                    copyCode()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .accessibilityHidden(true)
                        Text(isCopied ? "Copied" : "Copy")
                    }
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(isCopied ? Color.accentColor : Color.secondary)
                }
                .buttonStyle(.plain)
                .help("Copy code block to clipboard")
                .accessibilityLabel(isCopied ? "Copied \(language ?? "code")" : "Copy \(language ?? "code") block")
                .accessibilityHint("Copies the code block to the clipboard")
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 7)
            .background(Color.primary.opacity(0.04))

            Divider()

            ScrollView(.horizontal, showsIndicators: true) {
                // Highlighted when the language is one the native scanner
                // knows and the block is small enough; the MarkdownUI
                // rendering otherwise (which is also the streaming path for
                // big fences, keeping per-token re-renders cheap).
                if let highlighted = CodeSyntaxHighlighter.highlight(code, language: language) {
                    Text(highlighted)
                        .font(theme.code(.base))
                        // MarkdownUI's code block uses .em(0.225); at the
                        // base code size that is about three points.
                        .lineSpacing(3)
                        .textSelection(.enabled)
                } else {
                    content()
                }
            }
            .padding(12)
        }
        .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color(nsColor: .separatorColor).opacity(0.4), lineWidth: 0.5)
        )
        .markdownMargin(top: 8, bottom: 12)
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)
        _ = AccessibilityNotification.Announcement.post(
            .init("Copied \(language ?? "code") block to clipboard")
        )
        withAnimation(.easeInOut(duration: 0.15)) {
            isCopied = true
        }
        Task {
            try? await Task.sleep(for: .seconds(1.5))
            withAnimation(.easeOut(duration: 0.15)) {
                isCopied = false
            }
        }
    }
}
