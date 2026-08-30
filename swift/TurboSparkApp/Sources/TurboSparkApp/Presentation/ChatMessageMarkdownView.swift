import AppKit
import MarkdownUI
import SwiftUI

/// Renders markdown-formatted chat message content using MarkdownUI.
public struct ChatMessageMarkdownView: View {
    public let text: String

    public init(_ text: String) {
        self.text = text
    }

    public var body: some View {
        Markdown(text)
            .markdownTheme(.turboSpark)
            .textSelection(.enabled)
    }
}

extension Theme {
    /// Custom MarkdownUI theme tailored for TurboSpark chat transcripts.
    public static let turboSpark = Theme()
        .code {
            FontFamilyVariant(.monospaced)
            FontSize(.em(0.88))
            BackgroundColor(Color.primary.opacity(0.06))
        }
        .link {
            ForegroundColor(Color.accentColor)
            UnderlineStyle(.single)
        }
        .codeBlock { configuration in
            CodeBlockContainer(
                language: configuration.language,
                code: configuration.content
            ) {
                configuration.label
                    .relativeLineSpacing(.em(0.225))
                    .markdownTextStyle {
                        FontFamilyVariant(.monospaced)
                        FontSize(.em(0.88))
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

/// Container view for fenced code blocks featuring a language tag and copy-to-clipboard action.
private struct CodeBlockContainer<Content: View>: View {
    let language: String?
    let code: String
    @ViewBuilder let content: () -> Content

    @State private var isCopied = false

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack {
                if let language, !language.isEmpty {
                    Text(language.lowercased())
                        .font(.caption2.monospaced().weight(.semibold))
                        .foregroundStyle(.secondary)
                } else {
                    Text("code")
                        .font(.caption2.monospaced().weight(.semibold))
                        .foregroundStyle(.tertiary)
                }
                Spacer()
                Button {
                    copyCode()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                            .accessibilityHidden(true)
                        Text(isCopied ? "Copied" : "Copy")
                    }
                    .font(.caption2.weight(.medium))
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
                content()
                    .padding(12)
            }
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
