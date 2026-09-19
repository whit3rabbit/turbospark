import AppKit
import SwiftUI

/// View rendering tool execution results with ANSI stripping, tailing, and copy actions.
@MainActor
struct ToolResultOutputView: View {
    @Environment(\.appTheme) private var theme
    let output: String
    var isError: Bool = false
    var durationSeconds: Double = 0

    @State private var showAll = false
    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>? = nil

    /// The view's own tail bounds; the qwen-code tool rows tail too, and
    /// "Show all" is the same escape hatch.
    private let maxLines = 60
    private let maxChars = 3000

    private var tail: ToolOutputFormatter.OutputTail {
        ToolOutputFormatter.tailOutput(output, maxLines: maxLines, maxChars: maxChars)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            headerBar

            if !showAll && tail.isTruncated {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
                        showAll = true
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: "chevron.down.circle")
                            .themedFont(.tiny)
                        Text(hiddenNoticeText)
                            .themedFont(.tiny, weight: .medium)
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(Color.primary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                    .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help("Expand complete output")
            }

            ScrollView([.horizontal, .vertical], showsIndicators: true) {
                outputText
                    .font(theme.code(.small))
                    .foregroundStyle(isError ? Color.red : Color.primary)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxHeight: 220)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                isError
                    ? Color.red.opacity(0.05)
                    : theme.elevatedSurface.opacity(theme.surfaceOpacity(0.5))
            )
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(
                        isError
                            ? Color.red.opacity(0.3)
                            : theme.border,
                        lineWidth: 1
                    )
            )
        }
        .padding(.leading, 8)
        .overlay(
            Rectangle()
                .fill(isError ? Color.red.opacity(0.4) : Color.primary.opacity(0.15))
                .frame(width: 2),
            alignment: .leading
        )
    }

    /// The body text: colored when the output carries ANSI escapes (the
    /// qwen-code tool-row parity), plain otherwise. Truncation mirrors the
    /// tail above -- same line window, same plain-char cap -- so the
    /// "earlier lines hidden" notice describes what is actually not shown.
    @ViewBuilder
    private var outputText: some View {
        if let colored = coloredOutput {
            Text(colored)
        } else {
            Text(showAll ? tail.fullText : tail.visibleText)
        }
    }

    /// Colored rendering of the visible window, or nil for escape-free
    /// output. Escape sequences never carry newlines, so the LINE window
    /// computed on the raw text is the tail's own window; the char cap is
    /// applied per segment on plain character counts.
    private var coloredOutput: AttributedString? {
        guard output.contains("\u{1B}") else { return nil }
        var lines = output.components(separatedBy: "\n")
        if !showAll && lines.count > maxLines {
            lines = Array(lines.suffix(maxLines))
        }
        let source = showAll ? output : lines.joined(separator: "\n")
        var segments = ANSIColorizer.segments(in: source)
        if !showAll {
            let plainCount = segments.reduce(0) { $0 + $1.text.count }
            if plainCount > maxChars {
                var toDrop = plainCount - maxChars
                var index = 0
                while index < segments.count, segments[index].text.count <= toDrop {
                    toDrop -= segments[index].text.count
                    index += 1
                }
                var kept = Array(segments[min(index, segments.count)...])
                if !kept.isEmpty, toDrop > 0 {
                    kept[0].text = String(kept[0].text.dropFirst(toDrop))
                }
                segments = kept
            }
        }
        var attributed = AttributedString()
        let base = theme.code(.small)
        for segment in segments {
            var run = AttributedString(segment.text)
            if segment.foreground != .default {
                run.foregroundColor = Self.color(segment.foreground)
            }
            if segment.background != .default {
                run.backgroundColor = Self.color(segment.background)
            }
            if segment.bold || segment.italic {
                run.font = segment.bold && segment.italic
                    ? base.bold().italic() : segment.bold ? base.bold() : base.italic()
            }
            if segment.underline {
                run.underlineStyle = .single
            }
            // Faint has no font trait; a lower opacity reads the same.
            if segment.faint {
                run.foregroundColor = (segment.foreground != .default
                    ? Self.color(segment.foreground) : Color.primary).opacity(0.5)
            }
            attributed += run
        }
        return attributed
    }

    private static func color(_ palette: ANSIColorizer.Palette) -> Color {
        switch palette {
        case .black: return Color(red: 0, green: 0, blue: 0)
        case .red: return Color(red: 0.8, green: 0, blue: 0)
        case .green: return Color(red: 0, green: 0.8, blue: 0)
        case .yellow: return Color(red: 0.8, green: 0.8, blue: 0)
        case .blue: return Color(red: 0, green: 0, blue: 0.93)
        case .magenta: return Color(red: 0.8, green: 0, blue: 0.8)
        case .cyan: return Color(red: 0, green: 0.8, blue: 0.8)
        case .white: return Color(red: 0.9, green: 0.9, blue: 0.9)
        case .brightBlack: return Color(white: 0.5)
        case .brightRed: return Color(red: 1, green: 0.25, blue: 0.25)
        case .brightGreen: return Color(red: 0.25, green: 1, blue: 0.25)
        case .brightYellow: return Color(red: 1, green: 1, blue: 0.25)
        case .brightBlue: return Color(red: 0.4, green: 0.4, blue: 1)
        case .brightMagenta: return Color(red: 1, green: 0.25, blue: 1)
        case .brightCyan: return Color(red: 0.25, green: 1, blue: 1)
        case .brightWhite: return Color(white: 1)
        case .default: return .primary
        }
    }

    private var hiddenNoticeText: String {
        if tail.hiddenLineCount > 0 {
            return "Show all (\(tail.hiddenLineCount) earlier lines hidden)"
        }
        return "Show all (\(tail.hiddenCharCount) earlier chars hidden)"
    }

    private var headerBar: some View {
        HStack(spacing: 8) {
            Text(isError ? "ERROR" : "OUTPUT")
                .themedCode(.callout, weight: .bold)
                .foregroundStyle(isError ? Color.red : Color.secondary)

            if durationSeconds > 0 {
                Text(String(format: "(%.2fs)", durationSeconds))
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Button {
                copyOutput()
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .themedFont(.tiny)
                    Text(isCopied ? "Copied" : "Copy Output")
                        .themedFont(.tiny, weight: .medium)
                }
                .foregroundStyle(isCopied ? Color.green : Color.secondary)
            }
            .buttonStyle(.plain)
            .help("Copy full output to clipboard")
        }
        .padding(.vertical, 2)
    }

    private func copyOutput() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(tail.fullText, forType: .string)

        copyTask?.cancel()
        withAnimation(.easeInOut(duration: 0.15)) {
            isCopied = true
        }
        copyTask = Task {
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            withAnimation(.easeInOut(duration: 0.15)) {
                isCopied = false
            }
        }
    }
}
