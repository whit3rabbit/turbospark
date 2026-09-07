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

    private var tail: ToolOutputFormatter.OutputTail {
        ToolOutputFormatter.tailOutput(output, maxLines: 60, maxChars: 3000)
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
                            .themedFont(points: 10)
                        Text(hiddenNoticeText)
                            .themedFont(points: 11, weight: .medium)
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 3)
                    .background(Color.primary.opacity(0.06))
                    .clipShape(RoundedRectangle(cornerRadius: 4, style: .continuous))
                    .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
                .help("Expand complete output")
            }

            ScrollView([.horizontal, .vertical], showsIndicators: true) {
                Text(showAll ? tail.fullText : tail.visibleText)
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
                    : Color(nsColor: .textBackgroundColor).opacity(0.5)
            )
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(
                        isError
                            ? Color.red.opacity(0.3)
                            : Color(nsColor: .separatorColor).opacity(0.25),
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

    private var hiddenNoticeText: String {
        if tail.hiddenLineCount > 0 {
            return "Show all (\(tail.hiddenLineCount) earlier lines hidden)"
        }
        return "Show all (\(tail.hiddenCharCount) earlier chars hidden)"
    }

    private var headerBar: some View {
        HStack(spacing: 8) {
            Text(isError ? "ERROR" : "OUTPUT")
                .themedCode(points: 10, weight: .bold)
                .foregroundStyle(isError ? Color.red : Color.secondary)

            if durationSeconds > 0 {
                Text(String(format: "(%.2fs)", durationSeconds))
                    .themedFont(points: 10)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Button {
                copyOutput()
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .themedFont(points: 10)
                    Text(isCopied ? "Copied" : "Copy Output")
                        .themedFont(points: 11, weight: .medium)
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
