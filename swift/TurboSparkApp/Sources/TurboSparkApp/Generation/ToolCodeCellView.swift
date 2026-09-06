import AppKit
import SwiftUI

/// A code cell rendering tool commands, arguments, or scripts with copy and download actions.
@MainActor
struct ToolCodeCellView: View {
    @Environment(\.appTheme) private var theme
    let label: String
    let code: String
    var language: String = "bash"
    var downloadFilename: String? = nil

    @State private var isCopied = false
    @State private var copyTask: Task<Void, Never>? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            headerBar

            ScrollView(.horizontal, showsIndicators: true) {
                Text(code)
                    .font(theme.code(.small))
                    .foregroundStyle(.primary)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .textBackgroundColor).opacity(0.5))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.25), lineWidth: 1)
            )
        }
        .padding(.leading, 8)
        .overlay(
            Rectangle()
                .fill(Color.primary.opacity(0.15))
                .frame(width: 2),
            alignment: .leading
        )
    }

    private var headerBar: some View {
        HStack(spacing: 8) {
            Text(label.uppercased())
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundStyle(.secondary)

            Spacer()

            Button {
                copyCode()
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .font(.system(size: 10))
                    Text(isCopied ? "Copied" : "Copy")
                        .font(.system(size: 11, weight: .medium))
                }
                .foregroundStyle(isCopied ? Color.green : Color.secondary)
            }
            .buttonStyle(.plain)
            .help("Copy \(label) to clipboard")

            if let filename = downloadFilename, !filename.isEmpty {
                Button {
                    downloadCode(filename: filename)
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "arrow.down.doc")
                            .font(.system(size: 10))
                        Text("Download")
                            .font(.system(size: 11, weight: .medium))
                    }
                    .foregroundStyle(Color.secondary)
                }
                .buttonStyle(.plain)
                .help("Save \(label) to file")
            }
        }
        .padding(.vertical, 2)
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)

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

    private func downloadCode(filename: String) {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = filename
        if panel.runModal() == .OK, let url = panel.url {
            try? code.write(to: url, atomically: true, encoding: .utf8)
        }
    }
}
