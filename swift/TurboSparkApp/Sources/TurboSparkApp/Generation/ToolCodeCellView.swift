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
                    .foregroundStyle(.appText)
                    .textSelection(.enabled)
                    .padding(8)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appElevated)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .stroke(theme.border, lineWidth: 1)
            )
        }
        .padding(.leading, 8)
        .overlay(
            Rectangle()
                .fill(theme.border)
                .frame(width: 2),
            alignment: .leading
        )
    }

    private var headerBar: some View {
        HStack(spacing: 8) {
            Text(LocalizedStringKey(label), bundle: .module)
                .themedCode(.callout, weight: .bold)
                .foregroundStyle(.appSecondary)

            Spacer()

            Button {
                copyCode()
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: isCopied ? "checkmark" : "doc.on.doc")
                        .themedFont(.tiny)
                    (isCopied ? Text("Copied", bundle: .module) : Text("Copy", bundle: .module))
                        .themedFont(.tiny, weight: .medium)
                }
                .foregroundStyle(isCopied ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
            }
            .buttonStyle(.plain)
            .help(Text("Copy \(label) to clipboard", bundle: .module))

            if let filename = downloadFilename, !filename.isEmpty {
                Button {
                    downloadCode(filename: filename)
                } label: {
                    HStack(spacing: 3) {
                        Image(systemName: "arrow.down.doc")
                            .themedFont(.tiny)
                        Text("Download", bundle: .module)
                            .themedFont(.tiny, weight: .medium)
                    }
                    .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help(Text("Save \(label) to file", bundle: .module))
            }
        }
        .padding(.vertical, 2)
    }

    private func copyCode() {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(code, forType: .string)

        copyTask?.cancel()
        TSMotion.animate(TSMotion.press) {
            isCopied = true
        }
        copyTask = Task {
            try? await Task.sleep(for: .seconds(1.5))
            guard !Task.isCancelled else { return }
            TSMotion.animate(TSMotion.press) {
                isCopied = false
            }
        }
    }

    private func downloadCode(filename: String) {
        let panel = NSSavePanel()
        panel.canCreateDirectories = true
        panel.nameFieldStringValue = filename
        if panel.runModal() == .OK, let url = panel.url {
            do {
                try code.write(to: url, atomically: true, encoding: .utf8)
            } catch {
                NSAlert(error: error).runModal()
            }
        }
    }
}
