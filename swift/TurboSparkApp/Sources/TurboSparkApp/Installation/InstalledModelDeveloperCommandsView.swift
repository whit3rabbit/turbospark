import AppKit
import SwiftUI
import TurboSpark

/// Developer CLI commands card providing copyable snippets for terminal chat and API serving.
struct InstalledModelDeveloperCommandsView: View {
    let alias: String
    let defaultSystemPrompt: String
    let onShowToast: (String) -> Void

    private var serveCommand: String {
        let base = "turbospark-server --model \(alias)"
        let prompt = defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return base }
        return "\(base) --system \(ShellQuote.single(prompt))"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label("Terminal / CLI Commands", systemImage: "terminal")
                .font(.subheadline.weight(.semibold))

            cliSnippet(
                title: "Run CLI REPL Chat",
                command: "turbospark-check --model \(alias) --chat"
            )

            cliSnippet(
                title: "Serve OpenAI & Anthropic API",
                command: serveCommand
            )
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    private func cliSnippet(title: String, command: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption2).foregroundStyle(.secondary)
            HStack {
                Text(command)
                    .font(.caption.monospaced())
                    .lineLimit(1)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(command, forType: .string)
                    onShowToast("Command copied to clipboard")
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.caption)
                }
                .buttonStyle(.plain)
                .help("Copy command")
            }
            .padding(6)
            .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 4))
        }
    }
}
