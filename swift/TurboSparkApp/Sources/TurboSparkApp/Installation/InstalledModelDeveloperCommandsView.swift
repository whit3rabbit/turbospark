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
            Label { Text("Terminal / CLI Commands", bundle: .module) } icon: { Image(systemName: "terminal") }
                .themedFont(.small, weight: .semibold)

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
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
    }

    private func cliSnippet(title: String, command: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).themedFont(.tiny).foregroundStyle(.appSecondary)
            HStack {
                Text(command)
                    .themedCode(.small)
                    .lineLimit(1)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(command, forType: .string)
                    onShowToast("Command copied to clipboard")
                } label: {
                    Image(systemName: "doc.on.doc")
                        .themedFont(.small)
                }
                .buttonStyle(.plain)
                .help("Copy command")
            }
            .padding(6)
            .background(.appElevated, in: RoundedRectangle(cornerRadius: 4))
        }
    }
}
