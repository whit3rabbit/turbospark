import AppKit
import SwiftUI
import TurboSpark

/// Developer CLI commands card providing copyable snippets for terminal chat and API serving.
struct InstalledModelDeveloperCommandsView: View {
    /// Filesystem path of the install. The CLI accepts a path for `--model`,
    /// and unlike an alias it also resolves scanned (non-store) rows.
    let modelPath: String
    let defaultSystemPrompt: String
    let onShowToast: (String) -> Void

    /// The model reference is always single-quoted: aliases can hold spaces
    /// or parentheses ("gemma4 (LM Studio)") and scanned file names are
    /// attacker-influenced, so an unquoted paste could run `$(...)`.
    static func chatCommand(modelPath: String) -> String {
        "turbospark-check --model \(ShellQuote.single(modelPath)) --chat"
    }

    static func serveCommand(modelPath: String, defaultSystemPrompt: String) -> String {
        let base = "turbospark-server --model \(ShellQuote.single(modelPath))"
        let prompt = defaultSystemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !prompt.isEmpty else { return base }
        return "\(base) --system \(ShellQuote.single(prompt))"
    }

    private var serveCommand: String {
        Self.serveCommand(modelPath: modelPath, defaultSystemPrompt: defaultSystemPrompt)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Label { Text("Terminal / CLI Commands", bundle: .module) } icon: { Image(systemName: "terminal") }
                .themedFont(.small, weight: .semibold)

            cliSnippet(
                title: "Run CLI REPL Chat",
                command: Self.chatCommand(modelPath: modelPath)
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
