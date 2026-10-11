import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The subprocess half of the MCP server editor.
///
/// A dedicated subview taking bindings rather than an extension on the sheet:
/// an extension in another file cannot see `private` state, and widening that
/// state to satisfy a line guideline trades real encapsulation for a smaller
/// file. This is the split Gotcha 15 describes.
@MainActor
struct McpStdioTransportFields: View {
    @Binding var command: String
    @Binding var argsText: String
    @Binding var envText: String
    @Binding var cwdText: String
    @Binding var envPassthroughText: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Subprocess Configuration", bundle: .module)
                .themedFont(.small, weight: .semibold)

            labelledField(Text("Command / Executable", bundle: .module)) {
                TextField(text: $command, prompt: Text("npx, python3, uvx, docker, or absolute path", bundle: .module)) {
                    Text("Command / Executable", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
            }

            labelledField(Text("Arguments (one per line or space-separated)", bundle: .module)) {
                monospacedEditor(text: $argsText, height: 60, name: Text("Arguments (one per line or space-separated)", bundle: .module))
            }

            labelledField(Text("Environment Variables (KEY=VALUE per line)", bundle: .module)) {
                monospacedEditor(text: $envText, height: 60, name: Text("Environment Variables (KEY=VALUE per line)", bundle: .module))
            }

            workingDirectoryField
            environmentPassthroughField
        }
    }

    private var workingDirectoryField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Working Directory (Optional)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            HStack(spacing: 6) {
                TextField(text: $cwdText, prompt: Text("Defaults to the project root", bundle: .module)) {
                    Text("Working Directory (Optional)", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                Button { chooseWorkingDirectory() } label: { Text("Choose...", bundle: .module) }
                    .buttonStyle(.bordered)
                    .help(Text("Pick the directory this server is launched in", bundle: .module))
            }
            Text("Leave empty to launch the server in the active project's root.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private var environmentPassthroughField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Environment Passthrough (one variable NAME per line)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            monospacedEditor(text: $envPassthroughText, height: 50, name: Text("Environment Passthrough (one variable NAME per line)", bundle: .module))
            Text("Forwards these variables from this app's own environment. PATH, HOME, LANG and TMPDIR are always forwarded; nothing else is, so a server never receives a credential it was not named.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private func chooseWorkingDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = AppStrings.localized("Choose")
        panel.message = AppStrings.localized("Choose the directory this MCP server is launched in.")
        if panel.runModal() == .OK, let url = panel.url {
            cwdText = url.path
        }
    }

    private func labelledField<Content: View>(
        _ title: Text, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            title
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            content()
        }
    }

    private func monospacedEditor(text: Binding<String>, height: CGFloat, name: Text) -> some View {
        TextEditor(text: text)
            .themedCode(.small)
            .frame(height: height)
            .appEditorSurface()
            .accessibilityLabel(name)
    }
}

// Isolated explicitly: see above.
/// The remote-endpoint half of the MCP server editor.
@MainActor
struct McpRemoteTransportFields: View {
    @Binding var urlText: String
    @Binding var headersText: String
    let urlIsValid: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("SSE Endpoint Configuration", bundle: .module)
                .themedFont(.small, weight: .semibold)

            McpRemoteTransportFields.unavailableNotice

            VStack(alignment: .leading, spacing: 4) {
                Text("Server URL", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField(text: $urlText, prompt: Text(verbatim: "https://example.com/sse")) {
                    Text("Server URL", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                if !urlIsValid && !urlText.trimmingCharacters(in: .whitespaces).isEmpty {
                    Text("Not a valid URL.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.red)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Headers (Header: Value per line)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextEditor(text: $headersText)
                    .themedCode(.small)
                    .frame(height: 60)
                    .appEditorSurface()
                    .accessibilityLabel(Text("Headers (Header: Value per line)", bundle: .module))
            }
        }
    }

    /// **THE REMOTE TRANSPORT IS NOT IMPLEMENTED AND SAYS SO HERE.**
    ///
    /// `McpClientEngine.discoverToolsViaSSE` and `callToolViaSSE` both throw,
    /// deliberately: they used to return "SSE remote tool execution completed."
    /// and an empty tool list, so every remote call reported a fabricated
    /// success (swift/CLAUDE.md Gotcha 32, state#32). Saying nothing here left a
    /// user to find that out at the first tool call, in a transcript that looked
    /// like it had worked. Do not soften this text, and do not relabel the
    /// picker's arm as "Streamable HTTP" to match another client's UI: a better
    /// name on a throwing stub is a capability claim.
    static var unavailableNotice: some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.small)
                .foregroundStyle(.orange)
            Text("Remote MCP transport is not implemented in this build. A server saved here can be configured but every tool call against it will fail. Use a local subprocess for now.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
