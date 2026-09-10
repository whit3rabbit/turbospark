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

            labelledField("Command / Executable") {
                TextField("npx, python3, uvx, docker, or absolute path", text: $command)
                    .textFieldStyle(.roundedBorder)
            }

            labelledField("Arguments (one per line or space-separated)") {
                monospacedEditor(text: $argsText, height: 60)
            }

            labelledField("Environment Variables (KEY=VALUE per line)") {
                monospacedEditor(text: $envText, height: 60)
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
                TextField("Defaults to the project root", text: $cwdText)
                    .textFieldStyle(.roundedBorder)
                Button("Choose...") { chooseWorkingDirectory() }
                    .buttonStyle(.bordered)
                    .help("Pick the directory this server is launched in")
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
            monospacedEditor(text: $envPassthroughText, height: 50)
            Text(
                "Forwards these variables from this app's own environment. "
                + "PATH, HOME, LANG and TMPDIR are always forwarded; nothing else is, "
                + "so a server never receives a credential it was not named.")
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private func chooseWorkingDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose the directory this MCP server is launched in."
        if panel.runModal() == .OK, let url = panel.url {
            cwdText = url.path
        }
    }

    private func labelledField<Content: View>(
        _ title: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            content()
        }
    }

    private func monospacedEditor(text: Binding<String>, height: CGFloat) -> some View {
        TextEditor(text: text)
            .themedCode(.small)
            .frame(height: height)
            .padding(4)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
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
                TextField("https://example.com/sse", text: $urlText)
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
                    .padding(4)
                    .background(.appSurface)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(
                        RoundedRectangle(cornerRadius: 6)
                            .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
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
            Text(
                "Remote MCP transport is not implemented in this build. A server "
                + "saved here can be configured but every tool call against it will "
                + "fail. Use a local subprocess for now.")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}
