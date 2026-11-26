import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// One global MCP server in the MCP Servers pane: identity, the row actions
/// (test, edit, enable, overflow) and the discovered-tools disclosure.
///
/// Split out of `McpSettingsPaneView` so the pane stays near the file-size
/// guideline; the pane owns the state (which card is testing, expanded, or
/// showing a result) and passes it in.
@MainActor
struct McpServerCard: View {
    @ObservedObject var model: AppModel
    let server: McpServerConfig
    @Binding var isExpanded: Bool
    let isTesting: Bool
    let status: McpServerRuntimeStatus
    let result: (message: String, isError: Bool)?
    let onTest: () -> Void
    let onEdit: () -> Void

    @State private var isConfirmingDelete = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            // Adaptive: the four row actions drop under the server details
            // instead of squeezing its name at German and Extra Large text.
            AppAdaptiveHeader {
              HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .themedFont(.title3)
                    .foregroundStyle(server.isEnabled ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appSecondary))
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .themedFont(.base, weight: .semibold)
                            .foregroundStyle(.appText)

                        if !server.discoveredTools.isEmpty {
                            Text("Tools: \(server.discoveredTools.count)", bundle: .module)
                                .themedFont(.tiny, weight: .medium)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(.appElevated)
                                .clipShape(Capsule())
                        }
                    }

                    // Args often carry the same token as the env block.
                    Text(McpEnvMasking.scrub(server.commandSummary, secrets: server.envSecretValues))
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .help(Text(verbatim: McpEnvMasking.scrub(server.commandSummary, secrets: server.envSecretValues)))

                    McpServerStatusLabel(status: status, showsReason: result == nil)
                }
              }
            } actions: {
                rowActions
            }

            if let result {
                HStack(spacing: 6) {
                    Image(systemName: result.isError ? "exclamationmark.circle.fill" : "checkmark.circle.fill")
                        .themedFont(.small)
                    Text(result.message)
                        .themedFont(.small)
                }
                .foregroundStyle(result.isError ? AnyShapeStyle(AppAttention.color) : AnyShapeStyle(.appAccent))
                .padding(6)
                .background(.appElevated)
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius))
            }

            if !server.discoveredTools.isEmpty {
                toolsDisclosure
            }
        }
        .padding(12)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius))
        .overlay(RoundedRectangle(cornerRadius: AppMetrics.cardRadius).stroke(.appBorder, lineWidth: 0.5))
        .confirmationDialog(
            Text("Delete Server", bundle: .module),
            isPresented: $isConfirmingDelete,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                model.deleteGlobalMcpServer(id: server.id)
            } label: {
                Text("Delete Server", bundle: .module)
            }
            Button(role: .cancel) {} label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("Are you sure you want to delete this MCP server? This action cannot be undone.", bundle: .module)
        }
    }

    private var rowActions: some View {
        HStack(spacing: 8) {
            if isTesting {
                AppProgressBar(value: nil, label: Text("Test connection for \(server.name)", bundle: .module), size: .compact)
                    .frame(minWidth: 32, idealWidth: 48, maxWidth: 64)
            } else {
                AppIconButton("bolt.fill", label: Text("Test connection and discover tools", bundle: .module)) {
                    onTest()
                }
                .appAction(.quiet, size: .small)
                // The tooltip keeps the generic action; VoiceOver names the server.
                .accessibilityLabel(Text("Test connection for \(server.name)", bundle: .module))
            }

            AppIconButton("gearshape", label: Text("Edit server settings", bundle: .module)) {
                onEdit()
            }
            .appAction(.quiet, size: .small)
            .accessibilityLabel(Text("Edit settings for \(server.name)", bundle: .module))

            Toggle(isOn: Binding(
                get: { server.isEnabled },
                set: { model.toggleGlobalMcpServer(id: server.id, isEnabled: $0) }
            )) {
                Text("Enable \(server.name)", bundle: .module)
            }
            .labelsHidden()
            .toggleStyle(.switch)
            .controlSize(.small)

            Menu {
                Button(action: onEdit) {
                    Label { Text("Edit Server", bundle: .module) } icon: { Image(systemName: "pencil") }
                }
                Button(action: onTest) {
                    Label { Text("Re-query Tools", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                Divider()
                Button(role: .destructive) {
                    isConfirmingDelete = true
                } label: {
                    Label { Text("Delete Server", bundle: .module) } icon: { Image(systemName: "trash") }
                }
            } label: {
                Image(systemName: "ellipsis")
            }
            .appMenuAction()
            .help(Text("Server actions", bundle: .module))
            .accessibilityLabel(Text("Server actions for \(server.name)", bundle: .module))
        }
    }

    @ViewBuilder
    private var toolsDisclosure: some View {
        Button {
            isExpanded.toggle()
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.tiny)
                    .accessibilityHidden(true)
                Group {
                    if isExpanded {
                        Text("Hide Discovered Tools", bundle: .module)
                    } else {
                        Text("Show Discovered Tools (\(server.discoveredTools.count))", bundle: .module)
                    }
                }
                .themedFont(.tiny, weight: .medium)
            }
            .foregroundStyle(.appSecondary)
            .frame(minHeight: 20)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityLabel(isExpanded
            ? Text("Hide discovered tools for \(server.name)", bundle: .module)
            : Text("Show discovered tools for \(server.name)", bundle: .module))

        if isExpanded {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(server.discoveredTools) { tool in
                    HStack(alignment: .top, spacing: 6) {
                        Image(systemName: "wrench.and.screwdriver")
                            .themedFont(.tiny)
                            .foregroundStyle(.appAccent)
                            .padding(.top, 2)
                            .accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 1) {
                            Text(tool.name)
                                .themedCode(.small, weight: .semibold)
                            Text(tool.description)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                    .padding(.vertical, 2)
                }
            }
            .padding(8)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius))
        }
    }
}

/// One server's connection status as a glyph plus a word (StatusLabel guide),
/// with the failure reason under it. The reason is hidden while the test
/// result banner is showing, since that banner already carries the same error.
struct McpServerStatusLabel: View {
    let status: McpServerRuntimeStatus
    let showsReason: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 1) {
            Label {
                word
            } icon: {
                Image(systemName: glyph)
            }
            .themedFont(.tiny, weight: .medium)
            .foregroundStyle(tint)

            if showsReason, case .failed(let reason) = status, !reason.isEmpty {
                // Verbatim: the server or transport wrote it, not the app.
                Text(verbatim: reason)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(2)
                    .truncationMode(.tail)
                    .help(Text(verbatim: reason))
                    .textSelection(.enabled)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var word: Text {
        switch status {
        case .disabled: return Text("Disabled", bundle: .module)
        case .connecting: return Text("Connecting...", bundle: .module)
        case .connected: return Text("Connected", bundle: .module)
        case .failed: return Text("Failed", bundle: .module)
        case .notConnected: return Text("Not connected", bundle: .module)
        }
    }

    private var glyph: String {
        switch status {
        case .disabled: return "pause.circle"
        case .connecting: return "arrow.triangle.2.circlepath"
        case .connected: return "checkmark.circle.fill"
        case .failed: return "xmark.octagon.fill"
        case .notConnected: return "circle.dashed"
        }
    }

    private var tint: AnyShapeStyle {
        switch status {
        case .connected: return AnyShapeStyle(.appSuccess)
        case .failed: return AnyShapeStyle(.appDanger)
        case .disabled, .connecting, .notConnected: return AnyShapeStyle(.appSecondary)
        }
    }
}
