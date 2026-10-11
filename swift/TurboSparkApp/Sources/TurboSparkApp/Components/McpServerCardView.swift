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
    let result: (message: String, isError: Bool)?
    let onTest: () -> Void
    let onEdit: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .themedFont(.title3)
                    .foregroundStyle(server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)
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
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        }
                    }

                    Text(server.commandSummary)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }

                Spacer()

                rowActions
            }

            if let result {
                Text(result.message)
                    .themedFont(.small)
                    .foregroundStyle(result.isError ? .red : .green)
                    .padding(6)
                    .background(result.isError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if !server.discoveredTools.isEmpty {
                toolsDisclosure
            }
        }
        .padding(12)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius))
        .overlay(RoundedRectangle(cornerRadius: AppMetrics.cardRadius).stroke(.appBorder, lineWidth: 0.5))
    }

    private var rowActions: some View {
        HStack(spacing: 8) {
            Button(action: onTest) {
                if isTesting {
                    ProgressView()
                        .scaleEffect(0.5)
                        .frame(width: 16, height: 16)
                } else {
                    Image(systemName: "bolt.fill")
                        .themedFont(.small)
                }
            }
            .buttonStyle(.borderless)
            .help(Text("Test connection and discover tools", bundle: .module))
            .accessibilityLabel(Text("Test connection for \(server.name)", bundle: .module))

            Button(action: onEdit) {
                Image(systemName: "gearshape")
                    .themedFont(.small)
            }
            .buttonStyle(.borderless)
            .help(Text("Edit server settings", bundle: .module))
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
                    model.deleteGlobalMcpServer(id: server.id)
                } label: {
                    Label { Text("Delete Server", bundle: .module) } icon: { Image(systemName: "trash") }
                }
            } label: {
                Image(systemName: "ellipsis")
                    .themedFont(.small)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .frame(width: 18)
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
            .background(.appSurface.opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 6))
        }
    }
}
