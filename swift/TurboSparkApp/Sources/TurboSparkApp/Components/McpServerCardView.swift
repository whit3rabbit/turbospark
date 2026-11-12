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

    @State private var isConfirmingDelete = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
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

                    Text(server.commandSummary)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                }

                Spacer()

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
