import SwiftUI

/// A row view displaying a configured MCP server, its active status, test connection action, and discovered tools.
struct ProjectMcpServerRowView: View {
    @Environment(\.appTheme) private var theme
    let server: McpServerConfig
    let isExpanded: Bool
    let isTesting: Bool
    let testResultToast: (id: UUID, message: String, isError: Bool)?
    let onToggleExpand: () -> Void
    let onToggleEnabled: (Bool) -> Void
    let onTest: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var showDeleteConfirmation: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(server.isEnabled ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .themedFont(.base, weight: .semibold)
                        if let src = server.sourcePath {
                            Text(URL(fileURLWithPath: src).lastPathComponent)
                                .themedFont(.tiny)
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

                HStack(spacing: 8) {
                    Button {
                        onTest()
                    } label: {
                        if isTesting {
                            ProgressView()
                                .scaleEffect(0.5)
                                .frame(width: 16, height: 16)
                        } else {
                            Image(systemName: "bolt.fill")
                                .themedFont(.small)
                        }
                    }
                    .appAction(.quiet, size: .small)
                    .help(Text("Test connection", bundle: .module))
                    .accessibilityLabel(Text("Test connection", bundle: .module))

                    AppIconButton("gearshape", label: Text("Edit server settings", bundle: .module)) {
                        onEdit()
                    }
                    .appAction(.quiet, size: .small)

                    Toggle(isOn: Binding(
                        get: { server.isEnabled },
                        set: { onToggleEnabled($0) }
                    )) {
                        Text("Enabled", bundle: .module)
                    }
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)

                    Menu {
                        Button {
                            onEdit()
                        } label: {
                            Label { Text("Edit Server", bundle: .module) } icon: { Image(systemName: "pencil") }
                        }
                        Button {
                            onTest()
                        } label: {
                            Label { Text("Re-query Tools", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                        }
                        Divider()
                        Button(role: .destructive) {
                            showDeleteConfirmation = true
                        } label: {
                            Label { Text("Delete Server", bundle: .module) } icon: { Image(systemName: "trash") }
                        }
                    } label: {
                        Label { Text("Server actions", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                            .labelStyle(.iconOnly)
                            .themedFont(.small)
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .frame(minWidth: 18, minHeight: 18)
                    .help(Text("Server actions", bundle: .module))
                }
            }

            if let toast = testResultToast, toast.id == server.id {
                HStack(spacing: 5) {
                    Image(systemName: toast.isError ? "exclamationmark.triangle.fill" : "checkmark.circle.fill")
                        .themedFont(.tiny)
                    Text(toast.message)
                        .themedFont(.small)
                }
                .foregroundStyle(toast.isError ? AnyShapeStyle(AppAttention.color) : AnyShapeStyle(theme.accent))
                .padding(6)
                .background(
                    toast.isError ? AnyShapeStyle(AppAttention.color.opacity(0.12)) : AnyShapeStyle(theme.accent.opacity(0.12)),
                    in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                )
            }

            if !server.discoveredTools.isEmpty {
                Button {
                    onToggleExpand()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .themedFont(.tiny)
                        (isExpanded
                            ? Text("Hide Discovered Tools", bundle: .module)
                            : Text("Show Discovered Tools (\(server.discoveredTools.count))", bundle: .module))
                            .themedFont(.tiny, weight: .medium)
                    }
                    .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)

                if isExpanded {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(server.discoveredTools) { tool in
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "wrench.and.screwdriver")
                                    .themedFont(.tiny)
                                    .foregroundStyle(.appAccent)
                                    .padding(.top, 2)
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
                    .background(.appElevated)
                    .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                }
            }
        }
        .padding(12)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(theme.border, lineWidth: 1)
        )
        .confirmationDialog(
            Text("Delete MCP Server", bundle: .module),
            isPresented: $showDeleteConfirmation,
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                onDelete()
            } label: {
                Text("Delete Server", bundle: .module)
            }
            Button(role: .cancel) {
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: {
            Text("Are you sure you want to delete this MCP server configuration?", bundle: .module)
        }
    }
}
