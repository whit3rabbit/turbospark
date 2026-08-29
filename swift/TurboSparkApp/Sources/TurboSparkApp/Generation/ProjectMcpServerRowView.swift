import SwiftUI

/// A row view displaying a configured MCP server, its active status, test connection action, and discovered tools.
struct ProjectMcpServerRowView: View {
    let server: McpServerConfig
    let isExpanded: Bool
    let isTesting: Bool
    let testResultToast: (id: UUID, message: String, isError: Bool)?
    let onToggleExpand: () -> Void
    let onToggleEnabled: (Bool) -> Void
    let onTest: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .font(.headline)
                    .foregroundStyle(server.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 6) {
                        Text(server.name)
                            .font(.headline)
                        if let src = server.sourcePath {
                            Text(URL(fileURLWithPath: src).lastPathComponent)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(Color.secondary.opacity(0.12))
                                .clipShape(Capsule())
                        }
                    }
                    Text(server.commandSummary)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
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
                                .font(.caption)
                        }
                    }
                    .buttonStyle(.borderless)
                    .help("Test connection")

                    Button {
                        onEdit()
                    } label: {
                        Image(systemName: "gearshape")
                            .font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Edit server settings")

                    Toggle("", isOn: Binding(
                        get: { server.isEnabled },
                        set: { onToggleEnabled($0) }
                    ))
                    .labelsHidden()
                    .toggleStyle(.switch)
                    .controlSize(.small)

                    Menu {
                        Button("Edit Server", systemImage: "pencil") {
                            onEdit()
                        }
                        Button("Re-query Tools", systemImage: "arrow.clockwise") {
                            onTest()
                        }
                        Divider()
                        Button("Delete Server", systemImage: "trash", role: .destructive) {
                            onDelete()
                        }
                    } label: {
                        Image(systemName: "ellipsis")
                            .font(.caption)
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .frame(width: 18)
                    .help("Server actions")
                }
            }

            if let toast = testResultToast, toast.id == server.id {
                Text(toast.message)
                    .font(.caption)
                    .foregroundStyle(toast.isError ? .red : .green)
                    .padding(6)
                    .background(toast.isError ? Color.red.opacity(0.1) : Color.green.opacity(0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
            }

            if !server.discoveredTools.isEmpty {
                Button {
                    onToggleExpand()
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                            .font(.caption2)
                        Text(isExpanded ? "Hide Discovered Tools" : "Show Discovered Tools (\(server.discoveredTools.count))")
                            .font(.caption2.weight(.medium))
                    }
                    .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)

                if isExpanded {
                    VStack(alignment: .leading, spacing: 4) {
                        ForEach(server.discoveredTools) { tool in
                            HStack(alignment: .top, spacing: 6) {
                                Image(systemName: "wrench.and.screwdriver")
                                    .font(.caption2)
                                    .foregroundStyle(TurboSparkTheme.accentColor)
                                    .padding(.top, 2)
                                VStack(alignment: .leading, spacing: 1) {
                                    Text(tool.name)
                                        .font(.caption.monospaced().weight(.semibold))
                                    Text(tool.description)
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                            }
                            .padding(.vertical, 2)
                        }
                    }
                    .padding(8)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                }
            }
        }
        .padding(12)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.15), lineWidth: 0.5))
    }
}
