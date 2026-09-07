import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Approval sheet for MCP servers declared by the selected project's own
/// config files (`.mcp.json` and siblings), shown over the root view when
/// detection finds names nobody has decided on.
///
/// This is the port of the reference implementation's startup approval
/// dialog: a repo-provided server is never dialed or advertised until the
/// user has answered for it once. Approve imports it (enabled, never
/// auto-approved), "Approve All" additionally covers future servers this
/// project declares, Reject records the name so it never prompts again.
/// Dismissing the sheet defers -- undecided names re-prompt on the next
/// project selection.
struct ProjectMcpApprovalSheet: View {
    @ObservedObject var model: AppModel

    private var pending: PendingMcpServerApproval? {
        model.pendingMcpApprovals.first
    }

    var body: some View {
        Group {
            if let approval = pending {
                content(for: approval)
            } else {
                EmptyView()
            }
        }
    }

    @ViewBuilder
    private func content(for approval: PendingMcpServerApproval) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: "server.rack")
                    .themedFont(.title2)
                    .foregroundStyle(TurboSparkTheme.accentColor)
                VStack(alignment: .leading, spacing: 2) {
                    Text("MCP Server Approval", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                    Text("Project \"\(projectName(approval))\" declares MCP servers in its config files.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if model.pendingMcpApprovals.count > 1 {
                    Text("\(model.pendingMcpApprovals.count) pending", bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(approval.config.name)
                    .themedCode(.base, weight: .semibold)
                Text(approval.config.commandSummary)
                    .themedCode(.small)
                    .foregroundStyle(.secondary)
                    .textSelection(.enabled)
                    .lineLimit(3)
                Text("Declared in \(approval.sourceRelativePath)", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(TurboSparkTheme.hairlineColor, lineWidth: 1))

            Text("Approving runs this server's commands on this machine. Auto-approval stays off: tool calls still ask unless you opt in later.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                Button("Approve All Future") {
                    model.approvePendingMcpServer(id: approval.id, approveAllFuture: true)
                }
                .buttonStyle(.bordered)
                .help("Approve this server and every server this project declares in the future without prompting")

                Button("Reject") {
                    model.rejectPendingMcpServer(id: approval.id)
                }
                .buttonStyle(.bordered)
                .help("Never import or run this server, and do not ask again")

                Spacer()

                Button("Approve") {
                    model.approvePendingMcpServer(id: approval.id)
                }
                .buttonStyle(.borderedProminent)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(width: 460)
    }

    private func projectName(_ approval: PendingMcpServerApproval) -> String {
        model.projects.first(where: { $0.id == approval.projectID })?.name ?? "Unknown"
    }
}
