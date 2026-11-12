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
    @Environment(\.appTheme) private var theme
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
                    .foregroundStyle(.appAccent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("MCP Server Approval", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    Text("Project \"\(projectName(approval))\" declares MCP servers in its config files.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                if model.pendingMcpApprovals.count > 1 {
                    Text("Pending: \(model.pendingMcpApprovals.count)", bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(.appElevated)
                        .clipShape(Capsule())
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(approval.config.name)
                    .themedCode(.base, weight: .semibold)
                // The full argv, env, cwd and forwarded variables, never
                // truncated: this is what the approval actually runs.
                let summary = approval.config.approvalSummary
                ScrollView {
                    Text(verbatim: summary.text)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 140)
                if !summary.riskyEnvNames.isEmpty {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .themedFont(.micro)
                        Text(verbatim: "! " + summary.riskyEnvNames.joined(separator: ", "))
                            .themedCode(.small, weight: .semibold)
                    }
                    .foregroundStyle(AppAttention.color)
                }
                Text("Declared in \(approval.sourceRelativePath)", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(theme.border, lineWidth: 1)
            )

            Text("Approving runs this server's commands on this machine. Auto-approval stays off: tool calls still ask unless you opt in later.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)

            HStack(spacing: 10) {
                Button {
                    model.approvePendingMcpServer(id: approval.id, approveAllFuture: true)
                } label: { Text("Approve All Future", bundle: .module) }
                .appAction(.secondary)
                .help(Text("Approve this server and every server this project declares in the future without prompting", bundle: .module))

                Button {
                    model.rejectPendingMcpServer(id: approval.id)
                } label: { Text("Reject", bundle: .module) }
                .appAction(.secondary)
                .help(Text("Never import or run this server, and do not ask again", bundle: .module))

                Spacer()

                Button {
                    model.approvePendingMcpServer(id: approval.id)
                } label: { Text("Approve", bundle: .module) }
                .appAction(.primary)
            }
        }
        .padding(24)
        .frame(minWidth: 420, idealWidth: 460)
    }

    private func projectName(_ approval: PendingMcpServerApproval) -> String {
        model.projects.first(where: { $0.id == approval.projectID })?.name ?? AppStrings.localized("Unknown")
    }
}
