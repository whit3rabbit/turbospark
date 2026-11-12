import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Approval sheet for custom tools a project ships in `.turbospark/tools` or
/// `.agents/tools`. Such a tool runs commands with the user's privileges, so
/// it stays off until the user has read its complete definition and clicked
/// Approve for that tool. Reject is the default action (Return), Approve is
/// never the default, and there is no "approve all".
struct ProjectToolApprovalSheet: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    private var current: ProjectToolReview? {
        model.pendingProjectToolApprovals.first
    }

    var body: some View {
        Group {
            if let review = current {
                content(for: review)
            } else {
                EmptyView()
            }
        }
    }

    @ViewBuilder
    private func content(for review: ProjectToolReview) -> some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 10) {
                Image(systemName: "wrench.and.screwdriver")
                    .themedFont(.title2)
                    .foregroundStyle(.appAccent)
                VStack(alignment: .leading, spacing: 2) {
                    Text("Project Tool Approval", bundle: .module)
                        .themedFont(.base, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    Text("Review each tool in full before approving. An approved tool runs commands on this machine with your privileges. Rejecting leaves it off.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                if model.pendingProjectToolApprovals.count > 1 {
                    Text(verbatim: "\(model.pendingProjectToolApprovals.count)")
                        .themedFont(.small, weight: .medium)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 3)
                        .background(.appElevated)
                        .clipShape(Capsule())
                }
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(verbatim: review.name)
                    .themedCode(.base, weight: .semibold)
                    .textSelection(.enabled)
                HStack(spacing: 6) {
                    Text("Declared category", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Text(LocalizedStringKey(review.declaredCategory.label), bundle: .module)
                        .themedFont(.small, weight: .medium)
                }
                HStack(spacing: 6) {
                    Text("Effective category", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Text(LocalizedStringKey(review.effectiveCategory.label), bundle: .module)
                        .themedFont(.small, weight: .medium)
                }
                if review.categoryWasRaised {
                    HStack(spacing: 5) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .themedFont(.tiny)
                        Text("The declared category is weaker than what this tool really does, so the effective category is enforced instead.", bundle: .module)
                            .themedFont(.small)
                    }
                    .foregroundStyle(AppAttention.color)
                }
                kindLabel(review.executionKind)
                    .themedFont(.small, weight: .medium)

                Text("Full definition", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                // Never truncated and never line-limited: a long command is
                // exactly where something unwelcome hides. The scroll view
                // bounds the sheet height instead.
                ScrollView {
                    Text(verbatim: review.fullText)
                        .themedCode(.small)
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 240)
                .padding(8)
                .background(.appElevated)
                .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))

                if let source = review.sourcePath {
                    SettingsPathText(source, size: .tiny)
                }
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                    .stroke(theme.border, lineWidth: 1)
            )

            HStack(spacing: 10) {
                Spacer()
                Button {
                    model.approveProjectTool(review)
                } label: { Text("Approve", bundle: .module) }
                .appAction(.secondary)

                Button {
                    model.rejectProjectTool(review)
                } label: { Text("Reject", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(24)
        .frame(minWidth: 480, idealWidth: 560)
    }

    @ViewBuilder
    private func kindLabel(_ kind: CustomToolExecutionType) -> some View {
        switch kind {
        case .command: Text("Shell command", bundle: .module)
        case .script: Text("Script", bundle: .module)
        case .http: Text("HTTP request", bundle: .module)
        }
    }
}
