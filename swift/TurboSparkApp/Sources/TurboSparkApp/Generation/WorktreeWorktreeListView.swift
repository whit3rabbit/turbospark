import AppKit
import SwiftUI

/// Component displaying and managing Git worktrees for the current repository.
@MainActor
public struct WorktreeWorktreeListView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel

    public init(worktree: WorktreeModel) {
        self.worktree = worktree
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar
            Divider()

            if worktree.worktrees.isEmpty {
                emptyWorktreesView
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 6) {
                        ForEach(worktree.worktrees) { wt in
                            worktreeCard(wt)
                        }
                    }
                    .padding(10)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var headerBar: some View {
        AppAdaptiveHeader {
            Text("Linked Worktrees", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appText)
                .accessibilityAddTraits(.isHeader)
        } actions: {
            let count = worktree.worktrees.count
            (count == 1
                ? Text("\(count) worktree", bundle: .module)
                : Text("\(count) worktrees", bundle: .module))
                .themedFont(.tiny).monospacedDigit()
                .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(.appSurface)
    }

    private var emptyWorktreesView: some View {
        AppEmptyState(
            systemImage: "folder.badge.gearshape",
            title: Text("No Additional Worktrees", bundle: .module),
            message: Text("This repository only has the primary working directory.", bundle: .module)
        )
    }

    private func worktreeCard(_ wt: GitWorktreeInfo) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                    .themedFont(.tiny)
                    .foregroundStyle(.appAccent)

                Text(wt.branch)
                    .font(theme.code(.small, weight: .semibold))
                    .foregroundStyle(.appText)

                if wt.isCurrent {
                    Text("Current", bundle: .module)
                        .themedFont(.tiny, weight: .medium)
                        .foregroundStyle(theme.background)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(.appAccent, in: Capsule())
                }

                Spacer()

                Text(wt.head)
                    .font(theme.code(.small))
                    .foregroundStyle(.appSecondary)
            }

            SettingsPathText(wt.path, size: .tiny)

            if !wt.isCurrent {
                HStack {
                    Spacer()
                    Button {
                        worktree.selectWorktree(path: wt.path)
                    } label: {
                        Text("View in Git pane", bundle: .module)
                            .themedFont(.tiny, weight: .medium)
                    }
                    .appAction(.secondary, size: .small)
                    // This only retargets the Git pane. The agent's workspace
                    // root, hooks and snapshots stay on the project folder, so
                    // the label must not read as "move the agent here".
                    .help(Text("Shows this worktree in the Git pane only. The agent keeps working in the project folder.", bundle: .module))
                }
                .padding(.top, 2)
            }
        }
        .padding(8)
        .background(
            wt.isCurrent ? AnyShapeStyle(theme.accent.opacity(0.08)) : AnyShapeStyle(.appSurface),
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(wt.isCurrent ? AnyShapeStyle(theme.accent) : AnyShapeStyle(theme.border), lineWidth: 1)
        )
    }
}
