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
        HStack {
            Text("Linked Worktrees", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appText)

            Spacer()

            Text("\(worktree.worktrees.count) worktrees", bundle: .module)
                .themedFont(.tiny).monospacedDigit()
                .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.primary.opacity(0.02))
    }

    private var emptyWorktreesView: some View {
        VStack(spacing: 8) {
            Image(systemName: "folder.badge.gearshape")
                .themedFont(.hero)
                .foregroundStyle(.tertiary)
                .padding(.top, 40)
            Text("No Additional Worktrees", bundle: .module)
                .themedFont(.base, weight: .medium)
                .foregroundStyle(.appSecondary)
            Text("This repository only has the primary working directory.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
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
                        .foregroundStyle(.white)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(.appAccent, in: Capsule())
                }

                Spacer()

                Text(wt.head)
                    .font(theme.code(.small))
                    .foregroundStyle(.tertiary)
            }

            Text(wt.path)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .lineLimit(1)
                .truncationMode(.middle)

            if !wt.isCurrent {
                HStack {
                    Spacer()
                    Button {
                        worktree.selectWorktree(path: wt.path)
                    } label: {
                        Text("Switch to this worktree", bundle: .module)
                            .themedFont(.tiny, weight: .medium)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.mini)
                }
                .padding(.top, 2)
            }
        }
        .padding(8)
        .background(
            wt.isCurrent ? TurboSparkTheme.accentColor.opacity(0.08) : Color(nsColor: .controlBackgroundColor).opacity(0.5),
            in: RoundedRectangle(cornerRadius: 6)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(wt.isCurrent ? TurboSparkTheme.accentColor.opacity(0.3) : Color.primary.opacity(0.08), lineWidth: 1)
        )
    }
}
