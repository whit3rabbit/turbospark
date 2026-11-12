import AppKit
import SwiftUI

/// Collapsible drawer displaying recent repository commits ("All changes").
@MainActor
public struct WorktreeCommitListView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel
    @Binding var isExpanded: Bool

    public init(worktree: WorktreeModel, isExpanded: Binding<Bool>) {
        self.worktree = worktree
        self._isExpanded = isExpanded
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            if isExpanded {
                Divider()
                commitList
            }
        }
        .background(.appSurface)
    }

    private var header: some View {
        Button {
            TSMotion.animate(TSMotion.select) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)

                Text("All changes", bundle: .module)
                    .font(theme.ui(.small, weight: .semibold))
                    .foregroundStyle(.appText)

                Spacer()

                if !worktree.recentCommits.isEmpty {
                    Text(verbatim: "\(worktree.recentCommits.count)")
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(.appElevated, in: Capsule())
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }

    private var commitList: some View {
        Group {
            if worktree.recentCommits.isEmpty {
                Text("No recent commits found.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(10)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(worktree.recentCommits) { commit in
                            commitRow(commit)
                        }
                    }
                    .padding(.horizontal, 6)
                    .padding(.vertical, 4)
                }
                .frame(maxHeight: 180)
            }
        }
    }

    private func commitRow(_ commit: WorktreeCommit) -> some View {
        let isSelected = worktree.selectedCommit?.hash == commit.hash
        return Button {
            worktree.selectedCommit = isSelected ? nil : commit
            if !isSelected {
                worktree.setComparisonMode(.commit(hash: commit.hash, summary: commit.summary))
            } else {
                worktree.setComparisonMode(.uncommitted)
            }
        } label: {
            VStack(alignment: .leading, spacing: 2) {
                Text(commit.summary)
                    .font(theme.ui(.tiny, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appText))
                    .lineLimit(1)
                    .truncationMode(.tail)
                    .help(commit.summary)

                HStack(spacing: 6) {
                    Text(commit.shortHash)
                        .font(theme.code(.small))
                        .foregroundStyle(.appSecondary)

                    Text(verbatim: "-")
                        .foregroundStyle(.appSecondary)

                    Text(commit.author)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)

                    Text(verbatim: "-")
                        .foregroundStyle(.appSecondary)

                    Text(commit.relativeDate)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .background(
            isSelected ? AnyShapeStyle(theme.accent.opacity(0.12)) : AnyShapeStyle(Color.clear),
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
    }
}
