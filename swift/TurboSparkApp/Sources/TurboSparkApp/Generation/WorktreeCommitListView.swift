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
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.4))
    }

    private var header: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                isExpanded.toggle()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
                    .frame(width: 12)

                Text("All changes", bundle: .module)
                    .font(theme.ui(.small, weight: .semibold))
                    .foregroundStyle(.primary)

                Spacer()

                if !worktree.recentCommits.isEmpty {
                    Text("\(worktree.recentCommits.count)", bundle: .module)
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.tertiary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 1)
                        .background(Color.primary.opacity(0.04), in: Capsule())
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
                    .foregroundStyle(.tertiary)
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
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .primary)
                    .lineLimit(1)
                    .truncationMode(.tail)

                HStack(spacing: 6) {
                    Text(commit.shortHash)
                        .font(theme.code(.small))
                        .foregroundStyle(.tertiary)

                    Text("-", bundle: .module)
                        .foregroundStyle(.tertiary)

                    Text(commit.author)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)

                    Text("-", bundle: .module)
                        .foregroundStyle(.tertiary)

                    Text(commit.relativeDate)
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .frame(maxWidth: .infinity, alignment: .leading)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: RoundedRectangle(cornerRadius: 6)
        )
    }
}
