import AppKit
import SwiftUI

/// Interactive commit history timeline with node graph visualization and commit-scoped file inspection.
@MainActor
public struct WorktreeTimelineView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel

    public init(worktree: WorktreeModel) {
        self.worktree = worktree
    }

    public var body: some View {
        VStack(spacing: 0) {
            if worktree.recentCommits.isEmpty {
                emptyTimelineView
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 0) {
                        ForEach(worktree.recentCommits.indices, id: \.self) { index in
                            let commit = worktree.recentCommits[index]
                            let isLast = index == worktree.recentCommits.count - 1
                            timelineNodeRow(commit: commit, isLast: isLast)
                        }
                    }
                    .padding(.vertical, 8)
                    .padding(.horizontal, 10)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var emptyTimelineView: some View {
        AppEmptyState(
            systemImage: "clock.arrow.circlepath",
            title: Text("No Commit History Found", bundle: .module),
            message: Text("Commit your changes to view the project timeline.", bundle: .module)
        )
    }

    private func timelineNodeRow(commit: WorktreeCommit, isLast: Bool) -> some View {
        let isSelected = worktree.selectedTimelineCommit?.hash == commit.hash

        return HStack(alignment: .top, spacing: 10) {
            // Visual timeline node and connector rail
            VStack(spacing: 0) {
                Circle()
                    .fill(isSelected ? theme.accent : theme.secondaryText.opacity(0.6))
                    .frame(width: 9, height: 9)
                    .padding(.top, 6)

                if !isLast {
                    Rectangle()
                        .fill(.appBorder)
                        .frame(width: 1.5)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(width: 12)

            // Commit Card
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    TSMotion.animate(TSMotion.select) {
                        if isSelected {
                            worktree.deselectTimelineCommit()
                        } else {
                            worktree.selectTimelineCommit(commit)
                        }
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Text(commit.summary)
                                .font(theme.ui(.small, weight: isSelected ? .semibold : .medium))
                                .foregroundStyle(isSelected ? theme.accent : theme.foreground)
                                .lineLimit(2)

                            Spacer(minLength: 4)

                            Text(commit.shortHash)
                                .font(theme.code(.small))
                                .foregroundStyle(.appSecondary)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                        }

                        HStack(spacing: 6) {
                            Text(commit.author)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)

                            Text(verbatim: "-")
                                .foregroundStyle(.appSecondary)

                            Text(commit.relativeDate)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(
                    isSelected ? theme.selection : theme.surface,
                    in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                )
                .accessibilityAddTraits(isSelected ? [.isSelected] : [])

                // Expanded commit details and file list
                if isSelected {
                    commitDetailCard(commit: commit)
                        .padding(.top, 2)
                        .padding(.bottom, 6)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(.bottom, 6)
    }

    private func commitDetailCard(commit: WorktreeCommit) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text("Files changed in this commit:", bundle: .module)
                    .themedFont(.tiny, weight: .medium)
                    .foregroundStyle(.appSecondary)

                Spacer()

                if worktree.isLoadingCommitFiles {
                    AppProgressBar(value: nil, label: Text("Loading commit files...", bundle: .module), size: .compact)
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120)
                } else {
                    Text("\(worktree.selectedCommitFiles.count) files", bundle: .module)
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
            }

            if worktree.isLoadingCommitFiles {
                HStack(spacing: 6) {
                    AppProgressBar(value: nil, label: Text("Loading commit files...", bundle: .module), size: .compact)
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120)
                    Text("Loading commit files...", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                .padding(.vertical, 4)
            } else if worktree.selectedCommitFiles.isEmpty {
                Text("No modified files recorded.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            } else {
                VStack(spacing: 2) {
                    ForEach(worktree.selectedCommitFiles) { file in
                        commitFileRow(file)
                    }
                }
            }
        }
        .padding(8)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 1)
        )
    }

    private func commitFileRow(_ file: WorktreeFileChange) -> some View {
        let isSelectedFile = worktree.selectedFilePath == file.relativePath
        return Button {
            worktree.loadDiff(for: file.relativePath)
        } label: {
            HStack(spacing: 5) {
                Image(systemName: file.status.systemImage)
                    .themedFont(.tiny)
                    .foregroundStyle(file.status.color)
                    .accessibilityHidden(true)

                Text(file.fileName)
                    .font(theme.code(.small, weight: isSelectedFile ? .semibold : .regular))
                    .foregroundStyle(isSelectedFile ? theme.accent : theme.foreground)
                    .lineLimit(1)

                if !file.directoryPath.isEmpty {
                    Text(file.directoryPath)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }

                Spacer(minLength: 4)

                if file.additions > 0 || file.deletions > 0 {
                    HStack(spacing: 3) {
                        if file.additions > 0 {
                            Text(verbatim: "+\(file.additions)")
                                .themedFont(.tiny).monospacedDigit()
                                .foregroundStyle(.green)
                        }
                        if file.deletions > 0 {
                            Text(verbatim: "-\(file.deletions)")
                                .themedFont(.tiny).monospacedDigit()
                                .foregroundStyle(AppAttention.color)
                        }
                    }
                }
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 3)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            isSelectedFile ? theme.selection : Color.clear,
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .accessibilityAddTraits(isSelectedFile ? [.isSelected] : [])
    }
}
