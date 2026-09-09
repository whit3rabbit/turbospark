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
        VStack(spacing: 8) {
            Image(systemName: "clock.arrow.circlepath")
                .themedFont(.hero)
                .foregroundStyle(.tertiary)
                .padding(.top, 40)
            Text("No Commit History Found", bundle: .module)
                .themedFont(.base, weight: .medium)
                .foregroundStyle(.secondary)
            Text("Commit your changes to view the project timeline.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func timelineNodeRow(commit: WorktreeCommit, isLast: Bool) -> some View {
        let isSelected = worktree.selectedTimelineCommit?.hash == commit.hash

        return HStack(alignment: .top, spacing: 10) {
            // Visual timeline node and connector rail
            VStack(spacing: 0) {
                Circle()
                    .fill(isSelected ? TurboSparkTheme.accentColor : Color.secondary.opacity(0.6))
                    .frame(width: 9, height: 9)
                    .padding(.top, 6)

                if !isLast {
                    Rectangle()
                        .fill(Color.primary.opacity(0.1))
                        .frame(width: 1.5)
                        .frame(maxHeight: .infinity)
                }
            }
            .frame(width: 12)

            // Commit Card
            VStack(alignment: .leading, spacing: 4) {
                Button {
                    withAnimation(.easeInOut(duration: 0.15)) {
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
                                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .primary)
                                .lineLimit(2)

                            Spacer(minLength: 4)

                            Text(commit.shortHash)
                                .font(theme.code(.small))
                                .foregroundStyle(.tertiary)
                                .padding(.horizontal, 5)
                                .padding(.vertical, 1)
                                .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 4))
                        }

                        HStack(spacing: 6) {
                            Text(commit.author)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)

                            Text("-", bundle: .module)
                                .foregroundStyle(.tertiary)

                            Text(commit.relativeDate)
                                .themedFont(.tiny)
                                .foregroundStyle(.tertiary)
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .background(
                    isSelected ? TurboSparkTheme.accentColor.opacity(0.1) : Color.primary.opacity(0.02),
                    in: RoundedRectangle(cornerRadius: 6)
                )

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
                    .foregroundStyle(.secondary)

                Spacer()

                if worktree.isLoadingCommitFiles {
                    ProgressView().controlSize(.mini)
                } else {
                    Text("\(worktree.selectedCommitFiles.count) files", bundle: .module)
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.tertiary)
                }
            }

            if worktree.isLoadingCommitFiles {
                HStack(spacing: 6) {
                    ProgressView().controlSize(.mini)
                    Text("Loading commit files...", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                }
                .padding(.vertical, 4)
            } else if worktree.selectedCommitFiles.isEmpty {
                Text("No modified files recorded.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            } else {
                VStack(spacing: 2) {
                    ForEach(worktree.selectedCommitFiles) { file in
                        commitFileRow(file)
                    }
                }
            }
        }
        .padding(8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color.primary.opacity(0.08), lineWidth: 1)
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
                    .frame(width: 12)

                Text(file.fileName)
                    .font(theme.code(.small, weight: isSelectedFile ? .semibold : .regular))
                    .foregroundStyle(isSelectedFile ? TurboSparkTheme.accentColor : .primary)
                    .lineLimit(1)

                if !file.directoryPath.isEmpty {
                    Text(file.directoryPath)
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }

                Spacer(minLength: 4)

                if file.additions > 0 || file.deletions > 0 {
                    HStack(spacing: 3) {
                        if file.additions > 0 {
                            Text("+\(file.additions)", bundle: .module)
                                .themedFont(.tiny).monospacedDigit()
                                .foregroundStyle(.green)
                        }
                        if file.deletions > 0 {
                            Text("-\(file.deletions)", bundle: .module)
                                .themedFont(.tiny).monospacedDigit()
                                .foregroundStyle(.red)
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
            isSelectedFile ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: RoundedRectangle(cornerRadius: 4)
        )
    }
}
