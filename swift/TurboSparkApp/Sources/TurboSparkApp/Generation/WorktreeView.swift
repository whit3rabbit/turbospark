import AppKit
import SwiftUI

/// Right-hand working tree and git diff inspector matching modern coding agent environments.
struct WorktreeView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var worktree: WorktreeModel
    @State private var expandedFilePaths: Set<String> = []

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            subtitleBar
            Divider()
            fileList
        }
        .background(Color(nsColor: .windowBackgroundColor))
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var header: some View {
        HStack(spacing: 6) {
            Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                .font(.caption.weight(.semibold))
                .foregroundStyle(TurboSparkTheme.accentColor)
                .accessibilityHidden(true)

            Text(worktree.currentBranch)
                .font(theme.code(.large, weight: .semibold))
                .foregroundStyle(.primary)

            Image(systemName: "chevron.right")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)

            Text("working tree")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)

            Spacer()

            if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                HStack(spacing: 4) {
                    if worktree.totalAdditions > 0 {
                        Text("+\(worktree.totalAdditions)")
                            .font(.caption2.monospacedDigit().weight(.semibold))
                            .foregroundStyle(.green)
                    }
                    if worktree.totalDeletions > 0 {
                        Text("-\(worktree.totalDeletions)")
                            .font(.caption2.monospacedDigit().weight(.semibold))
                            .foregroundStyle(.red)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.primary.opacity(0.04), in: Capsule())
            }

            Button {
                worktree.refresh()
            } label: {
                Image(systemName: "arrow.clockwise")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .rotationEffect(worktree.isRefreshing ? .degrees(360) : .degrees(0))
                    .animation(worktree.isRefreshing ? .linear(duration: 0.8).repeatForever(autoreverses: false) : .default, value: worktree.isRefreshing)
            }
            .buttonStyle(.borderless)
            .help("Refresh working tree status")
            .accessibilityLabel("Refresh working tree")
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
    }

    private var subtitleBar: some View {
        HStack {
            Text("Files are collapsed for large diffs. Select a file to expand it.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
            Spacer()
            Text("\(worktree.files.count) \(worktree.files.count == 1 ? "file" : "files")")
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.02))
    }

    private var fileList: some View {
        Group {
            if worktree.files.isEmpty {
                emptyWorktree
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        ForEach(worktree.files) { file in
                            worktreeFileRow(file)
                        }
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 8)
                }
            }
        }
    }

    private var emptyWorktree: some View {
        VStack(spacing: 8) {
            Image(systemName: "checkmark.circle")
                .font(.system(size: 24))
                .foregroundStyle(.green.opacity(0.8))
                .padding(.top, 40)
            Text("Working tree clean")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Text("No modified, added, or deleted files in \(worktree.currentBranch).")
                .font(.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }

    private func worktreeFileRow(_ file: WorktreeFileChange) -> some View {
        let isExpanded = expandedFilePaths.contains(file.relativePath)

        return VStack(alignment: .leading, spacing: 0) {
            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    if isExpanded {
                        expandedFilePaths.remove(file.relativePath)
                    } else {
                        expandedFilePaths.insert(file.relativePath)
                        worktree.loadDiff(for: file.relativePath)
                    }
                }
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .frame(width: 12)

                    Image(systemName: file.status.systemImage)
                        .font(.caption2)
                        .foregroundStyle(file.status.color)

                    Text(file.fileName)
                        .font(.callout.weight(.medium))
                        .foregroundStyle(.primary)
                        .lineLimit(1)

                    if !file.directoryPath.isEmpty {
                        Text(file.directoryPath)
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                            .truncationMode(.head)
                    }

                    Spacer(minLength: 4)

                    if file.additions > 0 || file.deletions > 0 {
                        HStack(spacing: 4) {
                            if file.additions > 0 {
                                Text("+\(file.additions)")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.green)
                            }
                            if file.deletions > 0 {
                                Text("-\(file.deletions)")
                                    .font(.caption2.monospacedDigit())
                                    .foregroundStyle(.red)
                            }
                        }
                    } else {
                        Text(file.status.label)
                            .font(.caption2)
                            .foregroundStyle(file.status.color)
                    }
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 6)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .background(isExpanded ? Color.primary.opacity(0.04) : Color.clear, in: RoundedRectangle(cornerRadius: 6))

            if isExpanded {
                diffContainer(for: file)
                    .padding(.leading, 18)
                    .padding(.trailing, 4)
                    .padding(.vertical, 4)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
    }

    @ViewBuilder
    private func diffContainer(for file: WorktreeFileChange) -> some View {
        if worktree.isLoadingDiff && worktree.selectedFilePath == file.relativePath {
            HStack(spacing: 6) {
                TaskProgressFlameIcon(size: 14)
                Text("Loading diff...")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(8)
        } else if let diff = worktree.selectedFileDiff, worktree.selectedFilePath == file.relativePath {
            ScrollView(.horizontal, showsIndicators: true) {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(diff.components(separatedBy: .newlines).indices, id: \.self) { idx in
                        let line = diff.components(separatedBy: .newlines)[idx]
                        diffLine(line)
                    }
                }
                .padding(8)
            }
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 6, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 6, style: .continuous)
                    .stroke(Color.primary.opacity(0.08), lineWidth: 1)
            )
        } else {
            Text("Diff unavailable.")
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(6)
        }
    }

    private func diffLine(_ line: String) -> some View {
        let isAddition = line.hasPrefix("+") && !line.hasPrefix("+++")
        let isDeletion = line.hasPrefix("-") && !line.hasPrefix("---")
        let isHunk = line.hasPrefix("@@")

        let fgColor: Color = isAddition ? .green : (isDeletion ? .red : (isHunk ? TurboSparkTheme.accentColor : .primary))
        let bgColor: Color = isAddition ? Color.green.opacity(0.08) : (isDeletion ? Color.red.opacity(0.08) : (isHunk ? TurboSparkTheme.accentColor.opacity(0.06) : Color.clear))

        return Text(line.isEmpty ? " " : line)
            .font(theme.code(.small))
            .foregroundStyle(fgColor)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 4)
            .padding(.vertical, 1)
            .background(bgColor, in: RoundedRectangle(cornerRadius: 2))
    }
}
