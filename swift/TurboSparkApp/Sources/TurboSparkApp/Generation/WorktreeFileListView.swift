import AppKit
import SwiftUI

/// Component displaying changed files in either Hierarchical Tree or Flat List mode, with search filtering.
@MainActor
public struct WorktreeFileListView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var worktree: WorktreeModel
    @Binding var expandedDirectories: Set<String>

    public init(worktree: WorktreeModel, expandedDirectories: Binding<Set<String>>) {
        self.worktree = worktree
        self._expandedDirectories = expandedDirectories
    }

    public var body: some View {
        VStack(spacing: 0) {
            if worktree.isSearchVisible {
                searchBar
                Divider()
            }

            if worktree.filteredFiles.isEmpty {
                emptySearchResults
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 2) {
                        switch worktree.viewMode {
                        case .tree:
                            ForEach(worktree.treeNodes) { node in
                                WorktreeTreeNodeRow(
                                    worktree: worktree,
                                    node: node,
                                    depth: 0,
                                    expandedDirectories: $expandedDirectories
                                )
                            }
                        case .flat:
                            ForEach(worktree.filteredFiles) { file in
                                flatFileRow(file)
                            }
                        }
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 8)
                }
            }
        }
    }

    private var searchBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.caption2)
                .foregroundStyle(.secondary)

            TextField("Filter changed files...", text: $worktree.searchQuery)
                .textFieldStyle(.plain)
                .font(theme.ui(points: 12))

            if !worktree.searchQuery.isEmpty {
                Button {
                    worktree.searchQuery = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color.primary.opacity(0.03))
    }

    private var emptySearchResults: some View {
        VStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(.title3)
                .foregroundStyle(.tertiary)
                .padding(.top, 24)
            Text("No matching changed files")
                .font(.callout.weight(.medium))
                .foregroundStyle(.secondary)
            Text("Try clearing the search filter.")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .frame(maxWidth: .infinity, alignment: .center)
        .padding(.vertical, 20)
    }

    // MARK: - Flat List View

    private func flatFileRow(_ file: WorktreeFileChange) -> some View {
        let isSelected = worktree.selectedFilePath == file.relativePath
        return Button {
            worktree.loadDiff(for: file.relativePath)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: file.status.systemImage)
                    .font(.caption2)
                    .foregroundStyle(file.status.color)

                Text(file.fileName)
                    .font(.callout.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .primary)
                    .lineLimit(1)

                if !file.directoryPath.isEmpty {
                    Text(file.directoryPath)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                        .truncationMode(.head)
                }

                Spacer(minLength: 4)

                statPills(adds: file.additions, dels: file.deletions)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: RoundedRectangle(cornerRadius: 6)
        )
    }

    private func statPills(adds: Int, dels: Int) -> some View {
        HStack(spacing: 4) {
            if adds > 0 {
                Text("+\(adds)")
                    .font(.caption2.monospacedDigit().weight(.medium))
                    .foregroundStyle(.green)
            }
            if dels > 0 {
                Text("-\(dels)")
                    .font(.caption2.monospacedDigit().weight(.medium))
                    .foregroundStyle(.red)
            }
        }
    }
}

/// Recursive row component rendering a directory or file item in the working tree hierarchy.
@MainActor
struct WorktreeTreeNodeRow: View {
    @ObservedObject var worktree: WorktreeModel
    let node: WorktreeTreeNode
    let depth: Int
    @Binding var expandedDirectories: Set<String>

    private var isSearching: Bool {
        !worktree.searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private var isNodeExpanded: Bool {
        expandedDirectories.contains(node.relativePath) || isSearching
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            if node.isDirectory {
                directoryRow(node, depth: depth)
                if isNodeExpanded {
                    ForEach(node.children) { child in
                        WorktreeTreeNodeRow(
                            worktree: worktree,
                            node: child,
                            depth: depth + 1,
                            expandedDirectories: $expandedDirectories
                        )
                    }
                }
            } else if let file = node.fileChange {
                leafFileRow(file, depth: depth)
            }
        }
    }

    private func directoryRow(_ node: WorktreeTreeNode, depth: Int) -> some View {
        let isExpanded = isNodeExpanded
        return Button {
            withAnimation(.easeInOut(duration: 0.12)) {
                if expandedDirectories.contains(node.relativePath) {
                    expandedDirectories.remove(node.relativePath)
                } else {
                    expandedDirectories.insert(node.relativePath)
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .frame(width: 12)

                Image(systemName: isExpanded ? "folder.fill" : "folder")
                    .font(.caption)
                    .foregroundStyle(TurboSparkTheme.accentColor.opacity(0.85))

                Text(node.name)
                    .font(.callout.weight(.medium))
                    .foregroundStyle(.primary)
                    .lineLimit(1)

                Spacer(minLength: 4)

                if node.additions > 0 || node.deletions > 0 {
                    statPills(adds: node.additions, dels: node.deletions)
                }
            }
            .padding(.leading, CGFloat(depth * 14) + 6)
            .padding(.trailing, 8)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(Color.clear, in: RoundedRectangle(cornerRadius: 6))
    }

    private func leafFileRow(_ file: WorktreeFileChange, depth: Int) -> some View {
        let isSelected = worktree.selectedFilePath == file.relativePath
        return Button {
            worktree.loadDiff(for: file.relativePath)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: file.status.systemImage)
                    .font(.caption2)
                    .foregroundStyle(file.status.color)
                    .frame(width: 12)

                Text(file.fileName)
                    .font(.callout.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : .primary)
                    .lineLimit(1)

                Spacer(minLength: 4)

                statPills(adds: file.additions, dels: file.deletions)
            }
            .padding(.leading, CGFloat(depth * 14) + 18)
            .padding(.trailing, 8)
            .padding(.vertical, 4)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: RoundedRectangle(cornerRadius: 6)
        )
    }

    private func statPills(adds: Int, dels: Int) -> some View {
        HStack(spacing: 4) {
            if adds > 0 {
                Text("+\(adds)")
                    .font(.caption2.monospacedDigit().weight(.medium))
                    .foregroundStyle(.green)
            }
            if dels > 0 {
                Text("-\(dels)")
                    .font(.caption2.monospacedDigit().weight(.medium))
                    .foregroundStyle(.red)
            }
        }
    }
}
