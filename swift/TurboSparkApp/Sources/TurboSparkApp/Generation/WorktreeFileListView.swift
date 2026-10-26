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
        AppSearchField(
            text: $worktree.searchQuery,
            prompt: Text("Filter changed files...", bundle: .module),
            minWidth: 0, idealWidth: 240, maxWidth: .infinity)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
    }

    @ViewBuilder
    private var emptySearchResults: some View {
        // Two different empties: the filter matched nothing (the common
        // case while searching), and the repository simply has no changes.
        // Telling a clean worktree to "clear the search filter" it never
        // set is an instruction to do nothing.
        let hasQuery = !worktree.searchQuery
            .trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasQuery {
            AppEmptyState(
                systemImage: "magnifyingglass",
                title: Text("No matching changed files", bundle: .module),
                message: Text("Try clearing the search filter.", bundle: .module)
            )
        } else {
            AppEmptyState(
                systemImage: "checkmark.circle",
                title: Text("No changes", bundle: .module),
                message: Text("Files appear here when the selected comparison finds modifications.", bundle: .module)
            )
        }
    }

    // MARK: - Flat List View

    private func flatFileRow(_ file: WorktreeFileChange) -> some View {
        let isSelected = worktree.selectedFilePath == file.relativePath
        return Button {
            worktree.loadDiff(for: file.relativePath)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: file.status.systemImage)
                    .themedFont(.tiny)
                    .foregroundStyle(file.status.color)
                    .accessibilityHidden(true)

                Text(file.fileName)
                    .themedFont(.base, weight: isSelected ? .semibold : .regular)
                    .foregroundStyle(isSelected ? theme.accent : theme.foreground)
                    .lineLimit(1)

                if !file.directoryPath.isEmpty {
                    Text(file.directoryPath)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
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
            isSelected ? theme.selection : Color.clear,
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    private func statPills(adds: Int, dels: Int) -> some View {
        HStack(spacing: 4) {
            if adds > 0 {
                Text(verbatim: "+\(adds)")
                    .themedFont(.tiny, weight: .medium).monospacedDigit()
                    .foregroundStyle(.green)
            }
            if dels > 0 {
                Text(verbatim: "-\(dels)")
                    .themedFont(.tiny, weight: .medium).monospacedDigit()
                    .foregroundStyle(AppAttention.color)
            }
        }
    }
}

/// Recursive row component rendering a directory or file item in the working tree hierarchy.
@MainActor
struct WorktreeTreeNodeRow: View {
    @Environment(\.appTheme) private var theme
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
            TSMotion.animate(TSMotion.select) {
                if expandedDirectories.contains(node.relativePath) {
                    expandedDirectories.remove(node.relativePath)
                } else {
                    expandedDirectories.insert(node.relativePath)
                }
            }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: isExpanded ? "chevron.down" : "chevron.right")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)

                Image(systemName: isExpanded ? "folder.fill" : "folder")
                    .themedFont(.small)
                    .foregroundStyle(.appAccent.opacity(0.85))

                Text(node.name)
                    .themedFont(.base, weight: .medium)
                    .foregroundStyle(.appText)
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
        .background(Color.clear, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    private func leafFileRow(_ file: WorktreeFileChange, depth: Int) -> some View {
        let isSelected = worktree.selectedFilePath == file.relativePath
        return Button {
            worktree.loadDiff(for: file.relativePath)
        } label: {
            HStack(spacing: 6) {
                Image(systemName: file.status.systemImage)
                    .themedFont(.tiny)
                    .foregroundStyle(file.status.color)
                    .accessibilityHidden(true)

                Text(file.fileName)
                    .themedFont(.base, weight: isSelected ? .semibold : .regular)
                    .foregroundStyle(isSelected ? theme.accent : theme.foreground)
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
            isSelected ? theme.selection : Color.clear,
            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
        )
        .accessibilityAddTraits(isSelected ? [.isSelected] : [])
    }

    private func statPills(adds: Int, dels: Int) -> some View {
        HStack(spacing: 4) {
            if adds > 0 {
                Text(verbatim: "+\(adds)")
                    .themedFont(.tiny, weight: .medium).monospacedDigit()
                    .foregroundStyle(.green)
            }
            if dels > 0 {
                Text(verbatim: "-\(dels)")
                    .themedFont(.tiny, weight: .medium).monospacedDigit()
                    .foregroundStyle(AppAttention.color)
            }
        }
    }
}
