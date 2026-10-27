import AppKit
import SwiftUI

// Isolated explicitly: only body is isolated by the protocol on the macOS 14 SDK.
/// Right-hand working tree and git diff inspector matching modern coding agent environments.
@MainActor
struct WorktreeView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var worktree: WorktreeModel

    @State private var expandedDirectories: Set<String> = []
    @State private var isCommitsExpanded: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            tabBar
            Divider()

            if worktree.activeTab == .changes {
                subtitleBar
                Divider()
            }

            if !worktree.isGitRepository {
                nonGitRepositoryCard
            } else {
                tabContent
            }
        }
        .background(.appPage)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .onAppear {
            seedExpandedDirectories()
        }
        .onChange(of: worktree.files) {
            if expandedDirectories.isEmpty {
                seedExpandedDirectories()
            }
        }
    }

    private func seedExpandedDirectories() {
        if expandedDirectories.isEmpty {
            let topLevel = worktree.treeNodes.filter(\.isDirectory).map(\.relativePath)
            if !topLevel.isEmpty {
                expandedDirectories = Set(topLevel)
            }
        }
    }

    @ViewBuilder
    private var tabContent: some View {
        switch worktree.activeTab {
        case .changes:
            if worktree.isExpandedSplitMode {
                splitLayout
            } else {
                compactLayout
            }
        case .timeline:
            if worktree.isExpandedSplitMode {
                timelineSplitLayout
            } else {
                WorktreeTimelineView(worktree: worktree)
            }
        case .worktrees:
            WorktreeWorktreeListView(worktree: worktree)
        }
    }

    // MARK: - Tab Bar (Changes / Timeline / Worktrees)

    private var tabBar: some View {
        HStack(spacing: 4) {
            ForEach(WorktreeTabMode.allCases, id: \.self) { tab in
                Button {
                    TSMotion.animate(TSMotion.select) {
                        worktree.activeTab = tab
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: tab.systemImage)
                            .themedFont(.tiny)
                        Text(LocalizedStringKey(tab.title), bundle: .module)
                            .font(theme.ui(.tiny, weight: worktree.activeTab == tab ? .semibold : .regular))
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .foregroundStyle(worktree.activeTab == tab ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
                    .background(
                        worktree.activeTab == tab ? theme.accent.opacity(0.12) : Color.clear,
                        in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    )
                }
                .buttonStyle(.plain)
                .accessibilityAddTraits(worktree.activeTab == tab ? [.isSelected] : [])
            }
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(.appSurface)
    }

    // MARK: - Header matching Claude Code

    private var header: some View {
        HStack(spacing: 8) {
            if worktree.activeTab == .changes {
                Button {
                    TSMotion.animate(TSMotion.select) {
                        worktree.viewMode = worktree.viewMode == .tree ? .flat : .tree
                    }
                } label: {
                    Image(systemName: worktree.viewMode.systemImage)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help(worktree.viewMode == .tree
                    ? Text("Show as flat list", bundle: .module)
                    : Text("Show as tree", bundle: .module))
            }

            branchMenu

            Spacer()

            if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                HStack(spacing: 4) {
                    if worktree.totalAdditions > 0 {
                        Text(verbatim: "+\(worktree.totalAdditions)")
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.appText)
                    }
                    if worktree.totalDeletions > 0 {
                        Text(verbatim: "-\(worktree.totalDeletions)")
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(.appElevated, in: Capsule())
            }

            if worktree.activeTab == .changes {
                Button {
                    TSMotion.animate(TSMotion.select) {
                        worktree.isSearchVisible.toggle()
                        if !worktree.isSearchVisible { worktree.searchQuery = "" }
                    }
                } label: {
                    Image(systemName: "magnifyingglass")
                        .themedFont(.small)
                        .foregroundStyle(worktree.isSearchVisible ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
                }
                .buttonStyle(.plain)
                .help(Text("Filter changed files", bundle: .module))
            }

            Button {
                worktree.refresh()
            } label: {
                Image(systemName: "arrow.clockwise")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .tsPulse(active: worktree.isRefreshing)
            }
            .buttonStyle(.plain)
            .help(Text("Refresh Git changes", bundle: .module))

            Button {
                TSMotion.animate(TSMotion.pane) {
                    worktree.isExpandedSplitMode.toggle()
                }
            } label: {
                Image(systemName: worktree.isExpandedSplitMode ? "rectangle.split.2x1.fill" : "rectangle.split.2x1")
                    .themedFont(.small)
                    .foregroundStyle(worktree.isExpandedSplitMode ? AnyShapeStyle(theme.accent) : AnyShapeStyle(.appSecondary))
            }
            .buttonStyle(.plain)
            .help(worktree.isExpandedSplitMode ? Text("Collapse split view", bundle: .module) : Text("Expand into two-column diff view", bundle: .module))

            Button {
                NotificationCenter.default.post(name: .toggleInspector, object: nil)
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            .buttonStyle(.plain)
            .help(Text("Close inspector", bundle: .module))
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var branchMenu: some View {
        Menu {
            Section(header: Text("Comparison Target", bundle: .module)) {
                Button {
                    worktree.setComparisonMode(.againstBranch(worktree.currentBranch))
                } label: {
                    HStack {
                        Text("All changes vs \(worktree.currentBranch)", bundle: .module)
                        if case .againstBranch(let b) = worktree.comparisonMode, b == worktree.currentBranch {
                            Image(systemName: "checkmark")
                        }
                    }
                }

                Button {
                    worktree.setComparisonMode(.uncommitted)
                } label: {
                    HStack {
                        Text("Uncommitted changes", bundle: .module)
                        if case .uncommitted = worktree.comparisonMode {
                            Image(systemName: "checkmark")
                        }
                    }
                }
            }

            if !worktree.availableBranches.isEmpty {
                Section(header: Text("Switch Branch", bundle: .module)) {
                    ForEach(worktree.availableBranches, id: \.self) { branch in
                        if branch != worktree.currentBranch {
                            Button {
                                Task {
                                    let result = await worktree.switchBranch(to: branch)
                                    if !result.success {
                                        model.showToast(
                                            result.error ?? AppStrings.localized("Failed to switch branch."), style: .error)
                                    }
                                }
                            } label: {
                                Text("Switch to \(branch)", bundle: .module)
                            }
                            // A checkout swaps file contents under the agent
                            // mid-edit, so mirror the workspace-root guard.
                            .disabled(model.generating || model.submitting || model.pendingToolCall != nil)
                        }
                    }
                }

                Menu {
                    ForEach(worktree.availableBranches, id: \.self) { branch in
                        Button {
                            worktree.setComparisonMode(.againstBranch(branch))
                        } label: {
                            HStack {
                                Text(branch)
                                if case .againstBranch(let b) = worktree.comparisonMode, b == branch {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    Text("Compare against...", bundle: .module)
                }
            }

            if !worktree.recentCommits.isEmpty {
                Menu {
                    ForEach(worktree.recentCommits) { commit in
                        Button {
                            worktree.setComparisonMode(.commit(hash: commit.hash, summary: commit.summary))
                        } label: {
                            Text(verbatim: "\(commit.shortHash): \(commit.summary)")
                        }
                    }
                } label: {
                    Text("Commits >", bundle: .module)
                }
            }
        } label: {
            HStack(spacing: 4) {
                Text("\(comparisonTarget) -> working tree", bundle: .module)
                    .font(theme.code(.base, weight: .semibold))
                    .foregroundStyle(.appText)
                    .lineLimit(1)

                Image(systemName: "chevron.down")
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
        .help(Text("Select comparison branch or commit", bundle: .module))
        .accessibilityLabel(Text("Git comparison target", bundle: .module))
    }

    private var comparisonTarget: String {
        switch worktree.comparisonMode {
        case .uncommitted: return worktree.currentBranch
        case .againstBranch(let target): return target
        case .commit(let hash, _): return String(hash.prefix(7))
        }
    }

    private var subtitleBar: some View {
        HStack {
            Text("Files are collapsed for large diffs. Select a file to expand it.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            Spacer()
            if worktree.filteredFiles.count == 1 {
                Text("1 file", bundle: .module)
                    .themedFont(.tiny).monospacedDigit()
                    .foregroundStyle(.appSecondary)
            } else {
                Text("\(worktree.filteredFiles.count) files", bundle: .module)
                    .themedFont(.tiny).monospacedDigit()
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.appSurface)
    }

    // MARK: - Split Layout (Screenshot 4 Parity)

    private var splitLayout: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                WorktreeFileListView(worktree: worktree, expandedDirectories: $expandedDirectories)
                Divider()
                WorktreeCommitListView(worktree: worktree, isExpanded: $isCommitsExpanded)
            }
            .frame(minWidth: 240, idealWidth: 270 * AppMetrics.textScale(theme), maxWidth: 360)
            .frame(maxHeight: .infinity)

            Divider()

            diffSidePane
        }
    }

    private var timelineSplitLayout: some View {
        HStack(spacing: 0) {
            WorktreeTimelineView(worktree: worktree)
                .frame(minWidth: 260, idealWidth: 320 * AppMetrics.textScale(theme), maxWidth: 420)
                .frame(maxHeight: .infinity)

            Divider()

            diffSidePane
        }
    }

    private var diffSidePane: some View {
        Group {
            if let selectedPath = worktree.selectedFilePath,
               let file = worktree.scopedFiles.first(where: { $0.relativePath == selectedPath }) ?? worktree.files.first(where: { $0.relativePath == selectedPath }) {
                if worktree.isLoadingDiff {
                    VStack(spacing: 8) {
                        AppProgressBar(
                            value: nil,
                            label: Text("Loading diff for \(file.fileName)...", bundle: .module),
                            size: .compact
                        )
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120)
                        Text("Loading diff for \(file.fileName)...", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                    }
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else if let diff = worktree.selectedFileDiff {
                    WorktreeDiffView(
                        worktree: worktree,
                        file: file,
                        diff: diff,
                        onClose: { worktree.selectedFilePath = nil }
                    )
                    // Keyed on the file so the fold state inside the diff
                    // does not survive a switch to a different file: block
                    // ids are sequential per parse, so carried-over state
                    // would open fold #2 of file B because it was open in
                    // file A.
                    .id(file.relativePath)
                    .padding(8)
                } else {
                    Text("No diff available.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                AppEmptyState(
                    systemImage: "doc.text.magnifyingglass",
                    title: Text("Select a file to inspect diff", bundle: .module),
                    message: Text("Click any file to view modifications.", bundle: .module)
                )
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var compactLayout: some View {
        VStack(spacing: 0) {
            WorktreeFileListView(worktree: worktree, expandedDirectories: $expandedDirectories)

            if let selectedPath = worktree.selectedFilePath,
               let file = worktree.scopedFiles.first(where: { $0.relativePath == selectedPath }) ?? worktree.files.first(where: { $0.relativePath == selectedPath }) {
                Divider()
                if worktree.isLoadingDiff {
                    HStack(spacing: 6) {
                        AppProgressBar(
                            value: nil,
                            label: Text("Loading diff...", bundle: .module),
                            size: .compact
                        )
                        .frame(minWidth: 40, idealWidth: 80, maxWidth: 120)
                        Text("Loading diff...", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                    }
                    .padding(8)
                } else if let diff = worktree.selectedFileDiff {
                    WorktreeDiffView(
                        worktree: worktree,
                        file: file,
                        diff: diff,
                        onClose: { worktree.selectedFilePath = nil }
                    )
                    // Same fold-state reason as the split pane's copy.
                    .id(file.relativePath)
                    .frame(maxHeight: 260)
                    .padding(6)
                }
            }

            Divider()
            WorktreeCommitListView(worktree: worktree, isExpanded: $isCommitsExpanded)
        }
    }

    private var nonGitRepositoryCard: some View {
        AppEmptyState(
            systemImage: "point.topleft.down.to.point.bottomright.curvepath",
            title: Text("No Git Repository Detected", bundle: .module),
            message: Text("The current workspace folder is not tracked by Git. Initialize a repository to enable version control, change inspection, and diffs.", bundle: .module)
        ) {
            Button {
                worktree.initRepository()
            } label: {
                Label {
                    Text("Initialize Git Repository", bundle: .module)
                } icon: {
                    Image(systemName: "plus.circle.fill")
                }
            }
            .appAction(.primary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .padding(.top, 40)
    }
}
