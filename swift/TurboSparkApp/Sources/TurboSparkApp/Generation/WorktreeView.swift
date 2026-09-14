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
                    withAnimation(.easeInOut(duration: 0.12)) {
                        worktree.activeTab = tab
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: tab.systemImage)
                            .themedFont(.tiny)
                        Text(tab.title)
                            .font(theme.ui(.tiny, weight: worktree.activeTab == tab ? .semibold : .regular))
                    }
                    .padding(.horizontal, 9)
                    .padding(.vertical, 4)
                    .foregroundStyle(worktree.activeTab == tab ? TurboSparkTheme.accentColor : .secondary)
                    .background(
                        worktree.activeTab == tab ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 5)
                    )
                }
                .buttonStyle(.plain)
            }
            Spacer()
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 4)
        .background(Color.primary.opacity(0.02))
    }

    // MARK: - Header matching Claude Code

    private var header: some View {
        HStack(spacing: 8) {
            if worktree.activeTab == .changes {
                Button {
                    withAnimation(.easeInOut(duration: 0.12)) {
                        worktree.viewMode = worktree.viewMode == .tree ? .flat : .tree
                    }
                } label: {
                    Image(systemName: worktree.viewMode.systemImage)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .frame(width: 22, height: 22)
                        .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
                .help("Toggle Tree / List view (\(worktree.viewMode == .tree ? "Flat List" : "Tree View"))")
            }

            branchMenu

            Spacer()

            if worktree.totalAdditions > 0 || worktree.totalDeletions > 0 {
                HStack(spacing: 4) {
                    if worktree.totalAdditions > 0 {
                        Text(verbatim: "+\(worktree.totalAdditions)")
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.green)
                    }
                    if worktree.totalDeletions > 0 {
                        Text(verbatim: "-\(worktree.totalDeletions)")
                            .themedFont(.tiny, weight: .semibold).monospacedDigit()
                            .foregroundStyle(.red)
                    }
                }
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(Color.primary.opacity(0.04), in: Capsule())
            }

            if worktree.activeTab == .changes {
                Button {
                    withAnimation(.easeInOut(duration: 0.12)) {
                        worktree.isSearchVisible.toggle()
                        if !worktree.isSearchVisible { worktree.searchQuery = "" }
                    }
                } label: {
                    Image(systemName: "magnifyingglass")
                        .themedFont(.small)
                        .foregroundStyle(worktree.isSearchVisible ? TurboSparkTheme.accentColor : .secondary)
                }
                .buttonStyle(.plain)
                .help("Filter changed files")
            }

            Button {
                worktree.refresh()
            } label: {
                Image(systemName: "arrow.clockwise")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .rotationEffect(worktree.isRefreshing ? .degrees(360) : .degrees(0))
                    .animation(
                        worktree.isRefreshing ? .linear(duration: 0.8).repeatForever(autoreverses: false) : .default,
                        value: worktree.isRefreshing
                    )
            }
            .buttonStyle(.plain)
            .help("Refresh Git changes")

            Button {
                withAnimation(.easeInOut(duration: 0.15)) {
                    worktree.isExpandedSplitMode.toggle()
                }
            } label: {
                Image(systemName: worktree.isExpandedSplitMode ? "rectangle.split.2x1.fill" : "rectangle.split.2x1")
                    .themedFont(.small)
                    .foregroundStyle(worktree.isExpandedSplitMode ? TurboSparkTheme.accentColor : .secondary)
            }
            .buttonStyle(.plain)
            .help(worktree.isExpandedSplitMode ? "Collapse split view" : "Expand into two-column diff view")

            Button {
                NotificationCenter.default.post(name: .toggleInspector, object: nil)
            } label: {
                Image(systemName: "xmark")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            .buttonStyle(.plain)
            .help("Close inspector")
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
                                Task { _ = await worktree.switchBranch(to: branch) }
                            } label: {
                                Text("Switch to \(branch)", bundle: .module)
                            }
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
                Text(comparisonLabel)
                    .font(theme.code(.base, weight: .semibold))
                    .foregroundStyle(.appText)
                    .lineLimit(1)

                Image(systemName: "chevron.down")
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            }
        }
        .menuStyle(.borderlessButton)
        .fixedSize()
    }

    private var comparisonLabel: String {
        switch worktree.comparisonMode {
        case .uncommitted: return "\(worktree.currentBranch) -> working tree"
        case .againstBranch(let target): return "\(target) -> working tree"
        case .commit(let hash, _): return "\(String(hash.prefix(7))) -> working tree"
        }
    }

    private var subtitleBar: some View {
        HStack {
            Text("Files are collapsed for large diffs. Select a file to expand it.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.tertiary)
            Spacer()
            Text("\(worktree.filteredFiles.count) \(worktree.filteredFiles.count == 1 ? "file" : "files")", bundle: .module)
                .themedFont(.tiny).monospacedDigit()
                .foregroundStyle(.appSecondary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(Color.primary.opacity(0.02))
    }

    // MARK: - Split Layout (Screenshot 4 Parity)

    private var splitLayout: some View {
        HStack(spacing: 0) {
            VStack(spacing: 0) {
                WorktreeFileListView(worktree: worktree, expandedDirectories: $expandedDirectories)
                Divider()
                WorktreeCommitListView(worktree: worktree, isExpanded: $isCommitsExpanded)
            }
            .frame(width: 270)
            .frame(maxHeight: .infinity)

            Divider()

            diffSidePane
        }
    }

    private var timelineSplitLayout: some View {
        HStack(spacing: 0) {
            WorktreeTimelineView(worktree: worktree)
                .frame(width: 320)
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
                        ProgressView().controlSize(.small)
                        Text("Loading diff for \(file.fileName)...", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
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
                        .foregroundStyle(.tertiary)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }
            } else {
                VStack(spacing: 8) {
                    Image(systemName: "doc.text.magnifyingglass")
                        .themedFont(.hero)
                        .foregroundStyle(.tertiary)
                    Text("Select a file to inspect diff", bundle: .module)
                        .themedFont(.base, weight: .medium)
                        .foregroundStyle(.appSecondary)
                    Text("Click any file to view modifications.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.tertiary)
                }
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
                        ProgressView().controlSize(.mini)
                        Text("Loading diff...", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
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
        VStack(spacing: 12) {
            Image(systemName: "point.topleft.down.to.point.bottomright.curvepath")
                .themedFont(.hero)
                .foregroundStyle(.appAccent.opacity(0.8))
                .padding(.top, 40)

            Text("No Git Repository Detected", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .foregroundStyle(.appText)

            Text("The current workspace folder is not tracked by Git. Initialize a repository to enable version control, change inspection, and diffs.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 24)

            Button {
                worktree.initRepository()
            } label: {
                HStack(spacing: 6) {
                    Image(systemName: "plus.circle.fill")
                    Text("Initialize Git Repository", bundle: .module)
                }
                .themedFont(.base, weight: .medium)
                .padding(.horizontal, 14)
                .padding(.vertical, 7)
            }
            .buttonStyle(.borderedProminent)
            .tint(TurboSparkTheme.accentColor)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
    }
}
