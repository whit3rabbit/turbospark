import AppKit
import Foundation
import SwiftUI

/// A file changed or tracked in the working tree.
public struct WorktreeFileChange: Identifiable, Hashable, Sendable {
    public enum Status: String, Sendable {
        case modified = "M"
        case added = "A"
        case deleted = "D"
        case renamed = "R"
        case untracked = "?"
        case ignored = "!"
        case unknown = " "

        public var label: String {
            switch self {
            case .modified: return "Modified"
            case .added: return "Added"
            case .deleted: return "Deleted"
            case .renamed: return "Renamed"
            case .untracked: return "Untracked"
            case .ignored: return "Ignored"
            case .unknown: return "Changed"
            }
        }

        public var color: Color {
            switch self {
            case .modified: return .orange
            case .added, .untracked: return .green
            case .deleted: return .red
            case .renamed: return .blue
            case .ignored, .unknown: return .secondary
            }
        }

        public var systemImage: String {
            switch self {
            case .modified: return "pencil.circle"
            case .added, .untracked: return "plus.circle"
            case .deleted: return "minus.circle"
            case .renamed: return "arrow.triangle.swap"
            case .ignored, .unknown: return "doc"
            }
        }
    }

    public var id: String { relativePath }
    public let relativePath: String
    public let fileName: String
    public let directoryPath: String
    public let status: Status
    public var additions: Int
    public var deletions: Int

    public init(
        relativePath: String,
        status: Status,
        additions: Int = 0,
        deletions: Int = 0
    ) {
        self.relativePath = relativePath
        self.status = status
        self.additions = additions
        self.deletions = deletions

        let nsPath = relativePath as NSString
        self.fileName = nsPath.lastPathComponent
        let dir = nsPath.deletingLastPathComponent
        self.directoryPath = dir.isEmpty || dir == "." ? "" : dir
    }
}

/// Comparison target for diff computation in the working tree.
public enum WorktreeComparisonMode: Hashable, Sendable {
    case uncommitted
    case againstBranch(String)
    case commit(hash: String, summary: String)

    public var title: String {
        switch self {
        case .uncommitted:
            return "Uncommitted changes"
        case .againstBranch(let branch):
            return "All changes vs \(branch)"
        case .commit(let hash, let summary):
            let short = String(hash.prefix(7))
            return "\(short): \(summary)"
        }
    }
}

/// Layout mode for displaying changed files in the working tree.
public enum WorktreeViewMode: String, CaseIterable, Sendable {
    case tree
    case flat

    public var title: String {
        switch self {
        case .tree: return "Tree View"
        case .flat: return "Flat List"
        }
    }

    public var systemImage: String {
        switch self {
        case .tree: return "list.bullet.indent"
        case .flat: return "list.bullet"
        }
    }
}

/// Top-level view tab in the Git inspector.
public enum WorktreeTabMode: String, CaseIterable, Sendable {
    case changes
    case timeline
    case worktrees

    public var title: String {
        switch self {
        case .changes: return "Changes"
        case .timeline: return "Timeline"
        case .worktrees: return "Worktrees"
        }
    }

    public var systemImage: String {
        switch self {
        case .changes: return "arrow.triangle.swap"
        case .timeline: return "clock.arrow.circlepath"
        case .worktrees: return "folder.badge.gearshape"
        }
    }
}

/// Information about a linked or main Git worktree.
public struct GitWorktreeInfo: Identifiable, Hashable, Sendable {
    public var id: String { path }
    public let path: String
    public let head: String
    public let branch: String
    public let isCurrent: Bool

    public init(path: String, head: String, branch: String, isCurrent: Bool) {
        self.path = path
        self.head = head
        self.branch = branch
        self.isCurrent = isCurrent
    }
}

/// One Git commit record from recent history.
public struct WorktreeCommit: Identifiable, Hashable, Sendable {
    public var id: String { hash }
    public let hash: String
    public let shortHash: String
    public let author: String
    public let relativeDate: String
    public let summary: String

    public init(
        hash: String,
        shortHash: String,
        author: String,
        relativeDate: String,
        summary: String
    ) {
        self.hash = hash
        self.shortHash = shortHash
        self.author = author
        self.relativeDate = relativeDate
        self.summary = summary
    }
}

/// Observable coordinator managing Git worktree branch, status, changes, and diffs for a project directory.
@MainActor
public final class WorktreeModel: ObservableObject {
    @Published public var rootDirectoryPath: String
    @Published public var currentBranch: String = "main"
    @Published public var files: [WorktreeFileChange] = []
    @Published public var totalAdditions: Int = 0
    @Published public var totalDeletions: Int = 0
    @Published public var isRefreshing: Bool = false
    @Published public var lastRefreshTime: Date?
    @Published public var selectedFilePath: String?
    @Published public var selectedFileDiff: String?
    @Published public var isLoadingDiff: Bool = false
    @Published public var isGitRepository: Bool = false

    // Claude Code-style expansion, comparison, and view options
    @Published public var activeTab: WorktreeTabMode = .changes
    @Published public var comparisonMode: WorktreeComparisonMode = .againstBranch("main")
    @Published public var availableBranches: [String] = []
    @Published public var recentCommits: [WorktreeCommit] = []
    @Published public var worktrees: [GitWorktreeInfo] = []
    @Published public var viewMode: WorktreeViewMode = .tree
    @Published public var searchQuery: String = ""
    @Published public var isSearchVisible: Bool = false
    @Published public var isExpandedSplitMode: Bool = false
    @Published public var selectedCommit: WorktreeCommit? = nil

    // Timeline and branch diff scope
    @Published public var selectedTimelineCommit: WorktreeCommit? = nil
    @Published public var selectedCommitFiles: [WorktreeFileChange] = []
    @Published public var isLoadingCommitFiles: Bool = false
    @Published public var branchComparisonFiles: [WorktreeFileChange] = []

    private var refreshTask: Task<Void, Never>?
    /// Held so `loadDiff` can cancel its predecessor: an unstored `Task`'s
    /// `Task.isCancelled` is never true, which is what let two diffs race.
    private var diffTask: Task<Void, Never>?
    /// Bumped per `refresh()`. A cancelled predecessor's `defer` still runs,
    /// so ownership of `isRefreshing` is decided by this rather than by which
    /// task happens to finish last.
    private var refreshGeneration = 0

    public init(rootDirectoryPath: String) {
        self.rootDirectoryPath = rootDirectoryPath.isEmpty
            ? "" : PathContainment.canonical(URL(fileURLWithPath: rootDirectoryPath)).path
        refresh()
    }

    /// Repoints at another repository.
    ///
    /// **STANDARDIZED BEFORE COMPARING** (state#55). Raw string equality
    /// treats `/tmp/p`, `/tmp/p/` and `/private/tmp/p` as three roots, so a
    /// caller passing a cosmetically different spelling of the SAME path
    /// paid a full refresh, and one passing a symlinked spelling of a
    /// DIFFERENT path... also worked, by luck. Compare canonical, store
    /// canonical.
    public func updateRoot(path: String) {
        let canonical = path.isEmpty
            ? "" : PathContainment.canonical(URL(fileURLWithPath: path)).path
        guard canonical != rootDirectoryPath else { return }
        rootDirectoryPath = canonical
        refresh()
    }

    /// Asynchronously refreshes git branch, porcelain status, and line change stats.
    public func refresh() {
        refreshTask?.cancel()
        refreshGeneration += 1
        let generation = refreshGeneration
        refreshTask = Task { [weak self] in
            guard let self = self else { return }
            guard !self.rootDirectoryPath.isEmpty else {
                // Every field, not just `files`: leaving the previous
                // repository's branch name and line totals on screen beside an
                // empty file list reads as a repository with no changes.
                self.files = []
                self.isGitRepository = false
                self.currentBranch = ""
                self.totalAdditions = 0
                self.totalDeletions = 0
                self.selectedFilePath = nil
                self.selectedFileDiff = nil
                if self.refreshGeneration == generation {
                    self.isRefreshing = false
                }
                return
            }

            self.isRefreshing = true
            // A generation counter rather than `defer`: a cancelled
            // predecessor's `defer` still runs, and it ran AFTER the
            // successor had set `isRefreshing = true` -- so a quick second
            // refresh cleared the spinner while its own query was in flight.
            defer {
                if self.refreshGeneration == generation {
                    self.isRefreshing = false
                }
            }

            let path = self.rootDirectoryPath
            let (isGit, branch, changes, adds, dels) = await Self.queryGitStatus(rootPath: path)
            var branches: [String] = []
            var commits: [WorktreeCommit] = []
            var worktreeItems: [GitWorktreeInfo] = []
            if isGit {
                branches = await Self.queryGitBranches(rootPath: path)
                commits = await Self.queryRecentCommits(rootPath: path, maxCount: 25)
                worktreeItems = await Self.queryGitWorktrees(rootPath: path)
            }

            guard !Task.isCancelled, self.refreshGeneration == generation else { return }

            self.isGitRepository = isGit
            self.currentBranch = branch.isEmpty ? "main" : branch
            self.files = changes
            self.availableBranches = branches
            self.recentCommits = commits
            self.worktrees = worktreeItems
            self.totalAdditions = adds
            self.totalDeletions = dels
            self.lastRefreshTime = Date()

            if let selected = self.selectedFilePath, changes.contains(where: { $0.relativePath == selected }) {
                self.loadDiff(for: selected)
            } else {
                self.selectedFilePath = nil
                self.selectedFileDiff = nil
            }
        }
    }

    /// Selects a commit from the timeline to inspect its files and message.
    ///
    /// Also moves `comparisonMode` to that commit: `loadDiff` reads the
    /// ambient mode, so a file row clicked under a commit whose mode still
    /// said `.againstBranch` fetched the BRANCH diff -- empty for a file the
    /// commit touched but the working tree did not, which then fell through
    /// to the untracked fallback and showed the whole file as additions
    /// under a commit header.
    public func selectTimelineCommit(_ commit: WorktreeCommit) {
        selectedTimelineCommit = commit
        comparisonMode = .commit(hash: commit.hash, summary: commit.summary)
        isLoadingCommitFiles = true
        selectedCommitFiles = []
        let path = rootDirectoryPath
        let hash = commit.hash
        Task { [weak self] in
            guard let self = self else { return }
            let files = await Self.queryCommitFiles(rootPath: path, hash: hash)
            guard !Task.isCancelled else { return }
            self.selectedCommitFiles = files
            self.isLoadingCommitFiles = false
            if let first = files.first {
                self.loadDiff(for: first.relativePath)
            }
        }
    }

    /// Clears the timeline selection. The commit-scoped comparison mode
    /// goes with it: left behind, the next file click in the Changes list
    /// would query the stale commit's diff instead of the mode the list is
    /// showing.
    public func deselectTimelineCommit() {
        selectedTimelineCommit = nil
        selectedCommitFiles = []
        if case .commit = comparisonMode {
            comparisonMode = .uncommitted
        }
    }

    /// Loads files modified between the current branch and a base branch.
    public func loadBranchComparisonFiles(baseBranch: String) {
        let path = rootDirectoryPath
        Task { [weak self] in
            guard let self = self else { return }
            let files = await Self.queryBranchDiffFiles(rootPath: path, baseBranch: baseBranch)
            guard !Task.isCancelled else { return }
            self.branchComparisonFiles = files
        }
    }

    /// Switches the active project repository to another linked Git worktree.
    public func selectWorktree(path: String) {
        guard path != rootDirectoryPath else { return }
        updateRoot(path: path)
    }

    /// Checks out another local Git branch.
    public func switchBranch(to branchName: String) async -> (success: Bool, error: String?) {
        guard !rootDirectoryPath.isEmpty else { return (false, "No repository root") }
        let (exitCode, stderr) = await Self.runGitCheckout(rootPath: rootDirectoryPath, branch: branchName)
        if exitCode == 0 {
            refresh()
            return (true, nil)
        } else {
            return (false, stderr.isEmpty ? "Failed to switch branch." : stderr)
        }
    }

    /// Sets the active diff comparison target and reloads the current diff if any.
    public func setComparisonMode(_ mode: WorktreeComparisonMode) {
        self.comparisonMode = mode
        switch mode {
        case .againstBranch(let base):
            loadBranchComparisonFiles(baseBranch: base)
        case .commit(let hash, _):
            if let commit = recentCommits.first(where: { $0.hash == hash }) {
                selectTimelineCommit(commit)
            }
        case .uncommitted:
            break
        }
        if let selected = selectedFilePath {
            loadDiff(for: selected)
        }
    }

    /// Initializes a Git repository in the current root directory.
    public func initRepository() {
        guard !rootDirectoryPath.isEmpty else { return }
        let path = rootDirectoryPath
        Task { [weak self] in
            let success = await Self.runGitInit(rootPath: path)
            if success {
                self?.refresh()
            }
        }
    }

    /// File changes scoped to the active comparison mode (working tree, branch, or commit).
    public var scopedFiles: [WorktreeFileChange] {
        switch comparisonMode {
        case .uncommitted:
            return files
        case .againstBranch:
            return branchComparisonFiles.isEmpty ? files : branchComparisonFiles
        case .commit:
            return selectedCommitFiles.isEmpty ? files : selectedCommitFiles
        }
    }

    /// Filtered file changes matching the active search query within the active scope.
    public var filteredFiles: [WorktreeFileChange] {
        let q = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let source = scopedFiles
        guard !q.isEmpty else { return source }
        return source.filter {
            $0.relativePath.lowercased().contains(q) || $0.fileName.lowercased().contains(q)
        }
    }

    /// Hierarchical tree nodes built from filtered file changes.
    public var treeNodes: [WorktreeTreeNode] {
        Self.buildTree(from: filteredFiles)
    }

    /// Loads the git diff for a specific file path.
    ///
    /// **THE TASK IS STORED, AND THAT IS WHAT MAKES THE CANCEL CHECK REAL.**
    /// It used to be a bare `Task { ... }` referenced by nothing, so the
    /// `guard !Task.isCancelled` inside it could never be true: clicking file
    /// A then file B raced two `git diff` processes, and whichever finished
    /// last won -- showing A's diff under B's name. The path is re-checked
    /// after the await as well, since cancellation is cooperative and the
    /// process may already have finished.
    public func loadDiff(for relativePath: String) {
        diffTask?.cancel()
        selectedFilePath = relativePath
        selectedFileDiff = nil
        isLoadingDiff = true
        let path = rootDirectoryPath
        let mode = comparisonMode

        diffTask = Task { [weak self] in
            guard let self = self else { return }
            let diff = await Self.queryGitDiff(rootPath: path, file: relativePath, comparisonMode: mode)
            // Cancelled means a SUCCESSOR owns the spinner, so leave it up.
            guard !Task.isCancelled else { return }
            // Not cancelled but the selection moved anyway -- a `refresh()`
            // that no longer lists this file clears `selectedFilePath` out
            // from under us. Nothing else is loading, so the spinner is ours
            // to take down; leaving it up strands it forever.
            guard self.selectedFilePath == relativePath else {
                self.isLoadingDiff = false
                return
            }
            self.selectedFileDiff = diff
            self.isLoadingDiff = false
        }
    }
}
