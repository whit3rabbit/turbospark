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

            guard !Task.isCancelled, self.refreshGeneration == generation else { return }

            self.isGitRepository = isGit
            self.currentBranch = branch.isEmpty ? "main" : branch
            self.files = changes
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

    /// Loads the git diff for a specific file path.
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

        diffTask = Task { [weak self] in
            guard let self = self else { return }
            let diff = await Self.queryGitDiff(rootPath: path, file: relativePath)
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
