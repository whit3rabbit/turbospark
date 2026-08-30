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

    public init(rootDirectoryPath: String) {
        self.rootDirectoryPath = rootDirectoryPath
        refresh()
    }

    public func updateRoot(path: String) {
        guard path != rootDirectoryPath else { return }
        rootDirectoryPath = path
        refresh()
    }

    /// Asynchronously refreshes git branch, porcelain status, and line change stats.
    public func refresh() {
        refreshTask?.cancel()
        refreshTask = Task { [weak self] in
            guard let self = self else { return }
            guard !self.rootDirectoryPath.isEmpty else {
                self.files = []
                self.isGitRepository = false
                return
            }

            self.isRefreshing = true
            defer { self.isRefreshing = false }

            let path = self.rootDirectoryPath
            let (isGit, branch, changes, adds, dels) = await Self.queryGitStatus(rootPath: path)

            guard !Task.isCancelled else { return }

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
    public func loadDiff(for relativePath: String) {
        selectedFilePath = relativePath
        isLoadingDiff = true
        let path = rootDirectoryPath

        Task { [weak self] in
            guard let self = self else { return }
            let diff = await Self.queryGitDiff(rootPath: path, file: relativePath)
            guard !Task.isCancelled else { return }
            self.selectedFileDiff = diff
            self.isLoadingDiff = false
        }
    }

    // MARK: - Git Process Execution

    private static func runGitCommand(args: [String], rootPath: String) async -> (exitCode: Int32, stdout: String, stderr: String) {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let process = Process()
                process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
                process.currentDirectoryURL = URL(fileURLWithPath: rootPath)
                process.arguments = args

                let outPipe = Pipe()
                let errPipe = Pipe()
                process.standardOutput = outPipe
                process.standardError = errPipe

                do {
                    try process.run()
                    process.waitUntilExit()

                    let outData = outPipe.fileHandleForReading.readDataToEndOfFile()
                    let errData = errPipe.fileHandleForReading.readDataToEndOfFile()

                    let outStr = String(data: outData, encoding: .utf8) ?? ""
                    let errStr = String(data: errData, encoding: .utf8) ?? ""

                    continuation.resume(returning: (process.terminationStatus, outStr, errStr))
                } catch {
                    continuation.resume(returning: (-1, "", error.localizedDescription))
                }
            }
        }
    }

    private static func queryGitStatus(rootPath: String) async -> (isGit: Bool, branch: String, changes: [WorktreeFileChange], adds: Int, dels: Int) {
        let revParse = await runGitCommand(args: ["rev-parse", "--is-inside-work-tree"], rootPath: rootPath)
        guard revParse.exitCode == 0, revParse.stdout.trimmingCharacters(in: .whitespacesAndNewlines) == "true" else {
            return (false, "", [], 0, 0)
        }

        let branchResult = await runGitCommand(args: ["branch", "--show-current"], rootPath: rootPath)
        var branch = branchResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        if branch.isEmpty {
            let headResult = await runGitCommand(args: ["rev-parse", "--short", "HEAD"], rootPath: rootPath)
            branch = headResult.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // Fetch diff numstat for additions/deletions
        let numstatResult = await runGitCommand(args: ["diff", "HEAD", "--numstat"], rootPath: rootPath)
        var statMap: [String: (adds: Int, dels: Int)] = [:]
        for line in numstatResult.stdout.components(separatedBy: .newlines) {
            let parts = line.split(separator: "\t")
            if parts.count >= 3 {
                let adds = Int(parts[0]) ?? 0
                let dels = Int(parts[1]) ?? 0
                let file = String(parts[2]).trimmingCharacters(in: .whitespacesAndNewlines)
                statMap[file] = (adds, dels)
            }
        }

        // Fetch porcelain status
        let statusResult = await runGitCommand(args: ["status", "--porcelain=v1", "-uall"], rootPath: rootPath)
        var changes: [WorktreeFileChange] = []
        var totalAdds = 0
        var totalDels = 0

        for line in statusResult.stdout.components(separatedBy: .newlines) {
            guard line.count >= 4 else { continue }
            let indexCode = line[line.startIndex]
            let worktreeCode = line[line.index(line.startIndex, offsetBy: 1)]
            let filePath = String(line[line.index(line.startIndex, offsetBy: 3)...]).trimmingCharacters(in: .whitespacesAndNewlines)
            guard !filePath.isEmpty else { continue }

            let status: WorktreeFileChange.Status
            if indexCode == "?" || worktreeCode == "?" {
                status = .untracked
            } else if indexCode == "A" || worktreeCode == "A" {
                status = .added
            } else if indexCode == "D" || worktreeCode == "D" {
                status = .deleted
            } else if indexCode == "R" || worktreeCode == "R" {
                status = .renamed
            } else if indexCode == "M" || worktreeCode == "M" {
                status = .modified
            } else {
                status = .unknown
            }

            let stats = statMap[filePath] ?? (0, 0)
            totalAdds += stats.adds
            totalDels += stats.dels

            changes.append(
                WorktreeFileChange(
                    relativePath: filePath,
                    status: status,
                    additions: stats.adds,
                    deletions: stats.dels
                )
            )
        }

        // Sort by directory and name
        changes.sort { lhs, rhs in
            if lhs.directoryPath != rhs.directoryPath {
                return lhs.directoryPath < rhs.directoryPath
            }
            return lhs.fileName < rhs.fileName
        }

        return (true, branch, changes, totalAdds, totalDels)
    }

    private static func queryGitDiff(rootPath: String, file: String) async -> String {
        let diffResult = await runGitCommand(args: ["diff", "HEAD", "--", file], rootPath: rootPath)
        if diffResult.exitCode == 0 && !diffResult.stdout.isEmpty {
            return diffResult.stdout
        }
        // If untracked, read the file directly
        let fileURL = URL(fileURLWithPath: rootPath).appendingPathComponent(file)
        if let content = try? String(contentsOf: fileURL, encoding: .utf8) {
            let lines = content.components(separatedBy: .newlines)
            let formatted = lines.map { "+ \($0)" }.joined(separator: "\n")
            return "--- /dev/null\n+++ b/\(file)\n@@ -0,0 +1,\(lines.count) @@\n" + formatted
        }
        return "No diff available."
    }
}
