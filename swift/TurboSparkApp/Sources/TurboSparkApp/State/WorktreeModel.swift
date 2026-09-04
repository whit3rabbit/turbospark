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

    // MARK: - Git Process Execution

    /// Runs one `git` invocation in `rootPath`.
    ///
    /// **THROUGH `ProcessExecutor`, WHICH IS NOT A TIDY-UP** (state#54). The
    /// version this replaced called `waitUntilExit()` and THEN
    /// `readDataToEndOfFile()`, which deadlocks the moment the output exceeds
    /// one pipe buffer (64 KiB here): git blocks writing, this blocks
    /// waiting, and neither moves. `git diff` on any real change reaches that
    /// in one file. It also had no timeout and no handle, so a `git` blocked
    /// on an index lock or a network remote parked a global-queue worker for
    /// the life of the process with `isRefreshing` stuck true and the pane
    /// showing a spinner forever.
    ///
    /// `ProcessExecutor` drains both pipes concurrently with the wait and
    /// enforces a deadline, which is exactly the pair that was missing.
    private static func runGitCommand(args: [String], rootPath: String) async -> (
        exitCode: Int32, stdout: String, stderr: String
    ) {
        do {
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/usr/bin/git"),
                arguments: args,
                currentDirectoryURL: URL(fileURLWithPath: rootPath),
                timeoutSeconds: gitTimeoutSeconds
            )
            if result.timedOut {
                return (
                    -1, result.stdout,
                    "git \(args.first ?? "") timed out after \(Int(gitTimeoutSeconds))s"
                )
            }
            return (result.exitCode, result.stdout, result.stderr)
        } catch {
            return (-1, "", error.localizedDescription)
        }
    }

    /// Long enough for `git status` on a large repository, short enough that
    /// a `git` blocked on an index lock releases the pane.
    private static let gitTimeoutSeconds: TimeInterval = 20

    /// The path a `--porcelain=v1` line really names (state#54).
    ///
    /// Two things were taken literally. A RENAME is `R  old -> new`, and the
    /// whole string was used as a path -- so the row named a file that does
    /// not exist, and the `--numstat` lookup beside it missed, reporting 0
    /// additions and 0 deletions for every rename. And a path containing a
    /// space, a quote or a non-ASCII byte is C-QUOTED by git, so it arrived
    /// wrapped in `"` with `\\n` and `\\303\\251` style escapes intact.
    ///
    /// Pure, and separate from the loop, so both can be asserted without a
    /// repository (`swift/CLAUDE.md` Gotcha 26). `nonisolated` for the same
    /// reason `probe` is: the status parse runs off the main actor, and a
    /// test asserting a pure function should not have to hop onto it.
    nonisolated static func porcelainPath(_ raw: String) -> String {
        var path = raw
        // The arrow separates old from new; the NEW name is the file that
        // exists. Only outside quotes, so a literal " -> " inside a quoted
        // filename is left alone.
        if !path.hasPrefix("\"") , let arrow = path.range(of: " -> ") {
            path = String(path[arrow.upperBound...])
        } else if path.hasPrefix("\""), let arrow = path.range(of: "\" -> ") {
            path = String(path[arrow.upperBound...])
        }
        return unquotePorcelain(path.trimmingCharacters(in: .whitespaces))
    }

    /// The path `--numstat` names, in the same spelling `porcelainPath`
    /// produces (state#89).
    ///
    /// **A JOIN NEEDS BOTH SIDES NORMALIZED, AND ONLY ONE OF THEM WAS.**
    /// state#54 taught the PORCELAIN side about renames and C-quoting; the
    /// numstat side kept `String(parts[2])` raw, and the two formats do not
    /// even agree on how a rename is spelled -- porcelain writes
    /// `old -> new` and numstat writes `old => new` or the factored
    /// `dir/{old => new}/file`. So every rename and every non-ASCII path
    /// looked up a key that could not exist, read 0/0, and silently
    /// subtracted its real additions and deletions from the totals the pane
    /// shows.
    nonisolated static func numstatPath(_ raw: String) -> String {
        let unquoted = unquotePorcelain(raw.trimmingCharacters(in: .whitespaces))
        guard unquoted.contains(" => ") else { return unquoted }
        // The factored form: everything outside the braces is common to both
        // names, and the half after the arrow is the one that exists.
        if let open = unquoted.firstIndex(of: "{"),
            let close = unquoted[open...].firstIndex(of: "}"),
            let arrow = unquoted[open...close].range(of: " => ")
        {
            let prefix = String(unquoted[..<open])
            let renamed = String(unquoted[arrow.upperBound..<close])
            let suffix = String(unquoted[unquoted.index(after: close)...])
            // `dir/{name => }/file` factors an empty half, which would
            // otherwise leave a doubled separator.
            return (prefix + renamed + suffix).replacingOccurrences(of: "//", with: "/")
        }
        if let arrow = unquoted.range(of: " => ") {
            return String(unquoted[arrow.upperBound...])
        }
        return unquoted
    }

    /// Undoes git's C-style quoting. A path not wrapped in `"` is returned
    /// unchanged, which is the overwhelmingly common case.
    nonisolated static func unquotePorcelain(_ raw: String) -> String {
        guard raw.count >= 2, raw.hasPrefix("\""), raw.hasSuffix("\"") else { return raw }
        let body = Array(raw.dropFirst().dropLast().utf8)
        var bytes: [UInt8] = []
        var i = 0
        while i < body.count {
            guard body[i] == UInt8(ascii: "\\"), i + 1 < body.count else {
                bytes.append(body[i])
                i += 1
                continue
            }
            let next = body[i + 1]
            switch next {
            case UInt8(ascii: "n"): bytes.append(0x0A); i += 2
            case UInt8(ascii: "t"): bytes.append(0x09); i += 2
            case UInt8(ascii: "r"): bytes.append(0x0D); i += 2
            case UInt8(ascii: "\""), UInt8(ascii: "\\"): bytes.append(next); i += 2
            case UInt8(ascii: "0")...UInt8(ascii: "7") where i + 3 < body.count:
                // An octal escape is one BYTE, not one character: a UTF-8
                // name arrives as several in a row and only reassembles
                // correctly if they are collected as bytes and decoded once.
                let digits = body[(i + 1)...(i + 3)]
                let value = digits.reduce(0) { $0 * 8 + Int($1 - UInt8(ascii: "0")) }
                if digits.allSatisfy({ $0 >= UInt8(ascii: "0") && $0 <= UInt8(ascii: "7") }),
                    value <= 0xFF
                {
                    bytes.append(UInt8(value))
                    i += 4
                } else {
                    bytes.append(body[i])
                    i += 1
                }
            default:
                bytes.append(next)
                i += 2
            }
        }
        return String(decoding: bytes, as: UTF8.self)
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
                statMap[Self.numstatPath(String(parts[2]))] = (adds, dels)
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
            let rawPath = String(line[line.index(line.startIndex, offsetBy: 3)...])
                .trimmingCharacters(in: .whitespacesAndNewlines)
            let filePath = Self.porcelainPath(rawPath)
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

    /// **`nonisolated`, BOUNDED, AND CONTAINED** (state#90). The untracked
    /// fallback below read a whole file into a String with no size limit and
    /// no containment check, on the MAIN ACTOR -- so previewing an untracked
    /// 2 GB checkpoint or log froze the window, and a `file` argument that
    /// escaped the repository (git will not produce one, but this function's
    /// signature does not say so) was read and displayed.
    nonisolated private static func queryGitDiff(rootPath: String, file: String) async -> String {
        let diffResult = await runGitCommand(args: ["diff", "HEAD", "--", file], rootPath: rootPath)
        if diffResult.exitCode == 0 && !diffResult.stdout.isEmpty {
            return diffResult.stdout
        }
        // If untracked, read the file directly.
        let rootURL = URL(fileURLWithPath: rootPath, isDirectory: true)
        guard
            let fileURL = PathContainment.resolvedIfContained(
                rootURL.appendingPathComponent(file), in: rootURL)
        else {
            return "No diff available: that path resolves outside the repository."
        }
        guard let handle = try? FileHandle(forReadingFrom: fileURL) else {
            return "No diff available."
        }
        defer { try? handle.close() }
        // 64 KiB, which is far more than anyone reads in a diff pane and far
        // less than a checkpoint.
        let cap = 64 * 1024
        guard let data = try? handle.read(upToCount: cap + 1) else {
            return "No diff available."
        }
        let truncated = data.count > cap
        let content = String(decoding: truncated ? data.prefix(cap) : data, as: UTF8.self)
        let lines = content.components(separatedBy: .newlines)
        var formatted = lines.map { "+ \($0)" }.joined(separator: "\n")
        if truncated {
            formatted += "\n+ [truncated at \(cap / 1024) KiB]"
        }
        return "--- /dev/null\n+++ b/\(file)\n@@ -0,0 +1,\(lines.count) @@\n" + formatted
    }
}
