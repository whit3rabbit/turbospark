import Foundation

/// Talking to `git`, and reading what it says back.
///
/// **EVERYTHING HERE IS PURE OR PROCESS PLUMBING, AND THAT IS THE POINT.**
/// The published half next door cannot be asserted without a repository on
/// disk; these can, and every defect this code has carried was in the
/// READING rather than in the running. state#54 found `readDataToEndOfFile`
/// after `waitUntilExit` -- a deadlock past one pipe buffer -- alongside
/// renames and quoted paths taken literally, and state#89 found the numstat
/// half of that same join still raw a pass later. `porcelainPath`,
/// `numstatPath` and `unquotePorcelain` are `nonisolated` static so a test
/// can reach them without hopping onto the main actor for a string function.
extension WorktreeModel {
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

    // `internal` rather than `private` only because the published half is
    // in the file next door; nothing outside this module can see it.
    static func queryGitStatus(rootPath: String) async -> (isGit: Bool, branch: String, changes: [WorktreeFileChange], adds: Int, dels: Int) {
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

    /// Initializes a new Git repository in `rootPath`.
    static func runGitInit(rootPath: String) async -> Bool {
        let result = await runGitCommand(args: ["init"], rootPath: rootPath)
        return result.exitCode == 0
    }

    /// Queries local Git branches in `rootPath`.
    static func queryGitBranches(rootPath: String) async -> [String] {
        let result = await runGitCommand(
            args: ["branch", "--list", "--format=%(refname:short)"],
            rootPath: rootPath
        )
        guard result.exitCode == 0 else { return [] }
        return result.stdout
            .components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    /// Queries recent Git commits in `rootPath`.
    static func queryRecentCommits(rootPath: String, maxCount: Int = 25) async -> [WorktreeCommit] {
        let result = await runGitCommand(
            args: ["log", "-n", "\(maxCount)", "--pretty=format:%H%x09%h%x09%an%x09%cr%x09%s"],
            rootPath: rootPath
        )
        guard result.exitCode == 0 else { return [] }
        var commits: [WorktreeCommit] = []
        for line in result.stdout.components(separatedBy: .newlines) {
            let parts = line.components(separatedBy: "\t")
            guard parts.count >= 5 else { continue }
            commits.append(
                WorktreeCommit(
                    hash: parts[0],
                    shortHash: parts[1],
                    author: parts[2],
                    relativeDate: parts[3],
                    summary: parts[4]
                )
            )
        }
        return commits
    }

    /// Checks out or switches to another local Git branch.
    static func runGitCheckout(rootPath: String, branch: String) async -> (exitCode: Int32, error: String) {
        let result = await runGitCommand(args: ["switch", branch], rootPath: rootPath)
        if result.exitCode == 0 {
            return (0, "")
        }
        let fallback = await runGitCommand(args: ["checkout", branch], rootPath: rootPath)
        return (fallback.exitCode, fallback.stderr)
    }

    /// Queries registered Git worktrees for this repository.
    static func queryGitWorktrees(rootPath: String) async -> [GitWorktreeInfo] {
        let result = await runGitCommand(args: ["worktree", "list", "--porcelain"], rootPath: rootPath)
        guard result.exitCode == 0 else { return [] }
        return parseWorktreeListOutput(result.stdout, currentRoot: rootPath)
    }

    /// Queries the files modified in a specific Git commit.
    static func queryCommitFiles(rootPath: String, hash: String) async -> [WorktreeFileChange] {
        let numstat = await runGitCommand(args: ["show", hash, "--numstat", "--format="], rootPath: rootPath)
        let nameStatus = await runGitCommand(args: ["show", hash, "--name-status", "--format="], rootPath: rootPath)
        return parseNameStatusAndNumstat(numstatOutput: numstat.stdout, nameStatusOutput: nameStatus.stdout)
    }

    /// Queries files modified across a branch compared against a base branch.
    static func queryBranchDiffFiles(rootPath: String, baseBranch: String) async -> [WorktreeFileChange] {
        var numstat = await runGitCommand(args: ["diff", "\(baseBranch)...HEAD", "--numstat"], rootPath: rootPath)
        var nameStatus = await runGitCommand(args: ["diff", "\(baseBranch)...HEAD", "--name-status"], rootPath: rootPath)
        if numstat.exitCode != 0 || nameStatus.exitCode != 0 {
            numstat = await runGitCommand(args: ["diff", baseBranch, "--numstat"], rootPath: rootPath)
            nameStatus = await runGitCommand(args: ["diff", baseBranch, "--name-status"], rootPath: rootPath)
        }
        return parseNameStatusAndNumstat(numstatOutput: numstat.stdout, nameStatusOutput: nameStatus.stdout)
    }

    /// **`nonisolated`, BOUNDED, AND CONTAINED** (state#90). The untracked
    /// fallback below read a whole file into a String with no size limit and
    /// no containment check, on the MAIN ACTOR -- so previewing an untracked
    /// 2 GB checkpoint or log froze the window, and a `file` argument that
    /// escaped the repository (git will not produce one, but this function's
    /// signature does not say so) was read and displayed.
    nonisolated static func queryGitDiff(
        rootPath: String,
        file: String,
        comparisonMode: WorktreeComparisonMode = .uncommitted
    ) async -> String {
        let diffArgs: [String]
        switch comparisonMode {
        case .uncommitted:
            diffArgs = ["diff", "HEAD", "--", file]
        case .againstBranch(let branch):
            diffArgs = ["diff", "\(branch)...HEAD", "--", file]
        case .commit(let hash, _):
            diffArgs = ["show", hash, "--", file]
        }

        var diffResult = await runGitCommand(args: diffArgs, rootPath: rootPath)
        if diffResult.exitCode != 0 || diffResult.stdout.isEmpty {
            if case .againstBranch(let branch) = comparisonMode {
                diffResult = await runGitCommand(args: ["diff", branch, "--", file], rootPath: rootPath)
            }
        }
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
