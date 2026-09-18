import Foundation

/// The qwen-code git-command parity layer: `/diff`, `/log` and `/prs` local
/// commands over the selected chat's project root. Everything runs through
/// the same `git` / `gh` processes the worktree pane uses; none of it
/// generates, and all of it works while a turn is running, like every other
/// local command.
extension AppModel {
    nonisolated static let gitDiffLineLimit = 10_000

    /// Which view the git sheet is showing. One sheet for the three commands
    /// (qwen-code's GitDialog is the same container with `diff | log | prs`
    /// views), so switching commands mid-flight keeps the chrome.
    public enum GitInfoTab: String, CaseIterable, Identifiable {
        case diff
        case log
        case prs

        public var id: String { rawValue }
    }

    /// One row of `/prs` output, parsed from `gh pr list --json`.
    public struct GitPullRequestRow: Identifiable, Equatable {
        public let number: Int
        public let title: String
        public let state: String
        public let headBranch: String
        public let url: String

        public var id: Int { number }
    }

    /// Whether the git info sheet (`/diff`, `/log`, `/prs`) is showing,
    /// the fetched rows and the load error: stored on the class body
    /// (extensions cannot hold `@Published`), documented here beside the
    /// commands that drive them.

    /// The project root the git commands run against: the selected chat's
    /// project when it has one, else the app-wide selection. Nil refuses by
    /// name, exactly like the file and shell tools do for a projectless
    /// chat.
    public var gitCommandRootPath: String? {
        let project = turnProject(chatID: selectedChatID) ?? selectedProject
        guard let path = project?.rootDirectoryPath, !path.isEmpty else { return nil }
        return path
    }

    // MARK: - /diff

    public func runDiffCommand() {
        guard let root = gitCommandRootPath else {
            showToast("Pick a project with a folder first: /diff works on its git repository.", style: .warning)
            return
        }
        gitInfoTab = .diff
        gitDiffText = ""
        gitInfoError = nil
        isLoadingGitInfo = true
        showGitSheet = true
        Task {
            // One whole-tree diff against HEAD. Per-file diffs are the
            // worktree pane's job; this answers "what changed overall".
            let result = await Self.runProcess(
                executable: "/usr/bin/git",
                arguments: ["diff", "HEAD", "--stat"], workingDirectory: root)
            let full = await Self.runProcess(
                executable: "/usr/bin/git",
                arguments: ["diff", "HEAD"], workingDirectory: root)
            let stat = result.exitCode == 0 ? result.stdout : ""
            let body = full.exitCode == 0 ? Self.limitDiffLines(full.stdout) : ""
            if result.exitCode != 0 || full.exitCode != 0 {
                gitInfoError = (result.stderr.isEmpty ? full.stderr : result.stderr)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            }
            if body.isEmpty && gitInfoError == nil {
                gitInfoError = nil
            }
            gitDiffText = stat.isEmpty && body.isEmpty
                ? "" : (stat.isEmpty ? body : stat + "\n\n" + body)
            isLoadingGitInfo = false
        }
    }

    // MARK: - /log

    public func runLogCommand() {
        guard let root = gitCommandRootPath else {
            showToast("Pick a project with a folder first: /log works on its git repository.", style: .warning)
            return
        }
        gitInfoTab = .log
        gitCommits = []
        gitInfoError = nil
        isLoadingGitInfo = true
        showGitSheet = true
        Task {
            gitCommits = await WorktreeModel.queryRecentCommits(rootPath: root, maxCount: 50)
            if gitCommits.isEmpty {
                let probe = await Self.runProcess(
                    executable: "/usr/bin/git", arguments: ["status"], workingDirectory: root)
                if probe.exitCode != 0 {
                    gitInfoError = probe.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                }
            }
            isLoadingGitInfo = false
        }
    }

    // MARK: - /prs

    /// Re-runs the fetch for a tab whose data is absent, so a segment
    /// switch shows the real query instead of a never-fetched empty state
    /// ("No commits yet." for a log that was never asked for). Silent when
    /// there is no project root: the command's own toast already fired when
    /// the user ran the slash command, and repeating it on every tab switch
    /// would be noise.
    public func refreshGitInfoTabIfEmpty(_ tab: GitInfoTab) {
        guard !isLoadingGitInfo, gitCommandRootPath != nil else { return }
        switch tab {
        case .diff:
            if gitDiffText.isEmpty { runDiffCommand() }
        case .log:
            if gitCommits.isEmpty { runLogCommand() }
        case .prs:
            if gitPullRequests.isEmpty { runPrsCommand() }
        }
    }

    public func runPrsCommand() {
        guard let root = gitCommandRootPath else {
            showToast("Pick a project with a folder first: /prs works on its git repository.", style: .warning)
            return
        }
        gitInfoTab = .prs
        gitPullRequests = []
        gitInfoError = nil
        isLoadingGitInfo = true
        showGitSheet = true
        Task {
            let result = await Self.runProcess(
                executable: "/usr/bin/env",
                arguments: [
                    "gh", "pr", "list",
                    "--json", "number,title,state,headRefName,url", "--limit", "30",
                ],
                workingDirectory: root)
            guard result.exitCode == 0 else {
                let message = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                gitInfoError = message.isEmpty
                    ? "gh exited with status \(result.exitCode)."
                    : message
                isLoadingGitInfo = false
                return
            }
            gitPullRequests = Self.parsePullRequests(result.stdout)
            isLoadingGitInfo = false
        }
    }

    /// Parses `gh pr list --json` output. A malformed body is an empty list
    /// with the raw body kept as the error, never a crash: gh's shape is
    /// stable but not ours to guarantee.
    static func parsePullRequests(_ json: String) -> [GitPullRequestRow] {
        struct Row: Decodable {
            var number: Int
            var title: String
            var state: String
            var headRefName: String
            var url: String
        }
        guard let data = json.data(using: .utf8),
            let rows = try? JSONDecoder().decode([Row].self, from: data)
        else { return [] }
        return rows.map {
            GitPullRequestRow(
                number: $0.number, title: $0.title, state: $0.state,
                headBranch: $0.headRefName, url: $0.url)
        }
    }

    /// Bounds the number of SwiftUI rows a hostile or unusually noisy diff
    /// can create. ProcessExecutor independently caps the bytes captured from
    /// git; this second bound protects rendering when those bytes are mostly
    /// very short lines.
    static func limitDiffLines(_ text: String, maxLines: Int = gitDiffLineLimit) -> String {
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
        guard lines.count > maxLines else { return text }
        return lines.prefix(maxLines).joined(separator: "\n")
            + "\n... (diff truncated at \(maxLines) lines)"
    }

    // MARK: - Process helper

    /// One bounded git/gh invocation through the app-wide capped executor.
    /// Its truncation marker is intentionally retained for the git sheet.
    static func runProcess(
        executable: String, arguments: [String], workingDirectory: String
    ) async -> (exitCode: Int32, stdout: String, stderr: String) {
        do {
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: executable),
                arguments: arguments,
                currentDirectoryURL: URL(fileURLWithPath: workingDirectory, isDirectory: true),
                timeoutSeconds: 15)
            let error = result.timedOut && result.stderr.isEmpty
                ? "Timed out after 15 seconds."
                : result.stderr
            return (result.exitCode, result.stdout, error)
        } catch {
            return (-1, "", error.localizedDescription)
        }
    }
}
