import Foundation

/// Executor for git worktree operations (creating, switching, and cleaning up isolated worktrees).
public enum WorktreeExecutor {
    public static func enter(arguments: [String: String], rootURL: URL) async throws -> String {
        let branchName = arguments["name"] ?? arguments["branch"] ?? "turbospark-worktree-\(UUID().uuidString.prefix(6).lowercased())"
        let customPath = arguments["path"]
        let worktreeDirName = customPath ?? ".turbospark/worktrees/\(branchName)"
        let worktreeURL = try AppToolRegistry.resolveSecurePath(relPath: worktreeDirName, rootURL: rootURL)

        let branchCheck = try await runGit(arguments: ["check-ref-format", "--branch", branchName], rootURL: rootURL)
        guard branchCheck.exitCode == 0 else {
            throw toolError(code: 72, message: "Invalid git branch name", output: branchCheck.combinedText)
        }

        try FileManager.default.createDirectory(at: worktreeURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        var res = try await runGit(arguments: ["worktree", "add", "-b", branchName, worktreeURL.path, "HEAD"], rootURL: rootURL)
        if res.exitCode != 0 {
            res = try await runGit(arguments: ["worktree", "add", worktreeURL.path, branchName], rootURL: rootURL)
        }

        let errOutput = [res.stdout, res.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
        guard res.exitCode == 0 || FileManager.default.fileExists(atPath: worktreeURL.path) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 70,
                userInfo: [NSLocalizedDescriptionKey: "Failed to create git worktree: \(errOutput)"]
            )
        }

        return "Entered worktree at '\(worktreeDirName)' on branch '\(branchName)'."
    }

    public static func exit(arguments: [String: String], rootURL: URL) async throws -> String {
        let action = (arguments["action"] ?? "keep").lowercased()
        let path = arguments["path"] ?? arguments["worktree_path"]

        if action == "remove" {
            if let path {
                let worktreeURL = try AppToolRegistry.resolveSecurePath(relPath: path, rootURL: rootURL)
                var gitArguments = ["worktree", "remove"]
                if arguments["discard_changes"]?.lowercased() == "true" {
                    gitArguments.append("--force")
                }
                gitArguments.append(worktreeURL.path)
                let res = try await runGit(arguments: gitArguments, rootURL: rootURL)
                let removeErr = [res.stdout, res.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
                guard res.exitCode == 0 else {
                    throw NSError(
                        domain: "TurboSparkTool",
                        code: 71,
                        userInfo: [NSLocalizedDescriptionKey: "Failed to remove git worktree: \(removeErr)"]
                    )
                }
                return "Exited and removed git worktree at '\(path)'."
            } else {
                return "Exited git worktree and returned to main repository root."
            }
        }

        return "Exited git worktree (preserved at '\(path ?? "active worktree")') and returned to main repository root."
    }

    private static func runGit(arguments: [String], rootURL: URL) async throws -> ProcessExecutor.Output {
        try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/usr/bin/git"),
            arguments: arguments,
            currentDirectoryURL: rootURL,
            timeoutSeconds: 15.0
        )
    }

    private static func toolError(code: Int, message: String, output: String) -> NSError {
        let detail = output.isEmpty ? message : "\(message): \(output)"
        return NSError(
            domain: "TurboSparkTool",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: detail]
        )
    }
}
