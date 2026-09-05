import Foundation

/// Executor for git worktree operations (creating, switching, and cleaning up isolated worktrees).
public enum WorktreeExecutor {
    public static func enter(arguments: [String: String], rootURL: URL) async throws -> String {
        let branchName = arguments["name"] ?? arguments["branch"] ?? "turbospark-worktree-\(UUID().uuidString.prefix(6).lowercased())"
        let customPath = arguments["path"]
        let worktreeDirName = customPath ?? ".turbospark/worktrees/\(branchName)"
        let worktreeURL = rootURL.appendingPathComponent(worktreeDirName)

        try FileManager.default.createDirectory(at: worktreeURL.deletingLastPathComponent(), withIntermediateDirectories: true)

        let cmd = "git worktree add -b \"\(branchName)\" \"\(worktreeURL.path)\" HEAD 2>&1 || git worktree add \"\(worktreeURL.path)\" \"\(branchName)\" 2>&1"
        let res = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", cmd],
            currentDirectoryURL: rootURL,
            timeoutSeconds: 15.0
        )

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
                let discardFlag = (arguments["discard_changes"]?.lowercased() == "true") ? "--force" : ""
                let cmd = "git worktree remove \(discardFlag) \"\(worktreeURL.path)\" 2>&1"
                let res = try await ProcessExecutor.run(
                    executableURL: URL(fileURLWithPath: "/bin/zsh"),
                    arguments: ["-c", cmd],
                    currentDirectoryURL: rootURL,
                    timeoutSeconds: 15.0
                )
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
}
