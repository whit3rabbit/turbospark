import Foundation

/// The one git implementation both marketplaces use.
///
/// It replaces four raw `Process()` spawns in `SkillMarketplaceManager`, which
/// had three problems the MCP marketplace should not inherit:
///
/// 1. **No timeout and no output cap.** `waitUntilExit()` on `git clone` of an
///    unreachable host hangs the app with nothing to cancel it, and `git` is
///    happy to sit on a credential prompt forever when it thinks a terminal is
///    attached. Everything here goes through `ProcessExecutor.run`, which is
///    where that class of bug is fixed once (see its own header), and
///    `GIT_TERMINAL_PROMPT=0` makes the private-repo case fail rather than wait.
///
/// 2. **Only the clone's exit status was checked.** `pull`, `sparse-checkout
///    set` and `checkout HEAD` all ran unchecked, so a failed update was silent
///    and the caller read stale content as fresh -- which is worse than an
///    error, because a stale catalog looks exactly like a current one.
///
/// 3. **The cache directory was hardcoded** under `~/.turbospark`, outside
///    `AppStorageRoot`, so the test suite wrote the developer's real home
///    (swift/CLAUDE.md Gotcha 43). Callers pass their own directory here.
enum MarketplaceGit {
    /// A git step that did not exit 0.
    struct Failure: LocalizedError {
        let step: String
        let exitCode: Int32
        let stderr: String
        let timedOut: Bool

        var errorDescription: String? {
            if timedOut {
                return "git \(step) timed out. The repository may be unreachable, "
                    + "or it may require credentials this app cannot supply."
            }
            let detail = stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            if detail.isEmpty {
                return "git \(step) failed with exit code \(exitCode)."
            }
            return "git \(step) failed with exit code \(exitCode): \(detail)"
        }
    }

    static let executableURL = URL(fileURLWithPath: "/usr/bin/git")
    static let defaultTimeoutSeconds: TimeInterval = 120

    /// Clones `url` into `targetDir`, or pulls when a clone is already there.
    ///
    /// `sparsePaths`, when non-empty, switches the clone to a blobless
    /// `--no-checkout` fetch followed by a cone-mode sparse checkout, so a
    /// catalog repository holding hundreds of entries costs one subtree.
    static func cloneOrPull(
        url: String,
        targetDir: URL,
        ref: String?,
        sparsePaths: [String]?,
        timeoutSeconds: TimeInterval = defaultTimeoutSeconds
    ) async throws {
        let fileManager = FileManager.default
        let alreadyCloned = fileManager.fileExists(
            atPath: targetDir.appendingPathComponent(".git").path)

        if alreadyCloned {
            try await run(
                step: "pull",
                arguments: ["pull", "--quiet", "--ff-only"],
                currentDirectory: targetDir,
                timeoutSeconds: timeoutSeconds)
            return
        }

        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)

        let sparse = (sparsePaths ?? []).filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        var arguments = ["clone", "--depth", "1"]
        if !sparse.isEmpty {
            arguments.append(contentsOf: ["--filter=blob:none", "--no-checkout"])
        }
        if let ref, !ref.trimmingCharacters(in: .whitespaces).isEmpty {
            arguments.append(contentsOf: ["--branch", ref])
        }
        arguments.append(contentsOf: [url, targetDir.path])

        try await run(
            step: "clone", arguments: arguments, currentDirectory: nil,
            timeoutSeconds: timeoutSeconds)

        guard !sparse.isEmpty else { return }

        try await run(
            step: "sparse-checkout",
            arguments: ["sparse-checkout", "set", "--cone", "--"] + sparse,
            currentDirectory: targetDir,
            timeoutSeconds: timeoutSeconds)
        try await run(
            step: "checkout",
            arguments: ["checkout", "HEAD"],
            currentDirectory: targetDir,
            timeoutSeconds: timeoutSeconds)
    }

    /// A directory name safe to use for `url`'s clone cache.
    static func cacheDirectoryName(for url: String) -> String {
        let mapped = url.map { character -> Character in
            character.isLetter || character.isNumber || character == "." || character == "-"
                ? character
                : "-"
        }
        let slug = String(mapped)
        return slug.isEmpty ? "repository" : slug
    }

    private static func run(
        step: String,
        arguments: [String],
        currentDirectory: URL?,
        timeoutSeconds: TimeInterval
    ) async throws {
        let output = try await ProcessExecutor.run(
            executableURL: executableURL,
            arguments: arguments,
            currentDirectoryURL: currentDirectory,
            environment: childEnvironment(),
            timeoutSeconds: timeoutSeconds)

        guard output.exitCode == 0, !output.timedOut else {
            throw Failure(
                step: step, exitCode: output.exitCode, stderr: output.stderr,
                timedOut: output.timedOut)
        }
    }

    /// The parent environment plus the two variables that keep `git` from
    /// blocking on a prompt no one can answer. Unlike an MCP server, `git` here
    /// is this machine's own binary rather than third-party code, and it needs
    /// `HOME` for `.gitconfig` and its credential helper, so the environment is
    /// inherited rather than rebuilt.
    private static func childEnvironment() -> [String: String] {
        var environment = ProcessInfo.processInfo.environment
        environment["GIT_TERMINAL_PROMPT"] = "0"
        environment["GIT_OPTIONAL_LOCKS"] = "0"
        return environment
    }
}
