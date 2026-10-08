import Foundation

/// The one way app code runs `/usr/bin/git` against a project repository.
///
/// A repository's `.git/config` is attacker-influenceable (a model allowed to
/// write files can plant it), and several of its keys execute commands during
/// otherwise read-only operations: `core.fsmonitor` runs on `status` and
/// `diff`, `diff.external` / textconv on `diff`, `core.pager` on paged
/// output, and hooks on `checkout` / `switch`. Every override below turns one
/// of those off. Do not call `/usr/bin/git` directly for project repos.
enum HardenedGit {
    static let executableURL = URL(fileURLWithPath: "/usr/bin/git")

    /// `-c` overrides placed before the subcommand.
    private static let globalOverrides = [
        "--no-pager",
        "-c", "core.fsmonitor=false",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.pager=cat",
    ]

    /// Subcommands that can invoke an external diff driver or textconv filter.
    private static let diffLike: Set<String> = ["diff", "show", "log"]

    /// `args` with the hardening flags applied. A leading subcommand is
    /// expected (no caller passes global git options).
    static func arguments(_ args: [String]) -> [String] {
        var out = globalOverrides
        out.append(contentsOf: args)
        if let sub = args.first, diffLike.contains(sub) {
            // Inserted right after the subcommand so a trailing `--` pathspec
            // separator keeps its meaning.
            out.insert(contentsOf: ["--no-ext-diff", "--no-textconv"], at: globalOverrides.count + 1)
        }
        return out
    }

    /// Inherited environment plus the settings that keep git from locking the
    /// index for status refreshes, paging, or prompting for credentials.
    static func environment(base: [String: String] = ProcessInfo.processInfo.environment)
        -> [String: String]
    {
        var env = base
        env["GIT_OPTIONAL_LOCKS"] = "0"
        env["GIT_PAGER"] = "cat"
        env["GIT_TERMINAL_PROMPT"] = "0"
        // These would re-enable the very hooks the -c overrides disable.
        env.removeValue(forKey: "GIT_EXTERNAL_DIFF")
        return env
    }

    static func run(
        arguments args: [String], workingDirectory: String, timeoutSeconds: TimeInterval,
        mergeStreams: Bool = false
    ) async throws -> ProcessExecutor.Output {
        try await ProcessExecutor.run(
            executableURL: executableURL,
            arguments: arguments(args),
            currentDirectoryURL: URL(fileURLWithPath: workingDirectory, isDirectory: true),
            environment: environment(),
            timeoutSeconds: timeoutSeconds,
            mergeStreams: mergeStreams)
    }
}
