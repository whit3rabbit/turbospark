import Foundation

/// The shell tool's execution path: command wrapping, the timeout policy,
/// output shaping, cwd capture, and the background handoff.
///
/// Claude Code reference: `src/utils/shell/bashProvider.ts` (the wrapped
/// command with a cwd capture), `src/utils/timeouts.ts` (default 120 s, max
/// 600 s), and `BashTool.tsx` (`runShellCommand` plus the background info
/// text). What is deliberately NOT ported is the `eval` wrapper: the
/// command here is passed as one `zsh -c` argv element already, so the
/// suffix below can follow it as plain lines without re-quoting the user's
/// text at all.
enum ShellCommandRunner {
    /// Hard ceiling on one foreground command, whatever the model asked
    /// for. The default is `ProcessExecutor`'s 120 s; the ceiling is what
    /// stops a misread unit (`timeout: 3600000`, meaning milliseconds, read
    /// as a patient hour) from holding one tool call open that long.
    static let maxTimeoutSeconds: TimeInterval = 600

    static var defaultTimeoutSeconds: TimeInterval { ProcessExecutor.defaultTimeoutSeconds }

    static func clampedTimeoutSeconds(timeoutMs: Int?) -> TimeInterval {
        guard let timeoutMs, timeoutMs > 0 else { return defaultTimeoutSeconds }
        return min(TimeInterval(timeoutMs) / 1000.0, maxTimeoutSeconds)
    }

    /// The command actually run: the user's command verbatim, then a
    /// status-preserving cwd capture. The exit status is captured FIRST and
    /// exited LAST so the suffix cannot change what the caller sees.
    /// Newline separators, not `;`, keep a trailing comment (or a heredoc
    /// terminator) in the user command from swallowing the suffix. The one
    /// shape this breaks is a command ending in a line continuation, which
    /// splices the status line into its own arguments -- a visible failure
    /// in the output, not a silent one.
    static func wrappedCommand(_ command: String, captureFile: String) -> String {
        command + "\n"
            + "__turbospark_status=$?\n"
            + "pwd -P >| '\(captureFile)' 2>/dev/null || true\n"
            + "exit $__turbospark_status\n"
    }

    // MARK: - Entry

    static func run(
        command: String,
        rootURL: URL,
        timeoutMs: Int? = nil,
        runInBackground: Bool = false,
        description: String? = nil,
        chatID: UUID? = nil
    ) async throws -> String {
        let startDirectory = ShellCwdTracker.shared.startDirectory(for: rootURL)
        let environment = ShellOutputFormatting.shellEnvironment()

        if runInBackground {
            return try launchBackground(
                command: command, startDirectory: startDirectory,
                environment: environment, description: description, chatID: chatID)
        }
        return try await runForeground(
            command: command, startDirectory: startDirectory, rootURL: rootURL,
            environment: environment, timeoutMs: timeoutMs)
    }

    // MARK: - Foreground

    private static func runForeground(
        command: String,
        startDirectory: URL,
        rootURL: URL,
        environment: [String: String],
        timeoutMs: Int?
    ) async throws -> String {
        let timeoutSeconds = clampedTimeoutSeconds(timeoutMs: timeoutMs)
        let captureFile = NSTemporaryDirectory() + "turbospark-cwd-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: captureFile) }
        let wrapped = wrappedCommand(command, captureFile: captureFile)

        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", wrapped],
            currentDirectoryURL: startDirectory,
            environment: environment,
            timeoutSeconds: timeoutSeconds,
            mergeStreams: true
        )

        let combined = ShellOutputFormatting.stripANSI(result.combinedText)
        var notes: [String] = []
        // Where did the shell end up? Read before the deferred delete; a
        // failed read just skips the cwd update.
        if let finalCwd = try? String(contentsOfFile: captureFile, encoding: .utf8),
           let note = ShellCwdTracker.shared.record(finalDirectory: finalCwd, root: rootURL) {
            notes.append(note)
        }

        // state#81's rule holds for the timeout arm too: work that did not
        // finish is not a success string. Thrown, so `isError` follows, with
        // whatever output the command managed to produce.
        if result.timedOut {
            let partial = combined.isEmpty
                ? "" : "\n" + ShellOutputFormatting.compactWithSpill(combined, label: command)
            throw NSError(
                domain: "TurboSparkTool", code: 42,
                userInfo: [NSLocalizedDescriptionKey:
                    "(Command timed out after \(Int(timeoutSeconds))s and was terminated.)\(partial)"])
        }

        if result.exitCode != 0 {
            // grep 1 and friends are the command ANSWERING, not failing.
            // Reporting them as errors invites a retry of a question already
            // answered. Anything else throws, so `isError` follows.
            if let benign = ShellOutputFormatting.benignExitNote(
                command: command, exitCode: result.exitCode) {
                notes.append(benign)
            } else {
                throw NSError(
                    domain: "TurboSparkTool", code: 40,
                    userInfo: [NSLocalizedDescriptionKey:
                        "Command failed (exit \(result.exitCode)):\n"
                            + ShellOutputFormatting.compactWithSpill(combined, label: command)])
            }
        }

        var output = combined.isEmpty
            ? (result.exitCode == 0
                ? "(Command finished: success)"
                : "(Command finished: exit code \(result.exitCode))")
            : ShellOutputFormatting.compactWithSpill(combined, label: command)
        if !notes.isEmpty {
            output += "\n(" + notes.joined(separator: "; ") + ")"
        }
        return output
    }

    // MARK: - Background

    /// Launches and returns immediately. Background commands run the RAW
    /// command: the cwd suffix exists to move the NEXT foreground call's
    /// anchor, and a script finishing minutes later, unobserved, must not
    /// silently move it.
    private static func launchBackground(
        command: String,
        startDirectory: URL,
        environment: [String: String],
        description: String?,
        chatID: UUID?
    ) throws -> String {
        let record = try BackgroundShellManager.shared.launch(
            command: command, startDirectory: startDirectory,
            environment: environment, chatID: chatID, description: description)
        var lines = [
            "Command running in background with ID: \(record.id).",
            "Retrieve its output with the BashOutput tool (task_id: \"\(record.id)\"); "
                + "stop it with the KillShell tool.",
            "A background command has no timeout; it runs until it finishes or is killed.",
        ]
        if let description, !description.isEmpty {
            lines.insert("Description: \(description)", at: 1)
        }
        return lines.joined(separator: "\n")
    }

    /// The BashOutput tool: resolve an id within this chat, optionally wait
    /// for completion, return a status snapshot.
    static func backgroundOutput(arguments: [String: String], chatID: UUID?) async throws -> String {
        guard let id = arguments["task_id"] ?? arguments["bash_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "Missing 'task_id'. Pass the background shell id the Bash tool returned."])
        }
        guard let record = BackgroundShellManager.shared.record(id: id, chatID: chatID) else {
            throw NSError(
                domain: "TurboSparkTool", code: 9,
                userInfo: [NSLocalizedDescriptionKey: unknownIDMessage(id: id, chatID: chatID)])
        }
        let waitSeconds = TimeInterval(min(max(Int(arguments["wait_seconds"] ?? "") ?? 30, 0), 120))
        return await BackgroundShellManager.shared.waitAndOutput(record, seconds: waitSeconds)
    }

    /// The KillShell tool.
    static func killBackground(arguments: [String: String], chatID: UUID?) throws -> String {
        guard let id = arguments["task_id"] ?? arguments["bash_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool", code: 4,
                userInfo: [NSLocalizedDescriptionKey:
                    "Missing 'task_id'. Pass the background shell id the Bash tool returned."])
        }
        guard let record = BackgroundShellManager.shared.record(id: id, chatID: chatID) else {
            throw NSError(
                domain: "TurboSparkTool", code: 9,
                userInfo: [NSLocalizedDescriptionKey: unknownIDMessage(id: id, chatID: chatID)])
        }
        guard BackgroundShellManager.shared.kill(record) else {
            return "[\(record.id)] was already finished; nothing to kill.\n\(record.summaryLine)"
        }
        return "[\(record.id)] terminated (SIGTERM, then SIGKILL if it ignored that)."
    }

    private static func unknownIDMessage(id: String, chatID: UUID?) -> String {
        let known = BackgroundShellManager.shared.knownShellIDs(chatID: chatID)
        return "Unknown background shell id '\(id)' for this conversation. "
            + (known.isEmpty
                ? "No background shells exist here; start one with Bash and run_in_background: true."
                : "Known ids: \(known.joined(separator: ", "))")
    }
}
