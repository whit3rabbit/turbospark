import Foundation

extension AppHookExecutionEngine {
    func executeCommandHook(
        hook: AppHookCommand,
        event: AppHookEvent,
        sessionID: String,
        toolName: String?,
        toolArguments: [String: String]?,
        toolOutput: String?,
        toolDurationSeconds: Double?,
        isError: Bool?,
        workingDirectory: String?,
        prompt: String?,
        source: String?,
        reason: String?,
        stopHookActive: Bool?,
        agentID: String?,
        agentType: String?,
        startTime: Date
    ) async -> AppHookExecutionResult {
        // **A PROJECTLESS HOOK HAS NO WORKING DIRECTORY, AND THERE IS NO
        // DEFENSIBLE DEFAULT** (state#60). This fell back to
        // `currentDirectoryPath`, which is `/` for a Finder-launched process
        // -- so a `PreToolUse` hook written to inspect "the repository" ran
        // at the filesystem root and reported on it, which is
        // `swift/CLAUDE.md` Gotcha 30's finding arriving on the hook path.
        // A hook that names `${CLAUDE_PROJECT_DIR}` is refused below; one
        // that does not still runs, with no cwd claimed.
        let cwd = (workingDirectory?.isEmpty == false) ? workingDirectory! : ""
        let payloadDict = AppHookStdinPayload.build(
            event: event,
            sessionID: sessionID,
            transcriptPath: AppHookStdinPayload.transcriptPath,
            cwd: cwd,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            toolDurationSeconds: toolDurationSeconds,
            isError: isError,
            prompt: prompt,
            source: source,
            reason: reason,
            stopHookActive: stopHookActive,
            agentID: agentID,
            agentType: agentType
        )

        guard let payloadData = try? JSONSerialization.data(withJSONObject: payloadDict, options: []) else {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: "Failed to serialize hook input payload",
                durationSeconds: 0.0
            )
        }

        // Variable substitution (${user_config.KEY}, ${CLAUDE_PROJECT_DIR})
        let store = await AppHookStore.shared
        let allOptions = await store.optionValues
        var commandText = hook.command
        let sourceID = hook.pluginName != nil ? "plugin_\(hook.pluginName!)" : (hook.sourceType == .projectConfig ? "project_config" : "user_config")
        let options = allOptions[sourceID] ?? [:]

        // The sensitive set is needed HERE, not only at the environment
        // build below (state#60). A value marked sensitive was excluded from
        // the env and then substituted straight into the `-c` string, which
        // is the process's ARGV -- visible to every `ps` on the machine for
        // the life of the hook. It reaches the hook by environment alone now.
        let sensitiveKeys: Set<String> = await {
            let groups = await store.sourceGroups
            guard let group = groups.first(where: { $0.id == sourceID }) else { return [] }
            return Set(group.optionSpecs.filter(\.isSensitive).map(\.key))
        }()

        for (k, v) in options where !sensitiveKeys.contains(k) {
            commandText = commandText.replacingOccurrences(of: "${user_config.\(k)}", with: v)
        }
        if commandText.contains("${CLAUDE_PROJECT_DIR}") {
            guard !cwd.isEmpty else {
                return AppHookExecutionResult(
                    hookID: hook.id,
                    hookName: hook.name,
                    event: event,
                    exitCode: 1,
                    stdout: "",
                    stderr: "Hook names ${CLAUDE_PROJECT_DIR} but no project is selected.",
                    durationSeconds: Date().timeIntervalSince(startTime),
                    // No verdict, so a permission event asks rather than
                    // allows (state#40).
                    outcome: .unavailable(
                        reason: "\(hook.name) needs a project directory and none is selected.")
                )
            }
            // **QUOTED** (state#60). A path with a space split into two
            // arguments -- `cd ${CLAUDE_PROJECT_DIR} && git status` in
            // `/Users/me/My Projects/app` ran `cd /Users/me/My` -- and a path
            // carrying a `;` or a backtick executed whatever followed it.
            commandText = commandText.replacingOccurrences(
                of: "${CLAUDE_PROJECT_DIR}", with: Self.shellQuoted(cwd))
        }

        // Plugin variables: the install root and the persistent data dir,
        // resolved from the plugin this hook came from. Quoted for the same
        // reason as ${CLAUDE_PROJECT_DIR} (state#60), and ALSO placed in the
        // environment below -- a path is safer by env than by ARGV, and
        // scripts that receive the hook through a wrapper need the variable
        // even when the command string never mentions it.
        var pluginRootPath: String?
        var pluginDataPath: String?
        if let pluginName = hook.pluginName {
            let projectURL = (workingDirectory?.isEmpty == false)
                ? URL(fileURLWithPath: workingDirectory!, isDirectory: true) : nil
            if let plugin = PluginManager.shared.findPlugin(named: pluginName, projectURL: projectURL) {
                pluginRootPath = plugin.directoryURL.path
                pluginDataPath = PluginManager.shared.dataDirectory(for: plugin).path
                commandText = commandText
                    .replacingOccurrences(
                        of: "${CLAUDE_PLUGIN_ROOT}", with: Self.shellQuoted(pluginRootPath!))
                    .replacingOccurrences(
                        of: "${CLAUDE_PLUGIN_DATA}", with: Self.shellQuoted(pluginDataPath!))
            }
        }

        let executableURL: URL
        let arguments: [String]
        switch hook.shell {
        case .bash:
            executableURL = URL(fileURLWithPath: "/bin/bash")
            arguments = ["-c", commandText]
        case .zsh:
            executableURL = URL(fileURLWithPath: "/bin/zsh")
            arguments = ["-c", commandText]
        case .sh:
            executableURL = URL(fileURLWithPath: "/bin/sh")
            arguments = ["-c", commandText]
        case .pwsh:
            executableURL = URL(fileURLWithPath: "/usr/local/bin/pwsh")
            arguments = ["-Command", commandText]
        }

        // Build minimal environment (PATH, HOME, LANG, TMPDIR) plus hook-specific variables.
        // Never inherit the full parent environment to prevent leaking secrets to hook scripts.
        let parentEnvironment = ProcessInfo.processInfo.environment
        var env: [String: String] = [:]
        for key in ["PATH", "HOME", "LANG", "TMPDIR"] {
            if let value = parentEnvironment[key] { env[key] = value }
        }

        env["TURBOSPARK_HOOK_EVENT"] = event.rawValue
        env["TURBOSPARK_SESSION_ID"] = sessionID
        if let toolName { env["TURBOSPARK_TOOL_NAME"] = toolName }
        if let workingDirectory {
            env["TURBOSPARK_PROJECT_DIR"] = workingDirectory
            env["CLAUDE_PROJECT_DIR"] = workingDirectory
        }
        if let pluginRootPath {
            env["CLAUDE_PLUGIN_ROOT"] = pluginRootPath
            env["TURBOSPARK_PLUGIN_ROOT"] = pluginRootPath
        }
        if let pluginDataPath {
            env["CLAUDE_PLUGIN_DATA"] = pluginDataPath
            env["TURBOSPARK_PLUGIN_DATA"] = pluginDataPath
        }

        // Every option reaches the hook by ENVIRONMENT, sensitive ones
        // included: the env is per process and unreadable by other users,
        // where ARGV is world-readable (state#60). Excluding them here was
        // half a protection, since the substitution above put them in the
        // command string anyway.
        for (k, v) in options {
            let normalized = k.uppercased().replacingOccurrences(of: "-", with: "_")
            env["TURBOSPARK_OPTION_\(normalized)"] = v
            env["CLAUDE_PLUGIN_OPTION_\(normalized)"] = v
        }

        // Claude Code's own default is 600s. `SessionEnd` is capped far
        // tighter: nothing is waiting on its result, and a slow hook there
        // would hold up chat deletion / clearing for the whole timeout.
        var timeout = hook.timeoutSeconds > 0 ? hook.timeoutSeconds : 600.0
        if event == .sessionEnd { timeout = min(timeout, 1.5) }
        let workingDirectoryURL = (workingDirectory?.isEmpty == false) ? URL(fileURLWithPath: workingDirectory!) : nil

        do {
            // `ProcessExecutor` writes stdin off this call's thread and drains
            // stdout/stderr concurrently with the timeout wait. A previous
            // synchronous `pipeIn.write(contentsOf:)` embedded the tool's own
            // output in the hook payload and could block on it alone, before
            // the timeout loop even started; and reading stdout/stderr only
            // after `waitUntilExit()` meant a hook that wrote more than one
            // pipe buffer to stdout hung until the timeout fired and was then
            // misreported as "timed out" rather than as blocked on IO.
            let result = try await ProcessExecutor.run(
                executableURL: executableURL,
                arguments: arguments,
                currentDirectoryURL: workingDirectoryURL,
                environment: env,
                stdin: payloadData,
                timeoutSeconds: timeout
            )

            if result.timedOut {
                return AppHookExecutionResult(
                    hookID: hook.id,
                    hookName: hook.name,
                    event: event,
                    exitCode: 124, // Timeout exit code
                    stdout: result.stdout,
                    stderr: "Hook execution timed out after \(timeout) seconds.",
                    durationSeconds: Date().timeIntervalSince(startTime),
                    // Not nil (state#40): a timeout is a hook that never
                    // answered, and the aggregator reads a missing answer
                    // as permission.
                    outcome: .unavailable(
                        reason: "\(hook.name) timed out after \(timeout) seconds.")
                )
            }

            let duration = Date().timeIntervalSince(startTime)
            let outcome = AppHookResponseParser.parseHookOutput(
                stdout: result.stdout,
                stderr: result.stderr,
                exitCode: result.exitCode,
                event: event
            )

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: result.exitCode,
                stdout: result.stdout,
                stderr: result.stderr,
                durationSeconds: duration,
                outcome: outcome
            )
        } catch {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: error.localizedDescription,
                durationSeconds: Date().timeIntervalSince(startTime),
                // A shell that could not be spawned (state#40).
                outcome: .unavailable(
                    reason: "\(hook.name) could not run: \(error.localizedDescription)")
            )
        }
    }

    /// Wraps a string so `/bin/sh -c` reads it as one literal argument.
    ///
    /// Single quotes, with an embedded `'` spelled `'\\''` -- the only form
    /// safe for every byte, since nothing inside single quotes is special.
    static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }

    func executeHttpHook(
        hook: AppHookCommand,
        event: AppHookEvent,
        sessionID: String,
        toolName: String?,
        toolArguments: [String: String]?,
        toolOutput: String?,
        workingDirectory: String?,
        prompt: String?,
        source: String?,
        reason: String?,
        stopHookActive: Bool?,
        agentID: String?,
        agentType: String?,
        startTime: Date
    ) async -> AppHookExecutionResult {
        guard let url = URL(string: hook.command) else {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: "Invalid webhook URL: \(hook.command)",
                durationSeconds: 0.0,
                outcome: .unavailable(reason: "\(hook.name) has an invalid webhook URL.")
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let cwd = (workingDirectory?.isEmpty == false) ? workingDirectory! : ""
        let payloadDict = AppHookStdinPayload.build(
            event: event,
            sessionID: sessionID,
            transcriptPath: AppHookStdinPayload.transcriptPath,
            cwd: cwd,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            prompt: prompt,
            source: source,
            reason: reason,
            stopHookActive: stopHookActive,
            agentID: agentID,
            agentType: agentType
        )
        request.httpBody = try? JSONSerialization.data(withJSONObject: payloadDict, options: [])

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let httpResponse = response as? HTTPURLResponse
            let statusCode = Int32(httpResponse?.statusCode ?? 200)
            let stdout = String(data: data, encoding: .utf8) ?? ""
            let isSuccess = statusCode >= 200 && statusCode < 300

            // An HTTP hook has no exit code of its own, so a non-2xx status
            // maps to "exit 1" for the purposes of the shared parser -- an
            // HTTP hook cannot use Claude Code's exit-2 blocking convention
            // at all, only the JSON `hookSpecificOutput`/`decision` fields.
            let outcome = AppHookResponseParser.parseHookOutput(
                stdout: stdout,
                stderr: isSuccess ? "" : "HTTP status \(statusCode)",
                exitCode: isSuccess ? 0 : 1,
                event: event
            )

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: isSuccess ? 0 : 1,
                stdout: stdout,
                stderr: isSuccess ? "" : "HTTP status \(statusCode)",
                durationSeconds: Date().timeIntervalSince(startTime),
                outcome: outcome
            )
        } catch {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: error.localizedDescription,
                durationSeconds: Date().timeIntervalSince(startTime),
                outcome: .unavailable(
                    reason: "\(hook.name) could not be reached: \(error.localizedDescription)")
            )
        }
    }
}
