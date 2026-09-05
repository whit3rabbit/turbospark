import Foundation
import TurboSpark

/// Running one tool call.
///
/// The VOCABULARY -- which tools exist, what category each falls in, which
/// ones need a workspace root -- is `AppToolRegistry+Vocabulary`, and the
/// value types are `AppToolTypes`. What is here is the switch that turns a
/// parsed call into a result, which is the part with the arguments, the
/// errors and the refusals in it.
public enum AppToolRegistry {
    /// Executes a tool call asynchronously within the given project context.
    ///
    /// - Parameter chatID: the conversation the call belongs to, for tools
    ///   whose effect is per chat rather than per filesystem. `TodoWrite` is
    ///   the one today: without it, its `onTodosUpdated` callback falls back
    ///   to `selectedChatID` on the MAIN ACTOR, asynchronously, so a checklist
    ///   written by chat A's agent overwrites chat B's if the user switched
    ///   while the tool ran. Optional so the projectless and test call sites
    ///   need not invent one.
    /// - Parameter subagentDepth: how many subagents deep the caller already
    ///   is (state#47). The `agent` tool spawns another run, and nothing
    ///   carried a level, so a subagent could call `agent` which could call
    ///   `agent` without bound.
    public static func execute(
        call: AppToolCall, in project: AppProject?, chatID: UUID? = nil, subagentDepth: Int = 0
    ) async -> AppToolResult {
        let startTime = Date()

        // **NO PROJECT MEANS NO ROOT, AND THEREFORE NO FILE OR SHELL TOOL.**
        //
        // There is no defensible default here, which is what took two tries
        // to see. `FileManager.default.currentDirectoryPath` is "/" for a
        // Finder-launched process, making `resolveSecurePath`'s containment
        // check a no-op since every path is inside "/". The home directory
        // replaced it and is narrower in the way that counts least: `~`
        // holds `~/Library/Application Support`, browser profiles, SSH keys,
        // shell history and every API token on the machine, and
        // `isSensitivePath` knows about a dozen filenames out of all of that.
        // A model that asks to read a path in a projectless chat is asking
        // about a workspace the user never chose.
        //
        // Refusing by name is the honest answer: it costs a user one click
        // (pick a project) and it is the only version of this that does not
        // silently grant the model the whole account. Only the tools that
        // actually resolve a path or spawn a process are refused -- `skill`
        // and `todowrite` need no root and still work in a projectless chat,
        // which is the case the old fallback was really reaching for.
        //
        // **`workspaceRootedToolNames` IS STATIC AND A CUSTOM TOOL IS NOT IN
        // IT** (state#70). A user-defined tool resolves at the `default` arm
        // below and spawns `/bin/zsh -c` with `currentDirectoryURL` set to
        // the `/dev/null` placeholder -- so the one class of tool whose
        // command a user WRITES was the one class this refusal could not
        // name. The list stays static (it is the shipped vocabulary); the
        // custom names are resolved beside it.
        let resolvedRoot = project?.rootDirectoryURL
        let lowerName = call.name.lowercased()
        let needsWorkspaceRoot =
            workspaceRootedToolNames.contains(lowerName)
            || lowerName.hasPrefix("mcp__")
            || CustomToolManager.shared.resolveEffectiveTools(for: nil)
                .contains { $0.name.lowercased() == lowerName }
        if resolvedRoot == nil, needsWorkspaceRoot {
            let elapsed = Date().timeIntervalSince(startTime)
            return AppToolResult(
                callID: call.id,
                output: "Error: '\(call.name)' needs a project workspace. This chat has no "
                    + "project directory, so there is no root to resolve paths against and "
                    + "no command can be run. Attach a project in the sidebar first.",
                isError: true,
                durationSeconds: elapsed
            )
        }
        // Unreachable for the rooted tools above; the rootless ones never
        // read it.
        let rootURL = resolvedRoot ?? URL(fileURLWithPath: "/dev/null")

        do {
            let output: String
            switch call.name.lowercased() {
            case "list_directory", "list_dir", "ls", "glob":
                let relPath = call.arguments["path"] ?? call.arguments["pattern"] ?? "."
                SkillManager.shared.notePathTouched(relPath, projectURL: resolvedRoot)
                output = try listDirectory(relPath: relPath, rootURL: rootURL)

            case "read_file", "view_file", "cat", "fileread", "read":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] ?? call.arguments["resource"] else {
                    throw NSError(domain: "TurboSparkTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument."])
                }
                SkillManager.shared.notePathTouched(relPath, projectURL: resolvedRoot)
                let startLine = Int(call.arguments["start_line"] ?? call.arguments["offset"] ?? "")
                // Read from their OWN keys: `end_line` is an absolute bound
                // (what the schema advertises) and `limit` is a count (what
                // the Claude/OpenAI convention means). Collapsing them made
                // `end_line: 520` return 520 lines.
                let endLine = Int(call.arguments["end_line"] ?? "")
                let limit = Int(call.arguments["limit"] ?? "")
                output = try await readFile(
                    relPath: relPath, rootURL: rootURL, startLine: startLine, endLine: endLine,
                    limit: limit)

            case "write_file", "save_file", "filewrite", "write":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument."])
                }
                SkillManager.shared.notePathTouched(relPath, projectURL: resolvedRoot)
                let content = call.arguments["content"] ?? ""
                output = try await writeFile(relPath: relPath, content: content, rootURL: rootURL)

            case "edit_file", "fileedit", "edit":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'file_path' argument."])
                }
                SkillManager.shared.notePathTouched(relPath, projectURL: resolvedRoot)
                guard let oldStr = call.arguments["old_string"] ?? call.arguments["target"] ?? call.arguments["oldStr"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'old_string' argument."])
                }
                let newStr = call.arguments["new_string"] ?? call.arguments["replacement"] ?? call.arguments["newStr"] ?? ""
                let replaceAll = (call.arguments["replace_all"]?.lowercased() == "true")
                output = try await editFile(relPath: relPath, oldString: oldStr, newString: newStr, replaceAll: replaceAll, rootURL: rootURL)

            case "apply_patch", "applypatch":
                guard let patchText = call.arguments["patch_text"] ?? call.arguments["patchText"] ?? call.arguments["patch"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'patch_text' argument."])
                }
                let result = try ApplyPatchExecutor.apply(patchText: patchText, rootURL: rootURL)
                output = result.summary

            case "search_code", "grep", "search":
                guard let pattern = call.arguments["pattern"] ?? call.arguments["query"] else {
                    throw NSError(domain: "TurboSparkTool", code: 3, userInfo: [NSLocalizedDescriptionKey: "Missing 'pattern' argument."])
                }
                let relPath = call.arguments["path"] ?? "."
                output = try searchCode(pattern: pattern, relPath: relPath, rootURL: rootURL)

            case "run_command", "bash", "shell", "exec", "terminal":
                guard let command = call.arguments["command"] ?? call.arguments["cmd"] else {
                    throw NSError(domain: "TurboSparkTool", code: 4, userInfo: [NSLocalizedDescriptionKey: "Missing 'command' argument."])
                }
                let timeoutMs = Int(call.arguments["timeout"] ?? "")
                output = try await runCommand(command: command, rootURL: rootURL, timeoutMs: timeoutMs)

            case "websearch", "web_search", "search_web":
                guard let query = call.arguments["query"] ?? call.arguments["q"] ?? call.arguments["search_query"] else {
                    throw NSError(domain: "TurboSparkTool", code: 24, userInfo: [NSLocalizedDescriptionKey: "Missing 'query' argument for WebSearch tool call."])
                }
                let numResults = Int(call.arguments["num_results"] ?? call.arguments["numResults"] ?? call.arguments["limit"] ?? call.arguments["count"] ?? "")
                var allowedDomains: [String]?
                if let allowed = call.arguments["allowed_domains"] ?? call.arguments["allowedDomains"] {
                    allowedDomains = allowed.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
                }
                var blockedDomains: [String]?
                if let blocked = call.arguments["blocked_domains"] ?? call.arguments["blockedDomains"] {
                    blockedDomains = blocked.components(separatedBy: ",").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
                }
                let provider = call.arguments["provider"]
                let searchOutput = try await WebSearchExecutor.search(
                    query: query,
                    numResults: numResults,
                    allowedDomains: allowedDomains,
                    blockedDomains: blockedDomains,
                    provider: provider
                )
                output = searchOutput.formatMarkdown()

            case "webfetch", "web_fetch", "fetch_url", "read_url_content":
                guard let urlString = call.arguments["url"] ?? call.arguments["uri"] else {
                    throw NSError(domain: "TurboSparkTool", code: 23, userInfo: [NSLocalizedDescriptionKey: "Missing 'url' argument for WebFetch tool call."])
                }
                let format = call.arguments["format"] ?? "markdown"
                let timeoutSeconds = Int(call.arguments["timeout"] ?? "")
                output = try await WebFetchExecutor.fetch(url: urlString, format: format, timeout: timeoutSeconds)

            case "skill":
                guard let skillName = call.arguments["name"] ?? call.arguments["skill_name"] else {
                    throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [NSLocalizedDescriptionKey: "Missing 'name' argument for skill tool call."])
                }
                let effectiveSkills = SkillManager.shared.resolveEffectiveSkills(projectURL: project?.rootDirectoryURL)
                if let matched = effectiveSkills.first(where: { $0.name.lowercased() == skillName.lowercased() }) {
                    // `resolveEffectiveSkills` reports a skill's persisted
                    // `isEnabled`; this call site was the second half of
                    // state#12, ignoring that flag entirely and running a
                    // user-disabled skill just the same as an enabled one.
                    guard matched.isEnabled else {
                        throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [
                            NSLocalizedDescriptionKey: "Skill '\(matched.name)' is disabled and cannot be invoked."
                        ])
                    }
                    // `disable-model-invocation` was PARSED, DISPLAYED and
                    // enforced nowhere (state#48). This is the one place it
                    // can mean anything: the tool IS the model's invocation.
                    guard !(matched.manifest.disableModelInvocation ?? false) else {
                        throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [
                            NSLocalizedDescriptionKey:
                                "Skill '\(matched.name)' declares "
                                + "`disable-model-invocation: true` and can only be run by the "
                                + "user."
                        ])
                    }
                    let expanded = SkillManager.shared.substituteArguments(
                        content: matched.content,
                        arguments: call.arguments,
                        skillDirectoryURL: matched.skillDirectoryURL,
                        sessionID: nil
                    )
                    var res = "### Skill: \(matched.name) (\(matched.scope.label))\n\(expanded)"
                    if !matched.referenceFiles.isEmpty {
                        res += "\n\n*Reference Files in skill directory:* \(matched.referenceFiles.joined(separator: ", "))"
                    }
                    output = res
                } else {
                    // A skill the model may not invoke is not listed as
                    // available to it either, or the next turn simply asks
                    // for it again (state#48).
                    let available = effectiveSkills
                        .filter { $0.isEnabled && !($0.manifest.disableModelInvocation ?? false) }
                        .map { "- \($0.name): \($0.skillDescription)" }
                        .joined(separator: "\n")
                    output = "Skill '\(skillName)' was not found.\n\nAvailable skills:\n\(available.isEmpty ? "(No skills currently installed)" : available)"
                }

            case "todowrite", "todo_write":
                let res = try TodoWriteExecutor.execute(arguments: call.arguments, chatID: chatID)
                output = res.output

            case "agent", "subagent", "task":
                guard let prompt = call.arguments["prompt"] ?? call.arguments["task"] ?? call.arguments["instructions"] else {
                    throw NSError(domain: "TurboSparkTool", code: 20, userInfo: [NSLocalizedDescriptionKey: "Missing 'prompt' argument for Agent tool call."])
                }
                let subagentType = call.arguments["subagent_type"] ?? call.arguments["type"] ?? call.arguments["name"] ?? "general-purpose"
                let agentDef = AgentManager.shared.findAgent(name: subagentType, projectURL: project?.rootDirectoryURL)
                    ?? AgentManager.shared.findAgent(name: "general-purpose", projectURL: project?.rootDirectoryURL)
                    ?? AgentManager.shared.builtInAgents[0]
                guard agentDef.isEnabled else {
                    throw NSError(domain: "TurboSparkTool", code: 20, userInfo: [
                        NSLocalizedDescriptionKey: "Agent '\(agentDef.name)' is currently disabled."
                    ])
                }
                let activeSession = await activeSessionProvider?()
                let result = await SubagentRunner.run(
                    agent: agentDef, taskPrompt: prompt, session: activeSession, project: project,
                    chatID: chatID, depth: subagentDepth + 1)
                if result.status == "error" || result.status == "failed" {
                    throw NSError(domain: "TurboSparkTool", code: 21, userInfo: [NSLocalizedDescriptionKey: result.finalResponse])
                }
                output = "Subagent [\(agentDef.displayName)] completed in \(result.totalTurns) turn(s) (\(result.totalToolCalls) tool call(s), \(String(format: "%.2f", result.durationSeconds))s):\n\n\(result.finalResponse)"

            case "askuserquestion", "ask_user_question", "ask_question", "question":
                output = try AskUserQuestionExecutor.execute(arguments: call.arguments, chatID: chatID)

            case "enterplanmode", "enter_plan_mode", "plan_mode", "plan":
                output = PlanModeExecutor.enter(arguments: call.arguments, chatID: chatID)

            case "exitplanmode", "exit_plan_mode":
                output = PlanModeExecutor.exit(arguments: call.arguments, chatID: chatID)

            case "reportfindings", "report_findings", "findings":
                output = try ReportFindingsExecutor.execute(arguments: call.arguments)

            case "proposeskills", "propose_skills":
                output = try ProposeSkillsExecutor.execute(arguments: call.arguments, projectRootURL: resolvedRoot)

            case "proposegoal", "propose_goal":
                output = try ProposeGoalExecutor.execute(arguments: call.arguments)

            case "sendfeedback", "send_feedback":
                output = try SendFeedbackExecutor.execute(arguments: call.arguments)

            case "notebookedit", "notebook_edit":
                output = try await NotebookEditExecutor.execute(arguments: call.arguments, rootURL: rootURL)

            case "snip", "extract_snippet":
                output = try SnipExecutor.execute(arguments: call.arguments, rootURL: rootURL)

            case "senduserfile", "send_user_file":
                output = try SendUserFileExecutor.execute(arguments: call.arguments, rootURL: rootURL, chatID: chatID)

            case "taskcreate", "task_create", "task_add":
                output = try TaskManager.executeCreate(arguments: call.arguments, chatID: chatID)

            case "taskget", "task_get":
                output = try TaskManager.executeGet(arguments: call.arguments)

            case "tasklist", "task_list":
                output = TaskManager.executeList(arguments: call.arguments, chatID: chatID)

            case "taskupdate", "task_update":
                output = try TaskManager.executeUpdate(arguments: call.arguments, chatID: chatID)

            case "taskstop", "task_stop", "task_cancel":
                output = try TaskManager.executeStop(arguments: call.arguments, chatID: chatID)

            case "taskoutput", "task_output":
                output = try TaskManager.executeOutput(arguments: call.arguments)

            case "sleep", "delay":
                output = try await SleepExecutor.execute(arguments: call.arguments)

            case "pushnotification", "push_notification", "notify":
                output = try PushNotificationExecutor.execute(arguments: call.arguments)

            case "config", "config_tool":
                output = try ConfigToolExecutor.execute(arguments: call.arguments, project: project)

            case "ctxinspect", "ctx_inspect":
                output = CtxInspectExecutor.execute(arguments: call.arguments, chatID: chatID, project: project)

            case "enterworktree", "enter_worktree":
                output = try await WorktreeExecutor.enter(arguments: call.arguments, rootURL: rootURL)

            case "exitworktree", "exit_worktree":
                output = try await WorktreeExecutor.exit(arguments: call.arguments, rootURL: rootURL)

            case "listmcpresources", "list_mcp_resources", "list_resources":
                output = try await McpResourceExecutor.listResources(arguments: call.arguments, project: project, rootURL: rootURL)

            case "readmcpresource", "read_mcp_resource", "read_resource":
                output = try await McpResourceExecutor.readResource(arguments: call.arguments, project: project, rootURL: rootURL)

            case "call_mcp_tool", "callmcptool", "mcp_tool":
                guard let serverName = call.arguments["server"] ?? call.arguments["server_name"] else {
                    throw NSError(domain: "TurboSparkTool", code: 5, userInfo: [NSLocalizedDescriptionKey: "Missing 'server' argument for MCP tool call."])
                }
                guard let toolName = call.arguments["toolName"] ?? call.arguments["tool"] ?? call.arguments["name"] else {
                    throw NSError(domain: "TurboSparkTool", code: 6, userInfo: [NSLocalizedDescriptionKey: "Missing 'toolName' argument for MCP tool call."])
                }
                output = try await executeMcpCall(serverName: serverName, toolName: toolName, arguments: call.arguments, project: project, rootURL: rootURL)

            default:
                if call.name.contains("__") && call.name.lowercased().hasPrefix("mcp__") {
                    let parts = call.name.components(separatedBy: "__")
                    if parts.count >= 3 {
                        let serverName = parts[1]
                        let toolName = parts[2...].joined(separator: "__")
                        output = try await executeMcpCall(serverName: serverName, toolName: toolName, arguments: call.arguments, project: project, rootURL: rootURL)
                    } else {
                        throw NSError(domain: "TurboSparkTool", code: 19, userInfo: [
                            NSLocalizedDescriptionKey: "Malformed MCP tool name: '\(call.name)'."
                        ])
                    }
                } else if let customTool = CustomToolManager.shared.resolveEffectiveTools(for: project?.rootDirectoryURL).first(where: { $0.name.lowercased() == call.name.lowercased() }) {
                    guard customTool.isEnabled else {
                        throw NSError(domain: "TurboSparkTool", code: 22, userInfo: [
                            NSLocalizedDescriptionKey: "Custom tool '\(customTool.name)' is currently disabled."
                        ])
                    }
                    output = try await CustomToolExecutor.execute(tool: customTool, arguments: call.arguments, projectRootURL: rootURL)
                } else {
                    // Fabricating "Executed successfully" for a tool with no
                    // real handler let the model believe hallucinated results
                    // were verified (T5). An honest error is the only
                    // response `execute` can give for a name it does not
                    // implement.
                    throw NSError(domain: "TurboSparkTool", code: 19, userInfo: [
                        NSLocalizedDescriptionKey: "Tool '\(call.name)' is not implemented by this client and was not executed."
                    ])
                }
            }

            let elapsed = Date().timeIntervalSince(startTime)
            return AppToolResult(callID: call.id, output: output, isError: false, durationSeconds: elapsed)
        } catch is CancellationError {
            // **A STOP IS NOT A TOOL FAILURE** (state#64). `CancellationError`
            // localizes to "cancelled", so a command the USER stopped reached
            // the model as `Error: cancelled` -- indistinguishable from a
            // command that failed, which is an invitation to try again, and
            // the loop obligingly does. Said plainly, so a model that gets one
            // anyway stops rather than retries.
            let elapsed = Date().timeIntervalSince(startTime)
            return AppToolResult(
                callID: call.id,
                output: "Stopped by the user before it finished. Do not retry; wait for "
                    + "further instructions.",
                isError: true,
                durationSeconds: elapsed)
        } catch {
            let elapsed = Date().timeIntervalSince(startTime)
            return AppToolResult(callID: call.id, output: "Error: \(error.localizedDescription)", isError: true, durationSeconds: elapsed)
        }
    }

    private static func executeMcpCall(
        serverName: String,
        toolName: String,
        arguments: [String: String],
        project: AppProject?,
        rootURL: URL
    ) async throws -> String {
        // **A GLOBAL SERVER WINS A NAME COLLISION** (state#61). This put
        // PROJECT servers first, and `first(where:)` takes the first match --
        // so a `.mcp.json` in a cloned repository could take the name of a
        // server the user configured globally and be dialled in its place,
        // with a different command, while the approval card showed only the
        // name. Precedence is the opposite of the skills rule on purpose:
        // there, a project overriding a user skill is the FEATURE; here the
        // thing being overridden is a command the user chose to trust.
        let globalServers = GlobalMcpFileStore.load().servers
        let projectServers = project?.mcpServers ?? []
        let allServers = globalServers + projectServers

        guard let matchedServer = allServers.first(where: { $0.name.lowercased() == serverName.lowercased() }) else {
            throw NSError(domain: "TurboSparkTool", code: 7, userInfo: [NSLocalizedDescriptionKey: "MCP server '\(serverName)' not found in project or global configurations."])
        }

        guard matchedServer.isEnabled else {
            throw NSError(domain: "TurboSparkTool", code: 8, userInfo: [NSLocalizedDescriptionKey: "MCP server '\(serverName)' is currently disabled."])
        }

        return try await McpClientEngine.shared.callTool(
            config: matchedServer,
            toolName: toolName,
            arguments: arguments,
            workingDirectory: rootURL
        )
    }
}
