import Foundation
import TurboSpark

/// Result of an isolated subagent execution run.
public struct SubagentRunResult: Sendable, Equatable {
    public var agentName: String
    public var status: String
    public var finalResponse: String
    public var totalTurns: Int
    public var totalToolCalls: Int
    public var durationSeconds: Double

    public init(
        agentName: String,
        status: String = "completed",
        finalResponse: String,
        totalTurns: Int,
        totalToolCalls: Int,
        durationSeconds: Double
    ) {
        self.agentName = agentName
        self.status = status
        self.finalResponse = finalResponse
        self.totalTurns = totalTurns
        self.totalToolCalls = totalToolCalls
        self.durationSeconds = durationSeconds
    }
}

/// Executes subagent tasks in a clean, isolated context without parent conversation history.
public enum SubagentRunner {
    /// How many subagents may nest. Two, so an agent may delegate once and
    /// what it delegates to may not (state#47).
    public static let maxSubagentDepth = 2

    /// Builds the isolated system prompt for a subagent run.
    public static func buildSystemPrompt(for agent: AppAgentDefinition, project: AppProject?) -> String {
        var sections: [String] = []

        // 1. Agent's specialized role & instructions
        sections.append("## Subagent Role: \(agent.displayName)\n\(agent.systemPrompt)")

        // 2. Project workspace environment if attached
        if let project {
            if let root = project.rootDirectoryPath, !root.isEmpty {
                sections.append("## Workspace Environment\nRoot codebase directory: `\(root)`")
            }
            if !project.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let trimmedRules = project.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
                sections.append("## Project Specific Context\n\(trimmedRules)")
            }
        }

        // 3. Filtered Tool definitions
        //
        // **THE PROJECT'S SLICE, NOT THE WHOLE CATALOG** (state#47). This
        // read `AppToolCatalog.allTools`, so a subagent under an agent
        // profile with no `allowed-tools` clause was TOLD it had every tool
        // the app implements -- including ones its own project's agent type
        // does not offer. It then proposed them, `observation` refused them
        // one at a time, and the run burned its turn budget on calls that
        // were never available. `tools(for:)` is the same slice the main
        // loop advertises.
        let allTools = AppToolCatalog.tools(for: project?.agentType ?? .coder)
        let allowed = allTools.filter { agent.isToolAllowed($0.function.name) }

        if !allowed.isEmpty {
            var lines: [String] = []
            lines.append("## Available Tools")
            lines.append("You have access to the following developer tools:")
            for tool in allowed {
                lines.append("- `\(tool.function.name)`: \(tool.function.description)")
            }
            lines.append("")
            lines.append("To invoke a tool, output a tool call block:")
            lines.append("<tool_call>")
            lines.append("<name>tool_name</name>")
            lines.append("<arguments>{\"key\": \"value\"}</arguments>")
            lines.append("</tool_call>")
            sections.append(lines.joined(separator: "\n"))
        }

        return sections.joined(separator: "\n\n")
    }

    /// Executes an isolated subagent run to completion.
    /// - Parameter chatID: the conversation this run belongs to, threaded to
    ///   `AppToolRegistry.execute` (state#47). Without it `TodoWrite`'s
    ///   callback falls back to `selectedChatID` on the main actor,
    ///   asynchronously -- the hazard `AppTool.swift`'s own doc describes,
    ///   reached by the one caller that passed nothing.
    /// - Parameter depth: how many subagents deep this run already is. An
    ///   `agent` tool call from inside a subagent used to nest without any
    ///   bound at all, each level free to spawn its own.
    public static func run(
        agent: AppAgentDefinition,
        taskPrompt: String,
        session: TurboSparkSession?,
        project: AppProject?,
        chatID: UUID? = nil,
        depth: Int = 0,
        maxTurnsOverride: Int? = nil
    ) async -> SubagentRunResult {
        let startTime = Date()
        let maxTurns = maxTurnsOverride ?? agent.maxTurns
        guard depth <= maxSubagentDepth else {
            return SubagentRunResult(
                agentName: agent.name,
                status: "failed",
                finalResponse:
                    "Refused: subagents may nest at most \(maxSubagentDepth) deep and this run "
                    + "is already at \(depth). Run this task from the main conversation.",
                totalTurns: 0,
                totalToolCalls: 0,
                durationSeconds: Date().timeIntervalSince(startTime)
            )
        }

        // Verify session availability
        guard let session else {
            let duration = Date().timeIntervalSince(startTime)
            return SubagentRunResult(
                agentName: agent.name,
                status: "failed",
                finalResponse: "Error: No active model session available to execute subagent task.",
                totalTurns: 0,
                totalToolCalls: 0,
                durationSeconds: duration
            )
        }

        // Fresh, isolated history: zero parent message context
        var history: [ChatMessage] = []
        let sysPrompt = buildSystemPrompt(for: agent, project: project)
        history.append(ChatMessage(role: .system, content: sysPrompt))
        history.append(ChatMessage(role: .user, content: taskPrompt))

        var totalToolCalls = 0
        var currentTurn = 0
        var finalContent = ""
        /// How many turns `fitWindow` dropped across the whole run (state#75).
        var droppedTurns = 0

        while currentTurn < maxTurns {
            // `session.cancel()` ends the CURRENT stream cleanly rather than
            // throwing, so without this the loop simply started its next turn
            // and Stop looked like it had done nothing.
            if Task.isCancelled {
                return SubagentRunResult(
                    agentName: agent.name,
                    status: "cancelled",
                    finalResponse: finalContent.isEmpty
                        ? "Subagent run was cancelled." : finalContent,
                    totalTurns: currentTurn,
                    totalToolCalls: totalToolCalls,
                    durationSeconds: Date().timeIntervalSince(startTime)
                )
            }
            currentTurn += 1

            var options = GenerateOptions()
            options.temperature = 0.2
            options.maxNewTokens = 2048

            var generatedText = ""

            do {
                // **THE PROMPT BUDGET IS THE WINDOW MINUS WHAT GENERATION
                // NEEDS** (state#75, which is state#36 on this path). This
                // called `fitWindow` at its default bound -- the whole
                // `maxContext` -- so a history that "fits" leaves no room for
                // the reply, and the outcome was discarded entirely: a
                // truncated history was sent with nothing said about it, and
                // a no-room verdict still called `generate`, which clamps to
                // one token or throws a context overflow naming neither
                // cause. A subagent's history grows by a whole tool result
                // per turn, so it reaches the bound faster than a chat does.
                let promptBudget = session.info.maxContext > options.maxNewTokens
                    ? session.info.maxContext - options.maxNewTokens
                    : session.info.maxContext
                let fitted = try await session.fitWindow(
                    history, maxTokens: promptBudget, reasoning: .off)
                guard fitted.hasRoomForGeneration else {
                    return SubagentRunResult(
                        agentName: agent.name,
                        status: "context_overflow",
                        finalResponse:
                            "Subagent stopped: its context no longer fits. "
                            + "\(fitted.measuredTokens) prompt tokens against a "
                            + "\(session.info.maxContext)-token window with "
                            + "\(options.maxNewTokens) reserved for the reply. The text below is "
                            + "its last turn and is not a finished answer.\n\n\(finalContent)",
                        totalTurns: currentTurn,
                        totalToolCalls: totalToolCalls,
                        durationSeconds: Date().timeIntervalSince(startTime)
                    )
                }
                droppedTurns += fitted.removedTurnCount
                for try await event in session.generate(fitted.retained, options: options) {
                    try Task.checkCancellation()
                    if case .content(let chunk) = event {
                        generatedText += chunk
                    }
                }
            } catch {
                let duration = Date().timeIntervalSince(startTime)
                return SubagentRunResult(
                    agentName: agent.name,
                    status: "error",
                    finalResponse: "Subagent generation error: \(error.localizedDescription)",
                    totalTurns: currentTurn,
                    totalToolCalls: totalToolCalls,
                    durationSeconds: duration
                )
            }

            finalContent = generatedText
            let parsedCalls = extractToolCalls(
                from: generatedText, projectURL: project?.rootDirectoryURL)

            if parsedCalls.isEmpty {
                // Completed prose answer with no more tool calls
                break
            }

            // Append assistant response to isolated history
            history.append(ChatMessage(role: .assistant, content: generatedText))

            // Execute parsed tool calls
            for call in parsedCalls {
                totalToolCalls += 1
                history.append(
                    await observation(
                        for: call, agent: agent, project: project, chatID: chatID, depth: depth))
            }
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        // **A RUN THAT RAN OUT OF TURNS DID NOT COMPLETE.** `finalContent` here
        // is the last turn's raw text, which on this exit is a tool call the
        // loop never got to execute -- reported as `completed` it reaches the
        // parent as an answer, with unexecuted XML as its content.
        let exhausted = currentTurn >= maxTurns && !extractToolCalls(from: finalContent, projectURL: project?.rootDirectoryURL).isEmpty
        // A run whose own history was truncated says so (state#75). Dropping
        // turns is a legitimate outcome and a silent one is indistinguishable
        // from a subagent that simply forgot what it had already done.
        let truncationNote =
            droppedTurns > 0
            ? "[This run dropped \(droppedTurns) earlier "
                + "\(droppedTurns == 1 ? "turn" : "turns") to fit the context window.]\n\n"
            : ""
        return SubagentRunResult(
            agentName: agent.name,
            status: exhausted ? "max_turns" : "completed",
            finalResponse: truncationNote
                + (exhausted
                    ? "Subagent stopped after its \(maxTurns)-turn limit with work still in "
                        + "progress; the text below is its last turn and is not a finished "
                        + "answer.\n\n\(finalContent)"
                    : finalContent),
            totalTurns: currentTurn,
            totalToolCalls: totalToolCalls,
            durationSeconds: totalDuration
        )
    }

    /// Runs one proposed call and returns the observation to feed back.
    ///
    /// Split out of the loop so the GATE'S CALL SITE is testable and not just
    /// the gate: `run` needs a live model session, so a test over
    /// `permissionRefusal` alone stays green with the check deleted from the
    /// loop entirely, which is the exact defect this is guarding.
    ///
    /// **A SUBAGENT'S TOOL CALLS RUN THE LIFECYCLE HOOKS TOO** (state#68).
    /// This gated on `isToolAllowed`, `permissionRefusal` and the depth
    /// counter and then executed, while the main loop additionally runs
    /// `PreToolUse` and `PostToolUse` (`AppModel+AgentLoop.swift`). A deny
    /// hook -- which state#40 made fail CLOSED precisely so it can be relied
    /// on -- was therefore bypassed on the one path that runs unattended for
    /// `maxTurns` turns. `PermissionRequest` is deliberately NOT dispatched:
    /// it fires where the approval card would go up, and a subagent refuses
    /// `.ask` rather than surfacing one (state#18).
    ///
    /// The hook store is not rebound here: it follows the project, and a
    /// subagent runs under the project of the turn that reached it, which
    /// `AppModel`'s own dispatch already pointed it at (state#67).
    static func observation(
        for call: AppToolCall, agent: AppAgentDefinition, project: AppProject?,
        chatID: UUID? = nil, depth: Int = 0
    ) async -> ChatMessage {
        if !agent.isToolAllowed(call.name) {
            return errorObservation(
                "Tool '\(call.name)' is disallowed for agent profile '\(agent.name)'.")
        }
        // **THE NESTING BOUND IS ENFORCED WHERE THE CALL IS SEEN, NOT WHERE
        // THE RUN STARTS** (state#47). `run`'s own guard catches a run that
        // was started too deep; this catches the call that would start it,
        // and reports the reason to the model rather than letting it read a
        // failed subagent as a tool that broke.
        if call.name.lowercased() == "agent" || call.name.lowercased() == "task" {
            guard depth < maxSubagentDepth else {
                return errorObservation(
                    "Refused: subagents may nest at most \(maxSubagentDepth) deep and this one "
                        + "is already at \(depth). Do the work yourself or report back.")
            }
        }

        let sessionID = chatID?.uuidString ?? "subagent"
        let projectDirectory = project?.rootDirectoryPath
        let hookDecision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: sessionID,
            toolName: call.name,
            toolArguments: call.arguments,
            workingDirectory: projectDirectory)

        // `updatedInput` is applied BEFORE the permission gate, not after:
        // the rewritten command is what would run, so it is what has to be
        // evaluated. The main loop makes the same ordering choice.
        var call = call
        if let updated = hookDecision.updatedInput {
            for (key, value) in updated { call.arguments[key] = value }
        }
        if hookDecision.behavior == .deny {
            let reason = hookDecision.reason ?? "Blocked by PreToolUse hook"
            return errorObservation("Tool execution blocked by hook: \(reason)")
        }
        if hookDecision.behavior == .ask {
            let reason = hookDecision.reason ?? "a PreToolUse hook requested confirmation"
            return errorObservation(
                "Tool '\(call.name)' needs interactive approval (\(reason)), and a subagent runs "
                    + "with no approval UI. Ask the user to run this call in the main "
                    + "conversation.")
        }
        if let refusal = permissionRefusal(for: call, project: project) {
            return errorObservation(refusal)
        }

        let toolResult = await AppToolRegistry.execute(
            call: call, in: project, chatID: chatID, subagentDepth: depth)

        var results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: sessionID,
            toolName: call.name,
            toolArguments: call.arguments,
            toolOutput: toolResult.output,
            toolDurationSeconds: toolResult.durationSeconds,
            isError: toolResult.isError,
            workingDirectory: projectDirectory)
        if toolResult.isError {
            results += await AppHookExecutionEngine.shared.dispatch(
                event: .postToolUseFailure,
                sessionID: sessionID,
                toolName: call.name,
                toolArguments: call.arguments,
                toolOutput: toolResult.output,
                toolDurationSeconds: toolResult.durationSeconds,
                isError: true,
                workingDirectory: projectDirectory)
        }
        let postVerdict = AppHookDecisionAggregator.aggregate(results, event: .postToolUse)
        var output = toolResult.output
        // Feedback, never a block: the tool already ran. Same folding the
        // main loop's `runApprovedCall` does.
        if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
            output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
        }
        if let ctx = postVerdict.additionalContext, !ctx.isEmpty {
            output += "\n\n<hook_context>\n\(ctx)\n</hook_context>"
        }

        let tag = toolResult.isError ? "tool_error" : "tool_response"
        return ChatMessage(role: .tool, content: "<\(tag)>\n\(output)\n</\(tag)>")
    }

    /// A refusal fed back to the subagent.
    ///
    /// **`.tool`, NOT `.system`** (state#74, which is state#32 on this path).
    /// The Gemma, ChatML and DeepSeek fallback renderers refuse a mid-history
    /// system message outright and `fit_window` prices a failing render at
    /// `u64::MAX`, so it drops turns until the render stops failing -- the run
    /// silently loses its own history rather than reporting anything.
    static func errorObservation(_ message: String) -> ChatMessage {
        ChatMessage(role: .tool, content: "<tool_error>\n\(message)\n</tool_error>")
    }

    /// The reason a subagent may not run `call`, or nil when it may.
    ///
    /// **A SUBAGENT IS NOT EXEMPT FROM THE PERMISSION GATE** (state#18). This loop went
    /// straight to `AppToolRegistry.execute` after checking only
    /// `agent.isToolAllowed`, which is a tool NAME list -- and the built-in
    /// `general-purpose` agent declares no `disallowedTools` at all. So under
    /// a project whose terminal permission is `.ask` or `.deny`, a subagent
    /// reached from the `agent` tool or from `/explore` ran `/bin/zsh -c`
    /// unprompted, up to `maxTurns` times, on the strength of ONE approval of
    /// the agent call itself (swift/CLAUDE.md Gotchas 11 and 29).
    ///
    /// An isolated run has no UI to prompt with, so `.ask` DENIES rather than
    /// surfacing a card: the alternative is a hidden prompt nobody answers, or
    /// worse, treating "would have asked" as "may proceed". The terminal
    /// allowlist runs on top of that, so an auto-mode project still only
    /// auto-runs what `isAutoApprovable` accepts.
    static func permissionRefusal(for call: AppToolCall, project: AppProject?) -> String? {
        switch AppToolPermissionEngine.evaluate(call: call, project: project, sessionApproved: false) {
        case .deny(let reason):
            return "Tool '\(call.name)' is denied by project permissions: \(reason)"
        case .ask(_, let reason):
            return "Tool '\(call.name)' needs interactive approval (\(reason)), and a subagent "
                + "runs with no approval UI. Ask the user to run this call in the main "
                + "conversation, or widen the project's permissions."
        case .allow:
            break
        }

        // The positive gate from swift/CLAUDE.md Gotcha 29. It was written
        // because `permissive` returned `.allow` from `evaluate` before the
        // high-risk gate ran, so that mode alone would let a subagent run
        // `rm -rf ~` unattended; state#46 moved the mode below that gate, so
        // the engine no longer has the hole this was compensating for.
        //
        // **KEPT ANYWAY, AND NOT AS BELT-AND-BRACES.** A subagent cannot ask,
        // so `.ask` is a refusal here rather than a prompt, and the engine's
        // `.auto` and `.permissive` arms both return `.allow` for everything
        // the DENYLIST does not score `.high`. Gotcha 29 records 18 of 23
        // corpus strings surviving that denylist, so the positive allowlist
        // is what actually bounds an unattended shell -- a different question
        // from the one `evaluate` answers.
        if call.category == .terminal,
            let command = call.arguments["command"] ?? call.arguments["cmd"],
            !TerminalCommandClassifier.isAutoApprovable(command)
        {
            return "Command '\(command)' is not on the auto-approvable allowlist and a subagent "
                + "cannot ask. Run it in the main conversation instead."
        }
        return nil
    }

    // MARK: - Tool Call Parsing Helper

    /// - Parameter projectURL: the workspace root, so a PROJECT-scoped
    ///   custom tool is classified under its own declared category rather
    ///   than under the `default` arm (state#71).
    public static func extractToolCalls(
        from text: String, projectURL: URL? = nil
    ) -> [AppToolCall] {
        var calls: [AppToolCall] = []

        // XML Format: <tool_call> ... </tool_call>
        let xmlPattern = "<tool_call>([\\s\\S]*?)</tool_call>"
        if let xmlRegex = try? NSRegularExpression(pattern: xmlPattern, options: []) {
            let nsString = text as NSString
            let matches = xmlRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))
            for match in matches {
                guard match.numberOfRanges > 1 else { continue }
                let inner = nsString.substring(with: match.range(at: 1))
                let raw = nsString.substring(with: match.range(at: 0))

                var toolName = ""
                if let nameRegex = try? NSRegularExpression(pattern: "<name>([\\s\\S]*?)</name>", options: []),
                   let nameMatch = nameRegex.firstMatch(in: inner, options: [], range: NSRange(location: 0, length: (inner as NSString).length)) {
                    toolName = (inner as NSString).substring(with: nameMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                }

                var arguments: [String: String] = [:]
                if let argsRegex = try? NSRegularExpression(pattern: "<arguments>([\\s\\S]*?)</arguments>", options: []),
                   let argsMatch = argsRegex.firstMatch(in: inner, options: [], range: NSRange(location: 0, length: (inner as NSString).length)) {
                    let argsString = (inner as NSString).substring(with: argsMatch.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    arguments = parseJSONArguments(argsString)
                }

                if !toolName.isEmpty {
                    let category = AppToolRegistry.category(for: toolName, projectURL: projectURL)
                    let risk = ToolRiskClassifier.assessRisk(name: toolName, arguments: arguments)
                    calls.append(AppToolCall(
                        name: toolName,
                        arguments: arguments,
                        rawInvocation: raw,
                        status: .pendingApproval,
                        category: category,
                        riskAssessment: risk
                    ))
                }
            }
        }

        // Markdown block format fallback: ```tool_call ... ```
        if calls.isEmpty {
            let mdPattern = "```(?:tool_call|json_tool_call)\\s*\\n([\\s\\S]*?)\\n```"
            if let mdRegex = try? NSRegularExpression(pattern: mdPattern, options: []) {
                let nsString = text as NSString
                let matches = mdRegex.matches(in: text, options: [], range: NSRange(location: 0, length: nsString.length))
                for match in matches {
                    guard match.numberOfRanges > 1 else { continue }
                    let inner = nsString.substring(with: match.range(at: 1)).trimmingCharacters(in: .whitespacesAndNewlines)
                    let raw = nsString.substring(with: match.range(at: 0))

                    if let data = inner.data(using: .utf8),
                       let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       let toolName = (json["name"] as? String) ?? (json["tool"] as? String) {
                        var args: [String: String] = [:]
                        if let rawArgs = json["arguments"] as? [String: Any] {
                            for (k, v) in rawArgs {
                                args[k] = "\(v)"
                            }
                        }
                        let category = AppToolRegistry.category(for: toolName, projectURL: projectURL)
                        let risk = ToolRiskClassifier.assessRisk(name: toolName, arguments: args)
                        calls.append(AppToolCall(
                            name: toolName,
                            arguments: args,
                            rawInvocation: raw,
                            status: .pendingApproval,
                            category: category,
                            riskAssessment: risk
                        ))
                    }
                }
            }
        }

        return calls
    }

    private static func parseJSONArguments(_ string: String) -> [String: String] {
        guard let data = string.data(using: .utf8),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return [:]
        }
        var result: [String: String] = [:]
        for (key, val) in json {
            if let str = val as? String {
                result[key] = str
            } else if let num = val as? NSNumber {
                result[key] = num.stringValue
            } else {
                result[key] = "\(val)"
            }
        }
        return result
    }
}
