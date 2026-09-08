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
    /// Identifier emitted with the `SubagentStart` / `SubagentStop` hooks
    /// and reported to the parent in the result trailer, so a caller can
    /// refer back to a specific run.
    public var runID: String

    public init(
        agentName: String,
        status: String = "completed",
        finalResponse: String,
        totalTurns: Int,
        totalToolCalls: Int,
        durationSeconds: Double,
        runID: String = ""
    ) {
        self.agentName = agentName
        self.status = status
        self.finalResponse = finalResponse
        self.totalTurns = totalTurns
        self.totalToolCalls = totalToolCalls
        self.durationSeconds = durationSeconds
        self.runID = runID
    }
}

/// Executes subagent tasks in a clean, isolated context without parent conversation history.
public enum SubagentRunner {
    /// How many subagents may nest. Two, so an agent may delegate once and
    /// what it delegates to may not (state#47).
    public static let maxSubagentDepth = 2

    /// Builds the isolated system prompt for a subagent run.
    ///
    /// **THIS IS THE SECOND ASSEMBLER** and it shares no code with
    /// `AppModel.buildSystemPrompt`. Anything added to one and not the other
    /// silently applies to half this app's runs, which is already true of the
    /// skills listing. `userPrompt` is threaded in rather than read because
    /// this type is an `enum` with no `AppModel` to ask.
    ///
    /// A subagent inherits the app-wide DEFAULT only, never a per-chat
    /// override: a subagent runs in a fresh isolated context with zero parent
    /// history, so a prompt scoped to one conversation is not its scope.
    public static func buildSystemPrompt(
        for agent: AppAgentDefinition,
        project: AppProject?,
        userPrompt: String = ""
    ) -> String {
        var sections: [String] = []

        // 0. The user's own deployment-wide instructions, ahead of the agent
        //    role for the same reason they lead the main assembler: the role
        //    is a refinement of them, not a competitor.
        let trimmedUserPrompt = userPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedUserPrompt.isEmpty {
            sections.append(trimmedUserPrompt)
        }

        // 1. Agent's specialized role & instructions
        sections.append("## Subagent Role: \(agent.displayName)\n\(agent.systemPrompt)")

        // 2. Project workspace environment if attached
        if let project {
            if let root = project.rootDirectoryPath, !root.isEmpty {
                sections.append("## Workspace Environment\nRoot codebase directory: `\(root)`")
            }
            if !project.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                // An agent may opt out of the project's own instructions
                // (`omitsProjectInstructions`): Claude Code's Explore sets
                // `omitClaudeMd` because a fast search agent does not need
                // them and they can be long. DELIBERATE, so it does not read
                // as the divergence this file's header warns of: the flag
                // lives on the agent definition, and only `explore` sets it.
                if !agent.omitsProjectInstructions {
                    let trimmedRules = project.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines)
                    sections.append("## Project Specific Context\n\(trimmedRules)")
                }
            }
            // The same memory section the main assembler appends, from the
            // same builder. A subagent that could not see or save memories
            // would diverge from its parent about what this project already
            // knows -- the divergence this file's header exists to warn of.
            if MemoryStore.shared.isModelEnabled, let rootURL = project.rootDirectoryURL, !rootURL.path.isEmpty {
                sections.append(MemoryPromptBuilder.section(store: MemoryStore.shared, projectRoot: rootURL))
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
        let allTools = AppToolCatalog.tools(
            for: project?.agentType ?? .coder,
            projectURL: project?.rootDirectoryURL)
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

        // 4. Discovered MCP tools, under the agent's own allow-list too. The
        // main loop advertises these through `systemPromptAddendum`; a
        // subagent without them would propose `mcp__` calls the parent
        // demonstrably has, or never use servers the project enabled.
        if let project {
            let servers = AppToolCatalogMcp.visibleServers(
                global: GlobalMcpFileStore.load().servers, project: project)
            let mcpDefinitions = AppToolCatalogMcp.toolDefinitions(
                servers: servers, permissions: project.permissions)
                .filter { agent.isToolAllowed($0.function.name) }
            if !mcpDefinitions.isEmpty {
                var lines: [String] = [
                    "## MCP Server Tools",
                    "Tools discovered from connected MCP servers. Call them by their full `mcp__<server>__<tool>` name; each entry lists its arguments:",
                ]
                for tool in mcpDefinitions {
                    lines.append("- `\(tool.function.name)`: \(tool.function.description)")
                }
                sections.append(lines.joined(separator: "\n"))
            }
        }

        return sections.joined(separator: "\n\n")
    }

    /// Executes an isolated subagent run to completion, wrapped in the
    /// `SubagentStart` / `SubagentStop` lifecycle hooks. Notification-grade
    /// events: matching hooks run and their feedback is ignored, because a
    /// subagent has no interaction surface to resolve a block with (the same
    /// reason `.ask` denies on its tool path).
    ///
    /// The wrapper exists because `run`'s body exits early from five places
    /// (depth, no session, cancellation, context overflow, generation error);
    /// dispatching around it is the only shape that covers every one.
    /// - Parameter chatID: the conversation this run belongs to, threaded to
    ///   `AppToolRegistry.execute` (state#47). Without it `TodoWrite`'s
    ///   callback falls back to `selectedChatID` on the main actor,
    ///   asynchronously -- the hazard `AppTool.swift`'s own doc describes,
    ///   reached by the one caller that passed nothing.
    /// - Parameter depth: how many subagents deep this run already is. An
    ///   `agent` tool call from inside a subagent used to nest without any
    ///   bound at all, each level free to spawn its own.
    /// - Parameter progress: optional observer for the run's loop seams
    ///   (`SubagentProgressEvent`). Called and awaited in order from the
    ///   run's own task; a slow observer slows the run, which is what keeps
    ///   the events ordered. Nothing here reads the observer's result -- a
    ///   run is never steered by who is watching it.
    /// - Parameter taskDescription: the caller's short summary of the task,
    ///   reported in the `started` progress event. Prose for a card header,
    ///   never part of the prompt.
    public static func run(
        agent: AppAgentDefinition,
        taskPrompt: String,
        session: TurboSparkSession?,
        project: AppProject?,
        chatID: UUID? = nil,
        depth: Int = 0,
        maxTurnsOverride: Int? = nil,
        userSystemPrompt: String = "",
        samplingOptions: GenerateOptions = GenerateOptions(),
        progress: (@Sendable (SubagentProgressEvent) async -> Void)? = nil,
        taskDescription: String = ""
    ) async -> SubagentRunResult {
        let sessionID = chatID?.uuidString ?? "subagent"
        let runID = UUID().uuidString
        let workingDirectory = project?.rootDirectoryPath
        await progress?(.started(
            agentName: agent.name, displayName: agent.displayName,
            taskDescription: taskDescription, promptHead: head(of: taskPrompt),
            chatID: chatID))
        // The engine directly, exactly like `observation`: the hook store is
        // not rebound here (state#67 -- it follows the parent turn's
        // project, and this type has no AppModel to rebind it with).
        _ = await AppHookExecutionEngine.shared.dispatch(
            event: .subagentStart,
            sessionID: sessionID,
            workingDirectory: workingDirectory,
            agentID: runID,
            agentType: agent.name)

        let bodyResult = await runBody(
            agent: agent, taskPrompt: taskPrompt, session: session, project: project,
            chatID: chatID, depth: depth, maxTurnsOverride: maxTurnsOverride,
            userSystemPrompt: userSystemPrompt, samplingOptions: samplingOptions,
            progress: progress, runID: runID)
        var result = bodyResult
        result.runID = runID

        _ = await AppHookExecutionEngine.shared.dispatch(
            event: .subagentStop,
            sessionID: sessionID,
            workingDirectory: workingDirectory,
            stopHookActive: false,
            agentID: runID,
            agentType: agent.name)
        await progress?(.finished(status: result.status))
        return result
    }

    /// Bounded head of a task prompt for a progress card header. The full
    /// prompt stays in the run; a card shows enough to recognize it by.
    private static func head(of prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 300 else { return trimmed }
        return String(trimmed.prefix(300)) + "..."
    }

    /// The `agent` tool's result text for one finished run: the final
    /// answer, then a `<subagent_meta>` trailer (Claude Code's result
    /// contract -- the parent reads the trailer, not the UI). An empty
    /// answer becomes the explicit no-output sentence rather than silence,
    /// which a model would read as "the tool produced nothing".
    public static func toolOutput(for result: SubagentRunResult, agentName: String) -> String {
        let rawBody = result.finalResponse.trimmingCharacters(in: .whitespacesAndNewlines)
        let safeBody = escapingWrapperTags(["subagent_meta"], in: rawBody)
        let body = safeBody.isEmpty ? "(Subagent completed but returned no output.)" : safeBody
        return body
            + "\n\n<subagent_meta>\n"
            + "agent: \(agentName)\n"
            + "id: \(result.runID)\n"
            + "status: \(result.status)\n"
            + "turns: \(result.totalTurns)\n"
            + "tool_calls: \(result.totalToolCalls)\n"
            + "duration_s: \(String(format: "%.1f", result.durationSeconds))\n"
            + "</subagent_meta>"
    }

    /// The run itself, without the lifecycle dispatch (see `run`).
    private static func runBody(
        agent: AppAgentDefinition,
        taskPrompt: String,
        session: TurboSparkSession?,
        project: AppProject?,
        chatID: UUID? = nil,
        depth: Int = 0,
        maxTurnsOverride: Int? = nil,
        userSystemPrompt: String = "",
        samplingOptions: GenerateOptions = GenerateOptions(),
        progress: (@Sendable (SubagentProgressEvent) async -> Void)? = nil,
        runID: String
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
        let sysPrompt = buildSystemPrompt(
            for: agent, project: project, userPrompt: userSystemPrompt)
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
            await progress?(.turnStarted(number: currentTurn))

            // The caller's sampling preferences (temperature, top-k, top-p,
            // repetition penalty, seed, stop sequences), which every subagent
            // turn ignored until this parameter existed
            // (`swift/docs/SWIFT_SETTINGS_AUDIT.md`). `maxNewTokens` is still this
            // run's OWN turn budget rather than the interactive chat's reply
            // length: a subagent runs many turns of tool use, which is a
            // different quantity than one user-facing answer.
            var options = samplingOptions
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
                    switch event {
                    case .content(let chunk):
                        generatedText += chunk
                        await progress?(.content(chunk))
                    case .reasoning(let chunk):
                        await progress?(.content(chunk))
                    default:
                        break
                    }
                }
            } catch is CancellationError {
                let duration = Date().timeIntervalSince(startTime)
                return SubagentRunResult(
                    agentName: agent.name,
                    status: "cancelled",
                    finalResponse: finalContent.isEmpty
                        ? "Subagent run was cancelled." : finalContent,
                    totalTurns: currentTurn,
                    totalToolCalls: totalToolCalls,
                    durationSeconds: duration
                )
            } catch {
                let duration = Date().timeIntervalSince(startTime)
                if Task.isCancelled {
                    return SubagentRunResult(
                        agentName: agent.name,
                        status: "cancelled",
                        finalResponse: finalContent.isEmpty
                            ? "Subagent run was cancelled." : finalContent,
                        totalTurns: currentTurn,
                        totalToolCalls: totalToolCalls,
                        durationSeconds: duration
                    )
                }
                // **AN ERROR KEEPS THE PROGRESS IT HAD** (Claude Code's
                // partial-result rule). `finalContent` is the last COMPLETED
                // turn's text; discarding it reported an error that looked
                // like the run had done nothing, and the parent re-briefed a
                // fresh subagent from scratch over work that had happened.
                let partial = finalContent.trimmingCharacters(in: .whitespacesAndNewlines)
                let partialNote = partial.isEmpty
                    ? ""
                    : "\n\nPartial progress before the error:\n\(partial)"
                return SubagentRunResult(
                    agentName: agent.name,
                    status: "error",
                    finalResponse: "Subagent generation error: \(error.localizedDescription)"
                        + partialNote,
                    totalTurns: currentTurn,
                    totalToolCalls: totalToolCalls,
                    durationSeconds: duration
                )
            }

            finalContent = generatedText
            var parsedCalls = extractToolCalls(
                from: generatedText, projectURL: project?.rootDirectoryURL)

            let availableSpecs = availableTools(for: agent, project: project)
            let verdict = ForgeGuardrailsEngine.inspect(
                text: generatedText,
                parsedCalls: parsedCalls,
                availableTools: availableSpecs,
                requiresCall: false
            )

            switch verdict {
            case .accept:
                break
            case .rescued(let rescuedCalls, _):
                parsedCalls = rescuedCalls
            case .retry(let nudge):
                if currentTurn < maxTurns && !nudge.isEmpty {
                    history.append(ChatMessage(role: .assistant, content: generatedText))
                    history.append(ChatMessage(role: .user, content: nudge))
                    continue
                }
            }

            if parsedCalls.isEmpty {
                // Completed prose answer with no more tool calls
                break
            }

            // Append assistant response to isolated history
            history.append(ChatMessage(role: .assistant, content: generatedText))

            // Execute parsed tool calls
            for call in parsedCalls {
                if Task.isCancelled { break }
                totalToolCalls += 1
                await progress?(.toolStarted(name: call.name, summary: call.argumentsSummary))
                let observationMessage = await observation(
                    for: call, agent: agent, project: project, chatID: chatID, depth: depth,
                    session: session)
                await progress?(.toolFinished(
                    name: call.name, summary: call.argumentsSummary,
                    isError: observationMessage.content.hasPrefix("<tool_error>")))
                history.append(observationMessage)
            }

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
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        if Task.isCancelled {
            return SubagentRunResult(
                agentName: agent.name,
                status: "cancelled",
                finalResponse: finalContent.isEmpty
                    ? "Subagent run was cancelled." : finalContent,
                totalTurns: currentTurn,
                totalToolCalls: totalToolCalls,
                durationSeconds: totalDuration
            )
        }
        // **A RUN THAT RAN OUT OF TURNS DID NOT COMPLETE.** `finalContent` here
        // is the last turn's raw text, which on this exit is a tool call the
        // loop never got to execute -- reported as `completed` it reaches the
        // parent as an answer, with unexecuted XML as its content.
        let availableNames = Set(availableTools(for: agent, project: project).map { $0.function.name })
        let lastHasCalls = !extractToolCalls(from: finalContent, projectURL: project?.rootDirectoryURL).isEmpty
            || !ForgeGuardrailsEngine.rescueToolCalls(from: finalContent, availableToolNames: availableNames).isEmpty
        let exhausted = currentTurn >= maxTurns && lastHasCalls
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

    /// The tools available to this agent under the turn's project and MCP environment.
    public static func availableTools(for agent: AppAgentDefinition, project: AppProject?) -> [OpenAITool] {
        let baseTools = AppToolCatalog.tools(for: project?.agentType ?? .coder, projectURL: project?.rootDirectoryURL)
            .filter { agent.isToolAllowed($0.function.name) }
        guard let project else { return baseTools }
        let servers = AppToolCatalogMcp.visibleServers(
            global: GlobalMcpFileStore.load().servers, project: project)
        let mcpDefinitions = AppToolCatalogMcp.toolDefinitions(
            servers: servers, permissions: project.permissions)
            .filter { agent.isToolAllowed($0.function.name) }
        return baseTools + mcpDefinitions
    }

    /// **THE PARSER IS `ToolCallParser`'S NOW.** This file used to carry a
    /// byte-identical copy of it plus its own `parseJSONArguments`, which is
    /// the concrete reason state#68, state#74 and state#75 were three
    /// separate discoveries: a change to how the main loop reads a call had
    /// no way of reaching this one. The main loop's own guard is not shared,
    /// because a subagent is inside a project by construction.
    public static func extractToolCalls(
        from text: String, projectURL: URL? = nil
    ) -> [AppToolCall] {
        ToolCallParser.parse(from: text, projectURL: projectURL)
    }
}
