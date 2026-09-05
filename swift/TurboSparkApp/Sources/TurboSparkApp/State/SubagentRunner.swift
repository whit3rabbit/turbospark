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
        maxTurnsOverride: Int? = nil,
        userSystemPrompt: String = ""
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

    // MARK: - Tool Call Parsing

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
