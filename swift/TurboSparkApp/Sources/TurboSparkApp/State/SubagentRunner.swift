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
        let allTools = AppToolCatalog.allTools
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
    public static func run(
        agent: AppAgentDefinition,
        taskPrompt: String,
        session: TurboSparkSession?,
        project: AppProject?,
        maxTurnsOverride: Int? = nil
    ) async -> SubagentRunResult {
        let startTime = Date()
        let maxTurns = maxTurnsOverride ?? agent.maxTurns

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

        while currentTurn < maxTurns {
            currentTurn += 1

            var options = GenerateOptions()
            options.temperature = 0.2
            options.maxNewTokens = 2048

            var generatedText = ""

            do {
                let fitted = try await session.fitWindow(history, reasoning: .off)
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
            let parsedCalls = extractToolCalls(from: generatedText)

            if parsedCalls.isEmpty {
                // Completed prose answer with no more tool calls
                break
            }

            // Append assistant response to isolated history
            history.append(ChatMessage(role: .assistant, content: generatedText))

            // Execute parsed tool calls
            for call in parsedCalls {
                totalToolCalls += 1
                if !agent.isToolAllowed(call.name) {
                    let deniedMsg = "<tool_error>\nTool '\(call.name)' is disallowed for agent profile '\(agent.name)'.\n</tool_error>"
                    history.append(ChatMessage(role: .system, content: deniedMsg))
                    continue
                }

                let toolResult = await AppToolRegistry.execute(call: call, in: project)
                let tag = toolResult.isError ? "tool_error" : "tool_response"
                let resultMsg = "<\(tag)>\n\(toolResult.output)\n</\(tag)>"
                history.append(ChatMessage(role: .system, content: resultMsg))
            }
        }

        let totalDuration = Date().timeIntervalSince(startTime)
        return SubagentRunResult(
            agentName: agent.name,
            status: "completed",
            finalResponse: finalContent,
            totalTurns: currentTurn,
            totalToolCalls: totalToolCalls,
            durationSeconds: totalDuration
        )
    }

    // MARK: - Tool Call Parsing Helper

    public static func extractToolCalls(from text: String) -> [AppToolCall] {
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
                    let category = AppToolRegistry.category(for: toolName)
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
                        let category = AppToolRegistry.category(for: toolName)
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
