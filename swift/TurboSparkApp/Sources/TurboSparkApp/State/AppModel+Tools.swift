import Foundation
import TurboSpark

extension AppModel {
    /// Resolves effective permission level for a tool category.
    public func permission(for category: AppToolCategory) -> AppToolPermission {
        let permissions = selectedProject?.permissions ?? .standard
        switch category {
        case .fileRead: return permissions.fileRead
        case .fileWrite: return permissions.fileWrite
        case .terminal: return permissions.terminal
        case .web: return permissions.web
        case .mcp: return permissions.mcp
        case .automation: return permissions.automation
        }
    }

    /// Constructs comprehensive system prompt including agent instructions, project rules, and tool definitions.
    public func buildSystemPrompt(for project: AppProject?) -> String {
        var sections: [String] = []

        let agentType = project?.agentType ?? .coder
        sections.append(agentType.defaultSystemPrompt)

        if let project {
            if let root = project.rootDirectoryPath, !root.isEmpty {
                sections.append("## Workspace Environment\nRoot codebase directory: `\(root)`")
            }
            if !project.customInstructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                sections.append("## Project Specific Rules\n\(project.customInstructions)")
            }
        }

        let toolsPrompt = AppToolRegistry.systemPromptAddendum(for: agentType)
        sections.append(toolsPrompt)

        return sections.joined(separator: "\n\n")
    }

    /// Parses tool invocations from generated model text.
    public func extractToolCalls(from text: String) -> [AppToolCall] {
        var calls: [AppToolCall] = []

        // 1. XML Format: <tool_call> ... <name>X</name> ... <arguments>Y</arguments> ... </tool_call>
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

        // 2. Markdown Block Format: ```tool_call ... ```
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

    private func parseJSONArguments(_ string: String) -> [String: String] {
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

    /// User approval action for a pending tool call, with optional session-level persistence.
    public func approvePendingToolCall(id: UUID, alwaysAllowSession: Bool = false) {
        guard var call = pendingToolCall, call.id == id else { return }
        call.status = .running
        pendingToolCall = nil

        let sessionID = selectedChatID.uuidString
        let toolName = call.name
        let cmd = call.arguments["command"] ?? call.arguments["cmd"]

        Task {
            if alwaysAllowSession {
                await SessionApprovalStore.shared.allowTool(sessionID: sessionID, toolName: toolName)
                if let cmd {
                    await SessionApprovalStore.shared.allowCommandPrefix(sessionID: sessionID, prefix: cmd)
                }
            }

            let result = await AppToolRegistry.execute(call: call, in: self.selectedProject)
            call.status = result.isError ? .failed : .completed
            self.appendToolExecutionTurn(call: call, result: result)
            self.continueAgentLoop()
        }
    }

    /// User denial action for a pending tool call.
    public func denyPendingToolCall(id: UUID) {
        guard var call = pendingToolCall, call.id == id else { return }
        call.status = .denied
        pendingToolCall = nil

        let result = AppToolResult(
            callID: call.id,
            output: TOOL_REJECTED_MESSAGE,
            isError: true,
            durationSeconds: 0.0
        )
        self.appendToolExecutionTurn(call: call, result: result)
        self.continueAgentLoop()
    }

    /// Appends the tool execution and result turn to the conversation history.
    public func appendToolExecutionTurn(call: AppToolCall, result: AppToolResult) {
        guard let chatIndex = selectedChatIndex else { return }
        let turn = AppChatMessage(
            role: .assistant,
            content: "Invoking tool `\(call.name)` (\(call.argumentsSummary))",
            reasoning: "",
            stopReason: "tool_use",
            toolCalls: [call],
            toolResults: [result]
        )
        chats[chatIndex].messages.append(turn)
        chats[chatIndex].updatedAt = Date()
        persistChats()
    }
}
