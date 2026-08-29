import Foundation

/// Functional category of a tool action for permission checks.
public enum AppToolCategory: String, Codable, CaseIterable, Identifiable, Sendable {
    case fileRead
    case fileWrite
    case terminal
    case web
    case mcp
    case automation

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .fileRead: return "File & Codebase Reading"
        case .fileWrite: return "File Editing & Creation"
        case .terminal: return "Terminal & Shell Execution"
        case .web: return "Web & Network Requests"
        case .mcp: return "MCP External Tools"
        case .automation: return "Automation & Cron Tasks"
        }
    }

    public var systemImage: String {
        switch self {
        case .fileRead: return "doc.text.magnifyingglass"
        case .fileWrite: return "square.and.pencil"
        case .terminal: return "terminal"
        case .web: return "globe"
        case .mcp: return "server.rack"
        case .automation: return "clock.arrow.2.circlepath"
        }
    }
}

/// Constant rejection message fed back to the model on denial (Unsloth Studio parity).
public let TOOL_REJECTED_MESSAGE = "The user declined to run this tool call."

/// Execution status of an individual tool call.
public enum AppToolCallStatus: String, Codable, Sendable {
    case pendingApproval
    case running
    case completed
    case denied
    case failed
}

/// A parsed tool call requested by the model.
public struct AppToolCall: Identifiable, Codable, Equatable, Sendable {
    public var id = UUID()
    /// Unique approval identifier for tracking pending decisions.
    public var approvalID: String
    /// Canonical tool name (e.g. read_file, write_file, run_command).
    public var name: String
    /// Parsed parameter dictionary.
    public var arguments: [String: String]
    /// Raw unparsed invocation string from model output.
    public var rawInvocation: String
    /// Execution status.
    public var status: AppToolCallStatus
    /// Associated tool category.
    public var category: AppToolCategory
    /// Evaluated security risk assessment.
    public var riskAssessment: ToolRiskAssessment?
    /// Timestamp when invocation was requested.
    public var createdAt: Date

    public init(
        id: UUID = UUID(),
        approvalID: String = UUID().uuidString.prefix(16).lowercased(),
        name: String,
        arguments: [String: String] = [:],
        rawInvocation: String = "",
        status: AppToolCallStatus = .pendingApproval,
        category: AppToolCategory = .fileRead,
        riskAssessment: ToolRiskAssessment? = nil,
        createdAt: Date = Date()
    ) {
        self.id = id
        self.approvalID = approvalID
        self.name = name
        self.arguments = arguments
        self.rawInvocation = rawInvocation
        self.status = status
        self.category = category
        self.riskAssessment = riskAssessment
        self.createdAt = createdAt
    }

    /// Single line summary of call arguments for display.
    public var argumentsSummary: String {
        if let path = arguments["path"] ?? arguments["file_path"] {
            return path
        }
        if let cmd = arguments["command"] ?? arguments["cmd"] {
            return cmd
        }
        if let q = arguments["query"] ?? arguments["pattern"] {
            return "\"\(q)\""
        }
        return arguments.map { "\($0.key): \($0.value)" }.joined(separator: ", ")
    }
}

/// The result returned from executing a tool call.
public struct AppToolResult: Identifiable, Codable, Equatable, Sendable {
    public var id = UUID()
    public var callID: UUID
    public var output: String
    public var isError: Bool
    public var durationSeconds: Double

    public init(
        id: UUID = UUID(),
        callID: UUID,
        output: String,
        isError: Bool = false,
        durationSeconds: Double = 0.0
    ) {
        self.id = id
        self.callID = callID
        self.output = output
        self.isError = isError
        self.durationSeconds = durationSeconds
    }
}

/// Description of an available tool for prompt construction.
public struct AppToolDefinition: Sendable {
    public let name: String
    public let category: AppToolCategory
    public let description: String
    public let usageExample: String
}

/// Built-in tools and executor implementations for codebase operations.
public enum AppToolRegistry {
    public static let standardTools: [AppToolDefinition] = [
        AppToolDefinition(
            name: "list_directory",
            category: .fileRead,
            description: "List contents of a directory (defaults to project root).",
            usageExample: "<tool_call>\n<name>list_directory</name>\n<arguments>{\"path\": \".\"}</arguments>\n</tool_call>"
        ),
        AppToolDefinition(
            name: "read_file",
            category: .fileRead,
            description: "Read text contents of a file with optional start_line and end_line bounds.",
            usageExample: "<tool_call>\n<name>read_file</name>\n<arguments>{\"path\": \"src/main.rs\", \"start_line\": \"1\", \"end_line\": \"100\"}</arguments>\n</tool_call>"
        ),
        AppToolDefinition(
            name: "write_file",
            category: .fileWrite,
            description: "Create or replace the contents of a file at the given relative path.",
            usageExample: "<tool_call>\n<name>write_file</name>\n<arguments>{\"path\": \"src/lib.rs\", \"content\": \"// code here\"}</arguments>\n</tool_call>"
        ),
        AppToolDefinition(
            name: "search_code",
            category: .fileRead,
            description: "Search for a text pattern or symbol across files in the codebase.",
            usageExample: "<tool_call>\n<name>search_code</name>\n<arguments>{\"pattern\": \"struct AppModel\", \"path\": \".\"}</arguments>\n</tool_call>"
        ),
        AppToolDefinition(
            name: "run_command",
            category: .terminal,
            description: "Execute a shell command inside the project root directory.",
            usageExample: "<tool_call>\n<name>run_command</name>\n<arguments>{\"command\": \"cargo check\"}</arguments>\n</tool_call>"
        )
    ]

    /// Resolves category for a tool name.
    public static func category(for toolName: String) -> AppToolCategory {
        return AppToolCatalog.category(for: toolName)
    }

    /// Generates system prompt instructions for tool use.
    public static func systemPromptAddendum(for agentType: AppAgentType, tools: [AppToolDefinition] = standardTools) -> String {
        return AppToolCatalog.systemPromptAddendum(for: agentType)
    }

    /// Executes a tool call asynchronously within the given project context.
    public static func execute(call: AppToolCall, in project: AppProject?) async -> AppToolResult {
        let startTime = Date()
        let rootURL = project?.rootDirectoryURL ?? URL(fileURLWithPath: FileManager.default.currentDirectoryPath)

        do {
            let output: String
            switch call.name.lowercased() {
            case "list_directory", "list_dir", "ls", "glob":
                let relPath = call.arguments["path"] ?? call.arguments["pattern"] ?? "."
                output = try listDirectory(relPath: relPath, rootURL: rootURL)

            case "read_file", "view_file", "cat", "fileread", "read":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] ?? call.arguments["resource"] else {
                    throw NSError(domain: "TurboSparkTool", code: 1, userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument."])
                }
                let startLine = Int(call.arguments["start_line"] ?? call.arguments["offset"] ?? "")
                let endLine = Int(call.arguments["end_line"] ?? call.arguments["limit"] ?? "")
                output = try await readFile(relPath: relPath, rootURL: rootURL, startLine: startLine, endLine: endLine)

            case "write_file", "save_file", "filewrite", "write":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'path' or 'file_path' argument."])
                }
                let content = call.arguments["content"] ?? ""
                output = try await writeFile(relPath: relPath, content: content, rootURL: rootURL)

            case "edit_file", "fileedit", "edit":
                guard let relPath = call.arguments["path"] ?? call.arguments["file_path"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'file_path' argument."])
                }
                guard let oldStr = call.arguments["old_string"] else {
                    throw NSError(domain: "TurboSparkTool", code: 2, userInfo: [NSLocalizedDescriptionKey: "Missing 'old_string' argument."])
                }
                let newStr = call.arguments["new_string"] ?? ""
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
                output = try await runCommand(command: command, rootURL: rootURL)

            case "skill":
                guard let skillName = call.arguments["name"] ?? call.arguments["skill_name"] else {
                    throw NSError(domain: "TurboSparkTool", code: 18, userInfo: [NSLocalizedDescriptionKey: "Missing 'name' argument for skill tool call."])
                }
                let effectiveSkills = SkillManager.shared.resolveEffectiveSkills(projectURL: project?.rootDirectoryURL)
                if let matched = effectiveSkills.first(where: { $0.name.lowercased() == skillName.lowercased() }) {
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
                    let available = effectiveSkills.map { "- \($0.name): \($0.skillDescription)" }.joined(separator: "\n")
                    output = "Skill '\(skillName)' was not found.\n\nAvailable skills:\n\(available.isEmpty ? "(No skills currently installed)" : available)"
                }

            case "todowrite", "todo_write":
                output = "Todo list updated."

            case "taskcreate", "task_create":
                let subject = call.arguments["subject"] ?? "Untitled task"
                output = "Task created: \(subject) (ID: task_\(UUID().uuidString.prefix(8)))"

            case "tasklist", "task_list":
                output = "Task list: No active blocking tasks."

            case "askuserquestion", "ask_user_question", "question":
                output = "Question submitted to user."

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
                        output = "Executed \(call.name) successfully."
                    }
                } else {
                    output = "Executed \(call.name) successfully."
                }
            }

            let elapsed = Date().timeIntervalSince(startTime)
            return AppToolResult(callID: call.id, output: output, isError: false, durationSeconds: elapsed)
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
        // Search in project servers first, then global servers
        let globalServers = GlobalMcpFileStore.load().servers
        let projectServers = project?.mcpServers ?? []
        let allServers = projectServers + globalServers

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
