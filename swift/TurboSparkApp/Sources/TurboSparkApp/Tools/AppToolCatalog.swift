import Foundation

/// Central registry uniting all OpenAI-compatible tool definitions and categories for TurboSpark.
public enum AppToolCatalog {
    /// File and codebase navigation tools.
    public static let fileTools: [OpenAITool] = FileReadWriteToolDefinitions.all + FileSearchToolDefinitions.all + ApplyPatchToolDefinitions.all

    /// Terminal and execution tools.
    public static let terminalTools: [OpenAITool] = TerminalToolDefinitions.all

    /// Tasks, subagents, and todo management tools.
    public static let taskAgentTools: [OpenAITool] = AgentToolDefinitions.all + TaskItemToolDefinitions.all

    /// Web search and web fetching tools.
    public static let webTools: [OpenAITool] = WebToolDefinitions.all

    /// Project documentation, artifacts, and worktree tools.
    public static let projectArtifactTools: [OpenAITool] = ProjectDocDefinitions.all + ArtifactWorktreeDefinitions.all

    /// MCP server integration tools.
    public static let mcpTools: [OpenAITool] = McpToolDefinitions.all

    /// Interactive questions, plan mode, findings, skills, and feedback tools.
    public static let planningInteractiveTools: [OpenAITool] = PlanningInteractiveToolDefinitions.all + SkillToolDefinitions.all

    /// Automation, cron, monitoring, and notifications tools.
    public static let automationTools: [OpenAITool] = WorkflowCronDefinitions.all + MonitoringNotificationDefinitions.all

    /// Full suite of all supported OpenAI tool definitions.
    public static let allTools: [OpenAITool] = {
        var tools: [OpenAITool] = []
        tools.append(contentsOf: fileTools)
        tools.append(contentsOf: terminalTools)
        tools.append(contentsOf: taskAgentTools)
        tools.append(contentsOf: webTools)
        tools.append(contentsOf: projectArtifactTools)
        tools.append(contentsOf: mcpTools)
        tools.append(contentsOf: planningInteractiveTools)
        tools.append(contentsOf: automationTools)
        return tools
    }()

    /// Filters available tools appropriate for a specific agent profile.
    public static func tools(for agentType: AppAgentType) -> [OpenAITool] {
        switch agentType {
        case .coder:
            var list: [OpenAITool] = []
            list.append(contentsOf: fileTools)
            list.append(contentsOf: terminalTools)
            list.append(contentsOf: taskAgentTools)
            list.append(contentsOf: projectArtifactTools)
            list.append(contentsOf: webTools)
            list.append(contentsOf: planningInteractiveTools)
            return list

        case .researcher:
            var list: [OpenAITool] = []
            list.append(contentsOf: fileTools)
            list.append(contentsOf: webTools)
            list.append(contentsOf: projectArtifactTools)
            list.append(contentsOf: mcpTools)
            list.append(contentsOf: planningInteractiveTools)
            return list

        case .autonomous:
            return allTools

        case .general, .custom:
            var list: [OpenAITool] = []
            list.append(contentsOf: fileTools)
            list.append(contentsOf: terminalTools)
            list.append(contentsOf: webTools)
            list.append(contentsOf: taskAgentTools)
            return list
        }
    }

    /// Resolves the permission category for any tool name.
    public static func category(for toolName: String) -> AppToolCategory {
        let name = toolName.lowercased()
        if name.contains("__") || name.hasPrefix("mcp_") || name.hasPrefix("mcp.") {
            return .mcp
        }
        switch name {
        case "filewrite", "write_file", "write", "fileedit", "edit_file", "edit", "apply_patch", "applypatch", "notebookedit", "todowrite", "todo_write":
            return .fileWrite
        case "bash", "run_command", "repl", "shell", "exec", "terminal":
            return .terminal
        case "websearch", "web_search", "webfetch", "web_fetch", "fetch_url", "search_web", "read_url_content":
            return .web
        case "schedule", "cron", "manage_task", "monitoring", "notify", "notification":
            return .automation
        case "call_mcp_tool", "list_resources", "read_resource":
            return .mcp
        case "read_file", "view_file", "cat", "fileread", "read", "list_directory", "list_dir", "ls", "glob", "search_code", "grep", "search", "grep_search":
            return .fileRead
        default:
            return .fileRead
        }
    }

    /// Generates OpenAI-compatible tools JSON array payload.
    public static func openAIFormattedToolsJSON(for agentType: AppAgentType = .coder) -> String {
        let active = tools(for: agentType)
        return OpenAIToolSerializer.encodeJSONString(active)
    }

    /// Generates Markdown / XML system prompt guidance for tool use.
    public static func systemPromptAddendum(for agentType: AppAgentType) -> String {
        let active = tools(for: agentType)
        var lines: [String] = []
        lines.append("## Available Tools")
        lines.append("You have access to the following developer tools formatted in OpenAI function calling style:")
        for tool in active {
            lines.append("- `\(tool.function.name)`: \(tool.function.description)")
        }
        lines.append("")
        lines.append("To invoke a tool, output a tool call block:")
        lines.append("<tool_call>")
        lines.append("<name>tool_name</name>")
        lines.append("<arguments>{\"key\": \"value\"}</arguments>")
        lines.append("</tool_call>")
        lines.append("")
        lines.append("Or using JSON format:")
        lines.append("```tool_call")
        lines.append("{\"name\": \"tool_name\", \"arguments\": {\"key\": \"value\"}}")
        lines.append("```")
        return lines.joined(separator: "\n")
    }
}
