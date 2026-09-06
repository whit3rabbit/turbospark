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
    ///
    /// Filtered to `AppToolRegistry.isImplemented`: several definition
    /// lists above (web, task/agent, project/artifact, MCP-resource,
    /// planning) declare tools -- WebFetch, WebSearch, Agent, REPL,
    /// NotebookEdit, Cron*, Task{Get,Output,Stop,Update}, the MCP resource
    /// tools, and more -- with no executor behind them in
    /// `AppToolRegistry.execute`. Advertising them to the model let it call
    /// one, receive a fabricated "Executed successfully", and trust a
    /// result that never happened (T5); dropping them here means the model
    /// never sees them as an option in the first place. Add a name to
    /// `AppToolRegistry.supportedToolNames` (and a real `case` in
    /// `execute`) before removing it from this filter.
    /// Full suite of all supported OpenAI tool definitions.
    public static var allTools: [OpenAITool] {
        var tools: [OpenAITool] = []
        tools.append(contentsOf: fileTools)
        tools.append(contentsOf: terminalTools)
        tools.append(contentsOf: taskAgentTools)
        tools.append(contentsOf: webTools)
        tools.append(contentsOf: projectArtifactTools)
        tools.append(contentsOf: mcpTools)
        tools.append(contentsOf: planningInteractiveTools)
        tools.append(contentsOf: automationTools)
        let custom = CustomToolManager.shared.resolveEffectiveTools(for: nil).map { $0.openAITool }
        tools.append(contentsOf: custom)
        return tools.filter { AppToolRegistry.isImplemented($0.function.name) }
    }

    /// Filters available tools appropriate for a specific agent profile, including custom workspace tools.
    public static func tools(for agentType: AppAgentType, projectURL: URL? = nil) -> [OpenAITool] {
        var list: [OpenAITool]
        switch agentType {
        case .coder:
            var l: [OpenAITool] = []
            l.append(contentsOf: fileTools)
            l.append(contentsOf: terminalTools)
            l.append(contentsOf: taskAgentTools)
            l.append(contentsOf: projectArtifactTools)
            l.append(contentsOf: webTools)
            l.append(contentsOf: planningInteractiveTools)
            list = l

        case .researcher:
            var l: [OpenAITool] = []
            l.append(contentsOf: fileTools)
            l.append(contentsOf: webTools)
            l.append(contentsOf: projectArtifactTools)
            l.append(contentsOf: mcpTools)
            l.append(contentsOf: planningInteractiveTools)
            list = l

        case .autonomous:
            list = allTools

        case .general, .custom:
            var l: [OpenAITool] = []
            l.append(contentsOf: fileTools)
            l.append(contentsOf: terminalTools)
            l.append(contentsOf: webTools)
            l.append(contentsOf: taskAgentTools)
            list = l
        }

        let custom = CustomToolManager.shared.resolveEffectiveTools(for: projectURL).map { $0.openAITool }
        list.append(contentsOf: custom)
        if let idx = list.firstIndex(where: { $0.function.name == "skill" }) {
            list[idx] = SkillToolDefinitions.skillTool(projectURL: projectURL)
        }
        return list.filter { AppToolRegistry.isImplemented($0.function.name, projectURL: projectURL) }
    }

    /// Resolves the permission category for any tool name.
    ///
    /// **THE PROJECT ROOT IS PART OF THE QUESTION** (state#71). This resolved
    /// custom tools with `projectURL: nil`, which is the USER scope alone --
    /// so a project's own `.turbospark/tools/deploy.json` declaring
    /// `terminal` was not found here and fell through to the `default` arm's
    /// `.automation`, gating a shell tool on `permissions.automation` instead
    /// of `permissions.terminal`. Callers on the generation path pass the
    /// turn's project.
    public static func category(for toolName: String, projectURL: URL? = nil) -> AppToolCategory {
        let name = toolName.lowercased()
        if let custom = CustomToolManager.shared.resolveEffectiveTools(for: projectURL).first(where: { $0.name.lowercased() == name }) {
            return custom.category
        }
        if name.contains("__") || name.hasPrefix("mcp_") || name.hasPrefix("mcp.") {
            return .mcp
        }
        switch name {
        case "filewrite", "write_file", "write", "fileedit", "edit_file", "edit", "apply_patch", "applypatch", "notebookedit", "notebook_edit", "todowrite", "todo_write", "proposeskills", "propose_skills":
            return .fileWrite
        case "bash", "run_command", "repl", "shell", "exec", "terminal", "bashoutput", "bash_output", "killshell", "kill_shell", "enterworktree", "enter_worktree", "exitworktree", "exit_worktree":
            return .terminal
        case "websearch", "web_search", "webfetch", "web_fetch", "fetch_url", "search_web", "read_url_content":
            return .web
        case "schedule", "cron", "manage_task", "monitoring", "notify", "notification", "sleep", "delay", "pushnotification", "push_notification", "config", "config_tool", "ctxinspect", "ctx_inspect", "askuserquestion", "ask_user_question", "ask_question", "question", "enterplanmode", "enter_plan_mode", "plan_mode", "plan", "exitplanmode", "exit_plan_mode", "reportfindings", "report_findings", "findings", "proposegoal", "propose_goal", "sendfeedback", "send_feedback", "agent", "subagent", "task", "taskcreate", "task_create", "task_add", "taskget", "task_get", "tasklist", "task_list", "taskupdate", "task_update", "taskstop", "task_stop", "task_cancel", "taskoutput", "task_output":
            return .automation
        case "call_mcp_tool", "callmcptool", "mcp_tool", "list_resources", "listmcpresources", "list_mcp_resources", "read_resource", "readmcpresource", "read_mcp_resource":
            return .mcp
        case "read_file", "view_file", "cat", "fileread", "read", "list_directory", "list_dir", "ls", "glob", "search_code", "grep", "search", "grep_search", "snip", "extract_snippet", "senduserfile", "send_user_file":
            return .fileRead
        default:
            return .automation
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
        lines.append("")
        lines.append("## Task & Progress Tracking")
        lines.append("For multi-step or non-trivial tasks (3+ steps), proactively use `TodoWrite` to organize your plan, track progress, and update status in real-time. Mark a task as `in_progress` BEFORE working on it and `completed` IMMEDIATELY upon finishing.")
        return lines.joined(separator: "\n")
    }
}
