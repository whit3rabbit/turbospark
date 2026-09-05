import Foundation
import TurboSpark

/// Which tools exist, what each one is for, and how a call is classified.
///
/// **FOUR LISTS DESCRIBE ONE VOCABULARY AND THEY ARE KEPT IN STEP BY HAND.**
/// `standardTools` is what the model is TOLD about, `supportedToolNames` is
/// what `execute` actually implements, `workspaceRootedToolNames` is the
/// subset that cannot run without a project root, and `category(for:)` is
/// what the permission engine gates on. A name missing from one of them is a
/// silent, specific failure every time: absent from the second it is
/// refused as unimplemented; absent from the third it runs at the
/// `/dev/null` placeholder (state#70); absent from the fourth it falls to
/// `.automation` and is gated on the wrong switch (state#71).
///
/// They are in one file so the four can be read against each other. Merging
/// them into a single table is the real fix and is a bigger change than a
/// move: `standardTools` carries prose, `category` has arms for names none
/// of the others list (MCP prefixes, `notebookedit`), and custom tools enter
/// three of the four at runtime.
extension AppToolRegistry {
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

    /// Resolves category for a tool name. `projectURL` is what makes a
    /// PROJECT-scoped custom tool resolvable (state#71).
    public static func category(for toolName: String, projectURL: URL? = nil) -> AppToolCategory {
        return AppToolCatalog.category(for: toolName, projectURL: projectURL)
    }

    /// Active session provider for running subagent tasks.
    public static var activeSessionProvider: (@Sendable @MainActor () -> TurboSparkSession?)?

    /// The app-wide default system prompt, for a subagent spawned by the
    /// `agent` TOOL rather than from `AppModel+Agents`.
    ///
    /// A provider for `activeSessionProvider`'s reason: this type is not the
    /// model and cannot reach it. Without it the two ways to start a subagent
    /// would disagree about whether the user's prompt applies, which is the
    /// divergence `SubagentRunner.buildSystemPrompt`'s own comment warns about.
    public static var userSystemPromptProvider: (@Sendable @MainActor () -> String)?

    /// Tool names `execute(call:in:)` actually has a real handler for,
    /// independent of which `OpenAITool` DEFINITIONS `AppToolCatalog`
    /// advertises to the model. A name outside this set (and not a dynamic
    /// `mcp__server__tool` call, which resolves through `executeMcpCall`
    /// rather than a static name) falls to `execute`'s default case, which
    /// reports `isError` rather than fabricating success (T5). This set is
    /// also what lets `AppToolCatalog` avoid advertising a tool with no
    /// backing executor in the first place -- keep it in sync with the
    /// `switch` in `execute(call:in:)` below.
    static let supportedToolNames: Set<String> = [
        "list_directory", "list_dir", "ls", "glob",
        "read_file", "view_file", "cat", "fileread", "read",
        "write_file", "save_file", "filewrite", "write",
        "edit_file", "fileedit", "edit",
        "apply_patch", "applypatch",
        "search_code", "grep", "search",
        "run_command", "bash", "shell", "exec", "terminal",
        "websearch", "web_search", "search_web",
        "webfetch", "web_fetch", "fetch_url", "read_url_content",
        "skill",
        "todowrite", "todo_write",
        "agent", "subagent", "task",
        "askuserquestion", "ask_user_question", "ask_question", "question",
        "enterplanmode", "enter_plan_mode", "plan_mode", "plan",
        "exitplanmode", "exit_plan_mode",
        "reportfindings", "report_findings", "findings",
        "proposeskills", "propose_skills",
        "proposegoal", "propose_goal",
        "sendfeedback", "send_feedback",
        "notebookedit", "notebook_edit",
        "snip", "extract_snippet",
        "senduserfile", "send_user_file",
        "taskcreate", "task_create", "task_add",
        "taskget", "task_get",
        "tasklist", "task_list",
        "taskupdate", "task_update",
        "taskstop", "task_stop", "task_cancel",
        "taskoutput", "task_output",
        "sleep", "delay",
        "pushnotification", "push_notification", "notify",
        "config", "config_tool",
        "ctxinspect", "ctx_inspect",
        "enterworktree", "enter_worktree",
        "exitworktree", "exit_worktree",
        "call_mcp_tool", "callmcptool", "mcp_tool",
        "listmcpresources", "list_mcp_resources", "list_resources",
        "readmcpresource", "read_mcp_resource", "read_resource"
    ]

    /// Tool names whose handler resolves a filesystem path or spawns a
    /// process, and therefore cannot run without a project root.
    ///
    /// The complement of this set inside `supportedToolNames` is the group
    /// that works in a projectless chat (`skill`, `todowrite`, `agent`).
    /// Every `mcp__server__tool` call is treated as
    /// rooted too: `executeMcpCall` passes the root as the server's working
    /// directory, so there is no correct value to pass without one.
    static let workspaceRootedToolNames: Set<String> = [
        "list_directory", "list_dir", "ls", "glob",
        "read_file", "view_file", "cat", "fileread", "read",
        "write_file", "save_file", "filewrite", "write",
        "edit_file", "fileedit", "edit",
        "apply_patch", "applypatch",
        "search_code", "grep", "search",
        "run_command", "bash", "shell", "exec", "terminal",
        "notebookedit", "notebook_edit",
        "snip", "extract_snippet",
        "senduserfile", "send_user_file",
        "proposeskills", "propose_skills",
        "enterworktree", "enter_worktree",
        "exitworktree", "exit_worktree",
        "call_mcp_tool", "callmcptool", "mcp_tool",
        "listmcpresources", "list_mcp_resources", "list_resources",
        "readmcpresource", "read_mcp_resource", "read_resource"
    ]

    /// Whether `execute(call:in:)` has a real handler for `toolName`.
    public static func isImplemented(_ toolName: String, projectURL: URL? = nil) -> Bool {
        let lower = toolName.lowercased()
        if lower.contains("__") && lower.hasPrefix("mcp__") { return true }
        if supportedToolNames.contains(lower) { return true }
        let custom = CustomToolManager.shared.resolveEffectiveTools(for: projectURL)
        return custom.contains(where: { $0.name.lowercased() == lower })
    }

    /// Generates system prompt instructions for tool use.
    public static func systemPromptAddendum(for agentType: AppAgentType, tools: [AppToolDefinition] = standardTools) -> String {
        return AppToolCatalog.systemPromptAddendum(for: agentType)
    }
}
