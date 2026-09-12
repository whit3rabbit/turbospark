import Foundation

/// Bridges Claude Code's own tool names (`Bash`, `Write`, `Edit`, ...) to this
/// app's tool names (`run_command`, `write_file`, `edit_file`, ...) so a
/// matcher written for either vocabulary fires on the same tool call.
///
/// Built from `AppToolRegistry.supportedToolNames` (`Tools/Registry/AppToolRegistry+Vocabulary.swift`)
/// plus every Claude Code built-in tool name that has an equivalent here.
enum AppHookToolNameAliases {
    /// Claude Code name -> this app's canonical (lowercased) tool name(s).
    private static let claudeCodeToApp: [String: [String]] = [
        "bash": ["run_command", "bash", "shell", "exec", "terminal"],
        "bashoutput": ["bashoutput", "bash_output"],
        "killshell": ["killshell", "kill_shell"],
        "write": ["write_file", "save_file", "filewrite", "write"],
        "edit": ["edit_file", "fileedit", "edit", "editor"],
        "read": ["read_file", "view_file", "cat", "fileread", "read"],
        "glob": ["list_directory", "list_dir", "ls", "glob"],
        "grep": ["search_code", "grep", "search"],
        "notebookedit": ["notebookedit", "notebook_edit"],
        "task": ["skill"],
        "todowrite": ["todowrite", "todo_write"],
        "webfetch": ["webfetch", "web_fetch", "fetch_url", "read_url_content", "http_request", "httprequest"],
        "websearch": ["websearch", "web_search", "search_web"],
        "askuserquestion": ["askuserquestion", "ask_user_question", "ask_question", "question"],
        "enterplanmode": ["enterplanmode", "enter_plan_mode", "plan_mode", "plan"],
        "exitplanmode": ["exitplanmode", "exit_plan_mode"],
        "reportfindings": ["reportfindings", "report_findings", "findings"],
        "proposeskills": ["proposeskills", "propose_skills"],
        "proposegoal": ["proposegoal", "propose_goal"],
        "sendfeedback": ["sendfeedback", "send_feedback"],
        "snip": ["snip", "extract_snippet"],
        "senduserfile": ["senduserfile", "send_user_file"],
        "taskcreate": ["taskcreate", "task_create", "task_add"],
        "taskget": ["taskget", "task_get"],
        "tasklist": ["tasklist", "task_list"],
        "taskupdate": ["taskupdate", "task_update"],
        "taskstop": ["taskstop", "task_stop", "task_cancel"],
        "taskoutput": ["taskoutput", "task_output"],
        "sleep": ["sleep", "delay"],
        "pushnotification": ["pushnotification", "push_notification", "notify"],
        "config": ["config", "config_tool"],
        "ctxinspect": ["ctxinspect", "ctx_inspect"],
        "enterworktree": ["enterworktree", "enter_worktree"],
        "exitworktree": ["exitworktree", "exit_worktree"],
        "listmcpresources": ["listmcpresources", "list_mcp_resources", "list_resources"],
        "readmcpresource": ["readmcpresource", "read_mcp_resource", "read_resource"]
    ]

    /// Every alias (in either direction, lowercased) that should be treated
    /// as equivalent to `toolName` when evaluating a matcher.
    static func equivalentNames(for toolName: String) -> Set<String> {
        let lower = toolName.lowercased()
        var names: Set<String> = [lower]
        if let mapped = claudeCodeToApp[lower] {
            names.formUnion(mapped)
        }
        for (claudeCodeName, appNames) in claudeCodeToApp where appNames.contains(lower) {
            names.insert(claudeCodeName)
            names.formUnion(appNames)
        }
        return names
    }
}
