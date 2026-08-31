import Foundation

/// Bridges Claude Code's own tool names (`Bash`, `Write`, `Edit`, ...) to this
/// app's tool names (`run_command`, `write_file`, `edit_file`, ...) so a
/// matcher written for either vocabulary fires on the same tool call.
///
/// Built from `AppToolRegistry.supportedToolNames` (`State/AppTool.swift`)
/// plus every Claude Code built-in tool name that has an equivalent here.
enum AppHookToolNameAliases {
    /// Claude Code name -> this app's canonical (lowercased) tool name(s).
    private static let claudeCodeToApp: [String: [String]] = [
        "bash": ["run_command", "bash", "shell", "exec", "terminal"],
        "write": ["write_file", "save_file", "filewrite", "write"],
        "edit": ["edit_file", "fileedit", "edit"],
        "read": ["read_file", "view_file", "cat", "fileread", "read"],
        "glob": ["list_directory", "list_dir", "ls", "glob"],
        "grep": ["search_code", "grep", "search"],
        "notebookedit": ["edit_file", "fileedit", "edit"],
        "task": ["skill"],
        "todowrite": ["todowrite", "todo_write"],
        "askuserquestion": ["askuserquestion", "ask_user_question", "question"],
        "webfetch": ["webfetch", "web_fetch", "fetch_url", "read_url_content"],
        "websearch": ["call_mcp_tool", "callmcptool", "mcp_tool"]
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
