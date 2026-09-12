import Foundation

/// Summary metadata describing a tool call in a compact Claude-style format.
public struct ToolCallSummaryInfo: Equatable, Sendable {
    /// Action verb describing what the tool performed (e.g. "Edited", "Read", "Wrote", "Ran").
    public let action: String
    /// Primary target of the action (e.g. "messages.rs", "cargo check").
    public let target: String
    /// Number of lines added (for file edits/writes), displayed in green.
    public let additions: Int?
    /// Number of lines deleted (for file edits), displayed in red.
    public let deletions: Int?
    /// Optional line range (for file reads), e.g. "(563-722)".
    public let lineRange: String?

    public init(
        action: String,
        target: String,
        additions: Int? = nil,
        deletions: Int? = nil,
        lineRange: String? = nil
    ) {
        self.action = action
        self.target = target
        self.additions = additions
        self.deletions = deletions
        self.lineRange = lineRange
    }
}

/// Helper utility that extracts concise Claude-style diff statistics and summaries from tool calls.
public enum ToolCallDiffFormatter {
    /// Parses a tool call and extracts a compact summary with additions, deletions, and line ranges.
    public static func summarize(callName: String, arguments: [String: String]) -> ToolCallSummaryInfo {
        let lowerName = callName.lowercased()

        // 1. Task Management / TodoWrite (checked before write_file/create_file)
        if lowerName.contains("todo") {
            if let parsed = try? TodoWriteExecutor.parseTodos(from: arguments) {
                return ToolCallSummaryInfo(
                    action: "Tasks",
                    target: TodoChecklistSummary.text(parsed)
                )
            }
            return ToolCallSummaryInfo(
                action: "Updated",
                target: "task checklist"
            )
        }

        // 2. Subagent & Agent Execution
        if lowerName == "agent" || lowerName == "subagent"
            || (lowerName == "task" && (arguments["subagent_type"] != nil || arguments["subagentType"] != nil || arguments["prompt"] != nil || arguments["description"] != nil)) {
            let isBg = arguments["run_in_background"]?.lowercased() == "true"
                || arguments["runInBackground"]?.lowercased() == "true"
                || arguments["background"]?.lowercased() == "true"
            let agentType = arguments["subagent_type"] ?? arguments["subagentType"] ?? arguments["agent"] ?? arguments["agent_name"] ?? "agent"
            let desc = arguments["description"] ?? arguments["task"] ?? arguments["prompt"] ?? ""
            let target = desc.isEmpty ? agentType : "\(agentType): \(desc)"
            return ToolCallSummaryInfo(
                action: isBg ? "Background Agent" : "Agent",
                target: target
            )
        }

        if lowerName == "stop_agent" || lowerName == "stopagent" {
            let targetId = arguments["task_id"] ?? arguments["taskId"] ?? arguments["id"] ?? "agent"
            return ToolCallSummaryInfo(
                action: "Stop",
                target: targetId
            )
        }

        // 3. Structured Task System
        if lowerName.hasPrefix("task") {
            let subj = arguments["subject"] ?? arguments["taskId"] ?? arguments["task_id"] ?? "task"
            return ToolCallSummaryInfo(
                action: "Task",
                target: subj
            )
        }

        // 3. User Questions & Interactive Planning
        if lowerName.contains("question") {
            let header = arguments["header"] ?? "Question"
            return ToolCallSummaryInfo(
                action: "Asked",
                target: header
            )
        }

        if lowerName.contains("plan") {
            let act = lowerName.contains("enter") ? "Entered" : (lowerName.contains("exit") ? "Finalized" : "Plan")
            return ToolCallSummaryInfo(
                action: act,
                target: "plan mode"
            )
        }

        if lowerName.contains("findings") {
            return ToolCallSummaryInfo(
                action: "Reported",
                target: "code findings"
            )
        }

        // 4. Notebooks
        if lowerName.contains("notebook") {
            let targetPath = arguments["notebook_path"] ?? arguments["notebookPath"] ?? arguments["path"] ?? "notebook.ipynb"
            let fileName = (targetPath as NSString).lastPathComponent
            return ToolCallSummaryInfo(
                action: "Notebook",
                target: fileName.isEmpty ? "notebook.ipynb" : fileName
            )
        }

        // 5. Snippet extraction
        if lowerName == "snip" || lowerName == "extract_snippet" {
            let targetPath = arguments["path"] ?? arguments["file_path"] ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent
            let start = arguments["start_line"] ?? arguments["start"]
            let end = arguments["end_line"] ?? arguments["end"]
            let range = (start != nil && end != nil) ? "(\(start!)-\(end!))" : nil
            return ToolCallSummaryInfo(
                action: "Snippet",
                target: fileName.isEmpty ? "file" : fileName,
                lineRange: range
            )
        }

        // 6. Sleep / Notifications
        if lowerName == "sleep" || lowerName == "delay" {
            let sec = arguments["seconds"] ?? arguments["duration"] ?? "1"
            return ToolCallSummaryInfo(
                action: "Pause",
                target: "\(sec)s"
            )
        }

        if lowerName.contains("notification") || lowerName == "notify" {
            return ToolCallSummaryInfo(
                action: "Notified",
                target: "user"
            )
        }

        // 7. Patch Application
        if lowerName == "apply_patch" || lowerName == "applypatch" {
            let patch = arguments["patch_text"] ?? arguments["patchText"] ?? arguments["patch"] ?? ""
            var fileName = "patch"
            var additions = 0
            var deletions = 0
            for line in patch.components(separatedBy: "\n") {
                if line.starts(with: "+++ b/") {
                    fileName = (String(line.dropFirst(6)) as NSString).lastPathComponent
                } else if line.starts(with: "--- a/") && fileName == "patch" {
                    fileName = (String(line.dropFirst(6)) as NSString).lastPathComponent
                } else if line.starts(with: "+") && !line.starts(with: "+++") {
                    additions += 1
                } else if line.starts(with: "-") && !line.starts(with: "---") {
                    deletions += 1
                }
            }
            return ToolCallSummaryInfo(
                action: "Patched",
                target: fileName,
                additions: additions > 0 ? additions : nil,
                deletions: deletions > 0 ? deletions : nil
            )
        }

        // 8. File Edits: replace_file_content, edit_file, editor, etc.
        if lowerName.contains("replace") || lowerName.contains("edit") || lowerName == "editor" {
            let targetPath = arguments["TargetFile"]
                ?? arguments["AbsolutePath"]
                ?? arguments["path"]
                ?? arguments["file_path"]
                ?? arguments["filePath"]
                ?? arguments["file"]
                ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var additions: Int? = nil
            var deletions: Int? = nil

            if let targetContent = arguments["TargetContent"] ?? arguments["old_string"] ?? arguments["target"] {
                deletions = countLines(targetContent)
                // Zero is "nothing removed" (a pure insertion), which the
                // badge's absence says better than "-0".
                if deletions == 0 { deletions = nil }
            }
            if let replacementContent = arguments["ReplacementContent"] ?? arguments["new_string"] ?? arguments["replacement"] {
                additions = countLines(replacementContent)
                if additions == 0 { additions = nil }
            }

            // If multi_replace chunks are present in JSON
            if let chunksRaw = arguments["ReplacementChunks"], let data = chunksRaw.data(using: .utf8) {
                if let chunks = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
                    var totalAdd = 0
                    var totalDel = 0
                    for chunk in chunks {
                        if let target = chunk["TargetContent"] as? String {
                            totalDel += countLines(target)
                        }
                        if let repl = chunk["ReplacementContent"] as? String {
                            totalAdd += countLines(repl)
                        }
                    }
                    if totalAdd > 0 { additions = totalAdd }
                    if totalDel > 0 { deletions = totalDel }
                }
            }

            return ToolCallSummaryInfo(
                action: "Edited",
                target: fileName.isEmpty ? "file" : fileName,
                additions: additions,
                deletions: deletions
            )
        }

        // 9. File Writes: write_to_file, write_file, create_file
        if lowerName.contains("write") || lowerName.contains("create") {
            let targetPath = arguments["TargetFile"]
                ?? arguments["AbsolutePath"]
                ?? arguments["path"]
                ?? arguments["file_path"]
                ?? arguments["filePath"]
                ?? arguments["file"]
                ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var additions: Int? = nil
            if let content = arguments["CodeContent"] ?? arguments["content"] ?? arguments["text"] {
                additions = countLines(content)
            }

            return ToolCallSummaryInfo(
                action: "Wrote",
                target: fileName.isEmpty ? "file" : fileName,
                additions: additions
            )
        }

        // 10. Web Fetch & HTTP Request
        if lowerName.contains("fetch") || lowerName.contains("read_url") || lowerName.contains("http") {
            let method = arguments["method"]?.uppercased() ?? (lowerName.contains("http") ? "HTTP" : "Fetched")
            let rawUrl = arguments["url"] ?? arguments["uri"] ?? arguments["Url"] ?? arguments["URL"] ?? "url"
            let targetStr: String
            if let parsed = URL(string: rawUrl), let host = parsed.host {
                let path = parsed.path
                targetStr = path.isEmpty || path == "/" ? host : "\(host)\(path.prefix(24))"
            } else {
                targetStr = String(rawUrl.prefix(30))
            }
            return ToolCallSummaryInfo(
                action: method,
                target: targetStr
            )
        }

        // 11. Directory Listing
        if lowerName == "list_directory" || lowerName == "list_dir" || lowerName == "ls" || lowerName == "glob" {
            let path = arguments["path"] ?? arguments["pattern"] ?? arguments["DirectoryPath"] ?? arguments["dir"] ?? "."
            let target = path == "." ? "workspace" : (path as NSString).lastPathComponent
            return ToolCallSummaryInfo(
                action: "Listed",
                target: target.isEmpty ? path : target
            )
        }

        // 12. Skills & Feedback
        if lowerName == "skill" {
            let name = arguments["name"] ?? arguments["skill_name"] ?? "skill"
            return ToolCallSummaryInfo(
                action: "Skill",
                target: name
            )
        }

        if lowerName.contains("propose_skill") || lowerName.contains("proposeskill") {
            let name = arguments["name"] ?? arguments["skill_name"] ?? "skills"
            return ToolCallSummaryInfo(
                action: "Proposed",
                target: name
            )
        }

        if lowerName.contains("propose_goal") || lowerName.contains("proposegoal") {
            let title = arguments["title"] ?? arguments["goal"] ?? "goal"
            return ToolCallSummaryInfo(
                action: "Goal",
                target: title
            )
        }

        if lowerName.contains("feedback") {
            return ToolCallSummaryInfo(
                action: "Feedback",
                target: "session notes"
            )
        }

        // 13. File Reads: view_file, read_file
        if lowerName.contains("view") || lowerName.contains("read") {
            let targetPath = arguments["AbsolutePath"]
                ?? arguments["path"]
                ?? arguments["file_path"]
                ?? arguments["filePath"]
                ?? arguments["file"]
                ?? arguments["TargetFile"]
                ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var lineRange: String? = nil
            let start = arguments["StartLine"] ?? arguments["start_line"] ?? arguments["startLine"] ?? arguments["start"]
            let end = arguments["EndLine"] ?? arguments["end_line"] ?? arguments["endLine"] ?? arguments["end"]
            if let start, let end, !start.isEmpty, !end.isEmpty {
                lineRange = "(\(start)-\(end))"
            }

            return ToolCallSummaryInfo(
                action: "Read",
                target: fileName.isEmpty ? "file" : fileName,
                lineRange: lineRange
            )
        }

        // 14. Worktree
        if lowerName.contains("worktree") && (lowerName.contains("enter") || lowerName.contains("exit")) {
            let act = lowerName.contains("exit") ? "Exit Worktree" : "Worktree"
            let name = arguments["name"] ?? arguments["branch"] ?? arguments["path"] ?? "active"
            return ToolCallSummaryInfo(
                action: act,
                target: name
            )
        }

        // 15. Commands: run_command, bash, terminal
        if lowerName.contains("command") || lowerName.contains("bash") || lowerName.contains("exec") || lowerName.contains("terminal") {
            let cmd = arguments["CommandLine"] ?? arguments["command"] ?? arguments["cmd"] ?? arguments["name"] ?? ""
            let trimmed = cmd.trimmingCharacters(in: .whitespacesAndNewlines)
            let shortCmd = simplifyCommand(trimmed)

            return ToolCallSummaryInfo(
                action: "Ran",
                target: shortCmd.isEmpty ? "command" : shortCmd
            )
        }

        // 16. Grep / Search
        if lowerName.contains("grep") || lowerName.contains("search") {
            let query = arguments["Query"] ?? arguments["query"] ?? arguments["pattern"] ?? ""
            return ToolCallSummaryInfo(
                action: "Searched",
                target: query.isEmpty ? callName : "\"\(query)\""
            )
        }

        // 17. MCP Tools
        if lowerName.contains("mcp") {
            let server = arguments["server"] ?? arguments["server_name"] ?? arguments["ServerName"] ?? ""
            let tool = arguments["toolName"] ?? arguments["tool"] ?? arguments["name"] ?? arguments["ToolName"] ?? ""
            if !server.isEmpty && !tool.isEmpty {
                return ToolCallSummaryInfo(
                    action: "MCP: \(server)",
                    target: tool
                )
            } else if lowerName.hasPrefix("mcp__") {
                let parts = callName.components(separatedBy: "__")
                if parts.count >= 3 {
                    return ToolCallSummaryInfo(
                        action: "MCP: \(parts[1])",
                        target: parts[2...].joined(separator: "__")
                    )
                }
            } else if lowerName.contains("resource") {
                let uri = arguments["uri"] ?? arguments["Uri"] ?? "resource"
                return ToolCallSummaryInfo(
                    action: "MCP Resource",
                    target: (uri as NSString).lastPathComponent
                )
            }
        }

        // 18. Shared user files
        if lowerName == "senduserfile" || lowerName == "send_user_file" {
            let targetPath = arguments["path"] ?? arguments["file_path"] ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent
            return ToolCallSummaryInfo(
                action: "Shared",
                target: fileName.isEmpty ? "file" : fileName
            )
        }

        // 19. Generic fallback
        return ToolCallSummaryInfo(
            action: "Invoked",
            target: callName
        )
    }

    /// Counts the lines of the given text snippet. An empty body is ZERO
    /// lines (it used to read 1, so an empty write showed "1 lines" and a
    /// pure insertion showed "-1"), and a trailing newline does not start a
    /// phantom last line.
    public static func countLines(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var lines = text.components(separatedBy: "\n")
        if lines.last == "" { lines.removeLast() }
        return lines.count
    }

    /// Simplifies command strings for concise single-line presentation.
    private static func simplifyCommand(_ cmd: String) -> String {
        let firstLine = cmd.components(separatedBy: "\n").first ?? cmd
        if firstLine.count > 45 {
            return String(firstLine.prefix(42)) + "..."
        }
        return firstLine
    }
}
