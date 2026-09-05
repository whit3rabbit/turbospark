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
                let completed = parsed.filter { $0.isCompleted }.count
                let inProgress = parsed.filter { $0.isInProgress }.count
                let total = parsed.count
                let statusSummary = inProgress > 0 ? "\(completed)/\(total) done (1 in progress)" : "\(completed)/\(total) done"
                return ToolCallSummaryInfo(
                    action: "Tasks",
                    target: statusSummary
                )
            }
            return ToolCallSummaryInfo(
                action: "Updated",
                target: "task checklist"
            )
        }

        // 2. Structured Task System
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

        // 7. File Edits: replace_file_content, edit_file, etc.
        if lowerName.contains("replace") || lowerName.contains("edit") {
            let targetPath = arguments["TargetFile"] ?? arguments["path"] ?? arguments["file"] ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var additions: Int? = nil
            var deletions: Int? = nil

            if let targetContent = arguments["TargetContent"] {
                deletions = countLines(targetContent)
            }
            if let replacementContent = arguments["ReplacementContent"] {
                additions = countLines(replacementContent)
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

        // 8. File Writes: write_to_file, write_file, create_file
        if lowerName.contains("write") || lowerName.contains("create") {
            let targetPath = arguments["TargetFile"] ?? arguments["path"] ?? arguments["file"] ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var additions: Int? = nil
            if let content = arguments["CodeContent"] ?? arguments["content"] {
                additions = countLines(content)
            }

            return ToolCallSummaryInfo(
                action: "Wrote",
                target: fileName.isEmpty ? "file" : fileName,
                additions: additions
            )
        }

        // 9. File Reads: view_file, read_file
        if lowerName.contains("view") || lowerName.contains("read") {
            let targetPath = arguments["AbsolutePath"] ?? arguments["path"] ?? arguments["file"] ?? arguments["TargetFile"] ?? "file"
            let fileName = (targetPath as NSString).lastPathComponent

            var lineRange: String? = nil
            if let start = arguments["StartLine"], let end = arguments["EndLine"], !start.isEmpty, !end.isEmpty {
                lineRange = "(\(start)-\(end))"
            }

            return ToolCallSummaryInfo(
                action: "Read",
                target: fileName.isEmpty ? "file" : fileName,
                lineRange: lineRange
            )
        }

        // 10. Commands: run_command, bash, terminal
        if lowerName.contains("command") || lowerName.contains("bash") || lowerName.contains("exec") || lowerName.contains("worktree") {
            let cmd = arguments["CommandLine"] ?? arguments["command"] ?? arguments["cmd"] ?? arguments["name"] ?? ""
            let trimmed = cmd.trimmingCharacters(in: .whitespacesAndNewlines)
            let shortCmd = simplifyCommand(trimmed)

            return ToolCallSummaryInfo(
                action: "Ran",
                target: shortCmd.isEmpty ? "command" : shortCmd
            )
        }

        // 11. Grep / Search
        if lowerName.contains("grep") || lowerName.contains("search") {
            let query = arguments["Query"] ?? arguments["query"] ?? arguments["pattern"] ?? ""
            return ToolCallSummaryInfo(
                action: "Searched",
                target: query.isEmpty ? callName : "\"\(query)\""
            )
        }

        // 12. Generic fallback
        return ToolCallSummaryInfo(
            action: "Invoked",
            target: callName
        )
    }

    /// Counts non-empty or total lines in the given text snippet.
    public static func countLines(_ text: String) -> Int {
        let lines = text.components(separatedBy: "\n")
        return max(1, lines.count)
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
