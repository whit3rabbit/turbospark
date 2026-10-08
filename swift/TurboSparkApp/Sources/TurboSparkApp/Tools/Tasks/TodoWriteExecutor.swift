import Foundation

/// Executor for the `todowrite` / `todo_write` / `TodoWrite` tool call.
/// Robustly parses and normalizes checklist items across diverse LLM function call formats.
public enum TodoWriteExecutor {
    /// Global notification callback invoked whenever a tool call updates todos for a session/chat.
    public static var onTodosUpdated: (@Sendable (_ chatID: UUID?, _ todos: [TodoItem]) -> Void)?

    /// Parses raw arguments into a list of normalized `TodoItem`s.
    public static func parseTodos(from arguments: [String: String]) throws -> [TodoItem] {
        if arguments.isEmpty {
            return []
        }

        // Strategy 1: arguments["todos"] containing JSON array or JSON string
        if let rawTodos = arguments["todos"]?.trimmingCharacters(in: .whitespacesAndNewlines) {
            if rawTodos.isEmpty || rawTodos == "[]" {
                return []
            }
            if let items = tryParseJSONTodos(rawTodos) {
                return items
            }
        }

        // Strategy 2: arguments["input"] or arguments["arguments"]
        if let nested = arguments["input"] ?? arguments["arguments"] {
            if let items = tryParseJSONTodos(nested) {
                return items
            }
        }

        // Strategy 3: Check entire arguments dictionary if it was keyed with indexed entries or single item
        if let content = arguments["content"] ?? arguments["task"] ?? arguments["description"] {
            let status = arguments["status"] ?? "pending"
            let active = arguments["activeForm"] ?? arguments["active_form"] ?? content
            let id = arguments["id"] ?? UUID().uuidString
            return [TodoItem(id: id, content: content, status: status, activeForm: active)]
        }

        // Strategy 4: If arguments has JSON keys directly
        for (_, value) in arguments {
            if let items = tryParseJSONTodos(value) {
                return items
            }
        }

        return []
    }

    /// Attempts to parse JSON string representing either `[TodoItem]` or `{"todos": [TodoItem]}`.
    private static func tryParseJSONTodos(_ jsonString: String) -> [TodoItem]? {
        let trimmed = jsonString.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let data = trimmed.data(using: .utf8) else { return nil }

        // Attempt 1: Direct array of dictionaries: `[{"content": "...", "status": "..."}]`
        if let array = try? JSONSerialization.jsonObject(with: data) as? [[String: Any]] {
            return extractItems(from: array)
        }

        // Attempt 2: Object containing "todos" array: `{"todos": [...]}`
        if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let array = dict["todos"] as? [[String: Any]] {
            return extractItems(from: array)
        }

        // Attempt 3: Object with numbered keys or simple list of strings
        if let stringArray = try? JSONSerialization.jsonObject(with: data) as? [String] {
            return stringArray.map { TodoItem(content: $0, status: "pending", activeForm: $0) }
        }

        return nil
    }

    private static func extractItems(from array: [[String: Any]]) -> [TodoItem] {
        var items: [TodoItem] = []
        var seenIDs: Set<String> = []
        for obj in array {
            let content = (obj["content"] as? String)
                ?? (obj["task"] as? String)
                ?? (obj["description"] as? String)
                ?? (obj["subject"] as? String)
                ?? ""
            let trimmedContent = content.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmedContent.isEmpty else { continue }

            let status = (obj["status"] as? String) ?? "pending"
            let active = (obj["activeForm"] as? String)
                ?? (obj["active_form"] as? String)
                ?? (obj["active"] as? String)
                ?? trimmedContent
            // Ids drive row identity in the checklist view: derive a positional
            // one when the model omits it (a random UUID re-identified every row
            // on each update) and suffix collisions so ForEach never sees two
            // rows with the same identity.
            var id = (obj["id"] as? String) ?? "todo-\(items.count + 1)"
            if seenIDs.contains(id) {
                var n = 2
                while seenIDs.contains("\(id)-\(n)") { n += 1 }
                id = "\(id)-\(n)"
            }
            seenIDs.insert(id)

            items.append(TodoItem(
                id: id,
                content: trimmedContent,
                status: status,
                activeForm: active.isEmpty ? trimmedContent : active
            ))
        }
        return items
    }

    /// True when the caller deliberately asked for an empty list.
    private static func isExplicitClear(_ arguments: [String: String]) -> Bool {
        guard let raw = arguments["todos"]?.trimmingCharacters(in: .whitespacesAndNewlines)
        else { return false }
        if raw == "[]" { return true }
        if let data = raw.data(using: .utf8),
            let dict = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let list = dict["todos"] as? [Any], list.isEmpty
        {
            return true
        }
        return false
    }

    /// Executes the TodoWrite update, generates output text and triggers the update callback.
    public static func execute(
        arguments: [String: String],
        chatID: UUID? = nil
    ) throws -> (output: String, todos: [TodoItem]) {
        let newTodos = try parseTodos(from: arguments)
        // Only an explicit empty list clears the checklist. A truncated
        // payload, an unknown item key, or `{}` parses to nothing and used to
        // wipe the saved list while telling the model the update succeeded.
        if newTodos.isEmpty && !isExplicitClear(arguments) {
            throw NSError(
                domain: "TurboSparkTool", code: 80,
                userInfo: [NSLocalizedDescriptionKey:
                    "TodoWrite could not read any todo items from 'todos' (expected a JSON array "
                    + "of objects with 'content' and 'status'). The existing list was left unchanged. "
                    + "Pass \"[]\" to clear it."])
        }
        onTodosUpdated?(chatID, newTodos)

        let completedCount = newTodos.filter { $0.isCompleted }.count
        let inProgressCount = newTodos.filter { $0.isInProgress }.count
        let pendingCount = newTodos.filter { $0.isPending }.count
        let total = newTodos.count

        var lines: [String] = []
        lines.append("Todo list updated successfully (\(total) items: \(completedCount) completed, \(inProgressCount) in progress, \(pendingCount) pending).")
        lines.append("")
        lines.append("Current Task Checklist:")

        for (idx, item) in newTodos.enumerated() {
            let num = idx + 1
            let mark: String
            if item.isCompleted {
                mark = "[x]"
            } else if item.isInProgress {
                mark = "[>]"
            } else if item.isCancelled {
                mark = "[-]"
            } else {
                mark = "[ ]"
            }
            let activeNote = (item.isInProgress && !item.activeForm.isEmpty && item.activeForm != item.content)
                ? " (Active: \(item.activeForm))"
                : ""
            lines.append("\(num). \(mark) \(item.content)\(activeNote)")
        }

        lines.append("")
        lines.append("Ensure you continue to update the todo list as you progress through tasks.")

        return (output: lines.joined(separator: "\n"), todos: newTodos)
    }
}
