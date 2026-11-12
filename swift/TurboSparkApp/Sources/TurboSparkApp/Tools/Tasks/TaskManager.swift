import Foundation

/// Persistent and in-memory management for structured project tasks and subagent jobs.
public final class TaskManager: @unchecked Sendable {
    public static let shared = TaskManager()

    public struct TrackedTask: Codable, Sendable, Identifiable {
        public var id: String
        public var subject: String
        public var description: String
        public var status: String // "pending", "in_progress", "completed", "stopped", "failed"
        public var activeForm: String
        public var output: String
        public var createdAt: Date
        public var updatedAt: Date

        public init(
            id: String = UUID().uuidString.prefix(8).lowercased(),
            subject: String,
            description: String = "",
            status: String = "pending",
            activeForm: String = "",
            output: String = "",
            createdAt: Date = Date(),
            updatedAt: Date = Date()
        ) {
            self.id = id
            self.subject = subject
            self.description = description
            self.status = status
            self.activeForm = activeForm.isEmpty ? subject : activeForm
            self.output = output
            self.createdAt = createdAt
            self.updatedAt = updatedAt
        }
    }

    private let lock = NSLock()
    private var tasks: [String: TrackedTask] = [:]
    private var chatTasks: [String: [String]] = [:]

    public var onTasksUpdated: (@Sendable (UUID?, [TrackedTask]) -> Void)?

    /// Source of new task ids. Internal so tests can force collisions; the
    /// default is the first 8 hex characters of a UUID (32 bits).
    var idGenerator: @Sendable () -> String = { UUID().uuidString.prefix(8).lowercased() }

    private init() {}

    /// Statuses after which a task will not change again on its own. A
    /// blocking `TaskOutput` stops waiting once it sees one of these.
    static func isTerminalStatus(_ status: String) -> Bool {
        switch status.lowercased() {
        case "completed", "done", "stopped", "failed", "cancelled", "canceled", "deleted":
            return true
        default:
            return false
        }
    }

    public func createTask(subject: String, description: String, activeForm: String? = nil, chatID: UUID? = nil) -> TrackedTask {
        lock.lock()
        // 8 UUID characters can collide; an unchecked insert would silently
        // overwrite another task. Retry, then fall back to a full UUID.
        var id = idGenerator()
        var attempts = 0
        while tasks[id] != nil {
            attempts += 1
            id = attempts < 16 ? idGenerator() : UUID().uuidString.lowercased()
        }
        let task = TrackedTask(
            id: id,
            subject: subject,
            description: description,
            activeForm: activeForm ?? subject
        )
        tasks[task.id] = task
        let key = chatID?.uuidString ?? "default"
        var list = chatTasks[key] ?? []
        list.append(task.id)
        chatTasks[key] = list
        let allChatTasks = list.compactMap { tasks[$0] }
        lock.unlock()

        onTasksUpdated?(chatID, allChatTasks)
        return task
    }

    public func getTask(id: String) -> TrackedTask? {
        lock.lock()
        defer { lock.unlock() }
        return tasks[id]
    }

    public func listTasks(chatID: UUID? = nil) -> [TrackedTask] {
        lock.lock()
        defer { lock.unlock() }
        if let chatID {
            let key = chatID.uuidString
            let ids = chatTasks[key] ?? []
            return ids.compactMap { tasks[$0] }
        }
        return Array(tasks.values).sorted { $0.createdAt < $1.createdAt }
    }

    public func updateTask(
        id: String,
        subject: String? = nil,
        description: String? = nil,
        status: String? = nil,
        activeForm: String? = nil,
        output: String? = nil,
        chatID: UUID? = nil
    ) -> TrackedTask? {
        lock.lock()
        guard var task = tasks[id] else {
            lock.unlock()
            return nil
        }
        // "deleted" is the schema's way to remove a task. Drop it from the
        // store and from every chat list (the caller's chatID may not be the
        // one that created it), and return the final snapshot.
        if status?.lowercased() == "deleted" {
            task.status = "deleted"
            task.updatedAt = Date()
            tasks[id] = nil
            for key in chatTasks.keys {
                chatTasks[key]?.removeAll { $0 == id }
            }
            let key = chatID?.uuidString ?? "default"
            let remaining = (chatTasks[key] ?? []).compactMap { tasks[$0] }
            lock.unlock()
            onTasksUpdated?(chatID, remaining)
            return task
        }
        if let subject { task.subject = subject }
        if let description { task.description = description }
        if let status { task.status = status }
        if let activeForm { task.activeForm = activeForm }
        if let output { task.output = output }
        task.updatedAt = Date()
        tasks[id] = task

        let key = chatID?.uuidString ?? "default"
        let list = chatTasks[key] ?? []
        let allChatTasks = list.compactMap { tasks[$0] }
        lock.unlock()

        onTasksUpdated?(chatID, allChatTasks)
        return task
    }

    public func stopTask(id: String, chatID: UUID? = nil) -> Bool {
        lock.lock()
        guard var task = tasks[id] else {
            lock.unlock()
            return false
        }
        task.status = "stopped"
        task.updatedAt = Date()
        tasks[id] = task

        let key = chatID?.uuidString ?? "default"
        let list = chatTasks[key] ?? []
        let allChatTasks = list.compactMap { tasks[$0] }
        lock.unlock()

        onTasksUpdated?(chatID, allChatTasks)
        return true
    }

    // MARK: - Tool Call Executors

    public static func executeCreate(arguments: [String: String], chatID: UUID?) throws -> String {
        guard let subject = arguments["subject"] ?? arguments["title"] ?? arguments["name"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 50,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'subject' argument for TaskCreate."]
            )
        }
        let desc = arguments["description"] ?? arguments["details"] ?? ""
        let activeForm = arguments["activeForm"] ?? arguments["active_form"]
        let task = TaskManager.shared.createTask(subject: subject, description: desc, activeForm: activeForm, chatID: chatID)
        return "Task created with ID '\(task.id)': \(task.subject) (Status: \(task.status))"
    }

    public static func executeGet(arguments: [String: String]) throws -> String {
        guard let taskId = arguments["taskId"] ?? arguments["task_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 51,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'taskId' argument for TaskGet."]
            )
        }
        guard let task = TaskManager.shared.getTask(id: taskId) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 51,
                userInfo: [NSLocalizedDescriptionKey: "Task with ID '\(taskId)' not found."]
            )
        }
        var res = "Task details for '\(task.id)':\n"
        res += "- Subject: \(task.subject)\n"
        res += "- Status: \(task.status)\n"
        res += "- Description: \(task.description.isEmpty ? "(None)" : task.description)\n"
        if !task.output.isEmpty {
            res += "- Output: \(task.output)\n"
        }
        return res
    }

    public static func executeList(arguments: [String: String], chatID: UUID?) -> String {
        let tasks = TaskManager.shared.listTasks(chatID: chatID)
        if tasks.isEmpty {
            return "No active or tracked tasks in workspace."
        }
        var res = "Tracked Tasks (\(tasks.count)):\n"
        for t in tasks {
            let statusIcon: String
            switch t.status.lowercased() {
            case "completed", "done": statusIcon = "[✓]"
            case "in_progress", "running": statusIcon = "[▶]"
            case "stopped", "cancelled": statusIcon = "[x]"
            default: statusIcon = "[ ]"
            }
            res += "\n\(statusIcon) [\(t.id)] \(t.subject) (\(t.status))"
            if !t.description.isEmpty {
                res += "\n   Description: \(t.description)"
            }
        }
        return res
    }

    public static func executeUpdate(arguments: [String: String], chatID: UUID?) throws -> String {
        guard let taskId = arguments["taskId"] ?? arguments["task_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 52,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'taskId' argument for TaskUpdate."]
            )
        }
        let subject = arguments["subject"] ?? arguments["title"]
        let desc = arguments["description"]
        let status = arguments["status"]
        let activeForm = arguments["activeForm"] ?? arguments["active_form"]
        let output = arguments["output"]

        guard let updated = TaskManager.shared.updateTask(
            id: taskId,
            subject: subject,
            description: desc,
            status: status,
            activeForm: activeForm,
            output: output,
            chatID: chatID
        ) else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 52,
                userInfo: [NSLocalizedDescriptionKey: "Task with ID '\(taskId)' not found."]
            )
        }
        return "Updated task '\(updated.id)' (Status: \(updated.status), Subject: \(updated.subject))."
    }

    public static func executeStop(arguments: [String: String], chatID: UUID?) throws -> String {
        guard let taskId = arguments["taskId"] ?? arguments["task_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 53,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'task_id' argument for TaskStop."]
            )
        }
        let stopped = TaskManager.shared.stopTask(id: taskId, chatID: chatID)
        guard stopped else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 53,
                userInfo: [NSLocalizedDescriptionKey: "Task with ID '\(taskId)' not found or could not be stopped."]
            )
        }
        return "Task '\(taskId)' was successfully stopped."
    }

    /// Upper bound for a blocking wait, so a model-supplied timeout cannot park a tool call forever.
    static let maxOutputWaitMilliseconds = 600_000
    static let defaultOutputWaitMilliseconds = 10_000

    /// Reads the recorded `output` of a task from the task list. When `block`
    /// is true, waits (polling) until the task reaches a terminal status or
    /// `timeout` ms elapse. This does NOT read a background shell or subagent:
    /// tasks are not linked to those, so `output` is only what was stored with
    /// TaskUpdate.
    public static func executeOutput(arguments: [String: String]) async throws -> String {
        guard let taskId = arguments["taskId"] ?? arguments["task_id"] ?? arguments["id"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 54,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'task_id' argument for TaskOutput."]
            )
        }
        func lookup() throws -> TrackedTask {
            guard let task = TaskManager.shared.getTask(id: taskId) else {
                throw NSError(
                    domain: "TurboSparkTool",
                    code: 54,
                    userInfo: [NSLocalizedDescriptionKey: "Task with ID '\(taskId)' not found."]
                )
            }
            return task
        }

        let block = ["true", "1", "yes"].contains((arguments["block"] ?? "").lowercased())
        let requested = arguments["timeout"].flatMap { Int($0) } ?? defaultOutputWaitMilliseconds
        let timeoutMs = min(max(requested, 0), maxOutputWaitMilliseconds)

        var task = try lookup()
        var timedOut = false
        if block {
            let deadline = Date().addingTimeInterval(Double(timeoutMs) / 1000)
            while !isTerminalStatus(task.status) {
                if Date() >= deadline {
                    timedOut = true
                    break
                }
                try await Task.sleep(nanoseconds: 50_000_000)
                // A deleted task disappears, which ends the wait with a not-found error.
                task = try lookup()
            }
        }

        var res = task.output.isEmpty
            ? "Task '\(taskId)' has no recorded output yet (Status: \(task.status))."
            : task.output
        if timedOut {
            res += "\n[Timed out after \(timeoutMs) ms waiting for task '\(taskId)' to finish (Status: \(task.status))]"
        }
        return res
    }
}
