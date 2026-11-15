import Foundation

/// Tool schemas for the off-peak queue. Same permission category as the cron
/// tools (`.automation`): denied in Strict Read-Only, asked otherwise, and
/// never auto-approved by this feature.
public enum OffPeakTaskDefinitions {
    public static let enqueue = OpenAITool.function(
        name: "OffPeakEnqueue",
        description: "Queue a prompt to run later, only while the user allows off-peak tasks and the chat is idle. Runs in this chat, one task at a time, in the order queued. Use for long or low-priority work with no deadline.",
        parameters: .object(
            properties: [
                "prompt": .string(description: "The instruction to run when the machine is idle.")
            ],
            required: ["prompt"]
        )
    )

    public static let list = OpenAITool.function(
        name: "OffPeakList",
        description: "List off-peak tasks with their state.",
        parameters: .emptyObject()
    )

    public static let cancel = OpenAITool.function(
        name: "OffPeakCancel",
        description: "Cancel a queued off-peak task by identifier. A running task cannot be cancelled here.",
        parameters: .object(
            properties: [
                "id": .string(description: "Task ID to cancel.")
            ],
            required: ["id"]
        )
    )

    public static let all: [OpenAITool] = [enqueue, list, cancel]
}

/// Executors for the off-peak tools. Static like the cron executors: tool
/// dispatch reaches them without an `AppModel`. `store` is a parameter so a
/// test passes an isolated queue.
public enum OffPeakTaskExecutors {
    public static func executeEnqueue(
        arguments: [String: String], chatID: UUID?, chatEligible: Bool = true,
        store: OffPeakTaskStore = .shared
    ) throws -> String {
        guard let prompt = arguments["prompt"] ?? arguments["instruction"] else {
            throw NSError(domain: "TurboSparkOffPeak", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'prompt' argument for OffPeakEnqueue."
            ])
        }
        guard let chatID else {
            throw NSError(domain: "TurboSparkOffPeak", code: 9, userInfo: [
                NSLocalizedDescriptionKey:
                    "OffPeakEnqueue needs an active chat to run its prompt in."
            ])
        }
        // A ghost chat never persists conversation content, and a queued
        // prompt on disk is content.
        guard chatEligible else {
            throw NSError(domain: "TurboSparkOffPeak", code: 3, userInfo: [
                NSLocalizedDescriptionKey:
                    "This chat cannot hold off-peak tasks (temporary chats are never saved)."
            ])
        }
        let task = try store.enqueue(prompt: prompt, chatID: chatID, origin: .model)
        return "Off-peak task '\(task.id)' queued (\(store.activeCount) active). "
            + "It runs in this chat only when the user has turned on off-peak tasks "
            + "in Settings > Scheduled Tasks and the chat is open and idle. "
            + "It never approves tool calls on its own: a run that needs approval stops."
    }

    public static func executeList(
        arguments: [String: String], store: OffPeakTaskStore = .shared
    ) -> String {
        let tasks = store.list()
        if tasks.isEmpty { return "No off-peak tasks. Queue one with OffPeakEnqueue." }
        var lines = ["Off-peak tasks (\(tasks.count)):"]
        for task in tasks {
            let origin = task.origin == .model ? "assistant" : "user"
            lines.append("- [\(task.id)] \(task.state.rawValue) (added by \(origin))")
            lines.append("  Prompt: \(String(task.prompt.prefix(120)))")
        }
        return lines.joined(separator: "\n")
    }

    public static func executeCancel(
        arguments: [String: String], store: OffPeakTaskStore = .shared
    ) throws -> String {
        guard let id = arguments["id"] ?? arguments["taskId"] ?? arguments["task_id"] else {
            throw NSError(domain: "TurboSparkOffPeak", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Missing 'id' argument for OffPeakCancel."
            ])
        }
        try store.cancel(id: id)
        return "Off-peak task '\(id)' cancelled."
    }
}
