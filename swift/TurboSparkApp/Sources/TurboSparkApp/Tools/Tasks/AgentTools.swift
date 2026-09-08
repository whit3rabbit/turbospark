import Foundation

// MARK: - Agent Tool

/// Input parameters for spawning an autonomous subagent session.
public struct AgentInput: Codable, Sendable, Equatable {
    /// Concise summary of the subagent's objective.
    public var description: String
    /// Complete prompt and task instructions provided to the subagent.
    public var prompt: String
    /// Role profile or specialization template for the subagent.
    public var subagentType: String?
    /// Optional checkpoint model identifier override.
    public var model: String?
    /// Whether the subagent executes in background without blocking the parent conversation.
    public var runInBackground: Bool?
    /// Name or identifier label assigned to the spawned agent.
    public var name: String?
    /// Worktree or filesystem isolation policy.
    public var isolation: String?

    enum CodingKeys: String, CodingKey {
        case description
        case prompt
        case subagentType = "subagent_type"
        case model
        case runInBackground = "run_in_background"
        case name
        case isolation
    }

    public init(
        description: String,
        prompt: String,
        subagentType: String? = nil,
        model: String? = nil,
        runInBackground: Bool? = nil,
        name: String? = nil,
        isolation: String? = nil
    ) {
        self.description = description
        self.prompt = prompt
        self.subagentType = subagentType
        self.model = model
        self.runInBackground = runInBackground
        self.name = name
        self.isolation = isolation
    }
}

/// Result returned from spawning or completing a subagent task.
public enum AgentOutput: Codable, Sendable, Equatable {
    case completed(CompletedPayload)
    case asyncLaunched(AsyncPayload)
    case remoteLaunched(RemotePayload)

    /// Summary payload for a synchronously completed subagent task.
    public struct CompletedPayload: Codable, Sendable, Equatable {
        public var agentId: String
        public var status: String
        public var prompt: String
        public var totalToolUseCount: Int
        public var totalDurationMs: Double
        public var totalTokens: Int

        public init(
            agentId: String,
            status: String = "completed",
            prompt: String,
            totalToolUseCount: Int = 0,
            totalDurationMs: Double = 0.0,
            totalTokens: Int = 0
        ) {
            self.agentId = agentId
            self.status = status
            self.prompt = prompt
            self.totalToolUseCount = totalToolUseCount
            self.totalDurationMs = totalDurationMs
            self.totalTokens = totalTokens
        }
    }

    /// Status payload when a subagent is launched asynchronously.
    public struct AsyncPayload: Codable, Sendable, Equatable {
        public var status: String
        public var agentId: String
        public var description: String
        public var prompt: String
        public var outputFile: String

        public init(
            status: String = "async_launched",
            agentId: String,
            description: String,
            prompt: String,
            outputFile: String
        ) {
            self.status = status
            self.agentId = agentId
            self.description = description
            self.prompt = prompt
            self.outputFile = outputFile
        }
    }

    /// Status payload when a subagent is dispatched to a remote worker.
    public struct RemotePayload: Codable, Sendable, Equatable {
        public var status: String
        public var taskId: String
        public var sessionUrl: String
        public var description: String
        public var prompt: String

        public init(
            status: String = "remote_launched",
            taskId: String,
            sessionUrl: String,
            description: String,
            prompt: String
        ) {
            self.status = status
            self.taskId = taskId
            self.sessionUrl = sessionUrl
            self.description = description
            self.prompt = prompt
        }
    }
}

// MARK: - OpenAI Tool Definition for Agent

/// OpenAI tool definition schema for subagent task invocation.
public enum AgentToolDefinitions {
    public static let agent = OpenAITool.function(
        name: "Agent",
        description: "Spawn a specialized subagent that works in its own fresh context (it cannot "
            + "see this conversation) and reports its final answer back to you as the tool result. "
            + "Brief it fully: it knows only what you put in the prompt. You may issue SEVERAL "
            + "agent calls in one reply; they run concurrently and each returns its own result. "
            + "Set run_in_background to true to launch one without waiting: you get a task id "
            + "immediately and a <task-notification> message when it finishes.",
        parameters: .object(
            properties: [
                "description": .string(description: "A short 3-5 word summary of the task."),
                "prompt": .string(description: "Detailed instructions for the subagent to perform. Include everything it needs to know; it starts with zero context."),
                "subagent_type": .string(description: "Type of specialized agent profile. The agent types listed in the system prompt are what can be resolved here; an unknown name falls back to general-purpose."),
                "model": .string(description: "Optional model override for the subagent. Only 'inherit' (or omitting it) is supported today."),
                "run_in_background": .boolean(description: "Set true to run the subagent in the background. You will be notified with a <task-notification> when it completes; use stop_agent to cancel it."),
                "name": .string(description: "Unique label for the spawned agent.")
            ],
            required: ["description", "prompt"]
        )
    )

    /// Stops one background subagent by id.
    ///
    /// `taskstop` names the TASK-LIST system (`TaskManager`), not this one,
    /// so the kill verb for background agents gets its own name rather than
    /// overloading a word the model already knows to mean something else.
    public static let stopAgent = OpenAITool.function(
        name: "stop_agent",
        description: "Stop a background subagent you launched with run_in_background, by its id. "
            + "The subagent ends with whatever partial answer it had, and a <task-notification> "
            + "with status killed follows.",
        parameters: .object(
            properties: [
                "id": .string(description: "The background subagent id from its launch result (the `id:` line).")
            ],
            required: ["id"]
        )
    )

    public static let all: [OpenAITool] = [agent, stopAgent]
}
