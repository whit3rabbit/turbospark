import Foundation

// MARK: - Agent Tool

public struct AgentInput: Codable, Sendable, Equatable {
    public var description: String
    public var prompt: String
    public var subagentType: String?
    public var model: String?
    public var runInBackground: Bool?
    public var name: String?
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

public enum AgentOutput: Codable, Sendable, Equatable {
    case completed(CompletedPayload)
    case asyncLaunched(AsyncPayload)
    case remoteLaunched(RemotePayload)

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

public enum AgentToolDefinitions {
    public static let agent = OpenAITool.function(
        name: "Agent",
        description: "Spawn a specialized subagent to perform an autonomous task or workflow.",
        parameters: .object(
            properties: [
                "description": .string(description: "A short 3-5 word summary of the task."),
                "prompt": .string(description: "Detailed instructions for the subagent to perform."),
                "subagent_type": .string(description: "Type of specialized agent profile."),
                "model": .string(description: "Optional model override for the subagent."),
                "run_in_background": .boolean(description: "Whether to run the subagent asynchronously in the background."),
                "name": .string(description: "Unique label for the spawned agent.")
            ],
            required: ["description", "prompt"]
        )
    )

    public static let all: [OpenAITool] = [agent]
}
