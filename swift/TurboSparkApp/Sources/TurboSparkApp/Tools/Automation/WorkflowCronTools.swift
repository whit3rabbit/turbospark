import Foundation

// MARK: - Workflow Tool

public struct WorkflowInput: Codable, Sendable, Equatable {
    public var script: String?
    public var name: String?
    public var title: String?
    public var scriptPath: String?
    public var resumeFromRunId: String?

    public init(
        script: String? = nil,
        name: String? = nil,
        title: String? = nil,
        scriptPath: String? = nil,
        resumeFromRunId: String? = nil
    ) {
        self.script = script
        self.name = name
        self.title = title
        self.scriptPath = scriptPath
        self.resumeFromRunId = resumeFromRunId
    }
}

public struct WorkflowOutput: Codable, Sendable, Equatable {
    public var status: String
    public var taskId: String
    public var workflowName: String?
    public var runId: String?
    public var summary: String?
    public var scriptPath: String?
    public var error: String?

    public init(
        status: String = "async_launched",
        taskId: String,
        workflowName: String? = nil,
        runId: String? = nil,
        summary: String? = nil,
        scriptPath: String? = nil,
        error: String? = nil
    ) {
        self.status = status
        self.taskId = taskId
        self.workflowName = workflowName
        self.runId = runId
        self.summary = summary
        self.scriptPath = scriptPath
        self.error = error
    }
}

// MARK: - Cron Tools

public struct CronCreateInput: Codable, Sendable, Equatable {
    public var cron: String
    public var prompt: String
    public var recurring: Bool?
    public var durable: Bool?

    public init(cron: String, prompt: String, recurring: Bool? = true, durable: Bool? = false) {
        self.cron = cron
        self.prompt = prompt
        self.recurring = recurring
        self.durable = durable
    }
}

public struct CronCreateOutput: Codable, Sendable, Equatable {
    public var id: String
    public var humanSchedule: String
    public var recurring: Bool
    public var durable: Bool?

    public init(id: String, humanSchedule: String, recurring: Bool = true, durable: Bool? = false) {
        self.id = id
        self.humanSchedule = humanSchedule
        self.recurring = recurring
        self.durable = durable
    }
}

public struct CronDeleteInput: Codable, Sendable, Equatable {
    public var id: String

    public init(id: String) {
        self.id = id
    }
}

public struct CronDeleteOutput: Codable, Sendable, Equatable {
    public var id: String

    public init(id: String) {
        self.id = id
    }
}

public struct CronListInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct CronJobItem: Codable, Sendable, Equatable {
    public var id: String
    public var cron: String
    public var humanSchedule: String
    public var prompt: String
    public var recurring: Bool?
    public var durable: Bool?

    public init(
        id: String,
        cron: String,
        humanSchedule: String,
        prompt: String,
        recurring: Bool? = true,
        durable: Bool? = false
    ) {
        self.id = id
        self.cron = cron
        self.humanSchedule = humanSchedule
        self.prompt = prompt
        self.recurring = recurring
        self.durable = durable
    }
}

public struct CronListOutput: Codable, Sendable, Equatable {
    public var jobs: [CronJobItem]

    public init(jobs: [CronJobItem] = []) {
        self.jobs = jobs
    }
}

// MARK: - ScheduleWakeup Tool

public struct ScheduleWakeupInput: Codable, Sendable, Equatable {
    public var delaySeconds: Int?
    public var reason: String?
    public var prompt: String?
    public var stop: Bool?
    public var noop: Bool?

    public init(
        delaySeconds: Int? = nil,
        reason: String? = nil,
        prompt: String? = nil,
        stop: Bool? = nil,
        noop: Bool? = nil
    ) {
        self.delaySeconds = delaySeconds
        self.reason = reason
        self.prompt = prompt
        self.stop = stop
        self.noop = noop
    }
}

public struct ScheduleWakeupOutput: Codable, Sendable, Equatable {
    public var scheduledFor: Double
    public var clampedDelaySeconds: Int
    public var wasClamped: Bool
    public var stopped: Bool?

    public init(
        scheduledFor: Double,
        clampedDelaySeconds: Int,
        wasClamped: Bool = false,
        stopped: Bool? = nil
    ) {
        self.scheduledFor = scheduledFor
        self.clampedDelaySeconds = clampedDelaySeconds
        self.wasClamped = wasClamped
        self.stopped = stopped
    }
}

// MARK: - OpenAI Tool Definitions for Workflow and Cron

public enum WorkflowCronDefinitions {
    public static let workflow = OpenAITool.function(
        name: "Workflow",
        description: "Execute a multi-stage autonomous agent workflow script.",
        parameters: .object(
            properties: [
                "script": .string(description: "Complete workflow definition script."),
                "name": .string(description: "Name of predefined workflow."),
                "scriptPath": .string(description: "File path to workflow script.")
            ],
            required: []
        )
    )

    public static let cronCreate = OpenAITool.function(
        name: "CronCreate",
        description: "Schedule a recurring or delayed prompt execution via cron expression.",
        parameters: .object(
            properties: [
                "cron": .string(description: "Standard 5-field cron pattern (e.g. '*/5 * * * *')."),
                "prompt": .string(description: "Prompt to execute at schedule intervals."),
                "recurring": .boolean(description: "Whether to repeat or run once."),
                "durable": .boolean(description: "Whether to persist across sessions.")
            ],
            required: ["cron", "prompt"]
        )
    )

    public static let cronDelete = OpenAITool.function(
        name: "CronDelete",
        description: "Delete a scheduled cron job by identifier.",
        parameters: .object(
            properties: [
                "id": .string(description: "Job ID to remove.")
            ],
            required: ["id"]
        )
    )

    public static let cronList = OpenAITool.function(
        name: "CronList",
        description: "List all scheduled cron jobs.",
        parameters: .emptyObject()
    )

    public static let scheduleWakeup = OpenAITool.function(
        name: "ScheduleWakeup",
        description: "Schedule dynamic delayed wakeup for autonomous agent loops.",
        parameters: .object(
            properties: [
                "delaySeconds": .integer(description: "Seconds to delay wakeup (60 to 3600)."),
                "reason": .string(description: "Reason for the chosen delay."),
                "prompt": .string(description: "Prompt to fire on wakeup."),
                "stop": .boolean(description: "Set true to terminate dynamic loop.")
            ],
            required: []
        )
    )

    public static let all: [OpenAITool] = [
        workflow, cronCreate, cronDelete, cronList, scheduleWakeup
    ]
}
