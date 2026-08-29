import Foundation

// MARK: - Workflow Tool

/// Input parameters for initiating a multi-step agent workflow.
public struct WorkflowInput: Codable, Sendable, Equatable {
    /// Inline workflow script content.
    public var script: String?
    /// Identifier name of a predefined workflow.
    public var name: String?
    /// Human-readable title for the workflow run.
    public var title: String?
    /// Path to workflow definition file.
    public var scriptPath: String?
    /// Prior run ID to resume execution from.
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

/// Output returned when a workflow script is launched or completed.
public struct WorkflowOutput: Codable, Sendable, Equatable {
    /// Workflow run status.
    public var status: String
    /// Asynchronous task identifier.
    public var taskId: String
    /// Workflow name.
    public var workflowName: String?
    /// Unique workflow execution run ID.
    public var runId: String?
    /// Summary of workflow steps or results.
    public var summary: String?
    /// Script path executed.
    public var scriptPath: String?
    /// Error message string if workflow failed.
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

/// Input parameters for creating a scheduled prompt execution cron job.
public struct CronCreateInput: Codable, Sendable, Equatable {
    /// Standard 5-field cron expression.
    public var cron: String
    /// Prompt instruction executed when the cron fires.
    public var prompt: String
    /// Whether the cron repeats continuously or expires after one execution.
    public var recurring: Bool?
    /// Whether the cron persists across app restarts.
    public var durable: Bool?

    public init(cron: String, prompt: String, recurring: Bool? = true, durable: Bool? = false) {
        self.cron = cron
        self.prompt = prompt
        self.recurring = recurring
        self.durable = durable
    }
}

/// Output returned when a new cron job is registered.
public struct CronCreateOutput: Codable, Sendable, Equatable {
    /// Unique job identifier.
    public var id: String
    /// Human-friendly description of the schedule.
    public var humanSchedule: String
    /// Whether the job repeats.
    public var recurring: Bool
    /// Persistence status across sessions.
    public var durable: Bool?

    public init(id: String, humanSchedule: String, recurring: Bool = true, durable: Bool? = false) {
        self.id = id
        self.humanSchedule = humanSchedule
        self.recurring = recurring
        self.durable = durable
    }
}

/// Input payload for deleting a scheduled cron job.
public struct CronDeleteInput: Codable, Sendable, Equatable {
    /// Job identifier to delete.
    public var id: String

    public init(id: String) {
        self.id = id
    }
}

/// Output returned after deleting a cron job.
public struct CronDeleteOutput: Codable, Sendable, Equatable {
    /// Deleted job identifier.
    public var id: String

    public init(id: String) {
        self.id = id
    }
}

/// Input payload for listing active cron schedules.
public struct CronListInput: Codable, Sendable, Equatable {
    public init() {}
}

/// Description of an active cron schedule item.
public struct CronJobItem: Codable, Sendable, Equatable {
    /// Unique job ID.
    public var id: String
    /// Cron expression pattern.
    public var cron: String
    /// Human-readable schedule description.
    public var humanSchedule: String
    /// Prompt evaluated when triggered.
    public var prompt: String
    /// Whether recurring.
    public var recurring: Bool?
    /// Whether durable.
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

/// Output payload containing active cron job schedules.
public struct CronListOutput: Codable, Sendable, Equatable {
    /// Array of scheduled cron jobs.
    public var jobs: [CronJobItem]

    public init(jobs: [CronJobItem] = []) {
        self.jobs = jobs
    }
}

// MARK: - ScheduleWakeup Tool

/// Input payload for scheduling a one-shot wakeup timer.
public struct ScheduleWakeupInput: Codable, Sendable, Equatable {
    /// Delay interval in seconds.
    public var delaySeconds: Int?
    /// Reason explaining the timer requirement.
    public var reason: String?
    /// Prompt delivered on timer trigger.
    public var prompt: String?
    /// Whether to cancel / stop ongoing wakeup loop.
    public var stop: Bool?
    /// No-op test flag.
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

/// Output returned upon scheduling a wakeup timer.
public struct ScheduleWakeupOutput: Codable, Sendable, Equatable {
    /// Scheduled epoch timestamp in seconds.
    public var scheduledFor: Double
    /// Clamped delay in seconds within bounds.
    public var clampedDelaySeconds: Int
    /// Whether the requested delay exceeded range bounds.
    public var wasClamped: Bool
    /// Whether loop was terminated.
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

/// OpenAI tool definition schemas for workflows, cron schedules, and dynamic timer wakeups.
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
