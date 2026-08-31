import Foundation

// MARK: - Task Management Tools

/// Input parameters for creating a new structured project task item.
public struct TaskCreateInput: Codable, Sendable, Equatable {
    /// Brief summary title of the task.
    public var subject: String
    /// Detailed description and acceptance criteria.
    public var description: String
    /// Present continuous activity label shown while working on the task.
    public var activeForm: String?

    public init(subject: String, description: String, activeForm: String? = nil) {
        self.subject = subject
        self.description = description
        self.activeForm = activeForm
    }
}

/// Output payload returned upon creating a new task.
public struct TaskCreateOutput: Codable, Sendable, Equatable {
    /// Newly allocated task identifier.
    public var taskId: String
    /// Subject title of created task.
    public var subject: String

    public init(taskId: String, subject: String) {
        self.taskId = taskId
        self.subject = subject
    }
}

/// Input payload for fetching a specific task item.
public struct TaskGetInput: Codable, Sendable, Equatable {
    /// Task identifier.
    public var taskId: String

    public init(taskId: String) {
        self.taskId = taskId
    }
}

/// Detailed state information for a tracked task item.
public struct TaskItemInfo: Codable, Sendable, Equatable {
    /// Task ID.
    public var id: String
    /// Subject title.
    public var subject: String
    /// Detailed description.
    public var description: String
    /// Current task status ("pending", "in_progress", "completed", "deleted").
    public var status: String
    /// Task IDs blocked by this task.
    public var blocks: [String]
    /// Task IDs that block this task.
    public var blockedBy: [String]

    public init(
        id: String,
        subject: String,
        description: String = "",
        status: String = "pending",
        blocks: [String] = [],
        blockedBy: [String] = []
    ) {
        self.id = id
        self.subject = subject
        self.description = description
        self.status = status
        self.blocks = blocks
        self.blockedBy = blockedBy
    }
}

/// Output payload containing fetched task details.
public struct TaskGetOutput: Codable, Sendable, Equatable {
    /// Retrieved task info, or nil if not found.
    public var task: TaskItemInfo?

    public init(task: TaskItemInfo? = nil) {
        self.task = task
    }
}

/// Input parameters for modifying an existing task item.
public struct TaskUpdateInput: Codable, Sendable, Equatable {
    /// Target task identifier.
    public var taskId: String
    /// Updated title.
    public var subject: String?
    /// Updated description.
    public var description: String?
    /// Updated activity label.
    public var activeForm: String?
    /// New status ("pending", "in_progress", "completed", "deleted").
    public var status: String?
    /// Tasks to mark as blocked by this task.
    public var addBlocks: [String]?
    /// Tasks that block this task.
    public var addBlockedBy: [String]?
    /// Assigned owner label.
    public var owner: String?

    public init(
        taskId: String,
        subject: String? = nil,
        description: String? = nil,
        activeForm: String? = nil,
        status: String? = nil,
        addBlocks: [String]? = nil,
        addBlockedBy: [String]? = nil,
        owner: String? = nil
    ) {
        self.taskId = taskId
        self.subject = subject
        self.description = description
        self.activeForm = activeForm
        self.status = status
        self.addBlocks = addBlocks
        self.addBlockedBy = addBlockedBy
        self.owner = owner
    }
}

/// Output returned upon updating a task item.
public struct TaskUpdateOutput: Codable, Sendable, Equatable {
    /// Whether the update was applied.
    public var success: Bool
    /// Updated task identifier.
    public var taskId: String
    /// List of property names changed.
    public var updatedFields: [String]
    /// Error message if update failed.
    public var error: String?

    public init(
        success: Bool,
        taskId: String,
        updatedFields: [String] = [],
        error: String? = nil
    ) {
        self.success = success
        self.taskId = taskId
        self.updatedFields = updatedFields
        self.error = error
    }
}

/// Input payload for listing all tasks.
public struct TaskListInput: Codable, Sendable, Equatable {
    public init() {}
}

/// Output payload containing all project tasks.
public struct TaskListOutput: Codable, Sendable, Equatable {
    public var tasks: [TaskSummary]

    /// Summary info for a task item in listing.
    public struct TaskSummary: Codable, Sendable, Equatable {
        public var id: String
        public var subject: String
        public var status: String
        public var owner: String?
        public var blockedBy: [String]

        public init(
            id: String,
            subject: String,
            status: String,
            owner: String? = nil,
            blockedBy: [String] = []
        ) {
            self.id = id
            self.subject = subject
            self.status = status
            self.owner = owner
            self.blockedBy = blockedBy
        }
    }

    public init(tasks: [TaskSummary] = []) {
        self.tasks = tasks
    }
}

/// Input payload for terminating a background task.
public struct TaskStopInput: Codable, Sendable, Equatable {
    /// Task identifier to terminate.
    public var taskId: String?
    /// Shell session identifier.
    public var shellId: String?

    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case shellId = "shell_id"
    }

    public init(taskId: String? = nil, shellId: String? = nil) {
        self.taskId = taskId
        self.shellId = shellId
    }
}

/// Output returned after stopping a task.
public struct TaskStopOutput: Codable, Sendable, Equatable {
    /// Status message.
    public var message: String
    /// Stopped task identifier.
    public var taskId: String
    /// Kind of task stopped ("process", "subagent").
    public var taskType: String

    enum CodingKeys: String, CodingKey {
        case message
        case taskId = "task_id"
        case taskType = "task_type"
    }

    public init(message: String, taskId: String, taskType: String = "process") {
        self.message = message
        self.taskId = taskId
        self.taskType = taskType
    }
}

/// Input parameters for querying output from a background task.
public struct TaskOutputInput: Codable, Sendable, Equatable {
    /// Task identifier.
    public var taskId: String
    /// Whether to block waiting for completion.
    public var block: Bool
    /// Maximum wait timeout in milliseconds.
    public var timeout: Int

    enum CodingKeys: String, CodingKey {
        case taskId = "task_id"
        case block
        case timeout
    }

    public init(taskId: String, block: Bool = false, timeout: Int = 10000) {
        self.taskId = taskId
        self.block = block
        self.timeout = timeout
    }
}

// MARK: - TodoWrite Tool

/// A single checklist todo item for tracking agent progress across tasks.
public struct TodoItem: Codable, Sendable, Equatable, Identifiable {
    /// Unique identifier for the item.
    public var id: String
    /// Task item description (imperative form, e.g. "Run unit tests").
    public var content: String
    /// Status ("pending", "in_progress", "completed", "cancelled").
    public var status: String
    /// Active progress description (present continuous form, e.g. "Running unit tests").
    public var activeForm: String

    public init(
        id: String = UUID().uuidString,
        content: String,
        status: String = "pending",
        activeForm: String = ""
    ) {
        self.id = id
        self.content = content
        self.status = status
        self.activeForm = activeForm.isEmpty ? content : activeForm
    }

    public var isCompleted: Bool {
        let s = status.lowercased()
        return s == "completed" || s == "done" || s == "finished"
    }

    public var isInProgress: Bool {
        let s = status.lowercased()
        return s == "in_progress" || s == "in-progress" || s == "running" || s == "active"
    }

    public var isPending: Bool {
        let s = status.lowercased()
        return s == "pending" || s == "todo" || s.isEmpty
    }

    public var isCancelled: Bool {
        let s = status.lowercased()
        return s == "cancelled" || s == "canceled" || s == "deleted"
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        content = try container.decodeIfPresent(String.self, forKey: .content) ?? ""
        status = try container.decodeIfPresent(String.self, forKey: .status) ?? "pending"
        let rawActive = try container.decodeIfPresent(String.self, forKey: .activeForm) ?? ""
        activeForm = rawActive.isEmpty ? content : rawActive
    }
}

/// Input payload for updating the interactive todo list.
public struct TodoWriteInput: Codable, Sendable, Equatable {
    /// Array of updated todo items.
    public var todos: [TodoItem]

    public init(todos: [TodoItem] = []) {
        self.todos = todos
    }
}

/// Output payload from updating the todo checklist.
public struct TodoWriteOutput: Codable, Sendable, Equatable {
    /// Summary message of updated items.
    public var message: String
    /// Previous list of todos.
    public var oldTodos: [TodoItem]
    /// New list of todos.
    public var newTodos: [TodoItem]

    public init(message: String = "", oldTodos: [TodoItem] = [], newTodos: [TodoItem] = []) {
        self.message = message
        self.oldTodos = oldTodos
        self.newTodos = newTodos
    }
}

// MARK: - OpenAI Tool Definitions for Task Management

/// OpenAI tool definition schemas for TaskCreate, TaskGet, TaskUpdate, TaskList, TaskStop, TaskOutput, and TodoWrite.
public enum TaskItemToolDefinitions {
    public static let taskCreate = OpenAITool.function(
        name: "TaskCreate",
        description: "Create a new trackable task in the active project task list.",
        parameters: .object(
            properties: [
                "subject": .string(description: "Brief summary title for the task."),
                "description": .string(description: "Detailed requirements and acceptance criteria."),
                "activeForm": .string(description: "Present continuous label for progress display (e.g. 'Running tests').")
            ],
            required: ["subject", "description"]
        )
    )

    public static let taskGet = OpenAITool.function(
        name: "TaskGet",
        description: "Retrieve details and status for a specific task by ID.",
        parameters: .object(
            properties: [
                "taskId": .string(description: "The unique task identifier.")
            ],
            required: ["taskId"]
        )
    )

    public static let taskUpdate = OpenAITool.function(
        name: "TaskUpdate",
        description: "Update the status, description, or dependencies of an existing task.",
        parameters: .object(
            properties: [
                "taskId": .string(description: "The unique task identifier."),
                "subject": .string(description: "New title for the task."),
                "description": .string(description: "Updated description."),
                "status": .string(description: "New status: 'pending', 'in_progress', 'completed', or 'deleted'."),
                "activeForm": .string(description: "Present continuous activity label.")
            ],
            required: ["taskId"]
        )
    )

    public static let taskList = OpenAITool.function(
        name: "TaskList",
        description: "List all active, pending, and completed tasks in the workspace.",
        parameters: .emptyObject()
    )

    public static let taskStop = OpenAITool.function(
        name: "TaskStop",
        description: "Stop or cancel a running background task or subagent.",
        parameters: .object(
            properties: [
                "task_id": .string(description: "Identifier of the task or agent to terminate.")
            ],
            required: ["task_id"]
        )
    )

    public static let taskOutput = OpenAITool.function(
        name: "TaskOutput",
        description: "Retrieve stdout, stderr, or result output from a background task.",
        parameters: .object(
            properties: [
                "task_id": .string(description: "The unique task identifier."),
                "block": .boolean(description: "Whether to wait synchronously for task completion."),
                "timeout": .integer(description: "Max wait time in milliseconds.")
            ],
            required: ["task_id", "block", "timeout"]
        )
    )

    public static let todoWrite = OpenAITool.function(
        name: "TodoWrite",
        description: "Update the task checklist for the session. Use proactively for multi-step tasks (3+ steps), marking items in_progress BEFORE working and completed IMMEDIATELY after finishing. Keep at most ONE task in_progress at a time.",
        parameters: .object(
            properties: [
                "todos": .array(
                    items: .object(
                        properties: [
                            "id": .string(description: "Optional unique task ID."),
                            "content": .string(description: "Imperative task description (e.g. 'Run test suite')."),
                            "status": .string(description: "Status: 'pending', 'in_progress', 'completed', 'cancelled'."),
                            "activeForm": .string(description: "Present continuous label (e.g. 'Running test suite').")
                        ],
                        required: ["content", "status", "activeForm"]
                    ),
                    description: "Full updated list of todo items."
                )
            ],
            required: ["todos"]
        )
    )

    public static let all: [OpenAITool] = [
        taskCreate, taskGet, taskUpdate, taskList, taskStop, taskOutput, todoWrite
    ]
}
