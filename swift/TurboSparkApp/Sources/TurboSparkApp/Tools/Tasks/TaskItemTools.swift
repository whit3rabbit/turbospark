import Foundation

// MARK: - Task Management Tools

public struct TaskCreateInput: Codable, Sendable, Equatable {
    public var subject: String
    public var description: String
    public var activeForm: String?

    public init(subject: String, description: String, activeForm: String? = nil) {
        self.subject = subject
        self.description = description
        self.activeForm = activeForm
    }
}

public struct TaskCreateOutput: Codable, Sendable, Equatable {
    public var taskId: String
    public var subject: String

    public init(taskId: String, subject: String) {
        self.taskId = taskId
        self.subject = subject
    }
}

public struct TaskGetInput: Codable, Sendable, Equatable {
    public var taskId: String

    public init(taskId: String) {
        self.taskId = taskId
    }
}

public struct TaskItemInfo: Codable, Sendable, Equatable {
    public var id: String
    public var subject: String
    public var description: String
    public var status: String
    public var blocks: [String]
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

public struct TaskGetOutput: Codable, Sendable, Equatable {
    public var task: TaskItemInfo?

    public init(task: TaskItemInfo? = nil) {
        self.task = task
    }
}

public struct TaskUpdateInput: Codable, Sendable, Equatable {
    public var taskId: String
    public var subject: String?
    public var description: String?
    public var activeForm: String?
    public var status: String?
    public var addBlocks: [String]?
    public var addBlockedBy: [String]?
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

public struct TaskUpdateOutput: Codable, Sendable, Equatable {
    public var success: Bool
    public var taskId: String
    public var updatedFields: [String]
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

public struct TaskListInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct TaskListOutput: Codable, Sendable, Equatable {
    public var tasks: [TaskSummary]

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

public struct TaskStopInput: Codable, Sendable, Equatable {
    public var taskId: String?
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

public struct TaskStopOutput: Codable, Sendable, Equatable {
    public var message: String
    public var taskId: String
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

public struct TaskOutputInput: Codable, Sendable, Equatable {
    public var taskId: String
    public var block: Bool
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

public struct TodoItem: Codable, Sendable, Equatable {
    public var content: String
    public var status: String
    public var activeForm: String

    public init(content: String, status: String = "pending", activeForm: String = "") {
        self.content = content
        self.status = status
        self.activeForm = activeForm
    }
}

public struct TodoWriteInput: Codable, Sendable, Equatable {
    public var todos: [TodoItem]

    public init(todos: [TodoItem] = []) {
        self.todos = todos
    }
}

public struct TodoWriteOutput: Codable, Sendable, Equatable {
    public var oldTodos: [TodoItem]
    public var newTodos: [TodoItem]

    public init(oldTodos: [TodoItem] = [], newTodos: [TodoItem] = []) {
        self.oldTodos = oldTodos
        self.newTodos = newTodos
    }
}

// MARK: - OpenAI Tool Definitions for Task Management

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
        description: "Update the project todo list with current items and statuses.",
        parameters: .object(
            properties: [
                "todos": .array(
                    items: .object(
                        properties: [
                            "content": .string(description: "Description of the todo item."),
                            "status": .string(description: "Status: 'pending', 'in_progress', 'completed'."),
                            "activeForm": .string(description: "Progress action label.")
                        ],
                        required: ["content", "status", "activeForm"]
                    ),
                    description: "Array of updated todo items."
                )
            ],
            required: ["todos"]
        )
    )

    public static let all: [OpenAITool] = [
        taskCreate, taskGet, taskUpdate, taskList, taskStop, taskOutput, todoWrite
    ]
}
