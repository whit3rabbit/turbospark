import Foundation

// MARK: - Monitor Tool

public struct MonitorInput: Codable, Sendable, Equatable {
    public var description: String
    public var timeoutMs: Int
    public var persistent: Bool
    public var command: String?

    enum CodingKeys: String, CodingKey {
        case description
        case timeoutMs = "timeout_ms"
        case persistent
        case command
    }

    public init(description: String, timeoutMs: Int = 300000, persistent: Bool = false, command: String? = nil) {
        self.description = description
        self.timeoutMs = timeoutMs
        self.persistent = persistent
        self.command = command
    }
}

public struct MonitorOutput: Codable, Sendable, Equatable {
    public var taskId: String
    public var timeoutMs: Int
    public var persistent: Bool?

    public init(taskId: String, timeoutMs: Int, persistent: Bool? = nil) {
        self.taskId = taskId
        self.timeoutMs = timeoutMs
        self.persistent = persistent
    }
}

// MARK: - Notifications & Remote Triggers

public struct RemoteTriggerInput: Codable, Sendable, Equatable {
    public var action: String
    public var triggerId: String?
    public var sessionId: String?

    enum CodingKeys: String, CodingKey {
        case action
        case triggerId = "trigger_id"
        case sessionId = "session_id"
    }

    public init(action: String, triggerId: String? = nil, sessionId: String? = nil) {
        self.action = action
        self.triggerId = triggerId
        self.sessionId = sessionId
    }
}

public struct RemoteTriggerOutput: Codable, Sendable, Equatable {
    public var status: Int
    public var json: String
    public var summary: String?

    public init(status: Int, json: String, summary: String? = nil) {
        self.status = status
        self.json = json
        self.summary = summary
    }
}

public struct ReadNotificationsInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct NotificationItem: Codable, Sendable, Equatable {
    public var notificationId: String
    public var origin: String
    public var queuedAt: String
    public var content: String

    enum CodingKeys: String, CodingKey {
        case notificationId = "notification_id"
        case origin
        case queuedAt = "queued_at"
        case content
    }

    public init(notificationId: String, origin: String, queuedAt: String, content: String) {
        self.notificationId = notificationId
        self.origin = origin
        self.queuedAt = queuedAt
        self.content = content
    }
}

public struct ReadNotificationsOutput: Codable, Sendable, Equatable {
    public var notifications: [NotificationItem]
    public var remaining: Int

    public init(notifications: [NotificationItem] = [], remaining: Int = 0) {
        self.notifications = notifications
        self.remaining = remaining
    }
}

public struct PushNotificationInput: Codable, Sendable, Equatable {
    public var message: String
    public var status: String

    public init(message: String, status: String = "proactive") {
        self.message = message
        self.status = status
    }
}

public struct PushNotificationOutput: Codable, Sendable, Equatable {
    public var message: String
    public var pushSent: Bool?
    public var sentAt: String?

    public init(message: String, pushSent: Bool? = true, sentAt: String? = nil) {
        self.message = message
        self.pushSent = pushSent
        self.sentAt = sentAt
    }
}

public struct ShowOnboardingRolePickerInput: Codable, Sendable, Equatable {
    public init() {}
}

public struct ShowOnboardingRolePickerOutput: Codable, Sendable, Equatable {
    public var role: String?
    public var dismissed: Bool?

    public init(role: String? = nil, dismissed: Bool? = nil) {
        self.role = role
        self.dismissed = dismissed
    }
}

public struct ClaudeDesignInput: Codable, Sendable, Equatable {
    public var operation: String

    public init(operation: String) {
        self.operation = operation
    }
}

public struct ClaudeDesignOutput: Codable, Sendable, Equatable {
    public var operation: String
    public var isError: Bool?

    public init(operation: String, isError: Bool? = false) {
        self.operation = operation
        self.isError = isError
    }
}

// MARK: - OpenAI Tool Definitions for Monitoring and Notifications

public enum MonitoringNotificationDefinitions {
    public static let monitor = OpenAITool.function(
        name: "Monitor",
        description: "Monitor a command output or event stream in background.",
        parameters: .object(
            properties: [
                "description": .string(description: "Description of what is monitored."),
                "timeout_ms": .integer(description: "Timeout in milliseconds."),
                "persistent": .boolean(description: "Run for session lifetime."),
                "command": .string(description: "Shell command to monitor.")
            ],
            required: ["description", "timeout_ms", "persistent"]
        )
    )

    public static let pushNotification = OpenAITool.function(
        name: "PushNotification",
        description: "Send a user notification for important milestone events.",
        parameters: .object(
            properties: [
                "message": .string(description: "Notification text body."),
                "status": .string(description: "Status type: 'proactive'.")
            ],
            required: ["message", "status"]
        )
    )

    public static let all: [OpenAITool] = [
        monitor, pushNotification
    ]
}
