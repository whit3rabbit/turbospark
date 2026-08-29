import Foundation

// MARK: - Monitor Tool

/// Input payload for starting an asynchronous monitoring job.
public struct MonitorInput: Codable, Sendable, Equatable {
    /// Description of the process or event stream being monitored.
    public var description: String
    /// Maximum duration to monitor in milliseconds.
    public var timeoutMs: Int
    /// Whether the monitor should persist across turns until explicitly stopped.
    public var persistent: Bool
    /// Shell command or process to monitor.
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

/// Output returned when a background monitor is registered.
public struct MonitorOutput: Codable, Sendable, Equatable {
    /// Dispatched task identifier.
    public var taskId: String
    /// Configured timeout in milliseconds.
    public var timeoutMs: Int
    /// Persistence status of the monitor task.
    public var persistent: Bool?

    public init(taskId: String, timeoutMs: Int, persistent: Bool? = nil) {
        self.taskId = taskId
        self.timeoutMs = timeoutMs
        self.persistent = persistent
    }
}

// MARK: - Notifications & Remote Triggers

/// Input payload for registering or clearing remote event triggers.
public struct RemoteTriggerInput: Codable, Sendable, Equatable {
    /// Action method ("poll", "cancel", "listen").
    public var action: String
    /// Unique trigger identifier.
    public var triggerId: String?
    /// Associated session identifier.
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

/// Output payload from remote trigger polling or registration.
public struct RemoteTriggerOutput: Codable, Sendable, Equatable {
    /// Response status code.
    public var status: Int
    /// Raw JSON payload.
    public var json: String
    /// Human-readable trigger event summary.
    public var summary: String?

    public init(status: Int, json: String, summary: String? = nil) {
        self.status = status
        self.json = json
        self.summary = summary
    }
}

/// Input payload for draining pending system notifications.
public struct ReadNotificationsInput: Codable, Sendable, Equatable {
    public init() {}
}

/// A queued system notification event item.
public struct NotificationItem: Codable, Sendable, Equatable {
    /// Unique notification identifier.
    public var notificationId: String
    /// Source origin or subsystem of the notification.
    public var origin: String
    /// ISO timestamp when the event was queued.
    public var queuedAt: String
    /// Notification text message.
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

/// Output payload returning queued notification items.
public struct ReadNotificationsOutput: Codable, Sendable, Equatable {
    /// Array of retrieved notification events.
    public var notifications: [NotificationItem]
    /// Count of remaining unread notifications.
    public var remaining: Int

    public init(notifications: [NotificationItem] = [], remaining: Int = 0) {
        self.notifications = notifications
        self.remaining = remaining
    }
}

/// Input payload for posting a proactive user notification banner.
public struct PushNotificationInput: Codable, Sendable, Equatable {
    /// Message body text displayed in the notification.
    public var message: String
    /// Notification status classification.
    public var status: String

    public init(message: String, status: String = "proactive") {
        self.message = message
        self.status = status
    }
}

/// Output returned after sending a user push notification.
public struct PushNotificationOutput: Codable, Sendable, Equatable {
    /// Sent message text.
    public var message: String
    /// Whether delivery was successfully dispatched.
    public var pushSent: Bool?
    /// ISO timestamp when the notification was posted.
    public var sentAt: String?

    public init(message: String, pushSent: Bool? = true, sentAt: String? = nil) {
        self.message = message
        self.pushSent = pushSent
        self.sentAt = sentAt
    }
}

/// Input for presenting the interactive onboarding role picker.
public struct ShowOnboardingRolePickerInput: Codable, Sendable, Equatable {
    public init() {}
}

/// Output returned from the onboarding role picker sheet.
public struct ShowOnboardingRolePickerOutput: Codable, Sendable, Equatable {
    /// Selected role name if chosen.
    public var role: String?
    /// Whether the user dismissed the picker without choosing.
    public var dismissed: Bool?

    public init(role: String? = nil, dismissed: Bool? = nil) {
        self.role = role
        self.dismissed = dismissed
    }
}

/// Input payload for Claude Design integration operations.
public struct ClaudeDesignInput: Codable, Sendable, Equatable {
    /// Design tool operation name.
    public var operation: String

    public init(operation: String) {
        self.operation = operation
    }
}

/// Output payload from Claude Design tool invocation.
public struct ClaudeDesignOutput: Codable, Sendable, Equatable {
    /// Executed operation.
    public var operation: String
    /// Whether an error occurred.
    public var isError: Bool?

    public init(operation: String, isError: Bool? = false) {
        self.operation = operation
        self.isError = isError
    }
}

// MARK: - OpenAI Tool Definitions for Monitoring and Notifications

/// OpenAI tool definition schemas for background monitoring and user notifications.
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
