import Foundation
import AppKit

// MARK: - Sleep Executor

public enum SleepExecutor {
    public static func execute(arguments: [String: String]) async throws -> String {
        let seconds: Double
        if let secStr = arguments["seconds"] ?? arguments["duration"] ?? arguments["delay"] {
            seconds = Double(secStr) ?? 1.0
        } else if let msStr = arguments["ms"] ?? arguments["milliseconds"] {
            seconds = (Double(msStr) ?? 1000.0) / 1000.0
        } else {
            seconds = 1.0
        }

        let clampedSeconds = max(0.1, min(300.0, seconds))
        let nanos = UInt64(clampedSeconds * 1_000_000_000)
        try await Task.sleep(nanoseconds: nanos)

        return "Slept for \(String(format: "%.1f", clampedSeconds)) second(s)."
    }
}

// MARK: - PushNotification Executor

/// The agent's `PushNotification` tool. It never talks to
/// `UNUserNotificationCenter` and never asks macOS for authorization: it hands
/// a sanitized title and body to `deliver` (the shared `AppNotificationCenter`
/// Allow-only path, wired by `AppModel`). Without that hook, or without the
/// user's opt-in, nothing is shown and the result says so.
public enum PushNotificationExecutor {
    /// The in-app toast, fired only after a successful system post.
    public static var onNotificationPushed: (@Sendable (String, String) -> Void)?

    /// Set by `AppModel`: (title, body, chat) to the center's decision.
    static var deliver: (@Sendable (String, String, UUID?) async -> AgentPushResult)?

    static let defaultTitle = "TurboSpark Notice"
    static let maximumTitleLength = 80
    static let maximumBodyLength = 240

    /// Single line, no control or invisible formatting characters (newline
    /// and tab become a space, so a bidi override or an escape sequence
    /// cannot reach the banner), collapsed spaces, capped with an ellipsis.
    static func sanitize(_ text: String, limit: Int) -> String {
        var out = String.UnicodeScalarView()
        var lastWasSpace = true
        for scalar in text.unicodeScalars {
            let category = scalar.properties.generalCategory
            if category == .control || category == .format
                || category == .lineSeparator || category == .paragraphSeparator
                || category == .surrogate || category == .privateUse || category == .unassigned
            {
                // Whitespace controls separate words; every other one is dropped.
                if scalar == "\n" || scalar == "\t" || scalar == "\r" || category == .lineSeparator
                    || category == .paragraphSeparator
                {
                    if !lastWasSpace { out.append(" "); lastWasSpace = true }
                }
                continue
            }
            if scalar.properties.isWhitespace {
                if !lastWasSpace { out.append(" "); lastWasSpace = true }
                continue
            }
            out.append(scalar)
            lastWasSpace = false
        }
        var result = String(out).trimmingCharacters(in: .whitespaces)
        if result.count > limit {
            result = String(result.prefix(limit - 1)).trimmingCharacters(in: .whitespaces) + "\u{2026}"
        }
        return result
    }

    public static func execute(arguments: [String: String], chatID: UUID? = nil) async throws -> String {
        guard let rawMessage = arguments["message"] ?? arguments["body"] ?? arguments["text"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 65,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'message' argument for PushNotification."]
            )
        }
        let message = sanitize(rawMessage, limit: maximumBodyLength)
        guard !message.isEmpty else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 65,
                userInfo: [NSLocalizedDescriptionKey: "The 'message' argument for PushNotification is empty."]
            )
        }
        var title = sanitize(arguments["title"] ?? "", limit: maximumTitleLength)
        if title.isEmpty { title = defaultTitle }
        let status = sanitize(arguments["status"] ?? "proactive", limit: 32)

        // Only the sanitized title and message ever leave this function: no
        // other argument, tool output or file content is part of a banner.
        let result = await deliver?(title, message, chatID) ?? .optedOut
        switch result {
        case .posted:
            onNotificationPushed?(title, message)
            return "Notification posted as a system notification: [\(title)] \(message) (Status: \(status))."
        case .optedOut:
            return "Notification not sent: notifications from agents are not enabled in the app (Settings, Notifications, Allow agents to send notifications). The user was not prompted. Put the news in your reply instead."
        case .notAuthorized:
            return "Notification not sent: system notifications are not enabled for TurboSpark (the user allows them in Settings, Notifications, or in System Settings). The user was not prompted. Put the news in your reply instead."
        case .outranked:
            return "Notification not sent: this chat already has a pending approval or question notification. Put the news in your reply instead."
        case .failed:
            return "Notification not sent: the system refused to post it. Put the news in your reply instead."
        case .ghostChat:
            return "Notification not sent: Ghost chats never post system notifications. Put the news in your reply instead."
        }
    }
}

// MARK: - Config Tool Executor

public enum ConfigToolExecutor {
    public static func execute(arguments: [String: String], project: AppProject?) throws -> String {
        let action = (arguments["action"] ?? arguments["method"] ?? "get").lowercased()
        let key = arguments["key"] ?? arguments["name"] ?? ""

        switch action {
        case "get":
            if key.isEmpty {
                return listConfig(project: project)
            }
            return getConfigValue(key: key, project: project)
        case "set":
            // Nothing is writable from here. Reporting success for a no-op
            // would let the model (or user) believe e.g. guardrails were
            // turned off when nothing changed.
            throw NSError(
                domain: "TurboSparkTool",
                code: 67,
                userInfo: [NSLocalizedDescriptionKey:
                    "Config set is not supported and nothing was changed; the user changes settings in Settings."]
            )
        case "list":
            return listConfig(project: project)
        default:
            throw NSError(
                domain: "TurboSparkTool",
                code: 66,
                userInfo: [NSLocalizedDescriptionKey: "Unknown config action: '\(action)'. Valid: get, list."]
            )
        }
    }

    private static func listConfig(project: AppProject?) -> String {
        let settings = MacAppSettingsFileStore.load()
        var lines: [String] = ["### TurboSpark Configuration:"]
        lines.append("- interaction.mode: \(settings.interactionMode)")
        lines.append("- reasoning.effort: \(settings.reasoning)")
        lines.append("- guardrails.mode: \(settings.guardrailsMode)")
        lines.append("- power.profile: \(settings.powerProfile)")
        if let project {
            lines.append("- project.name: \(project.name)")
            lines.append("- project.root: \(project.rootDirectoryPath ?? "(None)")")
            lines.append("- project.permissions.mode: \(project.permissions.mode.label)")
        }
        return lines.joined(separator: "\n")
    }

    private static func getConfigValue(key: String, project: AppProject?) -> String {
        let settings = MacAppSettingsFileStore.load()
        switch key.lowercased() {
        case "interaction.mode", "interaction_mode":
            return "interaction.mode = \(settings.interactionMode)"
        case "reasoning.effort", "reasoning_effort":
            return "reasoning.effort = \(settings.reasoning)"
        case "guardrails.mode", "guardrails_mode":
            return "guardrails.mode = \(settings.guardrailsMode)"
        case "project.root", "project_root":
            return "project.root = \(project?.rootDirectoryPath ?? "(No project attached)")"
        default:
            return "\(key) = (Not explicitly configured or unknown)"
        }
    }
}

// MARK: - CtxInspect Executor

public enum CtxInspectExecutor {
    public static func execute(arguments: [String: String], chatID: UUID?, project: AppProject?) -> String {
        var lines: [String] = ["### Context Inspection:"]
        let key = chatID?.uuidString ?? "active"
        lines.append("- Chat ID: \(key)")
        if let p = project {
            lines.append("- Active Project: \(p.name) (\(p.rootDirectoryPath ?? "None"))")
        } else {
            lines.append("- Active Project: None (Projectless session)")
        }
        let availableTools = AppToolCatalog.tools(for: .coder, projectURL: project?.rootDirectoryURL)
        lines.append("- Registered Tools: \(availableTools.count) available")
        lines.append("- Guardrails & Approvals: Active")
        lines.append("- Memory Limits: Single file read cap 16 MiB, max scan files 5,000")
        return lines.joined(separator: "\n")
    }
}
