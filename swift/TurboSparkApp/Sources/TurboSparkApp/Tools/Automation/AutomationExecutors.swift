import Foundation
import UserNotifications
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

public enum PushNotificationExecutor {
    public static var onNotificationPushed: (@Sendable (String, String) -> Void)?

    public static func execute(arguments: [String: String]) async throws -> String {
        guard let message = arguments["message"] ?? arguments["body"] ?? arguments["text"] else {
            throw NSError(
                domain: "TurboSparkTool",
                code: 60,
                userInfo: [NSLocalizedDescriptionKey: "Missing 'message' argument for PushNotification."]
            )
        }
        let title = arguments["title"] ?? "TurboSpark Notice"
        let status = arguments["status"] ?? "proactive"

        onNotificationPushed?(title, message)

        // Report what actually happened. Without a granted authorization
        // macOS silently drops the request, and the model used to be told the
        // notification was posted when only the in-app toast had shown.
        var delivery = "shown in-app only (system notifications are unavailable outside the app bundle)"
        if Bundle.main.bundleURL.pathExtension == "app",
           Bundle.main.bundleIdentifier != nil {
            let center = UNUserNotificationCenter.current()
            let granted = (try? await center.requestAuthorization(options: [.alert, .sound])) ?? false
            if granted {
                let content = UNMutableNotificationContent()
                content.title = title
                content.body = message
                content.sound = .default
                let request = UNNotificationRequest(
                    identifier: UUID().uuidString, content: content, trigger: nil)
                do {
                    try await center.add(request)
                    delivery = "posted as a system notification"
                } catch {
                    delivery = "shown in-app only (system notification failed: \(error.localizedDescription))"
                }
            } else {
                delivery = "shown in-app only (system notification permission was not granted)"
            }
        }

        return "Notification \(delivery): [\(title)] \(message) (Status: \(status))."
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
                code: 62,
                userInfo: [NSLocalizedDescriptionKey:
                    "Config set is not supported and nothing was changed; the user changes settings in Settings."]
            )
        case "list":
            return listConfig(project: project)
        default:
            throw NSError(
                domain: "TurboSparkTool",
                code: 61,
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
