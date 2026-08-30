import Foundation
import CryptoKit

// MARK: - Lifecycle Hook Events

/// Standard lifecycle events that hooks can attach to.
public enum AppHookEvent: String, Codable, CaseIterable, Identifiable, Sendable {
    case preToolUse = "PreToolUse"
    case postToolUse = "PostToolUse"
    case postToolUseFailure = "PostToolUseFailure"
    case sessionStart = "SessionStart"
    case sessionEnd = "SessionEnd"
    case userPromptSubmit = "UserPromptSubmit"
    case stop = "Stop"
    case permissionRequest = "PermissionRequest"
    case notification = "Notification"

    public var id: String { rawValue }

    public var displayName: String { rawValue }

    public var eventDescription: String {
        switch self {
        case .preToolUse:
            return "Before a tool executes"
        case .postToolUse:
            return "After a tool executes"
        case .postToolUseFailure:
            return "When a tool execution encounters an error"
        case .sessionStart:
            return "When a new session starts"
        case .sessionEnd:
            return "When a session ends or is cleared"
        case .userPromptSubmit:
            return "When a user submits a prompt message"
        case .stop:
            return "When generation turn finishes"
        case .permissionRequest:
            return "When a permission prompt is triggered"
        case .notification:
            return "When an agent notification is dispatched"
        }
    }

    public var systemImage: String {
        switch self {
        case .preToolUse:
            return "arrow.triangle.pull"
        case .postToolUse:
            return "checkmark.seal"
        case .postToolUseFailure:
            return "exclamationmark.triangle"
        case .sessionStart:
            return "play.circle"
        case .sessionEnd:
            return "stop.circle"
        case .userPromptSubmit:
            return "paperplane"
        case .stop:
            return "flag.checkered"
        case .permissionRequest:
            return "lock.shield"
        case .notification:
            return "bell"
        }
    }
}

// MARK: - Hook Execution Type & Shell

public enum AppHookType: String, Codable, CaseIterable, Identifiable, Sendable {
    case command = "command"
    case http = "http"
    case prompt = "prompt"

    public var id: String { rawValue }

    public var title: String {
        switch self {
        case .command: return "Shell Command"
        case .http: return "HTTP Webhook"
        case .prompt: return "LLM Evaluator"
        }
    }
}

public enum AppHookShell: String, Codable, CaseIterable, Identifiable, Sendable {
    case bash = "bash"
    case zsh = "zsh"
    case sh = "sh"
    case pwsh = "powershell"

    public var id: String { rawValue }
}

// MARK: - Hook Source

public enum AppHookSourceType: String, Codable, CaseIterable, Identifiable, Sendable {
    case userConfig = "userConfig"
    case projectConfig = "projectConfig"
    case localConfig = "localConfig"
    case plugin = "plugin"
    case custom = "custom"

    public var id: String { rawValue }

    public var sectionTitle: String {
        switch self {
        case .userConfig: return "From Config"
        case .projectConfig: return "From Project"
        case .localConfig: return "Local Config"
        case .plugin: return "Plugins"
        case .custom: return "Custom Hooks"
        }
    }

    public var displayTitle: String {
        switch self {
        case .userConfig: return "User config"
        case .projectConfig: return "Project config"
        case .localConfig: return "Local settings"
        case .plugin: return "Plugin"
        case .custom: return "Custom"
        }
    }

    public var subtitle: String {
        switch self {
        case .userConfig: return "All projects (~/.turbospark or ~/.claude)"
        case .projectConfig: return "Current project (.turbospark or .claude)"
        case .localConfig: return "Local overrides (.settings.local.json)"
        case .plugin: return "Installed and enabled plugins"
        case .custom: return "App-managed lifecycle hooks"
        }
    }
}

// MARK: - Hook Command Model

public struct AppHookCommand: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID
    public var name: String
    public var event: AppHookEvent
    public var type: AppHookType
    public var command: String
    public var ifCondition: String?
    public var matcher: String?
    public var shell: AppHookShell
    public var timeoutSeconds: Double
    public var statusMessage: String?
    public var isAsync: Bool
    public var isEnabled: Bool
    public var sourceType: AppHookSourceType
    public var sourcePath: String?
    public var pluginName: String?
    public var options: [String: String]

    public init(
        id: UUID = UUID(),
        name: String,
        event: AppHookEvent,
        type: AppHookType = .command,
        command: String,
        ifCondition: String? = nil,
        matcher: String? = nil,
        shell: AppHookShell = .zsh,
        // Hand-authored/editor default: fail fast rather than let a human
        // typo in a custom hook silently block the UI for minutes. Hooks
        // DISCOVERED from a Claude Code config file get 600.0 when their
        // "timeout" key is absent instead, for Claude Code parity -- see
        // AppHookStore+Discovery.swift's parseHookObject and
        // AppHookCommandRunner.swift's runtime fallback (both commented
        // "Claude Code parity"). The two defaults serve different trust
        // contexts on purpose; do not unify them.
        timeoutSeconds: Double = 30.0,
        statusMessage: String? = nil,
        isAsync: Bool = false,
        isEnabled: Bool = true,
        sourceType: AppHookSourceType = .userConfig,
        sourcePath: String? = nil,
        pluginName: String? = nil,
        options: [String: String] = [:]
    ) {
        self.id = id
        self.name = name
        self.event = event
        self.type = type
        self.command = command
        self.ifCondition = ifCondition
        self.matcher = matcher
        self.shell = shell
        self.timeoutSeconds = timeoutSeconds
        self.statusMessage = statusMessage
        self.isAsync = isAsync
        self.isEnabled = isEnabled
        self.sourceType = sourceType
        self.sourcePath = sourcePath
        self.pluginName = pluginName
        self.options = options
    }

    /// Computes a cryptographic SHA-256 hash representing the immutable identity and executable content of this hook.
    ///
    /// Widened to include `timeoutSeconds` and `isAsync`: both change what
    /// the hook actually does (how long it may block, whether its decision
    /// can gate a call at all), so a change to either must re-trigger the
    /// trust prompt rather than silently keep the old approval.
    public var contentHash: String {
        let content = "\(event.rawValue):\(type.rawValue):\(command):\(ifCondition ?? ""):\(matcher ?? ""):\(shell.rawValue):\(timeoutSeconds):\(isAsync)"
        let digest = SHA256.hash(data: Data(content.utf8))
        return digest.compactMap { String(format: "%02x", $0) }.joined()
    }
}

// MARK: - UserConfig / Options Schema

public enum AppHookOptionType: String, Codable, CaseIterable, Identifiable, Sendable {
    case string = "string"
    case boolean = "boolean"
    case number = "number"
    case directory = "directory"
    case file = "file"

    public var id: String { rawValue }
}

public struct AppHookOptionSpec: Identifiable, Codable, Sendable, Equatable {
    public var id: String { key }
    public var key: String
    public var type: AppHookOptionType
    public var title: String
    public var description: String
    public var defaultValue: String?
    public var isRequired: Bool
    public var isSensitive: Bool

    public init(
        key: String,
        type: AppHookOptionType = .string,
        title: String,
        description: String,
        defaultValue: String? = nil,
        isRequired: Bool = false,
        isSensitive: Bool = false
    ) {
        self.key = key
        self.type = type
        self.title = title
        self.description = description
        self.defaultValue = defaultValue
        self.isRequired = isRequired
        self.isSensitive = isSensitive
    }
}

// MARK: - Hook Group by Source

public struct AppHookSourceGroup: Identifiable, Sendable, Equatable {
    public var id: String
    public var title: String
    public var subtitle: String
    public var sourceType: AppHookSourceType
    public var pluginName: String?
    public var hooks: [AppHookCommand]
    public var optionSpecs: [AppHookOptionSpec]
    public var unreviewedCount: Int

    public init(
        id: String,
        title: String,
        subtitle: String,
        sourceType: AppHookSourceType,
        pluginName: String? = nil,
        hooks: [AppHookCommand] = [],
        optionSpecs: [AppHookOptionSpec] = [],
        unreviewedCount: Int = 0
    ) {
        self.id = id
        self.title = title
        self.subtitle = subtitle
        self.sourceType = sourceType
        self.pluginName = pluginName
        self.hooks = hooks
        self.optionSpecs = optionSpecs
        self.unreviewedCount = unreviewedCount
    }
}

// MARK: - PreToolUse Decision & Execution Results

public enum AppHookPermissionBehavior: String, Codable, Sendable {
    case allow = "allow"
    case deny = "deny"
    case ask = "ask"
    case passthrough = "passthrough"
}

public struct AppHookPreToolUseDecision: Sendable {
    public var behavior: AppHookPermissionBehavior
    public var reason: String?
    public var blockedByHookName: String?
    /// `PreToolUse`'s `updatedInput`: replaces the corresponding keys in the
    /// tool call's arguments before it runs.
    public var updatedInput: [String: String]?
    /// Extra context a hook attached, folded beside the tool result rather
    /// than into the permission reason.
    public var additionalContext: String?

    public init(
        behavior: AppHookPermissionBehavior = .passthrough,
        reason: String? = nil,
        blockedByHookName: String? = nil,
        updatedInput: [String: String]? = nil,
        additionalContext: String? = nil
    ) {
        self.behavior = behavior
        self.reason = reason
        self.blockedByHookName = blockedByHookName
        self.updatedInput = updatedInput
        self.additionalContext = additionalContext
    }
}

public struct AppHookExecutionResult: Sendable {
    public var hookID: UUID
    public var hookName: String
    public var event: AppHookEvent
    public var exitCode: Int32
    public var stdout: String
    public var stderr: String
    public var durationSeconds: Double
    /// The interpreted outcome of this one hook's run, per
    /// `AppHookResponseParser`. `AppHookDecisionAggregator` is what folds
    /// several hooks' outcomes for the same event into one verdict.
    public var outcome: AppHookOutcome?

    public var isSuccess: Bool { exitCode == 0 }

    public init(
        hookID: UUID,
        hookName: String,
        event: AppHookEvent,
        exitCode: Int32,
        stdout: String,
        stderr: String,
        durationSeconds: Double,
        outcome: AppHookOutcome? = nil
    ) {
        self.hookID = hookID
        self.hookName = hookName
        self.event = event
        self.exitCode = exitCode
        self.stdout = stdout
        self.stderr = stderr
        self.durationSeconds = durationSeconds
        self.outcome = outcome
    }
}
