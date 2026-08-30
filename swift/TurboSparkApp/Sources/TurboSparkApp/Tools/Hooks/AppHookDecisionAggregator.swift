import Foundation

/// One event's combined verdict across every hook that ran for it.
public struct AppHookVerdict: Sendable {
    public var permissionDecision: AppHookPermissionBehavior?
    public var permissionReason: String?
    public var isBlocked: Bool
    public var blockReason: String?
    /// Non-blocking feedback (a `PostToolUse`/`PostToolUseFailure` exit-2,
    /// or any nonzero exit on an event that cannot block at all) -- shown
    /// to the model where the caller has somewhere to put it, but never
    /// treated as `isBlocked`.
    public var feedbackMessage: String?
    public var updatedInput: [String: String]?
    public var additionalContext: String?
    public var systemMessages: [String]

    public init(
        permissionDecision: AppHookPermissionBehavior? = nil,
        permissionReason: String? = nil,
        isBlocked: Bool = false,
        blockReason: String? = nil,
        feedbackMessage: String? = nil,
        updatedInput: [String: String]? = nil,
        additionalContext: String? = nil,
        systemMessages: [String] = []
    ) {
        self.permissionDecision = permissionDecision
        self.permissionReason = permissionReason
        self.isBlocked = isBlocked
        self.blockReason = blockReason
        self.feedbackMessage = feedbackMessage
        self.updatedInput = updatedInput
        self.additionalContext = additionalContext
        self.systemMessages = systemMessages
    }

    public static let passthrough = AppHookVerdict()
}

public enum AppHookDecisionAggregator {
    /// Folds one event's hook results into a single verdict: deny beats ask
    /// beats allow for `PreToolUse`/`PermissionRequest` (every matching hook
    /// runs; the worst decision wins), and the FIRST block wins for
    /// `Stop`/`UserPromptSubmit`/`PostToolUse`/`PostToolUseFailure` --
    /// later hooks still run and their `additionalContext`/`systemMessage`
    /// still aggregate.
    public static func aggregate(_ results: [AppHookExecutionResult], event: AppHookEvent) -> AppHookVerdict {
        var verdict = AppHookVerdict.passthrough
        var contexts: [String] = []
        var messages: [String] = []
        var feedback: [String] = []
        var updatedInput: [String: String] = [:]
        let isPermissionEvent = event == .preToolUse || event == .permissionRequest

        for result in results {
            guard let outcome = result.outcome else { continue }

            switch outcome {
            case .blocked(let reason):
                if isPermissionEvent {
                    if verdict.permissionDecision != .deny {
                        verdict.permissionDecision = .deny
                        verdict.permissionReason = reason
                    }
                } else if !verdict.isBlocked {
                    verdict.isBlocked = true
                    verdict.blockReason = reason
                }

            case .nonBlockingError(let message):
                if !message.isEmpty { feedback.append(message) }

            case .plainText:
                continue

            case .structured(let response):
                if let systemMessage = response.systemMessage, !systemMessage.isEmpty {
                    messages.append(systemMessage)
                }
                if let ctx = response.additionalContext, !ctx.isEmpty { contexts.append(ctx) }
                if let ctx = response.hookSpecificOutput?.additionalContext, !ctx.isEmpty { contexts.append(ctx) }
                for (k, v) in response.updatedInput ?? [:] { updatedInput[k] = v }
                for (k, v) in response.hookSpecificOutput?.updatedInput ?? [:] { updatedInput[k] = v }

                if let permission = response.hookSpecificOutput?.permissionDecision {
                    let behavior: AppHookPermissionBehavior?
                    switch permission.lowercased() {
                    case "deny", "block": behavior = .deny
                    case "ask": behavior = .ask
                    case "allow": behavior = .allow
                    default: behavior = nil
                    }
                    if let behavior, rank(behavior) > rank(verdict.permissionDecision) {
                        verdict.permissionDecision = behavior
                        verdict.permissionReason = response.hookSpecificOutput?.permissionDecisionReason ?? response.reason
                    }
                }

                if response.decision == "block", !isPermissionEvent, !verdict.isBlocked {
                    verdict.isBlocked = true
                    verdict.blockReason = response.reason ?? "Blocked by hook."
                }
            }
        }

        verdict.additionalContext = contexts.isEmpty ? nil : contexts.joined(separator: "\n\n")
        verdict.updatedInput = updatedInput.isEmpty ? nil : updatedInput
        verdict.systemMessages = messages
        verdict.feedbackMessage = feedback.isEmpty ? nil : feedback.joined(separator: "\n\n")
        return verdict
    }

    private static func rank(_ behavior: AppHookPermissionBehavior?) -> Int {
        switch behavior {
        case .deny: return 3
        case .ask: return 2
        case .allow: return 1
        case .passthrough, .none: return 0
        }
    }
}
