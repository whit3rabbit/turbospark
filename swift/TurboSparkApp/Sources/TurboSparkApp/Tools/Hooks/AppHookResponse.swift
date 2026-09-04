import Foundation

/// The `hookSpecificOutput` object Claude Code nests per-event fields under.
/// This app only reads the fields it acts on; anything else in that object
/// is ignored rather than modeled.
public struct AppHookSpecificOutput: Sendable {
    public var hookEventName: String?
    public var permissionDecision: String?
    public var permissionDecisionReason: String?
    public var updatedInput: [String: String]?
    public var additionalContext: String?

    public init(
        hookEventName: String? = nil,
        permissionDecision: String? = nil,
        permissionDecisionReason: String? = nil,
        updatedInput: [String: String]? = nil,
        additionalContext: String? = nil
    ) {
        self.hookEventName = hookEventName
        self.permissionDecision = permissionDecision
        self.permissionDecisionReason = permissionDecisionReason
        self.updatedInput = updatedInput
        self.additionalContext = additionalContext
    }
}

/// Parsed JSON response contract, matching Claude Code's hook output shape
/// (https://code.claude.com/docs/en/hooks): top-level `continue`/
/// `stopReason`/`systemMessage`/`decision`/`reason`/`additionalContext`/
/// `updatedInput`, plus a nested `hookSpecificOutput` for per-event fields.
public struct AppHookResponse: Sendable {
    public var continueGeneration: Bool
    public var stopReason: String?
    public var suppressOutput: Bool
    public var systemMessage: String?
    public var decision: String?
    public var reason: String?
    public var additionalContext: String?
    public var updatedInput: [String: String]?
    public var hookSpecificOutput: AppHookSpecificOutput?

    public init(
        continueGeneration: Bool = true,
        stopReason: String? = nil,
        suppressOutput: Bool = false,
        systemMessage: String? = nil,
        decision: String? = nil,
        reason: String? = nil,
        additionalContext: String? = nil,
        updatedInput: [String: String]? = nil,
        hookSpecificOutput: AppHookSpecificOutput? = nil
    ) {
        self.continueGeneration = continueGeneration
        self.stopReason = stopReason
        self.suppressOutput = suppressOutput
        self.systemMessage = systemMessage
        self.decision = decision
        self.reason = reason
        self.additionalContext = additionalContext
        self.updatedInput = updatedInput
        self.hookSpecificOutput = hookSpecificOutput
    }
}

/// One hook's outcome after its raw stdout/stderr/exit-code is interpreted
/// for the event it ran under.
public enum AppHookOutcome: Sendable {
    /// Exit 0 with a JSON object on stdout.
    case structured(AppHookResponse)
    /// Exit 0 with non-JSON stdout: advisory only, never blocking.
    case plainText(String)
    /// The event blocked (see `AppHookResponseParser.blockingExitTwoEvents`),
    /// with the reason to show.
    case blocked(reason: String)
    /// A nonzero, non-blocking exit: shown as feedback where the event
    /// supports it (`PostToolUse`'s exit 2 is exactly this shape -- the
    /// tool already ran, so nothing can be blocked, but Claude still sees
    /// stderr), otherwise logged and ignored.
    case nonBlockingError(String)
    /// The hook produced NO verdict at all: it timed out, its shell could
    /// not be spawned, or its transport failed (state#40).
    ///
    /// **DISTINCT FROM A NIL OUTCOME, WHICH IS WHAT THIS REPLACES.** The
    /// aggregator skips nil and the engine defaults to `.allow`, so all
    /// three failure arms let a call through -- a deny hook whose
    /// interpreter is missing permits every call it was written to block,
    /// silently and on every run. A hook that cannot answer is not consent.
    case unavailable(reason: String)
}

public enum AppHookResponseParser {
    /// Events where Claude Code's exit code 2 actually blocks something
    /// (https://code.claude.com/docs/en/hooks, "Exit Code 2 Blocking
    /// Behavior by Event"). `PostToolUse`/`PostToolUseFailure` are
    /// deliberately absent: the tool has already run by the time either
    /// fires, so exit 2 there is feedback, not a block; `PermissionRequest`
    /// is absent because it resolves via `permissionDecision` JSON instead.
    static let blockingExitTwoEvents: Set<AppHookEvent> = [.preToolUse, .userPromptSubmit, .stop]

    public static func parseHookOutput(stdout: String, stderr: String, exitCode: Int32, event: AppHookEvent) -> AppHookOutcome {
        let trimmedStdout = stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedStderr = stderr.trimmingCharacters(in: .whitespacesAndNewlines)

        if exitCode == 2 {
            let reason = trimmedStderr.isEmpty ? trimmedStdout : trimmedStderr
            let message = reason.isEmpty ? "Blocked by hook (exit code 2)." : reason
            return blockingExitTwoEvents.contains(event) ? .blocked(reason: message) : .nonBlockingError(message)
        }

        if exitCode != 0 {
            let message = trimmedStderr.isEmpty ? trimmedStdout : trimmedStderr
            return .nonBlockingError(message)
        }

        if let data = trimmedStdout.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            return .structured(parseResponseObject(json))
        }
        return .plainText(trimmedStdout)
    }

    private static func parseResponseObject(_ json: [String: Any]) -> AppHookResponse {
        var response = AppHookResponse()
        response.continueGeneration = (json["continue"] as? Bool) ?? true
        response.stopReason = json["stopReason"] as? String
        response.suppressOutput = (json["suppressOutput"] as? Bool) ?? false
        response.systemMessage = json["systemMessage"] as? String
        response.decision = json["decision"] as? String
        response.reason = json["reason"] as? String
        response.additionalContext = json["additionalContext"] as? String
        response.updatedInput = stringDictionary(json["updatedInput"])

        // Legacy top-level `permissionDecision`/`permissionDecisionReason`,
        // kept for hooks written before this app nested it under
        // `hookSpecificOutput` (and for the existing test suite).
        let legacyDecision = json["permissionDecision"] as? String
        let legacyReason = json["permissionDecisionReason"] as? String

        if let hso = json["hookSpecificOutput"] as? [String: Any] {
            response.hookSpecificOutput = AppHookSpecificOutput(
                hookEventName: hso["hookEventName"] as? String,
                permissionDecision: (hso["permissionDecision"] as? String) ?? legacyDecision,
                permissionDecisionReason: (hso["permissionDecisionReason"] as? String) ?? legacyReason,
                updatedInput: stringDictionary(hso["updatedInput"]),
                additionalContext: hso["additionalContext"] as? String
            )
        } else if legacyDecision != nil {
            response.hookSpecificOutput = AppHookSpecificOutput(
                permissionDecision: legacyDecision,
                permissionDecisionReason: legacyReason
            )
        }

        return response
    }

    private static func stringDictionary(_ value: Any?) -> [String: String]? {
        guard let dict = value as? [String: Any] else { return nil }
        return dict.reduce(into: [String: String]()) { acc, kv in
            switch kv.value {
            case let s as String: acc[kv.key] = s
            case let n as NSNumber: acc[kv.key] = n.stringValue
            default: acc[kv.key] = "\(kv.value)"
            }
        }
    }
}
