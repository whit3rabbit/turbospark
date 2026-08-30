import Foundation

/// Builds the JSON stdin payload for a command hook (and the HTTP hook body),
/// matching Claude Code's field names (`hook_event_name`, `session_id`,
/// `tool_input`, ...) so a hook script written for Claude Code reads a
/// familiar shape here too. See https://code.claude.com/docs/en/hooks.
enum AppHookStdinPayload {
    /// This app has no per-session transcript file (root swift/CLAUDE.md
    /// Gotcha 13: chats live in one shared archive), so `transcript_path`
    /// points at that archive rather than a session-specific log.
    static var transcriptPath: String {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        return appSupport.appendingPathComponent("TurboSpark/chats_archive.json").path
    }

    static func build(
        event: AppHookEvent,
        sessionID: String,
        transcriptPath: String,
        cwd: String,
        toolName: String? = nil,
        toolArguments: [String: String]? = nil,
        toolOutput: String? = nil,
        toolDurationSeconds: Double? = nil,
        isError: Bool? = nil,
        prompt: String? = nil,
        source: String? = nil,
        reason: String? = nil,
        stopHookActive: Bool? = nil
    ) -> [String: Any] {
        var payload: [String: Any] = [
            "session_id": sessionID,
            "transcript_path": transcriptPath,
            "cwd": cwd,
            "hook_event_name": event.rawValue,
            "timestamp": ISO8601DateFormatter().string(from: Date())
        ]

        switch event {
        case .preToolUse, .permissionRequest:
            payload["tool_name"] = toolName ?? ""
            payload["tool_input"] = toolArguments ?? [:]

        case .postToolUse, .postToolUseFailure:
            payload["tool_name"] = toolName ?? ""
            payload["tool_input"] = toolArguments ?? [:]
            payload["tool_response"] = [
                "output": toolOutput ?? "",
                "success": !(isError ?? false)
            ]
            payload["duration_ms"] = Int((toolDurationSeconds ?? 0) * 1000)

        case .userPromptSubmit:
            payload["prompt"] = prompt ?? ""

        case .sessionStart:
            payload["source"] = source ?? "startup"

        case .sessionEnd:
            payload["reason"] = reason ?? "other"

        case .stop:
            payload["stop_hook_active"] = stopHookActive ?? false

        case .notification:
            payload["message"] = reason ?? ""
        }

        return payload
    }
}
