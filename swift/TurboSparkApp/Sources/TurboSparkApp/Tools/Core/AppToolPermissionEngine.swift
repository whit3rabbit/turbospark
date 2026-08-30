import Foundation

/// Decision outcome for a tool call after evaluating policies, risk, and session approvals.
public enum ToolPermissionDecision: Equatable, Sendable {
    /// Action is permitted to execute automatically.
    case allow
    /// Action requires explicit user confirmation before executing.
    case ask(assessment: ToolRiskAssessment, reason: String)
    /// Action is strictly prohibited by policy and cannot be executed.
    case deny(reason: String)
}

/// Thread-safe in-memory session cache for "Always Allow" decisions made during a chat session.
public actor SessionApprovalStore {
    public static let shared = SessionApprovalStore()

    /// Map from sessionID -> Set of approved tool names.
    private var approvedTools: [String: Set<String>] = [:]
    /// Map from sessionID -> Set of approved command prefixes.
    private var approvedCommandPrefixes: [String: Set<String>] = [:]

    public init() {}

    /// Approves all future invocations of a specific tool name within the session.
    public func allowTool(sessionID: String, toolName: String) {
        let name = toolName.lowercased()
        var set = approvedTools[sessionID] ?? []
        set.insert(name)
        approvedTools[sessionID] = set
    }

    /// Approves future executions of commands starting with the given prefix.
    public func allowCommandPrefix(sessionID: String, prefix: String) {
        let clean = prefix.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !clean.isEmpty else { return }
        var set = approvedCommandPrefixes[sessionID] ?? []
        set.insert(clean)
        approvedCommandPrefixes[sessionID] = set
    }

    /// Checks if a tool call has been pre-approved for this session.
    public func isApproved(sessionID: String, toolName: String, command: String? = nil) -> Bool {
        let name = toolName.lowercased()
        if let set = approvedTools[sessionID], set.contains(name) {
            return true
        }

        if let cmd = command?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(),
           let prefixes = approvedCommandPrefixes[sessionID] {
            for p in prefixes {
                if cmd == p || cmd.hasPrefix(p + " ") {
                    return true
                }
            }
        }

        return false
    }

    /// Clears session approvals when a chat is deleted or reset.
    public func clear(sessionID: String) {
        approvedTools.removeValue(forKey: sessionID)
        approvedCommandPrefixes.removeValue(forKey: sessionID)
    }

    /// Clears all session approvals.
    public func reset() {
        approvedTools.removeAll()
        approvedCommandPrefixes.removeAll()
    }
}

/// Central permissions evaluation engine enforcing Unsloth-style security policies.
public enum AppToolPermissionEngine {
    /// Evaluates whether a tool call should be allowed, gated with user confirmation, or denied.
    ///
    /// Session approval, category-deny, and the high-risk gate are ordered
    /// deliberately: deny and high-risk are both checked BEFORE a session's
    /// "always allow" grant is consulted, so a prior approval of one call
    /// (e.g. `run_command git status`) can never be read as covering a later,
    /// unrelated call under the same tool name that turns out to be high-risk
    /// (e.g. `run_command rm -rf ~/Documents`). Do not move the session-approval
    /// check back above these two without re-adding a per-approval risk ceiling.
    public static func evaluate(
        call: AppToolCall,
        project: AppProject?,
        sessionApproved: Bool = false
    ) -> ToolPermissionDecision {
        // No project selected: fall back to the same guarded default a fresh
        // project would get, never to the wide-open `.auto` permission set
        // (fileWrite/terminal/mcp all `.allow`). A plain chat with no project
        // must still ask before running shell commands or writing files.
        let permissions = project?.permissions ?? .standard
        let category = call.category
        let risk = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)

        // 1. Strict Read-Only Mode. Absolute: session approval never applies here.
        if permissions.mode == .readOnly {
            if category == .fileRead && !risk.isHighRisk {
                return .allow
            }
            return .deny(reason: "Tool execution is denied in Strict Read-Only mode.")
        }

        // 2. Permissive Mode: Allow everything bounded by the filesystem sandbox
        if permissions.mode == .permissive {
            return .allow
        }

        // 3. Granular Category Permission Check (Explicit Deny wins, and a
        // session approval cannot resurrect a category the project denies)
        let categoryPermission: AppToolPermission
        switch category {
        case .fileRead: categoryPermission = permissions.fileRead
        case .fileWrite: categoryPermission = permissions.fileWrite
        case .terminal: categoryPermission = permissions.terminal
        case .web: categoryPermission = permissions.web
        case .mcp: categoryPermission = permissions.mcp
        case .automation: categoryPermission = permissions.automation
        }

        if categoryPermission == .deny {
            return .deny(reason: "The \(category.label) category is set to Deny in project settings.")
        }

        // 4. High-risk actions ALWAYS require a fresh confirmation, regardless
        // of any "always allow this session" grant recorded under this tool
        // name or command prefix. This is the fix for the bypass above: risk
        // is assessed on THIS call's actual arguments, not on whatever call
        // originally earned the session grant.
        if risk.isHighRisk {
            let reason = risk.reasons.isEmpty ? "High-risk action requires confirmation." : risk.reasons.joined(separator: "; ")
            return .ask(assessment: risk, reason: reason)
        }

        // 5. Session-level pre-approval bypasses prompts for repeats of a
        // call already vetted this session, now that deny and high-risk have
        // both had the first word.
        if sessionApproved {
            return .allow
        }

        // 6. MCP Server-level Auto-Approval Check
        if category == .mcp {
            let serverName: String?
            if call.name.hasPrefix("mcp__") {
                let parts = call.name.components(separatedBy: "__")
                serverName = parts.count >= 2 ? parts[1] : nil
            } else {
                serverName = call.arguments["server"] ?? call.arguments["server_name"]
            }

            if let sName = serverName {
                let allServers = (project?.mcpServers ?? []) + GlobalMcpFileStore.load().servers
                if let server = allServers.first(where: { $0.name.lowercased() == sName.lowercased() }),
                   server.autoApprove && !risk.isHighRisk {
                    return .allow
                }
            }
        }

        // 7. Always Ask Mode: Prompts on any mutating or external action
        if permissions.mode == .ask || categoryPermission == .ask {
            if category == .fileRead && risk.level == .safe {
                return .allow
            }
            let reason = risk.reasons.isEmpty ? "Manual confirmation required for \(call.name)." : risk.reasons.joined(separator: "; ")
            return .ask(assessment: risk, reason: reason)
        }

        // 8. Auto Mode ("Approve for me" - Unsloth Studio default): runs
        // safe and low-risk operations silently. High-risk was already
        // handled in step 4, above every other check in this function.
        if permissions.mode == .auto {
            return .allow
        }

        return .allow
    }
}
