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
    ///
    /// **AND `permissive` IS UNDER THEM TOO, WHICH IT WAS NOT** (state#46).
    /// That paragraph described every mode except the one that skipped all
    /// three gates in a single line at the top. `readOnly` is the only
    /// absolute arm here, and it is absolute in the SAFE direction.
    /// - Parameter globalServers: the app-level MCP configurations, passed in
    ///   rather than re-read from disk (state#61). Every caller has them in
    ///   memory already, and this function runs on the main actor.
    public static func evaluate(
        call: AppToolCall,
        project: AppProject?,
        sessionApproved: Bool = false,
        fallbackMode: AppPermissionMode? = nil,
        globalServers: [McpServerConfig] = GlobalMcpFileStore.load().servers
    ) -> ToolPermissionDecision {
        // No project selected: fall back to the guarded default or chosen fallback mode.
        let permissions = project?.permissions ?? AppProjectPermissions.preset(for: fallbackMode ?? .auto)
        let category = call.category
        let risk = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)

        // 1. Strict Read-Only Mode. Absolute: session approval never applies here.
        if permissions.mode == .readOnly {
            if category == .fileRead && !risk.isHighRisk {
                return .allow
            }
            return .deny(reason: "Tool execution is denied in Strict Read-Only mode.")
        }

        // Full Access Mode: Unrestricted execution without approval prompts.
        if permissions.mode == .fullAccess {
            return .allow
        }

        // 2. Granular Category Permission Check (Explicit Deny wins, and a
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

        // 3. High-risk actions ALWAYS require a fresh confirmation, regardless
        // of any "always allow this session" grant recorded under this tool
        // name or command prefix. This is the fix for the bypass above: risk
        // is assessed on THIS call's actual arguments, not on whatever call
        // originally earned the session grant.
        if risk.isHighRisk {
            let reason = risk.reasons.isEmpty ? "High-risk action requires confirmation." : risk.reasons.joined(separator: "; ")
            return .ask(assessment: risk, reason: reason)
        }

        // 4. Permissive Mode: no prompt for anything that got this far
        // (state#46).
        //
        // **IT USED TO SIT AT STEP 2 AND SHORT-CIRCUIT ALL THREE GATES
        // ABOVE**, which made the ordering paragraph on this function false
        // for one whole mode: deny and high-risk were documented as having
        // the first word, and a permissive project ran `rm -rf ~` unprompted.
        // `SubagentRunner` had noticed and compensated with its own positive
        // terminal gate; the MAIN loop had not, so the mode a user picks to
        // avoid being asked was also the mode that stopped asking about the
        // one class of call the whole engine exists for.
        //
        // Permissive now means "never ask, except for what you explicitly
        // denied and except for high risk", which is what the ordering
        // comment already claimed and what the `.readOnly` arm above models:
        // that one IS absolute, and in the safe direction.
        if permissions.mode == .permissive {
            return .allow
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
                // **GLOBAL FIRST, MATCHING THE EXECUTOR** (state#61). The two
                // resolved a name collision in OPPOSITE orders, so the
                // `autoApprove` flag consulted here could belong to a
                // different server from the one `executeMcpCall` then dialled
                // -- a project `.mcp.json` inheriting a global server's
                // "approved" bit for a command nobody approved. Read off
                // `globalMcpServers` where it is already in memory; this runs
                // on the main actor and `GlobalMcpFileStore.load()` is a disk
                // read per evaluation.
                let allServers = globalServers + (project?.mcpServers ?? [])
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
