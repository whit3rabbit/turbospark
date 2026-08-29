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
    public static func evaluate(
        call: AppToolCall,
        project: AppProject?,
        sessionApproved: Bool = false
    ) -> ToolPermissionDecision {
        let permissions = project?.permissions ?? .auto
        let category = call.category
        let risk = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)

        // 1. Session-level pre-approval bypasses prompts (unless strict read-only)
        if sessionApproved && permissions.mode != .readOnly {
            return .allow
        }

        // 2. Strict Read-Only Mode
        if permissions.mode == .readOnly {
            if category == .fileRead && !risk.isHighRisk {
                return .allow
            }
            return .deny(reason: "Tool execution is denied in Strict Read-Only mode.")
        }

        // 3. Permissive Mode: Allow everything bounded by the filesystem sandbox
        if permissions.mode == .permissive {
            return .allow
        }

        // 4. Granular Category Permission Check (Explicit Deny wins)
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

        // 5. MCP Server-level Auto-Approval Check
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

        // 6. Always Ask Mode: Prompts on any mutating or external action
        if permissions.mode == .ask || categoryPermission == .ask {
            if category == .fileRead && risk.level == .safe {
                return .allow
            }
            let reason = risk.reasons.isEmpty ? "Manual confirmation required for \(call.name)." : risk.reasons.joined(separator: "; ")
            return .ask(assessment: risk, reason: reason)
        }

        // 7. Auto Mode ("Approve for me" - Unsloth Studio default):
        // Automatically runs safe and low-risk operations; pauses for approval on high-risk operations.
        if permissions.mode == .auto {
            if risk.isHighRisk {
                let reason = risk.reasons.isEmpty ? "High-risk action requires confirmation." : risk.reasons.joined(separator: "; ")
                return .ask(assessment: risk, reason: reason)
            }
            return .allow
        }

        return .allow
    }
}
