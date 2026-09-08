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
    ///
    /// The MCP layers slot into that spine as follows. A persisted DENY rule
    /// sits immediately after the category deny (an explicitly denied tool
    /// must not be resurrected by a session approval, a allow rule, or an
    /// `autoApprove` flag). A persisted ALLOW rule sits AFTER the high-risk
    /// gate -- "always allow this tool" buys back the ask prompt, never the
    /// risk ceiling -- and BEFORE session approvals. And a server imported
    /// from a repository config asks once in `auto` mode unless the user
    /// explicitly opted it into `autoApprove` in-app, so a cloned
    /// `.mcp.json` cannot grant silent execution by being imported.
    ///
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

        // The server/tool pair this call addresses, in whichever spelling
        // the model used. Shared with the approval card and rule writing so
        // a rule is always matched with the parse that wrote it.
        let mcpTarget: (server: String, tool: String?)? = {
            guard category == .mcp else { return nil }
            return McpPermissionRule.targetOfCall(name: call.name, arguments: call.arguments)
        }()

        let risk: ToolRiskAssessment = {
            let base = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)
            // Server-declared annotations raise a safe verdict on a
            // destructive-marked tool; they never lower a heuristic one.
            guard category == .mcp, let target = mcpTarget, let tool = target.tool,
                  let annotations = McpToolCatalogCache.shared
                      .tools(forServerName: target.server)?
                      .first(where: { $0.name == tool })?.annotations else {
                return base
            }
            return ToolRiskClassifier.adjusting(base, annotations: annotations)
        }()

        // Whether THIS call's server has never completed discovery, so its
        // own destructive/read-only annotations are genuinely UNKNOWN rather
        // than merely absent. `McpToolCatalogCache.tools(forServerName:)`
        // returns nil ONLY in that case (never for "discovered, and this
        // tool has no annotations"), and `refreshEnabled` kicks discovery off
        // as a fire-and-forget background Task -- so a server just approved
        // or just enabled can have its first tool call reach here before
        // that Task has ever run. This is consulted ONLY at the two steps
        // below that would otherwise auto-allow with NO other gate in
        // between (permissive mode, and auto mode's own fallback): an
        // explicit deny rule, an explicit allow rule, a session approval and
        // ask-mode all already make their decision from the name heuristic
        // alone and are unaffected, because each of those is a distinct,
        // already-vetted signal rather than "nothing said no".
        let mcpAnnotationsUnknown: Bool = {
            guard category == .mcp, let target = mcpTarget else { return false }
            return McpToolCatalogCache.shared.tools(forServerName: target.server) == nil
        }()
        func askBecauseAnnotationsUnknown(defaultReason: String) -> ToolPermissionDecision {
            .ask(
                assessment: risk,
                reason: "MCP server '\(mcpTarget?.server ?? "")' has not finished advertising its tools "
                    + "yet, so \(defaultReason)")
        }

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

        // 2b. Persisted MCP deny rule: refused outright, ahead of every
        // gate below including session approvals and high-risk ask. The
        // same rules strip the tool from the advertised list
        // (`AppToolCatalogMcp`), so a call reaching here means the model
        // emitted a name it was never shown.
        if let target = mcpTarget, permissions.mcpDenyMatches(serverName: target.server, toolName: target.tool) {
            let toolPart = target.tool.map { "__\($0)" } ?? " (all tools)"
            return .deny(reason: "MCP tool 'mcp__\(target.server)\(toolPart)' is denied by a project permission rule.")
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
        //
        // EXCEPT when this is an MCP call whose server has not finished
        // discovery: this is the one auto-allow in the whole function with
        // NOTHING else standing between the call and execution, so it is
        // exactly where an undiscovered destructive tool would otherwise run
        // unprompted (see `mcpAnnotationsUnknown`'s own doc above).
        if permissions.mode == .permissive {
            if mcpAnnotationsUnknown {
                return askBecauseAnnotationsUnknown(
                    defaultReason: "this tool's own destructive/read-only annotations are not yet known.")
            }
            return .allow
        }

        // 5. Persisted MCP allow rule: the user granted this server or tool
        // across sessions, so it skips the ask prompt. It sits BELOW the
        // high-risk gate on purpose -- a grant is a statement about asking,
        // not about risk, and this call's own arguments were still assessed
        // first.
        if let target = mcpTarget, permissions.mcpAllowMatches(serverName: target.server, toolName: target.tool) {
            return .allow
        }

        // 6. Session-level pre-approval bypasses prompts for repeats of a
        // call already vetted this session, now that deny and high-risk have
        // both had the first word.
        if sessionApproved {
            return .allow
        }

        // 7. MCP Server-level Auto-Approval Check
        if let target = mcpTarget {
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
            let matched = allServers.first(where: { $0.name.lowercased() == target.server.lowercased() })
            if let server = matched, server.autoApprove && !risk.isHighRisk {
                return .allow
            }

            // 7b. A server imported from a repository config file asks in
            // auto and agent modes unless auto-approval was explicitly opted
            // into above. `sourcePath` records that origin for imports from
            // every supported format, and without this arm the auto mode's
            // trailing default allowed a cloned `.mcp.json`'s servers to
            // run non-high-risk calls silently the moment they were
            // imported -- the one gap the project approval lifecycle
            // (`AppModel+Mcp`) cannot close on its own for archives saved
            // before it existed.
            //
            // **AGENT MODE IS UNDER IT TOO, AND MUST STAY THERE** (see
            // `swift/docs/SWIFT_AGENT_MODE.md`). Without the `agentAuto`
            // arm, the ask would fall through to the classifier, and a
            // cloned config's tool would be judged by a model instead of
            // by the user -- exactly the silent-execution path this gate
            // exists to close. The ask is `hardGated` so the agent-mode
            // router parks it rather than classifying it.
            if (permissions.mode == .auto || permissions.mode == .agentAuto),
               let server = matched, !server.autoApprove,
               let source = server.sourcePath, !source.isEmpty {
                var importAsk = risk
                importAsk.hardGated = true
                return .ask(
                    assessment: importAsk,
                    reason: "MCP server '\(server.name)' was imported from a repository config (\(source)) and has not been marked auto-approved.")
            }
        }

        // 8. Always Ask Mode: Prompts on any mutating or external action
        if permissions.mode == .ask || categoryPermission == .ask {
            if category == .fileRead && risk.level == .safe {
                return .allow
            }
            let reason = risk.reasons.isEmpty ? "Manual confirmation required for \(call.name)." : risk.reasons.joined(separator: "; ")
            return .ask(assessment: risk, reason: reason)
        }

        // 9. Auto Mode ("Approve for me" - Unsloth Studio default): runs
        // safe and low-risk operations silently. High-risk was already
        // handled in step 4, above every other check in this function.
        //
        // Same MCP-discovery exception as step 4's permissive arm: this is
        // auto mode's own trailing default, reached only when no deny rule,
        // allow rule, session approval or repo-import gate (7b) has already
        // decided -- exactly the case an undiscovered destructive tool would
        // otherwise fall through to.
        if permissions.mode == .auto {
            if mcpAnnotationsUnknown {
                return askBecauseAnnotationsUnknown(
                    defaultReason: "this tool's own destructive/read-only annotations are not yet known.")
            }
            return .allow
        }

        return .allow
    }
}
