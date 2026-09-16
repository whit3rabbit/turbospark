import Foundation

extension AppToolRegistry {
    struct DeferredMcpContinuationStop: LocalizedError {
        let output: String
        let isError: Bool
        let reason: String

        var errorDescription: String? { output }
    }

    static func deferredMcpPreHookStop(
        _ decision: AppHookPreToolUseDecision
    ) -> DeferredMcpContinuationStop? {
        guard decision.preventContinuation else { return nil }
        let reason = decision.continuationStopReason
            ?? decision.reason
            ?? "A PreToolUse hook stopped the turn."
        return DeferredMcpContinuationStop(
            output: "Deferred MCP call blocked by hook: " + reason,
            isError: true,
            reason: reason)
    }

    /// Executes a bridge call only after rebuilding the underlying MCP call and
    /// running its own hooks and permission decision. A wrapped call cannot use
    /// the bridge tool's read-only category to bypass the target tool's gates.
    static func executeDeferredMcpCall(
        name: String,
        arguments: [String: Any],
        project: AppProject?,
        chatID: UUID?
    ) async throws -> String {
        let parts = name.components(separatedBy: "__")
        guard parts.count >= 3, parts[0].lowercased() == "mcp" else {
            throw NSError(domain: "TurboSparkToolSearch", code: 5, userInfo: [
                NSLocalizedDescriptionKey: "Malformed deferred MCP tool name: '\(name)'."
            ])
        }
        let serverName = parts[1]
        let toolName = parts.dropFirst(2).joined(separator: "__")
        var stringArguments = ToolSearchCatalog.stringArguments(from: arguments)
        let sessionID = chatID?.uuidString ?? "tool-search"
        let workingDirectory = project?.rootDirectoryPath
        let hookDecision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: sessionID,
            toolName: name,
            toolArguments: stringArguments,
            workingDirectory: workingDirectory,
            projectBoundHookDirectory: workingDirectory)
        if let updated = hookDecision.updatedInput {
            for (key, value) in updated { stringArguments[key] = value }
        }
        if let stop = deferredMcpPreHookStop(hookDecision) {
            throw stop
        }
        if hookDecision.behavior == .deny {
            throw NSError(domain: "TurboSparkToolSearch", code: 6, userInfo: [
                NSLocalizedDescriptionKey: "Deferred MCP call blocked by hook: "
                    + (hookDecision.reason ?? "PreToolUse denied the call.")
            ])
        }
        if hookDecision.behavior == .ask {
            throw approvalRequired(name: name, reason: hookDecision.reason)
        }

        let innerCall = AppToolCall(
            name: name,
            arguments: stringArguments,
            category: .mcp,
            riskAssessment: ToolRiskClassifier.assessRisk(
                name: name, arguments: stringArguments))
        let approved = await SessionApprovalStore.shared.isApproved(
            sessionID: sessionID, toolName: name)
        let decision = AppToolPermissionEngine.evaluate(
            call: innerCall,
            project: project,
            sessionApproved: approved,
            globalServers: GlobalMcpFileStore.load().servers)
        switch decision {
        case .allow:
            break
        case .ask(_, let reason):
            throw approvalRequired(name: name, reason: reason)
        case .deny(let reason):
            throw NSError(domain: "TurboSparkToolSearch", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "Deferred MCP call denied: " + reason
            ])
        }

        var output = try await executeMcpCall(
            serverName: serverName,
            toolName: toolName,
            arguments: stringArguments,
            project: project,
            rootURL: project?.rootDirectoryURL ?? URL(fileURLWithPath: "/dev/null"))
        let postResults = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: sessionID,
            toolName: name,
            toolArguments: stringArguments,
            toolOutput: output,
            toolDurationSeconds: 0,
            isError: false,
            workingDirectory: workingDirectory,
            projectBoundHookDirectory: workingDirectory)
        let postVerdict = AppHookDecisionAggregator.aggregate(postResults, event: .postToolUse)
        if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
            output += "\n\n<hook_feedback>\n" + note + "\n</hook_feedback>"
        }
        if let context = postVerdict.additionalContext, !context.isEmpty {
            output += "\n\n<hook_context>\n" + context + "\n</hook_context>"
        }
        if postVerdict.preventContinuation {
            throw DeferredMcpContinuationStop(
                output: output,
                isError: false,
                reason: postVerdict.continuationStopReason
                    ?? "A PostToolUse hook stopped the turn.")
        }
        return output
    }

    private static func approvalRequired(name: String, reason: String?) -> NSError {
        NSError(domain: "TurboSparkToolSearch", code: 8, userInfo: [
            NSLocalizedDescriptionKey: "Deferred MCP call '" + name + "' requires interactive approval"
                + (reason.map { ": " + $0 } ?? ".")
                + " Use the direct MCP tool in the main conversation so the approval card can be shown."
        ])
    }
}
