import Foundation
import TurboSpark

/// The static half of Agent-mode routing, shared by the main loop's router
/// and the subagent gate so the two cannot drift.
///
/// **THE CLASSIFIER IS THE ASK-BAND, NOTHING ELSE.** Deny rules and
/// `hardGated` verdicts (denylist regexes, structural shell checks,
/// sensitive paths, the repo-import MCP gate) keep their human, and the
/// classifier is consulted only where the ladder ran out of answer. Within
/// that band this mirrors Qwen Code's layering: MCP tools always classify
/// (their annotations are the server's own claims), while a non-MCP call
/// the static ladder scored `.safe`/`.low` runs on the fast path -- that is
/// the same population `.auto` already runs, so asking a model about `ls`
/// would only add latency.
enum AgentModeRouting {
    enum PreClassifierDecision: Equatable {
        /// The ordinary approval card (or, for a subagent with no UI, the
        /// ordinary refusal).
        case manualCard
        /// The static ladder vouched for the call; run it without a card.
        case fastAllow
        /// Ask the classifier.
        case classify
    }

    static func preClassifierDecision(
        call: AppToolCall, assessment: ToolRiskAssessment,
        suspended: Bool, skipClassifier: Bool
    ) -> PreClassifierDecision {
        if assessment.isHardGated {
            return .manualCard
        }
        if suspended || skipClassifier {
            return .manualCard
        }
        if call.category == .mcp {
            return .classify
        }
        if !assessment.isHighRisk {
            return .fastAllow
        }
        return .classify
    }

    /// The denial the model sees for a classifier block: the verdict, then
    /// qwen-code's don't-route-around guidance. Unrelated safe work stays
    /// allowed; only the denied intent is closed off. Nonisolated ON PURPOSE:
    /// the subagent gate feeds it to an error observation off the main actor.
    static func blockMessage(reason: String) -> String {
        "Blocked by Agent mode policy: \(reason). This denied action must not be completed "
            + "through another tool, shell indirection, a generated script, an alias, a "
            + "symlink, a config change, a hook, or an encoded payload. Unrelated safe work "
            + "and genuinely safer alternatives are still allowed; if this action is "
            + "genuinely required, stop and ask the user for explicit approval."
    }
}

/// Agent-mode routing for tool calls the static permission ladder asked
/// about (`swift/docs/SWIFT_AGENT_MODE.md`).
extension AppModel {
    /// What the agent-mode router decided about one `.ask`.
    enum AgentModeOutcome: Equatable {
        /// Run without a card. `byClassifier` distinguishes a classifier
        /// allow (the tool card marks it) from the static fast path, which
        /// runs exactly as `.auto` would and marks nothing.
        case run(byClassifier: Bool)
        /// Refuse with a policy reason, fed to the model like an engine deny.
        case denyWithReason(String)
        /// Park the ordinary approval card. The notice, when set, explains a
        /// classifier fallback (unavailable or skipped) on the card.
        case parkCard(notice: String?)
    }

    /// Resolves one engine `.ask` under Agent mode. Called AFTER the
    /// PermissionRequest hook declines to decide, from both the single-call
    /// path and the batch router, so the two cannot drift.
    ///
    /// - Parameters:
    ///   - call: the call as it will execute (post-hook `updatedInput`).
    ///   - assessment: the assessment the engine's `.ask` carried.
    ///   - chatID: the chat whose counters and transcript the decision uses.
    ///   - project: the TURN's project, whose mode governs -- never
    ///     `selectedProject` (state#30's rule, same as the engine's).
    func resolveAskUnderAgentMode(
        _ call: AppToolCall, assessment: ToolRiskAssessment,
        chatID: UUID, project: AppProject?
    ) async -> AgentModeOutcome {
        let mode = project?.permissions.mode ?? activePermissionMode
        guard mode == .agentAuto else { return .parkCard(notice: nil) }

        let sessionID = chatID.uuidString
        // Hard rules win, and a suspended session asked for this: both go to
        // the human before the classifier is ever consulted.
        let suspended = await AgentModeGate.shared.isSuspended(sessionID: sessionID)
        let skip = await AgentModeGate.shared.shouldSkipClassifier(sessionID: sessionID)
        let skipText = skip ? await AgentModeGate.shared.skipReason(sessionID: sessionID) : nil
        switch AgentModeRouting.preClassifierDecision(
            call: call, assessment: assessment, suspended: suspended, skipClassifier: skip)
        {
        case .manualCard:
            return .parkCard(notice: skipText)
        case .fastAllow:
            await AgentModeGate.shared.recordAllow(sessionID: sessionID)
            return .run(byClassifier: false)
        case .classify:
            break
        }

        let classifier = makeAgentModeClassifier(project: project)
        let request = ClassifierRequest(
            toolName: call.name,
            category: call.category,
            projectedCall: ToolCallProjection.projectedCall(call),
            recentUserIntent: recentUserIntent(chatID: chatID))
        let verdict = await classifier.classify(request)

        switch verdict {
        case .allow:
            await AgentModeGate.shared.recordAllow(sessionID: sessionID)
            return .run(byClassifier: true)
        case .block(let reason):
            await AgentModeGate.shared.recordBlock(sessionID: sessionID)
            return .denyWithReason(AgentModeRouting.blockMessage(reason: reason))
        case .unavailable(let reason):
            let streak = await AgentModeGate.shared.recordUnavailable(sessionID: sessionID)
            let thresholdNote =
                streak >= AgentModeGate.maxConsecutiveUnavailable
                ? " Further calls will skip the classifier until one is approved."
                : ""
            return .parkCard(
                notice: "Agent mode could not classify this call (\(reason))."
                    + thresholdNote)
        }
    }

    /// The fallback card's "Suspend Agent Mode" choice: stop consulting the
    /// classifier for the rest of THIS session. Session-scoped like qwen
    /// Code's switch-to-Default option -- settings are untouched, and
    /// re-selecting the mode is what ends a suspension deliberately.
    public func suspendAgentModeForSession() {
        let sessionID = (pendingToolCallChatID ?? selectedChatID).uuidString
        Task { await AgentModeGate.shared.suspend(sessionID: sessionID) }
    }

    /// The classifier for one decision, from the turn's own session and the
    /// mirrored hints. Tests inject `agentModeClassifierOverride` instead.
    func makeAgentModeClassifier(project: AppProject?) -> ToolCallClassifying {
        if let override = agentModeClassifierOverride {
            return override
        }
        return LocalModelToolClassifier(
            session: session,
            hints: agentModeHints,
            workspaceRoot: project?.rootDirectoryPath ?? "")
    }

    /// The recent user text, capped, so the classifier can weigh intent
    /// against the soft-deny list. The last two user messages are the
    /// signal; older turns are the transcript's job, not the verdict's.
    func recentUserIntent(chatID: UUID) -> String {
        let userMessages = turnMessages(for: chatID)
            .filter { $0.role == .user && !$0.content.isEmpty }
            .suffix(2)
            .map { $0.content }
        guard !userMessages.isEmpty else { return "" }
        let joined = userMessages.joined(separator: "\n---\n")
        guard joined.utf8.count > 1_500 else { return joined }
        return String(decoding: joined.utf8.suffix(1_500), as: UTF8.self)
    }
}
