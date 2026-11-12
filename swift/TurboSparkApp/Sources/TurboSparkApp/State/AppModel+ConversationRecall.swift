import Foundation
import SyntextContext
import TurboSpark

/// Wires conversation recall (`ChatRecallStore`, `ChatRecallPlanner`) into
/// the turn: journaling transcripts, adding recalled evidence to a fitted
/// prompt, and forgetting deleted chats.
extension AppModel {
    /// Pending debounced reconciles, one per chat.
    @MainActor private static var recallReconcileTasks: [UUID: Task<Void, Never>] = [:]

    /// Whether this chat's turns may be journaled and recalled. Never a
    /// Ghost chat (it exists only in memory) or a passphrase-protected
    /// profile (nothing readable while locked); never the bounded
    /// SKILL.state prompt, whose size does not grow with the conversation.
    func conversationRecallEligible(chatID: UUID) -> Bool {
        guard conversationRecallEnabled,
              let chat = chats.first(where: { $0.id == chatID }), !chat.isGhost,
              !ProfileVaultCoordinator.shared.isProtected
        else { return false }
        return turnProject(chatID: chatID)?.skillStateEnabled != true
    }

    /// Journals the chat's transcript now (queued, in order, off the main
    /// thread). The store compares against what it already holds.
    func reconcileConversationRecall(chatID: UUID) {
        Self.recallReconcileTasks[chatID]?.cancel()
        Self.recallReconcileTasks[chatID] = nil
        guard conversationRecallEligible(chatID: chatID) else { return }
        ChatRecallStore.shared.reconcile(
            chatID: chatID, turns: ChatRecallPlanner.recordable(turnMessages(for: chatID)))
    }

    /// Journals the chat shortly after a burst of transcript writes, so an
    /// edit or rewind that removes text takes it out of the store without
    /// waiting for the chat's next turn.
    func scheduleConversationRecallReconcile(chatID: UUID) {
        guard conversationRecallEnabled else { return }
        Self.recallReconcileTasks[chatID]?.cancel()
        Self.recallReconcileTasks[chatID] = Task { @MainActor [weak self] in
            try? await Task.sleep(for: .milliseconds(750))
            guard !Task.isCancelled else { return }
            self?.reconcileConversationRecall(chatID: chatID)
        }
    }

    /// The user deleted or cleared the chat: its turns leave the store, and
    /// the bytes leave the disk at the forced compaction that follows.
    func forgetConversationRecall(chatID: UUID) {
        Self.recallReconcileTasks[chatID]?.cancel()
        Self.recallReconcileTasks[chatID] = nil
        ChatRecallStore.shared.forget(chatID: chatID)
    }

    /// Adds recalled evidence to the fitted request when turns of this chat
    /// are outside it, or returns `fitted` unchanged (see
    /// `ChatRecallPlanner.applyConversationRecall`).
    ///
    /// - Parameters:
    ///   - projection: the projection the request was built from.
    ///   - requestMessages: the messages the fit actually ran on (after
    ///     microcompaction), which a tighter refit reuses.
    func applyConversationRecall(
        to fitted: AppChatInstructionPin.FitOutcome,
        chatID: UUID,
        projection: AppChatHistoryProjection,
        requestMessages: [ChatMessage],
        promptBudget: Int,
        session: TurboSparkSession,
        reasoning: GenerateOptions.Reasoning,
        tools: [ToolSpec]
    ) async -> AppChatInstructionPin.FitOutcome {
        let transcript = turnMessages(for: chatID)
        guard requestMessages.count == projection.sourceRowIndexByMessage.count,
              let questionRow = transcript.lastIndex(where: { $0.role == .user && !$0.isSynthetic })
        else { return fitted }
        let boundary = min(max(compactionState(chatID: chatID).boundary, 0), transcript.count)
        let rebasedQuestionRow = questionRow - boundary
        let question = requestMessages.indices.last {
            projection.sourceRowIndexByMessage[$0] == rebasedQuestionRow && requestMessages[$0].role == .user
        }.map { requestMessages[$0] }
        let questionText = ChatRecallPlanner.normalize(transcript[questionRow].content)
        let pinIndex = projection.instructionPinBlockIndex
        let modelPath = session.info.modelPath

        let result = await ChatRecallPlanner.applyConversationRecall(
            fitted: ChatRecallPlanner.Fit(
                retained: fitted.retained, measuredTokens: fitted.measuredTokens,
                removedTurnCount: fitted.removedTurnCount,
                hasRoomForGeneration: fitted.hasRoomForGeneration),
            inputs: ChatRecallPlanner.Inputs(
                projectionMessages: requestMessages,
                sourceRows: projection.sourceRowIndexByMessage,
                pinIndex: pinIndex,
                boundary: boundary,
                turns: ChatRecallPlanner.recordable(transcript),
                question: question,
                promptBudget: promptBudget),
            refit: { limit in
                guard limit > 0, let bound = UInt32(exactly: limit) else { return nil }
                let refitted = await self.fitRequestHistoryPreservingInstructionPins(
                    requestMessages, injectedBlockIndex: pinIndex, chatID: chatID,
                    maxTokens: bound, session: session, reasoning: reasoning, tools: tools)
                return ChatRecallPlanner.Fit(
                    retained: refitted.retained, measuredTokens: refitted.measuredTokens,
                    removedTurnCount: refitted.removedTurnCount,
                    hasRoomForGeneration: refitted.hasRoomForGeneration)
            },
            countTokens: { messages in
                try? await session.countTokens(messages, reasoning: reasoning, tools: tools)
            },
            evidence: { keptTurns, budget in
                await Self.recallValue(within: .milliseconds(1_500)) {
                    await ChatRecallStore.shared.evidence(
                        chatID: chatID, question: questionText, keptTurns: keptTurns,
                        budget: budget, modelPath: modelPath)
                }
            },
            policy: SyntextContextEngine.evidencePolicy)

        return AppChatInstructionPin.FitOutcome(
            retained: result.retained, measuredTokens: result.measuredTokens,
            removedTurnCount: result.removedTurnCount,
            hasRoomForGeneration: result.hasRoomForGeneration)
    }

    /// `work`'s result if it arrives within `limit`, else nil. The work is
    /// not cancelled, only no longer waited for: a blocking FFI call cannot
    /// be interrupted, and a task group would wait for it anyway. The turn
    /// then runs without recall rather than stalling behind a long
    /// reconcile.
    nonisolated static func recallValue(
        within limit: Duration, _ work: @escaping @Sendable () async -> String?
    ) async -> String? {
        let once = RecallResumeOnce()
        return await withCheckedContinuation { continuation in
            Task.detached {
                let value = await work()
                if once.claim() { continuation.resume(returning: value) }
            }
            Task.detached {
                try? await Task.sleep(for: limit)
                if once.claim() { continuation.resume(returning: nil) }
            }
        }
    }
}

/// Lets exactly one of two racing tasks resume a continuation.
private final class RecallResumeOnce: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false

    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !claimed else { return false }
        claimed = true
        return true
    }
}
