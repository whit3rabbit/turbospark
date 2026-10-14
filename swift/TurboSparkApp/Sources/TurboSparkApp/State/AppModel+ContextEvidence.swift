import Foundation
import SyntextContext
import TurboSpark

/// Retrieved evidence for a user turn (`swift/docs/SWIFT_CONTEXT_ENGINE.md`).
///
/// **EVIDENCE GOES INTO THE ROOM THE FIT LEFT, NEVER INTO ROOM IT TOOK.**
/// It is retrieved after compaction and the window fit have shaped the
/// history, budgeted from what is still free under the prompt budget, and
/// re-counted with the template before it is sent. A turn whose evidence
/// does not fit is sent without it, and any engine failure only costs the
/// evidence: retrieval never fails or delays a turn into an error.
///
/// It goes into the user row it was retrieved for, ahead of the question,
/// and is never written to the transcript. Agent steps and continuations of
/// the same turn re-attach the same text (`contextTurnEvidence`), so the
/// prompt prefix up to and through it is identical and its KV is reused.
extension AppModel {
    static let contextEvidenceMinimumTokens = 128
    static let contextEvidenceMaximumTokens = 4096
    /// Longest question text sent to retrieval. Pasted attachments ride in
    /// the user row and would otherwise become the query.
    static let contextQuestionCharacters = 2000

    /// The policy that tells the model evidence is data to cite, never
    /// instructions. In the system prompt of every project turn with the
    /// engine on, so that prompt is the same on every turn (engines run with
    /// `appendPolicy: false`). Recall-only chats rely on the evidence's own
    /// preamble and keep their system prompt unchanged.
    static var contextEvidencePolicy: String { SyntextContextEngine.evidencePolicy }

    /// Tokens a turn's evidence may use: three quarters of what the fit left
    /// free, at most a third of the window and 4,096; nothing below 128.
    static func contextEvidenceBudget(promptBudget: Int, measuredTokens: Int, window: Int) -> Int {
        let remaining = promptBudget - measuredTokens - max(32, window / 32)
        guard remaining > 0 else { return 0 }
        let budget = min(remaining * 3 / 4, window / 3, contextEvidenceMaximumTokens)
        return budget >= contextEvidenceMinimumTokens ? budget : 0
    }

    /// `retained` with the turn's evidence in its question, or unchanged.
    func messagesWithContextEvidence(
        _ retained: [ChatMessage],
        measuredTokens: Int,
        chatID: UUID,
        step: Int,
        project: AppProject?,
        promptBudget: Int,
        session: TurboSparkSession,
        reasoning: GenerateOptions.Reasoning,
        tools: [ToolSpec]
    ) async -> [ChatMessage] {
        let recall = contextRecallLocation(chatID: chatID)
        let projectID = contextEngineActive(for: project) ? project?.id : nil
        guard recall != nil || projectID != nil,
            let question = turnMessages(for: chatID).last(where: ContextRecall.isRecordedUserRow),
            let target = retained.lastIndex(where: {
                $0.role == .user && $0.content.contains(question.content)
            })
        else {
            contextTurnEvidence[chatID] = nil
            return retained
        }

        var evidence: ContextEvidence?
        if let cached = contextTurnEvidence[chatID], cached.anchor == question.id {
            evidence = cached.evidence
        } else if step == 0 {
            let budget = Self.contextEvidenceBudget(
                promptBudget: promptBudget, measuredTokens: measuredTokens,
                window: Int(session.info.maxContext))
            evidence = await retrieveContextEvidence(
                question: question.content, chatID: chatID, recall: recall, project: project,
                projectID: projectID, retained: retained, budget: budget,
                free: promptBudget - measuredTokens, session: session)
        }
        guard var current = evidence else {
            contextTurnEvidence[chatID] = nil
            return retained
        }

        // Counted with the template, as the fit was. One retry, smaller by
        // the overflow, on the step that retrieved; a later step whose
        // history has grown past it goes on without evidence.
        for attempt in 0..<2 {
            var candidate = retained
            candidate[target].content = current.text + "\n\n" + retained[target].content
            guard let counted = try? await session.countTokens(
                candidate, reasoning: reasoning, tools: tools)
            else { break }
            if counted <= promptBudget {
                contextTurnEvidence[chatID] = (question.id, current)
                return candidate
            }
            let smaller = current.tokens - (counted - promptBudget) - 64
            guard attempt == 0, step == 0, smaller >= Self.contextEvidenceMinimumTokens,
                let retried = await retrieveContextEvidence(
                    question: question.content, chatID: chatID, recall: recall, project: project,
                    projectID: projectID, retained: retained, budget: smaller,
                    free: promptBudget - measuredTokens, session: session)
            else { break }
            current = retried
        }
        contextTurnEvidence[chatID] = nil
        return retained
    }

    private func retrieveContextEvidence(
        question: String,
        chatID: UUID,
        recall: ContextChatLocation?,
        project: AppProject?,
        projectID: UUID?,
        retained: [ChatMessage],
        budget: Int,
        free: Int,
        session: TurboSparkSession
    ) async -> ContextEvidence? {
        guard budget >= Self.contextEvidenceMinimumTokens else { return nil }
        let rows = ContextRecall.rows(from: turnMessages(for: chatID), beforeLastQuestion: true)
        let request = ContextEvidenceRequest(
            chatID: chatID,
            recall: recall,
            projectID: projectID,
            question: String(question.prefix(Self.contextQuestionCharacters)),
            evidenceTokens: budget,
            recallRows: recall == nil ? [] : rows,
            keptTurns: ContextRecall.keptRows(rows, in: retained),
            // One or two results fit a small window; never more than 1,024.
            toolResultTokens: min(1024, max(256, free / 4)))
        let manager = ContextEngineManager.shared
        await manager.setModelPath(session.info.modelPath)
        // The selected project may not have been activated this launch: the
        // first turn opens and syncs it in the background (the sync does not
        // block this retrieval, which reads what the store already holds).
        if let projectID, projectID == selectedProjectID, let root = project?.rootDirectoryURL {
            await manager.activateProject(id: projectID, root: root, resync: false)
        }
        do {
            return try await manager.prepareEvidence(request)
        } catch {
            FileHandle.standardError.write(Data("TurboSpark: context evidence unavailable: \(error)\n".utf8))
            return nil
        }
    }
}
