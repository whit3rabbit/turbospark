import Foundation
import TurboSpark

/// A tool proposal parked because the chat that made it was not selected.
///
/// Holds exactly what `handleExtractedToolCalls` was called with, so the
/// replay is the ordinary call and the permission ladder (hooks, engine,
/// approval card, Agent mode) is evaluated at replay time, in the foreground.
struct DeferredToolProposal {
    let calls: [AppToolCall]
    let fullContent: String
    let reasoning: String
    let result: GenerationResult
    let step: Int
    let project: AppProject?
}

/// ZP-23: concurrent chats over the single turn lane.
///
/// Read `ChatGenerationCoordinator` first for what "concurrent" means here
/// and what is deferred. This file is the AppModel side: the per-chat guard,
/// the live-output scoping, the background tool deferral and the admission
/// that hands the lane to the next chat when a turn ends.
extension AppModel {
    // MARK: - Guard and scoping

    var laneSnapshot: ChatLaneSnapshot {
        ChatLaneSnapshot(
            generating: generating,
            generatingChatID: generatingChatID,
            submitting: submitting,
            approvalChatID: pendingToolCall == nil ? nil : pendingToolCallChatID,
            hasUnattributedApproval: pendingToolCall != nil && pendingToolCallChatID == nil)
    }

    func laneVerdict(_ op: ChatLaneOperation) -> ChatLaneVerdict {
        ChatGenerationCoordinator.verdict(op, laneSnapshot)
    }

    /// Whether the sidebar may offer New Chat right now.
    public var canCreateChat: Bool { laneVerdict(.select(target: nil)) == .allowed }

    /// Whether the sidebar may offer to open `chatID` right now.
    public func canSelectChat(id: UUID) -> Bool {
        id == selectedChatID || laneVerdict(.select(target: id)) == .allowed
    }

    /// `canSelectChat`, plus the project rule `openChat` enforces: moving to
    /// a chat under another project switches the workspace (file snapshot
    /// hashes, worktree, hook store), and those are not per-chat, so that
    /// move still waits for the running turn.
    public func canOpenChat(id: UUID) -> Bool {
        guard canSelectChat(id: id) else { return false }
        guard let chat = chats.first(where: { $0.id == id }) else { return false }
        return chat.projectID == selectedProjectID
            || (!generating && !submitting && pendingToolCall == nil)
    }

    /// Whether running work should disable row actions (rename, duplicate,
    /// delete, system prompt) for `chatID`: its own turn pins it, so does
    /// work that is not a chat turn, another chat's turn does not.
    public func isChatBusyForRowActions(_ chatID: UUID) -> Bool {
        guard generating || submitting else { return false }
        if submitting { return true }
        return generatingChatID == nil || generatingChatID == chatID
    }

    /// Whether the live stream buffer (`outputText` and friends) describes
    /// the selected chat. It is one buffer for the one running turn, so when
    /// that turn belongs to another chat the selected chat must not draw it.
    public var liveOutputBelongsToSelectedChat: Bool {
        generatingChatID == nil || generatingChatID == selectedChatID
    }

    /// The selected chat's own run state, as the composer and transcript see
    /// it. `isRunning` stays the app-wide "a turn is running" for guards.
    public var isRunningInSelectedChat: Bool {
        generating && liveOutputBelongsToSelectedChat
    }

    /// The running turn's chat title, for named refusals.
    func laneChatTitle(_ id: UUID?) -> String? {
        guard let id else { return nil }
        return chats.first(where: { $0.id == id })?.title
    }

    /// Says why the model cannot be switched: names the chat when one is
    /// attributed, otherwise the generic turn-running statement. Nothing is
    /// started, so the toast is the whole answer.
    func announceModelBusy() {
        guard laneVerdict(.switchModel) != .allowed else { return }
        if let title = laneChatTitle(generatingChatID) {
            showToast(
                AppStrings.localized(
                    "\(title) is generating on this model. Stop it before switching models."),
                style: .warning)
        } else {
            showToast(
                AppStrings.localized("Finish or stop the current turn before switching models."),
                style: .warning)
        }
    }

    // MARK: - The lane

    /// Marks `chatID` as owner of the running turn. Called wherever a chat
    /// turn raises `generating`. Forces the assignment: the lane is held by
    /// whichever turn the app-wide flag says is running, and a stale holder
    /// is a bug to heal, not a reason to refuse the turn.
    func claimLane(for chatID: UUID) {
        generatingChatID = chatID
        // A new turn starts clean: a stale error mark must not mute its finish cue.
        attentionErroredChats.remove(chatID)
        if turnCoordinator.laneHolder != chatID { turnCoordinator.releaseHolder() }
        turnCoordinator.requestLane(for: chatID)
    }

    /// What the sidebar shows for one chat.
    public func chatTurnState(for chatID: UUID) -> ChatTurnState {
        // A parked proposal outranks "generating": the proposal is made from
        // inside the turn, and the flag stays up until the turn's tail runs.
        if deferredToolProposals[chatID] != nil
            || (pendingToolCall != nil && pendingToolCallChatID == chatID)
        {
            return .awaitingTool
        }
        if generating && generatingChatID == chatID { return .generating }
        if pendingUserQuestions?.chatID == chatID { return .awaitingQuestion }
        if !(pendingUserMessages[chatID] ?? []).isEmpty
            || !(pendingTaskNotifications[chatID] ?? []).isEmpty
        {
            return .queued
        }
        return .idle
    }

    /// A chat gained input that waits for the lane. Keeps the admission
    /// order; never grants the lane.
    func noteParkedWork(chatID: UUID) {
        turnCoordinator.enqueue(chatID)
    }

    func forgetLaneState(chatID: UUID) {
        turnCoordinator.forget(chatID)
        deferredToolProposals[chatID] = nil
    }

    // MARK: - Background tool deferral

    /// A proposal made by a live turn of a chat the user is not looking at
    /// parks instead of running. Only the live turn defers: the guard on
    /// `generatingChatID` keeps direct callers (replay, tests) on the
    /// ordinary path.
    func shouldDeferToolProposal(chatID: UUID) -> Bool {
        chatID != selectedChatID && generatingChatID == chatID
    }

    func deferToolProposal(_ proposal: DeferredToolProposal, chatID: UUID) {
        deferredToolProposals[chatID] = proposal
        turnCoordinator.park(chatID, as: .awaitingTool)
        raiseAttention(.needsPermission, chatID: chatID, tool: proposal.calls.first?.name)
    }

    /// Replays a parked proposal through the ordinary path once the chat is
    /// foreground and the lane is free. Returns whether a replay started.
    ///
    /// Runs like the approve path: the app-wide flag goes up for the whole
    /// replay (so Send cannot start a second turn beside it), the epoch moves
    /// (so an older tail cannot clear it), and the shared tail runs after.
    @discardableResult
    func resumeDeferredProposalIfPossible(chatID: UUID) -> Bool {
        guard chatID == selectedChatID,
            let proposal = deferredToolProposals[chatID],
            !generating, !submitting, pendingToolCall == nil, pendingBatchCalls == nil,
            !isEjecting
        else { return false }
        deferredToolProposals[chatID] = nil
        turnCoordinator.unpark(chatID)
        generating = true
        claimLane(for: chatID)
        isCancellationPending = false
        generationEpoch += 1
        let myEpoch = generationEpoch
        toolExecutionTask = Task {
            defer {
                self.finishTurnTail(myEpoch: myEpoch, turnChatID: chatID, session: self.session)
                self.toolExecutionTask = nil
            }
            await self.handleExtractedToolCalls(
                proposal.calls, fullContent: proposal.fullContent,
                reasoning: proposal.reasoning, result: proposal.result,
                currentStep: proposal.step, chatID: chatID, project: proposal.project)
        }
        return true
    }

    // MARK: - Admission

    /// Whether a turn may start in `chatID` without the composer. Background
    /// scheduled prompts (cron, off-peak) use this; a typed prompt goes
    /// through the composer and a task notification still waits for its
    /// chat to be selected, both via `canInjectChatInput`.
    func canStartTurn(inChat chatID: UUID) -> Bool {
        !generating && !submitting && pendingToolCall == nil && pendingBatchCalls == nil
            && !isEjecting
            && deferredToolProposals[chatID] == nil
            && chats.contains(where: { $0.id == chatID })
    }

    /// Hands the lane to the next chat when a turn ends. Foreground first:
    /// the selected chat's parked proposal, then its queued prompt or
    /// notification, then the other chats in the order they parked. A chat
    /// with only a typed prompt parked while it is in the background waits
    /// for selection: that prompt needs the composer.
    func admitParkedWork() {
        guard !generating, !submitting, pendingToolCall == nil, pendingBatchCalls == nil,
            !isEjecting
        else { return }
        let foreground = selectedChatID
        if resumeDeferredProposalIfPossible(chatID: foreground) { return }
        drainPendingUserMessagesIfIdle(chatID: foreground)
        if generating || submitting { return }
        drainPendingTaskNotificationsIfIdle(chatID: foreground)
        if generating || submitting { return }
        var tried = Set<UUID>()
        while let next = turnCoordinator.nextAdmission(foreground: foreground),
            tried.insert(next).inserted
        {
            // Dequeued before the attempt so the loop always advances; a
            // chat that cannot start now is re-queued by whatever parks
            // work for it next.
            turnCoordinator.dequeue(next)
            if next == foreground { continue }
            // Scheduled prompts only: a task notification still waits for its
            // chat to be selected (`canInjectTaskNotification`).
            drainPendingUserMessagesIfIdle(chatID: next)
            if generating || submitting { return }
        }
    }

    // MARK: - Quit

    /// The chat whose live buffer a quit must flush: the running turn's own
    /// chat, which is not the selection when the user left it generating.
    var shutdownFlushChatID: UUID { generatingChatID ?? selectedChatID }

    /// Writes every parked proposal as an interrupted row so Continue and
    /// Retry have an anchor after a relaunch. The running turn is flushed by
    /// `shutdown()` from its own buffer; these chats have no buffer left.
    func flushDeferredProposalsForShutdown() {
        let label = AppStrings.localized("Incomplete tool call, not executed")
        for (chatID, proposal) in deferredToolProposals {
            mutateRecoveryState(for: chatID) { messages, anchor in
                if let preservation = AppStreamRecovery.preserveInterruptedOutput(
                    into: &messages,
                    content: proposal.fullContent,
                    reasoning: proposal.reasoning,
                    stopReason: "cancelled",
                    toolCallLabel: label)
                {
                    anchor = preservation.anchor
                }
            }
        }
        deferredToolProposals.removeAll()
    }
}

extension AppModel {
    /// Clears the live stream buffer, unless a turn is running: the buffer
    /// then belongs to that turn, whichever chat the user is looking at.
    func resetLiveOutputIfNoTurnRuns() {
        guard !generating else { return }
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
    }
}
