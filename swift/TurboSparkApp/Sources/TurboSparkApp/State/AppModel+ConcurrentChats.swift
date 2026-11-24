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

/// An approval card parked because the user left its chat (ZP-23).
///
/// A snapshot of the approval slot, keyed by chat in
/// `AppModel.parkedApprovals`. The proposal's transcript row already exists
/// at `.pendingApproval`, so nothing is replayed through the model: coming
/// back moves the card into the slot after `revalidateParkedApproval`.
struct ParkedApproval {
    let call: AppToolCall
    let chatID: UUID
    let step: Int
    /// The project the call was evaluated against. A SNAPSHOT: the live
    /// project is looked up again at restore, never trusted from here.
    let project: AppProject?
    let batchCalls: [AppToolCall]?
    let validationCall: AppToolCall?
    let classifierNotice: String?
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
    /// move still waits for the running turn. The selected chat's own
    /// approval card does not block it: `openChat` parks the card first, and
    /// the card keeps the project it was evaluated against.
    public func canOpenChat(id: UUID) -> Bool {
        guard canSelectChat(id: id) else { return false }
        guard let chat = chats.first(where: { $0.id == id }) else { return false }
        return chat.projectID == selectedProjectID
            || (!generating && !submitting && !slotHoldsForeignApproval)
    }

    /// An approval in the slot that is not the selected chat's, or has no
    /// chat, and so cannot be parked by leaving. Never true by construction
    /// of the slot (it follows the selection), kept as the guard's own term.
    var slotHoldsForeignApproval: Bool {
        pendingToolCall != nil && pendingToolCallChatID != selectedChatID
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
            || parkedApprovals[chatID] != nil
            || (pendingToolCall != nil && pendingToolCallChatID == chatID)
        {
            return .awaitingTool
        }
        // A question outranks "generating" for the same reason: the turn
        // holds the lane (the question blocks inside its tool task) but what
        // the chat is waiting for is the user.
        if pendingUserQuestions?.chatID == chatID { return .awaitingQuestion }
        if pendingPlanApproval?.chatID == chatID { return .awaitingQuestion }
        if generating && generatingChatID == chatID { return .generating }
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
        parkedApprovals[chatID] = nil
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
    func resumeDeferredProposalIfPossible(chatID: UUID, announce: Bool = true) -> Bool {
        guard chatID == selectedChatID,
            let proposal = deferredToolProposals[chatID],
            !generating, !submitting, pendingToolCall == nil, pendingBatchCalls == nil,
            !isEjecting
        else { return false }
        // The proposal replays in the workspace it was made in, or not at
        // all: the snapshot store, worktree and hook store are not per chat.
        guard alignWorkspace(toChat: chatID, announce: announce) else { return false }
        // The project snapshot is stale by definition. Replay against the
        // live one, so a permission edit made while the proposal waited
        // applies; a project deleted meanwhile means nothing runs.
        let live = liveProject(matching: proposal.project)
        deferredToolProposals[chatID] = nil
        turnCoordinator.unpark(chatID)
        if live.gone {
            writeInterruptedRow(for: chatID, proposal: proposal)
            showToast(
                AppStrings.localized(
                    "A tool call was dropped because its project no longer exists."),
                style: .warning)
            return false
        }
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
                currentStep: proposal.step, chatID: chatID, project: live.project)
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
            && parkedApprovals[chatID] == nil
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
        if resumeDeferredProposalIfPossible(chatID: foreground, announce: false) { return }
        if restoreParkedApprovalIfPossible(chatID: foreground, announce: false) { return }
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
        for (chatID, proposal) in deferredToolProposals {
            writeInterruptedRow(for: chatID, proposal: proposal)
        }
        deferredToolProposals.removeAll()
        // A parked approval needs no row: its proposal is already in the
        // transcript at `.pendingApproval`, which load reconciles.
        parkedApprovals.removeAll()
    }

    /// The interrupted row Continue and Retry anchor on, written from a
    /// proposal's captured prose.
    func writeInterruptedRow(for chatID: UUID, proposal: DeferredToolProposal) {
        let label = AppStrings.localized("Incomplete tool call, not executed")
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

// MARK: - Parked approvals (CUJ 2)

extension AppModel {
    /// Moves the slot's approval card into the per-chat store when the user
    /// leaves its chat. Returns whether a card moved.
    ///
    /// **THE SLOT DESCRIBES THE SELECTED CHAT ONLY.** Approve and deny read
    /// `pendingToolCallChatID` and the slot's call id, so a card can only be
    /// answered while it is in the slot, and only the chat that owns it ever
    /// puts it back. Chat A's approval therefore cannot be applied to chat B:
    /// leaving A empties the slot, and B's own card (if any) refills it.
    @discardableResult
    func parkSlotApproval() -> Bool {
        guard let call = pendingToolCall, let chatID = pendingToolCallChatID else { return false }
        parkedApprovals[chatID] = ParkedApproval(
            call: call, chatID: chatID, step: pendingToolCallStep,
            project: pendingToolCallProject, batchCalls: pendingBatchCalls,
            validationCall: pendingValidationCall,
            classifierNotice: pendingToolCallClassifierNotice)
        clearPendingToolCall()
        turnCoordinator.park(chatID, as: .awaitingTool)
        return true
    }

    /// Puts `chatID`'s parked card back in the slot, if the app is quiet
    /// enough to show it. Returns whether the card was restored.
    ///
    /// **A REPLAY IS A NEW DECISION POINT, NOT A RESUME.** The call was
    /// judged against the permission state of the moment it was proposed. By
    /// now the project may have been edited, switched to a stricter mode, or
    /// deleted, so the restore runs the engine again against the LIVE
    /// project: a refusal is applied (the call is denied with the engine's
    /// reason, and the model is told that, not that the user said no), and
    /// anything else shows the card. It never auto-approves: a call the
    /// policy has since relaxed still waits for the user.
    @discardableResult
    func restoreParkedApprovalIfPossible(chatID: UUID, announce: Bool = true) -> Bool {
        guard chatID == selectedChatID, let parked = parkedApprovals[chatID],
            pendingToolCall == nil, pendingBatchCalls == nil,
            !generating, !submitting, !isEjecting,
            deferredToolProposals[chatID] == nil
        else { return false }
        guard alignWorkspace(toChat: chatID, announce: announce) else { return false }
        parkedApprovals[chatID] = nil
        turnCoordinator.unpark(chatID)

        let live = liveProject(matching: parked.project)
        restoringParkedApproval = true
        pendingToolCall = parked.call
        pendingToolCallChatID = chatID
        pendingToolCallStep = parked.step
        pendingToolCallProject = live.project
        pendingBatchCalls = parked.batchCalls
        pendingValidationCall = parked.validationCall
        pendingToolCallClassifierNotice = parked.classifierNotice
        restoringParkedApproval = false

        if let reason = revalidationDenial(of: parked, project: live.project, gone: live.gone) {
            denyPendingToolCall(id: parked.call.id, policyReason: reason)
        }
        return true
    }

    /// The reason the permission state now refuses a parked card, or nil.
    /// Only a refusal is acted on; `ask` and `allow` both leave the card.
    func revalidationDenial(
        of parked: ParkedApproval, project: AppProject?, gone: Bool
    ) -> String? {
        if gone { return "The project this call was proposed in no longer exists." }
        let calls = (parked.batchCalls ?? [parked.call])
            + (parked.validationCall.map { [$0] } ?? [])
        for call in calls {
            // `sessionApproved: false` on purpose: a session grant can only
            // turn an ask into an allow, never lift a deny, and the card is
            // the user's answer for the ask case.
            let decision = AppToolPermissionEngine.evaluate(
                call: call, project: project, sessionApproved: false,
                fallbackMode: activePermissionMode, globalServers: globalMcpServers,
                browserContext: browserPermissionContext(for: call, project: project),
                networkHostApproved: false)
            if case .deny(let reason) = decision { return reason }
        }
        return nil
    }

    /// `captured` re-read from the live project list. `gone` is true when a
    /// project was captured and no longer exists; a call proposed with no
    /// project stays projectless.
    func liveProject(matching captured: AppProject?) -> (project: AppProject?, gone: Bool) {
        guard let captured else { return (nil, false) }
        guard let live = projects.first(where: { $0.id == captured.id }) else { return (nil, true) }
        return (live, false)
    }

    /// Makes the selected workspace the chat's own project before parked
    /// work replays. A project switch rebinds the worktree, the snapshot
    /// store and the hook store, so it keeps its idle requirement: while a
    /// turn runs this refuses and the work stays parked.
    func alignWorkspace(toChat chatID: UUID, announce: Bool) -> Bool {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return false }
        if chat.projectID == selectedProjectID { return true }
        if applyProjectSelection(id: chat.projectID) { return true }
        if announce {
            showToast(
                AppStrings.localized(
                    "Finish the running turn before opening this approval. It belongs to another project."
                ),
                style: .warning)
        }
        return false
    }
}

// MARK: - Background prompts (composer-free start)

extension AppModel {
    /// Whether a typed prompt may start without the composer. Slash
    /// commands, bang commands and quick-saves are dispatched by `run()` off
    /// the composer text and need the composer; plain prose does not.
    static func startsWithoutComposer(_ text: String) -> Bool {
        guard let first = text.trimmingCharacters(in: .whitespacesAndNewlines).first else {
            return false
        }
        return !"/!#".contains(first)
    }

    /// Starts the first queued TYPED prompt of a chat the user is not looking
    /// at. Returns the submission when one started.
    ///
    /// Not `run()`: that reads and clears the composer, which describes the
    /// selected chat. This mirrors `run()`'s body for a prompt that already
    /// left the composer (the steer assembly: mentions, attachments,
    /// `UserPromptSubmit`, sanitization) and then starts the turn in the
    /// chat's own id. `submitting` covers the hook window like `run()`'s, and
    /// Stop reaches it through `submissionTask`. A hook block re-parks the
    /// entry and surfaces its error, same as a mid-turn steer.
    func startBackgroundTypedPrompt(chatID: UUID) -> PendingUserSubmission? {
        guard chatID != selectedChatID, canStartTurn(inChat: chatID),
            hasModelReadyForScheduledDelivery,
            let queue = pendingUserMessages[chatID], let first = queue.first,
            !first.isScheduled, Self.startsWithoutComposer(first.text)
        else { return nil }
        submitting = true
        submissionTask = Task {
            defer {
                self.submitting = false
                self.submissionTask = nil
                if !self.generating { self.isCancellationPending = false }
            }
            self.stopHookReentryCount = 0
            self.resetGoalForUserPrompt(chatID: chatID)
            self.pendingResponseVariants[chatID] = nil
            self.pendingRecoveryRetryCounts[chatID] = nil
            let delivered = await self.deliverSteersAtBoundary(
                chatID: chatID, project: self.turnProject(chatID: chatID),
                visionAvailable: self.visionIsActive, maxEntries: 1)
            guard delivered, !Task.isCancelled, self.chats.contains(where: { $0.id == chatID })
            else { return }
            self.executeGenerationTurn(step: 0, chatID: chatID)
        }
        return PendingUserSubmission(
            text: first.text, attachments: first.attachments, presentation: .userSteer)
    }
}
