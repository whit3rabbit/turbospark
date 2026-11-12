import Foundation
import TurboSpark

/// How a turn stops when it was not allowed to finish.
///
/// **COOPERATIVE, AND `runTask` ALONE DOES NOT REACH THE LOOP.** An approved
/// pending call runs in `toolExecutionTask`, spawned from the approval card
/// after the proposing turn's stream already ended, so cancelling `runTask`
/// there cancels a task with nothing left to do. `isCancellationPending` is
/// the flag `continueAgentLoop` reads: a tool already in flight finishes, and
/// what Stop guarantees is that no FURTHER model turn or tool call starts.
///
/// Three state-review items landed on this one page -- state#34 (Stop was
/// refused while a call awaited approval), state#38 (a Stop hook appended an
/// orphan turn after a cancel) and state#99 (the flag latched when nothing
/// ran a tail afterwards) -- which is what makes it a subject rather than a
/// tail of the generation file.
extension AppModel {
    /// Records whatever the turn had produced before it stopped.
    ///
    /// `reason` distinguishes the two callers: a turn that threw was recorded
    /// as `"cancelled"` alongside a real user cancellation, so a transcript
    /// could not tell a user pressing Stop from the engine failing mid-turn --
    /// and the `error` banner beside it is transient while the stop reason is
    /// persisted.
    func finishCancelled(
        chatID: UUID? = nil,
        reason: String = "cancelled",
        continuationsUsed: Int = 0
    ) {
        let targetID = chatID ?? selectedChatID
        let targetProject = turnProject(chatID: targetID)
        let recoveryRetryAttempt = pendingRecoveryRetryCounts.removeValue(forKey: targetID)
        let hasFreshReplacement = recoveryRetryAttempt == nil
            && !(pendingResponseVariants[targetID] ?? []).isEmpty
        if !outputText.isEmpty || !outputReasoningText.isEmpty || hasFreshReplacement {
            // A Retry/Edit turn stopped mid-flight still owes its variants
            // an answer: the partial reply becomes the active version (its
            // stop reason marks it as unfinished) and the replaced one is
            // kept beside it. An empty failed replacement keeps an empty
            // interrupted row, so an edited prompt is never paired with its
            // previous answer as though generation succeeded. Consumed here
            // for the same once-only reason
            // `finishProseTurn` consumes it.
            let parkedVariants =
                self.pendingResponseVariants.removeValue(forKey: targetID) ?? []
            let selectedLanguage = AppLanguage.resolve(
                UserDefaults.standard.string(forKey: AppLanguage.storageKey)
                    ?? AppLanguage.system.rawValue)
            let toolCallLabel = String(
                localized: "Incomplete tool call, not executed",
                bundle: .module,
                locale: selectedLanguage.locale)
            let previousRetries = recoveryRetryAttempt ?? 0

            var preservation: InterruptedStreamPreservation?
            if let index = chats.firstIndex(where: { $0.id == targetID }), chats[index].isGhost {
                mutateGhostPayload(for: targetID) { payload in
                    preservation = AppStreamRecovery.preserveInterruptedOutput(
                        into: &payload.messages,
                        content: self.outputText,
                        reasoning: self.outputReasoningText,
                        stopReason: reason,
                        continuationsUsed: continuationsUsed,
                        retriesUsed: previousRetries,
                        alternates: parkedVariants,
                        toolCallLabel: toolCallLabel)
                    if let preservation {
                        payload.recoveryAnchor = preservation.anchor
                    }
                }
            } else if let index = chats.firstIndex(where: { $0.id == targetID }) {
                var messages = chats[index].messages
                preservation = AppStreamRecovery.preserveInterruptedOutput(
                    into: &messages,
                    content: outputText,
                    reasoning: outputReasoningText,
                    stopReason: reason,
                    continuationsUsed: continuationsUsed,
                    retriesUsed: previousRetries,
                    alternates: parkedVariants,
                    toolCallLabel: toolCallLabel)
                if let preservation {
                    chats[index].messages = messages
                    chats[index].recoveryAnchor = preservation.anchor
                    chats[index].updatedAt = Date()
                    persistChats()
                }
            }
            if let preservation {
                self.streamRecoveryEvents[targetID, default: []].append(contentsOf: preservation.events)
            }
            outputText = ""
            outputReasoningText = ""
        } else if recoveryRetryAttempt != nil {
            // A retry can fail before yielding any text. Restore its parked
            // interrupted row and advance the anchor so empty failures still
            // consume one of the bounded recovery attempts.
            let parkedVariants = pendingResponseVariants.removeValue(forKey: targetID) ?? []
            if let interrupted = parkedVariants.first {
                var updatedAnchor: RecoveryAnchor?
                mutateRecoveryState(for: targetID) { messages, anchor in
                    let retainedMessageCount = messages.count
                    if !messages.contains(where: { $0.id == interrupted.id }) {
                        messages.append(interrupted)
                    }
                    updatedAnchor = AppStreamRecovery.anchor(
                        retainedMessageCount: retainedMessageCount,
                        messageID: interrupted.id,
                        persistedRowContent: interrupted.content,
                        continuationsUsed: anchor?.continuationsUsed ?? 0,
                        retriesUsed: recoveryRetryAttempt ?? 0)
                    anchor = updatedAnchor
                }
                if let updatedAnchor {
                    streamRecoveryEvents[targetID, default: []].append(.anchorRecorded(updatedAnchor))
                }
            }
        }
        // `Stop` fires for observability even on a cancelled or errored
        // turn, but its block-and-continue capability does not apply here
        // on purpose: cancellation is user-initiated (the Stop button
        // should really stop), and re-entering after an error risks looping
        // straight back into the same failure. Fire-and-forget rather than
        // awaited, so a slow hook cannot delay the cancel/error UI update.
        Task {
            _ = await self.evaluateStop(
                stopHookActive: false, chatID: targetID, project: targetProject)
        }
    }

    /// Stops the turn: the token stream, the agent loop, and any tool the
    /// loop is currently running.
    ///
    /// **`runTask` ALONE DOES NOT REACH THE LOOP.** An approved pending call
    /// runs in `toolExecutionTask`, spawned from the approval card after the
    /// proposing turn's stream already ended, so cancelling `runTask` there
    /// cancels a task that has nothing left to do. `isCancellationPending` is
    /// the flag `continueAgentLoop` reads: cancellation is cooperative, so a
    /// tool already in flight finishes, and what Stop guarantees is that no
    /// FURTHER model turn or tool call is started.
    public func cancel() {
        if audioSummaryInFlight { audioWorkspace.cancel(); return }
        guard canCancel else { return }
        if imageGenerationTask != nil {
            isCancellationPending = true
            cancelImageGeneration()
            return
        }
        isCancellationPending = true
        // **A PENDING CALL IS DENIED, NOT DROPPED** (state#34). Clearing the
        // four fields alone left the persisted proposal at
        // `.pendingApproval` forever: the card rendered as still awaiting a
        // decision across relaunches, with nothing left able to answer it.
        if let batchCalls = pendingBatchCalls, batchCalls.count > 1 {
            let chatID = pendingToolCallChatID ?? selectedChatID
            for var batchCall in batchCalls {
                batchCall.status = .denied
                appendToolExecutionTurn(
                    call: batchCall,
                    result: AppToolResult(
                        callID: batchCall.id, output: TOOL_REJECTED_MESSAGE,
                        isError: true, durationSeconds: 0.0),
                    chatID: chatID)
            }
        } else if let pending = pendingToolCall {
            let chatID = pendingToolCallChatID ?? selectedChatID
            var stopped = pending
            stopped.status = .denied
            let result = AppToolResult(
                callID: pending.id,
                output: TOOL_REJECTED_MESSAGE,
                isError: true,
                durationSeconds: 0.0
            )
            appendToolExecutionTurn(call: stopped, result: result, chatID: chatID)
        }
        clearPendingToolCall()
        // No direct `session?.cancel()`: the session is shared with background
        // agents, /recap and memory capture, and a session-wide cancel lands
        // on whichever of them holds the serial queue (ending a subagent as
        // "cancelled" while the user's own queued prompt never starts).
        // Cancelling `runTask` / `toolExecutionTask` ends the consumed stream,
        // and `QueuedGenerationStream.Request` then cancels only the request
        // this turn owns (in Rust if running, or before it ever starts).
        runTask?.cancel()
        toolExecutionTask?.cancel()
        toolExecutionTask = nil
        // The submission window has no `runTask` to reach (state#33).
        submissionTask?.cancel()
        submissionTask = nil
        // Nothing downstream will run a tail to clear the flag when the only
        // thing stopped was an approval card, and a latched one refuses every
        // later `continueAgentLoop`.
        if !generating && !submitting {
            isCancellationPending = false
        }
    }

    /// Clears every field that describes the call awaiting approval.
    ///
    /// One helper rather than four sites: `cancel()` used to null the call and
    /// leave `pendingToolCallChatID` / `pendingToolCallStep` / the captured
    /// project behind, and approve never reset the step -- stale values that
    /// the NEXT pending call reads if anything fails to overwrite them.
    func clearPendingToolCall() {
        pendingToolCall = nil
        pendingToolCallChatID = nil
        pendingToolCallStep = 0
        pendingToolCallProject = nil
        pendingBatchCalls = nil
        pendingValidationCall = nil
        pendingToolCallClassifierNotice = nil
    }

    /// The Stop All command: the turn (`cancel()`), every running background
    /// agent, every running background shell, and an in-flight model
    /// install. The local server and the loaded model are deliberately NOT
    /// here -- they are explicit toggles a user starts on purpose, and
    /// stopping either from a "stop everything that is hung" gesture would
    /// surprise more than it rescues.
    ///
    /// The agent arm is `stopBackgroundAgent`'s insert-and-cancel shape
    /// inlined: that method is async and returns a model-facing string, and
    /// the completion watcher turns each cancelled run into a `killed`
    /// notification either way.
    public func stopAll() {
        cancel()
        cancelRunningBackgroundAgents()
        killAllBackgroundShells()
        if hasActiveModelDownload {
            cancelActiveModelDownload()
        }
    }

    /// The agent arm of Stop All: every running background agent is marked
    /// killed and its task cancelled, so the completion watcher reports it as
    /// `killed`. The task entry stays; that watcher removes it.
    func cancelRunningBackgroundAgents() {
        for (id, state) in backgroundAgentRuns where state.status == "running" {
            killedBackgroundAgentIDs.insert(id)
            backgroundAgentTasks[id]?.cancel()
        }
    }

    /// Cancels the work that keeps the chat session alive without being a
    /// turn, which `cancel()` deliberately does not reach (see its note):
    /// title generation, memory capture and every running background agent.
    /// Each holds the session strongly until it ends, so releasing the app's
    /// own reference frees nothing while one of them runs (the drive stays
    /// busy). Background agents go through the Stop All path, not a new kill.
    ///
    /// Returns true while any of it is still running, so a caller can poll
    /// and cancel again: a cancelled title task starts the next queued one
    /// from its tail.
    @discardableResult
    func cancelSessionBoundWork() -> Bool {
        titleGenerationTask?.cancel()
        memoryCaptureTask?.cancel()
        cancelRunningBackgroundAgents()
        return titleGenerationTask != nil || memoryCaptureTask != nil || !backgroundAgentTasks.isEmpty
    }

    /// One step of stopping everything that holds the chat session: the turn
    /// if one is started (`cancel()`), then title generation, memory capture
    /// and the background agents. Returns true while any of it is still
    /// running, so a caller polls it until it is quiet.
    ///
    /// The turn counts too: a turn that ends can be replaced in the same hop
    /// by the next one (a queued prompt, a parked notification), and a caller
    /// that waited for `generating` to drop once saw the replacement as the
    /// old turn still winding down.
    @discardableResult
    func cancelTurnAndSessionBoundWork() -> Bool {
        if generating || submitting || pendingToolCall != nil { cancel() }
        let helpersRunning = cancelSessionBoundWork()
        return helpersRunning || generating || submitting || pendingToolCall != nil
    }

    /// Whether a release for an eject is in progress. While it is, nothing may
    /// start a model turn on the chat session (`canInjectChatInput`).
    var isEjecting: Bool { ejectHoldCount > 0 }

    /// Counted, not a flag: a surprise removal can release while a requested
    /// eject is under way. Balanced by the coordinator on every exit.
    func beginEjectHold() {
        ejectHoldCount += 1
    }

    func endEjectHold() {
        ejectHoldCount = max(0, ejectHoldCount - 1)
    }
}
