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
    func finishCancelled(chatID: UUID? = nil, reason: String = "cancelled") {
        let targetID = chatID ?? selectedChatID
        let targetProject = turnProject(chatID: targetID)
        if !outputText.isEmpty || !outputReasoningText.isEmpty {
            // A Retry/Edit turn stopped mid-flight still owes its variants
            // an answer: the partial reply becomes the active version (its
            // stop reason marks it as unfinished) and the replaced one is
            // kept beside it. Consumed here for the same once-only reason
            // `finishProseTurn` consumes it.
            let parkedVariants =
                self.pendingResponseVariants.removeValue(forKey: targetID) ?? []
            mutateTurnMessages(for: targetID) {
                var message = AppChatMessage(
                    role: .assistant,
                    content: self.outputText,
                    reasoning: self.outputReasoningText,
                    stopReason: reason
                )
                if !parkedVariants.isEmpty {
                    message.alternates = parkedVariants.map { variant in
                        var flat = variant
                        flat.alternates = []
                        return flat
                    }
                }
                $0.append(message)
            }
            outputText = ""
            outputReasoningText = ""
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
        guard canCancel else { return }
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
        session?.cancel()
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
        for (id, state) in backgroundAgentRuns where state.status == "running" {
            killedBackgroundAgentIDs.insert(id)
            backgroundAgentTasks[id]?.cancel()
        }
        killAllBackgroundShells()
        if isInstallingModel {
            cancelInstall()
        }
    }
}
