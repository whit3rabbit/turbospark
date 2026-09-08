import Foundation
import TurboSpark

/// The goal LOOP proper: the stop-seam evaluation, the deferral and
/// check-in arms, the evaluator query, and the idle check-in timer.
/// Split from `AppModel+Goal.swift` (the `/goal` command, the state
/// access and the persistence) to keep both under the 400-line
/// guideline; `swift/docs/SWIFT_GOALS.md` is the spec both halves
/// implement.
extension AppModel {
    // MARK: - The stop seam

    /// The goal's turn-end evaluation. Returns TRUE when it continued the
    /// loop itself (a not-met verdict or a due check-in re-entered the
    /// agent loop), which tells `dispatchStopAndContinueIfBlocked` to stop
    /// without consulting the user's own Stop hooks -- a goal round is not
    /// a stop. False falls through to the ordinary Stop path.
    ///
    /// Order inside, CC's order: deferral first (background work running
    /// means NO evaluation, only maybe a check-in), then the judge.
    func handleGoalAtStop(chatID: UUID, project: AppProject?) async -> Bool {
        guard var goal = activeGoals[chatID], !goal.isPaused, !isCancellationPending
        else { return false }
        let tasks = runningGoalTasks(chatID: chatID)
        let now = Date()
        if !tasks.isEmpty {
            return await deferOrCheckin(
                goal: goal, tasks: tasks, chatID: chatID, project: project, now: now)
        }
        // Nothing defers the goal anymore: strip the deferral state, then
        // judge. (The timer, if armed, is cancelled with the state.)
        if goal.deferredSince != nil {
            goal.deferredSince = nil
            goal.checkinCount = 0
            goal.lastDeferralPassAt = nil
            updateGoal(goal, chatID: chatID)
        }
        cancelGoalIdleTimer(chatID: chatID)

        // Stall detection: the evaluator only ever sees what the
        // conversation surfaces, and a model replying in prose forever
        // would burn tokens circling. `GoalPolicy.stallThreshold`
        // consecutive tool-free evaluation rounds pause the goal; the next
        // user prompt lifts the pause.
        let messages = turnMessages(for: chatID)
        let sliceStart = min(goalEvalMessageCounts[chatID] ?? messages.count, messages.count)
        let slice = messages[sliceStart...]
        goalEvalMessageCounts[chatID] = messages.count
        let usedTools = slice.contains { !$0.toolCalls.isEmpty || !$0.toolResults.isEmpty }
        goalToolFreeEvals[chatID] = usedTools ? 0 : (goalToolFreeEvals[chatID] ?? 0) + 1
        if (goalToolFreeEvals[chatID] ?? 0) >= GoalPolicy.stallThreshold {
            goal.isPaused = true
            updateGoal(goal, chatID: chatID)
            appendGoalStatusRow(GoalEvaluator.stallRow, chatID: chatID)
            return false
        }

        // An unloaded model can never evaluate again: that is CC's
        // unrecoverable error, and the one case this app has.
        guard session != nil else {
            clearGoal(chatID: chatID, unrecoverable: true)
            return false
        }
        guard let verdict = await runGoalEvaluation(chatID: chatID, condition: goal.condition)
        else { return false }
        switch verdict {
        case .met(let reason):
            clearGoal(chatID: chatID)
            appendGoalStatusRow(
                GoalEvaluator.statusRow(for: .met(reason: reason), iterations: goal.iterations + 1),
                chatID: chatID)
            // A met goal is a real stop: the user's own Stop hooks still
            // get consulted about the turn that achieved it.
            return false
        case .impossible(let reason):
            clearGoal(chatID: chatID)
            appendGoalStatusRow(
                GoalEvaluator.statusRow(for: .impossible(reason: reason), iterations: goal.iterations),
                chatID: chatID)
            return false
        case .notMet(let reason):
            goal.iterations += 1
            goal.lastReason = reason
            updateGoal(goal, chatID: chatID)
            appendGoalStatusRow(
                GoalEvaluator.statusRow(for: verdict, iterations: goal.iterations - 1),
                chatID: chatID)
            // Fresh step allowance: the goal is standing user intent, and
            // the evaluation round is the unit of work now -- the loop had
            // already spent its `maxAutonomousSteps` getting here.
            continueAgentLoop(step: 0, chatID: chatID)
            return true
        }
    }

    /// The deferral arm: background work is running, so the goal is NOT
    /// evaluated on this pass. A due check-in is injected and the loop
    /// continues on it; otherwise the turn just ends and the idle timer
    /// carries the waiting.
    private func deferOrCheckin(
        goal: ChatGoalState, tasks: [GoalBackgroundTask], chatID: UUID,
        project: AppProject?, now: Date
    ) async -> Bool {
        var goal = goal
        let pass = GoalPolicy.deferralPass(goal: goal, tasks: tasks, now: now)
        goal.deferredSince = pass.deferredSince
        goal.checkinCount = pass.checkinCount
        goal.lastDeferralPassAt = now
        if pass.checkinDue {
            goal.checkinCount += 1
            goal.deferredSince = now
            updateGoal(goal, chatID: chatID)
            appendGoalCheckinRow(
                chatID: chatID,
                text: GoalPolicy.checkinMessage(
                    condition: goal.condition, tasks: tasks, announcingPause: false))
            // The check-in is a fresh turn (step 0): it is not a step of
            // the round that ended.
            continueAgentLoop(step: 0, chatID: chatID)
            armGoalIdleTimer(chatID: chatID)
            return true
        }
        updateGoal(goal, chatID: chatID)
        armGoalIdleTimer(chatID: chatID)
        return false
    }

    /// Appends one `<goal_checkin>` turn as the task-notification path
    /// does: a real (visible) user row, because it is prompt content the
    /// turn it starts must answer.
    private func appendGoalCheckinRow(chatID: UUID, text: String) {
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(role: .user, content: text))
        }
    }

    /// The compact verdict row: an assistant-role status note (the same
    /// shape as the guardrail-retry row). The MODEL also reads it back as
    /// its own transcript, which is fine -- the authoritative reason rides
    /// the assembly-time `<goal>` reminder regardless.
    func appendGoalStatusRow(_ text: String, chatID: UUID) {
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(role: .assistant, content: text, stopReason: "goal_check"))
        }
    }

    // MARK: - The evaluator query

    /// One judge query, with a 30 s timeout. Any failure -- timeout,
    /// cancellation, malformed output -- returns nil, which the caller
    /// treats as TRANSIENT: the goal stays, the turn ends, the next stop
    /// re-evaluates.
    private func runGoalEvaluation(chatID: UUID, condition: String) async -> GoalVerdict? {
        guard let session else { return nil }
        let messages = GoalEvaluator.judgeMessages(
            condition: condition, transcript: goalTranscriptTail(chatID: chatID))
        var options = GenerateOptions()
        options.reasoning = .off
        options.temperature = GoalPolicy.evaluatorTemperature
        options.maxNewTokens = UInt32(GoalPolicy.evaluatorMaxNewTokens)
        isEvaluatingGoal = true
        defer { isEvaluatingGoal = false }
        return await withTaskGroup(of: GoalVerdict?.self) { group in
            group.addTask {
                var text = ""
                do {
                    for try await event in session.generate(messages, options: options) {
                        try Task.checkCancellation()
                        if case .content(let chunk) = event { text += chunk }
                    }
                } catch {
                    return nil
                }
                return GoalEvaluator.parseVerdict(text)
            }
            group.addTask {
                try? await Task.sleep(
                    nanoseconds: UInt64(GoalPolicy.evaluatorTimeoutSeconds * 1_000_000_000))
                return nil
            }
            let first = await group.next() ?? nil
            group.cancelAll()
            return first
        }
    }

    /// The conversation slice the evaluator judges: everything since the
    /// last evaluation (or the goal's set point), falling back to the tail
    /// when the slice is somehow empty. Rendered in the same bracketed
    /// form compaction summarizes from.
    private func goalTranscriptTail(chatID: UUID) -> String {
        let messages = turnMessages(for: chatID)
        let sliceStart = min(goalEvalMessageCounts[chatID] ?? messages.count, messages.count)
        var slice = Array(messages[sliceStart...])
        if slice.isEmpty {
            slice = Array(messages.suffix(GoalPolicy.evaluatorTailMessages))
        }
        return AppChatCompaction.renderTranscript(slice)
    }

    // MARK: - Idle check-in timer

    /// (Re)arms the chat's idle check-in timer from the goal's current
    /// deferral state. A no-op when nothing defers: the timer only exists
    /// to wake a QUIET deferral up -- the turn-end seam covers every other
    /// moment.
    func armGoalIdleTimer(chatID: UUID) {
        goalIdleTimerTasks[chatID]?.cancel()
        goalIdleTimerTasks[chatID] = nil
        guard let goal = activeGoals[chatID], goal.deferredSince != nil, !goal.isPaused
        else { return }
        let pass = GoalPolicy.deferralPass(
            goal: goal, tasks: runningGoalTasks(chatID: chatID), now: Date())
        let delay = GoalPolicy.idleTimerDelay(pass: pass, now: Date())
        goalIdleTimerTasks[chatID] = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
            guard !Task.isCancelled else { return }
            self?.fireGoalIdleCheckin(chatID: chatID)
        }
    }

    func cancelGoalIdleTimer(chatID: UUID) {
        goalIdleTimerTasks[chatID]?.cancel()
        goalIdleTimerTasks[chatID] = nil
    }

    /// The idle timer fired. Delivery rules, CC's: the chat must be idle
    /// end to end (else retry shortly); the idle cap (3 per goal between
    /// prompts) makes the tick re-arm WITHOUT injecting once reached; the
    /// delivery that reaches the cap is the one that announces the pause.
    /// Deferring work that DRAINED while quiet gets the "no longer
    /// running" check-in, because nothing else will ever start that turn
    /// (shells have no completion notification; agents do).
    private func fireGoalIdleCheckin(chatID: UUID) {
        goalIdleTimerTasks[chatID] = nil
        guard var goal = activeGoals[chatID], !goal.isPaused else { return }
        let tasks = runningGoalTasks(chatID: chatID)
        guard canInjectTaskNotification(into: chatID) else {
            armGoalIdleTimer(chatID: chatID)
            return
        }
        if tasks.isEmpty {
            guard goal.deferredSince != nil else { return }
            goal.deferredSince = nil
            goal.checkinCount = 0
            goal.lastDeferralPassAt = nil
            goal.idleCheckinCount += 1
            updateGoal(goal, chatID: chatID)
            appendGoalCheckinRow(
                chatID: chatID,
                text: GoalPolicy.checkinMessage(
                    condition: goal.condition, tasks: [],
                    announcingPause: goal.idleCheckinCount >= GoalPolicy.maxIdleCheckins))
            executeGenerationTurn(step: 0, chatID: chatID)
            return
        }
        if !goal.canIdleCheckin {
            armGoalIdleTimer(chatID: chatID)
            return
        }
        let now = Date()
        let pass = GoalPolicy.deferralPass(goal: goal, tasks: tasks, now: now)
        guard pass.checkinDue else {
            armGoalIdleTimer(chatID: chatID)
            return
        }
        goal.idleCheckinCount += 1
        goal.checkinCount = pass.checkinCount + 1
        goal.deferredSince = now
        goal.lastDeferralPassAt = now
        updateGoal(goal, chatID: chatID)
        appendGoalCheckinRow(
            chatID: chatID,
            text: GoalPolicy.checkinMessage(
                condition: goal.condition, tasks: tasks,
                announcingPause: goal.idleCheckinCount >= GoalPolicy.maxIdleCheckins))
        executeGenerationTurn(step: 0, chatID: chatID)
    }

    // MARK: - Presentation helpers

    /// "3m" / "2h 15m" elapsed form, shared by the toast and the banner.
    /// `nonisolated`: pure date math that a test can call off the actor.
    nonisolated static func formatGoalElapsed(
        since date: Date, now: Date = Date()
    ) -> String {
        let minutes = max(0, Int(now.timeIntervalSince(date) / 60))
        if minutes < 60 { return "\(minutes)m" }
        let hours = minutes / 60
        let remainder = minutes % 60
        return remainder == 0 ? "\(hours)h" : "\(hours)h \(remainder)m"
    }
}
