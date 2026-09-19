import Foundation
import TurboSpark

/// The goal loop: `/goal` handling, the stop-seam evaluation, deferral and
/// idle check-ins, and the write-through persistence behind the published
/// `activeGoals` mirror.
///
/// Claude Code reference: the goal is a session-scoped prompt-based Stop
/// hook plus an idle check-in timer (`tengu_joyful_globe`). The stop seam
/// here is `dispatchStopAndContinueIfBlocked`, which every turn end (steps
/// exhausted or prose reply) already funnels through -- this file's
/// `handleGoalAtStop` runs before the user's own Stop hooks, exactly where
/// CC's goal hook sits among them. The pure decisions live in
/// `ChatGoal.swift` / `GoalEvaluator.swift`; this file only moves state.
extension AppModel {
    // MARK: - Command handling

    /// The spellings `/goal <one of these>` treats as "clear the goal"
    /// rather than as a new condition (CC's own subcommand list).
    static let goalClearAliases: Set<String> = ["clear", "stop", "off", "reset", "none", "cancel"]

    /// The `/goal` meta command: no argument shows a status toast, a clear
    /// alias clears, anything else SETS the goal and then feeds the
    /// condition through the ordinary submission path as the directive
    /// turn (CC: setting a goal immediately starts a turn on it).
    func handleGoalCommand(_ draft: String) {
        let chatID = selectedChatID
        let argument = draft.dropFirst("/goal".count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if argument.isEmpty {
            if let goal = activeGoals[chatID] {
                showToast(
                    "Goal active (\(goal.iterations) iteration\(goal.iterations == 1 ? "" : "s"), "
                        + Self.formatGoalElapsed(since: goal.setAt) + "): "
                        + GoalPolicy.conditionPreview(goal.condition),
                    style: .info)
            } else {
                showToast("No goal set. Use /goal <condition> to set one.", style: .info)
            }
            return
        }
        if Self.goalClearAliases.contains(argument.lowercased()) {
            if activeGoals[chatID] != nil {
                clearGoal(chatID: chatID)
                showToast("Goal cleared.", style: .success)
            } else {
                showToast("No goal set.", style: .info)
            }
            return
        }
        guard argument.count <= GoalPolicy.maxConditionLength else {
            error = "A goal condition is limited to \(GoalPolicy.maxConditionLength) "
                + "characters; this one is \(argument.count)."
            return
        }
        guard session != nil else {
            error = "Load a model before setting a goal."
            return
        }
        setGoal(condition: argument, chatID: chatID)
        // The condition itself is the directive turn, dispatched through
        // `run()` like any prompt: mentions resolve, the UserPromptSubmit
        // hook is consulted, and a busy chat queues it for the tail rather
        // than jumping the work already running.
        // Programmatic expansion of /goal, not a paste.
        writePromptTextDirectly(argument)
        run()
    }

    // MARK: - State access and write-through

    /// A chat's goal state, decrypting from the vault when ghost. What the
    /// banner and the reminder assembly read.
    func storedGoal(chatIndex: Int) -> ChatGoalState? {
        let chat = chats[chatIndex]
        if chat.isGhost {
            return ghostVault.payload(for: chat.id).goal
        }
        return chat.goal
    }

    /// Writes goal state through the same ghost-aware primitive as every
    /// other transcript mutation, and keeps the published mirror in step.
    func updateGoal(_ goal: ChatGoalState?, chatID: UUID) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return }
        activeGoals[chatID] = goal
        if chats[index].isGhost {
            mutateGhostPayload(for: chatID) { $0.goal = goal }
        } else {
            chats[index].goal = goal
            chats[index].updatedAt = Date()
            persistChats()
        }
    }

    /// Sets a fresh goal (replacing any active one) and primes the
    /// evaluation bookkeeping: the transcript slice the first evaluation
    /// reads starts HERE, so the directive turn's own reply is inside it.
    func setGoal(condition: String, chatID: UUID) {
        let trimmed = condition.trimmingCharacters(in: .whitespacesAndNewlines)
        let goal = ChatGoalState(
            condition: String(trimmed.prefix(GoalPolicy.maxConditionLength)))
        updateGoal(goal, chatID: chatID)
        goalEvalMessageCounts[chatID] = turnMessages(for: chatID).count
        goalToolFreeEvals[chatID] = 0
    }

    /// Clears the goal and every runtime piece of it. `unrecoverable` adds
    /// the CC warning: only an unavailable model (or, per the docs, credit
    /// or context exhaustion this app cannot hit) clears a goal on its own.
    func clearGoal(chatID: UUID, unrecoverable: Bool = false) {
        cancelGoalIdleTimer(chatID: chatID)
        goalEvalMessageCounts[chatID] = nil
        goalToolFreeEvals[chatID] = nil
        updateGoal(nil, chatID: chatID)
        if unrecoverable {
            showToast(
                "Goal cleared after an unrecoverable error: the model was unloaded. "
                    + "Run /goal again to continue.",
                style: .warning)
        }
    }

    /// The chat-teardown form: the row is going away, so there is nothing
    /// to persist -- drop the mirror, the timer and the bookkeeping.
    func teardownGoal(chatID: UUID) {
        cancelGoalIdleTimer(chatID: chatID)
        activeGoals[chatID] = nil
        goalEvalMessageCounts[chatID] = nil
        goalToolFreeEvals[chatID] = nil
    }

    /// Hydrates the published mirror from freshly loaded rows and resets
    /// everything runtime-shaped (CC's resume rule: condition and set time
    /// survive; counters, timers and baselines do not).
    func restoreGoalsFromRows() {
        activeGoals = [:]
        goalEvalMessageCounts = [:]
        goalToolFreeEvals = [:]
        for chat in chats where !chat.isGhost {
            guard let goal = chat.goal else { continue }
            activeGoals[chat.id] = goal.restoredForRelaunch()
            goalEvalMessageCounts[chat.id] = chat.messages.count
            goalToolFreeEvals[chat.id] = 0
        }
    }

    /// A user prompt resets everything that is "per goal between prompts":
    /// the idle check-in cap and a stall pause. CC's own resume rule --
    /// evaluation resumes after your next prompt.
    func resetGoalForUserPrompt(chatID: UUID) {
        guard var goal = activeGoals[chatID] else { return }
        goal.idleCheckinCount = 0
        goal.isPaused = false
        updateGoal(goal, chatID: chatID)
        armGoalIdleTimer(chatID: chatID)
    }

    /// Folds one turn's authoritative token count into the goal's display
    /// ledger. Called from the `.finished` fold; zero-token turns write
    /// nothing.
    func recordGoalTurnTokens(chatID: UUID, tokens: Int) {
        guard tokens > 0, var goal = activeGoals[chatID] else { return }
        goal.tokensSpent += tokens
        updateGoal(goal, chatID: chatID)
    }

    // MARK: - Background work inventory

    /// Everything running for this chat that defers evaluation, flattened
    /// out of the two registries (the same enumeration shape
    /// `stopBackgroundWork(forDeletedChat:)` uses). Oldest first.
    func runningGoalTasks(chatID: UUID) -> [GoalBackgroundTask] {
        var tasks: [GoalBackgroundTask] = backgroundAgentRuns.values
            .filter { $0.status == "running" && $0.chatID == chatID }
            .map { state in
                GoalBackgroundTask(
                    id: state.id,
                    kind: .agent,
                    label: state.displayName.isEmpty ? state.agentName : state.displayName,
                    detail: state.taskDescription.isEmpty ? state.promptHead : state.taskDescription,
                    startedAt: state.startedAt)
            }
        tasks += BackgroundShellManager.shared.runningRecords(chatID: chatID).map { record in
            let detail = record.displayDescription ?? record.command
            return GoalBackgroundTask(
                id: record.id,
                kind: .shell,
                label: "shell",
                detail: String(detail.prefix(120)),
                startedAt: record.startedAt)
        }
        return tasks.sorted { $0.startedAt < $1.startedAt }
    }
}
