import Foundation
import TurboSpark

struct PendingTaskNotification: Equatable, Sendable {
    let kind: MidTurnInputPresentation
    let note: String
}

/// What `send_message` needs to run a finished background child again: the
/// launch parameters it ran under (so the continuation keeps the same agent
/// definition, project, depth and permission gate) and the transcript its last
/// run ended with. In memory only, and pruned with the run's record.
struct RetainedBackgroundChild {
    let launch: BackgroundAgentLaunch
    var transcript: [ChatMessage]?
}

/// Everything needed to launch one background subagent, carried across the
/// provider boundary from `AppToolRegistry` into `AppModel`.
///
/// `@unchecked Sendable` for the same reason `BackgroundShellRecord` is: the
/// members are app value types this module controls, and boxing them in one
/// class is how the launcher closure stays a single sendable value.
public final class BackgroundAgentLaunch: @unchecked Sendable {
    public let agent: AppAgentDefinition
    public let taskPrompt: String
    public let taskDescription: String
    public let project: AppProject?
    public let chatID: UUID?
    public let depth: Int
    public let userSystemPrompt: String
    public let samplingOptions: GenerateOptions

    public init(
        agent: AppAgentDefinition, taskPrompt: String, taskDescription: String,
        project: AppProject?, chatID: UUID?, depth: Int, userSystemPrompt: String,
        samplingOptions: GenerateOptions = GenerateOptions()
    ) {
        self.agent = agent
        self.taskPrompt = taskPrompt
        self.taskDescription = taskDescription
        self.project = project
        self.chatID = chatID
        self.depth = depth
        self.userSystemPrompt = userSystemPrompt
        self.samplingOptions = samplingOptions
    }
}

/// Escapes any occurrence of an opening or closing `<tagName>` for each name
/// in `tagNames`, case-insensitively and tolerant of whitespace around the
/// name (`</ Result >`, `<TASK-NOTIFICATION>`), so a subagent's own generated
/// text cannot splice a forged close into a wrapper the parent model is
/// meant to read as trusted run metadata.
///
/// Plain `.replacingOccurrences(of:with:)` against exact-case literal
/// strings -- what this used to be -- only ever catches the one spelling it
/// names; a subagent's answer containing `</Result>` or `</ result >` passed
/// through unescaped, which is exactly the gap a wrapper like this exists to
/// close. Only the `<`/`>` characters of a MATCHED tag are replaced, so the
/// visible wording (case, internal spacing) is otherwise untouched.
func escapingWrapperTags(_ tagNames: [String], in text: String) -> String {
    guard !tagNames.isEmpty, !text.isEmpty else { return text }
    let namePattern = tagNames.map { NSRegularExpression.escapedPattern(for: $0) }.joined(separator: "|")
    guard let regex = try? NSRegularExpression(
        pattern: "<\\s*/?\\s*(?:\(namePattern))\\s*>", options: [.caseInsensitive]
    ) else {
        return text
    }
    let nsText = text as NSString
    let fullRange = NSRange(location: 0, length: nsText.length)
    var result = ""
    var lastEnd = 0
    regex.enumerateMatches(in: text, range: fullRange) { match, _, _ in
        guard let match else { return }
        let range = match.range
        result += nsText.substring(with: NSRange(location: lastEnd, length: range.location - lastEnd))
        let matched = nsText.substring(with: range)
        result += matched
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
        lastEnd = range.location + range.length
    }
    result += nsText.substring(from: lastEnd)
    return result
}

/// Formats the `<task-notification>` user turn a background subagent's
/// completion injects into its chat (Claude Code's task-notification shape,
/// reduced to what this app has: no output file, no usage block). Pure, so
/// the wording is pinned by test rather than by reading a live run.
enum BackgroundAgentNotification {
    static func text(
        id: String, status: String, agentName: String, displayName: String,
        result: SubagentRunResult
    ) -> String {
        let rawBody = result.finalResponse.trimmingCharacters(in: .whitespacesAndNewlines)
        let safeBody = escapingWrapperTags(["result", "task-notification"], in: rawBody)
        return """
        <task-notification>
        task_id: \(id)
        status: \(status)
        agent: \(displayName) (\(agentName))
        <result>
        \(safeBody.isEmpty ? "(No output.)" : safeBody)
        </result>
        turns: \(result.totalTurns), tool_calls: \(result.totalToolCalls), duration_s: \(String(format: "%.1f", result.durationSeconds))
        </task-notification>
        """
    }
}

extension AppModel {
    /// Ceiling on concurrently RUNNING background agents. Each one shares
    /// the single engine session with the main chat, so every running agent
    /// makes every other generation wait longer for the session queue; four
    /// interleaved conversations is already past the point where a fifth
    /// makes anything finish sooner.
    static let maxRunningBackgroundAgents = 4
    /// Finished background records kept for their cards and results; the
    /// oldest fall off the end (same shape as `BackgroundShellManager`).
    static let maxFinishedBackgroundAgents = 20

    // MARK: - Progress routing

    /// Routes one `SubagentProgressEvent` to the surface hosting its run.
    /// Installed as `AppToolRegistry.subagentProgressSink` at startup.
    ///
    /// Foreground runs are created lazily by their first event (the
    /// runner's `.started`, which carries the run's own chat) and removed
    /// on `.finished`, because the transcript's tool card takes over. A
    /// background run's record is created at LAUNCH and never removed by an
    /// event -- its card is the only representation it has.
    func applySubagentEvent(_ key: String, _ event: SubagentProgressEvent) {
        if backgroundAgentRuns[key] != nil {
            backgroundAgentRuns[key]?.apply(event)
            return
        }
        switch event {
        case .started(_, _, _, _, let chatID):
            guard liveSubagentRuns[key] == nil else {
                liveSubagentRuns[key]?.apply(event)
                return
            }
            let state = SubagentRunState(
                id: key, mode: .foreground, chatID: chatID)
            state.apply(event)
            liveSubagentRuns[key] = state
        case .finished:
            liveSubagentRuns[key]?.apply(event)
            liveSubagentRuns.removeValue(forKey: key)
        default:
            liveSubagentRuns[key]?.apply(event)
        }
    }

    // MARK: - Background launch

    /// Launches a background subagent and returns the immediate tool output
    /// naming its task id. The run happens in an unstructured `Task` that
    /// nothing else cancels; completion is observed by a watcher that
    /// injects the `<task-notification>` turn.
    ///
    /// **THE SESSION IS CAPTURED, NOT RE-READ.** The user may unload the
    /// model while the agent runs; holding the session here is what lets
    /// the run finish on the engine it started on instead of dying with the
    /// UI's notion of what is loaded. The chat's own Stop cannot reach the
    /// task either: it cancels `runTask`, and an unstructured task inherits
    /// no cancellation -- `stopBackgroundAgent` is the only kill path.
    func launchBackgroundAgent(_ launch: BackgroundAgentLaunch) async throws -> String {
        let running = backgroundAgentRuns.values.filter { $0.status == "running" }.count
        guard running < Self.maxRunningBackgroundAgents else {
            throw NSError(domain: "TurboSparkTool", code: 41, userInfo: [
                NSLocalizedDescriptionKey: "Too many background subagents are already running "
                    + "(\(running) of \(Self.maxRunningBackgroundAgents)). Wait for one to finish or "
                    + "stop it with stop_agent before launching another."
            ])
        }
        // The runner's own test seam counts as a model: it is the same port
        // the production session is wrapped in.
        guard session != nil || SubagentGenerationPortContext.current != nil else {
            throw NSError(domain: "TurboSparkTool", code: 26, userInfo: [
                NSLocalizedDescriptionKey: "No active model session to run the background subagent "
                    + "on. Load a model first."
            ])
        }
        let id = "bga_\(nextBackgroundAgentID)"
        nextBackgroundAgentID += 1

        let state = SubagentRunState(
            id: id, mode: .background, chatID: launch.chatID,
            agentName: launch.agent.name, displayName: launch.agent.displayName,
            taskDescription: launch.taskDescription,
            promptHead: headOfPrompt(launch.taskPrompt))
        backgroundAgentRuns[id] = state

        // Retained for `send_message`: what is needed to run this child again.
        // The transcript arrives with the result.
        retainedBackgroundChildren[id] = RetainedBackgroundChild(launch: launch, transcript: nil)
        startBackgroundRun(
            id: id, launch: launch, session: session, taskPrompt: launch.taskPrompt, prior: [])

        return """
            Background subagent launched; it does not block this conversation.

            id: \(id)
            agent: \(launch.agent.displayName) (\(launch.agent.name))
            task: \(launch.taskDescription.isEmpty ? state.promptHead : launch.taskDescription)

            You will receive a <task-notification> message here when it completes. Do not \
            wait for it; continue with other work or end your turn. Stop it with stop_agent \
            using the id above.
            """
    }

    /// Starts (or restarts, for `send_message`) the run behind a `bga_N` id and
    /// the watcher that delivers its completion.
    ///
    /// **`prior` IS NEVER NIL.** A non-nil `priorHistory` is what makes the
    /// runner hand its transcript back, and that transcript is the retained
    /// context a later `send_message` resumes from. A first launch passes an
    /// empty array, which leaves the run itself unchanged.
    ///
    /// The coordinator channel is bound here, as a task-local around the run:
    /// the child's `respond_to_coordinator` calls resolve their sender (this
    /// id) and chat from it, not from anything the model writes.
    private func startBackgroundRun(
        id: String, launch: BackgroundAgentLaunch, session: TurboSparkSession?,
        taskPrompt: String, prior: [ChatMessage]
    ) {
        let sink = AppToolRegistry.subagentProgressSink
        let agent = launch.agent
        let project = launch.project
        let chatID = launch.chatID
        let depth = launch.depth
        let userSystemPrompt = launch.userSystemPrompt
        let description = launch.taskDescription
        let options = launch.samplingOptions
        let channel = SubagentCoordinatorChannel(
            childID: id, chatID: chatID, agentName: agent.name, depth: depth)
        let runTask = Task<SubagentRunResult, Never> {
            await SubagentCoordinatorChannel.$current.withValue(channel) {
                await SubagentRunner.run(
                    agent: agent, taskPrompt: taskPrompt, session: session, project: project,
                    chatID: chatID, depth: depth, userSystemPrompt: userSystemPrompt,
                    samplingOptions: options,
                    progress: { event in await sink?(id, event) },
                    taskDescription: description,
                    priorHistory: prior)
            }
        }
        backgroundAgentTasks[id] = runTask

        Task { [weak self] in
            let result = await runTask.value
            await self?.completeBackgroundAgent(id: id, result: result)
        }
    }

    private func headOfPrompt(_ prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count > 300 else { return trimmed }
        return String(trimmed.prefix(300)) + "..."
    }

    // MARK: - Background completion and stopping

    /// Records a finished background run and injects its notification turn.
    func completeBackgroundAgent(id: String, result: SubagentRunResult) async {
        backgroundAgentTasks.removeValue(forKey: id)
        guard let state = backgroundAgentRuns[id] else { return }
        state.result = result
        // A stop request turns the runner's own `cancelled` exit into
        // `killed`, so the notification reads as what the user did.
        let wasKilled = killedBackgroundAgentIDs.remove(id) != nil
        let status = wasKilled && result.status == "cancelled"
            ? "killed" : result.status
        state.status = status
        // Context for `send_message`: kept only for a run that ended in a
        // shape worth continuing. A run that failed to start or overflowed its
        // window has nothing (or too much) to resume from.
        if let transcript = result.transcript,
            ["completed", "max_turns", "cancelled", "killed"].contains(status)
        {
            retainedBackgroundChildren[id]?.transcript = transcript
        }
        pruneFinishedBackgroundAgents()

        // Marked on every exit path from here, including the one below: once
        // this function has run, there is nothing further for it to deliver,
        // so the entry is exactly as "completed" whether or not the chat it
        // belonged to still exists. Doing this AFTER `pruneFinishedBackgroundAgents()`
        // matters only in that it must not be skipped by an early return
        // above this point -- there is none, but see `dismissBackgroundAgent`'s
        // own doc for why this flag exists rather than relying on `status`.
        defer { state.isRecordedComplete = true }

        guard let chatID = state.chatID, chats.contains(where: { $0.id == chatID }) else {
            // Nobody left to read what the child sent mid-task.
            subagentMessageHub.discard(childID: id)
            return
        }
        let note = BackgroundAgentNotification.text(
            id: id, status: status, agentName: state.agentName,
            displayName: state.displayName, result: result)
        let notification = PendingTaskNotification(kind: .taskNotification, note: note)
        if canInjectTaskNotification(into: chatID) {
            injectTaskNotification(notification, chatID: chatID)
        } else {
            pendingTaskNotifications[chatID, default: []].append(notification)
        }
    }

    /// Model-initiated stop: only agents launched from `callerChatID` are
    /// visible. An id owned by another chat is reported exactly like an
    /// unknown one, and the "Known ids" hint lists only the caller's own, so
    /// a chat can neither enumerate nor kill another chat's runs.
    func stopBackgroundAgent(_ id: String, callerChatID: UUID?) async throws -> String {
        guard let state = backgroundAgentRuns[id], state.chatID == callerChatID else {
            let known = backgroundAgentRuns.filter { $0.value.chatID == callerChatID }
                .keys.sorted().joined(separator: ", ")
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "No background subagent with id '\(id)'. "
                    + (known.isEmpty ? "None are registered." : "Known ids: \(known).")
            ])
        }
        return try await stopBackgroundAgent(id)
    }

    /// Stops a running background subagent by id. The runner notices at its
    /// next `Task.isCancelled` check, keeps whatever it had answered, and
    /// the completion path reports the run as killed.
    func stopBackgroundAgent(_ id: String) async throws -> String {
        guard let state = backgroundAgentRuns[id] else {
            let known = backgroundAgentRuns.keys.sorted().joined(separator: ", ")
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "No background subagent with id '\(id)'. "
                    + (known.isEmpty ? "None are registered." : "Known ids: \(known).")
            ])
        }
        guard state.status == "running", let task = backgroundAgentTasks[id] else {
            return "Background subagent '\(id)' has already finished (status: \(state.status))."
        }
        killedBackgroundAgentIDs.insert(id)
        task.cancel()
        return "Stop requested for background subagent '\(id)' (\(state.displayName)). It ends at "
            + "its next cancellation point and a <task-notification> with status killed will "
            + "follow."
    }

    // MARK: - Coordinator messaging

    /// `respond_to_coordinator`: queues a mid-task message from a RUNNING
    /// background child and delivers it to the chat that launched it.
    ///
    /// Delivery rides the `<task-notification>` queue: an idle coordinator is
    /// woken the same way a completion wakes it, a busy one reads the message
    /// at its next step boundary or at the turn tail. Nothing here interrupts
    /// a model call in flight.
    func postSubagentMessage(childID: String, chatID: UUID?, text: String) async throws -> String {
        guard let state = backgroundAgentRuns[childID], state.chatID == chatID,
            state.mode == .background, let chatID,
            chats.contains(where: { $0.id == chatID })
        else {
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "This run has no coordinator to message."
            ])
        }
        guard state.status == "running" else {
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "This run has already finished; its final answer is "
                    + "its report."
            ])
        }
        switch subagentMessageHub.post(
            childID: childID, chatID: chatID, agentName: state.agentName, text: text)
        {
        case .empty:
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "The message is empty."
            ])
        case .budgetExhausted:
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "Message budget reached (\(SubagentMessageHub.maxMessagesPerRun) "
                    + "per run). Keep working and put the rest in your final answer."
            ])
        case .queued(let truncated, let droppedOldest):
            drainPendingTaskNotificationsIfIdle(chatID: chatID)
            var reply = "Message queued for the coordinator; keep working."
            if truncated {
                reply += " It was cut to \(SubagentMessageHub.maxMessageBytes) bytes."
            }
            if droppedOldest {
                reply += " Your mailbox was full "
                    + "(\(SubagentMessageHub.maxQueuedPerChild) unread), so the oldest unread "
                    + "message was dropped. Send fewer, shorter messages."
            }
            return reply
        }
    }

    /// Appends any queued child messages to the chat as a user turn at a step
    /// boundary of a RUNNING agent loop, so the next model step reads them.
    /// Returns whether anything was appended. Refuses at an approval card or
    /// a pending cancel, for the same reason the steer boundary does.
    @discardableResult
    func deliverSubagentMessagesAtBoundary(chatID: UUID) -> Bool {
        guard pendingToolCall == nil, !isCancellationPending,
            subagentMessageHub.pendingCount(chatID: chatID) > 0
        else { return false }
        let batch = subagentMessageHub.drain(chatID: chatID)
        guard !batch.isEmpty else { return false }
        mutateTurnMessages(for: chatID) {
            $0.append(MidTurnInputPresentation.taskNotification.makeUserMessage(
                SubagentMessageHub.carrier(for: batch)))
        }
        return true
    }

    /// `send_message`: resumes a FINISHED background child with its retained
    /// context and a follow-up instruction.
    ///
    /// Ownership is the same wall `stop_agent` has: an id owned by another
    /// chat reads exactly like an unknown one. The resumed run is a background
    /// run again under the SAME agent definition, project, depth, tool filter
    /// and permission gate it launched with, and it counts against the
    /// running-agent cap, so continuing a child grants nothing a fresh launch
    /// would not.
    func resumeBackgroundAgent(_ id: String, message: String, callerChatID: UUID?) async throws -> String {
        guard let state = backgroundAgentRuns[id], state.chatID == callerChatID else {
            let known = backgroundAgentRuns.filter { $0.value.chatID == callerChatID }
                .keys.sorted().joined(separator: ", ")
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "No background subagent with id '\(id)'. "
                    + (known.isEmpty ? "None are registered." : "Known ids: \(known).")
            ])
        }
        guard state.status != "running" else {
            throw NSError(domain: "TurboSparkTool", code: 29, userInfo: [
                NSLocalizedDescriptionKey: "Background subagent '\(id)' is still running. Wait for "
                    + "its <task-notification>, or stop it with stop_agent."
            ])
        }
        guard state.isRecordedComplete else {
            throw NSError(domain: "TurboSparkTool", code: 29, userInfo: [
                NSLocalizedDescriptionKey: "Background subagent '\(id)' is still finishing. Try "
                    + "again after its <task-notification>."
            ])
        }
        guard let retained = retainedBackgroundChildren[id], let transcript = retained.transcript else {
            throw NSError(domain: "TurboSparkTool", code: 30, userInfo: [
                NSLocalizedDescriptionKey: "Background subagent '\(id)' cannot be continued: its "
                    + "context was not retained (it ended with status '\(state.status)', or the "
                    + "app restarted). Launch a new subagent and include what it needs in the prompt."
            ])
        }
        let followUp = message.trimmingCharacters(in: .whitespacesAndNewlines)
        guard followUp.utf8.count <= SubagentMessageHub.maxFollowUpBytes else {
            throw NSError(domain: "TurboSparkTool", code: 27, userInfo: [
                NSLocalizedDescriptionKey: "The message is over \(SubagentMessageHub.maxFollowUpBytes) "
                    + "bytes. Send a shorter instruction."
            ])
        }
        let running = backgroundAgentRuns.values.filter { $0.status == "running" }.count
        guard running < Self.maxRunningBackgroundAgents else {
            throw NSError(domain: "TurboSparkTool", code: 41, userInfo: [
                NSLocalizedDescriptionKey: "Too many background subagents are already running "
                    + "(\(running) of \(Self.maxRunningBackgroundAgents)). Wait for one to finish or "
                    + "stop it with stop_agent before continuing another."
            ])
        }
        // The runner's own test seam counts as a model: it is the same port
        // the production session is wrapped in.
        guard session != nil || SubagentGenerationPortContext.current != nil else {
            throw NSError(domain: "TurboSparkTool", code: 26, userInfo: [
                NSLocalizedDescriptionKey: "No active model session to run the background subagent "
                    + "on. Load a model first."
            ])
        }

        state.status = "running"
        state.result = nil
        state.isRecordedComplete = false
        state.turns = 0
        state.streamedText = ""
        state.toolRows = []
        state.promptHead = headOfPrompt(followUp)
        // The retained context stays with the record while it runs again; the
        // new result replaces it on completion.
        retainedBackgroundChildren[id]?.transcript = nil
        subagentMessageHub.resetBudget(childID: id)
        startBackgroundRun(
            id: id, launch: retained.launch, session: session,
            taskPrompt: "Follow-up from the coordinator:\n\(followUp)", prior: transcript)

        return """
            Background subagent '\(id)' (\(state.displayName)) resumed with its earlier context. \
            It runs in the background with the same tools and permissions as before; you will \
            receive a <task-notification> when it finishes. Stop it with stop_agent.
            """
    }

    // MARK: - Notification injection

    /// Whether an input may be injected into this chat now. A turn in flight
    /// parks it for the tail; so does an approval card, whose own approve
    /// path refuses to run while `generating` is true.
    ///
    /// **SO DOES A RELEASE FOR AN EJECT** (`isEjecting`). Stopping a chat's
    /// background agents makes each one's completion watcher inject a
    /// notification and start a model turn, and a cancelled turn's tail
    /// drains the queued and scheduled prompts the same way. Every one of
    /// those turns holds the session strongly again, so the first eject ended
    /// in `unloadBlocked` and only a second attempt worked. They PARK instead
    /// (`pendingTaskNotifications`, `pendingUserMessages`), not drop: they
    /// drain at the next idle tail after the eject, which is a later turn's
    /// tail or selecting the chat, exactly as they do after a busy turn.
    func canInjectChatInput(into chatID: UUID) -> Bool {
        canStartTurn(inChat: chatID) && selectedChatID == chatID
    }

    /// Background notifications wait behind queued user prompts and behind a
    /// prompt the tail drain restored to the composer but could not submit.
    func canInjectTaskNotification(into chatID: UUID) -> Bool {
        guard canInjectChatInput(into: chatID),
            (pendingUserMessages[chatID] ?? []).isEmpty
        else { return false }
        return promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && promptAttachments.isEmpty
    }

    /// Appends the notification as a user turn and answers it with a fresh
    /// generation turn (step 0: the notification is not an agent-loop step,
    /// so the turn gets the full `maxAutonomousSteps` budget).
    func injectTaskNotification(_ notification: PendingTaskNotification, chatID: UUID) {
        mutateTurnMessages(for: chatID) {
            $0.append(notification.kind.makeUserMessage(notification.note))
        }
        executeGenerationTurn(step: 0, chatID: chatID)
    }

    /// Injects whatever parked for this chat, if it is idle now. Called from
    /// the generation-turn tail (where `generating` just went false under
    /// the epoch guard) and from completion when the chat looked idle.
    func drainPendingTaskNotificationsIfIdle(chatID: UUID) {
        guard canInjectTaskNotification(into: chatID) else { return }
        var notes: [PendingTaskNotification] = []
        // Mid-task child messages first: they were sent before the child's
        // own completion notification, which may be parked behind them.
        let batch = subagentMessageHub.drain(chatID: chatID)
        if !batch.isEmpty {
            notes.append(PendingTaskNotification(
                kind: .taskNotification, note: SubagentMessageHub.carrier(for: batch)))
        }
        notes.append(contentsOf: pendingTaskNotifications[chatID] ?? [])
        guard !notes.isEmpty else { return }
        pendingTaskNotifications[chatID] = nil
        for notification in notes {
            mutateTurnMessages(for: chatID) {
                $0.append(notification.kind.makeUserMessage(notification.note))
            }
        }
        executeGenerationTurn(step: 0, chatID: chatID)
    }

    /// Removes one finished background run's card at the user's request. A
    /// running run cannot be dismissed -- stop it first, so an id the model
    /// may still reference cannot vanish from under `stop_agent`.
    ///
    /// **ALSO REFUSES BEFORE `completeBackgroundAgent` HAS RECORDED THE
    /// RESULT, EVEN THOUGH `status` MAY ALREADY SAY "not running".**
    /// `SubagentRunner.run` flips `status` via its own `.finished` progress
    /// event from INSIDE the run, before the separate watcher `Task` in
    /// `launchBackgroundAgent` has resumed and called `completeBackgroundAgent`
    /// -- so `status != "running"` alone is true during a real window in
    /// which dismissing would remove this entry out from under that
    /// completion, which then silently drops the `<task-notification>` (its
    /// own `guard let state = backgroundAgentRuns[id] else { return }`
    /// no-ops). `isRecordedComplete` is set only once that has genuinely
    /// happened.
    public func dismissBackgroundAgent(_ id: String) {
        guard let state = backgroundAgentRuns[id],
              state.status != "running", state.isRecordedComplete else { return }
        backgroundAgentRuns.removeValue(forKey: id)
        killedBackgroundAgentIDs.remove(id)
        retainedBackgroundChildren.removeValue(forKey: id)
        subagentMessageHub.resetBudget(childID: id)
    }

    /// Drops the oldest finished background records past the keep count.
    /// Running records are never pruned: their ids must keep resolving for
    /// `stop_agent`. Same reasoning as `dismissBackgroundAgent`'s own doc:
    /// `status` alone flips before `completeBackgroundAgent` has recorded the
    /// result, so a record not yet marked `isRecordedComplete` is excluded
    /// from what may be pruned even though it already reads as finished --
    /// otherwise a large enough backlog of completions landing back to back
    /// could prune a record out from under its own still-pending completion.
    private func pruneFinishedBackgroundAgents() {
        let finished = backgroundAgentRuns.values
            .filter { $0.status != "running" && $0.isRecordedComplete }
            .sorted { $0.startedAt < $1.startedAt }
        guard finished.count > Self.maxFinishedBackgroundAgents else { return }
        let doomed = finished.prefix(finished.count - Self.maxFinishedBackgroundAgents)
        for state in doomed {
            backgroundAgentRuns.removeValue(forKey: state.id)
            killedBackgroundAgentIDs.remove(state.id)
            retainedBackgroundChildren.removeValue(forKey: state.id)
            subagentMessageHub.resetBudget(childID: state.id)
        }
    }
}
