import Foundation
import TurboSpark

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
        guard let session = session else {
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

        let sink = AppToolRegistry.subagentProgressSink
        let agent = launch.agent
        let prompt = launch.taskPrompt
        let project = launch.project
        let chatID = launch.chatID
        let depth = launch.depth
        let userSystemPrompt = launch.userSystemPrompt
        let description = launch.taskDescription
        let options = launch.samplingOptions
        let runTask = Task<SubagentRunResult, Never> {
            await SubagentRunner.run(
                agent: agent, taskPrompt: prompt, session: session, project: project,
                chatID: chatID, depth: depth, userSystemPrompt: userSystemPrompt,
                samplingOptions: options,
                progress: { event in await sink?(id, event) },
                taskDescription: description)
        }
        backgroundAgentTasks[id] = runTask

        Task { [weak self] in
            let result = await runTask.value
            await self?.completeBackgroundAgent(id: id, result: result)
        }

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
            return
        }
        let note = BackgroundAgentNotification.text(
            id: id, status: status, agentName: state.agentName,
            displayName: state.displayName, result: result)
        if canInjectTaskNotification(into: chatID) {
            injectTaskNotification(note, chatID: chatID)
        } else {
            pendingTaskNotifications[chatID, default: []].append(note)
        }
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

    // MARK: - Notification injection

    /// Whether a notification turn may be injected and answered RIGHT NOW:
    /// the chat must be idle end to end. A turn in flight parks it for the
    /// tail; so does an approval card, whose own approve path refuses to run
    /// while `generating` is true -- injecting under it would eat the card.
    /// A live session is deliberately NOT required: with the model unloaded
    /// the notification still belongs in the transcript, and the generation
    /// it triggers self-guards on `session`.
    func canInjectTaskNotification(into chatID: UUID) -> Bool {
        !generating && !submitting && pendingToolCall == nil && pendingBatchCalls == nil
            && selectedChatID == chatID
            && chats.contains(where: { $0.id == chatID })
    }

    /// Appends the notification as a user turn and answers it with a fresh
    /// generation turn (step 0: the notification is not an agent-loop step,
    /// so the turn gets the full `maxAutonomousSteps` budget).
    func injectTaskNotification(_ note: String, chatID: UUID) {
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(role: .user, content: note))
        }
        executeGenerationTurn(step: 0, chatID: chatID)
    }

    /// Injects whatever parked for this chat, if it is idle now. Called from
    /// the generation-turn tail (where `generating` just went false under
    /// the epoch guard) and from completion when the chat looked idle.
    func drainPendingTaskNotificationsIfIdle(chatID: UUID) {
        guard canInjectTaskNotification(into: chatID),
            let notes = pendingTaskNotifications[chatID], !notes.isEmpty else { return }
        pendingTaskNotifications[chatID] = nil
        for note in notes {
            mutateTurnMessages(for: chatID) {
                $0.append(AppChatMessage(role: .user, content: note))
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
        }
    }
}
