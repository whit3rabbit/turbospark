import Foundation

extension AppModel {
    // MARK: - Local meta commands (qwen-code /stats /export /help parity)

    /// Dispatches the session-less meta commands: they never generate, so
    /// like `/memory` they work with no model loaded. Called from `run()`
    /// above the `canRun` gate.
    public func handleLocalMetaCommand(_ draft: String) {
        let words = draft.dropFirst().split(separator: " ").map(String.init)
        let name = words.first?.lowercased() ?? ""
        switch name {
        case "stats":
            showSessionStats = true
        case "help":
            showHelpSheet = true
        case "export":
            let argument = words.dropFirst().first?.lowercased() ?? ""
            let format: AppChatExportFormat
            switch argument {
            case "json": format = .json
            case "html": format = .html
            default: format = .markdown
            }
            exportSelectedChat(format: format)
        default:
            break
        }
    }

    // MARK: - Pinning

    /// Sets a chat's sidebar pin and persists. Ungated on purpose, like
    /// `renameChat`: the field is independent of the turn pipeline, so a
    /// pin during a run cannot corrupt anything.
    public func setChatPinned(id: UUID, pinned: Bool) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        guard chats[index].isPinned != pinned else { return }
        chats[index].isPinned = pinned
        persistChats()
    }

    // MARK: - Usage ledger

    /// Records one generate call into the chat's cumulative ledger.
    ///
    /// Ghost chats are skipped: the ledger would sit on a row that never
    /// persists, and the ghost invariant asserts ghost rows carry no
    /// conversation content -- a token total describing the sealed
    /// conversation is content.
    public func recordUsage(promptTokens: Int, outputTokens: Int, chatID: UUID) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }),
            !chats[index].isGhost
        else { return }
        if chats[index].usage == nil {
            chats[index].usage = AppChatUsageLedger()
        }
        chats[index].usage?.record(promptTokens: promptTokens, outputTokens: outputTokens)
        persistChats()
    }

    /// Recorded usage totals across every archivable chat, what the usage
    /// dashboard and stats sheet render.
    public var usageTotals: AppChatUsageTotals {
        AppChatUsageTotals(chats: chats.filter { !$0.isGhost })
    }

    // MARK: - Scheduled tasks (cron)

    /// The run loop retains repeating timers, so restart must invalidate
    /// the previous poller before replacing our reference to it.
    func startCronScheduler() {
        stopCronScheduler()
        CronScheduler.shared.fireHandler = { [weak self] job in
            guard let self else { return false }
            return await MainActor.run { self.fireCronJob(job) }
        }
        let timer = Timer(timeInterval: 20, repeats: true) { _ in
            Task { @MainActor in
                await CronScheduler.shared.fireDueJobs()
            }
        }
        RunLoop.main.add(timer, forMode: .common)
        cronPollTimer = timer
    }

    func stopCronScheduler() {
        cronPollTimer?.invalidate()
        cronPollTimer = nil
    }

    /// Delivers a cron prompt into the job's chat.
    ///
    /// Selected and idle: submit now, through `run()` like any prompt.
    /// Anything else: park it in the same queue the user's own mid-turn
    /// prompts take, which turn tails and `selectChat` both drain. A FALSE
    /// return records a missed run in the job's history, which is how the
    /// Scheduled Tasks pane says "its chat was deleted" -- the only failure
    /// mode a parked entry cannot recover from.
    private func fireCronJob(_ job: AppCronJob) -> Bool {
        guard chats.contains(where: { $0.id == job.chatID }) else { return false }
        if job.chatID == selectedChatID, canRun {
            // A cron job's own prompt landing in the composer, not a paste.
            writePromptTextDirectly(job.prompt)
            run()
            return true
        }
        pendingUserMessages[job.chatID, default: []].append(
            QueuedUserPrompt(text: job.prompt, attachments: []))
        return true
    }

    // MARK: - Turn navigation (qwen-code turn jump parity)

    /// User-role rows are the turn anchors: jumping moves the viewport to a
    /// prompt, which is what "turn" means in the qwen shell's navigator.
    private var turnAnchorIDs: [UUID] {
        selectedTurnMessages.filter { $0.role == .user }.map(\.id)
    }

    /// Moves the transcript viewport `delta` turns back (negative) or
    /// forward. Wraps at the ends rather than dead-ending, matching the
    /// chat prev/next commands.
    public func jumpTurn(delta: Int) {
        let anchors = turnAnchorIDs
        guard !anchors.isEmpty, !generating else { return }
        let newIndex: Int
        if let currentIndex = turnNavigationTargetID.flatMap({ id in
            anchors.firstIndex(where: { $0 == id })
        }) {
            newIndex = (currentIndex + delta + anchors.count) % anchors.count
        } else {
            // First jump, from wherever the viewport sits: "previous" is
            // the last prompt, "next" the first.
            newIndex = delta < 0 ? anchors.count - 1 : 0
        }
        turnNavigationTargetID = anchors[newIndex]
        turnNavigationToken += 1
    }
}
