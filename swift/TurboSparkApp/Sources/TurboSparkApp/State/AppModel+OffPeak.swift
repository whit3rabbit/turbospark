import Foundation

extension AppModel: OffPeakHost {
    // MARK: - Off-peak tasks

    /// Starts the off-peak poller. Own timer, own store, independent of
    /// `startCronScheduler`; the run loop retains a repeating timer, so a
    /// restart invalidates the previous one first. Nothing is admitted while
    /// `offPeakEnabled` is false (the default).
    func startOffPeakDispatcher() {
        offPeakTimer?.invalidate()
        AppToolRegistry.offPeakChatEligibleProvider = { [weak self] chatID in
            self?.chats.first(where: { $0.id == chatID }).map { !$0.isGhost } ?? false
        }
        let dispatcher = OffPeakTaskDispatcher(
            store: .shared, host: self,
            isEnabled: { [weak self] in self?.offPeakEnabled ?? false })
        offPeakDispatcher = dispatcher
        let timer = Timer(timeInterval: 5, repeats: true) { [weak dispatcher] _ in
            Task { @MainActor in dispatcher?.tick() }
        }
        RunLoop.main.add(timer, forMode: .common)
        offPeakTimer = timer
    }

    public func setOffPeakEnabled(_ enabled: Bool) {
        offPeakEnabled = enabled
        persistSettings()
    }

    func offPeakTurnDidFinish(chatID: UUID, cancelled: Bool, stillBusy: Bool) {
        offPeakDispatcher?.turnDidFinish(
            chatID: chatID, cancelled: cancelled, stillBusy: stillBusy)
    }

    public func offPeakChatExists(_ chatID: UUID) -> Bool {
        chats.contains(where: { $0.id == chatID })
    }

    /// The conditions a cron prompt needs (selected, idle, model loaded),
    /// plus a quiet composer and an empty queue so an off-peak turn never
    /// jumps ahead of, or lands on top of, something the user parked.
    public func offPeakCanDispatch(chatID: UUID) -> Bool {
        canInjectChatInput(into: chatID)
            && hasModelReadyForScheduledDelivery
            && (pendingUserMessages[chatID] ?? []).isEmpty
            && promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && promptAttachments.isEmpty
    }

    /// Delivered like a cron prompt: a plain synthetic user turn that never
    /// touches the composer and is never parsed for `!` or slash commands.
    public func offPeakDispatch(prompt: String, chatID: UUID) -> Bool {
        injectScheduledPrompt(prompt, chatID: chatID)
        return generating
    }

    public func offPeakIsAwaitingApproval(chatID: UUID) -> Bool {
        pendingToolCall != nil && (pendingToolCallChatID ?? chatID) == chatID
    }
}
