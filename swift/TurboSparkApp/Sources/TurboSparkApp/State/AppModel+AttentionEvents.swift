import AppKit
import Foundation

/// OC-02: the emit side of out-of-app notifications. Four hooks call
/// `raiseAttention` at the lifecycle points that already exist; the policy and
/// the center decide whether anything is shown.
///
/// Where each event comes from:
///
/// - `needsPermission`: `pendingToolCall` opening (the card) and a background
///   chat parking a proposal (`deferToolProposal`).
/// - `needsAnswer`: `waitForUserAnswers`, when the question card publishes.
/// - `error`: `finishCancelled` with an error reason.
/// - `turnFinished`: `finishTurnTail`, only when the chat is idle afterwards.
///   A user Stop, an error, a parked approval and a turn that handed over to
///   the next queued one do not fire it.
extension AppModel {
    func installAttentionEvents() {
        attentionCenter.openChat = { [weak self] id in self?.openChatFromNotification(id) }
        attentionActivationObserver = NotificationCenter.default.addObserver(
            forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.attentionCenter.appDidBecomeActive() }
        }
    }

    /// A banner click: bring the app forward and select the chat. A chat that
    /// no longer exists leaves the app in front on whatever it showed.
    func openChatFromNotification(_ id: UUID) {
        showMainWindow()
        _ = selectChatFromNotification(id)
    }

    /// The selection half of a click. False when the chat is gone.
    @discardableResult
    func selectChatFromNotification(_ id: UUID) -> Bool {
        guard chats.contains(where: { $0.id == id }) else { return false }
        selectChat(id: id)
        return true
    }

    // MARK: Settings

    func setAttentionEnabled(_ event: AttentionEvent, _ enabled: Bool) {
        guard attentionPreferences[event].enabled != enabled else { return }
        attentionPreferences[event].enabled = enabled
        persistSettings()
    }

    func setAttentionSound(_ event: AttentionEvent, _ sound: AttentionSound) {
        guard attentionPreferences[event].sound != sound.rawValue else { return }
        attentionPreferences[event].sound = sound.rawValue
        persistSettings()
    }

    func setAgentPushAllowed(_ allowed: Bool) {
        guard attentionPreferences.allowAgentPush != allowed else { return }
        attentionPreferences.allowAgentPush = allowed
        persistSettings()
    }

    // MARK: Emit

    /// The `PushNotification` tool's delivery. The title and body are what
    /// the agent supplied (already sanitized by the executor), never tool
    /// arguments or file contents. A Ghost chat never posts, for the same
    /// reason `raiseAttention` skips it.
    func deliverAgentPush(title: String, body: String, chatID: UUID?) async -> AgentPushResult {
        if let chatID, chats.first(where: { $0.id == chatID })?.isGhost == true {
            return .ghostChat
        }
        let notification = AttentionNotification(chatID: chatID ?? UUID(), title: title, body: body)
        return await attentionCenter.pushFromAgent(notification, preferences: attentionPreferences)
    }

    /// Raises one cue. Notification bodies carry the chat title and a generic
    /// line, never transcript content; a permission cue names the tool and
    /// not its arguments. A Ghost chat never raises one: its title is
    /// deliberately kept out of anything the system stores.
    @discardableResult
    func raiseAttention(_ event: AttentionEvent, chatID: UUID, tool: String? = nil) -> Task<Void, Never> {
        guard let chat = chats.first(where: { $0.id == chatID }), !chat.isGhost else {
            return Task {}
        }
        let notification = AttentionNotification(
            chatID: chatID, title: chat.title, body: Self.attentionBody(event, tool: tool))
        let preferences = attentionPreferences
        return Task { [attentionCenter] in
            await attentionCenter.notify(event, notification: notification, preferences: preferences)
        }
    }

    static func attentionBody(_ event: AttentionEvent, tool: String?) -> String {
        switch event {
        case .turnFinished:
            return AppStrings.localized("Finished replying")
        case .needsPermission:
            if let tool, !tool.isEmpty {
                return AppStrings.localized("Needs your approval to run \(tool)")
            }
            return AppStrings.localized("Needs your approval")
        case .needsAnswer:
            return AppStrings.localized("Has a question for you")
        case .error:
            return AppStrings.localized("Stopped because of an error")
        }
    }

    /// From `pendingToolCall.didSet`. The chat id is assigned after the call,
    /// so the state is read one hop later and only if the card is still up.
    func scheduleNeedsPermissionCue() {
        Task { @MainActor [weak self] in
            guard let self, let call = self.pendingToolCall,
                let chatID = self.pendingToolCallChatID
            else { return }
            await self.raiseAttention(.needsPermission, chatID: chatID, tool: call.name).value
        }
    }

    /// From `finishTurnTail`, after the drains and admission ran.
    func raiseTurnFinishedIfIdle(chatID: UUID, wasCancelled: Bool) {
        let errored = attentionErroredChats.remove(chatID) != nil
        guard !wasCancelled, !errored, !generating,
            chatTurnState(for: chatID) == .idle
        else { return }
        raiseAttention(.turnFinished, chatID: chatID)
    }

    /// From `finishCancelled`: only the two error reasons count.
    func noteTurnStopped(chatID: UUID, reason: String) {
        guard reason == StreamInterruptionReason.error.rawValue || reason == "context_overflow"
        else { return }
        attentionErroredChats.insert(chatID)
        raiseAttention(.error, chatID: chatID)
    }
}
