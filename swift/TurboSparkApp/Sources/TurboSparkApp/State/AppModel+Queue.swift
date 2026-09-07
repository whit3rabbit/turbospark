import Foundation

/// One prompt parked while its chat is busy.
///
/// Claude Code reference: `src/utils/messageQueueManager.ts` -- input typed
/// while a turn runs is QUEUED and sent when the turn ends, never silently
/// dropped. This app used to drop it: Return was swallowed whenever
/// `canRun` was false, so a user who kept typing into a long turn had to
/// notice, wait, and send again.
///
/// In-memory only, like `pendingTaskNotifications`: the queue describes the
/// gap between two turns of a live session, and a relaunch has nothing to
/// resume it against.
public struct QueuedUserPrompt: Identifiable, Equatable {
    public let id: UUID
    /// The draft text, trimmed at enqueue time the way `run()` trims it.
    public var text: String
    /// The draft attachments, MOVED out of the chat row at enqueue time.
    public var attachments: [AppPromptAttachment]

    public init(id: UUID = UUID(), text: String, attachments: [AppPromptAttachment] = []) {
        self.id = id
        self.text = text
        self.attachments = attachments
    }
}

extension AppModel {
    /// Moves the current draft into the queue for its chat.
    ///
    /// **MIRRORS `run()`'S CAPTURE AND CLEAR EXACTLY**, because the entry
    /// has to survive the same journey a sent prompt would: the text is
    /// trimmed here (the composer may still hold trailing whitespace), the
    /// attachments leave the row (so they cannot be mutated or re-sent by
    /// a user edit that lands before the drain), and the ghost draft is
    /// cleared through the vault payload, not the row. A ghost chat's
    /// queued text stays in memory only, the same exposure class as the
    /// composer holding it while it is typed.
    ///
    /// Only ever called for the SELECTED chat: it captures `promptText`
    /// and `promptAttachments`, both of which describe the selection.
    func enqueueCurrentDraft(chatID: UUID) {
        let text = promptText.trimmingCharacters(in: .whitespacesAndNewlines)
        let attachments = promptAttachments
        guard !text.isEmpty || !attachments.isEmpty else { return }
        pendingUserMessages[chatID, default: []].append(
            QueuedUserPrompt(text: text, attachments: attachments))
        if let index = chats.firstIndex(where: { $0.id == chatID }) {
            if chats[index].isGhost {
                mutateGhostPayload(for: chatID) { $0.draft = "" }
            } else {
                chats[index].draft = ""
            }
            chats[index].draftAttachments = []
            chats[index].updatedAt = Date()
        }
        promptText = ""
    }

    /// The selected chat's queued prompts, for the composer pill.
    public func queuedMessages(for chatID: UUID) -> [QueuedUserPrompt] {
        pendingUserMessages[chatID] ?? []
    }

    /// Puts one queued entry back into the draft for editing (the pill's
    /// remove affordance). Attachments return to the row first, so
    /// `promptAttachments` reads them the way a fresh attach would have.
    public func restoreQueuedMessage(id: UUID) {
        let chatID = selectedChatID
        guard var queued = pendingUserMessages[chatID],
            let index = queued.firstIndex(where: { $0.id == id })
        else { return }
        let entry = queued.remove(at: index)
        if queued.isEmpty {
            pendingUserMessages[chatID] = nil
        } else {
            pendingUserMessages[chatID] = queued
        }
        if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
            chats[rowIndex].draftAttachments += entry.attachments
        }
        promptText = promptText.isEmpty
            ? entry.text
            : "\(promptText)\n\n\(entry.text)"
    }

    /// Sends the chat's first queued prompt NOW, if the chat is idle.
    ///
    /// **GOES THROUGH `run()`, NOT AROUND IT.** The entry is restored into
    /// the composer and submitted the ordinary way, so a queued meta
    /// command dispatches as a command, `@` mentions resolve, the
    /// `UserPromptSubmit` hook runs, and the auto-title fires -- every one
    /// of those would be skipped by appending the message directly.
    ///
    /// Only the FIRST entry goes; the rest wait for the turn this starts,
    /// whose tail calls this again. Called from the generation-turn tail
    /// (before the task-notification drain, so USER intent goes before a
    /// background agent's completion note) and idempotent whenever the
    /// chat is not idle.
    func drainPendingUserMessagesIfIdle(chatID: UUID) {
        guard canInjectTaskNotification(into: chatID),
            let queued = pendingUserMessages[chatID], !queued.isEmpty
        else { return }
        let first = queued[0]
        if queued.count > 1 {
            pendingUserMessages[chatID] = Array(queued.dropFirst())
        } else {
            pendingUserMessages[chatID] = nil
        }
        if let rowIndex = chats.firstIndex(where: { $0.id == chatID }) {
            chats[rowIndex].draftAttachments = first.attachments
        }
        promptText = first.text
        run()
    }
}
