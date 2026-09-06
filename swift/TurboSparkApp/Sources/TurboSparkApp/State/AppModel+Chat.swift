import Foundation

extension AppModel {
    // MARK: - Reading the selected chat
    //
    // Moved off the base file, which the package's convention reserves for
    // published state and core lifecycle. These are chat operations reached
    // from chat views, and `materializeDraftChatIfNeeded` in particular is
    // called from exactly two places -- both of them the writers directly
    // below it.

    /// Index of the currently selected chat in `chats`.
    public var selectedChatIndex: Int? {
        chats.firstIndex { $0.id == selectedChatID }
    }

    /// The currently selected chat conversation.
    ///
    /// Pure: no mutation of `@Published` state on read (state#7 -- "Publishing
    /// changes from within view updates" is undefined behavior, and this
    /// getter is read from SwiftUI view bodies). `selectedChatID` is kept
    /// valid at the points where `chats` actually changes -- `loadChats()`,
    /// `selectChat`, `createChat`, `deleteChat` -- rather than patched
    /// lazily here; `activeDraftChat`'s identity is kept in sync by
    /// `selectedChatID`'s own `didSet` above.
    public var selectedChat: AppChat {
        if let index = selectedChatIndex {
            return chats[index]
        }
        return activeDraftChat
    }

    /// Active task checklist for the currently selected chat.
    public var currentTodos: [TodoItem] {
        // Ghost chats keep todos in the vault; their row fields are always
        // empty (see AppModel+Ghost.swift).
        selectedChat.isGhost
            ? ghostVault.payload(for: selectedChatID).todos
            : selectedChat.todos
    }

    /// Present continuous description of the active `in_progress` task, if any.
    public var activeTaskDescription: String? {
        if let inProgress = selectedChat.todos.first(where: { $0.isInProgress }) {
            return inProgress.activeForm.isEmpty ? inProgress.content : inProgress.activeForm
        }
        return nil
    }

    /// Updates the checklist items for a given chat and persists the change.
    public func updateTodos(for chatID: UUID, todos: [TodoItem]) {
        if let index = chats.firstIndex(where: { $0.id == chatID }) {
            if chats[index].isGhost {
                // Sealed in the vault; the row bump inside is what repaints
                // the checklist UI. Nothing reaches disk.
                mutateGhostPayload(for: chatID) { $0.todos = todos }
            } else {
                chats[index].todos = todos
                chats[index].updatedAt = Date()
                persistChats()
            }
        } else if activeDraftChat.id == chatID {
            activeDraftChat.todos = todos
            activeDraftChat.updatedAt = Date()
            materializeDraftChatIfNeeded()
        }
    }

    /// Draft prompt text for the currently selected chat.
    ///
    /// A ghost chat's draft is sealed in the vault like its messages, and
    /// its setter deliberately skips `persistChatsDebounced()`: the archive
    /// filter would exclude the row anyway, and skipping keeps a keystroke
    /// in Ghost Mode from rewriting every real chat to disk.
    public var promptText: String {
        get {
            if selectedChat.isGhost {
                return ghostVault.payload(for: selectedChatID).draft
            }
            return selectedChat.draft
        }
        set {
            if let index = selectedChatIndex {
                if chats[index].isGhost {
                    mutateGhostPayload(for: selectedChatID) { $0.draft = newValue }
                    updateTokenEstimate()
                    return
                }
                chats[index].draft = newValue
                chats[index].updatedAt = Date()
                // Debounced: this is one keystroke, and the store rewrites
                // every chat in the archive on each call.
                persistChatsDebounced()
            } else {
                activeDraftChat.draft = newValue
                activeDraftChat.updatedAt = Date()
                materializeDraftChatIfNeeded()
            }
            updateTokenEstimate()
        }
    }

    /// Document attachments attached to the current prompt draft.
    public var promptAttachments: [AppPromptAttachment] {
        selectedChat.draftAttachments
    }

    /// Promotes the transient draft into `chats` once it has content.
    ///
    /// Called from the two writers (`promptText`, `updateTodos`) rather than
    /// from a timer, so the promotion happens at the moment there is
    /// something to lose. An empty draft is left alone: a chat row that
    /// appears when the window opens, before the user has typed anything, is
    /// noise -- and `createChat` reuses an empty selected chat anyway, so
    /// nothing accumulates.
    @discardableResult
    func materializeDraftChatIfNeeded() -> Bool {
        guard selectedChatIndex == nil else { return false }
        let draft = activeDraftChat
        let hasContent =
            !draft.draft.isEmpty || !draft.draftAttachments.isEmpty || !draft.todos.isEmpty
            || !draft.messages.isEmpty
        guard hasContent else { return false }
        // A draft that predates a project selection carries none, and this is
        // the last point before it becomes a real row (state#79).
        var draftToInsert = draft
        if draftToInsert.projectID == nil {
            draftToInsert.projectID = selectedProjectID
        }
        chats.insert(draftToInsert, at: 0)
        // `selectedChatID` already equals `draft.id` (the `didSet` above keeps
        // them in step), so this needs no re-selection.
        persistChats()
        return true
    }

    // MARK: - Chat lifecycle

    @discardableResult
    public func createChat(projectID: UUID? = nil) -> UUID {
        // `pendingToolCall == nil` for `selectChat`'s reason (state#49):
        // `generating` is false while a call waits, so this was legal, and
        // starting a new chat there strands the approval card on a
        // conversation the user can no longer see.
        // `!submitting` beside the other two (state#77): `run()` awaits its
        // `UserPromptSubmit` hook with `generating` still false, and a chat
        // created in that window is one the awaited submission then appends
        // its user turn into by captured id.
        guard !generating, !submitting, pendingToolCall == nil else { return selectedChatID }
        activeSection = .chat
        let assignedProjectID = projectID ?? selectedProjectID
        // If current chat is already empty and matches the target project, reset and stay on it.
        // **NEVER "REUSE" A GHOST CHAT** for a plain New Chat: its row fields
        // are empty by design (the content is vaulted), so the emptiness
        // check alone would reset and keep the user in Ghost Mode after
        // they explicitly asked for a persisted chat.
        if !selectedChat.isGhost && selectedChat.messages.isEmpty && selectedChat.draft.isEmpty && selectedChat.draftAttachments.isEmpty && selectedChat.projectID == assignedProjectID {
            outputText = ""
            outputReasoningText = ""
            outputPromptText = ""
            updateTokenEstimate()
            return selectedChatID
        }
        let chat = AppChat(projectID: assignedProjectID)
        chats.insert(chat, at: 0)
        selectedChatID = chat.id
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        persistChats()
        updateTokenEstimate()
        let createdID = chat.id
        let createdProject = projects.first { $0.id == assignedProjectID }
        Task {
            _ = await self.dispatchLifecycleHook(
                event: .sessionStart, chatID: createdID, project: createdProject, source: "clear")
        }
        return chat.id
    }

    public func selectChat(id: UUID) {
        // A pending tool call already captures its own chat ID at proposal
        // time (state#9), so approving/denying it lands in the right place
        // regardless; this guard is the UX half -- switching away mid
        // approval, with `generating` already false, read as an ordinary
        // chat switch and made it easy to lose track of which chat is
        // waiting on a decision.
        guard !generating, !submitting, pendingToolCall == nil else { return }
        activeSection = .chat
        guard id != selectedChatID else { return }
        selectedChatID = id
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        persistChats()
        updateTokenEstimate()
    }

    public func renameChat(id: UUID, title: String) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        chats[index].title = String(trimmed.prefix(80))
        chats[index].updatedAt = Date()
        persistChats()
    }

    /// Sets or clears THIS chat's own system prompt.
    ///
    /// Blank clears it back to `nil` rather than storing `""`, so "no override"
    /// has one representation on disk instead of two. `resolvedUserSystemPrompt`
    /// treats them alike, which keeps a chat persisted before this normalizing
    /// existed behaving the same.
    public func setChatSystemPrompt(id: UUID, prompt: String) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        chats[index].systemPrompt = trimmed.isEmpty ? nil : trimmed
        chats[index].updatedAt = Date()
        persistChats()
    }

    public func deleteChat(id: UUID) {
        // **AND NOT MID-SUBMISSION** (state#77). `run()` captured this chat's
        // id before awaiting its hook and re-creates the row under the SAME
        // UUID if it is gone, so a chat deleted in that window comes back
        // carrying the prompt the user thought they had thrown away.
        guard !generating, !submitting, let index = chats.firstIndex(where: { $0.id == id })
        else { return }
        // Captured before the removal below: `SessionEnd` describes the chat
        // that is going away, and after `chats.remove` there is no row left
        // to resolve its project from (state#67).
        let deletedProject = project(forChat: id)
        // A pending approval on THIS chat has no chat left to land its
        // decision in once it's gone -- Approve/Deny would silently no-op
        // against a stale ID (state#9's own reasoning, applied to deletion
        // rather than to a chat switch).
        if pendingToolCallChatID == id {
            // Through the helper, not three of its four fields open-coded
            // (state#49): `pendingToolCallProject` was left behind, which is
            // exactly the stale value the helper exists to prevent.
            clearPendingToolCall()
        }
        // A ghost row's content lives only in the vault (`AppModel+Ghost.swift`),
        // never on this row, so removing the row alone leaves its sealed
        // ciphertext behind in memory for the rest of the process. Wiped
        // here rather than relying on every deletion path (the sidebar's
        // Delete action included) to call `endGhostChat()` instead.
        if chats[index].isGhost {
            ghostVault.wipe(for: id)
        }
        chats.remove(at: index)
        if chats.isEmpty {
            selectedChatID = UUID()
        } else if selectedChatID == id {
            // **FROM THE PROJECT'S OWN CHATS** (state#80). The replacement
            // was picked out of the UNFILTERED list, so deleting the last
            // chat under project A selected a project-B conversation the
            // sidebar does not even show -- the shape `selectProject`
            // already gets right. Ghost chats are excluded too: ending a
            // normal chat must not silently drop the user into the
            // temporary one.
            let siblings = chats.filter { $0.projectID == selectedProjectID && !$0.isGhost }
            if let replacement = siblings.first {
                selectedChatID = replacement.id
            } else if selectedProjectID == nil {
                selectedChatID = chats[min(index, chats.count - 1)].id
            } else {
                // Nothing left under this project. A fresh id rather than
                // another project's chat; `selectedChatID`'s own `didSet`
                // builds the draft with the project stamped on it.
                selectedChatID = UUID()
            }
        }
        outputText = ""
        outputReasoningText = ""
        persistChats()
        updateTokenEstimate()
        Task {
            // The grants belonged to a conversation that no longer exists,
            // and the ids are UUIDs, so nothing would ever collect them
            // (state#49).
            await SessionApprovalStore.shared.clear(sessionID: id.uuidString)
            _ = await self.dispatchSessionEnd(
                reason: "clear", chatID: id, project: deletedProject)
        }
    }

    /// Clears the conversation.
    ///
    /// **FOUR THINGS OUTLIVED IT** (state#49), each of which describes the run
    /// that was just erased: the SKILL.state bookkeeping, a call still
    /// awaiting approval, the checklist, and the session's "always allow"
    /// grants. `resetSkillState()` and `SessionApprovalStore.clear` both
    /// existed for this and neither had a caller -- so a cleared conversation
    /// came back with a tool pre-approved on the strength of a call the user
    /// could no longer read.
    public func clearOutput() {
        guard !generating, !submitting else { return }
        let clearedChatID = selectedChatID
        let clearedProject = project(forChat: clearedChatID)
        if let index = selectedChatIndex {
            if chats[index].isGhost {
                // The row fields are placeholders; the vault holds the
                // conversation, so that is what Clear empties.
                mutateGhostPayload(for: chats[index].id) {
                    $0.messages.removeAll()
                    $0.todos = []
                    $0.contextSummary = nil
                    $0.skillState = nil
                }
            } else {
                chats[index].messages.removeAll()
                chats[index].contextSummary = nil
                chats[index].skillState = nil
                chats[index].todos = []
                chats[index].updatedAt = Date()
            }
        }
        skillStateLastError = nil
        if pendingToolCallChatID == clearedChatID {
            clearPendingToolCall()
        }
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        diagnostics = nil
        error = nil
        persistChats()
        updateTokenEstimate()
        Task {
            await SessionApprovalStore.shared.clear(sessionID: clearedChatID.uuidString)
            _ = await self.dispatchSessionEnd(
                reason: "clear", chatID: clearedChatID, project: clearedProject)
        }
    }

    public func addPromptAttachment(_ attachment: AppPromptAttachment, toChatID chatID: UUID? = nil) {
        guard !generating, !submitting else {
            error = "Cannot attach files while generating."
            return
        }
        let targetID = chatID ?? selectedChatID
        if let index = chats.firstIndex(where: { $0.id == targetID }) {
            chats[index].draftAttachments.append(attachment)
            chats[index].updatedAt = Date()
            persistChats()
        } else {
            var chat = AppChat(id: targetID, projectID: selectedProjectID)
            chat.draftAttachments.append(attachment)
            chats.insert(chat, at: 0)
            selectedChatID = targetID
            persistChats()
        }
        updateTokenEstimate()
    }

    public func removePromptAttachment(id: UUID) {
        guard !generating, let index = selectedChatIndex else { return }
        chats[index].draftAttachments.removeAll { $0.id == id }
        chats[index].updatedAt = Date()
        persistChats()
        updateTokenEstimate()
    }

    public func clearPromptAttachments() {
        guard !generating, let index = selectedChatIndex else { return }
        chats[index].draftAttachments.removeAll()
        chats[index].updatedAt = Date()
        persistChats()
        updateTokenEstimate()
    }

    /// Returns chats filtered by the active project, ordered by most recently updated.
    public var orderedChats: [AppChat] {
        filteredChats.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Selects the previous chat in the sorted conversation list.
    public func selectPreviousChat() {
        guard !generating else { return }
        let list = orderedChats
        guard !list.isEmpty else { return }
        guard let currentIndex = list.firstIndex(where: { $0.id == selectedChatID }) else {
            if let first = list.first { selectChat(id: first.id) }
            return
        }
        if currentIndex > 0 {
            selectChat(id: list[currentIndex - 1].id)
        }
    }

    /// Selects the next chat in the sorted conversation list.
    public func selectNextChat() {
        guard !generating else { return }
        let list = orderedChats
        guard !list.isEmpty else { return }
        guard let currentIndex = list.firstIndex(where: { $0.id == selectedChatID }) else {
            if let first = list.first { selectChat(id: first.id) }
            return
        }
        if currentIndex < list.count - 1 {
            selectChat(id: list[currentIndex + 1].id)
        }
    }
}

