import Foundation

/// Ghost Mode: temporary chats that exist only in memory.
///
/// **THE DESIGN HAS TWO LAYERS, AND ONLY THE FIRST IS A GUARANTEE.**
/// Persistence is excluded by construction: both archive-construction points
/// filter `isGhost` rows, so no code path can write one to disk. Encryption
/// is the second layer: a ghost chat's messages, todos, draft, context
/// summary and skill state never sit on its row at all -- they are sealed in
/// `ghostVault` under a per-launch key, and the row keeps only its id,
/// project, title and timestamps.
///
/// Rules the rest of the codebase can rely on:
/// - A ghost row's `messages`, `todos`, `draft`, `contextSummary` and
///   `skillState` are ALWAYS empty. Read them through `turnMessages(for:)`,
///   `currentTodos`, `promptText` and the `storedSkillState` helpers, which
///   decrypt for ghosts; write them through `mutateTurnMessages(for:_:)`,
///   `mutateGhostPayload(for:_:)` or the specific setters, which re-seal.
/// - A ghost chat is NEVER persisted and dispatches NO lifecycle hooks that
///   carry prompt or tool content (`UserPromptSubmit` is skipped entirely);
///   hooks that see only ids still fire, because they carry no content.
extension AppModel {
    // MARK: - Reading ghost state

    /// Whether the selected conversation is a temporary chat.
    public var isInGhostChat: Bool {
        selectedChat.isGhost
    }

    /// The ghost chat row, when one exists.
    public var ghostChat: AppChat? {
        chats.first(where: { $0.isGhost })
    }

    /// Decrypts a ghost chat's payload. Empty for a non-ghost id.
    public func ghostPayload(for chatID: UUID) -> GhostChatPayload {
        ghostVault.payload(for: chatID)
    }

    /// Whether the ghost chat carries anything a user could lose, which is
    /// what the end-confirmation dialog keys on.
    public var ghostChatHasContent: Bool {
        guard let ghost = ghostChat else { return false }
        return ghostVault.payload(for: ghost.id).hasContent
    }

    /// A chat's messages, decrypting from the vault when it is a ghost.
    public func turnMessages(for chatID: UUID) -> [AppChatMessage] {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return [] }
        if chats[index].isGhost {
            return ghostVault.payload(for: chatID).messages
        }
        return chats[index].messages
    }

    /// Messages of the selected chat, vault-aware. What transcript views read.
    public var selectedTurnMessages: [AppChatMessage] {
        turnMessages(for: selectedChatID)
    }

    /// Whether a chat has any transcript content, vault-aware. What the
    /// sidebar's history list filters on.
    public func chatHasTranscript(_ chat: AppChat) -> Bool {
        if chat.isGhost {
            return !ghostVault.payload(for: chat.id).messages.isEmpty
        }
        return !chat.messages.isEmpty
    }

    /// A chat's stored skill state, decrypting from the vault when ghost.
    func storedSkillState(chatIndex: Int) -> AppSkillState? {
        let chat = chats[chatIndex]
        if chat.isGhost {
            return ghostVault.payload(for: chat.id).skillState
        }
        return chat.skillState
    }

    /// Sets a chat's stored skill state, re-sealing when ghost.
    func setStoredSkillState(chatIndex: Int, _ newState: AppSkillState?) {
        let chatID = chats[chatIndex].id
        if chats[chatIndex].isGhost {
            mutateGhostPayload(for: chatID) { $0.skillState = newState }
        } else {
            chats[chatIndex].skillState = newState
        }
    }

    // MARK: - Mutating ghost state

    /// Re-seals a ghost chat's payload after a change.
    ///
    /// The vault is not observable, so the row's `updatedAt` is bumped as
    /// the SwiftUI invalidation: `chats` is published, and every view that
    /// shows ghost content re-reads the payload off that bump.
    public func mutateGhostPayload(
        for chatID: UUID, _ change: (inout GhostChatPayload) -> Void
    ) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }),
            chats[index].isGhost
        else { return }
        var payload = ghostVault.payload(for: chatID)
        change(&payload)
        ghostVault.store(payload, for: chatID)
        chats[index].updatedAt = Date()
    }

    /// The ONE place a turn message is appended, replaced or removed.
    ///
    /// Ghost chats mutate the encrypted payload and touch nothing on disk;
    /// ordinary chats mutate the row and persist, exactly as the ~10 sites
    /// that now route through here did inline before. Taking a closure over
    /// the array (rather than an append-only helper) keeps the two
    /// in-place-repair sites -- `appendToolExecutionTurn`'s call replacement
    /// and the guardrail retry pair -- on the same primitive.
    public func mutateTurnMessages(
        for chatID: UUID, _ change: (inout [AppChatMessage]) -> Void
    ) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return }
        if chats[index].isGhost {
            mutateGhostPayload(for: chatID) { change(&$0.messages) }
        } else {
            change(&chats[index].messages)
            chats[index].updatedAt = Date()
            persistChats()
        }
    }

    // MARK: - Lifecycle

    /// Enters Ghost Mode: selects the temporary chat, creating it if needed.
    ///
    /// The ghost chat stamps the active project, so an agent project's
    /// persona, tools and permissions apply exactly as they would to a
    /// normal chat -- it is the PERSISTENCE that differs, not the run.
    /// Returns the ghost chat's id, or the current selection unchanged when
    /// the turn lifecycle refuses a switch.
    ///
    /// **ONE GHOST CHAT PER SESSION, STAMPED ONCE.** Re-entering after
    /// switching to a different project resumes the SAME ghost chat under
    /// whichever project was active the first time it was created -- the
    /// stamp does not follow later project switches, and nothing in the UI
    /// discloses which project a live ghost session is bound to. Deliberate
    /// (one session, one temporary conversation), not a bug; end the ghost
    /// chat first if a fresh one under the current project is wanted.
    @discardableResult
    public func enterGhostChat() -> UUID {
        guard !generating, !submitting, pendingToolCall == nil else { return selectedChatID }
        activeSection = .chat
        if let existing = ghostChat {
            if existing.id != selectedChatID {
                selectChat(id: existing.id)
            }
            return existing.id
        }
        var chat = AppChat(projectID: selectedProjectID)
        chat.isGhost = true
        chat.title = "Temporary Chat"
        chats.insert(chat, at: 0)
        selectedChatID = chat.id
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        updateTokenEstimate()
        return chat.id
    }

    /// Ends the ghost chat: wipes its sealed payload, removes the row and
    /// reselects a normal chat. The conversation is gone for good; the
    /// confirm-if-content dialog lives in the view that calls this.
    public func endGhostChat() {
        guard !generating, !submitting, pendingToolCall == nil, let ghost = ghostChat
        else { return }
        let ghostID = ghost.id
        ghostVault.wipe(for: ghostID)
        deleteChat(id: ghostID)
    }
}
