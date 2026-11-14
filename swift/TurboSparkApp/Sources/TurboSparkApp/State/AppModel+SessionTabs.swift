import Foundation

/// OC-04: session tabs, the AppModel side.
///
/// A tab is a chat id (`SessionTabStore`). Activating one goes through
/// `openChat`, the path a sidebar click takes, so the ZP-23 lane verdict, the
/// per-chat drafts, the read markers and the OC-05 navigation history all
/// behave exactly as for a click. There is no second conversation container.
///
/// **LIMITS INHERITED FROM ZP-23, ON PURPOSE.**
/// - Moving between chats of the same project is governed by the coordinator
///   (`canOpenChat`): a background turn does not block it, an open approval
///   card or a submitting turn does.
/// - Moving to a chat under ANOTHER project is a workspace switch and keeps
///   the blanket guard (no turn, no submit, no approval), so that tab switch
///   is refused with a toast.
/// - Still one turn at a time: tabs add no decode lane.
///
/// **CMD+W STAYS THE WINDOW'S.** It is in the OC-01 reserved list, so Close
/// Tab takes Cmd+Shift+W.
extension AppModel {
    var sessionTabsActive: Bool { sessionTabs.isActive }

    // MARK: - Validity

    /// A chat that may be a tab: a real, unarchived, non-ghost row. A ghost
    /// chat never reaches disk, so a tab for one could not be restored.
    func isSessionTabChat(_ id: UUID) -> Bool {
        guard let chat = chats.first(where: { $0.id == id }) else { return false }
        return !chat.isGhost && !chat.isArchived
    }

    /// Nothing to lose: no messages, text, attachments, checklist or
    /// per-chat sampling edit. Closing such a draft tab discards the row.
    func isEmptyDraftChat(_ chat: AppChat) -> Bool {
        chat.messages.isEmpty && chat.draft.isEmpty && chat.draftAttachments.isEmpty
            && chat.todos.isEmpty && chat.samplingOverride == nil
    }

    // MARK: - Persistence

    /// Restores the saved list at launch, after the chats load. Tabs for
    /// chats that were deleted, archived or are ghosts are dropped.
    func loadSessionTabs(from url: URL? = nil) {
        var store = SessionTabStore(archive: SessionTabFileStore.load(from: url))
        store.prune(isValid: isSessionTabChat)
        store.navigate(to: isSessionTabChat(selectedChatID) ? selectedChatID : nil)
        sessionTabs = store
    }

    func persistSessionTabs(to url: URL? = nil) {
        SessionTabFileStore.save(sessionTabs.archive, to: url)
    }

    private func mutateSessionTabs(_ change: (inout SessionTabStore) -> Void) {
        var next = sessionTabs
        change(&next)
        guard next != sessionTabs else { return }
        let listChanged = next.archive != sessionTabs.archive
        sessionTabs = next
        if listChanged { persistSessionTabs() }
    }

    // MARK: - Following the app's own selection

    /// Called from `selectedChatID.didSet`.
    func sessionTabsFollowChatSelection() {
        guard !sessionTabFollowSuppressed else { return }
        let target = isSessionTabChat(selectedChatID) ? selectedChatID : nil
        mutateSessionTabs { $0.navigate(to: target) }
    }

    /// Drops tabs and closed entries for chats that are gone or archived.
    func pruneSessionTabs() {
        mutateSessionTabs { $0.prune(isValid: isSessionTabChat) }
    }

    // MARK: - Commands

    /// Cmd+T. Reuses an empty draft tab that is already open (a second one
    /// would add nothing), otherwise creates a chat and opens it as a tab.
    func openNewSessionTab() {
        guard laneVerdict(.select(target: nil)) == .allowed else {
            announceSessionTabRefused()
            return
        }
        if let draftID = sessionTabs.tabs.first(where: { id in
            chats.first(where: { $0.id == id }).map(isEmptyDraftChat) ?? false
        }) {
            activateSessionTab(chatID: draftID)
            NotificationCenter.default.post(name: .focusPrompt, object: nil)
            return
        }
        guard sessionTabs.tabs.count < SessionTabStore.maxTabs else {
            announceSessionTabCapacity()
            return
        }
        let leaving = isSessionTabChat(selectedChatID) ? selectedChatID : nil
        // `createChat` moves the selection; without this the follow hook would
        // retarget the current tab to the new chat instead of adding one.
        sessionTabFollowSuppressed = true
        let created = createChat()
        sessionTabFollowSuppressed = false
        mutateSessionTabs { _ = $0.open(created, current: leaving) }
        NotificationCenter.default.post(name: .focusPrompt, object: nil)
    }

    /// Opens an existing chat as a tab and moves to it (sidebar "Open in New
    /// Tab"). An already-open chat just takes the selection.
    func openChatInSessionTab(id: UUID) {
        guard isSessionTabChat(id) else { return }
        guard canOpenChat(id: id) else {
            announceSessionTabRefused()
            return
        }
        let leaving = isSessionTabChat(selectedChatID) ? selectedChatID : nil
        var probe = sessionTabs
        guard probe.open(id, current: leaving) == .opened else {
            announceSessionTabCapacity()
            return
        }
        mutateSessionTabs { _ = $0.open(id, current: leaving) }
        openChat(id: id)
    }

    /// Moves to an open tab. Refused, with a reason, when the lane verdict or
    /// the project guard says no.
    func activateSessionTab(chatID: UUID) {
        guard chatID != selectedChatID else { return }
        guard isSessionTabChat(chatID) else {
            pruneSessionTabs()
            return
        }
        guard canOpenChat(id: chatID) else {
            announceSessionTabRefused()
            return
        }
        openChat(id: chatID)
    }

    func cycleSessionTab(delta: Int) {
        guard let target = sessionTabs.chatID(cycling: delta) else { return }
        activateSessionTab(chatID: target)
    }

    /// Closes a tab. Closing the selected one first moves to its neighbor, so
    /// a refused move keeps the tab. Closing never stops a turn: the chat
    /// keeps running with or without a tab. An EMPTY draft is discarded with
    /// its row (nothing to reopen); anything else is remembered for reopen,
    /// and a draft with text keeps its row and its text.
    func closeSessionTab(chatID: UUID) {
        guard sessionTabs.contains(chatID) else { return }
        if let neighbor = sessionTabs.neighborAfterClosing(chatID) {
            guard canOpenChat(id: neighbor) else {
                announceSessionTabRefused()
                return
            }
            openChat(id: neighbor)
            // The move can still have been refused (ZP-23 re-checks inside
            // `selectChat`); keep the tab rather than close the chat on screen.
            guard selectedChatID == neighbor else { return }
        }
        let chat = chats.first(where: { $0.id == chatID })
        let discard = chat.map(isEmptyDraftChat) ?? false
        let remember = !discard && chat != nil
        mutateSessionTabs { $0.close(chatID, remember: remember) }
        if discard { deleteChat(id: chatID) }
    }

    /// Reopens the most recent valid closed tab, or `chatID`'s entry, at its
    /// former strip position. A refused move leaves the entry in place.
    func reopenClosedSessionTab(chatID: UUID? = nil) {
        var probe = sessionTabs
        guard let entry = probe.takeClosed(chatID, isValid: isSessionTabChat) else {
            // Stale entries were dropped on the way; keep that.
            sessionTabs = probe
            return
        }
        guard canOpenChat(id: entry.chatID) else {
            announceSessionTabRefused()
            return
        }
        let leaving = isSessionTabChat(selectedChatID) ? selectedChatID : nil
        guard probe.open(entry.chatID, current: leaving, at: entry.index) == .opened else {
            announceSessionTabCapacity()
            return
        }
        let listChanged = probe.archive != sessionTabs.archive
        sessionTabs = probe
        if listChanged { persistSessionTabs() }
        openChat(id: entry.chatID)
    }

    /// The tab's badge: the chat's own ZP-02 state.
    func sessionTabBadge(for chatID: UUID) -> WorkspaceTabBadge {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return .none }
        return WorkspaceTabBadge.resolve(slots: [chatRowSlot(for: chat)])
    }

    // MARK: - Internals

    private func announceSessionTabRefused() {
        showToast(
            AppStrings.localized("Finish or stop the running turn before switching tabs."),
            style: .warning)
    }

    private func announceSessionTabCapacity() {
        showToast(
            AppStrings.localized(
                "You can keep up to \(SessionTabStore.maxTabs) session tabs open. Close one first."),
            style: .warning)
    }
}

extension AppModel {
    /// The title a tab or a Recently Closed row shows. An untitled chat shows
    /// the localized default name rather than the stored English one.
    func sessionTabTitle(for chatID: UUID) -> String {
        guard let chat = chats.first(where: { $0.id == chatID }) else { return "" }
        return chat.title == "New Chat" ? AppStrings.localized("New Chat") : chat.title
    }
}
