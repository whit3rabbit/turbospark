import Foundation

/// ZP-21: workspace tabs, the AppModel side.
///
/// A tab is a project reference plus the last chat open in it
/// (`WorkspaceTabStore`). Activating one goes through the same
/// `openChat` / `selectProject` paths the sidebar uses, so the guards, the
/// worktree rebinding and the per-chat drafts behave exactly as they do for
/// a sidebar click. There is no second context container.
///
/// **LIMITS INHERITED FROM ZP-23, ON PURPOSE.**
/// - One turn at a time app-wide, and a model shared by every chat. Tabs do
///   not add a lane or a cap.
/// - Moving to a chat under ANOTHER project keeps the blanket guard
///   (`canOpenChat`, ZP-23 remainder 4): a project switch resets the file
///   snapshot hashes, rebinds the worktree and the hook store. So a tab
///   switch is refused, with a toast, while a turn runs, is submitting, or an
///   approval card is open. Background generation therefore lives in the
///   project that is on screen; the other tabs show unread and queued state.
/// - An approval card pins you to its chat, which pins you to its tab.
extension AppModel {
    /// Whether the strip and the tab commands are live.
    public var workspaceTabsActive: Bool { workspaceTabs.isActive }

    /// Whether the app is in a state that may move to another project. The
    /// same three flags `applyProjectSelection` refuses on.
    public var canSwitchWorkspace: Bool {
        !generating && !submitting && pendingToolCall == nil
    }

    // MARK: - Persistence

    /// Restores the saved tab list at launch, after the projects load.
    /// Tabs for deleted projects are pruned, and the selection is aligned
    /// with the project the app restored (that archive is the source of
    /// truth for what is on screen).
    func loadWorkspaceTabs(from url: URL? = nil) {
        var store = WorkspaceTabStore(archive: WorkspaceTabFileStore.load(from: url))
        store.prune(validProjectIDs: Set(projects.map(\.id)))
        store.navigate(to: selectedProjectID)
        workspaceTabs = store
    }

    func persistWorkspaceTabs(to url: URL? = nil) {
        WorkspaceTabFileStore.save(workspaceTabs.archive, to: url)
    }

    /// Applies `change` to the store and writes it only when something moved.
    private func mutateWorkspaceTabs(_ change: (inout WorkspaceTabStore) -> Void) {
        var next = workspaceTabs
        change(&next)
        guard next != workspaceTabs else { return }
        workspaceTabs = next
        persistWorkspaceTabs()
    }

    // MARK: - Following the app's own selection

    /// Called from `selectedProjectID.didSet`: the sidebar, the palette and
    /// `openChat` all move the project, and the strip mirrors it.
    func workspaceTabsFollowProjectSelection() {
        mutateWorkspaceTabs { $0.navigate(to: selectedProjectID) }
    }

    /// Called from `selectedChatID.didSet`: remembers the chat for its
    /// project's tab. A chat that is not under the selected project is
    /// ignored, because a project switch assigns the project before the chat.
    func workspaceTabsNoteSelectedChat() {
        guard workspaceTabs.isActive, let projectID = selectedProjectID,
            let chat = chats.first(where: { $0.id == selectedChatID }),
            chat.projectID == projectID, !chat.isGhost
        else { return }
        let chatID = selectedChatID
        mutateWorkspaceTabs { $0.noteChat(chatID, in: projectID) }
    }

    /// Drops tabs whose project is gone (called after a project delete).
    func pruneWorkspaceTabs() {
        mutateWorkspaceTabs { $0.prune(validProjectIDs: Set(projects.map(\.id))) }
    }

    // MARK: - Commands

    /// Opens `projectID` in a tab and moves to it. "Open in New Tab" in the
    /// sidebar and the strip's plus menu both land here.
    func openWorkspaceTab(projectID: UUID) {
        guard projects.contains(where: { $0.id == projectID }) else { return }
        // Refuse before touching the list: a refused switch must not leave a
        // tab that the selection does not match.
        if projectID != selectedProjectID, !canSwitchWorkspace {
            announceWorkspaceSwitchRefused()
            return
        }
        var probe = workspaceTabs
        guard probe.open(projectID, current: selectedProjectID, currentChatID: selectedChatID) == .opened else {
            showToast(
                AppStrings.localized(
                    "You can keep up to \(WorkspaceTabStore.maxTabs) workspace tabs open. Close one first."),
                style: .warning)
            return
        }
        let leavingChat = selectedChatID
        mutateWorkspaceTabs { _ = $0.open(projectID, current: selectedProjectID, currentChatID: leavingChat) }
        if projectID != selectedProjectID { moveToWorkspace(projectID) }
    }

    /// Moves to an open tab. Refused, with a reason, while a turn runs.
    func activateWorkspaceTab(projectID: UUID) {
        guard projectID != selectedProjectID else { return }
        guard projects.contains(where: { $0.id == projectID }) else {
            pruneWorkspaceTabs()
            return
        }
        guard canSwitchWorkspace else {
            announceWorkspaceSwitchRefused()
            return
        }
        moveToWorkspace(projectID)
    }

    /// Cmd-free next/previous: wraps around the strip.
    func cycleWorkspaceTab(delta: Int) {
        guard interactionMode == .projects, let target = workspaceTabs.projectID(cycling: delta)
        else { return }
        activateWorkspaceTab(projectID: target)
    }

    /// Closes a tab. Closing the selected one first moves to its neighbor,
    /// so a refused move (a turn is running) keeps the tab. Closing never
    /// stops anything: a chat keeps its state with or without a tab.
    func closeWorkspaceTab(projectID: UUID) {
        guard workspaceTabs.contains(projectID) else { return }
        if let neighbor = workspaceTabs.neighborAfterClosing(projectID) {
            guard canSwitchWorkspace else {
                announceWorkspaceSwitchRefused()
                return
            }
            moveToWorkspace(neighbor)
        }
        mutateWorkspaceTabs { $0.remove(projectID) }
    }

    /// The tab's badge: the strongest ZP-02 state among the project's chats.
    func workspaceTabBadge(for projectID: UUID) -> WorkspaceTabBadge {
        let slots =
            chats
            .filter { $0.projectID == projectID && !$0.isGhost && !$0.isArchived }
            .map { chatRowSlot(for: $0) }
        return WorkspaceTabBadge.resolve(slots: slots)
    }

    // MARK: - Internals

    /// Activates the tab's remembered chat when it is still a live chat of
    /// that project, else the project's default selection.
    private func moveToWorkspace(_ projectID: UUID) {
        if let chatID = workspaceTabs.tab(for: projectID)?.lastChatID,
            let chat = chats.first(where: { $0.id == chatID }),
            chat.projectID == projectID, !chat.isGhost, !chat.isArchived
        {
            openChat(id: chatID)
        } else {
            selectProject(id: projectID)
        }
    }

    private func announceWorkspaceSwitchRefused() {
        showToast(
            AppStrings.localized("Finish or stop the running turn before switching projects."),
            style: .warning)
    }
}
