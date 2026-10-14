import Foundation

/// The context engine's lifecycle (`swift/docs/SWIFT_CONTEXT_ENGINE.md`):
/// which project's store is synced, where a chat's turns are recorded, and
/// what a deletion or a settings change removes. The per-turn evidence is
/// `AppModel+ContextEvidence`.
extension AppModel {
    /// Whether `project`'s files feed its turns: the global switch, the
    /// project's own opt-in, and a workspace to read.
    func contextEngineActive(for project: AppProject?) -> Bool {
        guard contextEngineEnabled, let project, project.contextEngineEnabled else { return false }
        return project.rootDirectoryURL != nil
    }

    /// Where `chatID`'s turns are recorded, or `nil` when recall is off for
    /// it. Ghost chats are never recorded: their content exists only in the
    /// in-memory vault, and the store is plaintext on disk.
    func contextRecallLocation(chatID: UUID) -> ContextChatLocation? {
        guard contextEngineEnabled, conversationRecallEnabled,
            let chat = chats.first(where: { $0.id == chatID }), !chat.isGhost
        else { return nil }
        if let project = project(forChat: chatID), contextEngineActive(for: project) {
            return ContextChatLocation(store: .project(project.id), chatID: chatID)
        }
        return ContextChatLocation(store: .chats, chatID: chatID)
    }

    /// Opens and syncs the selected project's store, or closes the previous
    /// one. Called wherever `refreshSyntextActivation` is.
    func refreshContextEngineActivation(for project: AppProject?) {
        let target = contextEngineActive(for: project) ? project : nil
        let id = target?.id
        let root = target?.rootDirectoryURL
        // The `context_*` tools are offered for this project only.
        AppToolRegistry.contextEngineRoot = root
        Task.detached(priority: .utility) { [weak self] in
            await self?.installContextEngineObserver()
            await ContextEngineManager.shared.activateProject(id: id, root: root)
        }
    }

    /// Mirrors the manager's sync progress into `contextEngineSyncStates`
    /// for the project settings sheet.
    func installContextEngineObserver() async {
        await ContextEngineManager.shared.setObserver { [weak self] id, state in
            Task { @MainActor [weak self] in
                self?.contextEngineSyncStates[id] = state
            }
        }
    }

    public func setContextEngineEnabled(_ enabled: Bool) {
        contextEngineEnabled = enabled
        persistSettingsDebounced()
        refreshContextEngineActivation(for: selectedProject)
    }

    /// Turning recall off deletes what it recorded. Every store is removed,
    /// since project stores hold their chats' turns too; the selected
    /// project's files are synced again.
    public func setConversationRecallEnabled(_ enabled: Bool) {
        conversationRecallEnabled = enabled
        persistSettingsDebounced()
        guard !enabled else { return }
        clearContextEngineData()
    }

    /// Deletes every context store (recall data and project indexes), then
    /// resyncs the selected project when its engine is on.
    public func clearContextEngineData() {
        contextEngineSyncStates = [:]
        Task.detached(priority: .utility) { [weak self] in
            await ContextEngineManager.shared.removeAllStores()
            await MainActor.run {
                guard let self else { return }
                self.refreshContextEngineActivation(for: self.selectedProject)
            }
        }
    }

    /// Re-reads a project from scratch.
    public func rebuildContextEngineIndex(for project: AppProject) {
        guard contextEngineActive(for: project), let root = project.rootDirectoryURL else { return }
        let id = project.id
        Task.detached(priority: .utility) { [weak self] in
            await self?.installContextEngineObserver()
            await ContextEngineManager.shared.rebuildProject(id: id, root: root)
        }
    }

    /// Deletes a project's store: the index of its files and its chats'
    /// recorded turns.
    public func removeContextEngineIndex(for projectID: UUID) {
        contextEngineSyncStates[projectID] = nil
        Task.detached(priority: .utility) {
            await ContextEngineManager.shared.removeProjectStore(id: projectID)
        }
    }

    /// Records `chatID`'s finished turn: the store is reconciled with the
    /// whole transcript, so edits and retries made since are carried too.
    func recordContextTurn(chatID: UUID) {
        guard let location = contextRecallLocation(chatID: chatID) else { return }
        let rows = ContextRecall.rows(from: turnMessages(for: chatID), beforeLastQuestion: false)
        Task.detached(priority: .utility) {
            await ContextEngineManager.shared.recordTurn(
                chatID: chatID, store: location.store, rows: rows)
        }
    }

    /// A deleted chat's recorded turns leave the store too. Called for every
    /// non-ghost deletion, whatever the settings say now: a chat recorded
    /// before recall was switched off still has turns to remove.
    func forgetContextChat(chatID: UUID, projectID: UUID?) {
        Task.detached(priority: .utility) {
            await ContextEngineManager.shared.forgetChat(chatID, projectID: projectID)
        }
    }
}
