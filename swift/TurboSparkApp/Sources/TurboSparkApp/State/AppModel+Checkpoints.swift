import Foundation

/// What the transcript knows about checkpoints: the manifests of the chats it
/// has shown, and the restore waiting for confirmation.
@MainActor
final class TurnCheckpointIndex: ObservableObject {
    /// chat id -> turn id (the user message that opened the turn) -> manifest.
    @Published private(set) var manifests: [UUID: [UUID: TurnCheckpointManifest]] = [:]
    @Published var pendingRestore: PendingRestore?
    private var loadedChats: Set<UUID> = []

    struct PendingRestore: Identifiable, Equatable {
        var chatID: UUID
        var turnID: UUID
        var plan: TurnCheckpointRestorePlan
        var id: UUID { turnID }
    }

    func manifest(chatID: UUID, turnID: UUID) -> TurnCheckpointManifest? {
        manifests[chatID]?[turnID]
    }

    /// For a view body: starts the chat's one-time load and answers from
    /// what is in memory. The load publishes later, never during the update.
    func manifestLoading(chatID: UUID, turnID: UUID) -> TurnCheckpointManifest? {
        if !loadedChats.contains(chatID) { loadIfNeeded(chatID: chatID) }
        return manifests[chatID]?[turnID]
    }

    func record(_ manifest: TurnCheckpointManifest) {
        manifests[manifest.chatID, default: [:]][manifest.turnID] = manifest
    }

    /// Loads a chat's checkpoints once per process, off the main thread: a
    /// capture can hold the store lock while it copies a large file.
    func loadIfNeeded(chatID: UUID, store: TurnCheckpointStore = .shared) {
        guard !loadedChats.contains(chatID) else { return }
        loadedChats.insert(chatID)
        Task.detached(priority: .utility) {
            let loaded = store.manifests(chatID: chatID)
            await MainActor.run { [weak self] in
                guard let self else { return }
                var byTurn = self.manifests[chatID] ?? [:]
                for manifest in loaded where byTurn[manifest.turnID] == nil {
                    byTurn[manifest.turnID] = manifest
                }
                self.manifests[chatID] = byTurn
            }
        }
    }

    func forget(chatID: UUID) {
        manifests[chatID] = nil
        loadedChats.remove(chatID)
        if pendingRestore?.chatID == chatID { pendingRestore = nil }
    }
}

extension AppModel {
    /// Wires `AppToolRegistry.execute` to the store. The turn is looked up on
    /// the main actor; the copy itself runs off it.
    func installTurnCheckpointCapture(store: TurnCheckpointStore = .shared) {
        AppToolRegistry.turnCheckpointCapturer = { [weak self] call, rootURL, chatID in
            guard let self, let turnID = await self.checkpointTurnID(forChat: chatID) else { return }
            let paths = TurnCheckpointPolicy.targetPaths(for: call)
            guard !paths.isEmpty,
                  let manifest = store.capture(paths: paths, rootURL: rootURL, chatID: chatID, turnID: turnID)
            else { return }
            await MainActor.run { [weak self] in self?.turnCheckpoints.record(manifest) }
        }
    }

    /// The turn a file change belongs to: the chat's newest user message.
    /// Nil, so nothing is captured, for a Ghost chat or a protected profile:
    /// blobs are plaintext copies, and both promise nothing lands on disk.
    func checkpointTurnID(forChat chatID: UUID) -> UUID? {
        guard !ProfileVaultCoordinator.shared.isProtected,
              let chat = chats.first(where: { $0.id == chatID }), !chat.isGhost
        else { return nil }
        return chat.messages.last(where: { $0.role == .user })?.id
    }

    /// Whether Restore is offered for a chat at all.
    func checkpointsAvailable(inChat chatID: UUID) -> Bool {
        guard !ProfileVaultCoordinator.shared.isProtected,
              let chat = chats.first(where: { $0.id == chatID })
        else { return false }
        return !chat.isGhost
    }

    /// Builds the plan and asks for confirmation. Nothing is written here.
    func requestCheckpointRestore(chatID: UUID, turnID: UUID, store: TurnCheckpointStore = .shared) {
        guard let refusal = checkpointRestoreRefusal(chatID: chatID, turnID: turnID) else {
            guard let plan = store.restorePlan(chatID: chatID, turnID: turnID) else {
                showToast(AppStrings.localized("This checkpoint could not be read."), style: .error)
                return
            }
            guard !plan.items.isEmpty else {
                showToast(AppStrings.localized("Nothing to restore for this checkpoint."), style: .info)
                return
            }
            turnCheckpoints.pendingRestore = .init(chatID: chatID, turnID: turnID, plan: plan)
            return
        }
        showToast(refusal, style: .warning)
    }

    /// Why a restore cannot run now, or nil when it can. Checked again on
    /// confirm, because a turn can start while the dialog is open.
    func checkpointRestoreRefusal(chatID: UUID, turnID: UUID) -> String? {
        if isRunning {
            return AppStrings.localized("Wait for the current turn to finish before restoring.")
        }
        guard checkpointsAvailable(inChat: chatID),
              let manifest = turnCheckpoints.manifest(chatID: chatID, turnID: turnID)
        else { return AppStrings.localized("This checkpoint could not be read.") }
        guard let root = project(forChat: chatID)?.rootDirectoryURL,
              PathContainment.canonical(root).path == manifest.rootPath
        else { return AppStrings.localized("This checkpoint belongs to a different project folder.") }
        return nil
    }

    /// Runs the confirmed restore. Chat messages are not touched; a toast
    /// says what changed on disk.
    /// `pending` comes from the dialog's presented value: the dialog clears
    /// `pendingRestore` as it dismisses, which can be before this runs.
    func confirmCheckpointRestore(
        _ pending: TurnCheckpointIndex.PendingRestore,
        store: TurnCheckpointStore = .shared,
        trash: @escaping @Sendable (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }
    ) async {
        turnCheckpoints.pendingRestore = nil
        if let refusal = checkpointRestoreRefusal(chatID: pending.chatID, turnID: pending.turnID) {
            showToast(refusal, style: .warning)
            return
        }
        let chatID = pending.chatID
        let turnID = pending.turnID
        let outcome = await Task.detached(priority: .userInitiated) {
            store.restore(chatID: chatID, turnID: turnID, trash: trash)
        }.value
        guard let outcome else {
            showToast(AppStrings.localized("This checkpoint could not be read."), style: .error)
            return
        }
        if let manifest = store.manifest(chatID: chatID, turnID: turnID) {
            turnCheckpoints.record(manifest)
        }
        showToast(
            TurnCheckpointRestoreText.summary(outcome),
            style: outcome.failed.isEmpty && outcome.skipped.isEmpty ? .success : .warning,
            duration: 6)
    }

    /// Drops a deleted chat's checkpoints, on disk and in memory.
    func deleteTurnCheckpoints(chatID: UUID, store: TurnCheckpointStore = .shared) {
        turnCheckpoints.forget(chatID: chatID)
        Task.detached(priority: .utility) { store.delete(chatID: chatID) }
    }
}

/// Copy for the confirmation and the toast, kept out of the view so it can
/// be tested.
enum TurnCheckpointRestoreText {
    static let listedItemLimit = 12

    static func line(_ item: TurnCheckpointRestorePlan.Item) -> String {
        let path = item.relativePath
        let base: String
        switch item.action {
        case .overwrite: base = AppStrings.localized("Overwrite: \(path)")
        case .recreate: base = AppStrings.localized("Recreate: \(path)")
        case .trash: base = AppStrings.localized("Move to Trash: \(path)")
        case .cannotRestore: base = AppStrings.localized("Cannot restore: \(path)")
        }
        return item.changedInLaterTurn
            ? AppStrings.localized("\(base) (changed again in a later turn)") : base
    }

    /// Actionable files first, then the ones that cannot be restored, capped
    /// so a large turn does not make a dialog taller than the screen.
    static func message(_ plan: TurnCheckpointRestorePlan) -> String {
        let ordered = plan.actionable + plan.unrestorable
        var lines = ordered.prefix(listedItemLimit).map(line)
        if ordered.count > listedItemLimit {
            lines.append(AppStrings.localized("More files not shown: \(ordered.count - listedItemLimit)"))
        }
        lines.append("")
        lines.append(AppStrings.localized("Edits made by shell commands are not in the checkpoint. Chat messages are kept."))
        return lines.joined(separator: "\n")
    }

    static func summary(_ outcome: TurnCheckpointStore.RestoreOutcome) -> String {
        let notRestored = outcome.failed.count + outcome.skipped.count
        if notRestored == 0 {
            return AppStrings.localized(
                "Checkpoint restored: \(outcome.restored.count) written, \(outcome.trashed.count) moved to the Trash.")
        }
        return AppStrings.localized(
            "Checkpoint restored: \(outcome.restored.count) written, \(outcome.trashed.count) moved to the Trash, \(notRestored) not restored.")
    }
}
