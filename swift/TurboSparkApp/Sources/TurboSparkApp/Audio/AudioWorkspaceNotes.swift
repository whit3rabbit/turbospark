import Foundation

/// The user's own notes on a take. Typing stages the text and a short
/// debounce saves it through `persistMutation`, which re-reads the record,
/// so a capture autosave and a notes save never overwrite each other.
extension AudioWorkspaceController {
    /// The text to show for an item's notes: what is still being typed, else
    /// what is saved.
    func userNotes(for item: AudioLibraryItem) -> String {
        pendingUserNotes[item.id] ?? item.userNotes ?? ""
    }

    func stageUserNotes(_ text: String, for id: UUID) {
        guard !isShutDown else { return }
        pendingUserNotes[id] = text
        userNotesSaveTasks[id]?.cancel()
        let token = epoch
        userNotesSaveTasks[id] = Task { [weak self] in
            try? await Task.sleep(for: .milliseconds(600))
            guard !Task.isCancelled, let self, self.valid(token), let latest = self.pendingUserNotes[id] else { return }
            _ = self.saveUserNotes(latest, for: id)
        }
    }

    /// Saves now. Called by the debounce and when the editor loses focus, so
    /// switching takes never drops the last keystrokes.
    func flushUserNotes(for id: UUID) {
        guard let latest = pendingUserNotes[id] else { return }
        userNotesSaveTasks.removeValue(forKey: id)?.cancel()
        _ = saveUserNotes(latest, for: id)
    }

    @discardableResult func saveUserNotes(_ text: String, for id: UUID) -> Bool {
        guard !isShutDown else { return false }
        do {
            try persistMutation(id) { item in
                let value: String? = text.isEmpty ? nil : text
                guard item.userNotes != value else { return }
                item.userNotes = value
                item.updatedAt = Date()
            }
            // Keep newer typing that arrived while this save ran.
            if pendingUserNotes[id] == text { pendingUserNotes.removeValue(forKey: id) }
            return true
        } catch {
            fail(error)
            return false
        }
    }
}
