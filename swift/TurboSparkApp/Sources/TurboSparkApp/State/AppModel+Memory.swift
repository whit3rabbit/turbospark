import AppKit
import Foundation

/// The user-facing memory entry points: the `/memory` meta command and the
/// `#` quick-save.
///
/// Neither one generates, so both are intercepted in `run()` ABOVE the
/// `canRun` guard -- a quick-save with no model loaded still saves, which is
/// the point of the feature. `/compact` stays below that guard because it
/// needs the session's own model to summarize with.
extension AppModel {
    /// The project whose memory a submission acts on. The same resolution
    /// `run()`'s hook phase makes: the turn's own project when the chat has
    /// one, the selection when it is about to be created with one.
    func memoryProject() -> AppProject? {
        turnProject(chatID: selectedChatID)
            ?? (interactionMode == .projects ? selectedProject : nil)
    }

    /// `/memory`: opens the selected chat's project memory folder in Finder.
    /// There are no arguments to parse; the draft is left alone, because
    /// unlike `/compact` this command is not a submission and has nothing to
    /// clear. The opener is a parameter so a test can observe the target
    /// instead of really opening Finder.
    func handleMemoryCommand(opener: (URL) -> Void = { NSWorkspace.shared.open($0) }) {
        guard let root = memoryProject()?.rootDirectoryURL, !root.path.isEmpty else {
            showToast(
                "Attach a project first: memories are stored per project directory.",
                style: .warning)
            return
        }
        opener(MemoryStore.shared.directory(forProjectRoot: root))
    }

    /// The `#` quick-save: writes the text as a `user`-type memory and
    /// appends a `<user-memory-input>` transcript row, with no generation
    /// turn. Type `user` because the content is verbatim from the user --
    /// classifying it into feedback/project/reference is extraction work
    /// this path deliberately does not do.
    ///
    /// A failed save keeps the draft: the user's text is still in the
    /// composer, nothing was lost, and they can retry after fixing the
    /// project. A successful one clears it, like any submission.
    func handleMemoryQuickSave(_ trimmedDraft: String) {
        guard !generating, !submitting, pendingToolCall == nil, !opening else { return }
        let chatID = selectedChatID
        guard let root = memoryProject()?.rootDirectoryURL, !root.path.isEmpty else {
            showToast(
                "Attach a project first: memories are stored per project directory.",
                style: .warning)
            return
        }
        let text = UserMemoryInputMessage.memoryText(of: trimmedDraft)
        do {
            let firstLine = SkillParser.normalizedLines(text).first { !$0.trimmingCharacters(in: .whitespaces).isEmpty } ?? text
            let trimmedLine = firstLine.trimmingCharacters(in: .whitespaces)
            let hook = trimmedLine.count > 120 ? String(trimmedLine.prefix(120)) + "..." : trimmedLine
            let saved = try MemoryStore.shared.saveTopic(
                projectRoot: root,
                name: MemoryStore.slug(from: text, dated: true),
                type: .user,
                description: hook,
                body: text)
            promptText = ""
            let row = AppChatMessage(
                role: .user, content: UserMemoryInputMessage.wrapping(text))
            mutateTurnMessages(for: chatID) { $0.append(row) }
            showToast("Saved to memory as \(saved.url.lastPathComponent).", style: .success)
        } catch {
            showToast("Could not save memory: \(error.localizedDescription)", style: .error)
        }
    }
}
