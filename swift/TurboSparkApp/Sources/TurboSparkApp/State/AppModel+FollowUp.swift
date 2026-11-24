import Foundation

extension AppModel {
    /// A follow-up chip's click: puts `text` in the composer (after a blank
    /// line when a draft exists, the `composeInChat` rule) and focuses it.
    /// Never sends, so the user can edit or discard the ask.
    func draftFollowUp(_ text: String) {
        // Programmatic insert, like Quote in reply, so the paste-to-chip
        // conversion never sees it.
        writePromptTextDirectly(Self.draft(appending: text, to: promptText))
        NotificationCenter.default.post(name: .focusPrompt, object: nil)
    }
}
