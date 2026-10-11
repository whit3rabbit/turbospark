import Foundation

extension AppModel {
    /// Brings the chat forward with `text` in the composer, for a surface
    /// whose items are created by asking the assistant (Scheduled Tasks has
    /// no form of its own: jobs come from the model's scheduling tool).
    ///
    /// An existing draft is kept and the text appended after it, so the
    /// shortcut can never discard what the user was typing. Nothing is sent:
    /// the user reads, edits and submits the request themselves.
    public func composeInChat(_ text: String) {
        showMainWindow(navigatingTo: .chat)
        promptText = Self.draft(appending: text, to: promptText)
        NotificationCenter.default.post(name: .focusPrompt, object: nil)
    }

    /// The draft after `composeInChat`, as a pure function for the test.
    static func draft(appending text: String, to existing: String) -> String {
        existing.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            ? text
            : existing + "\n\n" + text
    }
}
