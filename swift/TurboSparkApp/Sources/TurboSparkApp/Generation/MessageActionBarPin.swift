import Foundation

/// Which message keeps its action bar visible without hover: the newest
/// assistant reply with text. Hover-only actions are hard to find with a
/// keyboard, VoiceOver, or no trackpad, and the newest reply is where copy
/// and retry are wanted most. Every other row stays hover-only so the
/// transcript does not fill with icons.
enum MessageActionBarPin {
    /// Nil while a turn runs (the previous reply would sit above the new
    /// prompt looking like its answer) and when the newest human prompt has
    /// no reply yet, for the same reason.
    static func pinnedMessageID(in messages: [AppChatMessage], isRunning: Bool) -> UUID? {
        guard !isRunning else { return nil }
        for message in messages.reversed() {
            if message.role == .user && !message.isSynthetic { return nil }
            if message.role == .assistant
                && !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return message.id
            }
        }
        return nil
    }
}
