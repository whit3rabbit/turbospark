import Foundation

/// Opt-in follow-up chips under the newest reply (component audit,
/// suggestion-chips). The chips are fixed templates, never generated: asking
/// the loaded model for suggestions would cost a second pass on the same GPU
/// the user just waited on. A chip only fills the composer draft.
enum FollowUpSuggestions {
    /// Catalog keys, shown and drafted through `AppStrings.localized`. Kept
    /// to three general asks that fit any reply from a local assistant.
    static let templates = [
        "Explain that more simply",
        "Give an example",
        "Summarize in 3 bullets",
    ]

    /// The row that shows the chips: the row whose action bar is pinned, so
    /// it inherits the same rules (nil while busy, nil when the newest prompt
    /// has no reply, nil in an empty chat, whose example cards already cover
    /// what to ask first). Nil when the setting is off.
    static func rowID(in messages: [AppChatMessage], isBusy: Bool, isEnabled: Bool) -> UUID? {
        guard isEnabled else { return nil }
        return MessageActionBarPin.pinnedMessageID(in: messages, isRunning: isBusy)
    }
}
