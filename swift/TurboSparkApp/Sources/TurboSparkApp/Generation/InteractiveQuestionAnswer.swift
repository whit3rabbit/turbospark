import Foundation

/// The answer string one AskUserQuestion question sends, including the
/// card's "Other" free text. It returns the same comma-joined label string
/// the card always sent, so the waiter, the answers map and the tool result
/// keep their shape: free text is simply one more answer in that string.
enum InteractiveQuestionAnswer {
    /// The typed text with outer whitespace removed, or nil when nothing is
    /// left: a blank field is no answer, never an empty one.
    static func freeText(_ raw: String?) -> String? {
        guard let trimmed = raw?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty
        else { return nil }
        return trimmed
    }

    /// One question's answer, or nil when nothing is picked or typed.
    /// Single-select: typed text wins over a pick (the card clears one when
    /// the other is chosen, so both are set only transiently). Multi-select:
    /// sorted picked labels, then the typed text. Text that equals an option
    /// label, ignoring case, IS that option, spelled as the model wrote it,
    /// so the model never sees the same choice twice or in two spellings.
    static func compose(
        selected: Set<String>, freeText raw: String?, options: [UserQuestionOption], multiSelect: Bool
    ) -> String? {
        let typed = freeText(raw)
        let matched = typed.flatMap { text in
            options.first { $0.label.caseInsensitiveCompare(text) == .orderedSame }?.label
        }
        if !multiSelect {
            if let typed { return matched ?? typed }
            return selected.sorted().first
        }
        var labels = selected
        if let matched { labels.insert(matched) }
        var parts = labels.sorted()
        if let typed, matched == nil { parts.append(typed) }
        return parts.isEmpty ? nil : parts.joined(separator: ", ")
    }
}
