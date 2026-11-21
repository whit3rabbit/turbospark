import Foundation

extension AppModel {
    /// Appends `text` to the composer draft as a markdown blockquote and
    /// moves focus to the composer. Never sends: the user writes the reply.
    /// Returns false when there is nothing to quote.
    @discardableResult
    func quoteInReply(_ text: String) -> Bool {
        guard let quote = MessageQuote.blockquote(text) else { return false }
        // Programmatic insert: a long quote must not be converted into a
        // paste attachment chip.
        writePromptTextDirectly(MessageQuote.appending(quote: quote, to: promptText))
        NotificationCenter.default.post(name: .focusPrompt, object: nil)
        return true
    }
}
