import Foundation

/// String-level preview for long message and attachment text.
///
/// `CollapsibleMessageContentView`'s frame clamp only hides the overflow
/// AFTER `Text`/MarkdownUI has laid the whole string out, and transcript
/// rows re-evaluate on every streamed token (swift/AGENTS.md Gotcha 16), so
/// a dump-sized message pays layout on every body eval forever. This gate
/// shrinks the STRING instead: past the collapse threshold a collapsed row
/// renders a head+tail window and only the expanded state receives the full
/// text. The underlying message content is never touched -- only what the
/// view is handed.
enum MessageContentPreview {
    /// Messages longer than this many characters collapse to a preview
    /// while unexpanded. Short enough that ordinary long replies never hit
    /// it; far below where Text layout starts to stutter.
    static let collapseThreshold = 6_000

    /// Leading characters a preview keeps.
    static let headLimit = 4_000
    /// Trailing characters a preview keeps, so the end of a paste (logs,
    /// stack traces) is visible without expanding.
    static let tailLimit = 500

    /// The computed preview: what to render and how much it hides.
    struct Preview: Equatable {
        let visible: String
        let hiddenCharacterCount: Int
    }

    /// Builds the preview, or nil when `text` is short enough to render in
    /// full (the caller then behaves exactly as before this gate existed).
    ///
    /// `includesTail == false` keeps the head only, for renderer paths
    /// (markdown) whose collapsed frame already shows the top of the
    /// message and whose readers expand from there.
    static func make(
        _ text: String,
        collapseThreshold: Int = collapseThreshold,
        headLimit: Int = headLimit,
        tailLimit: Int = tailLimit,
        includesTail: Bool = true
    ) -> Preview? {
        guard text.count > collapseThreshold else { return nil }
        let hiddenCharacterCount = text.count - headLimit - (includesTail ? tailLimit : 0)
        let marker = String(
            localized: "[... \(hiddenCharacterCount) characters hidden ...]",
            bundle: .module)
        if includesTail {
            let visible =
                String(text.prefix(headLimit))
                + "\n\n" + marker + "\n\n"
                + String(text.suffix(tailLimit))
            return Preview(visible: visible, hiddenCharacterCount: hiddenCharacterCount)
        }
        let visible = String(text.prefix(headLimit)) + "\n\n" + marker
        return Preview(visible: visible, hiddenCharacterCount: hiddenCharacterCount)
    }
}
