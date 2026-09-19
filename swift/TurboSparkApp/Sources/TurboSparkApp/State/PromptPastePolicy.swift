import Foundation

/// Pure policy for large pasted text in the composer (ChatGPT-style).
///
/// A paste over `attachmentThreshold` characters is lifted out of the draft
/// into a `Pasted Text.txt` attachment chip, so the plain `TextEditor` never
/// has to lay out, measure, and persist a dump-sized string on every
/// keystroke. Detection is post-hoc on the draft's before/after values
/// rather than a pasteboard hook, because SwiftUI's `TextEditor` offers no
/// paste interception and an NSTextView representable rewrite would drag the
/// autocomplete, the two-stage Escape, the history walk, and the measured
/// sizer with it (swift/AGENTS.md Gotcha 18). A paste is the only ordinary
/// gesture that grows a draft by thousands of characters in one change.
///
/// Everything here is a pure function so the thresholds and the UTF-8
/// splitting are testable without an `AppModel`.
enum PromptPastePolicy {
    /// A single draft change that inserts at least this many characters is
    /// treated as a paste and moved into an attachment. Roughly one page or
    /// about a thousand tokens; below it a paste stays inline as before.
    static let attachmentThreshold = 4_000

    /// The characters-per-token ratio every client-side estimate in this app
    /// already uses for attachments (`buildEstimateParts`,
    /// `estimatedContextTokens`).
    static let charactersPerToken = 4

    /// The result of recognizing one draft change as a paste.
    struct Split: Equatable {
        /// The draft with the pasted middle removed (the unchanged prefix
        /// and suffix of the edit), which replaces the caller's text.
        var draft: String
        /// The inserted text, the part that becomes the attachment.
        var pasted: String
    }

    /// Splits one draft change into its pasted middle, or nil when the
    /// change does not read as a paste.
    ///
    /// The edit is recovered as longest common prefix + suffix over UTF-8
    /// bytes: byte boundaries of two valid UTF-8 strings are always
    /// character boundaries (a multibyte character cannot half-match), so
    /// this is multibyte-safe and O(insert) rather than O(text). Paste over
    /// a selection and paste into the middle both come out with the
    /// untouched surroundings kept in `draft`.
    static func splitPaste(
        previous: String,
        current: String,
        threshold: Int = attachmentThreshold
    ) -> Split? {
        // Cheap gate first: character-count growth below the threshold can
        // never qualify, whatever the edit was (typing, autocomplete,
        // deletions).
        guard current.count - previous.count >= threshold else { return nil }

        let oldBytes = Array(previous.utf8)
        let newBytes = Array(current.utf8)
        let shared = min(oldBytes.count, newBytes.count)

        var prefix = 0
        while prefix < shared && oldBytes[prefix] == newBytes[prefix] {
            prefix += 1
        }
        var suffix = 0
        while suffix < shared - prefix
            && oldBytes[oldBytes.count - 1 - suffix] == newBytes[newBytes.count - 1 - suffix]
        {
            suffix += 1
        }

        let insertedByteCount = newBytes.count - prefix - suffix
        guard insertedByteCount > 0 else { return nil }
        let pasted = String(
            decoding: newBytes[prefix..<(prefix + insertedByteCount)], as: UTF8.self)
        // The threshold is stated in characters (what the chip reports and
        // what the estimates price), so a multibyte middle re-checks here.
        guard pasted.count >= threshold else { return nil }

        let draft =
            String(decoding: newBytes[..<prefix], as: UTF8.self)
            + String(decoding: newBytes[(newBytes.count - suffix)...], as: UTF8.self)
        return Split(draft: draft, pasted: pasted)
    }

    /// How pasted (or extracted) text sits against the model's free context
    /// window. `freeTokens` is the window minus the generation reserve, the
    /// same budget `fitConversationWindow` enforces at send time.
    enum ContextFit: Equatable {
        /// Under the budget; store as-is.
        case fits
        /// Over the budget; store the kept head and flag the attachment
        /// truncated, matching what `DocumentTextExtractor` does at its own
        /// 240k-character ceiling.
        case truncated(String)
        /// No room at all; the caller keeps the text inline and refuses.
        case noRoom
    }

    /// Head-only truncation of `text` to `freeTokens`.
    static func fit(_ text: String, freeTokens: Int) -> ContextFit {
        guard freeTokens > 0 else { return .noRoom }
        let allowedCharacters = freeTokens * charactersPerToken
        guard text.count > allowedCharacters else { return .fits }
        return .truncated(String(text.prefix(allowedCharacters)))
    }
}
