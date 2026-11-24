import Foundation

/// Formats a transcript message as a markdown blockquote for the composer
/// ("Quote in reply", component audit selection-toolbar).
///
/// SwiftUI `Text` with `.textSelection(.enabled)` never exposes the selected
/// range, so the quote is the whole message; the user trims it in the
/// composer. Pure so the formatting rules are tested without a view.
enum MessageQuote {
    /// A long reply quoted whole would bury the user's own words and eat the
    /// context window; past this many characters the quote is cut.
    static let maxQuotedCharacters = 2_000

    /// The quote, or nil when the text has nothing to quote.
    static func blockquote(_ text: String, maxCharacters: Int = maxQuotedCharacters) -> String? {
        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        var lines = trimmedLines(normalized)
        guard !lines.isEmpty, maxCharacters > 0 else { return nil }

        // The cap counts the trimmed text, so outer blank lines and trailing
        // spaces never trigger a cut on their own.
        let joined = lines.joined(separator: "\n")
        let truncated = joined.count > maxCharacters
        if truncated {
            // Re-trim after the cut: it can end on spaces or a blank line.
            // Character-based, so no grapheme cluster is split.
            lines = trimmedLines(String(joined.prefix(maxCharacters)))
        }

        if truncated {
            if let openFence = unclosedFence(in: lines) {
                // An unclosed fence would turn the rest of the draft into
                // code, so close it and put the ellipsis outside.
                lines.append(openFence)
                lines.append("\u{2026}")
            } else {
                lines[lines.count - 1] += "\u{2026}"
            }
        }

        // Existing "> " lines become "> > ", a nested quote, which is what
        // quoting a quote means in markdown. Blank lines keep a bare marker
        // so the blockquote does not end early.
        return lines.map { $0.isEmpty ? ">" : "> " + $0 }.joined(separator: "\n")
    }

    /// The draft with the quote appended after it, and a blank line left for
    /// the user's reply. Nothing is sent.
    static func appending(quote: String, to draft: String) -> String {
        let head = draft.trimmingTrailingWhitespace()
        return head.isEmpty ? quote + "\n\n" : head + "\n\n" + quote + "\n\n"
    }

    /// Lines with trailing whitespace stripped and outer blank lines dropped.
    private static func trimmedLines(_ text: String) -> [String] {
        var lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map { String($0).trimmingTrailingWhitespace() }
        while lines.first?.isEmpty == true { lines.removeFirst() }
        while lines.last?.isEmpty == true { lines.removeLast() }
        return lines
    }

    /// The opening fence marker still open at the end, if any. CommonMark:
    /// up to three leading spaces, three or more backticks or tildes, closed
    /// by the same character at least as long.
    private static func unclosedFence(in lines: [String]) -> String? {
        var open: String?
        for line in lines {
            let indent = line.prefix(while: { $0 == " " }).count
            guard indent <= 3 else { continue }
            let body = line.dropFirst(indent)
            guard let first = body.first, first == "`" || first == "~" else { continue }
            let run = String(body.prefix(while: { $0 == first }))
            guard run.count >= 3 else { continue }
            if let current = open {
                let rest = body.dropFirst(run.count)
                if run.first == current.first, run.count >= current.count,
                   rest.allSatisfy({ $0 == " " || $0 == "\t" }) {
                    open = nil
                }
            } else {
                open = run
            }
        }
        return open
    }
}

private extension String {
    func trimmingTrailingWhitespace() -> String {
        var end = endIndex
        while end > startIndex {
            let previous = index(before: end)
            guard self[previous].isWhitespace else { break }
            end = previous
        }
        return String(self[..<end])
    }
}
