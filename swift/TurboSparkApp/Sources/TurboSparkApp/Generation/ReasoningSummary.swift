import Foundation

/// The one-line summary a collapsed reasoning header shows after
/// "Thought for N".
///
/// Local models emit one raw reasoning blob with no step structure, so this
/// only quotes the trace: its first markdown heading (models that outline
/// their thinking start sections that way), else its first sentence. It
/// never synthesizes a summary, which would need a second model pass.
enum ReasoningSummary {
    /// Characters kept before the ellipsis. The header truncates again to
    /// the row width; this bound keeps VoiceOver from reading a paragraph.
    static let defaultLimit = 80

    /// Lines scanned for a heading. A heading deep in a long trace describes
    /// a late step, not the thought as a whole.
    static let headingScanLines = 12

    /// Nil when the trace has no readable text.
    static func headline(of reasoning: String, limit: Int = defaultLimit) -> String? {
        let lines = reasoning.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        let scanned = lines.prefix(headingScanLines)
        if let heading = scanned.lazy.compactMap(headingText).first {
            return truncate(heading, limit: limit)
        }
        guard let first = lines.lazy.map(clean).first(where: { !$0.isEmpty }) else { return nil }
        return truncate(firstSentence(of: first), limit: limit)
    }

    /// "## Title" or a line wholly in bold ("**Title**"). Nil otherwise or
    /// when the marker wraps nothing.
    private static func headingText(_ line: String) -> String? {
        if line.hasPrefix("#") {
            let rest = line.drop(while: { $0 == "#" })
            guard line.count - rest.count <= 6, rest.first == " " else { return nil }
            let text = clean(String(rest))
            return text.isEmpty ? nil : text
        }
        if line.hasPrefix("**"), line.hasSuffix("**"), line.count > 4 {
            let text = clean(String(line.dropFirst(2).dropLast(2)))
            return text.isEmpty ? nil : text
        }
        return nil
    }

    /// Drops emphasis and code markers. Underscores stay: they are usually
    /// part of an identifier (`read_file`), not emphasis.
    private static func clean(_ text: String) -> String {
        let stripped = text.filter { $0 != "*" && $0 != "`" }
        return stripped.drop(while: { $0 == "#" || $0 == " " })
            .trimmingCharacters(in: .whitespaces)
    }

    /// Up to and including the first ".", "?" or "!" that is followed by
    /// whitespace, so "1.82" or "e.g" mid-word does not end the sentence.
    private static func firstSentence(of line: String) -> String {
        var index = line.startIndex
        while index < line.endIndex {
            let next = line.index(after: index)
            if ".?!".contains(line[index]), next == line.endIndex || line[next].isWhitespace {
                return String(line[..<next])
            }
            index = next
        }
        return line
    }

    /// Cuts at the last word boundary within `limit` and adds an ellipsis;
    /// an unbroken run is cut at `limit`.
    private static func truncate(_ text: String, limit: Int) -> String {
        guard limit > 0, text.count > limit else { return text }
        let window = text.prefix(limit)
        let cut = window.lastIndex(where: \.isWhitespace).map { window[..<$0] } ?? window
        let body = cut.trimmingCharacters(in: .whitespaces)
        return (body.isEmpty ? String(window) : body) + "\u{2026}"
    }
}
