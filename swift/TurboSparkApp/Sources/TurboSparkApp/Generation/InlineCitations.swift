import Foundation

/// Inline `[n]` citation markers in a finished assistant reply (component
/// audit, inline-citation).
///
/// A marker is only a CLAIM by the model that a sentence came from source n.
/// Small local models emit these unreliably and sometimes make them up, so a
/// marker becomes a link only when n names one of the turn's listed sources
/// (`1...sourceCount`), and only in prose: code spans, fences, indented
/// code, markdown links, defined reference labels and table rows keep their
/// brackets exactly as written. When unsure, the text is left alone.
///
/// Pure, one pass over the text in practice, and called only for a committed
/// reply row that already has sources, so the streaming row never pays for it.
enum InlineCitations {
    /// The scheme of the rewritten link. The transcript's `openURL` handler
    /// claims it; it is never handed to the system.
    static let scheme = "turbospark-cite"

    /// A marker's source number and its character offsets in the text.
    struct Marker: Equatable {
        let number: Int
        let range: Range<Int>
    }

    /// The reply with its markers rewritten as citation links, and the cited
    /// source numbers in first-mention order without repeats.
    struct Rendering: Equatable {
        let markdown: String
        let numbers: [Int]
    }

    /// The valid markers of `text`, in order.
    static func markers(in text: String, sourceCount: Int) -> [Marker] {
        guard sourceCount > 0, text.contains("[") else { return [] }
        let chars = Array(text)
        let lines = lineRanges(chars)
        let labels = definedLabels(chars, lines: lines)
        var found: [Marker] = []
        for segment in proseSegments(chars, lines: lines) {
            scan(chars, segment, labels: labels, sourceCount: sourceCount, into: &found)
        }
        return found
    }

    /// `text` with each valid marker replaced by a link to its source, or nil
    /// when there is no valid marker (the caller then renders `text` as is).
    static func apply(to text: String, sourceCount: Int) -> Rendering? {
        let found = markers(in: text, sourceCount: sourceCount)
        guard !found.isEmpty else { return nil }
        let chars = Array(text)
        var output = ""
        var cursor = 0
        var numbers: [Int] = []
        for marker in found {
            output += String(chars[cursor..<marker.range.lowerBound])
            // Escaped brackets keep the visible "[n]" and cannot be read as
            // a nested link or reference by the markdown parser.
            output += "[\\[\(marker.number)\\]](\(scheme):\(marker.number))"
            cursor = marker.range.upperBound
            if !numbers.contains(marker.number) { numbers.append(marker.number) }
        }
        output += String(chars[cursor...])
        return Rendering(markdown: output, numbers: numbers)
    }

    /// The source number a citation link names, or nil for any other URL or
    /// a number outside `1...sourceCount`.
    static func number(from url: URL, sourceCount: Int) -> Int? {
        guard sourceCount > 0, url.scheme?.lowercased() == scheme else { return nil }
        let rest = url.absoluteString.dropFirst(scheme.count + 1)
        guard !rest.isEmpty, rest.count <= 3, rest.allSatisfy(\.isASCIIDigit),
              let number = Int(rest), (1...sourceCount).contains(number)
        else { return nil }
        return number
    }

    // MARK: Block level

    /// Character ranges of each line, without the newline.
    private static func lineRanges(_ chars: [Character]) -> [Range<Int>] {
        var ranges: [Range<Int>] = []
        var start = 0
        for (index, char) in chars.enumerated() where char.isNewline {
            ranges.append(start..<index)
            start = index + 1
        }
        ranges.append(start..<chars.count)
        return ranges
    }

    private static func leadingSpaces(_ chars: [Character], _ line: Range<Int>) -> Int {
        var count = 0
        for index in line {
            switch chars[index] {
            case " ": count += 1
            case "\t": count += 4
            default: return count
            }
        }
        return count
    }

    private static func isBlank(_ chars: [Character], _ line: Range<Int>) -> Bool {
        line.allSatisfy { chars[$0].isWhitespace }
    }

    /// The fence a line opens or closes: its character and run length.
    private static func fence(_ chars: [Character], _ line: Range<Int>) -> (Character, Int)? {
        guard leadingSpaces(chars, line) <= 3 else { return nil }
        let start = line.first { chars[$0] != " " } ?? line.upperBound
        guard start < line.upperBound, chars[start] == "`" || chars[start] == "~" else { return nil }
        let mark = chars[start]
        var run = 0
        while start + run < line.upperBound, chars[start + run] == mark { run += 1 }
        return run >= 3 ? (mark, run) : nil
    }

    /// `[label]: destination` at the start of a line.
    private static func definitionLabel(_ chars: [Character], _ line: Range<Int>) -> String? {
        guard leadingSpaces(chars, line) <= 3,
              let open = line.first(where: { chars[$0] != " " }), chars[open] == "[",
              let close = (open + 1..<line.upperBound).first(where: { chars[$0] == "]" }),
              close > open + 1, close + 1 < line.upperBound, chars[close + 1] == ":"
        else { return nil }
        return String(chars[(open + 1)..<close]).lowercased()
    }

    private static func isTableRow(_ chars: [Character], _ line: Range<Int>) -> Bool {
        line.contains { chars[$0] == "|" }
    }

    /// Lines walked once for fences, so a label defined inside code is not
    /// mistaken for a reference definition.
    private static func definedLabels(_ chars: [Character], lines: [Range<Int>]) -> Set<String> {
        var labels = Set<String>()
        var openFence: (Character, Int)?
        for line in lines {
            if let open = openFence {
                if let close = fence(chars, line), close.0 == open.0, close.1 >= open.1 { openFence = nil }
                continue
            }
            if let open = fence(chars, line) { openFence = open; continue }
            if let label = definitionLabel(chars, line) { labels.insert(label) }
        }
        return labels
    }

    /// Runs of consecutive prose lines (a paragraph or list), as character
    /// ranges. A code span can cross a line break inside one, never across.
    private static func proseSegments(_ chars: [Character], lines: [Range<Int>]) -> [Range<Int>] {
        var segments: [Range<Int>] = []
        var current: Range<Int>?
        var openFence: (Character, Int)?
        var previousBlank = true
        var inIndentedCode = false

        func flush() {
            if let segment = current { segments.append(segment) }
            current = nil
        }

        for line in lines {
            if let open = openFence {
                if let close = fence(chars, line), close.0 == open.0, close.1 >= open.1 { openFence = nil }
                previousBlank = false
                continue
            }
            if let open = fence(chars, line) {
                flush()
                openFence = open
                previousBlank = false
                continue
            }
            if isBlank(chars, line) {
                flush()
                previousBlank = true
                continue
            }
            // Indented code cannot interrupt a paragraph, so only a line
            // after a blank (or another code line) starts or continues it.
            // A list's indented continuation is caught here too, which only
            // costs a citation link, never a wrong rewrite.
            let indented = leadingSpaces(chars, line) >= 4
            if indented && (previousBlank || inIndentedCode) {
                flush()
                inIndentedCode = true
                previousBlank = false
                continue
            }
            inIndentedCode = false
            previousBlank = false
            if definitionLabel(chars, line) != nil || isTableRow(chars, line) {
                flush()
                continue
            }
            if let segment = current {
                current = segment.lowerBound..<line.upperBound
            } else {
                current = line
            }
        }
        flush()
        return segments
    }

    // MARK: Inline level

    private static func scan(
        _ chars: [Character],
        _ segment: Range<Int>,
        labels: Set<String>,
        sourceCount: Int,
        into found: inout [Marker]
    ) {
        let end = segment.upperBound
        var index = segment.lowerBound
        while index < end {
            let char = chars[index]
            switch char {
            case "\\":
                index += 2
            case "`":
                index = skipCodeSpan(chars, from: index, end: end)
            case "<":
                index = skipAutolink(chars, from: index, end: end)
            case "[":
                index = handleBracket(
                    chars, at: index, end: end, labels: labels, sourceCount: sourceCount, into: &found)
            default:
                index += 1
            }
        }
    }

    /// Past a code span opened at `start`, or one past its backticks when
    /// no closing run of the same length follows (then they are literal).
    private static func skipCodeSpan(_ chars: [Character], from start: Int, end: Int) -> Int {
        var run = 0
        while start + run < end, chars[start + run] == "`" { run += 1 }
        var index = start + run
        while index < end {
            guard chars[index] == "`" else { index += 1; continue }
            var closing = 0
            while index + closing < end, chars[index + closing] == "`" { closing += 1 }
            if closing == run { return index + closing }
            index += closing
        }
        return start + run
    }

    /// Past `<scheme:...>` or `<user@host>`, else one character.
    private static func skipAutolink(_ chars: [Character], from start: Int, end: Int) -> Int {
        var index = start + 1
        var sawScheme = false
        while index < end {
            let char = chars[index]
            if char == ">" { return sawScheme ? index + 1 : start + 1 }
            if char.isWhitespace || char == "<" { return start + 1 }
            if char == ":" || char == "@" { sawScheme = true }
            index += 1
        }
        return start + 1
    }

    /// The index of the `]` that closes the `[` at `open`, counting nesting
    /// and skipping escapes, or nil.
    private static func closingBracket(_ chars: [Character], open: Int, end: Int) -> Int? {
        var depth = 0
        var index = open
        while index < end {
            switch chars[index] {
            case "\\": index += 1
            case "[": depth += 1
            case "]":
                depth -= 1
                if depth == 0 { return index }
            default: break
            }
            index += 1
        }
        return nil
    }

    /// The index of the `)` that closes the `(` at `open`, or nil.
    private static func closingParen(_ chars: [Character], open: Int, end: Int) -> Int? {
        var depth = 0
        var index = open
        while index < end {
            switch chars[index] {
            case "\\": index += 1
            case "(": depth += 1
            case ")":
                depth -= 1
                if depth == 0 { return index }
            default: break
            }
            index += 1
        }
        return nil
    }

    /// Whether the `[` at `open` (with matching `close`) starts a link the
    /// markdown parser will render, and where that link ends.
    private static func linkEnd(
        _ chars: [Character], open: Int, close: Int, end: Int, labels: Set<String>
    ) -> Int? {
        let next = close + 1
        if next < end, chars[next] == "(",
           let paren = closingParen(chars, open: next, end: end) {
            return paren + 1
        }
        if next < end, chars[next] == "[",
           let refClose = closingBracket(chars, open: next, end: end) {
            let ref = String(chars[(next + 1)..<refClose]).lowercased()
            let text = String(chars[(open + 1)..<close]).lowercased()
            if labels.contains(ref) || (ref.isEmpty && labels.contains(text)) { return refClose + 1 }
        }
        if labels.contains(String(chars[(open + 1)..<close]).lowercased()) { return next }
        return nil
    }

    private static func handleBracket(
        _ chars: [Character],
        at open: Int,
        end: Int,
        labels: Set<String>,
        sourceCount: Int,
        into found: inout [Marker]
    ) -> Int {
        guard let close = closingBracket(chars, open: open, end: end) else { return open + 1 }
        // An image (`![...]`) is never a citation and its alt text is not prose.
        let isImage = open > 0 && chars[open - 1] == "!"
        if let linkEnd = linkEnd(chars, open: open, close: close, end: end, labels: labels) {
            return linkEnd
        }
        let inner = chars[(open + 1)..<close]
        guard !isImage, (1...3).contains(inner.count), inner.allSatisfy(\.isASCIIDigit),
              inner.first != "0", let number = Int(String(inner)),
              (1...sourceCount).contains(number)
        else {
            // Not a marker: scan inside, where plain text may hold one.
            return open + 1
        }
        found.append(Marker(number: number, range: open..<(close + 1)))
        return close + 1
    }
}

private extension Character {
    var isASCIIDigit: Bool { isASCII && isNumber }
}
