import Foundation

/// Hides a dangling markdown marker at the very end of a reply that is
/// still streaming, so a lone `**`, `_`, backtick run or `[` does not flash
/// as literal text for one token before the model writes what it opens.
///
/// Only a marker with nothing after it is held, and only when it cannot be
/// closing something earlier on the same line: real text is never held,
/// and a finished message never goes through here. The scan looks at the
/// current line only and at most `scanLimit` characters of it, so the cost
/// per streamed token does not grow with the reply. The tradeoff of that
/// bound (and of not tracking code fences, which would need the whole
/// reply) is that a marker is sometimes held one token longer than needed;
/// it is never dropped.
enum StreamingMarkdownTail {
    static let scanLimit = 1_024

    static func visible(_ text: String) -> String {
        let scalars = text.unicodeScalars
        guard !scalars.isEmpty else { return text }
        let lastIndex = scalars.index(before: scalars.endIndex)
        let last = scalars[lastIndex]

        if last == "[" {
            return String(text.unicodeScalars[..<lastIndex])
        }
        guard last == "*" || last == "_" || last == "`" else { return text }

        // The run of the same marker at the end.
        var runStart = lastIndex
        while runStart > scalars.startIndex {
            let previous = scalars.index(before: runStart)
            guard scalars[previous] == last else { break }
            runStart = previous
        }

        // Same-marker count and the character before the run, current line
        // only, bounded.
        var countBefore = 0
        var charBefore: Unicode.Scalar?
        var index = runStart
        var steps = 0
        while index > scalars.startIndex, steps < scanLimit {
            index = scalars.index(before: index)
            let scalar = scalars[index]
            if scalar == "\n" || scalar == "\r" { break }
            if charBefore == nil { charBefore = scalar }
            if scalar == last { countBefore += 1 }
            steps += 1
        }

        let hold: Bool
        if last == "`" {
            // Code spans pair by count: an even number before means no span
            // is open, so this run opens one (or a fence).
            hold = countBefore % 2 == 0
        } else if let charBefore, !charBefore.properties.isWhitespace {
            // After a word the run may close an earlier opener; hold it only
            // when there is none on the line.
            hold = countBefore == 0
        } else {
            // After a space or at line start it cannot close anything.
            hold = true
        }
        return hold ? String(scalars[..<runStart]) : text
    }
}
