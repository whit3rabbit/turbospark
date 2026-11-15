import Foundation

/// OC-08: turns a streaming assistant reply into sentences worth speaking.
///
/// It is fed the CUMULATIVE reply text and returns only the sentences that
/// became complete since the last call, so the call can speak sentence by
/// sentence while the reply streams. What it will not read aloud:
///
/// - fenced code, replaced once per block by a short spoken placeholder;
/// - file paths and URLs, replaced inside the sentence;
/// - markdown scaffolding (heading marks, bullets, emphasis, backticks) and
///   table rows, which are noise when spoken.
///
/// Tool calls and tool output never reach here: the call reads the assistant's
/// own text, and tool results live in separate message fields. A reply that
/// restarts (the agent loop clears the live buffer between steps) resets the
/// planner by itself.
struct VoiceCallSpeechPlanner {
    struct Placeholders: Equatable {
        var code: String
        var path: String
        var link: String

        static var localized: Placeholders {
            Placeholders(
                code: AppStrings.localized("Code block skipped."),
                path: AppStrings.localized("a file path"),
                link: AppStrings.localized("a link"))
        }
    }

    private let placeholders: Placeholders
    private var previous = ""
    /// Characters of `previous` already spoken or deliberately skipped.
    private var consumed = 0

    init(placeholders: Placeholders) { self.placeholders = placeholders }

    mutating func reset() {
        previous = ""
        consumed = 0
    }

    /// - Parameter final: the reply is complete, so a trailing fragment with
    ///   no terminator and an unclosed code fence are both flushed.
    mutating func ingest(_ text: String, final: Bool) -> [String] {
        // The live buffer was cleared or rewritten (agent loop step): start over.
        if !text.hasPrefix(previous) { consumed = 0 }
        previous = text
        let chars = Array(text)
        let count = chars.count
        var out: [String] = []
        var i = min(consumed, count)

        while i < count {
            if Self.isFence(chars, at: i) {
                if let close = Self.nextFence(chars, from: i + 3) {
                    var j = close + 3
                    while j < count, chars[j] != "\n" { j += 1 }
                    out.append(placeholders.code)
                    i = min(count, j + 1)
                    consumed = i
                    continue
                } else if final {
                    out.append(placeholders.code)
                    i = count
                    consumed = count
                    break
                } else {
                    // An open fence: hold everything after it until it closes.
                    break
                }
            }
            let proseEnd = Self.nextFence(chars, from: i) ?? count
            var unitStart = i
            var k = i
            while k < proseEnd {
                let c = chars[k]
                if c == "\n" {
                    emit(chars[unitStart..<k], into: &out)
                    unitStart = k + 1
                    consumed = unitStart
                } else if c == "." || c == "!" || c == "?" {
                    let nextIsBreak = k + 1 < proseEnd ? chars[k + 1].isWhitespace : (k + 1 == proseEnd && (proseEnd < count || final))
                    if nextIsBreak, !(c == "." && Self.endsAbbreviation(chars, periodAt: k, from: unitStart)) {
                        emit(chars[unitStart...k], into: &out)
                        unitStart = k + 1
                        consumed = unitStart
                    }
                }
                k += 1
            }
            if proseEnd < count || final {
                emit(chars[unitStart..<proseEnd], into: &out)
                consumed = proseEnd
            } else {
                // The tail may still grow into a longer sentence: hold it.
                consumed = unitStart
            }
            i = proseEnd
            if proseEnd >= count { break }
        }
        if final { consumed = count }
        return out
    }

    private func emit(_ slice: ArraySlice<Character>, into out: inout [String]) {
        if let sentence = speakable(String(slice)) { out.append(sentence) }
    }

    // MARK: Cleaning

    private func speakable(_ raw: String) -> String? {
        var s = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        // A table row reads as punctuation soup.
        if s.hasPrefix("|") { return nil }
        s = s.replacingOccurrences(of: #"^(?:#{1,6}\s+|[-*+]\s+|>\s*)"#, with: "", options: .regularExpression)
        // The last character is not punctuation, so a sentence's own period stays.
        s = s.replacingOccurrences(of: #"https?://\S*[^\s.,;:!?)]"#, with: placeholders.link, options: .regularExpression)
        // Absolute, home and dot-relative paths, then relative paths that end
        // in a file extension. "and/or" is neither.
        s = s.replacingOccurrences(
            of: #"(?<![\w/])(?:~|\.{1,2})?/(?:[\w.@+\-]+/)*[\w.@+\-]*[\w@+\-]/?"#,
            with: placeholders.path, options: .regularExpression)
        s = s.replacingOccurrences(
            of: #"(?<![\w/])(?:[\w.@+\-]+/)+[\w\-]+(?:\.[\w\-]+)*\.[A-Za-z0-9]{1,6}\b"#,
            with: placeholders.path, options: .regularExpression)
        for mark in ["**", "__", "`"] { s = s.replacingOccurrences(of: mark, with: "") }
        s = s.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespaces)
        guard s.contains(where: { $0.isLetter || $0.isNumber }) else { return nil }
        return s
    }

    // MARK: Scanning

    private static func isFence(_ chars: [Character], at index: Int) -> Bool {
        index + 2 < chars.count && chars[index] == "`" && chars[index + 1] == "`" && chars[index + 2] == "`"
    }

    private static func nextFence(_ chars: [Character], from start: Int) -> Int? {
        var i = max(0, start)
        while i + 2 < chars.count {
            if isFence(chars, at: i) { return i }
            i += 1
        }
        return nil
    }

    private static let abbreviations: Set<String> = ["e.g", "i.e", "vs", "etc", "mr", "mrs", "ms", "dr", "st", "no"]

    /// A period that closes "1.", "v2.", "e.g." or "Dr." is not a sentence end.
    private static func endsAbbreviation(_ chars: [Character], periodAt index: Int, from unitStart: Int) -> Bool {
        var start = index
        while start > unitStart, !chars[start - 1].isWhitespace { start -= 1 }
        let token = String(chars[start..<index]).lowercased()
        if token.isEmpty { return false }
        if token.allSatisfy(\.isNumber) { return true }
        return abbreviations.contains(token)
    }
}
