import Foundation

/// The parts of a code fence info string the chat code block understands:
/// the language, an optional file title and optional marked lines.
///
/// MarkdownUI hands the WHOLE info string over as the block's "language", so
/// a fence like ```swift title="Foo.swift" {3}``` used to show that entire
/// string as the tag and defeat the highlighter. There is no standard syntax
/// for these hints, so only forms that parse unambiguously are accepted:
///
/// - `lang title="path"`, `lang title=path`, `lang filename='path'`
/// - `lang:path` (the whole first word)
/// - `{3,5-7}` as its own word or right after the language (`js{4}`)
///
/// Anything else (an unknown attribute, an unclosed quote or brace, a bad
/// range) returns the raw info string as the language with no title and no
/// marks, so the block renders exactly as it did before.
struct CodeFenceInfo: Equatable {
    let language: String?
    let title: String?
    /// One-based line numbers to tint. Nil when none were given.
    let markedLines: IndexSet?

    /// Larger than any block the highlighter or a chat row should draw; a
    /// line number past it is more likely noise than a real mark.
    static let maxMarkedLine = 100_000

    private static let titleKeys: Set<String> = ["title", "filename", "file"]

    static func parse(_ info: String?) -> CodeFenceInfo {
        let raw = CodeFenceInfo(language: info, title: nil, markedLines: nil)
        guard let info, let tokens = tokenize(info), let head = tokens.first else { return raw }
        // The common case: a bare language tag stays byte for byte.
        if tokens.count == 1, !head.contains(":"), !head.contains("{") { return raw }

        var language = head
        var title: String?
        var marks: IndexSet?

        if let colon = head.firstIndex(of: ":") {
            let lang = String(head[..<colon])
            let path = String(head[head.index(after: colon)...])
            guard isLanguageTag(lang), !path.isEmpty else { return raw }
            language = lang
            title = path
        } else if let brace = head.firstIndex(of: "{") {
            let lang = String(head[..<brace])
            guard isLanguageTag(lang), let parsed = parseMarks(String(head[brace...])) else { return raw }
            language = lang
            marks = parsed
        } else if !isLanguageTag(head) {
            return raw
        }

        for token in tokens.dropFirst() {
            if token.hasPrefix("{") {
                guard marks == nil, let parsed = parseMarks(token) else { return raw }
                marks = parsed
            } else if let equals = token.firstIndex(of: "=") {
                let key = token[..<equals].lowercased()
                guard titleKeys.contains(key), title == nil,
                      let value = unquote(String(token[token.index(after: equals)...]))
                else { return raw }
                title = value
            } else {
                return raw
            }
        }
        return CodeFenceInfo(language: language, title: title, markedLines: marks)
    }

    /// Splits the code text into display lines, keeping each run's
    /// attributes (the highlighter's colors). A final newline does not make
    /// an extra empty line, matching how the unmarked block draws.
    static func lines(of text: AttributedString) -> [AttributedString] {
        var result: [AttributedString] = []
        let chars = text.characters
        var start = chars.startIndex
        var index = chars.startIndex
        while index < chars.endIndex {
            // isNewline, not == "\n": "\r\n" is one Character in Swift.
            if chars[index].isNewline {
                result.append(AttributedString(text[start..<index]))
                start = chars.index(after: index)
            }
            index = chars.index(after: index)
        }
        if start < chars.endIndex || result.isEmpty {
            result.append(AttributedString(text[start..<chars.endIndex]))
        }
        return result
    }

    // MARK: - Parts

    /// Words split on whitespace, except inside quotes or braces. Nil when
    /// a quote or brace is left open.
    private static func tokenize(_ info: String) -> [String]? {
        var tokens: [String] = []
        var current = ""
        var quote: Character?
        var inBraces = false
        for char in info {
            if let open = quote {
                if char == open { quote = nil }
                current.append(char)
            } else if char == "\"" || char == "'" {
                quote = char
                current.append(char)
            } else if char == "{" {
                guard !inBraces else { return nil }
                inBraces = true
                current.append(char)
            } else if char == "}" {
                guard inBraces else { return nil }
                inBraces = false
                current.append(char)
            } else if char.isWhitespace {
                if inBraces { continue }
                if !current.isEmpty { tokens.append(current) }
                current = ""
            } else {
                current.append(char)
            }
        }
        guard quote == nil, !inBraces else { return nil }
        if !current.isEmpty { tokens.append(current) }
        return tokens
    }

    private static func isLanguageTag(_ tag: String) -> Bool {
        !tag.isEmpty && tag.allSatisfy { $0.isLetter || $0.isNumber || "+#._-".contains($0) }
    }

    /// `"a b"`, `'a'` or a bare word with no quotes in it.
    private static func unquote(_ value: String) -> String? {
        if let first = value.first, first == "\"" || first == "'" {
            guard value.count >= 2, value.last == first else { return nil }
            let inner = String(value.dropFirst().dropLast())
            guard !inner.isEmpty, !inner.contains(first) else { return nil }
            return inner
        }
        guard !value.isEmpty, !value.contains("\""), !value.contains("'") else { return nil }
        return value
    }

    /// `{3,5-7}`: each item is N or N-M with 1 <= N <= M <= maxMarkedLine.
    private static func parseMarks(_ token: String) -> IndexSet? {
        guard token.hasPrefix("{"), token.hasSuffix("}"), token.count > 2 else { return nil }
        var set = IndexSet()
        for item in token.dropFirst().dropLast().split(separator: ",", omittingEmptySubsequences: false) {
            let bounds = item.split(separator: "-", omittingEmptySubsequences: false)
            guard (1...2).contains(bounds.count),
                  let low = Int(bounds[0]), let high = Int(bounds[bounds.count - 1]),
                  low >= 1, low <= high, high <= maxMarkedLine
            else { return nil }
            set.insert(integersIn: low...high)
        }
        return set
    }
}
