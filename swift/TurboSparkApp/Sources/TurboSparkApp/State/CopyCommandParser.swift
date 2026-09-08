import Foundation

/// The `/copy` command family, the qwen-code `copyCommand.ts` parity: copy a
/// fenced code block or a LaTeX span out of the last assistant message by
/// language and ordinal, without selecting text by hand.
///
/// Grammar (argument words after `/copy`):
/// - (nothing) or `code`: the first fenced code block
/// - `code <lang> [index]`: the index-th fenced block whose language is
///   `<lang>` (1-based)
/// - `<lang>`: the first fenced block whose language is `<lang>`
/// - `latex [index]`: the index-th display-math block
/// - `inline-latex [index]`: the index-th inline math span
///
/// All scanning is pure so the extraction rules are testable without a
/// transcript. Unterminated fences are skipped rather than half-copied: a
/// fence the model never closed is more likely mid-stream prose than code.
enum CopyCommandParser {
    struct Request: Equatable {
        enum Target: Equatable {
            /// Fenced code block, optionally narrowed to a language.
            case code(language: String?)
            /// Display math (`$$...$$` or an equation environment).
            case latex
            /// Inline math (`$...$` or `\(...\)`).
            case inlineLatex
        }

        let target: Target
        /// 1-based ordinal WITHIN the filtered set. 1 when absent.
        let index: Int
    }

    enum Outcome: Equatable {
        case copied(String)
        /// The draft parsed but the message offered nothing to copy; the
        /// payload says what was looked for, which is the toast's text.
        case nothingFound(String)
        /// Not a `/copy` draft at all.
        case notACopyCommand
        /// Parsed, but the ordinal is past the end; the payload says how
        /// many candidates existed.
        case indexOutOfRange(requested: Int, available: Int, noun: String)
    }

    static func parse(_ draft: String) -> Request? {
        // The caller hands over the COMPOSER DRAFT, slash included.
        // Requiring the slash here means a prose sentence starting "copy
        // this" can never read as the command.
        let words = draft.split(separator: " ").map(String.init)
        guard words.first?.lowercased() == "/copy" else { return nil }
        let args = Array(words.dropFirst())

        switch args.first?.lowercased() {
        case nil:
            return Request(target: .code(language: nil), index: 1)
        case "code":
            // `/copy code [lang] [index]`: a following number is an
            // overall ordinal, a word narrows the language first.
            var language: String? = nil
            var ordinalIndex = 1
            if args.count > 1 {
                if let parsed = Int(args[1]), parsed > 0 {
                    ordinalIndex = parsed
                } else {
                    language = args[1].lowercased()
                    if args.count > 2, let parsed = Int(args[2]), parsed > 0 {
                        ordinalIndex = parsed
                    }
                }
            }
            return Request(target: .code(language: language), index: ordinalIndex)
        case "latex":
            return Request(target: .latex, index: ordinal(args.dropFirst()))
        case "inline-latex", "inline_latex":
            return Request(target: .inlineLatex, index: ordinal(args.dropFirst()))
        default:
            // A bare word is a language. A trailing number is an ordinal
            // WITHIN that language, so `/copy python 2` is the second python
            // block, not the second block overall.
            let language = args[0]
            if args.count > 1, let explicit = Int(args[1]), explicit > 0 {
                return Request(target: .code(language: language.lowercased()), index: explicit)
            }
            return Request(target: .code(language: language.lowercased()), index: 1)
        }
    }

    /// Runs a parsed request against message content. The draft form is
    /// re-parsed here so the caller hands over one string.
    static func resolve(draft: String, in message: String) -> Outcome {
        guard let request = parse(draft) else { return .notACopyCommand }
        switch request.target {
        case .code(let language):
            let blocks = fencedCodeBlocks(in: message)
            let candidates = blocks.filter { block in
                guard let language else { return true }
                return block.language == language
            }
            if candidates.isEmpty {
                let noun = language.map { "\($0) code block" } ?? "code block"
                return .nothingFound(blocks.isEmpty ? "no code block" : "no \(noun) in this response")
            }
            guard let block = candidates[safe: request.index - 1] else {
                return .indexOutOfRange(
                    requested: request.index, available: candidates.count, noun: "code block")
            }
            return .copied(block.code)
        case .latex:
            let spans = displayLatexSpans(in: message)
            if spans.isEmpty { return .nothingFound("no LaTeX block") }
            guard let span = spans[safe: request.index - 1] else {
                return .indexOutOfRange(
                    requested: request.index, available: spans.count, noun: "LaTeX block")
            }
            return .copied(span)
        case .inlineLatex:
            let spans = inlineLatexSpans(in: message)
            if spans.isEmpty { return .nothingFound("no inline LaTeX") }
            guard let span = spans[safe: request.index - 1] else {
                return .indexOutOfRange(
                    requested: request.index, available: spans.count, noun: "inline LaTeX span")
            }
            return .copied(span)
        }
    }

    // MARK: - Fenced code blocks

    struct CodeBlock: Equatable {
        let language: String?
        let code: String
    }

    /// Line scan for fenced blocks. An info string's FIRST word is the
    /// language (`swift run` fences as `swift`); a fence opened with 3+
    /// backticks closes at a line whose backtick run is at least as long.
    static func fencedCodeBlocks(in text: String) -> [CodeBlock] {
        var blocks: [CodeBlock] = []
        let lines = text.split(separator: "\n", omittingEmptySubsequences: false)
            .map(String.init)
        var index = 0
        while index < lines.count {
            let fence = openingFenceTickCount(lines[index])
            guard fence >= 3 else { index += 1; continue }
            let info = lines[index]
                .dropFirst(fence)
                .trimmingCharacters(in: .whitespaces)
            let language = info.split(separator: " ").first.map(String.init)
            var body: [String] = []
            var closed = false
            var cursor = index + 1
            while cursor < lines.count {
                if closingFenceTickCount(lines[cursor]) >= fence {
                    closed = true
                    break
                }
                body.append(lines[cursor])
                cursor += 1
            }
            if closed {
                // A trailing newline belongs to the fence, not the code, so
                // the copy is what an editor would hold for the same text.
                var code = body.joined(separator: "\n")
                if !code.isEmpty { code += "\n" }
                blocks.append(CodeBlock(language: language, code: code))
                index = cursor + 1
            } else {
                // Unterminated with nothing after it (the only way a fence
                // can be unterminated -- a later ``` would have closed this
                // one). Nothing further to scan.
                break
            }
        }
        return blocks
    }

    private static func openingFenceTickCount(_ line: String) -> Int {
        let trimmed = line.drop { $0 == " " }
        guard trimmed.hasPrefix("```") else { return 0 }
        return trimmed.prefix { $0 == "`" }.count
    }

    private static func closingFenceTickCount(_ line: String) -> Int {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("```"), trimmed.allSatisfy({ $0 == "`" }) else { return 0 }
        return trimmed.count
    }

    // MARK: - LaTeX

    /// Display math: `$$...$$` spans (multiline) and the equation-family
    /// environments. The `$$` delimiters are NOT part of the copy: the point
    /// is the source you can paste into an editor.
    static func displayLatexSpans(in text: String) -> [String] {
        var spans: [String] = []
        var rest = Substring(text)
        while let open = rest.range(of: "$$") {
            rest = rest[open.upperBound...]
            guard let close = rest.range(of: "$$") else { break }
            spans.append(String(rest[..<close.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines))
            rest = rest[close.upperBound...]
        }
        for environment in ["equation", "align", "gather", "eqnarray"] {
            let openTag = "\\begin{\(environment)}"
            let closeTag = "\\end{\(environment)}"
            var remaining = text
            while let openRange = remaining.range(of: openTag) {
                guard let closeRange = remaining.range(of: closeTag, range: openRange.upperBound..<remaining.endIndex) else { break }
                spans.append(String(remaining[openRange.upperBound..<closeRange.lowerBound])
                    .trimmingCharacters(in: .whitespacesAndNewlines))
                remaining = String(remaining[closeRange.upperBound...])
            }
        }
        return spans
    }

    /// Inline math: `$...$` on one line with the usual no-adjacent-space
    /// rule (so "$5 and $6" is currency, not math), plus `\(...\)`.
    static func inlineLatexSpans(in text: String) -> [String] {
        var spans: [String] = []
        // \( ... \) first; a parenthetical span can hold dollar signs.
        var rest = Substring(text)
        while let open = rest.range(of: "\\(") {
            rest = rest[open.upperBound...]
            guard let close = rest.range(of: "\\)") else { break }
            spans.append(String(rest[..<close.lowerBound])
                .trimmingCharacters(in: .whitespacesAndNewlines))
            rest = rest[close.upperBound...]
        }
        // Then $...$ per line: inline math does not cross a newline, and
        // scanning per line keeps a stray `$$` block from pairing oddly.
        for line in text.split(separator: "\n") {
            var cursor = line.startIndex
            while let open = line[cursor...].firstIndex(of: "$") {
                let after = line.index(after: open)
                guard after < line.endIndex,
                    line[after] != "$", line[after] != " ",
                    line[line.startIndex..<open].last != "\\"
                else { cursor = after; continue }
                guard let close = line[after...].firstIndex(of: "$") else { break }
                let inner = line[after..<close]
                guard let last = inner.last, last != " ", last != "\\" else {
                    cursor = after
                    continue
                }
                spans.append(String(inner))
                cursor = line.index(after: close)
            }
        }
        return spans
    }

    private static func ordinal(_ words: ArraySlice<String>) -> Int {
        guard let first = words.first, let value = Int(first), value > 0 else { return 1 }
        return value
    }
}

extension Array {
    /// Bounds-checked element, the read every ordinal guard above wants.
    subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
