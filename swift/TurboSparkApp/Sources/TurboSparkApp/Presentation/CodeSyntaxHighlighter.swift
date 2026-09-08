import SwiftUI

/// Lightweight native syntax highlighting for transcript code blocks, the
/// qwen-code web-shell's Shiki role at a deliberate fraction of the
/// machinery: one linear scanner, five token classes, no external
/// dependency and no per-language grammar tables.
///
/// Returns nil -- and the caller falls back to the plain MarkdownUI
/// rendering -- when the language tag is not one this scanner knows or the
/// block is over the size cap, so a huge streamed block cannot turn every
/// stream token into a full re-tokenization.
enum CodeSyntaxHighlighter {
    /// Over this, render plain. Streaming re-renders a growing fence on
    /// every token; this keeps that linear-per-token work bounded.
    static let maxHighlightLength = 12_000

    enum Family {
        /// // and /* */ comments; C-family keyword set.
        case cLike
        /// # comments; shell/python/ruby keyword set.
        case hash
        /// -- comments; SQL keyword set.
        case sql
        /// <tag> markup.
        case markup
    }

    static func family(forLanguage language: String?) -> Family? {
        switch language?.lowercased() {
        case "c", "cpp", "c++", "h", "hpp", "objc", "objectivec", "java", "kotlin",
            "cs", "csharp", "go", "rust", "rs", "swift", "js", "javascript", "ts",
            "typescript", "json", "dart", "scala", "php", "zig", "scss":
            return .cLike
        case "python", "py", "ruby", "rb", "sh", "bash", "zsh", "shell", "console",
            "yaml", "yml", "toml", "r", "perl", "makefile", "dockerfile", "ini",
            "tf", "hcl", "nginx", "gitignore":
            return .hash
        case "sql":
            return .sql
        case "html", "htm", "xml", "svg", "vue":
            return .markup
        default:
            return nil
        }
    }

    enum TokenKind {
        case plain
        case comment
        case string
        case number
        case keyword
        case literal
        case tag
    }

    /// Colors are SwiftUI's adaptive named hues: fixed hue families that
    /// keep workable contrast across the light and dark appearance themes
    /// this app ships, without a second palette keyed by appearance.
    static func color(for kind: TokenKind) -> Color? {
        switch kind {
        case .plain: return nil
        case .comment: return Color.secondary
        case .string: return .green
        case .number: return .orange
        case .keyword: return .purple
        case .literal: return .pink
        case .tag: return .blue
        }
    }

    /// Tokenizes and colors. The scanner is one pass with a small state
    /// machine (in-string, in-line-comment, in-block-comment); block-comment
    /// and string state carry ACROSS lines, so a block comment opened on
    /// line 1 colors through line 40 correctly.
    static func highlight(_ code: String, language: String?) -> AttributedString? {
        guard let family = family(forLanguage: language), code.count <= maxHighlightLength else {
            return nil
        }
        let keywords: Set<String>
        let literals: Set<String>
        switch family {
        case .cLike:
            keywords = [
                "abstract", "as", "async", "await", "break", "case", "catch", "class",
                "const", "continue", "debugger", "def", "default", "defer", "do", "else",
                "enum", "export", "extends", "extension", "final", "finally",
                "fn", "for", "foreach", "func", "function", "get", "go", "if", "impl",
                "implements", "import", "in", "init", "instanceof", "interface", "let",
                "match", "mut", "namespace", "new", "operator", "package", "private",
                "protected", "protocol", "public", "pub", "record", "return", "self",
                "Self", "static", "struct", "super", "switch", "throw", "throws", "trait",
                "try", "type", "typealias", "union", "unsafe", "use", "using", "var",
                "virtual", "void", "where", "while", "with", "yield",
            ]
            literals = ["true", "false", "null", "nil", "undefined", "none", "None"]
        case .hash:
            keywords = [
                "alias", "and", "as", "assert", "async", "await", "begin", "break",
                "case", "class", "def", "do", "done", "elif", "else", "end", "ensure",
                "except", "exec", "export", "fi", "finally", "for", "from", "function",
                "global", "if", "import", "in", "is", "lambda", "local", "module", "next",
                "not", "or", "pass", "raise", "require", "rescue", "return", "set",
                "then", "try", "unless", "unset", "until", "while", "with", "yield",
            ]
            literals = ["true", "false", "none", "null", "self", "yes", "no"]
        case .sql:
            keywords = [
                "add", "alter", "and", "as", "asc", "begin", "between", "by", "commit",
                "create", "delete", "desc", "distinct", "drop", "else", "end", "exists",
                "foreign", "from", "group", "having", "in", "index", "inner", "insert",
                "into", "join", "key", "left", "limit", "not", "null", "offset", "on",
                "or", "order", "outer", "primary", "references", "rollback", "select",
                "set", "table", "then", "transaction", "union", "unique", "update",
                "values", "view", "when", "where", "with",
            ]
            literals = ["true", "false", "null"]
        case .markup:
            keywords = []
            literals = []
        }

        var out = AttributedString()
        var segment = String()
        var kind: TokenKind = .plain

        func flush() {
            guard !segment.isEmpty else { return }
            var text = AttributedString(segment)
            if color(for: kind) != nil, kind != .plain {
                text.foregroundColor = color(for: kind)
            }
            out += text
            segment = ""
        }

        func begin(_ newKind: TokenKind) {
            if newKind != kind { flush() }
            kind = newKind
        }

        let chars = Array(code)
        var index = 0
        var word = String()

        func flushWord() {
            guard !word.isEmpty else { return }
            let decided: TokenKind
            if literals.contains(word) {
                decided = .literal
            } else if keywords.contains(word) {
                decided = .keyword
            } else {
                decided = .plain
            }
            // Flush whatever precedes the word FIRST. The word then becomes
            // its own run under its decided kind. (Checking the kind before
            // flushing -- the first version of this -- silently dropped the
            // pending text when a plain word followed plain text, which the
            // round-trip assertion in QwenParityFeaturesTests caught as a
            // missing space.)
            flush()
            kind = decided
            segment = word
            word = ""
            flush()
            kind = .plain
        }

        while index < chars.count {
            let char = chars[index]
            let two = index + 1 < chars.count ? String(chars[index...index + 1]) : ""

            switch family {
            case .cLike:
                if two == "//" { flushWord(); begin(.comment); segment = "//"; index += 2
                    while index < chars.count, chars[index] != "\n" { segment.append(chars[index]); index += 1 }
                    flush(); continue }
                if two == "/*" {
                    flushWord(); begin(.comment); segment = "/*"; index += 2
                    while index < chars.count {
                        if chars[index] == "*" && index + 1 < chars.count && chars[index + 1] == "/" {
                            segment.append("*/")
                            index += 2
                            break
                        }
                        segment.append(chars[index])
                        index += 1
                    }
                    flush(); continue }
                if char == "\"" || char == "'" {
                    flushWord(); begin(.string); segment.append(char); index += 1
                    while index < chars.count {
                        segment.append(chars[index])
                        if chars[index] == "\\" && index + 1 < chars.count {
                            index += 1
                            if index < chars.count { segment.append(chars[index]) }
                        } else if chars[index] == char || chars[index] == "\n" {
                            index += 1
                            break
                        }
                        index += 1
                    }
                    flush(); continue }
            case .hash:
                if char == "#" { flushWord(); begin(.comment); segment = "#"; index += 1
                    while index < chars.count, chars[index] != "\n" { segment.append(chars[index]); index += 1 }
                    flush(); continue }
                if char == "\"" || char == "'" {
                    flushWord(); begin(.string); segment.append(char); index += 1
                    while index < chars.count {
                        segment.append(chars[index])
                        if chars[index] == "\\" && index + 1 < chars.count {
                            index += 1
                            if index < chars.count { segment.append(chars[index]) }
                        } else if chars[index] == char || chars[index] == "\n" {
                            index += 1
                            break
                        }
                        index += 1
                    }
                    flush(); continue }
            case .sql:
                if two == "--" { flushWord(); begin(.comment); segment = "--"; index += 2
                    while index < chars.count, chars[index] != "\n" { segment.append(chars[index]); index += 1 }
                    flush(); continue }
                if char == "'" {
                    flushWord(); begin(.string); segment.append(char); index += 1
                    while index < chars.count {
                        segment.append(chars[index])
                        if chars[index] == "'" {
                            index += 1
                            break
                        }
                        index += 1
                    }
                    flush(); continue }
            case .markup:
                if char == "<", index + 1 < chars.count,
                    chars[index + 1] == "/" || chars[index + 1].isLetter {
                    flushWord(); begin(.tag); segment.append(char); index += 1
                    while index < chars.count, chars[index] != ">" {
                        segment.append(chars[index])
                        index += 1
                    }
                    if index < chars.count { segment.append(">"); index += 1 }
                    flush(); continue }
            }

            // Word accumulation for keyword matching (cLike/hash/sql only).
            if family != .markup, char.isLetter || char == "_" {
                word.append(char)
                index += 1
                continue
            }
            flushWord()
            if char.isNumber {
                begin(.number)
                segment.append(char)
                index += 1
                while index < chars.count,
                    chars[index].isNumber || chars[index] == "." || chars[index] == "x"
                        || chars[index] == "b" || chars[index] == "f" || chars[index] == "L" {
                    segment.append(chars[index])
                    index += 1
                }
                flush()
                kind = .plain
                continue
            }
            begin(.plain)
            segment.append(char)
            index += 1
        }
        flushWord()
        flush()
        return out
    }
}
