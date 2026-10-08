import Foundation
import JavaScriptCore

/// Result of the public JSCheckScriptSyntax probe that decides whether a
/// script can run natively as a classic script.
struct REPLClassicSyntaxProbe: Sendable, Equatable {
    var ok: Bool
    var message: String?
}

/// Shared decision machinery for the top-level-await strategy proven by
/// REPLTopLevelAwaitPrototype in task 1.5 and adopted by REPLWorkerContext in
/// task 1.6. This file holds the single bounded token-based scanner and the
/// async-wrapper lowering; the prototype and the production worker context
/// both use it, so there is exactly one parser in the subsystem.
///
/// Strategy, in decision order:
/// 1. JSCheckScriptSyntax probes the code as a classic script. Valid classic
///    scripts evaluate natively and keep native lexical declaration
///    semantics (no transform).
/// 2. Classic scripts cannot contain top-level await, so a failed probe hands
///    the source to the bounded parser below (a real lexer and top-level
///    statement scanner, never a regex over source text).
/// 3. If the parser finds top-level await and every top-level statement is
///    inside the supported subset, the program is lowered to an async wrapper
///    whose top-level declarations are rewritten to persistent assignments on
///    the global object. Declarations never move into a wrapper scope.
/// 4. Anything outside the subset is reported as unsupported syntax before
///    any evaluation runs, leaving the session unchanged.
///
/// Recorded bounds inherited from the proof (design bounds, not defects):
/// - Lowered declarations persist as writable global properties. const
///   enforcement, temporal dead zones, and lexical redeclaration errors are
///   not preserved, and a native lexical binding of the same name shadows a
///   lowered property assignment.
/// - The bounded subset rejects destructuring declarations, declarations with
///   multiple declarators, labeled statements, top-level return, and static
///   import/export.
/// - Awaits inside template substitutions are not detected by the scanner and
///   surface as parse errors instead. Awaits inside shorthand class or object
///   methods may be overcounted, which stays safe because valid classic
///   scripts never reach the parser.
/// - A final compound statement completes as undefined because the lowering
///   does not capture block completion values.
enum REPLClassicScriptSyntax {
    /// Probes the code with the public JSCheckScriptSyntax entry point.
    static func check(context: JSContext, code: String) -> REPLClassicSyntaxProbe {
        let string = JSStringCreateWithCFString(code as CFString)
        defer { JSStringRelease(string) }
        var exception: JSValueRef?
        let ok = JSCheckScriptSyntax(context.jsGlobalContextRef, string, nil, 1, &exception)
        if ok {
            return REPLClassicSyntaxProbe(ok: true, message: nil)
        }
        let message = exception
            .map { JSValue(jsValueRef: $0, in: context).toString() ?? "SyntaxError" }
            ?? "SyntaxError (no message)"
        return REPLClassicSyntaxProbe(ok: false, message: message)
    }
}

/// Lock-guarded one-shot outcome box for the lowered wrapper's settlement.
/// The Swift settle callback records the wrapper promise's outcome here and
/// the queue-confined settlement loop reads it.
final class REPLSettlementBox: @unchecked Sendable {
    private let lock = NSLock()
    private var fulfilled: Bool?
    private var text: String?

    var isSettled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return fulfilled != nil
    }

    var outcome: (fulfilled: Bool, text: String)? {
        lock.lock()
        defer { lock.unlock() }
        guard let fulfilled else { return nil }
        return (fulfilled, text ?? "")
    }

    func record(fulfilled: Bool, text: String) {
        lock.lock()
        self.fulfilled = fulfilled
        self.text = text
        lock.unlock()
    }

    func reset() {
        lock.lock()
        fulfilled = nil
        text = nil
        lock.unlock()
    }
}

/// Builds the lowered async wrapper for a bounded program.
///
/// Persistent declarations are never moved into a wrapper scope. Instead:
/// - Function and class declarations are installed as global properties
///   before every other statement (function declarations hoist to the top of
///   a classic script, so the lowering preserves that install order).
/// - let/const/var declarations become explicit assignments on the global
///   object; a declaration without an initializer only reserves the binding
///   when the property does not exist yet.
/// - The final top-level expression becomes the wrapper's return value, so
///   its rendered form is the call's completion text.
enum REPLLoweredWrapperBuilder {
    static func build(
        program: REPLBoundedProgram,
        source: String,
        settleGlobal: String,
        renderGlobal: String
    ) -> String {
        let characters = Array(source)
        let tokens = program.tokens

        func text(of range: Range<Int>) -> String {
            guard range.lowerBound < range.upperBound, range.upperBound <= tokens.count else {
                return ""
            }
            var upper = range.upperBound - 1
            while upper > range.lowerBound,
                tokens[upper].kind == .punctuator, tokens[upper].text == ";"
            {
                upper -= 1
            }
            let start = tokens[range.lowerBound].start
            let end = tokens[upper].end
            guard start <= end, end <= characters.count else { return "" }
            return String(characters[start..<end])
        }

        func bindingName(_ statement: REPLBoundedStatement) -> String {
            switch statement.kind {
            case let .declaration(binding, _):
                return binding
            case let .functionDeclaration(name):
                return name
            case let .classDeclaration(name):
                return name
            case .compound, .expression:
                return ""
            }
        }

        var body: [String] = []
        for statement in program.statements {
            if case .functionDeclaration = statement.kind {
                body.append(
                    "globalThis.\(bindingName(statement)) = \(text(of: statement.tokenRange));")
            }
        }
        let finalIndex = program.statements.count - 1
        for (index, statement) in program.statements.enumerated() {
            let isFinal = index == finalIndex
            switch statement.kind {
            case .functionDeclaration:
                continue
            case let .declaration(binding, hasInitializer):
                if hasInitializer {
                    let initializer = statement.initializerTokenRange.map { text(of: $0) } ?? ""
                    body.append("globalThis.\(binding) = (\(initializer));")
                } else {
                    body.append(
                        "if (!(\"\(binding)\" in globalThis)) globalThis.\(binding) = undefined;")
                }
            case .classDeclaration:
                body.append(
                    "globalThis.\(bindingName(statement)) = \(text(of: statement.tokenRange));")
            case .expression:
                let statementText = text(of: statement.tokenRange)
                // A bare `;` (an empty statement) has no value and must not
                // become `return (;);`.
                guard !statementText.isEmpty,
                    !statementText.allSatisfy({ $0 == ";" || $0.isWhitespace })
                else { continue }
                if isFinal {
                    body.append("return (\(statementText));")
                } else {
                    body.append("\(statementText);")
                }
            case .compound:
                let statementText = text(of: statement.tokenRange)
                guard !statementText.isEmpty else { continue }
                body.append("\(statementText);")
            }
        }

        return """
        (async () => {
        \(body.joined(separator: "\n"))
        })().then(
          value => \(settleGlobal)(true, \(renderGlobal)(value)),
          error => \(settleGlobal)(
            false,
            String(error) + (error && error.stack ? '\\n' + error.stack : '')));
        """
    }
}

// MARK: - Bounded token-based parser

enum REPLBoundedParseOutcome {
    case program(REPLBoundedProgram)
    case unsupported(reason: String, usesTopLevelAwait: Bool)
    case lexFailed(String)
}

struct REPLBoundedProgram {
    var statements: [REPLBoundedStatement]
    var tokens: [REPLLexToken]
    var usesTopLevelAwait: Bool
}

struct REPLBoundedStatement {
    enum Kind {
        case declaration(binding: String, hasInitializer: Bool)
        case functionDeclaration(name: String)
        case classDeclaration(name: String)
        case compound
        case expression
    }

    var kind: Kind
    var tokenRange: Range<Int>
    var initializerTokenRange: Range<Int>?
    var usesTopLevelAwait: Bool
}

struct REPLLexToken {
    enum Kind {
        case identifier
        case number
        case string
        case template
        case regex
        case punctuator
    }

    var kind: Kind
    var start: Int
    var end: Int
    var startsLine: Bool
    var text: String
}

struct REPLLexError: Error {
    var message: String
}

struct REPLBoundedLexer {
    let characters: [Character]

    private static let punctuators: [String] = [
        ">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=", "??=",
        "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=",
        "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>",
        "{", "}", "(", ")", "[", "]", ";", ",", "<", ">", "+", "-", "*", "/", "%",
        "&", "|", "^", "!", "~", "?", ":", "=", "."
    ]

    private static let regexPrecedingKeywords: Set<String> = [
        "return", "typeof", "case", "in", "of", "instanceof", "new", "delete",
        "void", "do", "else", "yield", "await", "throw"
    ]

    static func isLineTerminator(_ character: Character) -> Bool {
        character == "\n" || character == "\r" || character == "\u{2028}" || character == "\u{2029}"
    }

    func lex() throws -> [REPLLexToken] {
        var tokens: [REPLLexToken] = []
        var index = 0
        var pendingLineStart = false
        let count = characters.count

        func append(_ kind: REPLLexToken.Kind, _ start: Int, _ end: Int) {
            tokens.append(
                REPLLexToken(
                    kind: kind,
                    start: start,
                    end: end,
                    startsLine: pendingLineStart,
                    text: String(characters[start..<end])))
            pendingLineStart = false
        }

        func isIdentifierStart(_ character: Character) -> Bool {
            character.isLetter || character == "_" || character == "$" || character.asciiValue == nil
        }

        func isIdentifierPart(_ character: Character) -> Bool {
            character.isLetter || character.isNumber || character == "_" || character == "$"
                || character.asciiValue == nil
        }

        // Decides whether a "/" at the current position starts a regex or is
        // division, based on the previous significant token. After a "}" the
        // scanner chooses division, which is the documented approximation.
        func slashIsRegex() -> Bool {
            guard let previous = tokens.last else { return true }
            switch previous.kind {
            case .identifier:
                return Self.regexPrecedingKeywords.contains(previous.text)
            case .number, .string, .template, .regex:
                return false
            case .punctuator:
                return ![")", "]", "}", "++", "--"].contains(previous.text)
            }
        }

        while index < count {
            let character = characters[index]

            if character == " " || character == "\t" || character == "\u{0B}"
                || character == "\u{0C}" || character == "\u{FEFF}"
            {
                index += 1
                continue
            }
            if Self.isLineTerminator(character) {
                pendingLineStart = true
                index += 1
                continue
            }
            if character == "/" && index + 1 < count && characters[index + 1] == "/" {
                index = skipLineComment(from: index)
                continue
            }
            if character == "/" && index + 1 < count && characters[index + 1] == "*" {
                let (next, crossedNewline) = try skipBlockComment(from: index)
                if crossedNewline { pendingLineStart = true }
                index = next
                continue
            }
            if character == "\"" || character == "'" {
                let end = try skipStringLiteral(from: index)
                append(.string, index, end)
                index = end
                continue
            }
            if character == "`" {
                let end = try skipTemplateLiteral(from: index)
                append(.template, index, end)
                index = end
                continue
            }
            if isIdentifierStart(character) {
                var end = index
                while end < count && isIdentifierPart(characters[end]) {
                    end += 1
                }
                append(.identifier, index, end)
                index = end
                continue
            }
            if character.isNumber
                || (character == "." && index + 1 < count && characters[index + 1].isNumber)
            {
                var end = index
                while end < count,
                    characters[end].isLetter || characters[end].isNumber
                        || characters[end] == "." || characters[end] == "_"
                {
                    end += 1
                }
                append(.number, index, end)
                index = end
                continue
            }
            if character == "/" && slashIsRegex() {
                var end = index + 1
                var closed = false
                while end < count {
                    if characters[end] == "\\" {
                        end += 2
                        continue
                    }
                    if characters[end] == "/" {
                        end += 1
                        closed = true
                        break
                    }
                    if Self.isLineTerminator(characters[end]) {
                        throw REPLLexError(message: "unterminated regular expression")
                    }
                    end += 1
                }
                guard closed else { throw REPLLexError(message: "unterminated regular expression") }
                while end < count, characters[end].isLetter {
                    end += 1
                }
                append(.regex, index, end)
                index = end
                continue
            }
            if let punctuator = Self.matchPunctuator(in: characters, at: index) {
                append(.punctuator, index, index + punctuator.count)
                index += punctuator.count
                continue
            }
            throw REPLLexError(message: "unexpected character")
        }
        return tokens
    }

    private static func matchPunctuator(in characters: [Character], at index: Int) -> String? {
        let remaining = characters.count - index
        for punctuator in punctuators where punctuator.count <= remaining {
            var matches = true
            for (offset, character) in punctuator.enumerated() where characters[index + offset] != character {
                matches = false
                break
            }
            if matches { return punctuator }
        }
        return nil
    }

    private func skipLineComment(from index: Int) -> Int {
        var end = index + 2
        while end < characters.count && !Self.isLineTerminator(characters[end]) {
            end += 1
        }
        return end
    }

    private func skipBlockComment(from index: Int) throws -> (end: Int, crossedNewline: Bool) {
        var end = index + 2
        var crossedNewline = false
        while end < characters.count {
            let character = characters[end]
            if Self.isLineTerminator(character) {
                crossedNewline = true
            }
            if character == "*" && end + 1 < characters.count && characters[end + 1] == "/" {
                return (end + 2, crossedNewline)
            }
            end += 1
        }
        throw REPLLexError(message: "unterminated block comment")
    }

    private func skipStringLiteral(from index: Int) throws -> Int {
        let quote = characters[index]
        var end = index + 1
        while end < characters.count {
            let character = characters[end]
            if character == "\\" {
                end += 2
                continue
            }
            if character == quote {
                return end + 1
            }
            if character == "\n" || character == "\r" {
                throw REPLLexError(message: "unterminated string literal")
            }
            end += 1
        }
        throw REPLLexError(message: "unterminated string literal")
    }

    private func skipTemplateLiteral(from index: Int) throws -> Int {
        var end = index + 1
        while end < characters.count {
            let character = characters[end]
            if character == "\\" {
                end += 2
                continue
            }
            if character == "`" {
                return end + 1
            }
            if character == "$" && end + 1 < characters.count && characters[end + 1] == "{" {
                end = try skipBalancedCurly(from: end + 1)
                continue
            }
            end += 1
        }
        throw REPLLexError(message: "unterminated template literal")
    }

    /// Skips from an opening "{" to its matching "}", accounting for nested
    /// brackets, strings, templates, regular expressions, and comments.
    private func skipBalancedCurly(from index: Int) throws -> Int {
        var depth = 1
        var end = index + 1
        var previous: Character = "{"
        while end < characters.count {
            let character = characters[end]
            if character == "\"" || character == "'" {
                end = try skipStringLiteral(from: end)
                previous = "\""
                continue
            }
            if character == "`" {
                end = try skipTemplateLiteral(from: end)
                previous = "`"
                continue
            }
            if character == "/" && end + 1 < characters.count && characters[end + 1] == "/" {
                end = skipLineComment(from: end)
                previous = ";"
                continue
            }
            if character == "/" && end + 1 < characters.count && characters[end + 1] == "*" {
                let (next, _) = try skipBlockComment(from: end)
                end = next
                previous = ";"
                continue
            }
            if character == "/" && regexAllowed(after: previous) {
                var regexEnd = end + 1
                var closed = false
                while regexEnd < characters.count {
                    if characters[regexEnd] == "\\" {
                        regexEnd += 2
                        continue
                    }
                    if characters[regexEnd] == "/" {
                        regexEnd += 1
                        closed = true
                        break
                    }
                    regexEnd += 1
                }
                guard closed else { throw REPLLexError(message: "unterminated regular expression") }
                while regexEnd < characters.count, characters[regexEnd].isLetter {
                    regexEnd += 1
                }
                previous = "a"
                end = regexEnd
                continue
            }
            if character == "{" || character == "(" || character == "[" {
                depth += 1
            } else if character == "}" {
                depth -= 1
                if depth == 0 {
                    return end + 1
                }
            } else if character == ")" || character == "]" {
                throw REPLLexError(message: "unbalanced brackets inside a template substitution")
            }
            previous = character
            end += 1
        }
        throw REPLLexError(message: "unbalanced braces inside a template substitution")
    }

    private func regexAllowed(after previous: Character) -> Bool {
        switch previous {
        case "{", "(", "[", ",", "=", ":", "?", "!", "&", "|", "+", "-", "*", "%", "^", "~", ";":
            return true
        default:
            return false
        }
    }
}

enum REPLBoundedScriptParser {
    private static let continuationPunctuators: Set<String> = [
        "(", "[", "+", "-", "*", "/", "%", "=", "<", ">", "?", ":", ".", ",",
        "&", "|", "^", "=>", "==", "!=", "===", "!==", "<=", ">=", "&&", "||", "??",
        "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "**", "<<", ">>", ">>>",
        "?.", "...", "**=", "<<=", ">>=", ">>>=", "&&=", "||=", "??="
    ]

    private static let awaitOperandPunctuators: Set<String> = [
        "=", "(", "[", "{", ",", ";", "?", ":", "=>", "&&", "||", "??", "+", "-",
        "*", "/", "%", "<", ">", "<=", ">=", "==", "!=", "===", "!==", "&", "|",
        "^", "!", "~", "..."
    ]

    private static let awaitOperandKeywords: Set<String> = [
        "return", "typeof", "case", "new", "delete", "void", "in", "of",
        "instanceof", "throw", "yield", "for", "await", "else", "do"
    ]

    private static let compoundKeywords: Set<String> = [
        "if", "for", "while", "do", "switch", "try", "with"
    ]

    static func parse(source: String) -> REPLBoundedParseOutcome {
        let characters = Array(source)
        let tokens: [REPLLexToken]
        do {
            tokens = try REPLBoundedLexer(characters: characters).lex()
        } catch let error as REPLLexError {
            return .lexFailed(error.message)
        } catch {
            return .lexFailed("lexical scan failed")
        }
        guard !tokens.isEmpty else {
            return .program(REPLBoundedProgram(statements: [], tokens: [], usesTopLevelAwait: false))
        }

        // Pass 1: split the token stream into top-level statements and flag
        // awaits that live outside nested function bodies.
        var rawStatements: [REPLRawStatement] = []
        var statementStart = 0
        var parenDepth = 0
        var bracketDepth = 0
        var braceDepth = 0
        var braceIsFunction: [Bool] = []
        // Parallel to braceIsFunction: whether each `{` opened an object-literal
        // EXPRESSION (after `=`, `?`, `:`, `||`, ...) rather than a block. Only a
        // block-closing `}` ends a statement; an expression `}` falls through to
        // the normal continuation / ASI rule so `x = {a: 1}[k]` and
        // `c ? {..} : {..}` are not cut in half.
        var braceIsExpression: [Bool] = []
        var lastClosedBraceWasExpression = false
        var functionDepth = 0
        var pendingFunctionBody = false
        var previousInStatement: REPLLexToken?
        var statementUsesAwait = false

        func flush(through endIndex: Int) {
            guard endIndex >= statementStart else { return }
            rawStatements.append(
                REPLRawStatement(
                    tokenRange: statementStart..<(endIndex + 1),
                    usesTopLevelAwait: statementUsesAwait))
            statementStart = endIndex + 1
            previousInStatement = nil
            statementUsesAwait = false
        }

        var index = 0
        while index < tokens.count {
            let token = tokens[index]

            if token.kind == .identifier {
                switch token.text {
                case "function":
                    pendingFunctionBody = true
                case "await":
                    if functionDepth == 0 && isOperandPosition(previousInStatement) {
                        statementUsesAwait = true
                    }
                default:
                    break
                }
            } else if token.kind == .punctuator {
                switch token.text {
                case "(":
                    parenDepth += 1
                case ")":
                    parenDepth -= 1
                    if parenDepth < 0 { return .lexFailed("unbalanced parentheses") }
                case "[":
                    bracketDepth += 1
                case "]":
                    bracketDepth -= 1
                    if bracketDepth < 0 { return .lexFailed("unbalanced brackets") }
                case "{":
                    let isFunctionBody =
                        pendingFunctionBody
                        || (previousInStatement?.kind == .punctuator
                            && previousInStatement?.text == "=>")
                    braceIsFunction.append(isFunctionBody)
                    braceIsExpression.append(
                        !isFunctionBody && previousInStatement != nil
                            && isOperandPosition(previousInStatement)
                            && !(previousInStatement?.kind == .identifier
                                && ["else", "do"].contains(previousInStatement?.text ?? "")))
                    if isFunctionBody { functionDepth += 1 }
                    braceDepth += 1
                    pendingFunctionBody = false
                case "}":
                    braceDepth -= 1
                    if braceDepth < 0 { return .lexFailed("unbalanced braces") }
                    if let wasFunction = braceIsFunction.popLast(), wasFunction {
                        functionDepth -= 1
                    }
                    lastClosedBraceWasExpression = braceIsExpression.popLast() ?? false
                case "=>":
                    if index + 1 < tokens.count && tokens[index + 1].text == "{" {
                        pendingFunctionBody = true
                    }
                default:
                    break
                }
            }
            previousInStatement = token

            if parenDepth == 0 && bracketDepth == 0 && braceDepth == 0 {
                let next = index + 1 < tokens.count ? tokens[index + 1] : nil
                if token.kind == .punctuator && token.text == ";" {
                    flush(through: index)
                } else if token.kind == .punctuator && token.text == "}"
                    && !lastClosedBraceWasExpression
                {
                    if let next, next.kind == .punctuator, next.text == ";" {
                        // `const o = {...};` -- let the `;` close the whole
                        // statement instead of orphaning it as its own.
                    } else if let next, next.kind == .identifier,
                        ["else", "catch", "finally", "while"].contains(next.text)
                    {
                        // The statement continues (if/else, try/catch, do/while).
                    } else if let next, next.kind == .punctuator, next.text == "." {
                        // A method chain continuing the same expression.
                    } else {
                        flush(through: index)
                    }
                } else if let next, next.startsLine, !isContinuation(next, after: token) {
                    // Automatic semicolon insertion at the statement level:
                    // a newline terminates the statement unless the next token
                    // can continue the current expression.
                    flush(through: index)
                }
            }
            index += 1
        }
        flush(through: tokens.count - 1)

        let usesTopLevelAwait = rawStatements.contains { $0.usesTopLevelAwait }

        // Pass 2: classify each top-level statement. The first statement
        // outside the supported subset defines the unsupported-syntax reason.
        var statements: [REPLBoundedStatement] = []
        for raw in rawStatements {
            switch classify(raw: raw, tokens: tokens) {
            case let .statement(statement):
                statements.append(statement)
            case let .unsupported(reason):
                return .unsupported(reason: reason, usesTopLevelAwait: usesTopLevelAwait)
            }
        }

        return .program(
            REPLBoundedProgram(
                statements: statements,
                tokens: tokens,
                usesTopLevelAwait: usesTopLevelAwait))
    }

    private static func isOperandPosition(_ previous: REPLLexToken?) -> Bool {
        guard let previous else { return true }
        switch previous.kind {
        case .punctuator:
            return awaitOperandPunctuators.contains(previous.text)
        case .identifier:
            return awaitOperandKeywords.contains(previous.text)
        case .number, .string, .template, .regex:
            return false
        }
    }

    private static func isContinuation(_ next: REPLLexToken, after previous: REPLLexToken) -> Bool {
        if next.kind == .punctuator {
            return continuationPunctuators.contains(next.text)
        }
        if next.kind == .identifier {
            return ["in", "of", "instanceof"].contains(next.text)
        }
        if next.kind == .template {
            // A tagged template continues only after a possible tag.
            if previous.kind == .identifier { return true }
            if previous.kind == .punctuator && [")", "]"].contains(previous.text) { return true }
            return false
        }
        return false
    }

    private enum Classification {
        case statement(REPLBoundedStatement)
        case unsupported(String)
    }

    private static func classify(raw: REPLRawStatement, tokens: [REPLLexToken]) -> Classification {
        classifyTokens(raw.tokenRange, usesAwait: raw.usesTopLevelAwait, tokens: tokens)
    }

    private static func classifyTokens(
        _ range: Range<Int>,
        usesAwait: Bool,
        tokens: [REPLLexToken]
    ) -> Classification {
        let lower = range.lowerBound
        let upper = range.upperBound - 1
        let first = tokens[lower]
        let second = lower + 1 <= upper ? tokens[lower + 1] : nil

        func statement(_ kind: REPLBoundedStatement.Kind, initializer: Range<Int>? = nil)
            -> Classification
        {
            .statement(
                REPLBoundedStatement(
                    kind: kind,
                    tokenRange: range,
                    initializerTokenRange: initializer,
                    usesTopLevelAwait: usesAwait))
        }

        if first.kind == .identifier {
            switch first.text {
            case "let", "const", "var":
                guard let second, second.kind == .identifier else {
                    if let second, second.kind == .punctuator,
                        second.text == "{" || second.text == "["
                    {
                        return .unsupported(
                            "destructuring declarations are outside the bounded subset")
                    }
                    return .unsupported("declaration without a simple binding name")
                }
                var declarators = 0
                var depth = 0
                var initializerStart: Int?
                var scan = lower + 2
                while scan <= upper {
                    let token = tokens[scan]
                    if token.kind == .punctuator {
                        switch token.text {
                        case "(", "[", "{":
                            depth += 1
                        case ")", "]", "}":
                            depth -= 1
                        case "," where depth == 0:
                            declarators += 1
                        case "=" where depth == 0 && initializerStart == nil:
                            initializerStart = scan + 1
                        default:
                            break
                        }
                    }
                    scan += 1
                }
                if declarators > 0 {
                    return .unsupported(
                        "multiple declarators in one declaration are outside the bounded subset")
                }
                return statement(
                    .declaration(binding: second.text, hasInitializer: initializerStart != nil),
                    initializer: initializerStart.map { $0..<(upper + 1) })
            case "function", "async":
                // `function f`, `function* g`, `async function h`,
                // `async function* i`: the name is the first identifier after
                // `function` (and an optional `*`). Without this the async and
                // generator forms were not persisted across calls.
                var nameIndex = lower + 1
                if first.text == "async" {
                    guard let second, second.kind == .identifier, second.text == "function" else {
                        return statement(.expression)
                    }
                    nameIndex = lower + 2
                }
                if nameIndex <= upper, tokens[nameIndex].kind == .punctuator,
                    tokens[nameIndex].text == "*"
                {
                    nameIndex += 1
                }
                guard nameIndex <= upper, tokens[nameIndex].kind == .identifier else {
                    return statement(.expression)
                }
                return statement(.functionDeclaration(name: tokens[nameIndex].text))
            case "class":
                guard let second, second.kind == .identifier else {
                    return statement(.expression)
                }
                return statement(.classDeclaration(name: second.text))
            case "import", "export":
                return .unsupported("module import and export syntax is not supported")
            case "return":
                return .unsupported("top-level return is not supported")
            default:
                if let second, second.kind == .punctuator, second.text == ":" {
                    return .unsupported("labeled statements are outside the bounded subset")
                }
                if compoundKeywords.contains(first.text) {
                    return statement(.compound)
                }
            }
        } else if first.kind == .punctuator && first.text == "{" {
            return statement(.compound)
        }

        return statement(.expression)
    }
}

/// Raw statement shape handed from the splitter pass to the classifier.
struct REPLRawStatement {
    var tokenRange: Range<Int>
    var usesTopLevelAwait: Bool
}
