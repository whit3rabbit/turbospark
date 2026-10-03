import Foundation
import JavaScriptCore

/// Result of the public JSCheckScriptSyntax probe used by the proof suite.
struct REPLClassicSyntaxProbe: Sendable, Equatable {
    var ok: Bool
    var message: String?
}

/// Isolated proof prototype for task 1.5 of the js-repl-tool spec.
///
/// Decision flow per call, all on one private serial queue:
/// 1. JSCheckScriptSyntax probes the code as a classic script. Valid classic
///    scripts are evaluated natively with evaluateScript and keep native
///    lexical declaration semantics (no transform, design point 1).
/// 2. Classic scripts cannot contain top-level await, so a failed probe hands
///    the source to a bounded token-based parser (a real lexer and top-level
///    statement scanner, never a regex over source text).
/// 3. If the parser finds top-level await and every top-level statement is
///    inside the supported subset, the program is lowered to an async wrapper
///    whose top-level declarations are rewritten to persistent assignments on
///    the global object. The wrapper's promise settles through native
///    callbacks, with a microtask-drain loop plus a native timer wheel so
///    host-delayed promises settle too.
/// 4. Anything outside the subset returns the dedicated unsupportedSyntax
///    status before any evaluation runs, leaving the session unchanged.
///
/// Recorded bounds of this prototype (task 1.6 must resolve each):
/// - Declarations persist as writable global properties (var-like). const
///   enforcement, temporal dead zones, and lexical redeclaration errors are
///   not preserved, and a native lexical binding of the same name shadows a
///   lowered property assignment (see the divergence proof test).
/// - awaits inside template substitutions are not detected and surface as
///   parseError; awaits inside shorthand class or object methods may be
///   overcounted, which stays safe because valid classic scripts never reach
///   the parser.
/// - A final compound statement completes as undefined because the prototype
///   does not capture block completion values.
/// - The settlement deadline cannot terminate the context in-process; the
///   production supervisor terminates the worker process instead (task 3.2),
///   which also prevents a late settlement from a timed-out call recording
///   into the next call's outcome.
final class REPLTopLevelAwaitPrototype: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.turbospark.repl.tla-proof")
    private var context: JSContext?
    private var renderer: JSValue?
    private var capturedException: String?
    private var hasCreatedSession = false
    private var hostTimers: [HostTimer] = []
    private let outcomeBox = SettledBox()

    private struct HostTimer {
        var fireDate: Date
        var value: JSValue
        var resolve: JSValue
    }

    private final class SettledBox: @unchecked Sendable {
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

    // MARK: Public proof surface

    /// Evaluates code through the decision flow described in the class docs.
    func evaluate(code: String, settlementTimeout: TimeInterval = 5) async -> REPLCallResult {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                continuation.resume(
                    returning: evaluateOnQueue(code: code, settlementTimeout: settlementTimeout))
            }
        }
    }

    /// Probes the code with the public JSCheckScriptSyntax entry point.
    func classicSyntaxProbe(code: String) async -> REPLClassicSyntaxProbe {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                continuation.resume(returning: classicSyntaxProbeOnQueue(code: code))
            }
        }
    }

    /// Runs a native probe closure on the prototype queue so tests can touch
    /// the JavaScriptCore C API without breaking single-thread confinement.
    func runProbe<T: Sendable>(_ body: @escaping (JSContext) -> T) async -> T {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard let context = context ?? makeContext() else {
                    preconditionFailure("JavaScriptCore context unavailable")
                }
                continuation.resume(returning: body(context))
            }
        }
    }

    // MARK: Queue-confined implementation

    private func evaluateOnQueue(code: String, settlementTimeout: TimeInterval) -> REPLCallResult {
        let sessionCreated = !hasCreatedSession
        capturedException = nil
        hostTimers.removeAll()

        guard let context = context ?? makeContext() else {
            hasCreatedSession = true
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: "JavaScriptCore could not create an evaluation context.",
                sessionCreated: sessionCreated)
        }
        self.context = context
        hasCreatedSession = true
        context.exception = nil

        let probe = checkClassicSyntax(context: context, code: code)
        if probe.ok {
            return evaluateNatively(context: context, code: code, sessionCreated: sessionCreated)
        }
        let nativeMessage = probe.message ?? "unknown syntax error"

        switch BoundedParser.parse(source: code) {
        case let .lexFailed(reason):
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: "\(nativeMessage) (bounded scanner: \(reason))",
                sessionCreated: sessionCreated)
        case let .unsupported(reason, usesTopLevelAwait):
            if usesTopLevelAwait {
                return makeResult(
                    status: .unsupportedSyntax,
                    completionText: nil,
                    errorText: "unsupported top-level syntax: \(reason). "
                        + "The session was not changed.",
                    sessionCreated: sessionCreated)
            }
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: nativeMessage,
                sessionCreated: sessionCreated)
        case let .program(program):
            guard program.usesTopLevelAwait else {
                return makeResult(
                    status: .parseError,
                    completionText: nil,
                    errorText: nativeMessage,
                    sessionCreated: sessionCreated)
            }
            return evaluateLowered(
                context: context,
                program: program,
                source: code,
                settlementTimeout: settlementTimeout,
                sessionCreated: sessionCreated)
        }
    }

    private func evaluateNatively(
        context: JSContext,
        code: String,
        sessionCreated: Bool
    ) -> REPLCallResult {
        let value = context.evaluateScript(code)
        let failure = capturedException ?? context.exception?.toString()
        context.exception = nil
        if let failure {
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: failure,
                sessionCreated: sessionCreated)
        }
        return makeResult(
            status: .completed,
            completionText: renderCompletion(value),
            errorText: nil,
            sessionCreated: sessionCreated)
    }

    private func evaluateLowered(
        context: JSContext,
        program: BoundedProgram,
        source: String,
        settlementTimeout: TimeInterval,
        sessionCreated: Bool
    ) -> REPLCallResult {
        let wrapper = Self.buildAsyncWrapper(program: program, source: source)
        outcomeBox.reset()
        capturedException = nil
        context.exception = nil
        _ = context.evaluateScript(wrapper)

        if let compileError = capturedException ?? context.exception?.toString() {
            context.exception = nil
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: "\(compileError) (from the lowered wrapper; the session was not changed)",
                sessionCreated: sessionCreated)
        }

        let deadline = Date().addingTimeInterval(settlementTimeout)
        while true {
            if outcomeBox.isSettled { break }
            fireDueHostTimers(context: context)
            drainMicrotasks(context: context)
            if outcomeBox.isSettled { break }
            if Date() >= deadline {
                hostTimers.removeAll()
                return makeResult(
                    status: .timedOut,
                    completionText: nil,
                    errorText: "top-level await did not settle within \(settlementTimeout) s; "
                        + "the in-process prototype cannot terminate the context",
                    sessionCreated: sessionCreated)
            }
            Thread.sleep(forTimeInterval: 0.005)
        }

        guard let outcome = outcomeBox.outcome else {
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: "settlement loop ended without an outcome",
                sessionCreated: sessionCreated)
        }
        if outcome.fulfilled {
            return makeResult(
                status: .completed,
                completionText: outcome.text,
                errorText: nil,
                sessionCreated: sessionCreated)
        }
        return makeResult(
            status: .failed,
            completionText: nil,
            errorText: outcome.text,
            sessionCreated: sessionCreated)
    }

    private func classicSyntaxProbeOnQueue(code: String) -> REPLClassicSyntaxProbe {
        guard let context = context ?? makeContext() else {
            return REPLClassicSyntaxProbe(ok: false, message: "context unavailable")
        }
        self.context = context
        return checkClassicSyntax(context: context, code: code)
    }

    private func checkClassicSyntax(context: JSContext, code: String) -> REPLClassicSyntaxProbe {
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

    private func fireDueHostTimers(context: JSContext) {
        let now = Date()
        let due = hostTimers.filter { $0.fireDate <= now }
        guard !due.isEmpty else { return }
        hostTimers.removeAll { $0.fireDate <= now }
        for timer in due {
            _ = timer.resolve.call(withArguments: [timer.value])
        }
        context.exception = nil
    }

    /// Any script evaluation drains the microtask queue at its end, which is
    /// the settlement mechanism this prototype relies on. Mutation evidence:
    /// JSValue.call also drains microtasks after a native resolve, so this
    /// explicit drain is a defensive checkpoint rather than the sole
    /// load-bearing step; production (task 1.6) should keep an explicit
    /// checkpoint instead of depending on call-side draining.
    private func drainMicrotasks(context: JSContext) {
        _ = context.evaluateScript("0")
        context.exception = nil
    }

    private func makeContext() -> JSContext? {
        guard let context = JSContext() else { return nil }
        context.exceptionHandler = { [weak self] _, exception in
            self?.capturedException = exception?.toString()
        }

        let box = outcomeBox
        let settled: @convention(block) (Bool, String) -> Void = { fulfilled, text in
            box.record(fulfilled: fulfilled, text: text)
        }
        context.setObject(settled, forKeyedSubscript: "__replProofSettled" as NSString)

        let schedule: @convention(block) (Double, JSValue, JSValue) -> Void = { [weak self]
            milliseconds, value, resolve in
            guard let self else { return }
            self.hostTimers.append(
                HostTimer(
                    fireDate: Date().addingTimeInterval(milliseconds / 1000),
                    value: value,
                    resolve: resolve))
        }
        context.setObject(schedule, forKeyedSubscript: "__replProofScheduleDelay" as NSString)

        context.evaluateScript(Self.hostSetupScript)
        context.evaluateScript("globalThis.__replProofRender = \(Self.rendererScript);")
        renderer = context.evaluateScript("globalThis.__replProofRender")
        context.exception = nil
        return context
    }

    private func renderCompletion(_ value: JSValue?) -> String {
        guard let value else { return "undefined" }
        return renderer?.call(withArguments: [value])?.toString() ?? value.toString()
    }

    private func makeResult(
        status: REPLCallResult.Status,
        completionText: String?,
        errorText: String?,
        sessionCreated: Bool
    ) -> REPLCallResult {
        REPLCallResult(
            status: status,
            outputEvents: [],
            consoleText: "",
            errorText: errorText,
            completionText: completionText,
            images: [],
            truncated: false,
            sessionCreated: sessionCreated,
            sessionReset: false)
    }

    // MARK: Installed host scripts

    private static let hostSetupScript = """
    globalThis.__replProofDelayedResolve = (milliseconds, value) =>
      new Promise(resolve => __replProofScheduleDelay(milliseconds, value, resolve));
    """

    private static let rendererScript = """
    (() => {
      const stringify = JSON.stringify;
      const stringValue = String;
      return value => {
        if (typeof value === "object" && value !== null) {
          try {
            const encoded = stringify(value);
            if (encoded !== undefined) return encoded;
          } catch (_) {}
        }
        return stringValue(value);
      };
    })()
    """

    // MARK: Lowering

    private static func buildAsyncWrapper(program: BoundedProgram, source: String) -> String {
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

        func bindingName(_ statement: BoundedStatement) -> String {
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
        // Function declarations hoist to the top of a classic script, so the
        // lowering installs them before every other statement.
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
                guard !statementText.isEmpty else { continue }
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
          value => __replProofSettled(true, __replProofRender(value)),
          error => __replProofSettled(
            false,
            String(error) + (error && error.stack ? '\\n' + error.stack : '')));
        """
    }
}

// MARK: - Bounded token-based parser

private enum BoundedParseOutcome {
    case program(BoundedProgram)
    case unsupported(reason: String, usesTopLevelAwait: Bool)
    case lexFailed(String)
}

private struct BoundedProgram {
    var statements: [BoundedStatement]
    var tokens: [LexToken]
    var usesTopLevelAwait: Bool
}

private struct BoundedStatement {
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

private struct LexToken {
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

private struct LexError: Error {
    var message: String
}

private struct BoundedLexer {
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

    func lex() throws -> [LexToken] {
        var tokens: [LexToken] = []
        var index = 0
        var pendingLineStart = false
        let count = characters.count

        func append(_ kind: LexToken.Kind, _ start: Int, _ end: Int) {
            tokens.append(
                LexToken(
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
                        throw LexError(message: "unterminated regular expression")
                    }
                    end += 1
                }
                guard closed else { throw LexError(message: "unterminated regular expression") }
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
            throw LexError(message: "unexpected character")
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
        throw LexError(message: "unterminated block comment")
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
                throw LexError(message: "unterminated string literal")
            }
            end += 1
        }
        throw LexError(message: "unterminated string literal")
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
        throw LexError(message: "unterminated template literal")
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
                guard closed else { throw LexError(message: "unterminated regular expression") }
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
                throw LexError(message: "unbalanced brackets inside a template substitution")
            }
            previous = character
            end += 1
        }
        throw LexError(message: "unbalanced braces inside a template substitution")
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

private enum BoundedParser {
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

    static func parse(source: String) -> BoundedParseOutcome {
        let characters = Array(source)
        let tokens: [LexToken]
        do {
            tokens = try BoundedLexer(characters: characters).lex()
        } catch let error as LexError {
            return .lexFailed(error.message)
        } catch {
            return .lexFailed("lexical scan failed")
        }
        guard !tokens.isEmpty else {
            return .program(BoundedProgram(statements: [], tokens: [], usesTopLevelAwait: false))
        }

        // Pass 1: split the token stream into top-level statements and flag
        // awaits that live outside nested function bodies.
        var rawStatements: [RawStatement] = []
        var statementStart = 0
        var parenDepth = 0
        var bracketDepth = 0
        var braceDepth = 0
        var braceIsFunction: [Bool] = []
        var functionDepth = 0
        var pendingFunctionBody = false
        var previousInStatement: LexToken?
        var statementUsesAwait = false

        func flush(through endIndex: Int) {
            guard endIndex >= statementStart else { return }
            rawStatements.append(
                RawStatement(
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
                    if isFunctionBody { functionDepth += 1 }
                    braceDepth += 1
                    pendingFunctionBody = false
                case "}":
                    braceDepth -= 1
                    if braceDepth < 0 { return .lexFailed("unbalanced braces") }
                    if let wasFunction = braceIsFunction.popLast(), wasFunction {
                        functionDepth -= 1
                    }
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
                } else if token.kind == .punctuator && token.text == "}" {
                    if let next, next.kind == .identifier,
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
        var statements: [BoundedStatement] = []
        for raw in rawStatements {
            switch classify(raw: raw, tokens: tokens) {
            case let .statement(statement):
                statements.append(statement)
            case let .unsupported(reason):
                return .unsupported(reason: reason, usesTopLevelAwait: usesTopLevelAwait)
            }
        }

        return .program(
            BoundedProgram(
                statements: statements,
                tokens: tokens,
                usesTopLevelAwait: usesTopLevelAwait))
    }

    private static func isOperandPosition(_ previous: LexToken?) -> Bool {
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

    private static func isContinuation(_ next: LexToken, after previous: LexToken) -> Bool {
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
        case statement(BoundedStatement)
        case unsupported(String)
    }

    private static func classify(raw: RawStatement, tokens: [LexToken]) -> Classification {
        classifyTokens(raw.tokenRange, usesAwait: raw.usesTopLevelAwait, tokens: tokens)
    }

    private static func classifyTokens(
        _ range: Range<Int>,
        usesAwait: Bool,
        tokens: [LexToken]
    ) -> Classification {
        let lower = range.lowerBound
        let upper = range.upperBound - 1
        let first = tokens[lower]
        let second = lower + 1 <= upper ? tokens[lower + 1] : nil

        func statement(_ kind: BoundedStatement.Kind, initializer: Range<Int>? = nil)
            -> Classification
        {
            .statement(
                BoundedStatement(
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
            case "function":
                guard let second, second.kind == .identifier else {
                    return statement(.expression)
                }
                return statement(.functionDeclaration(name: second.text))
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
private struct RawStatement {
    var tokenRange: Range<Int>
    var usesTopLevelAwait: Bool
}
