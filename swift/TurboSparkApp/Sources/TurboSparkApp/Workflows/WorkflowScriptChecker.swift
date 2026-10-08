import CryptoKit
import Foundation

/// UTF-8 byte offset with one-based Unicode-scalar line and column coordinates.
struct WorkflowSourceLocation: Equatable, Sendable {
    let byteOffset: Int
    let line: Int
    let column: Int
}

/// Half-open UTF-8 source range.
struct WorkflowSourceRange: Equatable, Sendable {
    let start: WorkflowSourceLocation
    let end: WorkflowSourceLocation
}

enum WorkflowScriptDiagnosticRule: String, Equatable, Sendable {
    case sourceByteLimitExceeded
    case tokenLimitExceeded
    case astNodeLimitExceeded
    case nestingLimitExceeded
    case invalidEntryForm
    case unexpectedToken
    case unsupportedStatement
    case unsupportedExpression
    case unsupportedLoopForm
    case unsupportedOperator
    case unsupportedCall
    case invalidAwait
    case invalidLiteral
    case unterminatedString
    case invalidEscape
    case unterminatedComment
    case misplacedFacadeCall
    case unsupportedWorldReadOperation
    case facadeCallInsideLoop
    case runDispatchExecutableOverride
    case literalOnlyPosition
    case literalStructureRequired
    case literalRequiredFieldMissing
    case literalUnknownField
    case literalDuplicateField
    case literalValueOutOfRange
    case literalEnumValueUnsupported
    case executablePathMustBeAbsolute
    case shapeMustBeStatic
    case staticAnalysisBudgetExceeded
    case worldOperationMustBeStatic
    case runCommandKeyMustBeLiteral
    case runRequiresNamedDynamicValues
    case duplicateActorName
    case unknownActor
    case duplicateCommandPin
    case unknownCommandPin
    case misplacedPhaseMarker
    case duplicateGraphNodeID
    case unknownGraphDependency
    case graphCycle
    case graphNodeLimitExceeded
    case unknownGraphReference
    case graphNotJoinedOnEveryPath
    case graphJoinedMoreThanOnce
    case graphJoinInsideLoop
    case unserializablePublishedPayload
    case undeclaredWorkflowArgument
    case executableIdentityUnavailable
    case reservedActorName
    case undeclaredIdentifier
    case duplicateBinding
    case reservedBindingName
    case duplicateCommandSlot
}

struct WorkflowScriptDiagnostic: Equatable, Sendable {
    let rule: WorkflowScriptDiagnosticRule
    let message: String
    let location: WorkflowSourceLocation
}

enum WorkflowScriptLiteral: Equatable, Sendable {
    case null
    case boolean(Bool)
    case number(Double)
    case string(String)
}

enum WorkflowBinaryOperator: String, Equatable, Sendable {
    case logicalOr
    case logicalAnd
    case strictEqual
    case strictNotEqual
    case lessThan
    case lessThanOrEqual
    case greaterThan
    case greaterThanOrEqual
}

enum WorkflowCallTarget: String, CaseIterable, Equatable, Sendable {
    case agent
    case ask
    case parallel
    case join
    case criticLoop
    case phase
    case worldRead
    case command
    case run
    case report
    case artifact
    case glob
    case read
    case grep
    case git
}

struct WorkflowCall: Equatable, Sendable {
    let target: WorkflowCallTarget
    let arguments: [WorkflowExpression]
    let sourceRange: WorkflowSourceRange
}

struct WorkflowObjectMember: Equatable, Sendable {
    let name: String
    let nameRange: WorkflowSourceRange
    let value: WorkflowExpression
    let sourceRange: WorkflowSourceRange
}

struct WorkflowExpression: Equatable, Sendable {
    indirect enum Kind: Equatable, Sendable {
        case literal(WorkflowScriptLiteral)
        case identifier(String)
        case member(base: WorkflowExpression, name: String)
        case array([WorkflowExpression])
        case object([WorkflowObjectMember])
        case unaryNot(WorkflowExpression)
        case binary(left: WorkflowExpression, op: WorkflowBinaryOperator, right: WorkflowExpression)
        case call(WorkflowCall)
        case awaited(WorkflowCall)
    }

    let kind: Kind
    let sourceRange: WorkflowSourceRange
    let treeDepth: Int
}

struct WorkflowBlock: Equatable, Sendable {
    let statements: [WorkflowStatement]
    let sourceRange: WorkflowSourceRange
}

struct WorkflowStatement: Equatable, Sendable {
    indirect enum Kind: Equatable, Sendable {
        case declaration(name: String, nameRange: WorkflowSourceRange, value: WorkflowExpression)
        case expression(WorkflowExpression)
        case conditional(
            condition: WorkflowExpression,
            thenBlock: WorkflowBlock,
            elseBlock: WorkflowBlock?)
        case forOf(name: String, nameRange: WorkflowSourceRange, sequence: WorkflowExpression, body: WorkflowBlock)
    }

    let kind: Kind
    let sourceRange: WorkflowSourceRange
}

struct WorkflowScriptAST: Equatable, Sendable {
    static let currentVersion = 1

    let version: Int
    let facadeVersion: Int
    let sourceRange: WorkflowSourceRange
    let body: WorkflowBlock
}

struct WorkflowScriptParseResult: Equatable, Sendable {
    let ast: WorkflowScriptAST?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        ast != nil && diagnostics.isEmpty
    }
}

struct WorkflowCheckedProgram: Equatable, Sendable {
    let ast: WorkflowScriptAST
    let descriptor: WorkflowRunDescriptor
}

struct WorkflowCheckOutcome: Equatable, Sendable {
    let checked: WorkflowCheckedProgram?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        checked != nil && diagnostics.isEmpty
    }
}

struct WorkflowDefinitionCheckOutcome: Equatable, Sendable {
    let ast: WorkflowScriptAST?
    let manifest: WorkflowLaunchManifest?
    let diagnostics: [WorkflowScriptDiagnostic]

    var isValid: Bool {
        ast != nil && manifest != nil && diagnostics.isEmpty
    }
}

/// A non-evaluating parser and source-visible semantic checker for Workflow facade v1.
///
/// It validates syntax, literal positions, actor and graph rules, then emits the
/// complete AST with diagnostics. Saved-definition argument binding is checked
/// when declarations are available. It never evaluates source.
enum WorkflowScriptChecker {
    struct Limits: Equatable, Sendable {
        let maxSourceBytes: Int
        let maxTokens: Int
        let maxASTNodes: Int
        let maxNesting: Int

        static let production = Limits(
            maxSourceBytes: 131_072,
            maxTokens: 20_000,
            maxASTNodes: 20_000,
            maxNesting: 64)

        init(
            maxSourceBytes: Int = 131_072,
            maxTokens: Int = 20_000,
            maxASTNodes: Int = 20_000,
            maxNesting: Int = 64
        ) {
            self.maxSourceBytes = maxSourceBytes
            self.maxTokens = maxTokens
            self.maxASTNodes = maxASTNodes
            self.maxNesting = maxNesting
        }
    }

    static func parse(_ source: String) -> WorkflowScriptParseResult {
        parse(source, limits: .production)
    }

    /// The limits overload keeps resource-boundary tests small. Production call
    /// sites use parse(_:) and therefore always use the versioned limits.
    static func parse(_ source: String, limits: Limits) -> WorkflowScriptParseResult {
        let sourceBytes = source.utf8
        guard sourceBytes.count <= limits.maxSourceBytes else {
            let prefix = Array(sourceBytes.prefix(limits.maxSourceBytes))
            return failure(
                rule: .sourceByteLimitExceeded,
                message: "Source exceeds the \(limits.maxSourceBytes)-byte limit.",
                bytes: prefix,
                offset: limits.maxSourceBytes)
        }
        let bytes = Array(sourceBytes)

        do {
            var lexer = WorkflowLexer(bytes: bytes, maximumTokens: limits.maxTokens)
            let tokens = try lexer.lex()
            var parser = WorkflowParser(tokens: tokens, limits: limits)
            let ast = try parser.parse()
            let diagnostics = WorkflowScriptLiteralValidation.validate(ast)
                + WorkflowScriptSemanticValidation.validate(ast)
            return WorkflowScriptParseResult(ast: ast, diagnostics: diagnostics)
        } catch let error as WorkflowParseFailure {
            return WorkflowScriptParseResult(ast: nil, diagnostics: [error.diagnostic])
        } catch {
            let location = WorkflowSourceMap.location(in: bytes, offset: 0)
            return WorkflowScriptParseResult(ast: nil, diagnostics: [
                WorkflowScriptDiagnostic(
                    rule: .unexpectedToken,
                    message: "Workflow source could not be parsed.",
                    location: location)
            ])
        }
    }

    static func checkDefinition(
        _ source: String,
        declarations: [WorkflowArgumentDeclaration]
    ) -> WorkflowScriptParseResult {
        var result = parse(source)
        guard let ast = result.ast else { return result }
        result = WorkflowScriptParseResult(
            ast: ast,
            diagnostics: result.diagnostics + WorkflowScriptSemanticValidation.validateDefinitionArguments(
                ast,
                declaredNames: Set(declarations.map(\.name))))
        return result
    }

    static func check(source: String, name: String, args: [String: String]) -> WorkflowCheckOutcome {
        let result = parse(source)
        guard let ast = result.ast else {
            return WorkflowCheckOutcome(checked: nil, diagnostics: result.diagnostics)
        }
        guard result.diagnostics.isEmpty else {
            return WorkflowCheckOutcome(checked: nil, diagnostics: result.diagnostics)
        }

        let build = WorkflowScriptManifestCompiler.build(ast)
        guard let manifest = build.manifest, build.diagnostics.isEmpty else {
            return WorkflowCheckOutcome(checked: nil, diagnostics: result.diagnostics + build.diagnostics)
        }

        let descriptor = WorkflowRunDescriptor(
            id: UUID(),
            name: name,
            source: source,
            sourceHash: sourceHash(source),
            facadeVersion: ast.facadeVersion,
            args: WorkflowFrozenArguments(args),
            manifest: manifest)
        return WorkflowCheckOutcome(
            checked: WorkflowCheckedProgram(ast: ast, descriptor: descriptor),
            diagnostics: [])
    }

    static func checkDefinition(
        source: String,
        name _: String,
        declarations: [WorkflowArgumentDeclaration]
    ) -> WorkflowDefinitionCheckOutcome {
        let result = checkDefinition(source, declarations: declarations)
        guard let ast = result.ast, result.diagnostics.isEmpty else {
            return WorkflowDefinitionCheckOutcome(ast: result.ast, manifest: nil, diagnostics: result.diagnostics)
        }

        let build = WorkflowScriptManifestCompiler.build(ast)
        return WorkflowDefinitionCheckOutcome(
            ast: ast,
            manifest: build.manifest,
            diagnostics: result.diagnostics + build.diagnostics)
    }

    private static func sourceHash(_ source: String) -> String {
        SHA256.hash(data: Data(source.utf8)).map { String(format: "%02x", $0) }.joined()
    }

    private static func failure(
        rule: WorkflowScriptDiagnosticRule,
        message: String,
        bytes: [UInt8],
        offset: Int
    ) -> WorkflowScriptParseResult {
        WorkflowScriptParseResult(ast: nil, diagnostics: [
            WorkflowScriptDiagnostic(
                rule: rule,
                message: message,
                location: WorkflowSourceMap.location(in: bytes, offset: offset))
        ])
    }
}

private struct WorkflowParseFailure: Error {
    let diagnostic: WorkflowScriptDiagnostic
}

private enum WorkflowSourceMap {
    static func location(in bytes: [UInt8], offset requestedOffset: Int) -> WorkflowSourceLocation {
        let end = min(max(requestedOffset, 0), bytes.count)
        var index = 0
        var line = 1
        var column = 1
        var previousWasCarriageReturn = false

        while index < end {
            let byte = bytes[index]
            index += 1
            if byte == 0x0d {
                line += 1
                column = 1
                previousWasCarriageReturn = true
            } else if byte == 0x0a {
                if !previousWasCarriageReturn {
                    line += 1
                }
                column = 1
                previousWasCarriageReturn = false
            } else {
                previousWasCarriageReturn = false
                if byte & 0xc0 != 0x80 {
                    column += 1
                }
            }
        }

        return WorkflowSourceLocation(byteOffset: end, line: line, column: column)
    }
}

private struct WorkflowToken: Equatable {
    enum Kind: Equatable {
        case identifier(String)
        case number(Double)
        case string(String)
        case symbol(WorkflowSymbol)
        case end
    }

    let kind: Kind
    let range: WorkflowSourceRange
}

private enum WorkflowSymbol: Equatable {
    case leftParen
    case rightParen
    case leftBrace
    case rightBrace
    case leftBracket
    case rightBracket
    case dot
    case comma
    case colon
    case semicolon
    case assign
    case bang
    case strictEqual
    case strictNotEqual
    case andAnd
    case orOr
    case lessThan
    case lessThanOrEqual
    case greaterThan
    case greaterThanOrEqual
    case unsupportedOperator(String)
}

private struct WorkflowLexer {
    private let bytes: [UInt8]
    private let maximumTokens: Int
    private var offset = 0
    private var line = 1
    private var column = 1
    private var previousWasCarriageReturn = false
    private var tokens: [WorkflowToken] = []

    init(bytes: [UInt8], maximumTokens: Int) {
        self.bytes = bytes
        self.maximumTokens = maximumTokens
    }

    mutating func lex() throws -> [WorkflowToken] {
        while true {
            try skipTrivia()
            guard !isAtEnd else { break }
            guard tokens.count < maximumTokens else {
                throw failure(
                    .tokenLimitExceeded,
                    "Source exceeds the \(maximumTokens)-token limit.",
                    at: currentLocation)
            }
            tokens.append(try nextToken())
        }

        let end = currentLocation
        tokens.append(WorkflowToken(
            kind: .end,
            range: WorkflowSourceRange(start: end, end: end)))
        return tokens
    }

    private var isAtEnd: Bool {
        offset >= bytes.count
    }

    private var currentByte: UInt8? {
        isAtEnd ? nil : bytes[offset]
    }

    private func peekByte(_ distance: Int = 1) -> UInt8? {
        let index = offset + distance
        return index < bytes.count ? bytes[index] : nil
    }

    private var currentLocation: WorkflowSourceLocation {
        WorkflowSourceLocation(byteOffset: offset, line: line, column: column)
    }

    private mutating func advance() -> UInt8? {
        guard !isAtEnd else { return nil }
        let byte = bytes[offset]
        offset += 1

        if byte == 0x0d {
            line += 1
            column = 1
            previousWasCarriageReturn = true
        } else if byte == 0x0a {
            if !previousWasCarriageReturn {
                line += 1
            }
            column = 1
            previousWasCarriageReturn = false
        } else {
            previousWasCarriageReturn = false
            if byte & 0xc0 != 0x80 {
                column += 1
            }
        }
        return byte
    }

    private mutating func skipTrivia() throws {
        while let byte = currentByte {
            if isWhitespace(byte) {
                _ = advance()
                continue
            }

            guard byte == 0x2f, let next = peekByte() else { return }
            if next == 0x2f {
                _ = advance()
                _ = advance()
                while let commentByte = currentByte, commentByte != 0x0a, commentByte != 0x0d {
                    _ = advance()
                }
                continue
            }

            if next == 0x2a {
                let start = currentLocation
                _ = advance()
                _ = advance()
                var closed = false
                while let commentByte = currentByte {
                    if commentByte == 0x2a, peekByte() == 0x2f {
                        _ = advance()
                        _ = advance()
                        closed = true
                        break
                    }
                    _ = advance()
                }
                guard closed else {
                    throw failure(
                        .unterminatedComment,
                        "Block comment is not terminated.",
                        at: start)
                }
                continue
            }
            return
        }
    }

    private mutating func nextToken() throws -> WorkflowToken {
        let start = currentLocation
        guard let byte = currentByte else {
            return WorkflowToken(
                kind: .end,
                range: WorkflowSourceRange(start: start, end: start))
        }

        if isIdentifierStart(byte) {
            _ = advance()
            while let next = currentByte, isIdentifierContinue(next) {
                _ = advance()
            }
            let value = String(decoding: bytes[start.byteOffset..<offset], as: UTF8.self)
            return token(.identifier(value), start: start)
        }

        if isDigit(byte) || (byte == 0x2d && peekByte().map(isDigit) == true) {
            return try numberToken(start: start)
        }

        if byte == 0x22 || byte == 0x27 {
            return try stringToken(start: start, quote: byte)
        }

        switch byte {
        case 0x28: _ = advance(); return token(.symbol(.leftParen), start: start)
        case 0x29: _ = advance(); return token(.symbol(.rightParen), start: start)
        case 0x7b: _ = advance(); return token(.symbol(.leftBrace), start: start)
        case 0x7d: _ = advance(); return token(.symbol(.rightBrace), start: start)
        case 0x5b: _ = advance(); return token(.symbol(.leftBracket), start: start)
        case 0x5d: _ = advance(); return token(.symbol(.rightBracket), start: start)
        case 0x2e: _ = advance(); return token(.symbol(.dot), start: start)
        case 0x2c: _ = advance(); return token(.symbol(.comma), start: start)
        case 0x3a: _ = advance(); return token(.symbol(.colon), start: start)
        case 0x3b: _ = advance(); return token(.symbol(.semicolon), start: start)
        case 0x3d:
            if peekByte() == 0x3d, peekByte(2) == 0x3d {
                _ = advance(); _ = advance(); _ = advance()
                return token(.symbol(.strictEqual), start: start)
            }
            if peekByte() == 0x3e {
                throw failure(.unsupportedOperator, "Arrow functions are not supported.", at: start)
            }
            _ = advance()
            return token(.symbol(.assign), start: start)
        case 0x21:
            if peekByte() == 0x3d, peekByte(2) == 0x3d {
                _ = advance(); _ = advance(); _ = advance()
                return token(.symbol(.strictNotEqual), start: start)
            }
            if peekByte() == 0x3d {
                throw failure(.unsupportedOperator, "Only strict inequality is supported.", at: start)
            }
            _ = advance()
            return token(.symbol(.bang), start: start)
        case 0x26:
            guard peekByte() == 0x26 else {
                throw failure(.unsupportedOperator, "Bitwise operators are not supported.", at: start)
            }
            _ = advance(); _ = advance()
            return token(.symbol(.andAnd), start: start)
        case 0x7c:
            guard peekByte() == 0x7c else {
                throw failure(.unsupportedOperator, "Bitwise operators are not supported.", at: start)
            }
            _ = advance(); _ = advance()
            return token(.symbol(.orOr), start: start)
        case 0x3c:
            _ = advance()
            if currentByte == 0x3d {
                _ = advance()
                return token(.symbol(.lessThanOrEqual), start: start)
            }
            return token(.symbol(.lessThan), start: start)
        case 0x3e:
            _ = advance()
            if currentByte == 0x3d {
                _ = advance()
                return token(.symbol(.greaterThanOrEqual), start: start)
            }
            return token(.symbol(.greaterThan), start: start)
        case 0x2b, 0x2a, 0x25, 0x2f, 0x7e:
            _ = advance()
            return token(.symbol(.unsupportedOperator(String(UnicodeScalar(byte)))), start: start)
        case 0x3f:
            throw failure(.unsupportedExpression, "Optional chaining and conditional expressions are not supported.", at: start)
        case 0x60:
            throw failure(.unsupportedExpression, "Template strings are not supported.", at: start)
        default:
            throw failure(.unsupportedExpression, "This source character is not supported.", at: start)
        }
    }

    private mutating func numberToken(start: WorkflowSourceLocation) throws -> WorkflowToken {
        let startOffset = offset
        if currentByte == 0x2d {
            _ = advance()
        }

        if currentByte == 0x30 {
            _ = advance()
            if currentByte.map(isDigit) == true {
                throw failure(.invalidLiteral, "Numeric literals cannot have leading zeroes.", at: start)
            }
        } else {
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "A number must contain an integer part.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        if currentByte == 0x2e {
            _ = advance()
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "A decimal point must be followed by digits.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        if currentByte == 0x65 || currentByte == 0x45 {
            _ = advance()
            if currentByte == 0x2b || currentByte == 0x2d {
                _ = advance()
            }
            guard currentByte.map(isDigit) == true else {
                throw failure(.invalidLiteral, "An exponent must contain digits.", at: start)
            }
            while currentByte.map(isDigit) == true {
                _ = advance()
            }
        }

        let text = String(decoding: bytes[startOffset..<offset], as: UTF8.self)
        guard let value = Double(text), value.isFinite else {
            throw failure(.invalidLiteral, "Numeric literals must be finite.", at: start)
        }
        return token(.number(value), start: start)
    }

    private mutating func stringToken(start: WorkflowSourceLocation, quote: UInt8) throws -> WorkflowToken {
        _ = advance()
        var decoded: [UInt8] = []

        while let byte = currentByte {
            if byte == quote {
                _ = advance()
                let value = String(decoding: decoded, as: UTF8.self)
                return token(.string(value), start: start)
            }
            if byte == 0x0a || byte == 0x0d {
                throw failure(.unterminatedString, "String literals cannot contain raw line breaks.", at: start)
            }
            if byte != 0x5c {
                decoded.append(advance()!)
                continue
            }

            let escapeLocation = currentLocation
            _ = advance()
            guard let escaped = currentByte else {
                throw failure(.unterminatedString, "String literal ends after an escape marker.", at: start)
            }
            _ = advance()
            switch escaped {
            case 0x22: decoded.append(0x22)
            case 0x27: decoded.append(0x27)
            case 0x5c: decoded.append(0x5c)
            case 0x2f: decoded.append(0x2f)
            case 0x62: decoded.append(0x08)
            case 0x66: decoded.append(0x0c)
            case 0x6e: decoded.append(0x0a)
            case 0x72: decoded.append(0x0d)
            case 0x74: decoded.append(0x09)
            case 0x76: decoded.append(0x0b)
            case 0x30:
                guard currentByte.map(isDigit) != true else {
                    throw failure(.invalidEscape, "Octal string escapes are not supported.", at: escapeLocation)
                }
                decoded.append(0)
            case 0x75:
                let first = try unicodeEscape(at: escapeLocation)
                let scalarValue: UInt32
                if (0xd800...0xdbff).contains(first) {
                    guard currentByte == 0x5c, peekByte() == 0x75 else {
                        throw failure(.invalidEscape, "A high surrogate needs a following low surrogate.", at: escapeLocation)
                    }
                    _ = advance()
                    _ = advance()
                    let second = try unicodeEscape(at: escapeLocation)
                    guard (0xdc00...0xdfff).contains(second) else {
                        throw failure(.invalidEscape, "A high surrogate must be followed by a low surrogate.", at: escapeLocation)
                    }
                    scalarValue = 0x10000 + ((UInt32(first) - 0xd800) << 10) + (UInt32(second) - 0xdc00)
                } else {
                    guard !(0xdc00...0xdfff).contains(first) else {
                        throw failure(.invalidEscape, "A low surrogate cannot appear by itself.", at: escapeLocation)
                    }
                    scalarValue = UInt32(first)
                }
                guard let scalar = UnicodeScalar(scalarValue) else {
                    throw failure(.invalidEscape, "Unicode escape is outside the scalar range.", at: escapeLocation)
                }
                decoded.append(contentsOf: String(scalar).utf8)
            default:
                throw failure(.invalidEscape, "This string escape is not supported.", at: escapeLocation)
            }
        }

        throw failure(.unterminatedString, "String literal is not terminated.", at: start)
    }

    private mutating func unicodeEscape(at location: WorkflowSourceLocation) throws -> UInt16 {
        var value: UInt16 = 0
        for _ in 0..<4 {
            guard let byte = currentByte, let digit = hexValue(byte) else {
                throw failure(.invalidEscape, "Unicode escapes need four hexadecimal digits.", at: location)
            }
            _ = advance()
            value = (value << 4) | UInt16(digit)
        }
        return value
    }

    private func hexValue(_ byte: UInt8) -> UInt8? {
        switch byte {
        case 0x30...0x39: byte - 0x30
        case 0x41...0x46: byte - 0x41 + 10
        case 0x61...0x66: byte - 0x61 + 10
        default: nil
        }
    }

    private func token(_ kind: WorkflowToken.Kind, start: WorkflowSourceLocation) -> WorkflowToken {
        WorkflowToken(kind: kind, range: WorkflowSourceRange(start: start, end: currentLocation))
    }

    private func failure(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) -> WorkflowParseFailure {
        WorkflowParseFailure(diagnostic: WorkflowScriptDiagnostic(
            rule: rule,
            message: message,
            location: location))
    }

    private func isWhitespace(_ byte: UInt8) -> Bool {
        byte == 0x20 || byte == 0x09 || byte == 0x0a || byte == 0x0d || byte == 0x0c
    }

    private func isDigit(_ byte: UInt8) -> Bool {
        (0x30...0x39).contains(byte)
    }

    private func isIdentifierStart(_ byte: UInt8) -> Bool {
        (0x41...0x5a).contains(byte) || (0x61...0x7a).contains(byte) || byte == 0x5f || byte == 0x24
    }

    private func isIdentifierContinue(_ byte: UInt8) -> Bool {
        isIdentifierStart(byte) || isDigit(byte)
    }
}

private struct WorkflowParser {
    private let tokens: [WorkflowToken]
    private let limits: WorkflowScriptChecker.Limits
    private var index = 0
    private var nodeCount = 0
    private var nesting = 0
    private var loopBodyDepth = 0

    init(tokens: [WorkflowToken], limits: WorkflowScriptChecker.Limits) {
        self.tokens = tokens
        self.limits = limits
    }

    mutating func parse() throws -> WorkflowScriptAST {
        let start = current.range.start
        guard consumeKeyword("async") != nil,
              consumeKeyword("function") != nil,
              consumeExactIdentifier("workflow") != nil,
              consumeSymbol(.leftParen) != nil,
              consumeSymbol(.rightParen) != nil
        else {
            throw failure(.invalidEntryForm, "Entry must be async function workflow() { ... }.", at: current.range.start)
        }

        let body = try parseBlock()
        guard isAtEnd else {
            throw failure(.invalidEntryForm, "Only the workflow function body is accepted.", at: current.range.start)
        }
        let end = current.range.end
        try countNode(at: start)
        return WorkflowScriptAST(
            version: WorkflowScriptAST.currentVersion,
            facadeVersion: WorkflowFacade.version,
            sourceRange: WorkflowSourceRange(start: start, end: end),
            body: body)
    }

    private var current: WorkflowToken {
        tokens[min(index, tokens.count - 1)]
    }

    private var isAtEnd: Bool {
        if case .end = current.kind { return true }
        return false
    }

    private func peekIdentifier(_ name: String) -> Bool {
        if case .identifier(name) = current.kind { return true }
        return false
    }

    private func peekSymbol(_ symbol: WorkflowSymbol) -> Bool {
        if case .symbol(symbol) = current.kind { return true }
        return false
    }

    private mutating func advance() -> WorkflowToken {
        let token = current
        if !isAtEnd {
            index += 1
        }
        return token
    }

    private mutating func consumeKeyword(_ keyword: String) -> WorkflowToken? {
        guard peekIdentifier(keyword) else { return nil }
        return advance()
    }

    private mutating func consumeExactIdentifier(_ name: String) -> WorkflowToken? {
        guard case .identifier(name) = current.kind else { return nil }
        return advance()
    }

    private mutating func consumeSymbol(_ symbol: WorkflowSymbol) -> WorkflowToken? {
        guard peekSymbol(symbol) else { return nil }
        return advance()
    }

    private mutating func expectSymbol(
        _ symbol: WorkflowSymbol,
        rule: WorkflowScriptDiagnosticRule = .unexpectedToken
    ) throws -> WorkflowToken {
        if case .symbol(.unsupportedOperator(let spelling)) = current.kind {
            let diagnosticRule: WorkflowScriptDiagnosticRule = rule == .unsupportedLoopForm
                ? .unsupportedLoopForm
                : .unsupportedOperator
            throw failure(diagnosticRule, "Operator '\(spelling)' is not supported.", at: current.range.start)
        }
        guard let token = consumeSymbol(symbol) else {
            throw failure(rule, "Expected \(symbolName(symbol)).", at: current.range.start)
        }
        return token
    }

    private mutating func expectKeyword(
        _ keyword: String,
        rule: WorkflowScriptDiagnosticRule = .unexpectedToken
    ) throws -> WorkflowToken {
        if case .symbol(.unsupportedOperator(let spelling)) = current.kind {
            let diagnosticRule: WorkflowScriptDiagnosticRule = rule == .unsupportedLoopForm
                ? .unsupportedLoopForm
                : .unsupportedOperator
            throw failure(diagnosticRule, "Operator '\(spelling)' is not supported here.", at: current.range.start)
        }
        guard let token = consumeKeyword(keyword) else {
            throw failure(rule, "Expected '\(keyword)'.", at: current.range.start)
        }
        return token
    }

    private mutating func expectIdentifier(
        rule: WorkflowScriptDiagnosticRule = .unexpectedToken
    ) throws -> WorkflowToken {
        guard case .identifier(let name) = current.kind, !Self.reservedWords.contains(name) else {
            throw failure(rule, "Expected an identifier.", at: current.range.start)
        }
        return advance()
    }

    private mutating func expectSemicolon() throws -> WorkflowToken {
        if peekSymbol(.assign) {
            throw failure(.unsupportedOperator, "Assignment expressions are not supported.", at: current.range.start)
        }
        if case .symbol(.unsupportedOperator(let spelling)) = current.kind {
            throw failure(.unsupportedOperator, "Operator '\(spelling)' is not supported.", at: current.range.start)
        }
        return try expectSymbol(.semicolon)
    }

    private mutating func parseBlock() throws -> WorkflowBlock {
        let open = try expectSymbol(.leftBrace)
        try enterNesting(at: open.range.start)
        defer { nesting -= 1 }
        var statements: [WorkflowStatement] = []

        while !peekSymbol(.rightBrace) {
            guard !isAtEnd else {
                throw failure(.unexpectedToken, "Block is not terminated.", at: current.range.start)
            }
            statements.append(try parseStatement())
        }

        let close = try expectSymbol(.rightBrace)
        try countNode(at: open.range.start)
        return WorkflowBlock(
            statements: statements,
            sourceRange: WorkflowSourceRange(start: open.range.start, end: close.range.end))
    }

    private mutating func parseStatement() throws -> WorkflowStatement {
        if peekKeyword("const") {
            return try parseDeclaration()
        }
        if peekKeyword("if") {
            return try parseConditional()
        }
        if peekKeyword("for") {
            return try parseForOf()
        }
        if isUnsupportedStatementStart {
            throw failure(.unsupportedStatement, "This statement form is not supported.", at: current.range.start)
        }

        let expression = try parseExpression()
        let semicolon = try expectSemicolon()
        guard isSupportedExpressionStatement(expression) else {
            throw failure(.unsupportedStatement, "Only declared facade statements are accepted here.", at: expression.sourceRange.start)
        }
        try validateStatementExpression(expression)
        try countNode(at: expression.sourceRange.start)
        return WorkflowStatement(
            kind: .expression(expression),
            sourceRange: WorkflowSourceRange(start: expression.sourceRange.start, end: semicolon.range.end))
    }

    private mutating func parseDeclaration() throws -> WorkflowStatement {
        let start = advance()
        let nameToken = try expectIdentifier()
        guard case .identifier(let name) = nameToken.kind else {
            throw failure(.unexpectedToken, "Expected a binding name.", at: nameToken.range.start)
        }
        _ = try expectSymbol(.assign)
        let value = try parseExpression()
        let semicolon = try expectSemicolon()
        try validateInitializer(value)
        try countNode(at: start.range.start)
        return WorkflowStatement(
            kind: .declaration(name: name, nameRange: nameToken.range, value: value),
            sourceRange: WorkflowSourceRange(start: start.range.start, end: semicolon.range.end))
    }

    private mutating func parseConditional() throws -> WorkflowStatement {
        let start = advance()
        _ = try expectSymbol(.leftParen)
        let condition = try parseExpression()
        _ = try expectSymbol(.rightParen)
        guard isPure(condition) else {
            throw failure(.unsupportedExpression, "A branch condition must use pure expressions only.", at: condition.sourceRange.start)
        }
        let thenBlock = try parseBlock()
        let elseBlock: WorkflowBlock?
        if consumeKeyword("else") != nil {
            elseBlock = try parseBlock()
        } else {
            elseBlock = nil
        }
        try countNode(at: start.range.start)
        return WorkflowStatement(
            kind: .conditional(condition: condition, thenBlock: thenBlock, elseBlock: elseBlock),
            sourceRange: WorkflowSourceRange(
                start: start.range.start,
                end: elseBlock?.sourceRange.end ?? thenBlock.sourceRange.end))
    }

    private mutating func parseForOf() throws -> WorkflowStatement {
        let start = advance()
        _ = try expectSymbol(.leftParen, rule: .unsupportedLoopForm)
        guard consumeKeyword("const") != nil else {
            throw failure(.unsupportedLoopForm, "Only for (const name of expression) loops are supported.", at: current.range.start)
        }
        let nameToken = try expectIdentifier(rule: .unsupportedLoopForm)
        guard case .identifier(let name) = nameToken.kind else {
            throw failure(.unsupportedLoopForm, "Loop binding must be an identifier.", at: nameToken.range.start)
        }
        _ = try expectKeyword("of", rule: .unsupportedLoopForm)
        let sequence = try parseExpression()
        guard isPure(sequence) else {
            throw failure(.unsupportedLoopForm, "A for-of sequence must be a pure expression.", at: sequence.sourceRange.start)
        }
        _ = try expectSymbol(.rightParen, rule: .unsupportedLoopForm)

        loopBodyDepth += 1
        let body: WorkflowBlock
        do {
            body = try parseBlock()
            loopBodyDepth -= 1
        } catch {
            loopBodyDepth -= 1
            throw error
        }

        try countNode(at: start.range.start)
        return WorkflowStatement(
            kind: .forOf(name: name, nameRange: nameToken.range, sequence: sequence, body: body),
            sourceRange: WorkflowSourceRange(start: start.range.start, end: body.sourceRange.end))
    }

    private mutating func parseExpression() throws -> WorkflowExpression {
        try parseOr()
    }

    private mutating func parseOr() throws -> WorkflowExpression {
        var expression = try parseAnd()
        while let token = consumeSymbol(.orOr) {
            let right = try parseAnd()
            expression = try makeExpression(
                .binary(left: expression, op: .logicalOr, right: right),
                range: WorkflowSourceRange(start: expression.sourceRange.start, end: right.sourceRange.end),
                location: token.range.start)
        }
        return expression
    }

    private mutating func parseAnd() throws -> WorkflowExpression {
        var expression = try parseEquality()
        while let token = consumeSymbol(.andAnd) {
            let right = try parseEquality()
            expression = try makeExpression(
                .binary(left: expression, op: .logicalAnd, right: right),
                range: WorkflowSourceRange(start: expression.sourceRange.start, end: right.sourceRange.end),
                location: token.range.start)
        }
        return expression
    }

    private mutating func parseEquality() throws -> WorkflowExpression {
        var expression = try parseComparison()
        while true {
            let op: WorkflowBinaryOperator
            let token: WorkflowToken
            if let found = consumeSymbol(.strictEqual) {
                op = .strictEqual
                token = found
            } else if let found = consumeSymbol(.strictNotEqual) {
                op = .strictNotEqual
                token = found
            } else {
                break
            }
            let right = try parseComparison()
            expression = try makeExpression(
                .binary(left: expression, op: op, right: right),
                range: WorkflowSourceRange(start: expression.sourceRange.start, end: right.sourceRange.end),
                location: token.range.start)
        }
        return expression
    }

    private mutating func parseComparison() throws -> WorkflowExpression {
        var expression = try parseUnary()
        while true {
            let op: WorkflowBinaryOperator
            let token: WorkflowToken
            if let found = consumeSymbol(.lessThan) {
                op = .lessThan
                token = found
            } else if let found = consumeSymbol(.lessThanOrEqual) {
                op = .lessThanOrEqual
                token = found
            } else if let found = consumeSymbol(.greaterThan) {
                op = .greaterThan
                token = found
            } else if let found = consumeSymbol(.greaterThanOrEqual) {
                op = .greaterThanOrEqual
                token = found
            } else {
                break
            }
            let right = try parseUnary()
            expression = try makeExpression(
                .binary(left: expression, op: op, right: right),
                range: WorkflowSourceRange(start: expression.sourceRange.start, end: right.sourceRange.end),
                location: token.range.start)
        }
        return expression
    }

    private mutating func parseUnary() throws -> WorkflowExpression {
        if let token = consumeSymbol(.bang) {
            try enterNesting(at: token.range.start)
            defer { nesting -= 1 }
            let operand = try parseUnary()
            return try makeExpression(
                .unaryNot(operand),
                range: WorkflowSourceRange(start: token.range.start, end: operand.sourceRange.end),
                location: token.range.start)
        }

        if let token = consumeKeyword("await") {
            let awaited = try parsePostfix()
            guard case .call(let call) = awaited.kind, Self.awaitableTargets.contains(call.target) else {
                throw failure(.invalidAwait, "await must apply directly to a supported facade operation.", at: token.range.start)
            }
            return try makeExpression(
                .awaited(call),
                range: WorkflowSourceRange(start: token.range.start, end: awaited.sourceRange.end),
                location: token.range.start)
        }

        return try parsePostfix()
    }

    private mutating func parsePostfix() throws -> WorkflowExpression {
        var expression = try parsePrimary()

        while true {
            if let dot = consumeSymbol(.dot) {
                let nameToken = try expectIdentifier(rule: .unsupportedExpression)
                guard case .identifier(let name) = nameToken.kind else {
                    throw failure(.unsupportedExpression, "Property access needs a static identifier.", at: nameToken.range.start)
                }
                guard !Self.prototypeProperties.contains(name) else {
                    throw failure(.unsupportedExpression, "Prototype-related property access is not supported.", at: nameToken.range.start)
                }
                if case .identifier("world") = expression.kind, name != "read" {
                    throw failure(.unsupportedCall, "The only world facade member is read.", at: dot.range.start)
                }
                expression = try makeExpression(
                    .member(base: expression, name: name),
                    range: WorkflowSourceRange(start: expression.sourceRange.start, end: nameToken.range.end),
                    location: dot.range.start)
                continue
            }

            if peekSymbol(.leftBracket) {
                throw failure(.unsupportedExpression, "Computed property access is not supported.", at: current.range.start)
            }

            guard let open = consumeSymbol(.leftParen) else { break }
            let arguments = try parseArguments(after: open)
            let target = try callTarget(for: expression)
            if loopBodyDepth > 0, target == .parallel || target == .criticLoop {
                throw failure(.facadeCallInsideLoop, "\(target.rawValue) cannot appear inside a for-of body.", at: expression.sourceRange.start)
            }
            try validateArity(target, count: arguments.count, at: expression.sourceRange.start)
            let callRange = WorkflowSourceRange(start: expression.sourceRange.start, end: currentPreviousEnd)
            try countNode(at: callRange.start)
            let call = WorkflowCall(target: target, arguments: arguments, sourceRange: callRange)
            expression = try makeExpression(
                .call(call),
                range: callRange,
                location: callRange.start)
        }

        return expression
    }

    private mutating func parseArguments(after open: WorkflowToken) throws -> [WorkflowExpression] {
        try enterNesting(at: open.range.start)
        defer { nesting -= 1 }

        var arguments: [WorkflowExpression] = []
        if consumeSymbol(.rightParen) != nil {
            return arguments
        }
        while true {
            arguments.append(try parseExpression())
            if consumeSymbol(.rightParen) != nil {
                return arguments
            }
            _ = try expectSymbol(.comma)
            if consumeSymbol(.rightParen) != nil {
                return arguments
            }
        }
    }

    private mutating func parsePrimary() throws -> WorkflowExpression {
        let token = current
        switch token.kind {
        case .number(let value):
            _ = advance()
            return try makeExpression(.literal(.number(value)), range: token.range, location: token.range.start)
        case .string(let value):
            _ = advance()
            return try makeExpression(.literal(.string(value)), range: token.range, location: token.range.start)
        case .identifier(let name):
            _ = advance()
            switch name {
            case "true":
                return try makeExpression(.literal(.boolean(true)), range: token.range, location: token.range.start)
            case "false":
                return try makeExpression(.literal(.boolean(false)), range: token.range, location: token.range.start)
            case "null":
                return try makeExpression(.literal(.null), range: token.range, location: token.range.start)
            case "new", "this", "super", "delete", "void", "typeof", "instanceof", "in", "yield":
                throw failure(.unsupportedExpression, "This expression form is not supported.", at: token.range.start)
            default:
                guard !Self.reservedWords.contains(name) else {
                    throw failure(.unsupportedExpression, "Reserved words are not values.", at: token.range.start)
                }
                return try makeExpression(.identifier(name), range: token.range, location: token.range.start)
            }
        case .symbol(.leftParen):
            let open = advance()
            try enterNesting(at: open.range.start)
            defer { nesting -= 1 }
            let value = try parseExpression()
            let close = try expectSymbol(.rightParen)
            return try makeExpression(
                value.kind,
                range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                location: open.range.start)
        case .symbol(.leftBracket):
            return try parseArray()
        case .symbol(.leftBrace):
            return try parseObject()
        case .symbol(.assign):
            throw failure(.unsupportedOperator, "Assignment expressions are not supported.", at: token.range.start)
        case .end:
            throw failure(.unexpectedToken, "Expected an expression.", at: token.range.start)
        case .symbol(.rightParen), .symbol(.rightBrace), .symbol(.rightBracket),
             .symbol(.dot), .symbol(.comma), .symbol(.colon), .symbol(.semicolon), .symbol(.bang),
             .symbol(.strictEqual), .symbol(.strictNotEqual), .symbol(.andAnd), .symbol(.orOr),
             .symbol(.lessThan), .symbol(.lessThanOrEqual), .symbol(.greaterThan),
             .symbol(.greaterThanOrEqual):
            throw failure(.unsupportedExpression, "Expected a literal, value, array, or object expression.", at: token.range.start)
        case .symbol(.unsupportedOperator(let spelling)):
            throw failure(.unsupportedOperator, "Operator '\(spelling)' is not supported.", at: token.range.start)
        }
    }

    private mutating func parseArray() throws -> WorkflowExpression {
        let open = advance()
        try enterNesting(at: open.range.start)
        defer { nesting -= 1 }
        var values: [WorkflowExpression] = []
        if let close = consumeSymbol(.rightBracket) {
            return try makeExpression(
                .array(values),
                range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                location: open.range.start)
        }

        while true {
            guard !peekSymbol(.comma) else {
                throw failure(.unsupportedExpression, "Array holes are not supported.", at: current.range.start)
            }
            values.append(try parseExpression())
            if let close = consumeSymbol(.rightBracket) {
                return try makeExpression(
                    .array(values),
                    range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                    location: open.range.start)
            }
            _ = try expectSymbol(.comma)
            if let close = consumeSymbol(.rightBracket) {
                return try makeExpression(
                    .array(values),
                    range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                    location: open.range.start)
            }
        }
    }

    private mutating func parseObject() throws -> WorkflowExpression {
        let open = advance()
        try enterNesting(at: open.range.start)
        defer { nesting -= 1 }
        var members: [WorkflowObjectMember] = []
        if let close = consumeSymbol(.rightBrace) {
            return try makeExpression(
                .object(members),
                range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                location: open.range.start)
        }

        while true {
            let nameToken = current
            let name: String
            switch nameToken.kind {
            case .identifier(let value) where !Self.reservedWords.contains(value):
                name = value
                _ = advance()
            case .string(let value):
                name = value
                _ = advance()
            default:
                throw failure(.unsupportedExpression, "Object keys must be static identifiers or string literals.", at: nameToken.range.start)
            }
            _ = try expectSymbol(.colon, rule: .unsupportedExpression)
            let value = try parseExpression()
            try countNode(at: nameToken.range.start)
            members.append(WorkflowObjectMember(
                name: name,
                nameRange: nameToken.range,
                value: value,
                sourceRange: WorkflowSourceRange(start: nameToken.range.start, end: value.sourceRange.end)))

            if let close = consumeSymbol(.rightBrace) {
                return try makeExpression(
                    .object(members),
                    range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                    location: open.range.start)
            }
            _ = try expectSymbol(.comma)
            if let close = consumeSymbol(.rightBrace) {
                return try makeExpression(
                    .object(members),
                    range: WorkflowSourceRange(start: open.range.start, end: close.range.end),
                    location: open.range.start)
            }
        }
    }

    private mutating func validateInitializer(_ expression: WorkflowExpression) throws {
        switch expression.kind {
        case .call(let call) where call.target == .parallel:
            try validateFacadeArguments(call)
        case .awaited(let call) where Self.initializerAwaitableTargets.contains(call.target):
            try validateFacadeArguments(call)
        default:
            guard isPure(expression) else {
                throw failure(.misplacedFacadeCall, "Facade operations must appear as a declaration value or statement.", at: expression.sourceRange.start)
            }
        }
    }

    private mutating func validateStatementExpression(_ expression: WorkflowExpression) throws {
        switch expression.kind {
        case .call(let call) where [.agent, .phase, .command].contains(call.target):
            try validateFacadeArguments(call)
        case .awaited(let call) where [.report, .artifact].contains(call.target):
            try validateFacadeArguments(call)
        default:
            throw failure(.misplacedFacadeCall, "This facade operation is not a supported standalone statement.", at: expression.sourceRange.start)
        }
    }

    private mutating func validateFacadeArguments(_ call: WorkflowCall) throws {
        if call.target == .worldRead {
            guard call.arguments.count == 1 else {
                throw failure(.unsupportedWorldReadOperation, "world.read takes exactly one read operation.", at: call.sourceRange.start)
            }
            let operation = call.arguments[0]
            if case .call(let operationCall) = operation.kind,
               [.glob, .read, .grep, .git].contains(operationCall.target) {
                try validatePureArguments(operationCall)
            } else if !isPure(operation) {
                throw failure(.unsupportedWorldReadOperation, "world.read needs a fixed read operation.", at: operation.sourceRange.start)
            }
            return
        }

        try validatePureArguments(call)
        if call.target == .run, call.arguments.count == 2,
           case .object(let members) = call.arguments[1].kind,
           members.contains(where: { ["executable", "workingDirectory", "argv"].contains($0.name) }) {
            throw failure(
                .runDispatchExecutableOverride,
                "run accepts only named dynamic values for a prior command key.",
                at: call.arguments[1].sourceRange.start)
        }
    }

    private mutating func validatePureArguments(_ call: WorkflowCall) throws {
        for argument in call.arguments where !isPure(argument) {
            throw failure(.unsupportedExpression, "Facade arguments cannot contain nested facade calls.", at: argument.sourceRange.start)
        }
    }

    private func isPure(_ expression: WorkflowExpression) -> Bool {
        switch expression.kind {
        case .literal, .identifier:
            true
        case .member(let base, _), .unaryNot(let base):
            isPure(base)
        case .array(let values):
            values.allSatisfy(isPure)
        case .object(let members):
            members.allSatisfy { isPure($0.value) }
        case .binary(let left, _, let right):
            isPure(left) && isPure(right)
        case .call, .awaited:
            false
        }
    }

    private func isSupportedExpressionStatement(_ expression: WorkflowExpression) -> Bool {
        switch expression.kind {
        case .call(let call):
            [.agent, .phase, .command].contains(call.target)
        case .awaited(let call):
            [.report, .artifact].contains(call.target)
        default:
            false
        }
    }

    private mutating func callTarget(for expression: WorkflowExpression) throws -> WorkflowCallTarget {
        let name: String
        switch expression.kind {
        case .identifier(let value):
            name = value
        case .member(let base, let member):
            if case .identifier("world") = base.kind, member == "read" {
                return .worldRead
            }
            throw failure(.unsupportedCall, "Only declared facade and world-read operations may be called.", at: expression.sourceRange.start)
        default:
            throw failure(.unsupportedCall, "Only declared facade and world-read operations may be called.", at: expression.sourceRange.start)
        }
        guard let target = WorkflowCallTarget(rawValue: name) else {
            throw failure(.unsupportedCall, "Call to '\(name)' is not part of facade v1.", at: expression.sourceRange.start)
        }
        return target
    }

    private mutating func validateArity(
        _ target: WorkflowCallTarget,
        count: Int,
        at location: WorkflowSourceLocation
    ) throws {
        let accepted: Bool
        switch target {
        case .agent, .command, .run, .read:
            accepted = count == 2
        case .ask:
            accepted = count == 3
        case .parallel, .join, .criticLoop, .phase, .worldRead, .report, .artifact, .glob, .git:
            accepted = count == 1
        case .grep:
            accepted = (1...2).contains(count)
        }
        guard accepted else {
            throw failure(.unsupportedCall, "Call to '\(target.rawValue)' has an unsupported argument count.", at: location)
        }
    }

    private mutating func makeExpression(
        _ kind: WorkflowExpression.Kind,
        range: WorkflowSourceRange,
        location: WorkflowSourceLocation
    ) throws -> WorkflowExpression {
        try countNode(at: location)
        let depth: Int
        switch kind {
        case .literal, .identifier:
            depth = 1
        case .member(let base, _), .unaryNot(let base):
            depth = base.treeDepth + 1
        case .array(let values):
            depth = 1 + (values.map(\.treeDepth).max() ?? 0)
        case .object(let members):
            depth = 1 + (members.map { $0.value.treeDepth }.max() ?? 0)
        case .binary(let left, _, let right):
            depth = 1 + max(left.treeDepth, right.treeDepth)
        case .call(let call), .awaited(let call):
            depth = 2 + (call.arguments.map(\.treeDepth).max() ?? 0)
        }
        guard depth <= limits.maxNesting else {
            throw failure(
                .nestingLimitExceeded,
                "AST expression depth exceeds the \(limits.maxNesting)-level limit.",
                at: location)
        }
        return WorkflowExpression(kind: kind, sourceRange: range, treeDepth: depth)
    }

    private mutating func countNode(at location: WorkflowSourceLocation) throws {
        guard nodeCount < limits.maxASTNodes else {
            throw failure(
                .astNodeLimitExceeded,
                "AST exceeds the \(limits.maxASTNodes)-node limit.",
                at: location)
        }
        nodeCount += 1
    }

    private mutating func enterNesting(at location: WorkflowSourceLocation) throws {
        guard nesting < limits.maxNesting else {
            throw failure(
                .nestingLimitExceeded,
                "Syntax nesting exceeds the \(limits.maxNesting)-level limit.",
                at: location)
        }
        nesting += 1
    }

    private func failure(
        _ rule: WorkflowScriptDiagnosticRule,
        _ message: String,
        at location: WorkflowSourceLocation
    ) -> WorkflowParseFailure {
        WorkflowParseFailure(diagnostic: WorkflowScriptDiagnostic(
            rule: rule,
            message: message,
            location: location))
    }

    private var currentPreviousEnd: WorkflowSourceLocation {
        tokens[max(0, index - 1)].range.end
    }

    private var isUnsupportedStatementStart: Bool {
        guard case .identifier(let name) = current.kind else {
            return peekSymbol(.semicolon) || peekSymbol(.leftBrace)
        }
        return Self.unsupportedStatementWords.contains(name)
    }

    private func peekKeyword(_ keyword: String) -> Bool {
        peekIdentifier(keyword)
    }

    private func symbolName(_ symbol: WorkflowSymbol) -> String {
        switch symbol {
        case .leftParen: "("
        case .rightParen: ")"
        case .leftBrace: "{"
        case .rightBrace: "}"
        case .leftBracket: "["
        case .rightBracket: "]"
        case .dot: "."
        case .comma: ","
        case .colon: ":"
        case .semicolon: ";"
        case .assign: "="
        case .bang: "!"
        case .strictEqual: "==="
        case .strictNotEqual: "!=="
        case .andAnd: "&&"
        case .orOr: "||"
        case .lessThan: "<"
        case .lessThanOrEqual: "<="
        case .greaterThan: ">"
        case .greaterThanOrEqual: ">="
        case .unsupportedOperator(let spelling): spelling
        }
    }

    private static let reservedWords: Set<String> = [
        "async", "await", "const", "else", "false", "for", "function", "if", "null", "of", "true", "workflow"
    ]

    private static let unsupportedStatementWords: Set<String> = [
        "break", "class", "continue", "debugger", "do", "export", "import", "let", "return", "switch",
        "throw", "try", "var", "while", "with"
    ]

    private static let prototypeProperties: Set<String> = ["__proto__", "constructor", "prototype"]

    private static let awaitableTargets: Set<WorkflowCallTarget> = [
        .ask, .join, .criticLoop, .worldRead, .run, .report, .artifact
    ]

    private static let initializerAwaitableTargets: Set<WorkflowCallTarget> = [
        .ask, .join, .criticLoop, .worldRead, .run
    ]
}
