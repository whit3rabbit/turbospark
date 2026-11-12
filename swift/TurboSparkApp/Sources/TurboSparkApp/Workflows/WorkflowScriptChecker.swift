import CryptoKit
import Foundation

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
