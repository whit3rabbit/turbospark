import XCTest
@testable import TurboSparkApp

final class WorkflowScriptCheckerTests: XCTestCase {
    func testAcceptedScriptProducesCompleteVersionedAST() throws {
        let source = """
        async function workflow() {
          agent("writer", "Drafts a report");
          command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: ["build"] });
          phase("draft");
          const shape = { result: "string", flags: [true, false, null, 2.5] };
          const argsList = [args.input, "fixed"];
          const answer = await ask("writer", args.prompt, shape);
          const graph = parallel([{ id: "draft", actor: "writer", prompt: answer, shape: shape, dependsOn: [], maxRetries: 1 }]);
          const results = await join(graph);
          const review = await criticLoop({ producer: { actor: "writer", prompt: results, shape: shape }, critic: { actor: "writer", prompt: results }, verdictField: "verdict", feedbackField: "feedback", maxIterations: 3 });
          const files = await world.read(glob("Sources/**"));
          const contents = await world.read(read("Sources/main.swift", 4096));
          const matches = await world.read(grep("Workflow", "Sources"));
          const status = await world.read(git("status"));
          const commandResult = await run("build", { sourcePath: args.sourcePath });
          if (!args.ready && review.accepted === true || 2 < 3 && 4 >= 4) {
            await report({ value: answer });
          } else {
            await artifact({ value: results });
          }
          for (const item of files) {
            if (item !== null) { await report(item); } else { await artifact(item); }
          }
        }
        """

        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
        let ast = try XCTUnwrap(result.ast)
        XCTAssertEqual(ast.version, WorkflowScriptAST.currentVersion)
        XCTAssertEqual(ast.facadeVersion, WorkflowFacade.version)
        XCTAssertEqual(ast.sourceRange.start, WorkflowSourceLocation(byteOffset: 0, line: 1, column: 1))
        XCTAssertEqual(ast.body.statements.count, 16)

        var statementKinds: Set<String> = []
        var expressionKinds: Set<String> = []
        var callTargets: Set<WorkflowCallTarget> = []
        collect(
            ast.body,
            statementKinds: &statementKinds,
            expressionKinds: &expressionKinds,
            callTargets: &callTargets)

        XCTAssertEqual(statementKinds, ["declaration", "expression", "conditional", "forOf"])
        XCTAssertEqual(
            expressionKinds,
            ["literal", "identifier", "member", "array", "object", "unaryNot", "binary", "call", "awaited"])
        XCTAssertEqual(callTargets, Set(WorkflowCallTarget.allCases))
    }

    func testIfKeepsBothBranchesAndTheirSourceRanges() throws {
        let source = """
        async function workflow() {
          if (args.enabled) {
            await report("then");
          } else {
            await artifact("else");
          }
        }
        """
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
        let ast = try XCTUnwrap(result.ast)
        guard case .conditional(let condition, let thenBlock, let elseBlock?) = ast.body.statements[0].kind else {
            return XCTFail("Expected a conditional with both blocks")
        }

        XCTAssertEqual(condition.sourceRange.start.line, 2)
        XCTAssertEqual(thenBlock.statements.count, 1)
        XCTAssertEqual(elseBlock.statements.count, 1)
        XCTAssertLessThan(thenBlock.sourceRange.start.byteOffset, elseBlock.sourceRange.start.byteOffset)
        XCTAssertEqual(thenBlock.statements[0].sourceRange.start.line, 3)
        XCTAssertEqual(elseBlock.statements[0].sourceRange.start.line, 5)
    }

    func testForOfParsesBindingSequenceAndNestedBody() throws {
        let source = """
        async function workflow() {
          const names = args.names;
          for (const name of names) {
            const prefix = args.prefix;
            if (name === prefix) { await report(name); } else { await artifact(name); }
          }
        }
        """
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
        let ast = try XCTUnwrap(result.ast)

        guard case .forOf(let name, _, let sequence, let body) = ast.body.statements[1].kind else {
            return XCTFail("Expected a for-of statement")
        }
        XCTAssertEqual(name, "name")
        XCTAssertEqual(sequence.kind, .identifier("names"))
        XCTAssertEqual(body.statements.count, 2)
        guard case .conditional(_, let thenBlock, let elseBlock?) = body.statements[1].kind else {
            return XCTFail("Expected the loop body to contain both conditional branches")
        }
        XCTAssertEqual(thenBlock.statements.count, 1)
        XCTAssertEqual(elseBlock.statements.count, 1)
    }

    func testUnsupportedStatementExpressionAndLoopFormsHaveNamedRules() {
        let cases: [(String, WorkflowScriptDiagnosticRule)] = [
            ("while (ready) {}", .unsupportedStatement),
            ("do {} while (ready);", .unsupportedStatement),
            ("return value;", .unsupportedStatement),
            ("let value = 1;", .unsupportedStatement),
            ("fetch(\"url\");", .unsupportedCall),
            ("const value = new Date();", .unsupportedExpression),
            ("const value = input[field];", .unsupportedExpression),
            ("const value = input?.field;", .unsupportedExpression),
            ("const value = 1 + 2;", .unsupportedOperator),
            ("value = 1;", .unsupportedOperator),
            ("for (let index = 0; index < 2; index += 1) {}", .unsupportedLoopForm),
            ("for (const index in values) {}", .unsupportedLoopForm),
            ("for (const index = 0; index < 2; index = index + 1) {}", .unsupportedLoopForm),
            ("for (;;) {}", .unsupportedLoopForm),
            ("for await (const item of items) {}", .unsupportedLoopForm)
        ]

        for (body, expectedRule) in cases {
            let result = WorkflowScriptChecker.parse(wrapped(body))
            XCTAssertNil(result.ast, body)
            XCTAssertEqual(result.diagnostics.first?.rule, expectedRule, body)
        }
    }

    func testFacadeOperationsThatCannotBeParsedInsideForOfAreRejected() {
        let cases = [
            "for (const item of items) { const graph = parallel([]); }",
            "for (const item of items) { const review = await criticLoop({}); }"
        ]

        for body in cases {
            let result = WorkflowScriptChecker.parse(wrapped(body))
            XCTAssertEqual(result.diagnostics.first?.rule, .facadeCallInsideLoop, body)
        }
    }

    func testRunDispatchCannotSupplyAnExecutableOrArgv() {
        let result = WorkflowScriptChecker.parse(wrapped(
            "const result = await run(\"build\", { executable: \"/usr/bin/tool\", argv: [\"--version\"] });"))
        XCTAssertNil(result.ast)
        XCTAssertEqual(result.diagnostics.first?.rule, .runDispatchExecutableOverride)
    }

    func testDiagnosticLocationIsStableInUTF8BytesAndSourceLines() throws {
        let source = """
        async function workflow() {
          const ok = 1;
          while (true) {}
        }
        """
        let result = WorkflowScriptChecker.parse(source)
        let diagnostic = try XCTUnwrap(result.diagnostics.first)
        let expectedOffset = try XCTUnwrap(source.range(of: "while")).lowerBound
        let expectedByteOffset = source[..<expectedOffset].utf8.count

        XCTAssertEqual(diagnostic.rule, .unsupportedStatement)
        XCTAssertEqual(diagnostic.location.byteOffset, expectedByteOffset)
        XCTAssertEqual(diagnostic.location.line, 3)
        XCTAssertEqual(diagnostic.location.column, 3)
    }

    func testEachProductionResourceLimitHasItsNamedDiagnostic() {
        XCTAssertEqual(WorkflowScriptChecker.Limits.production.maxSourceBytes, 131_072)
        XCTAssertEqual(WorkflowScriptChecker.Limits.production.maxTokens, 20_000)
        XCTAssertEqual(WorkflowScriptChecker.Limits.production.maxASTNodes, 20_000)
        XCTAssertEqual(WorkflowScriptChecker.Limits.production.maxNesting, 64)

        let utf8Oversize = String(repeating: "\u{00E9}", count: 65_537)
        let byteResult = WorkflowScriptChecker.parse(utf8Oversize)
        XCTAssertEqual(byteResult.diagnostics.first?.rule, .sourceByteLimitExceeded)
        XCTAssertEqual(byteResult.diagnostics.first?.location.byteOffset, 131_072)

        let tokenOversize = wrapped(String(repeating: "const item = 1;\n", count: 4_000))
        let tokenResult = WorkflowScriptChecker.parse(tokenOversize)
        XCTAssertEqual(tokenResult.diagnostics.first?.rule, .tokenLimitExceeded)

        let astLimited = WorkflowScriptChecker.parse(
            wrapped("const item = 1;"),
            limits: WorkflowScriptChecker.Limits(maxASTNodes: 2))
        XCTAssertEqual(astLimited.diagnostics.first?.rule, .astNodeLimitExceeded)

        let nested = wrapped("const item = " + String(repeating: "(", count: 64)
            + "true" + String(repeating: ")", count: 64) + ";")
        let nestingResult = WorkflowScriptChecker.parse(nested)
        XCTAssertEqual(nestingResult.diagnostics.first?.rule, .nestingLimitExceeded)
    }

    func testNestingAtConfiguredBoundaryStillParses() {
        let expression = String(repeating: "(", count: 63) + "true" + String(repeating: ")", count: 63)
        let result = WorkflowScriptChecker.parse(wrapped("const item = \(expression);"))
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
    }

    func testIterativeExpressionChainsAlsoRespectMaximumASTDepth() {
        let memberAtLimit = "args" + String(repeating: ".value", count: 63)
        let memberBoundary = WorkflowScriptChecker.parse(wrapped("const item = \(memberAtLimit);"))
        XCTAssertTrue(memberBoundary.isValid, memberBoundary.diagnostics.map(\.message).joined(separator: "\n"))

        let memberOverLimit = "args" + String(repeating: ".value", count: 64)
        let memberOverflow = WorkflowScriptChecker.parse(wrapped("const item = \(memberOverLimit);"))
        XCTAssertEqual(memberOverflow.diagnostics.first?.rule, .nestingLimitExceeded)

        let logicalAtLimit = Array(repeating: "args", count: 64).joined(separator: " && ")
        let logicalBoundary = WorkflowScriptChecker.parse(wrapped("if (\(logicalAtLimit)) {}"))
        XCTAssertTrue(logicalBoundary.isValid, logicalBoundary.diagnostics.map(\.message).joined(separator: "\n"))

        let logicalOverLimit = Array(repeating: "args", count: 65).joined(separator: " && ")
        let logicalOverflow = WorkflowScriptChecker.parse(wrapped("if (\(logicalOverLimit)) {}"))
        XCTAssertEqual(logicalOverflow.diagnostics.first?.rule, .nestingLimitExceeded)
    }

    private func wrapped(_ body: String) -> String {
        "async function workflow() {\n\(body)\n}"
    }

    private func collect(
        _ block: WorkflowBlock,
        statementKinds: inout Set<String>,
        expressionKinds: inout Set<String>,
        callTargets: inout Set<WorkflowCallTarget>
    ) {
        for statement in block.statements {
            switch statement.kind {
            case .declaration(_, _, let value):
                statementKinds.insert("declaration")
                collect(value, expressionKinds: &expressionKinds, callTargets: &callTargets)
            case .expression(let expression):
                statementKinds.insert("expression")
                collect(expression, expressionKinds: &expressionKinds, callTargets: &callTargets)
            case .conditional(let condition, let thenBlock, let elseBlock):
                statementKinds.insert("conditional")
                collect(condition, expressionKinds: &expressionKinds, callTargets: &callTargets)
                collect(thenBlock, statementKinds: &statementKinds, expressionKinds: &expressionKinds, callTargets: &callTargets)
                if let elseBlock {
                    collect(elseBlock, statementKinds: &statementKinds, expressionKinds: &expressionKinds, callTargets: &callTargets)
                }
            case .forOf(_, _, let sequence, let body):
                statementKinds.insert("forOf")
                collect(sequence, expressionKinds: &expressionKinds, callTargets: &callTargets)
                collect(body, statementKinds: &statementKinds, expressionKinds: &expressionKinds, callTargets: &callTargets)
            }
        }
    }

    private func collect(
        _ expression: WorkflowExpression,
        expressionKinds: inout Set<String>,
        callTargets: inout Set<WorkflowCallTarget>
    ) {
        switch expression.kind {
        case .literal:
            expressionKinds.insert("literal")
        case .identifier:
            expressionKinds.insert("identifier")
        case .member(let base, _):
            expressionKinds.insert("member")
            collect(base, expressionKinds: &expressionKinds, callTargets: &callTargets)
        case .array(let values):
            expressionKinds.insert("array")
            values.forEach { collect($0, expressionKinds: &expressionKinds, callTargets: &callTargets) }
        case .object(let members):
            expressionKinds.insert("object")
            members.forEach { collect($0.value, expressionKinds: &expressionKinds, callTargets: &callTargets) }
        case .unaryNot(let value):
            expressionKinds.insert("unaryNot")
            collect(value, expressionKinds: &expressionKinds, callTargets: &callTargets)
        case .binary(let left, _, let right):
            expressionKinds.insert("binary")
            collect(left, expressionKinds: &expressionKinds, callTargets: &callTargets)
            collect(right, expressionKinds: &expressionKinds, callTargets: &callTargets)
        case .call(let call):
            expressionKinds.insert("call")
            collect(call, expressionKinds: &expressionKinds, callTargets: &callTargets)
        case .awaited(let call):
            expressionKinds.insert("awaited")
            collect(call, expressionKinds: &expressionKinds, callTargets: &callTargets)
        }
    }

    private func collect(
        _ call: WorkflowCall,
        expressionKinds: inout Set<String>,
        callTargets: inout Set<WorkflowCallTarget>
    ) {
        callTargets.insert(call.target)
        call.arguments.forEach { collect($0, expressionKinds: &expressionKinds, callTargets: &callTargets) }
    }
}
