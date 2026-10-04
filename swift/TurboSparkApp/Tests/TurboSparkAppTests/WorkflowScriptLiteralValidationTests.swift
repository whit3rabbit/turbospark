import XCTest
@testable import TurboSparkApp

final class WorkflowScriptLiteralValidationTests: XCTestCase {
    func testEveryLiteralOnlyFieldHasNamedDiagnosticAtItsSourceLocation() {
        let cases: [FailureCase] = [
            .init("agent name", "agent(args.actorName, \"role\");", .literalOnlyPosition, "args.actorName"),
            .init("agent role", "agent(\"writer\", args.actorRole);", .literalOnlyPosition, "args.actorRole"),
            .init("phase name", "phase(args.phaseName);", .literalOnlyPosition, "args.phaseName"),
            .init("ask actor", "const answer = await ask(args.askActor, \"prompt\", shape);", .literalOnlyPosition, "args.askActor"),
            .init("ask shape", "const answer = await ask(\"writer\", \"prompt\", args.askShape);", .shapeMustBeStatic, "args.askShape"),
            .init("command key", "command(args.commandKey, { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [] });", .literalOnlyPosition, "args.commandKey"),
            .init("command definition", "command(\"build\", args.commandDefinition);", .literalStructureRequired, "args.commandDefinition"),
            .init("command executable", "command(\"build\", { executable: args.executable, workingDirectory: \"workspace\", argv: [] });", .literalOnlyPosition, "args.executable"),
            .init("absolute executable", "command(\"build\", { executable: \"tools/build\", workingDirectory: \"workspace\", argv: [] });", .executablePathMustBeAbsolute, "\"tools/build"),
            .init("working directory", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: args.cwd, argv: [] });", .literalOnlyPosition, "args.cwd"),
            .init("argv array and fixed count", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: args.argv });", .literalStructureRequired, "args.argv"),
            .init("fixed argv item", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [args.fixedArg] });", .literalOnlyPosition, "args.fixedArg"),
            .init("slot name", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: args.slotName, kind: \"workspaceInputPath\" }] });", .literalOnlyPosition, "args.slotName"),
            .init("slot kind", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"path\", kind: args.slotKind }] });", .literalOnlyPosition, "args.slotKind"),
            .init("slot enum values array", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"mode\", kind: \"allowedValue\", values: args.values }] });", .literalStructureRequired, "args.values"),
            .init("slot enum item", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"mode\", kind: \"allowedValue\", values: [args.value] }] });", .literalOnlyPosition, "args.value"),
            .init("slot kind enum", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"mode\", kind: \"shell\" }] });", .literalEnumValueUnsupported, "\"shell"),
            .init("bounded text byte limit", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"note\", kind: \"boundedText\", maximumBytes: args.byteLimit }] });", .literalOnlyPosition, "args.byteLimit"),
            .init("bounded text range", "command(\"build\", { executable: \"/bin/tool\", workingDirectory: \"workspace\", argv: [{ name: \"note\", kind: \"boundedText\", maximumBytes: 4097 }] });", .literalValueOutOfRange, "4097"),
            .init("parallel array literal", "const graph = parallel(args.nodes);", .literalStructureRequired, "args.nodes"),
            .init("parallel node id", "const graph = parallel([{ id: args.nodeID, actor: \"writer\", prompt: args.prompt, shape: shape, dependsOn: [], maxRetries: 1 }]);", .literalOnlyPosition, "args.nodeID"),
            .init("parallel node actor", "const graph = parallel([{ id: \"node\", actor: args.nodeActor, prompt: args.prompt, shape: shape, dependsOn: [], maxRetries: 1 }]);", .literalOnlyPosition, "args.nodeActor"),
            .init("parallel node shape", "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: args.prompt, shape: args.nodeShape, dependsOn: [], maxRetries: 1 }]);", .shapeMustBeStatic, "args.nodeShape"),
            .init("parallel dependency array", "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: args.prompt, shape: shape, dependsOn: args.dependencies, maxRetries: 1 }]);", .literalStructureRequired, "args.dependencies"),
            .init("parallel dependency item", "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: args.prompt, shape: shape, dependsOn: [args.dependency], maxRetries: 1 }]);", .literalOnlyPosition, "args.dependency"),
            .init("parallel retry bound literal", "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: args.prompt, shape: shape, dependsOn: [], maxRetries: args.retries }]);", .literalOnlyPosition, "args.retries"),
            .init("parallel retry range", "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: args.prompt, shape: shape, dependsOn: [], maxRetries: 4 }]);", .literalValueOutOfRange, "4"),
            .init("critic policy object", "const review = await criticLoop(args.policy);", .literalStructureRequired, "args.policy"),
            .init("critic producer actor", "const review = await criticLoop({ producer: { actor: args.producerActor, prompt: args.producerPrompt, shape: shape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: 2 });", .literalOnlyPosition, "args.producerActor"),
            .init("critic reviewer actor", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: shape }, critic: { actor: args.criticActor, prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: 2 });", .literalOnlyPosition, "args.criticActor"),
            .init("critic producer shape", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: args.criticShape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: 2 });", .shapeMustBeStatic, "args.criticShape"),
            .init("critic verdict field", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: shape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: args.verdictField, feedbackField: \"feedback\", maxIterations: 2 });", .literalOnlyPosition, "args.verdictField"),
            .init("critic feedback field", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: shape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: args.feedbackField, maxIterations: 2 });", .literalOnlyPosition, "args.feedbackField"),
            .init("critic iteration bound literal", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: shape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: args.iterations });", .literalOnlyPosition, "args.iterations"),
            .init("critic iteration range", "const review = await criticLoop({ producer: { actor: \"writer\", prompt: args.producerPrompt, shape: shape }, critic: { actor: \"reviewer\", prompt: args.criticPrompt }, verdictField: \"verdict\", feedbackField: \"feedback\", maxIterations: 11 });", .literalValueOutOfRange, "11"),
            .init("run command key", "const result = await run(args.commandKey, { path: args.path });", .runCommandKeyMustBeLiteral, "args.commandKey"),
            .init("run values object", "const result = await run(\"build\", \"not a named values object\");", .runRequiresNamedDynamicValues, "\"not a named values object"),
            .init("world operation", "const result = await world.read(args.operation);", .worldOperationMustBeStatic, "args.operation"),
            .init("glob pattern", "const result = await world.read(glob(args.globPattern));", .literalOnlyPosition, "args.globPattern"),
            .init("read path", "const result = await world.read(read(args.readPath, 4096));", .literalOnlyPosition, "args.readPath"),
            .init("read byte limit", "const result = await world.read(read(\"README.md\", args.readLimit));", .literalOnlyPosition, "args.readLimit"),
            .init("grep pattern", "const result = await world.read(grep(args.grepPattern));", .literalOnlyPosition, "args.grepPattern"),
            .init("grep path hint", "const result = await world.read(grep(\"query\", args.pathHint));", .literalOnlyPosition, "args.pathHint"),
            .init("git operation literal", "const result = await world.read(git(args.gitOperation));", .literalOnlyPosition, "args.gitOperation"),
            .init("git operation enum", "const result = await world.read(git(\"execute\"));", .literalEnumValueUnsupported, "\"execute")
        ]

        for testCase in cases {
            let source = wrapped("const shape = { type: \"string\" };\n" + testCase.body)
            let result = WorkflowScriptChecker.parse(source)
            XCTAssertNotNil(result.ast, "\(testCase.name): syntax should parse before literal validation")
            let diagnostic = result.diagnostics.first
            XCTAssertEqual(diagnostic?.rule, testCase.rule, testCase.name)

            guard let diagnostic, let match = source.range(of: testCase.marker, options: .backwards) else { continue }
            let prefix = source[..<match.lowerBound]
            XCTAssertEqual(diagnostic.location.byteOffset, prefix.utf8.count, testCase.name)
            XCTAssertEqual(diagnostic.location.line, prefix.filter(\.isNewline).count + 1, testCase.name)
            let columnPrefix = prefix.split(separator: "\n", omittingEmptySubsequences: false).last ?? ""
            XCTAssertEqual(diagnostic.location.column, columnPrefix.unicodeScalars.count + 1, testCase.name)
        }
    }

    func testStaticShapeAliasesPreserveArbitraryLiteralContents() throws {
        let source = wrapped(
            "const schemaSource = { anyKey: [\"opaque\", true, null, { nested: 2 }] };\n" +
                "const shape = schemaSource;\n" +
                "agent(\"writer\", \"Writes\");\n" +
                "const answer = await ask(\"writer\", args.prompt, shape);\n" +
                "const graph = parallel([{ id: \"node\", actor: \"writer\", prompt: answer, shape: shape, dependsOn: [], maxRetries: 0 }]);\n" +
                "const joined = await join(graph);")
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))

        let ast = try XCTUnwrap(result.ast)
        guard case .declaration(_, _, let schema) = ast.body.statements.first?.kind else {
            return XCTFail("Expected the original static shape expression to remain in the AST")
        }
        guard case .object(let members) = schema.kind,
              members.count == 1,
              members[0].name == "anyKey",
              case .array(let entries) = members[0].value.kind,
              entries.count == 4
        else {
            return XCTFail("Shape contents should be preserved without schema interpretation")
        }
        XCTAssertEqual(entries[0].kind, .literal(.string("opaque")))
        XCTAssertEqual(entries[1].kind, .literal(.boolean(true)))
        XCTAssertEqual(entries[2].kind, .literal(.null))
        guard case .object(let nestedMembers) = entries[3].kind else {
            return XCTFail("Expected the nested static object to remain in the shape")
        }
        XCTAssertEqual(nestedMembers.first?.name, "nested")
        XCTAssertEqual(nestedMembers.first?.value.kind, .literal(.number(2)))
    }

    func testDocumentedDynamicPromptArgumentAndPayloadPositionsRemainAccepted() {
        let source = wrapped("""
        const shape = { type: "object", properties: { answer: ["string", { nullable: true }] } };
        agent("writer", "Writes a report");
        phase("draft");
        command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: ["build", { name: "path", kind: "workspaceInputPath" }, { name: "mode", kind: "allowedValue", values: ["fast", "safe"] }, { name: "note", kind: "boundedText", maximumBytes: 4096 }] });
        const answer = await ask("writer", args.prompt, shape);
        const graph = parallel([{ id: "draft", actor: "writer", prompt: answer, shape: shape, dependsOn: [], maxRetries: 1 }]);
        const results = await join(graph);
        const review = await criticLoop({ producer: { actor: "writer", prompt: results, shape: shape }, critic: { actor: "writer", prompt: args.reviewPrompt }, verdictField: "verdict", feedbackField: "feedback", maxIterations: 3 });
        const files = await world.read(glob("Sources/**"));
        const contents = await world.read(read("README.md", 4096));
        const matches = await world.read(grep("Workflow", "Sources"));
        const status = await world.read(git("status"));
        const commandResult = await run("build", { path: args.sourcePath, mode: args.mode, note: args.note });
        const canonicalRunResult = await run("build", args);
        await report({ answer: answer, results: results, review: review, files: files, contents: contents, matches: matches, status: status, command: commandResult, canonicalCommand: canonicalRunResult });
        await artifact(args.artifactValue);
        """)
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
    }

    func testStaticShapeResolverRejectsDynamicBindingAndNestedDynamicValue() {
        let cases = [
            ("const shape = args.schema;\nconst answer = await ask(\"writer\", \"prompt\", shape);", "shape)"),
            ("const shape = { type: args.schemaType };\nconst answer = await ask(\"writer\", \"prompt\", shape);", "shape)"),
            ("const first = second;\nconst second = first;\nconst answer = await ask(\"writer\", \"prompt\", first);", "first)")
        ]
        for (body, marker) in cases {
            let source = wrapped("const unused = 0;\n" + body)
            let result = WorkflowScriptChecker.parse(source)
            XCTAssertEqual(result.diagnostics.first?.rule, .shapeMustBeStatic)
            if let diagnostic = result.diagnostics.first,
               let range = source.range(of: marker, options: .backwards)
            {
                XCTAssertEqual(diagnostic.location.byteOffset, source[..<range.lowerBound].utf8.count)
            }
        }
    }

    func testExponentialShapeAliasesStopAtTheSharedWorkBudget() {
        let source = wrapped("agent(\"writer\", \"Writes\");\n" + doublingBindings(levels: 20)
            + "\nconst answer = await ask(\"writer\", \"prompt\", shape20);")
        XCTAssertLessThan(source.utf8.count, WorkflowScriptChecker.Limits.production.maxSourceBytes)
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertNotNil(result.ast, "The source stays within parser byte, token, and AST limits")
        XCTAssertEqual(result.diagnostics.first?.rule, .staticAnalysisBudgetExceeded)
    }

    func testStaticAnalysisBudgetIsSharedAcrossShapePositions() {
        let uses = [
            "agent(\"writer\", \"Writes\");",
            "const first = await ask(\"writer\", \"prompt\", shape13);",
            "const second = await ask(\"writer\", \"prompt\", shape13);"
        ].joined(separator: "\n")
        let source = wrapped(doublingBindings(levels: 13) + "\n" + uses)
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertNotNil(result.ast)
        XCTAssertEqual(result.diagnostics.first?.rule, .staticAnalysisBudgetExceeded)
        XCTAssertEqual(result.diagnostics.count, 1)

        if let diagnostic = result.diagnostics.first,
           let range = source.range(of: "shape13", options: .backwards)
        {
            XCTAssertEqual(diagnostic.location.byteOffset, source[..<range.lowerBound].utf8.count)
        }
    }

    func testStaticShapeAliasResolutionUsesAnIterativeWalk() {
        var declarations = ["const shape0 = { opaque: [\"value\"] };"]
        for index in 1...2_000 {
            declarations.append("const shape\(index) = shape\(index - 1);")
        }
        declarations.append("agent(\"writer\", \"Writes\");")
        declarations.append("const answer = await ask(\"writer\", args.prompt, shape2000);")
        let result = WorkflowScriptChecker.parse(wrapped(declarations.joined(separator: "\n")))
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
    }

    private func wrapped(_ body: String) -> String {
        "async function workflow() {\n\(body)\n}"
    }

    private func doublingBindings(levels: Int) -> String {
        var declarations = ["const shape0 = {};"]
        for index in 1...levels {
            let prior = "shape\(index - 1)"
            declarations.append("const shape\(index) = [\(prior), \(prior)];")
        }
        return declarations.joined(separator: "\n")
    }

    private struct FailureCase {
        let name: String
        let body: String
        let rule: WorkflowScriptDiagnosticRule
        let marker: String
        init(_ name: String, _ body: String, _ rule: WorkflowScriptDiagnosticRule, _ marker: String) {
            self.name = name
            self.body = body
            self.rule = rule
            self.marker = marker
        }
    }
}
