import XCTest
@testable import TurboSparkApp

final class WorkflowScriptSemanticValidationTests: XCTestCase {
    func testCompleteScriptWithBranchingAndJoinedGraphHasNoSemanticDiagnostics() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        agent("writer", "Writes drafts");
        phase("draft");
        const graph = parallel([
          { id: "first", actor: "writer", prompt: "draft", shape: { answer: "string" }, dependsOn: [], maxRetries: 0 },
          { id: "second", actor: "writer", prompt: "review", shape: { answer: "string" }, dependsOn: ["first"], maxRetries: 1 }
        ]);
        const results = await join(graph);
        if (args.ready) { await report({ results: results }); } else { await artifact(["pending"]); }
        """))

        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"))
    }

    func testDuplicateActorNamePointsAtSecondDeclaration() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        agent("writer", "First");
        agent("writer", "Second");
        """))

        assertDiagnostic("duplicateActorName", in: result, sourceLine: 3)
    }

    func testActorNameCannotClaimReservedEntryLane() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        agent("main", "Writes drafts");
        await report("done");
        """))

        assertDiagnostic("reservedActorName", in: result, sourceLine: 2)
    }

    func testAskAndGraphReferencesRequirePreviouslyDeclaredActors() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        const answer = await ask("missing", "prompt", {});
        const graph = parallel([{ id: "node", actor: "later", prompt: answer, shape: {}, dependsOn: [], maxRetries: 0 }]);
        agent("later", "Declared too late");
        const joined = await join(graph);
        """))

        XCTAssertEqual(result.diagnostics.filter { $0.rule.rawValue == "unknownActor" }.count, 2)
        XCTAssertTrue(result.diagnostics.filter { $0.rule.rawValue == "unknownActor" }.allSatisfy { $0.location.line > 1 })
    }

    func testCriticProducerAndReviewerActorsMustBeDeclared() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        const review = await criticLoop({
          producer: { actor: "writer", prompt: "draft", shape: {} },
          critic: { actor: "reviewer", prompt: "review" },
          verdictField: "verdict", feedbackField: "feedback", maxIterations: 2
        });
        """))

        XCTAssertEqual(result.diagnostics.filter { $0.rule.rawValue == "unknownActor" }.count, 2)
        XCTAssertTrue(result.diagnostics.filter { $0.rule.rawValue == "unknownActor" }.allSatisfy { $0.location.line > 1 })
    }

    func testCommandPinsMustBeUniqueAndDeclaredBeforeRun() {
        let forwardReference = WorkflowScriptChecker.parse(wrapped("""
        const result = await run("build", {});
        command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: [] });
        """))
        assertDiagnostic("unknownCommandPin", in: forwardReference, sourceLine: 2)

        let duplicate = WorkflowScriptChecker.parse(wrapped("""
        command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: [] });
        command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: [] });
        const result = await run("build", {});
        """))
        assertDiagnostic("duplicateCommandPin", in: duplicate, sourceLine: 3)
    }

    func testGraphRejectsDuplicateNodeIDsAndUnknownDependencies() {
        let duplicate = WorkflowScriptChecker.parse(wrapped("""
        agent("writer", "Writes");
        const graph = parallel([
          { id: "same", actor: "writer", prompt: "a", shape: {}, dependsOn: [], maxRetries: 0 },
          { id: "same", actor: "writer", prompt: "b", shape: {}, dependsOn: ["absent"], maxRetries: 0 }
        ]);
        const joined = await join(graph);
        """))
        assertDiagnostic("duplicateGraphNodeID", in: duplicate, sourceLine: 5)
        assertDiagnostic("unknownGraphDependency", in: duplicate, sourceLine: 5)
    }

    func testGraphRejectsDependencyCyclesAtCycleEdge() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        agent("writer", "Writes");
        const graph = parallel([
          { id: "first", actor: "writer", prompt: "a", shape: {}, dependsOn: ["second"], maxRetries: 0 },
          { id: "second", actor: "writer", prompt: "b", shape: {}, dependsOn: ["first"], maxRetries: 0 }
        ]);
        const joined = await join(graph);
        """))

        assertDiagnostic("graphCycle", in: result, sourceLine: 5)
    }

    func testGraphNodeCountIsBounded() {
        let nodes = (0..<101).map { index in
            "{ id: \"node-\(index)\", actor: \"writer\", prompt: \"work\", shape: {}, dependsOn: [], maxRetries: 0 }"
        }.joined(separator: ",\n")
        let result = WorkflowScriptChecker.parse(wrapped("""
        agent("writer", "Writes");
        const graph = parallel([
        \(nodes)
        ]);
        const joined = await join(graph);
        """))

        assertDiagnostic("graphNodeLimitExceeded", in: result, sourceLine: 3)
    }

    func testGraphMustBeJoinedOnEveryConditionalPath() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        const graph = parallel([]);
        if (args.ready) { const joined = await join(graph); } else { await report("skipped"); }
        """))

        assertDiagnostic("graphNotJoinedOnEveryPath", in: result, sourceLine: 2)
    }

    func testGraphMayBeJoinedInEachBranchButCannotBeJoinedTwice() {
        let bothBranches = WorkflowScriptChecker.parse(wrapped("""
        const graph = parallel([]);
        if (args.ready) { const a = await join(graph); } else { const b = await join(graph); }
        """))
        XCTAssertTrue(bothBranches.isValid, bothBranches.diagnostics.map(\.message).joined(separator: "\n"))

        let repeated = WorkflowScriptChecker.parse(wrapped("""
        const graph = parallel([]);
        const first = await join(graph);
        const second = await join(graph);
        """))
        assertDiagnostic("graphJoinedMoreThanOnce", in: repeated, sourceLine: 4)
    }

    func testJoinCannotBePlacedInsideARepeatableLoop() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        const graph = parallel([]);
        for (const item of items) { const joined = await join(graph); }
        """))

        assertDiagnostic("graphJoinInsideLoop", in: result, sourceLine: 3)
    }

    func testGraphJoinMustReferenceAnEarlierGraphBinding() {
        let result = WorkflowScriptChecker.parse(wrapped("const joined = await join(missing);"))
        assertDiagnostic("unknownGraphReference", in: result, sourceLine: 2)
    }

    func testActorCommandAndPhaseDeclarationsMustStayAtEntryBlockTopLevel() {
        let result = WorkflowScriptChecker.parse(wrapped("""
        if (args.enabled) {
          agent("writer", "Writes");
          command("build", { executable: "/usr/bin/tool", workingDirectory: "workspace", argv: [] });
          phase("nested");
        }
        """))

        XCTAssertEqual(result.diagnostics.filter { $0.rule.rawValue == "misplacedFacadeCall" }.count, 2)
        assertDiagnostic("misplacedPhaseMarker", in: result, sourceLine: 5)
    }

    func testPublishedPayloadsMustBeSerializableAndWithinTheStaticSizeLimit() {
        let invalid = WorkflowScriptChecker.parse(wrapped("await report({ value: true, value: false });"))
        assertDiagnostic("unserializablePublishedPayload", in: invalid, sourceLine: 2)

        let largeString = String(repeating: "x", count: 66_000)
        let oversized = WorkflowScriptChecker.parse(wrapped("""
        const payload = { dynamic: args.value, text: "\(largeString)" };
        await artifact(payload);
        """))
        assertDiagnostic("unserializablePublishedPayload", in: oversized, sourceLine: 3)
    }

    func testDefinitionCheckRejectsUndeclaredArgumentsAcrossAllBranches() {
        let result = WorkflowScriptChecker.checkDefinition(
            wrapped("""
            if (args.allowed) { await report(args.title); } else { await artifact(args.missing.value); }
            """),
            declarations: [WorkflowArgumentDeclaration(name: "allowed", required: false, maximumUTF8Bytes: 100),
                           WorkflowArgumentDeclaration(name: "title", required: true, maximumUTF8Bytes: 100)])

        assertDiagnostic("undeclaredWorkflowArgument", in: result, sourceLine: 2)
        let argumentDiagnostic = result.diagnostics.first(where: { $0.rule.rawValue == "undeclaredWorkflowArgument" })
        XCTAssertEqual(argumentDiagnostic?.location.column, 76)
        XCTAssertEqual(result.diagnostics.filter { $0.rule.rawValue == "undeclaredWorkflowArgument" }.count, 1)
    }

    func testPublishedPayloadAliasExpansionStopsAtTheStaticAnalysisBudget() {
        var declarations = ["const value0 = 0;"]
        for index in 1...16 {
            declarations.append("const value\(index) = [value\(index - 1), value\(index - 1)];")
        }
        let result = WorkflowScriptChecker.parse(wrapped(
            declarations.joined(separator: "\n") + "\nawait report(value16);"))

        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.diagnostics.filter { $0.rule == .staticAnalysisBudgetExceeded }.count, 1)
    }

    func testPublishedPayloadAliasDepthIsBoundedAcrossReportPositions() {
        var declarations = ["const value0 = 0;"]
        for index in 1...90 {
            declarations.append("const value\(index) = value\(index - 1);")
        }
        let result = WorkflowScriptChecker.parse(wrapped(
            declarations.joined(separator: "\n") + "\nawait report(value90);\nawait artifact(value90);"))

        XCTAssertFalse(result.isValid)
        XCTAssertEqual(result.diagnostics.filter { $0.rule == .staticAnalysisBudgetExceeded }.count, 1)
    }

    private func wrapped(_ body: String) -> String {
        "async function workflow() {\n\(body)\n}"
    }

    private func assertDiagnostic(
        _ rule: String,
        in result: WorkflowScriptParseResult,
        sourceLine: Int,
        file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let diagnostic = result.diagnostics.first { $0.rule.rawValue == rule }
        XCTAssertNotNil(diagnostic, "Expected \(rule), got: \(result.diagnostics.map(\.message))", file: file, line: line)
        XCTAssertEqual(diagnostic?.location.line, sourceLine, file: file, line: line)
        XCTAssertGreaterThan(diagnostic?.location.byteOffset ?? 0, 0, file: file, line: line)
        XCTAssertFalse(diagnostic?.message.isEmpty ?? true, file: file, line: line)
    }
}
