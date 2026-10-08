import XCTest
@testable import TurboSparkApp

final class WorkflowInterpreterTests: XCTestCase {
    func testIntegerObservationsAndLengthsCompareToNumericLiterals() async throws {
        let program = try parse("""
        async function workflow() {
          const observation = await world.read(git("status"));
          if (observation.exitStatus === 0) { await report("success"); }
          else { await report("failure"); }
          const items = ["first", "second"];
          await report(items.length === 2);
          await report(items.length !== 2);
        }
        """)
        let sink = WorkflowInterpreterTestSink(resolutions: [
            .immediate(.object(["exitStatus": .integer(0)])),
            .immediate(.null), .immediate(.null), .immediate(.null),
        ])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        try await interpreter.execute(program: program)

        let reports = await sink.requests().compactMap { request -> WorkflowCanonicalValue? in
            guard case .report(let value) = request.operation else { return nil }
            return value
        }
        XCTAssertEqual(reports, [.string("success"), .boolean(true), .boolean(false)])
    }

    func testAwaitedFacadeCallDoesNotContinueBeforeEngineResolution() async throws {
        let program = try parse("""
        async function workflow() {
          const contents = await world.read(git("status"));
          await report(contents);
        }
        """)
        let sink = WorkflowInterpreterTestSink(resolutions: [.held, .immediate(.null)])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        let execution = Task { try await interpreter.execute(program: program) }

        await sink.waitForRequestCount(1)
        let beforeResolution = await sink.requests()
        XCTAssertEqual(beforeResolution.count, 1)
        guard case .worldRead = beforeResolution[0].operation else {
            return XCTFail("The world read should reach the engine first")
        }

        await sink.resolveHeld(with: .string("engine-result"))
        await sink.waitForRequestCount(2)
        let afterResolution = await sink.requests()
        XCTAssertEqual(afterResolution.count, 2)
        XCTAssertEqual(afterResolution[1].operation, .report(value: .string("engine-result")))
        try await execution.value
    }

    func testEachAttemptGetsFreshBindingsAndFrozenArgumentValues() async throws {
        let source = """
        async function workflow() {
          const observation = await world.read(git("status"));
          await report({ observation: observation, input: args.input });
        }
        """
        let firstProgram = try parse(source, args: ["input": "first"])
        let secondProgram = try parse(source, args: ["input": "second"])
        let sink = WorkflowInterpreterTestSink(resolutions: [
            .immediate(.string("attempt-one")), .immediate(.null),
            .immediate(.string("attempt-two")), .immediate(.null)
        ])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        try await interpreter.execute(program: firstProgram)
        try await interpreter.execute(program: secondProgram)

        let reports = await sink.requests().compactMap { request -> WorkflowCanonicalValue? in
            guard case .report(let value) = request.operation else { return nil }
            return value
        }
        XCTAssertEqual(reports, [
            .object(["observation": .string("attempt-one"), "input": .string("first")]),
            .object(["observation": .string("attempt-two"), "input": .string("second")])
        ])
    }

    func testConcurrentAttemptsUseOneSerialExecutor() async throws {
        let program = try parse("async function workflow() { await report(\"once\"); }")
        let sink = WorkflowInterpreterTestSink(resolutions: [.held, .immediate(.null)])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        let first = Task { try await interpreter.execute(program: program) }

        await sink.waitForRequestCount(1)
        let second = Task { try await interpreter.execute(program: program) }
        try await Task.sleep(for: .milliseconds(50))
        let whileFirstIsHeld = await sink.requestCount()
        XCTAssertEqual(whileFirstIsHeld, 1, "A second attempt must wait for the first attempt to finish")

        await sink.resolveHeld(with: .null)
        await sink.waitForRequestCount(2)
        try await first.value
        try await second.value
        let completedRequestCount = await sink.requestCount()
        XCTAssertEqual(completedRequestCount, 2)
    }

    func testCancellingQueuedAttemptReturnsBeforeActiveAttemptFinishes() async throws {
        let program = try parse("async function workflow() { await report(\"once\"); }")
        let sink = WorkflowInterpreterTestSink(resolutions: [.held, .immediate(.null)])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        let first = Task { try await interpreter.execute(program: program) }

        await sink.waitForRequestCount(1)
        let secondFinished = WorkflowInterpreterCompletionProbe()
        let second = Task {
            do {
                try await interpreter.execute(program: program)
                await secondFinished.finish()
            } catch {
                await secondFinished.finish()
                throw error
            }
        }
        try await Task.sleep(for: .milliseconds(50))
        second.cancel()

        let completedBeforeRelease = await secondFinished.waitForCompletion(within: .milliseconds(100))
        await sink.resolveHeld(with: .null)
        try await first.value
        do {
            try await second.value
            XCTFail("A canceled queued attempt must not run")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .cancelled)
        }
        XCTAssertTrue(completedBeforeRelease, "Queued cancellation should return while the active attempt is still held")
    }

    func testDeadlineCancelsAnAwaitedOperationWithResourceLimitStatus() async throws {
        let program = try parse("async function workflow() { const result = await world.read(git(\"status\")); }")
        let sink = WorkflowInterpreterTestSink(resolutions: [.waitForCancellation])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        let deadline = ContinuousClock().now.advanced(by: .milliseconds(150))
        let execution = Task {
            try await interpreter.execute(program: program, deadline: deadline)
        }

        await sink.waitForRequestCount(1)
        do {
            try await execution.value
            XCTFail("An expired attempt deadline must fail the attempt")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .resourceLimit)
            XCTAssertEqual(error.message, "Workflow attempt deadline exceeded.")
        }
    }

    func testRequestCancelReachesTheActiveEngineOperation() async throws {
        let program = try parse("async function workflow() { const result = await world.read(git(\"status\")); }")
        let sink = WorkflowInterpreterTestSink(resolutions: [.waitForCancellation])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        let execution = Task { try await interpreter.execute(program: program) }

        await sink.waitForRequestCount(1)
        interpreter.requestCancel()
        do {
            try await execution.value
            XCTFail("The active attempt should stop after cancellation")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .cancelled)
        }
    }

    func testDynamicCommandValuesAreValidatedBeforeRunDispatch() async throws {
        let program = try parse("""
        async function workflow() {
          command("build", { executable: "/usr/bin/true", workingDirectory: "workspace", argv: [{ name: "sourcePath", kind: "workspaceInputPath" }] });
          const observation = await world.read(git("status"));
          const commandResult = await run("build", { sourcePath: observation });
        }
        """)
        let sink = WorkflowInterpreterTestSink(resolutions: [.immediate(.array([.string("not-a-string-slot")]))])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        do {
            try await interpreter.execute(program: program)
            XCTFail("A non-string dynamic command value must be rejected before dispatch")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .validation)
        }

        let requests = await sink.requests()
        XCTAssertEqual(requests.count, 1)
        guard case .worldRead = requests[0].operation else {
            return XCTFail("The invalid command must not reach the engine")
        }
    }

    func testRunChecksPinnedSlotNamesAllowedValuesAndTextBoundsBeforeDispatch() async throws {
        let source = """
        async function workflow() {
          command("build", { executable: "/usr/bin/true", workingDirectory: "workspace", argv: [
            { name: "mode", kind: "allowedValue", values: ["debug", "release"] },
            { name: "summary", kind: "boundedText", maximumBytes: 4 }
          ] });
          const result = await run("build", { mode: args.mode, summary: args.summary });
        }
        """
        let program = try parse(source, args: ["mode": "debug", "summary": "ok"])
        let unknownSlotProgram = try parse(source.replacingOccurrences(
            of: "{ mode: args.mode, summary: args.summary }",
            with: "{ mode: args.mode, summary: args.summary, unexpected: args.unexpected }"),
            args: ["mode": "debug", "summary": "ok", "unexpected": "value"])
        let sink = WorkflowInterpreterTestSink()
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        for invalidArguments in [
            ["mode": "unsafe", "summary": "ok"],
            ["mode": "debug", "summary": "longer"]
        ] {
            do {
                let invalidProgram = try parse(source, args: invalidArguments)
                try await interpreter.execute(program: invalidProgram)
                XCTFail("Invalid dynamic command values must fail before dispatch")
            } catch let error as WorkflowError {
                XCTAssertEqual(error.kind, .validation)
            }
        }
        do {
            try await interpreter.execute(
                program: unknownSlotProgram)
            XCTFail("Undeclared dynamic command slots must fail before dispatch")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .validation)
        }
        let rejectedRequestCount = await sink.requestCount()
        XCTAssertEqual(rejectedRequestCount, 0)

        try await interpreter.execute(program: program)
        let requests = await sink.requests()
        XCTAssertEqual(requests.map(\.operation), [
            .run(commandKey: "build", values: ["mode": .string("debug"), "summary": .string("ok")])
        ])
    }

    func testDynamicCommandValuesRejectNulAndDotPathComponents() async throws {
        let source = """
        async function workflow() {
          command("build", { executable: "/usr/bin/true", workingDirectory: "workspace", argv: [
            { name: "summary", kind: "boundedText", maximumBytes: 16 },
            { name: "sourcePath", kind: "workspaceInputPath" }
          ] });
          const result = await run("build", { summary: args.summary, sourcePath: args.sourcePath });
        }
        """
        let sink = WorkflowInterpreterTestSink()
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)
        for invalid in [
            ["summary": "ok\u{0}--dangerous", "sourcePath": "a/b"],
            ["summary": "ok", "sourcePath": "./a"],
            ["summary": "ok", "sourcePath": "a/./b"],
        ] {
            do {
                try await interpreter.execute(program: try parse(source, args: invalid))
                XCTFail("NUL bytes and '.' path components must fail before dispatch")
            } catch let error as WorkflowError {
                XCTAssertEqual(error.kind, .validation)
            }
        }
        let rejected = await sink.requestCount()
        XCTAssertEqual(rejected, 0)

        try await interpreter.execute(
            program: try parse(source, args: ["summary": "ok", "sourcePath": "a/b"]))
        let accepted = await sink.requestCount()
        XCTAssertEqual(accepted, 1)
    }

    func testHostGlobalsAreRefusedByTheRuntimeFacadeBoundary() async throws {
        // The checker now rejects the undeclared `process` statically; this
        // test is about the interpreter's own defense in depth.
        let program = try parseBypassingScopeChecks("async function workflow() { const value = process.env; await report(value); }")
        let sink = WorkflowInterpreterTestSink()
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        do {
            try await interpreter.execute(program: program)
            XCTFail("Host globals are outside the workflow facade")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
        let requestCount = await sink.requestCount()
        XCTAssertEqual(requestCount, 0)
    }

    func testFacadeArgumentsCannotBeShadowedByALocalBinding() async throws {
        // Same: the checker reports the reserved name, the interpreter must
        // still refuse it if a program ever reaches it unchecked.
        let program = try parseBypassingScopeChecks("async function workflow() { const args = { input: \"shadow\" }; await report(args.input); }")
        let sink = WorkflowInterpreterTestSink()
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        do {
            try await interpreter.execute(program: program)
            XCTFail("The frozen facade arguments cannot be shadowed")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .sandboxRefusal)
        }
        let requestCount = await sink.requestCount()
        XCTAssertEqual(requestCount, 0)
    }

    func testOversizedLoopArrayIsRejectedBeforeItsFirstBodyOperation() async throws {
        let program = try parse("""
        async function workflow() {
          const items = await world.read(git("status"));
          for (const item of items) { await report(item); }
        }
        """)
        let sink = WorkflowInterpreterTestSink(resolutions: [
            .immediate(.array(Array(repeating: .string("item"), count: 101)))
        ])
        let interpreter = WorkflowInterpreter(limits: WorkflowInterpreterLimits(), engine: sink)

        do {
            try await interpreter.execute(program: program)
            XCTFail("Loop arrays above 100 items must be refused")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .resourceLimit)
        }

        let requests = await sink.requests()
        XCTAssertEqual(requests.count, 1)
        guard case .worldRead = requests[0].operation else {
            return XCTFail("No loop body operation should run before the array bound check")
        }
    }

    func testEveryLoopIterationConsumesTheExecutedNodeBudget() async throws {
        let limits = WorkflowInterpreterLimits(maximumExecutedNodes: 3)
        let emptyProgram = try parse("async function workflow() { for (const item of []) {} }")
        try await WorkflowInterpreter(limits: limits, engine: WorkflowInterpreterTestSink())
            .execute(program: emptyProgram)

        let oneItemProgram = try parse("async function workflow() { for (const item of [1]) {} }")
        do {
            try await WorkflowInterpreter(limits: limits, engine: WorkflowInterpreterTestSink())
                .execute(program: oneItemProgram)
            XCTFail("The loop body iteration must consume a run budget node")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .resourceLimit)
        }
    }

    func testFacadeOperationBudgetStopsBeforeDispatchOverrun() async throws {
        let program = try parse("async function workflow() { await report(1); await report(2); }")
        let sink = WorkflowInterpreterTestSink()
        let interpreter = WorkflowInterpreter(
            limits: WorkflowInterpreterLimits(maximumFacadeOperations: 1),
            engine: sink)

        do {
            try await interpreter.execute(program: program)
            XCTFail("The second facade operation must exceed the configured budget")
        } catch let error as WorkflowError {
            XCTAssertEqual(error.kind, .resourceLimit)
        }
        let requests = await sink.requests()
        XCTAssertEqual(requests.map(\.operation), [.report(value: .number(1))])
    }

    /// Builds a checked program while ignoring ONLY the scope diagnostics, so
    /// the interpreter's runtime refusals stay covered.
    private func parseBypassingScopeChecks(_ source: String) throws -> WorkflowCheckedProgram {
        let scopeRules: Set<WorkflowScriptDiagnosticRule> = [
            .undeclaredIdentifier, .duplicateBinding, .reservedBindingName,
        ]
        let result = WorkflowScriptChecker.parse(source)
        XCTAssertTrue(result.diagnostics.allSatisfy { scopeRules.contains($0.rule) })
        let ast = try XCTUnwrap(result.ast)
        let manifest = try XCTUnwrap(WorkflowScriptManifestCompiler.build(ast).manifest)
        return WorkflowCheckedProgram(
            ast: ast,
            descriptor: WorkflowRunDescriptor(
                id: UUID(), name: "Test workflow", source: source, sourceHash: "test",
                facadeVersion: ast.facadeVersion, args: WorkflowFrozenArguments([:]),
                manifest: manifest))
    }

    private func parse(
        _ source: String,
        args: [String: String] = [:],
        file: StaticString = #filePath,
        line: UInt = #line
    ) throws -> WorkflowCheckedProgram {
        let result = WorkflowScriptChecker.check(source: source, name: "Test workflow", args: args)
        XCTAssertTrue(result.isValid, result.diagnostics.map(\.message).joined(separator: "\n"), file: file, line: line)
        return try XCTUnwrap(result.checked, file: file, line: line)
    }
}

private struct WorkflowInterpreterRecordedRequest: Equatable, Sendable {
    let operation: WorkflowInterpreterOperation
    let site: WorkflowSiteKey?
}

private actor WorkflowInterpreterTestSink: WorkflowCommandSink {
    enum Resolution: Sendable {
        case immediate(WorkflowCanonicalValue)
        case held
        case waitForCancellation
    }

    private var resolutions: [Resolution]
    private var recordedRequests: [WorkflowInterpreterRecordedRequest] = []
    private var requestWaiters: [(Int, CheckedContinuation<Void, Never>)] = []
    private var heldResolution: CheckedContinuation<WorkflowCanonicalValue, Error>?

    init(resolutions: [Resolution] = []) {
        self.resolutions = resolutions
    }

    func perform(
        _ operation: WorkflowInterpreterOperation,
        at site: WorkflowSiteKey?,
        context: WorkflowAttemptContext
    ) async throws -> WorkflowCanonicalValue {
        recordedRequests.append(WorkflowInterpreterRecordedRequest(operation: operation, site: site))
        resumeSatisfiedWaiters()

        let resolution = resolutions.isEmpty ? .immediate(.null) : resolutions.removeFirst()
        switch resolution {
        case .immediate(let value):
            return value
        case .held:
            return try await withCheckedThrowingContinuation { continuation in
                heldResolution = continuation
            }
        case .waitForCancellation:
            _ = await context.cancellation.waitUntilCancelled()
            throw CancellationError()
        }
    }

    func resolveHeld(with value: WorkflowCanonicalValue) {
        heldResolution?.resume(returning: value)
        heldResolution = nil
    }

    func waitForRequestCount(_ count: Int) async {
        guard recordedRequests.count < count else { return }
        await withCheckedContinuation { continuation in
            requestWaiters.append((count, continuation))
        }
    }

    func requestCount() -> Int {
        recordedRequests.count
    }

    func requests() -> [WorkflowInterpreterRecordedRequest] {
        recordedRequests
    }

    private func resumeSatisfiedWaiters() {
        let ready = requestWaiters.filter { recordedRequests.count >= $0.0 }
        requestWaiters.removeAll { recordedRequests.count >= $0.0 }
        for (_, continuation) in ready {
            continuation.resume()
        }
    }
}

private actor WorkflowInterpreterCompletionProbe {
    private var completed = false
    private var waiter: CheckedContinuation<Bool, Never>?

    func finish() {
        completed = true
        waiter?.resume(returning: true)
        waiter = nil
    }

    func waitForCompletion(within duration: Duration) async -> Bool {
        if completed { return true }
        return await withCheckedContinuation { continuation in
            waiter = continuation
            Task {
                try? await Task.sleep(for: duration)
                self.timeout()
            }
        }
    }

    private func timeout() {
        waiter?.resume(returning: completed)
        waiter = nil
    }
}
