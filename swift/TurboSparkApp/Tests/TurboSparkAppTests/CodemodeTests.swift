import XCTest
@testable import TurboSparkApp

/// Codemode sandbox tests: unit coverage for the pure pieces (identifier
/// sanitizing, @options parsing, payload capping, result formatting, the
/// per-chat store) and process-level coverage through the real packaged
/// executable for the wire protocol, VM surface, caps, deadline kills, and
/// settlement semantics. The nested-call handler is injected, so process
/// tests exercise the full sandbox without MCP servers.
final class CodemodeTests: XCTestCase {
    // MARK: Executable helper (same contract as REPLWorkerProcessTests)

    private func makeExecutableURL() throws -> URL {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        return try XCTUnwrap(
            [
                ".build/out/Products/Debug/TurboSparkApp",
                ".build/out/Products/Release/TurboSparkApp",
                ".build/arm64-apple-macosx/debug/TurboSparkApp",
                ".build/arm64-apple-macosx/release/TurboSparkApp",
                ".build/debug/TurboSparkApp",
                ".build/release/TurboSparkApp"
            ]
            .map { packageRoot.appendingPathComponent($0, isDirectory: false) }
            .first { FileManager.default.isExecutableFile(atPath: $0.path) },
            "The TurboSparkApp executable should be available in the SwiftPM build products")
    }

    private func makeEntry(
        _ name: String = "mcp__demo__echo", jsName: String? = nil
    ) -> CodemodeToolEntry {
        CodemodeToolEntry(
            name: name,
            jsName: jsName ?? CodemodeIdentifier.sanitize(name),
            description: "echoes its input")
    }

    private func echoHandler() -> CodemodeWorkerSupervisor.CallHandler {
        { name, arguments in
            .success(name + " saw " + (arguments ?? "no arguments"))
        }
    }

    private func runScript(
        _ code: String,
        entries: [CodemodeToolEntry],
        handler: @escaping CodemodeWorkerSupervisor.CallHandler,
        limits: CodemodeLimits = CodemodeLimits(),
        timeout: TimeInterval? = nil
    ) async throws -> CodemodeResult {
        var effectiveLimits = limits
        if let timeout {
            effectiveLimits.defaultTimeout = timeout
        }
        return await CodemodeSandbox.run(
            code: code,
            entries: entries,
            storeKey: "test-" + UUID().uuidString,
            callHandler: handler,
            limits: effectiveLimits,
            executableURL: try makeExecutableURL())
    }

    // MARK: Identifier sanitizing

    func testSanitizeKeepsValidIdentifiersUntouched() {
        XCTAssertEqual(CodemodeIdentifier.sanitize("mcp__demo__echo"), "mcp__demo__echo")
        XCTAssertEqual(CodemodeIdentifier.sanitize("a1_$ok"), "a1_$ok")
    }

    func testSanitizeReplacesInvalidCharactersAndGuardsShapes() {
        XCTAssertEqual(CodemodeIdentifier.sanitize("my-tool.name"), "my_tool_name")
        XCTAssertEqual(CodemodeIdentifier.sanitize("2fa"), "_2fa")
        XCTAssertEqual(CodemodeIdentifier.sanitize("class"), "class_")
        XCTAssertEqual(CodemodeIdentifier.sanitize(""), "_")
    }

    func testEntriesDedupeCollisionsFirstWins() {
        let descriptors = [
            DeferredToolDescriptor(
                name: "mcp__a__tool", serverName: "a", toolName: "tool",
                description: "first", inputSchemaJSON: "{}"),
            DeferredToolDescriptor(
                name: "mcp__a__tool", serverName: "a", toolName: "tool",
                description: "duplicate", inputSchemaJSON: "{}"),
        ]
        let entries = CodemodeIdentifier.entries(for: descriptors)
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.description, "first")
    }

    // MARK: @options source parsing

    func testParseWithoutOptionsReturnsCodeUnchanged() throws {
        let parsed = try CodemodeSandbox.ParsedSource.parse(
            "return 1;", limits: CodemodeLimits())
        XCTAssertEqual(parsed.code, "return 1;")
        XCTAssertNil(parsed.timeoutOverride)
        XCTAssertNil(parsed.maxOutputCharsOverride)
    }

    func testParseOptionsLineStripsItselfAndAppliesBounds() throws {
        let parsed = try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"timeout_ms\": 5000, \"max_output_chars\": 100}\nreturn 1;",
            limits: CodemodeLimits())
        XCTAssertFalse(parsed.code.contains("@options"))
        XCTAssertTrue(parsed.code.hasPrefix("\n"), "the options line is replaced by an empty line")
        XCTAssertEqual(parsed.timeoutOverride, 5)
        XCTAssertEqual(parsed.maxOutputCharsOverride, 100)
    }

    func testParseOptionsClampToTheMaximumTimeout() throws {
        let parsed = try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"timeout_ms\": 99999999}\nreturn 1;",
            limits: CodemodeLimits())
        XCTAssertEqual(parsed.timeoutOverride, CodemodeLimits().maximumTimeout)
    }

    func testParseRejectsUnknownFieldsAndMalformedJSON() {
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"wat\": 1}\nreturn 1;", limits: CodemodeLimits()))
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "// @options: not json\nreturn 1;", limits: CodemodeLimits()))
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "// @options: {}", limits: CodemodeLimits()),
            "options without any script code is a source error")
    }

    // MARK: Payload capping and result formatting

    func testTruncatedPayloadKeepsHeadAndTailUnderTheCap() {
        let text = String(repeating: "a", count: 1_000)
        let capped = CodemodeWorkerSupervisor.truncatedPayload(text, cap: 200)
        XCTAssertTrue(capped.contains("characters truncated"))
        XCTAssertLessThanOrEqual(capped.count, 300)
        XCTAssertEqual(CodemodeWorkerSupervisor.truncatedPayload("short", cap: 200), "short")
    }

    func testPromptOutputJoinsItemsThenTheReturnValue() {
        let ok = CodemodeResult(
            ok: true,
            valueJSON: "\"done\"",
            output: [
                CodemodeOutputItem(kind: .console, text: "log line", level: "log"),
                CodemodeOutputItem(kind: .text, text: "text line", level: nil)
            ],
            calls: [],
            error: nil,
            storeWrites: [])
        let formatted = CodemodeSandbox.promptOutput(for: ok)
        XCTAssertFalse(formatted.isError)
        XCTAssertTrue(formatted.output.contains("log line"))
        XCTAssertTrue(formatted.output.contains("text line"))
        XCTAssertTrue(formatted.output.contains("Return value: \"done\""))
    }

    func testPromptOutputReportsFailureWithStackAndKeepsPartialOutput() {
        let failed = CodemodeResult(
            ok: false,
            valueJSON: nil,
            output: [CodemodeOutputItem(kind: .text, text: "partial", level: nil)],
            calls: [],
            error: CodemodeError(
                kind: .script, name: "TypeError", message: "x is not defined",
                stack: "at codemode.js:3:5"),
            storeWrites: [])
        let formatted = CodemodeSandbox.promptOutput(for: failed)
        XCTAssertTrue(formatted.isError)
        XCTAssertTrue(formatted.output.hasPrefix("partial"))
        XCTAssertTrue(formatted.output.contains("TypeError"))
        XCTAssertTrue(formatted.output.contains("x is not defined"))
        XCTAssertTrue(formatted.output.contains("codemode.js:3:5"))
    }

    func testArchivalCallsCarryTelemetryOnly() {
        let result = CodemodeResult(
            ok: true, valueJSON: nil, output: [],
            calls: [CodemodeCallRecord(name: "mcp__a__b", status: .ok, durationMs: 12)],
            error: nil, storeWrites: [])
        let archival = CodemodeSandbox.archivalCalls(for: result)
        XCTAssertEqual(archival, "{\"calls\":[{\"duration_ms\":12,\"name\":\"mcp__a__b\",\"status\":\"ok\"}]}")
        XCTAssertNil(CodemodeSandbox.archivalCalls(for: CodemodeResult(
            ok: true, valueJSON: nil, output: [], calls: [], error: nil, storeWrites: [])))
    }

    // MARK: Per-chat store

    func testStoreBoxAppliesWritesAndDeletes() async {
        let box = CodemodeSandbox.StoreBox()
        let key = "chat"
        await box.apply(
            [CodemodeStoreWrite(key: "a", valueJSON: "{\"n\":1}")], forKey: key)
        let snapshot = await box.snapshot(for: key)
        XCTAssertEqual(snapshot["a"], "{\"n\":1}")
        await box.apply([CodemodeStoreWrite(key: "a", valueJSON: nil)], forKey: key)
        let afterDelete = await box.snapshot(for: key)
        XCTAssertNil(afterDelete["a"])
    }

    // MARK: Process-level sandbox runs

    func testReturnAndTopLevelAwaitWork() async throws {
        let result = try await runScript("return 40 + 2;", entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "42")

        let awaited = try await runScript(
            #"const v = await Promise.resolve("hi"); return v + "!";"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(awaited.ok, awaited.error?.message ?? "")
        XCTAssertEqual(awaited.valueJSON, "\"hi!\"")
    }

    func testToolCallRoundTripCarriesNameArgumentsAndJSONTextResult() async throws {
        final class Recorder: @unchecked Sendable {
            var name: String?
            var arguments: String?
        }
        let recorder = Recorder()
        let handler: CodemodeWorkerSupervisor.CallHandler = { name, arguments in
            recorder.name = name
            recorder.arguments = arguments
            return .success("echo result text")
        }
        let result = try await runScript(
            #"const r = await tools.mcp__demo__echo({msg: "abc"}); return r;"#,
            entries: [makeEntry()], handler: handler)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(recorder.name, "mcp__demo__echo")
        XCTAssertEqual(recorder.arguments, #"{"msg":"abc"}"#)
        XCTAssertEqual(result.valueJSON, "\"echo result text\"",
                       "the tool's text output crosses JSON-encoded and resolves as a string")
        XCTAssertEqual(result.calls.first?.status, .ok)
        XCTAssertEqual(result.calls.first?.name, "mcp__demo__echo")
    }

    func testBracketAccessByAdvertisedNameAlsoWorks() async throws {
        let result = try await runScript(
            #"const r = await tools["mcp__demo__echo"]({}); return r;"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
    }

    func testParallelCallsAllSettle() async throws {
        let result = try await runScript(
            #"const [a, b] = await Promise.all([tools.mcp__demo__echo({i: 1}), tools.mcp__demo__echo({i: 2})]); return a + "|" + b;"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.calls.filter { $0.status == .ok }.count, 2)
        XCTAssertTrue((result.valueJSON ?? "").contains("mcp__demo__echo saw"))
    }

    func testFailedNestedCallRejectsInsideTheScript() async throws {
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            .failure(CodemodeCallError(message: "permission denied by the engine"))
        }
        let result = try await runScript(
            #"try { await tools.mcp__demo__echo({}); return "not reached"; } catch (e) { return "caught: " + e.message; }"#,
            entries: [makeEntry()], handler: handler)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertTrue((result.valueJSON ?? "").contains("permission denied by the engine"))
        XCTAssertEqual(result.calls.first?.status, .error)
    }

    func testUnknownToolThrowsWithSuggestions() async throws {
        let result = try await runScript(
            #"try { tools.nope(); return "no"; } catch (e) { return e.message; }"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok)
        let message = result.valueJSON ?? ""
        XCTAssertTrue(message.contains("not an available tool"), message)
        XCTAssertTrue(message.contains("ALL_TOOLS"), message)
    }

    func testAllToolsListsBindings() async throws {
        let result = try await runScript(
            #"return ALL_TOOLS.length + ":" + ALL_TOOLS[0].name;"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "\"1:mcp__demo__echo\"")
    }

    func testOutputItemsKeepOrderAcrossConsoleAndText() async throws {
        let result = try await runScript(
            #"console.log("first"); text("second"); console.warn("third"); return "end";"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.output.map(\.text), ["first", "second", "third"])
        XCTAssertEqual(result.output.first?.kind, .console)
        XCTAssertEqual(result.output.first?.level, "log")
        XCTAssertEqual(result.output[1].kind, .text)
        XCTAssertEqual(result.output.last?.level, "warn")
    }

    func testOutputCapBreachFailsEvenWhenTheErrorIsCaught() async throws {
        var limits = CodemodeLimits()
        limits.maximumOutputCharacters = 200
        let plain = try await runScript(
            #"for (let i = 0; i < 40; i++) text("x".repeat(50)); return "never";"#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertFalse(plain.ok)
        XCTAssertEqual(plain.error?.name, "RangeError")

        let caught = try await runScript(
            #"try { for (let i = 0; i < 40; i++) text("x".repeat(50)); } catch (e) {} text("after catch"); return "resumed";"#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertFalse(caught.ok,
                       "catching the cap breach must not resume output or flip the result to ok")
        XCTAssertEqual(caught.error?.name, "RangeError")
    }

    func testStoreAndLoadRoundTripAndReportWrites() async throws {
        let result = try await runScript(
            #"store("k", {n: 1}); const v = load("k"); store("gone", 5); store("gone", undefined); return v.n;"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "1")
        XCTAssertEqual(result.storeWrites.count, 2, "the set and the delete are reported")
        XCTAssertTrue(result.storeWrites.contains(CodemodeStoreWrite(key: "k", valueJSON: "{\"n\":1}")))
        XCTAssertTrue(result.storeWrites.contains(CodemodeStoreWrite(key: "gone", valueJSON: nil)))
    }

    func testStoreValueCapThrows() async throws {
        var limits = CodemodeLimits()
        limits.maximumStoreValueCharacters = 16
        let result = try await runScript(
            #"try { store("k", "0123456789012345678901234567890"); return "stored"; } catch (e) { return e.name; }"#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertTrue(result.ok)
        XCTAssertEqual(result.valueJSON, "\"RangeError\"")
    }

    func testStalledPromiseFailsFast() async throws {
        let started = Date()
        let result = try await runScript(
            #"await new Promise(() => {});"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertTrue(
            (result.error?.message ?? "").contains("can never settle"),
            result.error?.message ?? "")
        XCTAssertLessThan(Date().timeIntervalSince(started), 30,
                          "the stall must fail fast, not burn the deadline")
    }

    func testExitKeepsOutputAndStoreWritesAndSkipsTheReturnValue() async throws {
        let result = try await runScript(
            #"text("before"); store("kept", true); exit(); return "after";"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(result.ok)
        XCTAssertTrue(result.output.contains(where: { $0.text == "before" }))
        XCTAssertTrue(result.storeWrites.contains(CodemodeStoreWrite(key: "kept", valueJSON: "true")))
        XCTAssertNil(result.valueJSON)
    }

    func testUnawaitedCallIsReportedCancelled() async throws {
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            try? await Task.sleep(nanoseconds: 400_000_000)
            return .success("late")
        }
        let result = try await runScript(
            #"tools.mcp__demo__echo({}); return "quick";"#,
            entries: [makeEntry()], handler: handler)
        XCTAssertTrue(result.ok)
        XCTAssertEqual(result.calls.filter { $0.status == .cancelled }.count, 1,
                       "the in-flight call at script end is reported cancelled")
    }

    func testSpinningScriptIsKilledAtTheDeadline() async throws {
        let started = Date()
        let result = try await runScript(
            #"text("started"); while (true) {}"#,
            entries: [makeEntry()], handler: echoHandler(),
            limits: CodemodeLimits(defaultTimeout: 3))
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .timeout)
        XCTAssertTrue(result.output.contains(where: { $0.text == "started" }),
                      "partial output survives the kill")
        XCTAssertLessThan(Date().timeIntervalSince(started), 15)
    }

    func testCancelledRunIsReportedAborted() async throws {
        let task = Task {
            try await runScript(
                "while (true) {}",
                entries: [makeEntry()],
                handler: echoHandler(),
                limits: CodemodeLimits(defaultTimeout: 60))
        }
        try await Task.sleep(nanoseconds: 500_000_000)
        task.cancel()
        let result = try await task.value
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .aborted)
    }

    func testSyntaxErrorIsAScriptFailure() async throws {
        let result = try await runScript(
            "return (;", entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .script)
        XCTAssertNotNil(result.error?.message)
    }

    func testOversizedScriptIsRejectedLocallyWithoutSpawning() async throws {
        var limits = CodemodeLimits()
        limits.maximumScriptBytes = 32
        let executableURL = try makeExecutableURL()
        let result = await CodemodeSandbox.run(
            code: String(repeating: "1", count: 200),
            entries: [makeEntry()],
            storeKey: "test-oversize",
            callHandler: echoHandler(),
            limits: limits,
            executableURL: executableURL)
        XCTAssertFalse(result.ok)
        XCTAssertTrue((result.error?.message ?? "").contains("byte limit"))
    }

    func testEmptyEntriesFailBeforeSpawning() async throws {
        let result = try await runScript(
            "return 1;", entries: [], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertTrue((result.error?.message ?? "").contains("no MCP tools"))
    }
}
