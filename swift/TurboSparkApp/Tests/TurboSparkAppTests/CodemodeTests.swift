import XCTest
@testable import TurboSparkApp

/// Thread-safe record of what an injected call handler observed. Handlers run
/// on detached tasks, so every field is lock guarded.
private final class CallLog: @unchecked Sendable {
    private let lock = NSLock()
    private var invocationCount = 0
    private var currentlyRunning = 0
    private var peakRunning = 0
    private var didStart = false
    private var didCancel = false

    /// Marks a handler entering; tracks the concurrency high-water mark.
    func begin() {
        lock.lock()
        defer { lock.unlock() }
        invocationCount += 1
        currentlyRunning += 1
        peakRunning = max(peakRunning, currentlyRunning)
        didStart = true
    }

    func end() {
        lock.lock()
        defer { lock.unlock() }
        currentlyRunning -= 1
    }

    func markCancelled() {
        lock.lock()
        defer { lock.unlock() }
        didCancel = true
    }

    var invocations: Int { lock.withLock { invocationCount } }
    var peakConcurrency: Int { lock.withLock { peakRunning } }
    var started: Bool { lock.withLock { didStart } }
    var cancelled: Bool { lock.withLock { didCancel } }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}

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

    private func descriptor(
        _ name: String, description: String = "a tool", schema: String = "{}"
    ) -> DeferredToolDescriptor {
        let parts = name.components(separatedBy: "__")
        return DeferredToolDescriptor(
            name: name,
            serverName: parts.count > 1 ? parts[1] : "srv",
            toolName: parts.count > 2 ? parts[2...].joined(separator: "__") : name,
            description: description,
            inputSchemaJSON: schema)
    }

    func testEntriesDropRepeatedAdvertisedNames() {
        // The same advertised name twice (even with different case) is one
        // tool; the first description wins.
        let entries = CodemodeIdentifier.entries(for: [
            descriptor("mcp__a__tool", description: "first"),
            descriptor("MCP__A__TOOL", description: "duplicate"),
        ])
        XCTAssertEqual(entries.count, 1)
        XCTAssertEqual(entries.first?.description, "first")
    }

    func testCollidingIdentifiersAreDisambiguatedNotDropped() {
        // Two DIFFERENT names that sanitize to one identifier. The second used
        // to vanish, reachable by neither spelling.
        let entries = CodemodeIdentifier.entries(for: [
            descriptor("mcp__a-b__c"),
            descriptor("mcp__a_b__c"),
            descriptor("mcp__a.b__c"),
        ])
        XCTAssertEqual(entries.map(\.name), ["mcp__a-b__c", "mcp__a_b__c", "mcp__a.b__c"])
        XCTAssertEqual(entries.map(\.jsName), ["mcp__a_b__c", "mcp__a_b__c_2", "mcp__a_b__c_3"])
        XCTAssertEqual(Set(entries.map(\.jsName)).count, 3, "every tool has its own binding")
    }

    func testCollidingToolsAreEachCallableInAScript() async throws {
        let entries = CodemodeIdentifier.entries(for: [
            descriptor("mcp__a-b__c"), descriptor("mcp__a_b__c"),
        ])
        let result = try await runScript(
            #"return (await tools.mcp__a_b__c({})) + "|" + (await tools.mcp__a_b__c_2({}));"#,
            entries: entries, handler: echoHandler())
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(
            result.valueJSON, "\"mcp__a-b__c saw {}|mcp__a_b__c saw {}\"")
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

    func testOptionsLineAfterLeadingBlankLinesIsHonoredAndKeepsLineNumbers() throws {
        // A model often starts with a blank line; silently ignoring the
        // options line then dropped the timeout it asked for.
        let raw = "\n\n// @options: {\"timeout_ms\": 5000}\nreturn 1;"
        let parsed = try CodemodeSandbox.ParsedSource.parse(raw, limits: CodemodeLimits())
        XCTAssertEqual(parsed.timeoutOverride, 5)
        XCTAssertFalse(parsed.code.contains("@options"))
        XCTAssertEqual(
            parsed.code.components(separatedBy: "\n").count,
            raw.components(separatedBy: "\n").count,
            "the options line is blanked, not removed, so line numbers do not move")
        XCTAssertTrue(parsed.code.hasSuffix("return 1;"))
    }

    func testOptionsLineAfterOtherCodeIsLeftAlone() throws {
        let raw = "const a = 1;\n// @options: {\"timeout_ms\": 5000}\nreturn a;"
        let parsed = try CodemodeSandbox.ParsedSource.parse(raw, limits: CodemodeLimits())
        XCTAssertNil(parsed.timeoutOverride, "only the first non-blank line is an options line")
        XCTAssertEqual(parsed.code, raw)
    }

    func testTimeoutHasAFloorAndOptionsRejectBooleans() throws {
        let tiny = try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"timeout_ms\": 1}\nreturn 1;", limits: CodemodeLimits())
        XCTAssertEqual(tiny.timeoutOverride, CodemodeSandbox.ParsedSource.minimumTimeout,
                       "a worker cannot start in a millisecond, so a smaller ask could only time out")
        // JSON `true` bridges to NSNumber 1; it is not a number the author meant.
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"timeout_ms\": true}\nreturn 1;", limits: CodemodeLimits()))
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "// @options: {\"max_output_chars\": true}\nreturn 1;", limits: CodemodeLimits()))
        XCTAssertThrowsError(try CodemodeSandbox.ParsedSource.parse(
            "   \n \n", limits: CodemodeLimits()), "an all-blank script is empty")
    }

    // MARK: Payload capping and result formatting

    func testResultLimitIsMeasuredInUTF8BytesAndNeverTruncates() {
        XCTAssertFalse(CodemodeWorkerSupervisor.resultExceedsLimit("short", cap: 200))
        XCTAssertFalse(CodemodeWorkerSupervisor.resultExceedsLimit(String(repeating: "a", count: 200), cap: 200))
        XCTAssertTrue(CodemodeWorkerSupervisor.resultExceedsLimit(String(repeating: "a", count: 201), cap: 200))
        // Four bytes per scalar: 60 emoji are 60 characters but 240 bytes.
        XCTAssertTrue(CodemodeWorkerSupervisor.resultExceedsLimit(String(repeating: "\u{1F600}", count: 60), cap: 200))
    }

    func testFirstScriptLineReadsTheFirstScriptFrame() {
        XCTAssertEqual(
            CodemodeWorkerSupervisor.firstScriptLine(in: "f@codemode.js:12:5\n@codemode.js:30:1"), 12)
        XCTAssertNil(CodemodeWorkerSupervisor.firstScriptLine(in: "f@\nglobal code@"))
        XCTAssertNil(CodemodeWorkerSupervisor.firstScriptLine(in: "@other.js:3:4"))
    }

    func testPromptOutputNamesTheFailingLine() {
        let failed = CodemodeResult(
            ok: false, valueJSON: nil, output: [], calls: [],
            error: CodemodeError(
                kind: .script, name: "TypeError", message: "x is not defined",
                stack: nil, line: 7),
            storeWrites: [])
        let formatted = CodemodeSandbox.promptOutput(for: failed)
        XCTAssertTrue(formatted.isError)
        XCTAssertTrue(formatted.output.contains("(line 7)"), formatted.output)
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

    func testStoreBoxKeyOverflowKeepsWhatTheScriptJustWritten() async {
        let box = CodemodeSandbox.StoreBox()
        let chat = "chat"
        await box.apply(
            (0..<250).map { CodemodeStoreWrite(key: String(format: "old-%03d", $0), valueJSON: "1") },
            forKey: chat)
        let fresh = (0..<20).map { CodemodeStoreWrite(key: "new-\($0)", valueJSON: "2") }
        await box.apply(fresh, forKey: chat)
        let snapshot = await box.snapshot(for: chat)
        XCTAssertEqual(snapshot.count, CodemodeSandbox.StoreBox.maximumKeys)
        for write in fresh {
            XCTAssertNotNil(snapshot[write.key], "\(write.key) was just written and must survive")
        }
        // Deterministic: the untouched keys that go are the first by name.
        XCTAssertNil(snapshot["old-000"])
        XCTAssertNotNil(snapshot["old-249"])
    }

    func testStoreBoxDropsTheLeastRecentlyUsedChat() async {
        let box = CodemodeSandbox.StoreBox()
        let write = [CodemodeStoreWrite(key: "k", valueJSON: "1")]
        for index in 0..<CodemodeSandbox.StoreBox.maximumChats {
            await box.apply(write, forKey: "chat-\(index)")
        }
        // Reading chat-0 makes it the most recently used.
        let touched = await box.snapshot(for: "chat-0")
        XCTAssertEqual(touched["k"], "1")
        await box.apply(write, forKey: "overflow")
        let evicted = await box.snapshot(for: "chat-1")
        XCTAssertTrue(evicted.isEmpty, "chat-1 was the least recently used")
        let kept = await box.snapshot(for: "chat-0")
        XCTAssertEqual(kept["k"], "1", "a recently read chat is kept")
        let newest = await box.snapshot(for: "overflow")
        XCTAssertEqual(newest["k"], "1")
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

    // MARK: Call admission (count and concurrency caps)

    func testCallCountIsCappedAndConcurrencyIsBounded() async throws {
        let log = CallLog()
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            log.begin()
            // Long enough that admitted calls overlap and the slot limit,
            // not call speed, is what bounds concurrency.
            try? await Task.sleep(nanoseconds: 60_000_000)
            log.end()
            return .success("ok")
        }
        var limits = CodemodeLimits()
        limits.maximumCalls = 10
        limits.maximumConcurrentCalls = 3
        let result = try await runScript(
            #"""
            const settled = await Promise.allSettled(
              Array.from({ length: 30 }, (_, i) => tools.mcp__demo__echo({ i })));
            const rejected = settled.filter((s) => s.status === "rejected");
            const message = rejected.length > 0 ? rejected[0].reason.message : "";
            return settled.length - rejected.length + ":" + rejected.length + ":" + message;
            """#,
            entries: [makeEntry()], handler: handler, limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(log.invocations, 10, "calls past the cap must never reach the handler")
        XCTAssertLessThanOrEqual(log.peakConcurrency, 3, "in-flight calls must respect the slot limit")
        XCTAssertGreaterThanOrEqual(log.peakConcurrency, 2, "admitted calls still run concurrently")
        let value = result.valueJSON ?? ""
        XCTAssertTrue(value.hasPrefix("\"10:20:"), "ten ran, twenty were rejected: \(value)")
        XCTAssertTrue(value.contains("at most 10 tool calls"), value)
        XCTAssertEqual(result.calls.filter { $0.status == .ok }.count, 10)
    }

    func testQueuedCallsWaitForASlotInsteadOfFailing() async throws {
        // Fewer calls than the cap but more than the slots: all must succeed.
        var limits = CodemodeLimits()
        limits.maximumConcurrentCalls = 2
        let result = try await runScript(
            #"""
            const out = await Promise.all(
              Array.from({ length: 12 }, (_, i) => tools.mcp__demo__echo({ i })));
            return out.length;
            """#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "12")
    }

    // MARK: Settled scripts cannot make tool calls

    func testCallsAfterExitNeverReachTheHandler() async throws {
        let log = CallLog()
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            log.begin()
            log.end()
            return .success("ran")
        }
        // exit() settles the result but throws a sentinel; a script that
        // catches it keeps running. None of its later calls may run a tool.
        let result = try await runScript(
            #"""
            try { exit(); } catch (e) {}
            for (let i = 0; i < 50; i++) {
              try { tools.mcp__demo__echo({ i }); } catch (e) {}
            }
            try { await tools.mcp__demo__echo({ late: true }); } catch (e) {}
            """#,
            entries: [makeEntry()], handler: handler)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(log.invocations, 0, "no tool may run after the script has finished")
        XCTAssertTrue(result.calls.isEmpty, "\(result.calls)")
    }

    func testCallsAfterACaughtOutputCapBreachNeverReachTheHandler() async throws {
        let log = CallLog()
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            log.begin()
            log.end()
            return .success("ran")
        }
        var limits = CodemodeLimits()
        limits.maximumOutputCharacters = 100
        let result = try await runScript(
            #"""
            try { for (let i = 0; i < 40; i++) text("x".repeat(50)); } catch (e) {}
            for (let i = 0; i < 20; i++) {
              try { tools.mcp__demo__echo({ i }); } catch (e) {}
            }
            """#,
            entries: [makeEntry()], handler: handler, limits: limits)
        XCTAssertFalse(result.ok, "the breach is the result, however it is caught")
        XCTAssertEqual(log.invocations, 0)
    }

    // MARK: Abandoned calls are cancelled

    func testAbandonedCallIsCancelledWhenTheScriptEnds() async throws {
        let log = CallLog()
        let slow = makeEntry("mcp__demo__slow")
        let handler: CodemodeWorkerSupervisor.CallHandler = { name, _ in
            if name == "mcp__demo__slow" {
                return await withTaskCancellationHandler {
                    log.begin()
                    try? await Task.sleep(nanoseconds: 20_000_000_000)
                    return Result<String, CodemodeCallError>.success("late")
                } onCancel: {
                    log.markCancelled()
                }
            }
            // The awaited call returns only after the slow one is running, so
            // the script cannot end before the slow call has started.
            for _ in 0..<200 where !log.started {
                try? await Task.sleep(nanoseconds: 10_000_000)
            }
            return .success("quick")
        }
        let result = try await runScript(
            #"""
            tools.mcp__demo__slow({});
            await tools.mcp__demo__echo({});
            return "done";
            """#,
            entries: [makeEntry(), slow], handler: handler)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.calls.filter { $0.status == .cancelled }.count, 1)
        for _ in 0..<200 where !log.cancelled {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertTrue(log.cancelled,
                      "the host call must actually be cancelled, not just reported cancelled")
    }

    // MARK: Return value shares the output budget

    func testReturnValueCannotBypassTheOutputBudget() async throws {
        let oversized = try await runScript(
            #"return "x".repeat(40000);"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(oversized.ok)
        XCTAssertEqual(oversized.error?.name, "RangeError")
        XCTAssertNil(oversized.valueJSON)

        let combined = try await runScript(
            #"text("y".repeat(20000)); return "x".repeat(20000);"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(combined.ok, "text() and the return value draw on one budget")
        XCTAssertEqual(combined.error?.name, "RangeError")

        let fits = try await runScript(
            #"return "x".repeat(100);"#,
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertTrue(fits.ok, fits.error?.message ?? "")
        XCTAssertEqual(fits.valueJSON, "\"" + String(repeating: "x", count: 100) + "\"")
    }

    func testHostStopsAWorkerThatNeverEndsALine() async throws {
        // Streams well past the wire bound with no newline, then idles.
        let bytes = CodemodeWire.maximumLineBytes + 4 * 1_024 * 1_024
        let started = Date()
        let result = try await runFakeWorker(
            "head -c \(bytes) /dev/zero | tr '\\000' a; exec sleep 30",
            // If the line cap regresses this fails at the deadline instead of
            // waiting out the default two minutes.
            limits: CodemodeLimits(defaultTimeout: 45))
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .sandbox)
        XCTAssertTrue((result.error?.message ?? "").contains("byte limit"), result.error?.message ?? "")
        XCTAssertLessThan(Date().timeIntervalSince(started), 45)
    }

    // MARK: A finished worker's last line is never lost

    func testDoneLineWrittenJustBeforeWorkerExitIsHonored() async throws {
        // Writes its terminal line and exits immediately, like the real
        // worker does. The result must be the script's, not "the worker
        // exited before the script finished".
        let result = try await runFakeWorker(
            "printf '{\"type\":\"done\",\"ok\":true,\"value\":\"7\"}\\n'")
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "7", "`done.value` is the JSON text of the return value")
    }

    func testConcurrentSuccessfulScriptsAreNeverReportedAsWorkerExits() async throws {
        // Regression guard for the lost-tail race: the worker writes `done`
        // and exits at once, and a reader cleared on termination could miss
        // the line. It showed as roughly 1 run in 10 under CPU load, so a
        // burst of concurrent runs is what exposes a return of it.
        let executableURL = try makeExecutableURL()
        let entries = [makeEntry()]
        let handler = echoHandler()
        let failures = await withTaskGroup(of: [String].self) { group in
            for _ in 0..<8 {
                group.addTask {
                    var bad: [String] = []
                    for _ in 0..<8 {
                        let result = await CodemodeSandbox.run(
                            code: "return 1;",
                            entries: entries,
                            storeKey: "test-" + UUID().uuidString,
                            callHandler: handler,
                            executableURL: executableURL)
                        if !(result.ok && result.valueJSON == "1") {
                            bad.append(result.error?.message ?? "not ok")
                        }
                    }
                    return bad
                }
            }
            var all: [String] = []
            for await part in group { all += part }
            return all
        }
        XCTAssertTrue(failures.isEmpty, "\(failures.count) of 64 runs failed: \(failures.prefix(3))")
    }

    // MARK: Error attribution

    func testRuntimeErrorReportsTheScriptLine() async throws {
        let result = try await runScript(
            "const a = 1;\nconst b = 2;\nnull.boom;\nreturn a + b;",
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .script)
        XCTAssertEqual(result.error?.line, 3)
        XCTAssertTrue((result.error?.stack ?? "").contains("codemode.js:3"), result.error?.stack ?? "")
        let formatted = CodemodeSandbox.promptOutput(for: result)
        XCTAssertTrue(formatted.output.contains("(line 3)"), formatted.output)
    }

    func testErrorBuiltInsideThePreludePointsAtTheCallingScriptLine() async throws {
        // `tools.nope` throws from the prelude's Proxy; the model must see
        // its own line (2), never a prelude frame or a prelude line number.
        let result = try await runScript(
            "const a = 1;\ntools.nope();\nreturn a;",
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.line, 2)
        XCTAssertFalse((result.error?.stack ?? "").contains("codemode-prelude.js"),
                       result.error?.stack ?? "")
        XCTAssertFalse((result.error?.stack ?? "").contains("global code@"),
                       result.error?.stack ?? "")
    }

    func testSyntaxErrorReportsTheScriptLine() async throws {
        let result = try await runScript(
            "const a = 1;\nconst b = ;\nreturn a;",
            entries: [makeEntry()], handler: echoHandler())
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .script)
        XCTAssertEqual(result.error?.name, "SyntaxError")
        XCTAssertEqual(result.error?.line, 2)
        XCTAssertFalse((result.error?.message ?? "").hasPrefix("SyntaxError:"),
                       "the name is reported separately: \(result.error?.message ?? "")")
    }

    // MARK: Nested results are complete or rejected, never clipped

    func testLargeNestedResultReachesTheScriptIntact() async throws {
        // About 300 KB of JSON: far past the old 64 Ki cap, which clipped it
        // and made JSON.parse fail. Filtering a big result in-script is what
        // codemode is for.
        let items = (0..<5_000).map { "{\"id\":\($0),\"name\":\"item-\($0)-padding-padding\"}" }
        let json = "{\"items\":[" + items.joined(separator: ",") + "]}"
        XCTAssertGreaterThan(json.utf8.count, 200_000)
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in .success(json) }
        let result = try await runScript(
            #"const d = JSON.parse(await tools.mcp__demo__echo({})); return d.items.length + ":" + d.items[4999].id;"#,
            entries: [makeEntry()], handler: handler)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertEqual(result.valueJSON, "\"5000:4999\"")
    }

    // MARK: Store key cap and call-argument cap (inside the VM)

    func testStoreKeyCapThrowsAtTheWriteAndDeletingFreesASlot() async throws {
        var limits = CodemodeLimits()
        limits.maximumStoreKeys = 3
        let result = try await runScript(
            #"""
            store("a", 1); store("b", 2); store("c", 3);
            let message = "";
            try { store("d", 4); } catch (e) { message = e.name + ":" + e.message; }
            store("a", undefined);           // frees a slot
            store("d", 4);                   // now fits
            store("b", 20);                  // overwriting an existing key never counts as new
            return message;
            """#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        XCTAssertTrue((result.valueJSON ?? "").hasPrefix("\"RangeError:store is full"), result.valueJSON ?? "")
        XCTAssertTrue(result.storeWrites.contains(CodemodeStoreWrite(key: "d", valueJSON: "4")))
    }

    func testOversizedCallArgumentsAreRejectedBeforeTheyReachTheHost() async throws {
        let log = CallLog()
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            log.begin(); log.end(); return .success("ran")
        }
        var limits = CodemodeLimits()
        limits.maximumCallArgumentCharacters = 1_000
        let result = try await runScript(
            #"""
            try { await tools.mcp__demo__echo({ blob: "a".repeat(5000) }); return "sent"; }
            catch (e) { return e.name + ":" + e.message; }
            """#,
            entries: [makeEntry()], handler: handler, limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        let value = result.valueJSON ?? ""
        XCTAssertTrue(value.hasPrefix("\"RangeError:"), value)
        XCTAssertTrue(value.contains("character limit"), value)
        XCTAssertEqual(log.invocations, 0, "an oversized call must never reach the handler")
    }

    // MARK: Worker memory

    func testPhysicalFootprintReadsTheKernelAccounting() {
        let own = CodemodeWorkerSupervisor.physicalFootprint(of: getpid())
        XCTAssertNotNil(own)
        XCTAssertGreaterThan(own ?? 0, 1_000_000, "a running test process uses more than a megabyte")
        XCTAssertNil(CodemodeWorkerSupervisor.physicalFootprint(of: 0x7fff_fff0),
                     "a pid that does not exist has no footprint")
    }

    func testRunawayAllocationIsKilledByTheMemoryLimit() async throws {
        var limits = CodemodeLimits()
        limits.maximumWorkerMemoryBytes = 512 * 1_024 * 1_024
        limits.defaultTimeout = 60
        let started = Date()
        let result = try await runScript(
            #"const keep = []; while (true) { keep.push(new Array(1000000).fill(1)); }"#,
            entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .sandbox, result.error?.message ?? "")
        XCTAssertTrue((result.error?.message ?? "").contains("memory"), result.error?.message ?? "")
        XCTAssertLessThan(Date().timeIntervalSince(started), 45,
                          "the watchdog must end it long before the 60 s deadline")
    }

    func testOrdinaryScriptStaysFarBelowTheMemoryLimit() async throws {
        // The real worker is the whole app binary, so its baseline footprint is
        // not obviously small. This pins that it is well under the default
        // limit's headroom: a trivial run must pass with only 512 MiB.
        var limits = CodemodeLimits()
        limits.maximumWorkerMemoryBytes = 512 * 1_024 * 1_024
        let result = try await runScript(
            "return 1;", entries: [makeEntry()], handler: echoHandler(), limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
    }

    // MARK: Worker exit diagnostics and environment

    /// Runs `shellBody` as the worker via `/bin/sh -c`. Deliberately NOT a
    /// script file written for the test: macOS holds the first exec of a new
    /// file while it assesses it, and with several builds running that took
    /// minutes and made these tests time out. `/bin/sh` is a system binary.
    private func runFakeWorker(
        _ shellBody: String, limits: CodemodeLimits = CodemodeLimits(defaultTimeout: 30)
    ) async throws -> CodemodeResult {
        await CodemodeSandbox.run(
            code: "return 1;",
            entries: [makeEntry()],
            storeKey: nil,
            callHandler: echoHandler(),
            limits: limits,
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            workerArguments: ["-c", shellBody])
    }

    func testWorkerThatDiesReportsItsExitStatusAndStderr() async throws {
        let result = try await runFakeWorker("echo 'fatal: JSC heap corrupted' >&2\nexit 3")
        XCTAssertFalse(result.ok)
        XCTAssertEqual(result.error?.kind, .sandbox)
        let message = result.error?.message ?? ""
        XCTAssertTrue(message.contains("exit status 3"), message)
        XCTAssertTrue(message.contains("fatal: JSC heap corrupted"), message)
    }

    func testWorkerKilledBySignalReportsTheSignal() async throws {
        let result = try await runFakeWorker("kill -9 $$")
        XCTAssertFalse(result.ok)
        let message = result.error?.message ?? ""
        XCTAssertTrue(message.contains("terminated by signal 9"), message)
    }

    func testWorkerDoesNotInheritTheAppEnvironment() async throws {
        setenv("CODEMODE_TEST_SECRET", "hunter2", 1)
        defer { unsetenv("CODEMODE_TEST_SECRET") }
        // The fake worker reports what it can see through its stderr tail.
        let result = try await runFakeWorker(
            "printf 'secret=%s path=%s\\n' \"${CODEMODE_TEST_SECRET-unset}\" \"${PATH:+set}\" >&2\nexit 5")
        let message = result.error?.message ?? ""
        XCTAssertTrue(message.contains("secret=unset"), message)
        XCTAssertFalse(message.contains("hunter2"), message)
        XCTAssertTrue(message.contains("path=set"), "the baseline PATH is still passed: \(message)")
    }

    // MARK: Agent allow-lists reach inside the registry

    func testFilteredCatalogKeepsOnlyWhatTheAgentMayUse() {
        let catalog = [descriptor("mcp__demo__allowed"), descriptor("mcp__demo__secret")]
        XCTAssertEqual(ToolSearchCatalog.filtered(catalog, by: nil).map(\.name),
                       ["mcp__demo__allowed", "mcp__demo__secret"], "no filter means no change")
        XCTAssertEqual(
            ToolSearchCatalog.filtered(catalog, by: { $0 != "mcp__demo__secret" }).map(\.name),
            ["mcp__demo__allowed"])
        XCTAssertNil(AppToolRegistry.callerToolFilter, "the main conversation has no filter")
    }

    func testCallerFilterReachesChildTasksButNotDetachedOnes() async {
        // Pins WHY codemode narrows its catalog in the registry's own task and
        // not in the worker's call handler (which runs on detached tasks).
        let filter: @Sendable (String) -> Bool = { $0 == "ok" }
        let seen: (child: Bool, detached: Bool) = await AppToolRegistry.$callerToolFilter.withValue(filter) {
            async let inChild = AppToolRegistry.callerToolFilter != nil
            let inDetached = await Task.detached { AppToolRegistry.callerToolFilter != nil }.value
            return (await inChild, inDetached)
        }
        XCTAssertTrue(seen.child)
        XCTAssertFalse(seen.detached, "a detached task does not inherit task-locals")
        XCTAssertNil(AppToolRegistry.callerToolFilter, "the value is gone outside the scope")
    }

    func testToolSearchBridgeHidesAndRefusesToolsTheCallingAgentMayNotUse() async throws {
        let server = McpServerConfig(
            name: "demo", transport: .stdio(command: "/bin/echo"),
            isEnabled: true, autoApprove: false, sourcePath: nil)
        func tool(_ name: String) -> McpDiscoveredTool {
            McpDiscoveredTool(
                name: name, description: "does \(name)", inputSchemaJSON: "{}",
                serverName: "demo", annotations: nil)
        }
        McpToolCatalogCache.shared.removeAll()
        McpToolCatalogCache.shared.setTools([tool("allowed_tool"), tool("secret_tool")], for: server)
        defer { McpToolCatalogCache.shared.removeAll() }
        var project = AppProject(name: "allow-list", permissions: AppProjectPermissions(mode: .auto))
        project.mcpServers = [server]

        func bridge(_ name: String, _ arguments: [String: String]) -> AppToolCall {
            AppToolCall(name: name, arguments: arguments, rawInvocation: name, category: .fileRead)
        }
        let search = bridge("tool_search", ["queries": "tool"])
        let describe = bridge("tool_describe", ["names": "mcp__demo__secret_tool"])
        let invoke = bridge("tool_call", ["name": "mcp__demo__secret_tool", "arguments": "{}"])

        // Control: with no filter the denied-to-be tool is findable.
        let open = try await ToolSearchExecutor.execute(call: search, project: project, chatID: nil)
        XCTAssertTrue(open.contains("mcp__demo__secret_tool"), open)

        let filter: @Sendable (String) -> Bool = { $0 != "mcp__demo__secret_tool" }
        let narrowed = try await AppToolRegistry.$callerToolFilter.withValue(filter) {
            try await ToolSearchExecutor.execute(call: search, project: project, chatID: nil)
        }
        XCTAssertTrue(narrowed.contains("mcp__demo__allowed_tool"), narrowed)
        XCTAssertFalse(narrowed.contains("mcp__demo__secret_tool"), "search must not reveal it: \(narrowed)")

        let described = try await AppToolRegistry.$callerToolFilter.withValue(filter) {
            try await ToolSearchExecutor.execute(call: describe, project: project, chatID: nil)
        }
        XCTAssertTrue(described.contains("not_found"), described)

        do {
            _ = try await AppToolRegistry.$callerToolFilter.withValue(filter) {
                try await ToolSearchExecutor.execute(call: invoke, project: project, chatID: nil)
            }
            XCTFail("calling a tool outside the agent's allow-list must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("not found in the current granted"),
                          error.localizedDescription)
        }
    }

    // MARK: Worker confinement (OS sandbox)

    /// Runs the real packaged executable in its sandbox-probe mode and parses
    /// the `name=result` lines it prints.
    private func runSandboxProbe(unconfined: Bool, readFile: String?) throws -> [String: String] {
        let process = Process()
        process.executableURL = try makeExecutableURL()
        var probeArguments = [REPLWorkerMain.workerModeArgument, CodemodeWorkerSandbox.probeArgument]
        if unconfined { probeArguments.append(CodemodeWorkerSandbox.unconfinedArgument) }
        if let readFile { probeArguments.append(readFile) }
        process.arguments = probeArguments
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        try process.run()
        // The probe finishes in well under a second; this only stops a stuck
        // child from hanging the suite.
        DispatchQueue.global().asyncAfter(deadline: .now() + 60) {
            if process.isRunning { process.terminate() }
        }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        var results: [String: String] = [:]
        for line in String(decoding: data, as: UTF8.self).split(separator: "\n") {
            guard let separator = line.firstIndex(of: "=") else { continue }
            results[String(line[..<separator])] = String(line[line.index(after: separator)...])
        }
        return results
    }

    func testSandboxedWorkerCannotReachTheOutsideWorld() throws {
        let home = NSHomeDirectory()
        try XCTSkipUnless(home.hasPrefix("/Users/"),
                          "the profile protects /Users, so the user-file probe needs a home under it")
        let userFile = home + "/.codemode-probe-\(UUID().uuidString)"
        try Data("private".utf8).write(to: URL(fileURLWithPath: userFile))
        defer { try? FileManager.default.removeItem(atPath: userFile) }

        // The control: with no profile every capability is really there, so a
        // denial below cannot be a probe that was never able to succeed.
        let control = try runSandboxProbe(unconfined: true, readFile: userFile)
        XCTAssertEqual(control["applied"], "skipped")
        XCTAssertEqual(control["javascript"], "ok")
        for capability in ["read_user_file", "read_own_bundle", "write_file", "exec", "spawn", "signal_parent"] {
            XCTAssertEqual(control[capability], "allowed", "control: \(capability) -> \(control)")
        }
        XCTAssertNotEqual(control["connect"], "denied", "control: \(control)")
        XCTAssertEqual(control["exec_inplace"], "attempting",
                       "control: the in-place exec succeeds, so the process ends there: \(control)")

        let confined = try runSandboxProbe(unconfined: false, readFile: userFile)
        XCTAssertEqual(confined["applied"], "ok", "\(confined)")
        // JavaScriptCore must still work under the profile.
        XCTAssertEqual(confined["javascript"], "ok", "\(confined)")
        XCTAssertLessThan(Double(confined["javascript_seconds"] ?? "99") ?? 99, 5,
                          "the JIT must not be crippled: \(confined)")
        // The worker's own bundle stays readable: the allow rule has to win
        // over the broader deny, because a development build lives under /Users.
        XCTAssertEqual(confined["read_own_bundle"], "allowed", "\(confined)")
        // Everything the profile exists to take away.
        XCTAssertEqual(confined["read_user_file"], "denied", "\(confined)")
        XCTAssertEqual(confined["write_file"], "denied", "\(confined)")
        XCTAssertEqual(confined["exec"], "denied", "\(confined)")
        XCTAssertEqual(confined["spawn"], "denied", "posix_spawn has no fork to be stopped by: \(confined)")
        XCTAssertEqual(confined["exec_inplace"], "denied",
                       "an in-place execve has no fork either; only process-exec* stops it: \(confined)")
        XCTAssertEqual(confined["connect"], "denied", "\(confined)")
        XCTAssertEqual(confined["signal_parent"], "denied", "\(confined)")
        // Reading the clipboard is a Mach service lookup. Only meaningful
        // where the control could reach the service at all.
        if control["mach_lookup"] == "allowed" {
            XCTAssertEqual(confined["mach_lookup"], "denied", "\(confined)")
        }
    }

    func testProfileQuotesTheAllowedReadRootAndPutsItAfterTheDeny() {
        let profile = CodemodeWorkerSandbox.profile(allowedReadRoot: "/tmp/a \"b\" \\c")
        XCTAssertTrue(profile.contains(#"(allow file-read* (subpath "/tmp/a \"b\" \\c"))"#), profile)
        let deny = profile.range(of: "(deny file-read*")
        let allow = profile.range(of: "(allow file-read*")
        XCTAssertNotNil(deny)
        XCTAssertNotNil(allow)
        if let deny, let allow {
            XCTAssertLessThan(deny.lowerBound, allow.lowerBound, "a later rule overrides an earlier one")
        }
        // A path with a control character cannot be quoted, so no rule is added.
        XCTAssertFalse(
            CodemodeWorkerSandbox.profile(allowedReadRoot: "/tmp/a\nb").contains("(allow file-read*"))
        XCTAssertFalse(CodemodeWorkerSandbox.profile(allowedReadRoot: nil).contains("(allow file-read*"))
    }

    func testOwnReadRootsAreRealPathsOfTheBundleAndExecutableDirectory() {
        let roots = CodemodeWorkerSandbox.ownReadRoots()
        XCTAssertFalse(roots.isEmpty)
        XCTAssertEqual(Set(roots).count, roots.count, "no duplicates")
        for root in roots {
            XCTAssertEqual(
                root, URL(fileURLWithPath: root).resolvingSymlinksInPath().path,
                "a Seatbelt subpath must name the resolved path, not a symlink to it: \(root)")
        }
        let profile = CodemodeWorkerSandbox.profile(allowedReadRoots: roots)
        for root in roots {
            XCTAssertTrue(profile.contains("(allow file-read* (subpath \"\(root)\"))"), profile)
        }
    }

    func testApplyingAnInvalidProfileFailsAndLeavesTheProcessUnconfined() throws {
        // `sandbox_init` compiles the profile before applying anything, so a
        // profile that does not parse is rejected and nothing is confined.
        // (This must never be given a profile that is valid and restrictive:
        // it would confine the test process itself, irreversibly.)
        let failure = CodemodeWorkerSandbox.apply(profile: "(((")
        XCTAssertNotNil(failure)
        let scratch = NSTemporaryDirectory() + "codemode-unconfined-\(UUID().uuidString)"
        XCTAssertTrue(FileManager.default.createFile(atPath: scratch, contents: Data("x".utf8)),
                      "the test process can still write")
        try? FileManager.default.removeItem(atPath: scratch)
    }

    // MARK: Typed declarations

    private func arguments(_ schema: String, maximumCharacters: Int = CodemodeSchemaRenderer.defaultMaximumCharacters) -> String {
        CodemodeSchemaRenderer.argumentsDeclaration(schemaJSON: schema, maximumCharacters: maximumCharacters)
    }

    func testRendererOrdersRequiredFirstAndMarksOptional() {
        XCTAssertEqual(
            arguments(#"{"type":"object","properties":{"verbose":{"type":"boolean"},"path":{"type":"string"},"limit":{"type":"integer"}},"required":["path"]}"#),
            "args: { path: string; limit?: number; verbose?: boolean }")
        XCTAssertEqual(
            arguments(#"{"type":"object","properties":{"q":{"type":"string"}}}"#),
            "args?: { q?: string }", "nothing required makes the whole parameter optional")
        XCTAssertEqual(arguments(#"{"type":"object","properties":{}}"#), "args?: {}")
    }

    func testRendererHandlesEnumsArraysNestingAndUnions() {
        XCTAssertEqual(
            arguments(#"""
            {"type":"object","properties":{
              "mode":{"enum":["fast","slow"]},
              "tags":{"type":"array","items":{"type":"string"}},
              "opts":{"type":"object","properties":{"deep":{"type":"boolean"}},"required":["deep"]},
              "ids":{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"number"}]}},
              "note":{"type":["string","null"]}}}
            """#),
            #"args?: { ids?: (string | number)[]; mode?: "fast" | "slow"; note?: string | null; opts?: { deep: boolean }; tags?: string[] }"#)
    }

    func testRendererQuotesNonIdentifierKeysAndResolvesReferences() {
        XCTAssertEqual(
            arguments(#"""
            {"type":"object","properties":{"weird-key":{"type":"string"},"2fa":{"type":"boolean"},"name":{"$ref":"#/definitions/Name"}},
             "required":["name"],"definitions":{"Name":{"type":"string"}}}
            """#),
            #"args: { name: string; "2fa"?: boolean; "weird-key"?: string }"#)
    }

    func testRendererSurvivesACyclicReference() {
        let declaration = arguments(#"""
            {"type":"object","properties":{"node":{"$ref":"#/definitions/Node"}},
             "definitions":{"Node":{"type":"object","properties":{"next":{"$ref":"#/definitions/Node"}}}}}
            """#)
        XCTAssertEqual(declaration, "args?: { node?: { next?: unknown } }")
    }

    func testRendererFallsBackForUnknownMalformedAndOversizedSchemas() {
        XCTAssertEqual(arguments("{}"), "args?: Record<string, unknown>")
        XCTAssertEqual(arguments("not json"), "args?: Record<string, unknown>")
        XCTAssertEqual(arguments("[1,2]"), "args?: Record<string, unknown>")

        // Many nested objects: too long in full, short once nested objects
        // collapse to `object`, so the second pass keeps the top-level names.
        let inner = (0..<8).map { "\"field_number_\($0)\":{\"type\":\"string\"}" }.joined(separator: ",")
        let outer = (0..<6).map { "\"section_\($0)\":{\"type\":\"object\",\"properties\":{\(inner)}}" }
            .joined(separator: ",")
        let nested = arguments("{\"type\":\"object\",\"properties\":{\(outer)}}")
        XCTAssertLessThanOrEqual(nested.count, CodemodeSchemaRenderer.defaultMaximumCharacters)
        XCTAssertTrue(nested.contains("section_0?: object"), nested)
        XCTAssertNotEqual(nested, "args: Record<string, unknown>")

        // Too many top-level properties even once collapsed.
        let wide = (0..<80).map { "\"property_name_number_\($0)\":{\"type\":\"string\"}" }.joined(separator: ",")
        XCTAssertEqual(
            arguments("{\"type\":\"object\",\"properties\":{\(wide)}}"),
            "args: Record<string, unknown>")
    }

    // MARK: Merged listing

    private func twoToolCatalog() -> [DeferredToolDescriptor] {
        [
            descriptor(
                "mcp__github__create_issue", description: "Create an issue in a repository.",
                schema: #"{"type":"object","properties":{"repo":{"type":"string"},"title":{"type":"string"},"body":{"type":"string"}},"required":["repo","title"]}"#),
            descriptor(
                "mcp__slack__send_message", description: "Send a message to a Slack channel.",
                schema: #"{"type":"object","properties":{"channel":{"type":"string"},"text":{"type":"string"}},"required":["channel","text"]}"#),
        ]
    }

    func testListingHasOneHeadingAndTypedSignatures() {
        let listing = CodemodeCatalog.promptListing(descriptors: twoToolCatalog(), enabled: true)
        XCTAssertEqual(listing.components(separatedBy: "## Deferred MCP Tools").count - 1, 1)
        XCTAssertFalse(listing.contains("## Codemode"), "no second section for the same tools")
        XCTAssertTrue(listing.contains("tool_search"), "the discovery instructions stay")
        XCTAssertTrue(listing.contains(
            "- `tools.mcp__github__create_issue(args: { repo: string; title: string; body?: string }): Promise<string>`: Create an issue in a repository."),
            listing)
        XCTAssertTrue(listing.contains("tools.mcp__slack__send_message(args: { channel: string; text: string })"), listing)
        XCTAssertEqual(listing.components(separatedBy: "mcp__github__create_issue").count - 1, 1,
                       "each tool is named once, not once per listing")
    }

    func testListingIsThePlainDeferredListingWhenCodemodeIsOffOrNotOffered() {
        let plain = ToolSearchCatalog.promptListing(descriptors: twoToolCatalog())
        XCTAssertEqual(CodemodeCatalog.promptListing(descriptors: twoToolCatalog(), enabled: false), plain)
        XCTAssertEqual(
            CodemodeCatalog.promptListing(
                descriptors: twoToolCatalog(), codemodeOffered: false, enabled: true),
            plain, "an agent without the codemode tool gets no bindings")
    }

    func testListingNotesTheToolNameWhenTheBindingDiffers() {
        let listing = CodemodeCatalog.promptListing(
            descriptors: [descriptor("mcp__my-server__run", description: "Run it.")], enabled: true)
        XCTAssertTrue(listing.contains("tools.mcp__my_server__run("), listing)
        XCTAssertTrue(listing.contains("(tool name: `mcp__my-server__run`)"), listing)
    }

    private var tenFieldSchema: String {
        "{\"type\":\"object\",\"properties\":{"
            + (0..<10).map { "\"argument_\($0)\":{\"type\":\"string\"}" }.joined(separator: ",") + "}}"
    }

    func testLongCatalogIsCutWithinTheBudgetAndTheCutIsAnnounced() {
        let catalog = (0..<150).map {
            descriptor(String(format: "mcp__bulk__tool_%03d", $0),
                       description: "Does bulk thing number \($0).", schema: tenFieldSchema)
        }
        let listing = CodemodeCatalog.promptListing(descriptors: catalog, enabled: true)
        XCTAssertLessThanOrEqual(listing.count, CodemodeCatalog.listingCharacterLimit)
        XCTAssertTrue(listing.contains("tools.mcp__bulk__tool_000(args?: {"), "early tools are typed")
        XCTAssertTrue(listing.contains("more tools are not listed"), "the cut is announced")
    }

    func testToolDegradesFromTypedToPlainLineAtTheBudgetBoundary() {
        // Sweep the budget one character at a time around a single tool's
        // typed line. Somewhere the typed line stops fitting while the plain
        // one still does; no budget may be overrun.
        let first = descriptor("mcp__bulk__one", description: "First.", schema: tenFieldSchema)
        let second = descriptor("mcp__bulk__two", description: "Second.", schema: tenFieldSchema)
        let entry = CodemodeIdentifier.entries(for: [first])[0]
        let typedLength = CodemodeCatalog.typedLine(entry, schemaJSON: tenFieldSchema).count
        let overhead = CodemodeCatalog.promptListing(descriptors: [first], enabled: true).count
            - 1 - typedLength

        var sawTyped = false
        var sawPlain = false
        for limit in max(600, overhead)...(overhead + typedLength + 200) {
            let listing = CodemodeCatalog.promptListing(
                descriptors: [first, second], contextTokens: limit * 20, enabled: true)
            let firstLine = listing.components(separatedBy: "\n")
                .first { $0.contains("mcp__bulk__one") }
            if firstLine?.contains("Promise<string>") == true { sawTyped = true }
            if firstLine?.contains("(args)`: First.") == true { sawPlain = true }
            // Once there is room for the header, a tool line and the closing
            // note, the listing must respect the budget it was given.
            if limit >= overhead + 200 {
                XCTAssertLessThanOrEqual(listing.count, limit, "limit \(limit)")
            }
        }
        XCTAssertTrue(sawTyped, "with room, the tool is typed")
        XCTAssertTrue(sawPlain, "at the boundary, the tool degrades to a plain line instead of vanishing")
    }

    // MARK: Nested runner (the re-gating entry point)

    private func nestedHandler(
        _ descriptors: [DeferredToolDescriptor], project: AppProject? = nil
    ) -> CodemodeWorkerSupervisor.CallHandler {
        CodemodeNestedRunner.callHandler(descriptors: descriptors, project: project, chatID: UUID())
    }

    private func failureMessage(_ result: Result<String, CodemodeCallError>) -> String? {
        if case let .failure(error) = result { return error.message }
        return nil
    }

    func testNestedRunnerRefusesCodemodeCallingItself() async {
        let handler = nestedHandler([descriptor("mcp__demo__echo")])
        let message = failureMessage(await handler("codemode", "{}"))
        XCTAssertTrue((message ?? "").contains("cannot invoke itself"), message ?? "succeeded")
        let shouting = failureMessage(await handler("CodeMode", nil))
        XCTAssertTrue((shouting ?? "").contains("cannot invoke itself"), "the check is case-insensitive")
    }

    func testNestedRunnerRefusesNamesOutsideTheGrantedCatalog() async {
        let handler = nestedHandler([descriptor("mcp__demo__echo")])
        for name in ["mcp__demo__other", "bash", "write_file", "tool_call"] {
            let message = failureMessage(await handler(name, "{}"))
            XCTAssertTrue((message ?? "").contains("not in the currently granted"), "\(name): \(message ?? "succeeded")")
        }
    }

    func testNestedRunnerRejectsArgumentsThatAreNotAJSONObject() async {
        let handler = nestedHandler([descriptor("mcp__demo__echo")])
        for arguments in ["[1,2]", "not json", "\"text\"", "42"] {
            let message = failureMessage(await handler("mcp__demo__echo", arguments))
            XCTAssertTrue((message ?? "").contains("not a JSON object"), "\(arguments): \(message ?? "succeeded")")
        }
    }

    func testNestedCallThePermissionEngineDeniesStaysDeniedInsideAScript() async {
        // The point of the whole design: wrapping a call in a script must not
        // route around its gate. A persisted deny rule rejects it.
        let project = AppProject(
            name: "nested-deny",
            permissions: AppProjectPermissions(mode: .auto, mcpDenyRules: ["mcp__demo__echo"]))
        let handler = nestedHandler([descriptor("mcp__demo__echo")], project: project)
        let message = failureMessage(await handler("mcp__demo__echo", "{}"))
        XCTAssertTrue((message ?? "").contains("denied"), message ?? "succeeded")
    }

    func testNestedCallThatNeedsApprovalFailsClosedInsideAScript() async {
        let project = AppProject(
            name: "nested-ask", permissions: AppProjectPermissions(mode: .ask))
        let handler = nestedHandler([descriptor("mcp__demo__echo")], project: project)
        let message = failureMessage(await handler("mcp__demo__echo", "{}"))
        XCTAssertTrue((message ?? "").contains("requires interactive approval"), message ?? "succeeded")
        XCTAssertTrue((message ?? "").contains("direct MCP tool"),
                      "the script is told what to do instead: \(message ?? "")")
    }

    func testOversizedNestedResultRejectsInsteadOfTruncating() async throws {
        var limits = CodemodeLimits()
        limits.maximumCallResultBytes = 1_000
        let handler: CodemodeWorkerSupervisor.CallHandler = { _, _ in
            .success(String(repeating: "z", count: 5_000))
        }
        let result = try await runScript(
            #"try { const r = await tools.mcp__demo__echo({}); return "got " + r.length; } catch (e) { return e.message; }"#,
            entries: [makeEntry()], handler: handler, limits: limits)
        XCTAssertTrue(result.ok, result.error?.message ?? "")
        let message = result.valueJSON ?? ""
        XCTAssertTrue(message.contains("5000 bytes"), message)
        XCTAssertTrue(message.contains("1000 byte limit"), message)
        XCTAssertFalse(message.hasPrefix("\"got"), "the script must not receive a clipped result: \(message)")
        XCTAssertEqual(result.calls.first?.status, .error)
    }
}
