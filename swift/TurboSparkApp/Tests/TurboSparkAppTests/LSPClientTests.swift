import XCTest
@testable import TurboSparkApp

/// OC-10: framing, handshake, diagnostics routing and failure paths of the
/// language server client, against a scripted in-process server.
final class LSPClientTests: XCTestCase {
    private let uri = "file:///tmp/proj/A.swift"

    private func started(_ behavior: ScriptedLSPServer.Behavior = .init(), timeouts: LSPTimeouts = LSPTimeouts())
        async throws -> (LSPClient, ScriptedLSPServer)
    {
        let server = ScriptedLSPServer(behavior)
        let client = LSPClient(transport: server, rootURL: URL(fileURLWithPath: "/tmp/proj"), timeouts: timeouts)
        try await client.start()
        return (client, server)
    }

    // MARK: Framing

    func testFramingRoundTripsAcrossArbitraryChunkBoundaries() throws {
        let one = LSPFraming.encode(Data(#"{"a":1}"#.utf8))
        let two = LSPFraming.encode(Data(#"{"b":"é"}"#.utf8))
        let stream = one + two
        var decoder = LSPFrameDecoder()
        var bodies: [Data] = []
        for byte in stream { bodies += try decoder.append(Data([byte])) }
        XCTAssertEqual(bodies.map { String(decoding: $0, as: UTF8.self) }, [#"{"a":1}"#, #"{"b":"é"}"#])
        var whole = LSPFrameDecoder()
        XCTAssertEqual(try whole.append(stream).count, 2)
    }

    func testFramingRejectsOversizeAndMalformedHeaders() {
        var decoder = LSPFrameDecoder()
        XCTAssertThrowsError(try decoder.append(Data("Content-Length: 999999999\r\n\r\n".utf8))) {
            XCTAssertEqual($0 as? LSPFraming.Failure, .oversizeFrame(999_999_999))
        }
        var missing = LSPFrameDecoder()
        XCTAssertThrowsError(try missing.append(Data("Content-Type: x\r\n\r\n{}".utf8))) {
            XCTAssertEqual($0 as? LSPFraming.Failure, .malformedHeader)
        }
        var negative = LSPFrameDecoder()
        XCTAssertThrowsError(try negative.append(Data("Content-Length: -5\r\n\r\n".utf8)))
        var endless = LSPFrameDecoder()
        XCTAssertThrowsError(try endless.append(Data(repeating: 0x41, count: LSPFraming.maximumHeaderBytes + 10)))
    }

    // MARK: Handshake and diagnostics

    func testHandshakeSendsInitializeThenInitializedWithTheRoot() async throws {
        let (client, server) = try await started()
        let serverName = await client.serverName
        XCTAssertEqual(serverName, "fake-ls")
        XCTAssertEqual(Array(server.receivedMethods().prefix(2)), ["initialize", "initialized"])
        let params = try XCTUnwrap(server.received.first?["params"] as? [String: Any])
        XCTAssertEqual(params["rootUri"] as? String, "file:///tmp/proj")
        await client.shutdown()
        XCTAssertTrue(server.killed)
    }

    func testDiagnosticsArriveForTheSyncedDocumentAndAreParsed() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.diagnostics = { _, _ in [ScriptedLSPServer.error(line: 4, character: 2, message: "boom")] }
        let (client, server) = try await started(behavior)
        let sync = try await client.sync(uri: uri, languageID: "swift", text: "x")
        let outcome = await client.awaitDiagnostics(uri: uri, version: sync.version, baseline: sync.baseline, budget: 2)
        guard case .fresh(let list) = outcome else { return XCTFail("expected fresh, got \(outcome)") }
        XCTAssertEqual(list.count, 1)
        XCTAssertEqual(list[0].message, "boom")
        XCTAssertEqual(list[0].range.start, LSPPosition(line: 4, character: 2))
        XCTAssertEqual(list[0].code, "7")
        XCTAssertTrue(list[0].isError)
        XCTAssertEqual(server.receivedMethods().suffix(2), ["textDocument/didOpen", "textDocument/didSave"])
        // The second sync is a full-text change with the next version.
        let second = try await client.sync(uri: uri, languageID: "swift", text: "y")
        XCTAssertEqual(second.version, 2)
        _ = await client.awaitDiagnostics(uri: uri, version: second.version, baseline: second.baseline, budget: 2)
        XCTAssertTrue(server.receivedMethods().contains("textDocument/didChange"))
        await client.shutdown()
    }

    /// The budget is the contract: a server that never publishes must not
    /// hold the caller past it. Mutation check: remove the timer in
    /// `awaitDiagnostics` (or multiply the budget) and this fails.
    func testSilentServerTimesOutWithinTheBudget() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.publishes = false
        let (client, _) = try await started(behavior)
        let sync = try await client.sync(uri: uri, languageID: "swift", text: "x")
        let started = ContinuousClock.now
        let outcome = await client.awaitDiagnostics(uri: uri, version: sync.version, baseline: sync.baseline, budget: 0.2)
        let elapsed = ContinuousClock.now - started
        XCTAssertEqual(outcome, .timedOut)
        XCTAssertLessThan(elapsed, .seconds(1.5))
        await client.shutdown()
    }

    func testStaleVersionPublishDoesNotCountAsFresh() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.versionOffset = -1
        behavior.diagnostics = { _, _ in [ScriptedLSPServer.error(line: 0, message: "old")] }
        let (client, _) = try await started(behavior)
        let sync = try await client.sync(uri: uri, languageID: "swift", text: "x")
        let outcome = await client.awaitDiagnostics(uri: uri, version: sync.version, baseline: sync.baseline, budget: 0.3)
        XCTAssertEqual(outcome, .timedOut)
        await client.shutdown()
    }

    func testPublishForAnUnopenedDocumentIsNotStored() async throws {
        let (client, server) = try await started()
        let params: [String: Any] = ["uri": "file:///tmp/proj/Other.swift", "diagnostics": [ScriptedLSPServer.error(line: 0, message: "x")]]
        server.inject(try LSPFraming.encode(object: ["jsonrpc": "2.0", "method": "textDocument/publishDiagnostics", "params": params]))
        try await Task.sleep(nanoseconds: 100_000_000)
        let known = await client.knownDiagnostics(uri: "file:///tmp/proj/Other.swift")
        XCTAssertNil(known)
        await client.shutdown()
    }

    // MARK: Requests

    func testRequestTimeoutFailsTheCallAndKeepsTheClientUsable() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.silentMethods = ["textDocument/documentSymbol"]
        var timeouts = LSPTimeouts()
        timeouts.request = 0.15
        let (client, _) = try await started(behavior, timeouts: timeouts)
        let began = ContinuousClock.now
        defer { XCTAssertLessThan(ContinuousClock.now - began, .seconds(3), "the request timeout is enforced") }
        do {
            _ = try await client.documentSymbols(uri: uri)
            XCTFail("expected a timeout")
        } catch let error as LSPClientError {
            guard case .timedOut = error else { return XCTFail("wrong error \(error)") }
        }
        let alive = await client.isAlive
        XCTAssertTrue(alive)
        await client.shutdown()
    }

    func testInitializeTimeoutKillsTheServer() async {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.silentMethods = ["initialize"]
        var timeouts = LSPTimeouts()
        timeouts.initialize = 0.15
        let server = ScriptedLSPServer(behavior)
        let client = LSPClient(transport: server, rootURL: URL(fileURLWithPath: "/tmp/proj"), timeouts: timeouts)
        do { try await client.start(); XCTFail("expected a timeout") } catch {}
        XCTAssertTrue(server.killed)
    }

    func testDocumentAndWorkspaceSymbolsParseBothReplyShapes() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        let range: [String: Any] = ["start": ["line": 1, "character": 0], "end": ["line": 3, "character": 1]]
        behavior.documentSymbols = [
            ["name": "Outer", "kind": 23, "range": range, "selectionRange": range,
             "children": [["name": "inner", "kind": 6, "range": range, "selectionRange": range]]],
        ]
        behavior.workspaceSymbols = [
            ["name": "Thing", "kind": 5, "containerName": "Mod", "location": ["uri": "file:///tmp/proj/T.swift", "range": range]],
            ["name": "broken"],
        ]
        let (client, _) = try await started(behavior)
        let tree = try await client.documentSymbols(uri: uri)
        XCTAssertEqual(tree.first?.name, "Outer")
        XCTAssertEqual(tree.first?.children.first?.name, "inner")
        let found = try await client.workspaceSymbols(query: "Thi")
        XCTAssertEqual(found.map(\.name), ["Thing"])
        XCTAssertEqual(found.first?.container, "Mod")
        await client.shutdown()
    }

    // MARK: Server-initiated traffic and hostile bytes

    func testServerRequestsAreAnsweredAndNothingIsPerformed() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.serverRequestAfterInit = ["jsonrpc": "2.0", "id": 99, "method": "client/registerCapability", "params": [String: Any]()]
        let (client, server) = try await started(behavior)
        try await Task.sleep(nanoseconds: 150_000_000)
        let answer = server.received.first { ($0["id"] as? Int) == 99 }
        XCTAssertNotNil(answer)
        XCTAssertTrue(answer?["result"] is NSNull)
        await client.shutdown()
    }

    func testOversizeFrameFromTheServerKillsIt() async throws {
        let (client, server) = try await started()
        server.inject(Data("Content-Length: \(LSPFraming.maximumFrameBytes + 1)\r\n\r\n".utf8))
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertTrue(server.killed)
        let alive = await client.isAlive
        XCTAssertFalse(alive)
    }

    func testServerExitFailsPendingRequestsAndMarksItDead() async throws {
        var behavior = ScriptedLSPServer.Behavior()
        behavior.silentMethods = ["workspace/symbol"]
        let (client, server) = try await started(behavior)
        let pending = Task { try await client.workspaceSymbols(query: "x") }
        try await Task.sleep(nanoseconds: 100_000_000)
        server.kill()
        do { _ = try await pending.value; XCTFail("expected failure") }
        catch { XCTAssertEqual(error as? LSPClientError, .notRunning) }
        let alive = await client.isAlive
        XCTAssertFalse(alive)
    }
}
