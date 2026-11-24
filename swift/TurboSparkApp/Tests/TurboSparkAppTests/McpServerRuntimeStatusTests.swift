import XCTest
@testable import TurboSparkApp

private actor StatusDiscoveryGate {
    typealias Reply = CheckedContinuation<[McpDiscoveredTool], Error>
    private var pending: [Reply] = []
    private var waiter: CheckedContinuation<Reply, Never>?

    func discover() async throws -> [McpDiscoveredTool] {
        try await withCheckedThrowingContinuation { reply in
            if let waiter {
                self.waiter = nil
                waiter.resume(returning: reply)
            } else {
                pending.append(reply)
            }
        }
    }

    func next() async -> Reply {
        if !pending.isEmpty { return pending.removeFirst() }
        return await withCheckedContinuation { waiter = $0 }
    }
}

private struct FixtureError: LocalizedError {
    let errorDescription: String?
}

/// The MCP Servers pane status: the pure mapping, and the cache events it
/// reads (failure recorded, cleared by success, scoped to the transport).
@MainActor
final class McpServerRuntimeStatusTests: XCTestCase {
    private func server(_ command: String = "old", enabled: Bool = true) -> McpServerConfig {
        McpServerConfig(name: "fixture", transport: .stdio(command: command), isEnabled: enabled)
    }

    private func tools(_ count: Int) -> [McpDiscoveredTool] {
        (0..<count).map { McpDiscoveredTool(name: "t\($0)", description: "fixture", serverName: "fixture") }
    }

    // MARK: - Mapping

    func testDisabledWinsOverEverything() {
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: false, isTesting: true, isDiscovering: true,
                cachedToolCount: 3, lastFailure: "boom"),
            .disabled)
    }

    func testWorkInFlightReadsConnecting() {
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: true, isTesting: true, isDiscovering: false,
                cachedToolCount: nil, lastFailure: "boom"),
            .connecting)
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: true, isTesting: false, isDiscovering: true,
                cachedToolCount: 2, lastFailure: nil),
            .connecting)
    }

    func testFailureBeatsOlderCachedTools() {
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: true, isTesting: false, isDiscovering: false,
                cachedToolCount: 4, lastFailure: "  exited (status 1)\n"),
            .failed(reason: "exited (status 1)"))
    }

    func testCachedToolsReadConnectedWithCount() {
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: true, isTesting: false, isDiscovering: false,
                cachedToolCount: 0, lastFailure: nil),
            .connected(toolCount: 0))
    }

    func testNothingKnownReadsNotConnected() {
        XCTAssertEqual(
            McpServerRuntimeStatus.resolve(
                isEnabled: true, isTesting: false, isDiscovering: false,
                cachedToolCount: nil, lastFailure: nil),
            .notConnected)
    }

    // MARK: - Cache events

    func testBackgroundFailureIsRecordedThenClearedBySuccess() async throws {
        let gate = StatusDiscoveryGate()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        let task = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        XCTAssertTrue(cache.isDiscovering(server()))
        let reply = await gate.next()
        reply.resume(throwing: FixtureError(errorDescription: "server exited"))
        await task.value
        XCTAssertFalse(cache.isDiscovering(server()))
        XCTAssertEqual(cache.lastFailure(for: server()), "server exited")
        XCTAssertNil(cache.cachedToolCount(for: server()))

        // A failure leaves the server stale, so a refresh still retries it.
        let retry = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        let retryReply = await gate.next()
        retryReply.resume(returning: tools(2))
        await retry.value
        XCTAssertNil(cache.lastFailure(for: server()))
        XCTAssertEqual(cache.cachedToolCount(for: server()), 2)
    }

    func testFailureIsScopedToTheTransportThatFailed() {
        let cache = McpToolCatalogCache(discover: { _, _ in [] })
        cache.recordFailure("bad command", for: server())
        XCTAssertEqual(cache.lastFailure(for: server()), "bad command")
        XCTAssertNil(cache.lastFailure(for: server("edited")))
    }

    func testExplicitFailureKeepsAdvertisedTools() {
        let cache = McpToolCatalogCache(discover: { _, _ in [] })
        cache.setTools(tools(3), for: server())
        cache.recordFailure("timed out", for: server())
        XCTAssertEqual(cache.tools(forServerName: "fixture")?.count, 3)
        XCTAssertEqual(cache.lastFailure(for: server()), "timed out")
        cache.setTools(tools(1), for: server())
        XCTAssertNil(cache.lastFailure(for: server()))
    }

    func testRemovalAndResetClearFailures() {
        let cache = McpToolCatalogCache(discover: { _, _ in [] })
        cache.recordFailure("x", for: server())
        cache.removeServer(named: "FIXTURE")
        XCTAssertNil(cache.lastFailure(for: server()))
        cache.recordFailure("y", for: server())
        cache.removeAll()
        XCTAssertNil(cache.lastFailure(for: server()))
    }

    func testRevokedDiscoveryRecordsNoFailure() async throws {
        let gate = StatusDiscoveryGate()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        let task = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        let reply = await gate.next()
        cache.removeAll()
        reply.resume(throwing: FixtureError(errorDescription: "late"))
        await task.value
        XCTAssertNil(cache.lastFailure(for: server()))
    }

    func testChangesPostTheNotification() {
        let cache = McpToolCatalogCache(discover: { _, _ in [] })
        let posted = expectation(forNotification: McpToolCatalogCache.didChangeNotification, object: cache)
        cache.recordFailure("x", for: server())
        wait(for: [posted], timeout: 1)
    }
}
