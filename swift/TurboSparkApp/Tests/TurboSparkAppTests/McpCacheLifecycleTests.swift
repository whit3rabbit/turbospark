import XCTest
@testable import TurboSparkApp

private actor ControlledDiscovery {
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

@MainActor
final class McpCacheLifecycleTests: XCTestCase {
    private func server(_ command: String = "old", enabled: Bool = true) -> McpServerConfig {
        McpServerConfig(name: "fixture", transport: .stdio(command: command), isEnabled: enabled)
    }

    private func tools(_ name: String) -> [McpDiscoveredTool] {
        [McpDiscoveredTool(name: name, description: "fixture", serverName: "fixture")]
    }

    private func invalidation(_ invalidate: (McpToolCatalogCache) -> Void) async throws {
        let gate = ControlledDiscovery()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        let task = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        let reply = await gate.next()
        invalidate(cache)
        reply.resume(returning: tools("late"))
        await task.value
        XCTAssertNil(cache.tools(forServerName: "fixture"))
    }

    func testResetRejectsPendingDiscovery() async throws {
        try await invalidation { $0.removeAll() }
    }

    func testRemovalRejectsPendingDiscovery() async throws {
        try await invalidation { $0.removeServer(named: "FIXTURE") }
    }

    func testDisableRejectsPendingDiscovery() async throws {
        try await invalidation {
            XCTAssertTrue($0.refreshEnabled(servers: [server(enabled: false)], workingDirectory: nil).isEmpty)
        }
    }

    func testPruningRejectsPendingDiscovery() async throws {
        try await invalidation {
            XCTAssertTrue($0.refreshEnabled(servers: [], workingDirectory: nil).isEmpty)
        }
    }

    private func replacement(oldFails: Bool, reset: Bool = false) async throws {
        let gate = ControlledDiscovery()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        let oldTask = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        let oldReply = await gate.next()
        XCTAssertTrue(cache.refreshEnabled(servers: [server()], workingDirectory: nil).isEmpty)
        let current = reset ? server() : server("new")
        if reset { cache.removeAll() }
        let newTask = try XCTUnwrap(cache.refreshTasks(servers: [current], workingDirectory: nil).first?.task)
        let newReply = await gate.next()
        if oldFails {
            oldReply.resume(throwing: NSError(domain: "fixture", code: 1))
        } else {
            oldReply.resume(returning: tools("old"))
        }
        await oldTask.value
        XCTAssertNil(cache.tools(forServerName: "fixture"))
        // An obsolete completion must not clear the replacement's reservation.
        let duplicate = cache.refreshTasks(servers: [current], workingDirectory: nil)
        XCTAssertTrue(duplicate.isEmpty)
        newReply.resume(returning: tools("new"))
        await newTask.value
        for item in duplicate {
            let reply = await gate.next()
            reply.resume(returning: tools("new"))
            await item.task.value
        }
        XCTAssertEqual(cache.tools(forServerName: "fixture"), tools("new"))
        XCTAssertFalse(cache.isStale(for: current))
    }

    func testTransportReplacementRejectsOldSuccess() async throws {
        try await replacement(oldFails: false)
    }

    func testResetReplacementRejectsOldSuccess() async throws {
        try await replacement(oldFails: false, reset: true)
    }

    func testTransportReplacementRejectsOldFailure() async throws {
        try await replacement(oldFails: true)
    }

    func testExplicitToolsSupersedePendingDiscovery() async throws {
        let gate = ControlledDiscovery()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        let task = try XCTUnwrap(cache.refreshTasks(servers: [server()], workingDirectory: nil).first?.task)
        let reply = await gate.next()
        cache.setTools(tools("manual"), for: server())
        reply.resume(returning: tools("late"))
        await task.value
        XCTAssertEqual(cache.tools(forServerName: "fixture"), tools("manual"))
    }

    func testFailureRetainsCacheAndAllowsRetry() async throws {
        let gate = ControlledDiscovery()
        let cache = McpToolCatalogCache(discover: { _, _ in try await gate.discover() })
        cache.setTools(tools("cached"), for: server())
        let task = try XCTUnwrap(cache.refreshTasks(servers: [server("new")], workingDirectory: nil).first?.task)
        let reply = await gate.next()
        reply.resume(throwing: NSError(domain: "fixture", code: 1))
        await task.value
        XCTAssertEqual(cache.tools(forServerName: "fixture"), tools("cached"))
        let retry = try XCTUnwrap(cache.refreshTasks(servers: [server("new")], workingDirectory: nil).first?.task)
        let retryReply = await gate.next()
        retryReply.resume(returning: tools("new"))
        await retry.value
        XCTAssertEqual(cache.tools(forServerName: "fixture"), tools("new"))
        XCTAssertTrue(cache.refreshEnabled(servers: [server("new")], workingDirectory: nil).isEmpty)
    }
}
