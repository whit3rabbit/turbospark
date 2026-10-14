import XCTest
import SyntextContext

@testable import TurboSparkApp

/// `ContextEngineManager` against the real engine, compiled into the same
/// archive the app links: a project synced and retrieved from, a chat's
/// turns recalled, isolated, rewritten and forgotten down to the bytes on
/// disk, and the tools run with the turn's scopes. Stores live under the
/// test process's own `AppStorageRoot`.
final class ContextEngineManagerTests: XCTestCase {
    private var root: URL!
    private let manager = ContextEngineManager.shared

    override func setUp() async throws {
        await manager.removeAllStores()
        await manager.setModelPath(nil)
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("context-engine-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("docs"), withIntermediateDirectories: true)
        try """
            # Deploys

            ## Rolling back

            To roll back a deploy, run `bin/rollback --to previous`. This is the zebracorn procedure.
            """.write(to: root.appendingPathComponent("docs/deploy.md"), atomically: true, encoding: .utf8)
        try """
            /// Limits requests per client.
            func rateLimiter(requestsPerMinute: Int) -> Bool { requestsPerMinute < 120 }
            """.write(to: root.appendingPathComponent("Limiter.swift"), atomically: true, encoding: .utf8)
    }

    override func tearDown() async throws {
        await manager.removeAllStores()
        try? FileManager.default.removeItem(at: root)
    }

    /// Records sync states the manager reports.
    private final class States: @unchecked Sendable {
        private let lock = NSLock()
        private var states: [ContextEngineSyncState] = []
        func append(_ state: ContextEngineSyncState) { lock.lock(); states.append(state); lock.unlock() }
        var last: ContextEngineSyncState? { lock.lock(); defer { lock.unlock() }; return states.last }
    }

    private func syncProject(_ id: UUID) async throws {
        let states = States()
        await manager.setObserver { projectID, state in
            if projectID == id { states.append(state) }
        }
        await manager.activateProject(id: id, root: root)
        let deadline = Date().addingTimeInterval(60)
        while Date() < deadline {
            switch states.last {
            case .ready: return
            case .failed(let message): return XCTFail("sync failed: \(message)")
            default: try await Task.sleep(nanoseconds: 50_000_000)
            }
        }
        XCTFail("sync did not finish")
    }

    private func request(
        chatID: UUID, recall: ContextChatLocation?, projectID: UUID?, question: String,
        rows: [ContextRecall.Row] = []
    ) -> ContextEvidenceRequest {
        ContextEvidenceRequest(
            chatID: chatID, recall: recall, projectID: projectID, question: question,
            evidenceTokens: 800, recallRows: rows, keptTurns: 0, toolResultTokens: 512)
    }

    func testProjectEvidenceComesFromTheProjectsFiles() async throws {
        let projectID = UUID()
        try await syncProject(projectID)
        let evidence = try await manager.prepareEvidence(request(
            chatID: UUID(), recall: nil, projectID: projectID,
            question: "How do I roll back a deploy?"))
        let text = try XCTUnwrap(evidence?.text)
        XCTAssertTrue(text.contains("bin/rollback"), text)
        XCTAssertTrue(text.contains("<evidence"), "rendered as evidence blocks")
        XCTAssertLessThanOrEqual(evidence?.tokens ?? .max, 800)
        XCTAssertEqual(evidence?.lowestTrust, .workspace)

        let stats = await manager.stats(projectID: projectID)
        XCTAssertEqual(stats?.documents, 2)
    }

    func testFilesTheAppWritesAreSyncedWithoutAWalk() async throws {
        let projectID = UUID()
        try await syncProject(projectID)
        let note = root.appendingPathComponent("docs/billing.md")
        try "Invoices are issued on the quokkaday of each month.\n"
            .write(to: note, atomically: true, encoding: .utf8)
        await manager.filesChanged([note], root: root)
        var found = false
        for _ in 0..<100 where !found {
            let evidence = try await manager.prepareEvidence(request(
                chatID: UUID(), recall: nil, projectID: projectID,
                question: "When are invoices issued quokkaday"))
            found = evidence?.text.contains("quokkaday") ?? false
            if !found { try await Task.sleep(nanoseconds: 50_000_000) }
        }
        XCTAssertTrue(found, "a written file is searchable without a full sync")
    }

    func testRecallBringsBackAChatsOwnTurnsOnly() async throws {
        let chatID = UUID()
        let location = ContextChatLocation(store: .chats, chatID: chatID)
        let rows = [
            ContextRecall.Row(role: .user, text: "My locker code is 4417."),
            ContextRecall.Row(role: .assistant, text: "Noted, locker 4417."),
        ]
        await manager.recordTurn(chatID: chatID, store: .chats, rows: rows)
        let recalled = try await manager.prepareEvidence(request(
            chatID: chatID, recall: location, projectID: nil,
            question: "What is my locker code?", rows: rows))
        XCTAssertTrue(recalled?.text.contains("4417") ?? false)
        XCTAssertEqual(recalled?.lowestTrust, .memory)

        // Another chat's scope is not granted to this one.
        let other = UUID()
        let elsewhere = try await manager.prepareEvidence(request(
            chatID: other, recall: ContextChatLocation(store: .chats, chatID: other), projectID: nil,
            question: "What is my locker code?"))
        XCTAssertFalse(elsewhere?.text.contains("4417") ?? false)
    }

    func testReconcileFollowsAnEditedTranscript() async throws {
        let chatID = UUID()
        let location = ContextChatLocation(store: .chats, chatID: chatID)
        let original = [
            ContextRecall.Row(role: .user, text: "Deploy to staging."),
            ContextRecall.Row(role: .assistant, text: "Deployed to staging."),
        ]
        try await manager.reconcile(location, rows: original)
        let appended = original + [ContextRecall.Row(role: .user, text: "And production?")]
        let conversation = try await manager.reconcile(location, rows: appended)
        XCTAssertEqual(try conversation.history().map(\.content), appended.map(\.text))

        let edited = [
            ContextRecall.Row(role: .user, text: "Deploy to canary."),
            ContextRecall.Row(role: .assistant, text: "Deployed to canary."),
        ]
        try await manager.reconcile(location, rows: edited)
        XCTAssertEqual(try conversation.history().map(\.content), edited.map(\.text))
        // And what a fresh handle reads from the store agrees.
        await manager.removeConversationHandlesForTesting()
        let reopened = try await manager.reconcile(location, rows: edited)
        XCTAssertEqual(try reopened.history().map(\.content), edited.map(\.text))
    }

    func testForgettingAChatRemovesItsTurnsFromDisk() async throws {
        let chatID = UUID()
        let location = ContextChatLocation(store: .chats, chatID: chatID)
        let rows = [ContextRecall.Row(role: .user, text: "The vault phrase is xylozebra.")]
        await manager.recordTurn(chatID: chatID, store: .chats, rows: rows)
        let directory = ContextEngineManager.directory(for: .chats)
        XCTAssertTrue(Self.bytes(under: directory).contains("xylozebra"))

        await manager.forgetChat(chatID, projectID: nil)
        let after = try await manager.prepareEvidence(request(
            chatID: chatID, recall: location, projectID: nil, question: "What is the vault phrase?"))
        XCTAssertFalse(after?.text.contains("xylozebra") ?? false)

        // Gone from every file once the debounced compaction has run.
        let deadline = Date().addingTimeInterval(20)
        while Date() < deadline, Self.bytes(under: directory).contains("xylozebra") {
            try await Task.sleep(nanoseconds: 200_000_000)
        }
        XCTAssertFalse(Self.bytes(under: directory).contains("xylozebra"))
    }

    func testToolsRunWithTheTurnsOptions() async throws {
        let projectID = UUID()
        try await syncProject(projectID)
        let chatID = UUID()
        _ = try await manager.prepareEvidence(request(
            chatID: chatID, recall: nil, projectID: projectID, question: "rate limiter"))

        let outline = try await manager.callTool(
            name: "context_outline", arguments: "{}", chatID: chatID, projectID: projectID)
        XCTAssertFalse(outline.isError, outline.content)
        // The top level lists documents and directories.
        XCTAssertTrue(outline.content.contains("docs"), outline.content)
        XCTAssertTrue(outline.content.contains("Limiter.swift"), outline.content)

        let read = try await manager.callTool(
            name: "context_read", arguments: #"{"source": "docs/deploy.md"}"#,
            chatID: chatID, projectID: projectID)
        XCTAssertTrue(read.content.contains("zebracorn"), read.content)

        let missing = try await manager.callTool(
            name: "context_read", arguments: #"{"source": "nope.md"}"#,
            chatID: chatID, projectID: projectID)
        XCTAssertTrue(missing.isError, "a source that does not exist is the model's to fix")
    }

    /// The definitions are written with the app's schema builders; this is
    /// what catches a syntext bump that renames a tool or an argument.
    func testToolDefinitionsMatchTheEngines() throws {
        let json = try SyntextContextEngine.toolDefinitionsJSON()
        let list = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [[String: Any]])
        var engine: [String: (properties: Set<String>, required: Set<String>)] = [:]
        for entry in list {
            let function = try XCTUnwrap(entry["function"] as? [String: Any])
            let name = try XCTUnwrap(function["name"] as? String)
            let parameters = try XCTUnwrap(function["parameters"] as? [String: Any])
            let properties = Set((parameters["properties"] as? [String: Any] ?? [:]).keys)
            let required = Set(parameters["required"] as? [String] ?? [])
            engine[name] = (properties, required)
        }
        XCTAssertEqual(Set(engine.keys), ContextEngineToolDefinitions.names)
        for tool in ContextEngineToolDefinitions.all {
            let name = tool.function.name
            XCTAssertEqual(
                Set(tool.function.parameters.properties?.keys.map { $0 } ?? []),
                engine[name]?.properties, name)
            XCTAssertEqual(Set(tool.function.parameters.required ?? []), engine[name]?.required, name)
        }
    }

    private static func bytes(under directory: URL) -> String {
        guard let walker = FileManager.default.enumerator(at: directory, includingPropertiesForKeys: nil)
        else { return "" }
        var all = Data()
        for case let url as URL in walker {
            if let data = try? Data(contentsOf: url) { all.append(data) }
        }
        return String(decoding: all, as: UTF8.self)
    }
}
