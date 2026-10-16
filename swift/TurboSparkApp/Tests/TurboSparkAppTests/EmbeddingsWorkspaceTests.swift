import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The API Embeddings tab: embedding ids are listed apart from chat models,
/// and Attach states why it is unavailable.
@MainActor
final class EmbeddingsWorkspaceTests: XCTestCase {
    private func info(models: [String], embedding: [String]) throws -> ServerInfo {
        let json = """
        {"port":8080,"host":"127.0.0.1","modelId":"\(models.first ?? "")","models":\(models),
         "embeddingModels":\(embedding),"authEnabled":false}
        """
        return try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
    }

    func testEmbeddingIdsStayOutOfTheTextRows() throws {
        let model = AppModel()
        defer { model.stopCronScheduler() }
        model.serverInfo = try info(
            models: ["gemma4.gturbo", "embeddinggemma-2-coreml"],
            embedding: ["embeddinggemma-2-coreml"])
        XCTAssertEqual(model.serverEmbeddingModelIDs, ["embeddinggemma-2-coreml"])
        XCTAssertEqual(model.serverTextModelIDs, ["gemma4.gturbo"])
        XCTAssertEqual(model.serverModelRows.map(\.id), ["gemma4.gturbo"])
    }

    private func probe(runnable: Bool, reason: String? = nil) throws -> EmbeddingProbe {
        let why = reason.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"input":"x","resolved_path":"/x","runnable":\(runnable),"family":"gte",
         "backend":null,"verified":false,"profile_alias":null,"refused_because":\(why)}
        """
        return try JSONDecoder().decode(EmbeddingProbe.self, from: Data(json.utf8))
    }

    func testAttachBlockStatesWhy() throws {
        typealias M = AppModel
        XCTAssertEqual(
            M.embeddingAttachBlock(serverRunning: false, selection: "m", probe: nil, busy: false, attached: []),
            .serverStopped)
        XCTAssertEqual(
            M.embeddingAttachBlock(serverRunning: true, selection: "  ", probe: nil, busy: false, attached: []),
            .noSelection)
        XCTAssertEqual(
            M.embeddingAttachBlock(serverRunning: true, selection: "m", probe: nil, busy: true, attached: []),
            .busy)
        XCTAssertEqual(
            M.embeddingAttachBlock(serverRunning: true, selection: " m ", probe: nil, busy: false, attached: ["m"]),
            .alreadyAttached)
        XCTAssertEqual(
            M.embeddingAttachBlock(
                serverRunning: true, selection: "m",
                probe: try probe(runnable: false, reason: "gte is not supported"),
                busy: false, attached: []),
            .refused("gte is not supported"))
        XCTAssertNil(M.embeddingAttachBlock(
            serverRunning: true, selection: "m", probe: try probe(runnable: true), busy: false, attached: []))
    }

    /// A stop pressed while an embedding attach holds the server's lock is
    /// recorded instead of blocking the main thread on that lock.
    func testStopDuringAnAttachIsDeferred() throws {
        let model = AppModel()
        defer { model.stopCronScheduler() }
        let server = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { server.stop() }
        model.server = server
        model.serverEmbeddingAttachInFlight = true
        model.stopServer()
        XCTAssertTrue(model.serverStopAfterEmbeddingAttach)
        XCTAssertNotNil(model.server, "the stop waits for the attach")
        model.serverEmbeddingAttachInFlight = false
        model.serverStopAfterEmbeddingAttach = false
        model.stopServer()
        XCTAssertNil(model.server)
    }
}
