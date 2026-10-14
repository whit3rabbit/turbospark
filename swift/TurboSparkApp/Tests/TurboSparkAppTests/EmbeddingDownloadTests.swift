import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The pinned embedding download joins the shared queue like every other
/// model kind, and a saved row survives a relaunch.
@MainActor
final class EmbeddingDownloadTests: XCTestCase {
    private func makeModel() -> AppModel {
        let model = AppModel()
        model.downloadHistoryWritable = false
        model.modelDownloads = []
        model.activeModelDownloadID = nil
        model.embeddingProfiles = []
        return model
    }

    private func profile(installed: String?, canRun: Bool = true) throws -> EmbeddingProfile {
        let path = installed.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"alias":"embeddinggemma-2-coreml","display_name":"EmbeddingGemma 2 (Core ML)",
         "family":"embeddinggemma2","backend":"coreml","repository":"FluidInference/embeddinggemma-2-coreml",
         "revision":"3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4","license":"Apache-2.0",
         "modalities":["text","image","audio"],"dimensions":768,"matryoshka_dims":[768,512,256,128],
         "download_bytes":1481903308,"installed_path":\(path),"can_run":\(canRun),"unavailable_reason":null}
        """
        return try JSONDecoder().decode(EmbeddingProfile.self, from: Data(json.utf8))
    }

    func testEmbeddingRequestHasItsOwnQueueKeyAndRoundTrips() throws {
        let request = ModelDownload.Request.embedding(alias: "embeddinggemma-2-coreml")
        XCTAssertEqual(request.alias, "embeddinggemma-2-coreml")
        XCTAssertEqual(request.queueKey, "embedding:embeddinggemma-2-coreml")
        // The same alias as a text install is a different queue entry.
        XCTAssertNotEqual(request.queueKey, ModelDownload.Request.catalog(alias: "embeddinggemma-2-coreml").queueKey)
        let row = ModelDownload(request: request, status: .failed)
        let decoded = try JSONDecoder().decode(ModelDownload.self, from: JSONEncoder().encode(row))
        XCTAssertEqual(decoded.request, request)
        XCTAssertEqual(decoded.status, .failed)
    }

    func testEnqueueDispatchesOnceAndSkipsInstalledOrUnrunnable() throws {
        let model = makeModel()
        defer { model.stopCronScheduler() }
        var started: [ModelDownload.Request] = []
        model.modelDownloadStartOverride = { started.append($0) }

        XCTAssertFalse(model.enqueueEmbeddingDownload(try profile(installed: nil, canRun: false)),
                       "a model this Mac cannot run is not queued")
        let fresh = try profile(installed: nil)
        XCTAssertTrue(model.enqueueEmbeddingDownload(fresh))
        XCTAssertEqual(started, [.embedding(alias: fresh.alias)])
        XCTAssertFalse(model.enqueueEmbeddingDownload(fresh), "a pending download is not queued twice")

        let installed = try profile(installed: "/tmp/x")
        model.embeddingProfiles = [installed]
        XCTAssertTrue(model.isModelAlreadyInstalled(.embedding(alias: installed.alias)))
    }

    func testProfileDecodesSnakeCaseFields() throws {
        let p = try profile(installed: "/store/models/embeddings/embeddinggemma-2-coreml")
        XCTAssertEqual(p.displayName, "EmbeddingGemma 2 (Core ML)")
        XCTAssertEqual(p.matryoshkaDims, [768, 512, 256, 128])
        XCTAssertEqual(p.downloadBytes, 1_481_903_308)
        XCTAssertTrue(p.canRun)
        XCTAssertNotNil(p.installedPath)
    }
}
