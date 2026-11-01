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

    nonisolated private static let pin = "3f978ddf92cac9aa07eb2dcf6eb36cf450f3dba4"

    private func profile(
        installed: String?, canRun: Bool = true, status: String = "verified",
        revision: String = EmbeddingDownloadTests.pin
    ) throws -> EmbeddingProfile {
        let path = installed.map { "\"\($0)\"" } ?? "null"
        let json = """
        {"alias":"embeddinggemma-2-coreml","display_name":"EmbeddingGemma 2 (Core ML)",
         "family":"embeddinggemma2","backend":"coreml","repository":"FluidInference/embeddinggemma-2-coreml",
         "revision":"\(revision)","license":"Apache-2.0",
         "modalities":["text","image","audio"],"dimensions":768,"matryoshka_dims":[768,512,256,128],
         "status":"\(status)","gates":["coreml_embeddinggemma2"],
         "download_bytes":1481903308,"installed_path":\(path),"can_run":\(canRun),"unavailable_reason":null}
        """
        return try JSONDecoder().decode(EmbeddingProfile.self, from: Data(json.utf8))
    }

    func testEmbeddingRequestHasItsOwnQueueKeyAndRoundTrips() throws {
        let request = ModelDownload.Request.embedding(alias: "embeddinggemma-2-coreml", revision: Self.pin)
        XCTAssertEqual(request.alias, "embeddinggemma-2-coreml")
        XCTAssertTrue(request.queueKey.hasPrefix("embedding:"))
        // The same alias as a text install is a different queue entry.
        XCTAssertNotEqual(request.queueKey, ModelDownload.Request.catalog(alias: "embeddinggemma-2-coreml").queueKey)
        // A different pin is a different download: a retry cannot adopt it.
        XCTAssertNotEqual(
            request.queueKey,
            ModelDownload.Request.embedding(alias: "embeddinggemma-2-coreml", revision: String(repeating: "1", count: 40)).queueKey)
        let row = ModelDownload(request: request, status: .failed)
        let decoded = try JSONDecoder().decode(ModelDownload.self, from: JSONEncoder().encode(row))
        XCTAssertEqual(decoded.request, request)
        XCTAssertEqual(decoded.status, .failed)
    }

    /// A history row saved before the revision was recorded still decodes,
    /// so the history file does not fail to load as a whole.
    func testLegacyEmbeddingRowWithoutRevisionDecodes() throws {
        let legacy = #"{"embedding":{"alias":"embeddinggemma-2-coreml"}}"#
        let request = try JSONDecoder().decode(ModelDownload.Request.self, from: Data(legacy.utf8))
        XCTAssertEqual(request, .embedding(alias: "embeddinggemma-2-coreml", revision: nil))
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
        XCTAssertEqual(started, [.embedding(alias: fresh.alias, revision: fresh.revision)])
        XCTAssertFalse(model.enqueueEmbeddingDownload(fresh), "a pending download is not queued twice")

        let installed = try profile(installed: "/tmp/x")
        model.embeddingProfiles = [installed]
        XCTAssertTrue(model.isModelAlreadyInstalled(.embedding(alias: installed.alias, revision: installed.revision)))
        // A request for another pin is not satisfied by this install.
        XCTAssertFalse(model.isModelAlreadyInstalled(
            .embedding(alias: installed.alias, revision: String(repeating: "1", count: 40))))
    }

    func testProfileStatusComesFromTheCatalogRow() throws {
        XCTAssertTrue(try profile(installed: nil).isVerified)
        XCTAssertFalse(try profile(installed: nil, status: "runs").isVerified)
        // A row from an engine without the field never claims a test.
        let old = """
        {"alias":"a","display_name":"A","family":"f","backend":"coreml","repository":"o/r",
         "revision":"\(Self.pin)","license":"MIT","modalities":["text"],"dimensions":8,
         "matryoshka_dims":[8],"download_bytes":1,"installed_path":null,"can_run":true,
         "unavailable_reason":null}
        """
        let decoded = try JSONDecoder().decode(EmbeddingProfile.self, from: Data(old.utf8))
        XCTAssertFalse(decoded.isVerified)
        XCTAssertEqual(decoded.installState, "absent")
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
