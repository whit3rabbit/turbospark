import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9: the destination rides on the download row, so a queued install, a
/// retry and a restart keep it, and a row written before libraries reads as This
/// Mac without any change to the saved bytes.
final class ModelDownloadLibraryTests: XCTestCase {
    private func encoded(_ download: ModelDownload) throws -> [String: Any] {
        let data = try JSONEncoder().encode(download)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testTheDestinationSurvivesASaveAndALoad() throws {
        let download = ModelDownload(request: .catalog(alias: "qwen"), status: .queued, library: "t7-id")
        let decoded = try JSONDecoder().decode(ModelDownload.self, from: JSONEncoder().encode(download))
        XCTAssertEqual(decoded.library, "t7-id")
        XCTAssertEqual(decoded.request, .catalog(alias: "qwen"))
        XCTAssertEqual(decoded.id, download.id)
    }

    func testAThisMacRowSavesNoLibraryKey() throws {
        let row = try encoded(ModelDownload(request: .catalog(alias: "qwen"), status: .queued))
        XCTAssertNil(row["library"], "a This Mac row is the same bytes it was before libraries")
    }

    func testARowSavedBeforeLibrariesReadsAsThisMac() throws {
        let id = UUID().uuidString
        let json = """
            {"id":"\(id)","request":{"catalog":{"alias":"qwen"}},"status":"failed","failure":"x",
             "downloadedBytes":5,"startedAt":0,"updatedAt":0}
            """
        let decoded = try JSONDecoder().decode(ModelDownload.self, from: Data(json.utf8))
        XCTAssertNil(decoded.library)
        XCTAssertEqual(decoded.status, .failed)
    }

    func testTheDestinationIsNotPartOfTheQueueKey() {
        // Alias plus modality is unique across every library, so one model cannot
        // be queued twice for two drives.
        let request = ModelDownload.Request.catalog(alias: "qwen")
        XCTAssertEqual(request.queueKey, "text:qwen")
    }

    func testOnlyEmbeddingRequestsAreHomeOnly() {
        XCTAssertTrue(ModelDownload.Request.embedding(alias: "e").installsOnlyOnThisMac)
        XCTAssertFalse(ModelDownload.Request.catalog(alias: "q").installsOnlyOnThisMac)
        XCTAssertFalse(ModelDownload.Request.repository(repo: "o/r", alias: "q", file: nil, sidecarRepo: nil)
            .installsOnlyOnThisMac)
        XCTAssertFalse(ModelDownload.Request.image(alias: "i").installsOnlyOnThisMac)
        let identity = AudioProfileIdentity(
            task: .textToSpeech, alias: "a", repository: "o/a", revision: "r", assetFingerprint: "f")
        XCTAssertFalse(ModelDownload.Request.audio(identity: identity).installsOnlyOnThisMac)
    }
}
