import TurboSpark
import XCTest

@testable import TurboSparkApp

@MainActor
final class VisionSidecarSupportTests: XCTestCase {
    private struct EngineFailure: LocalizedError {
        let errorDescription: String?
    }

    private func row(_ alias: String, kind: String? = nil) -> InstalledModel {
        InstalledModel(alias: alias, repo: "owner/\(alias)", path: "/tmp/\(alias)", family: "llama", kind: kind)
    }

    func testEveryLoadAsksForAnInstalledSidecar() {
        XCTAssertEqual(AppModel().buildOpenOptions().visionSidecar, "auto")
    }

    func testASidecarIsNotALoadableModel() {
        let rows = [row("chat"), row("tower", kind: "vision-tower")]
        XCTAssertEqual(LoadableModelList.filter(installed: rows, live: rows).map(\.alias), ["chat"])
    }

    func testASidecarFailureRetriesTextOnly() async throws {
        var asked: [String?] = []
        var options = OpenOptions()
        options.visionSidecar = VisionSidecar.automatic
        let result = try await VisionSidecar.open(options) { attempt -> Int in
            asked.append(attempt.visionSidecar)
            if attempt.visionSidecar != nil {
                throw EngineFailure(errorDescription: "visionSidecar auto: two towers pair with it")
            }
            return 7
        }
        XCTAssertEqual(result.session, 7)
        XCTAssertEqual(result.skipped, "visionSidecar auto: two towers pair with it")
        XCTAssertEqual(asked, ["auto", nil])
    }

    func testAnyOtherFailureIsNotRetried() async {
        var attempts = 0
        var options = OpenOptions()
        options.visionSidecar = VisionSidecar.automatic
        do {
            _ = try await VisionSidecar.open(options) { _ -> Int in
                attempts += 1
                throw EngineFailure(errorDescription: "not enough memory")
            }
            XCTFail("the failure must reach the caller")
        } catch {
            XCTAssertEqual(error.localizedDescription, "not enough memory")
        }
        XCTAssertEqual(attempts, 1)
    }

    /// The engine decodes the option by this name; a rename would silently
    /// leave every sidecar unattached.
    func testTheOptionReachesTheEngineByName() throws {
        var options = OpenOptions()
        options.visionSidecar = VisionSidecar.automatic
        let json = String(decoding: try JSONEncoder().encode(options), as: UTF8.self)
        XCTAssertTrue(json.contains("\"visionSidecar\":\"auto\""), json)
    }
}
