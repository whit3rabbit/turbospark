import XCTest
@testable import TurboSparkApp

/// Regression test for state#11: `setModelURL`'s `defer { opening = false }`
/// was attached to the SYNCHRONOUS function body, not to the `Task` doing
/// the actual (async) work, so it fired the instant `setModelURL` returned --
/// immediately after spawning the Task -- rather than when the model
/// finished (successfully or not) loading. The loading indicator never had
/// a chance to render.
@MainActor
final class ModelLoadingStateTests: XCTestCase {
    func testOpeningStaysTrueUntilTheLoadActuallyFinishes() async throws {
        let appModel = AppModel()
        XCTAssertFalse(appModel.opening)

        // A path with no real checkpoint at it: `TurboSparkSession(modelPath:options:)`
        // will fail quickly, but the point is what happens BEFORE that
        // failure is observed, not after.
        let bogusPath = FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).gturbo")

        appModel.setModelURL(bogusPath)

        // Immediately after `setModelURL` returns (before the spawned Task
        // has had any chance to run), `opening` must still be true. Under
        // the old code the `defer` had already fired by this point.
        XCTAssertTrue(appModel.opening, "opening must stay true immediately after setModelURL returns, until the async load actually completes.")

        let deadline = Date().addingTimeInterval(10)
        while appModel.opening && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(appModel.opening, "opening must become false once the (failed) load completes.")
        XCTAssertNotNil(appModel.error, "A bogus path should fail to open and surface an error.")
    }
}
