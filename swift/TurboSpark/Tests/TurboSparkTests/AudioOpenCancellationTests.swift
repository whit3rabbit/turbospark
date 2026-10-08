import XCTest

@testable import TurboSpark

final class AudioOpenCancellationTests: XCTestCase {
    private var fixture: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("crates/audio/testdata/minimax_music3/converted_plain").path
    }

    func testACancelledTokenStopsTheOpenWithTheCancelCode() throws {
        let token = try AudioOpenToken()
        token.cancel()
        token.cancel()  // idempotent
        XCTAssertThrowsError(
            try AudioSession(
                modelPath: "/missing/audio/model", task: .music, openToken: token, onProgress: nil)
        ) { error in
            XCTAssertEqual((error as? TurboSparkError)?.code, .cancelled, "\(error)")
        }
    }

    func testAnUncancelledTokenOpensAndReportsNoProgressForAnUnmanagedFolder() throws {
        final class Count: @unchecked Sendable { var n = 0 }
        let count = Count()
        let token = try AudioOpenToken()
        let session = try AudioSession(
            modelPath: fixture, task: .music, openToken: token, onProgress: { _ in count.n += 1 })
        XCTAssertEqual(count.n, 0, "a folder you chose is not verified, so there is nothing to report")
        session.close()
    }

    func testTheOriginalInitStillWorksAndAnOpenWithNoTokenIsUnchanged() throws {
        let session = try AudioSession(modelPath: fixture, task: .music)
        session.close()
        XCTAssertThrowsError(try AudioSession(modelPath: "/missing/audio/model", task: .music)) {
            XCTAssertNotEqual(($0 as? TurboSparkError)?.code, .cancelled)
        }
    }

    func testCancelIsCallableFromAnotherThread() throws {
        // Only proves `cancel` returns promptly off the main thread. That it
        // also works DURING a verification is covered a layer down, where the
        // hash is stopped inside a file (`crates/model-io` sha256 tests).
        let token = try AudioOpenToken()
        let done = expectation(description: "cancel returned")
        DispatchQueue.global().async { token.cancel(); done.fulfill() }
        wait(for: [done], timeout: 2)
    }
}
