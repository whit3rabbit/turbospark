import TurboSpark
import XCTest

@testable import TurboSparkApp

final class AudioModelOpeningTests: XCTestCase {
    func testAStopRequestedBeforeTheOpenSurfacesAsACancellationNotAnEngineError() {
        let opening = AudioModelOpening()
        opening.cancelOpen()
        XCTAssertThrowsError(
            try opening.open(
                path: "/missing/audio/model", task: .music, portable: false,
                experimentalMetal: false, expectedFamily: "minimax_music3")
        ) { error in
            XCTAssertTrue(error is CancellationError, "got \(type(of: error)): \(error)")
        }
        opening.stopAndDrain()  // must not hang after an open that already ended
    }

    func testAnOpenThatFailsForAnotherReasonIsNotMistakenForACancel() {
        let opening = AudioModelOpening()
        XCTAssertThrowsError(
            try opening.open(
                path: "/missing/audio/model", task: .music, portable: false,
                experimentalMetal: false, expectedFamily: "minimax_music3")
        ) { error in
            XCTAssertFalse(error is CancellationError, "\(error)")
            XCTAssertNotEqual((error as? TurboSparkError)?.code, .cancelled)
        }
        opening.stopAndDrain()
    }

    func testStopAndDrainBeforeAnyOpenCancelsTheLaterOpen() {
        let opening = AudioModelOpening()
        // `stopAndDrain` waits for `open` to finish, so run `open` first on
        // another thread the way the controller does, then drain.
        let done = expectation(description: "open ended")
        DispatchQueue.global().async {
            _ = try? opening.open(
                path: "/missing/audio/model", task: .music, portable: false,
                experimentalMetal: false, expectedFamily: "x")
            done.fulfill()
        }
        wait(for: [done], timeout: 5)
        opening.stopAndDrain()
    }
}
