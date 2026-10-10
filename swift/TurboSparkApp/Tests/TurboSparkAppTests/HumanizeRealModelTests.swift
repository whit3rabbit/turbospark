import XCTest
import TurboSpark

@testable import TurboSparkApp

/// The controller on the real `humanizer-12b` install, through the real
/// Swift binding and C boundary. Skipped unless
/// `TURBOSPARK_HUMANIZER_INSTALL_DIR` names the install: it maps 12.7 GB.
@MainActor
final class HumanizeRealModelTests: XCTestCase {
    func testTheControllerRewritesADraftOnTheRealModel() async throws {
        guard let dir = ProcessInfo.processInfo.environment["TURBOSPARK_HUMANIZER_INSTALL_DIR"]
        else { throw XCTSkip("set TURBOSPARK_HUMANIZER_INSTALL_DIR to run on the real model") }

        let controller = HumanizeController()
        controller.installedPath = { dir }
        controller.draft = """
            In conclusion, it is important to note that our team shipped 37 fixes across \
            12 services in March 2024. Moreover, the on-call load dropped by 40 percent, \
            which is truly a testament to the quality of the work.
            """

        controller.run()
        XCTAssertTrue(controller.isBusy, "run() must flip the phase synchronously")
        for _ in 0..<600 where controller.isBusy {
            try await Task.sleep(nanoseconds: 100_000_000)
        }

        XCTAssertNil(controller.errorMessage)
        XCTAssertEqual(controller.stopReason, .eos)
        XCTAssertGreaterThan(controller.rewrite.split(separator: " ").count, 10)
        let audit = try XCTUnwrap(controller.audit)
        print("REWRITE: \(controller.rewrite)")
        print("AUDIT: \(audit)")

        // A second run reuses the resident model and produces a new sample.
        let first = controller.rewrite
        controller.run()
        for _ in 0..<600 where controller.isBusy {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(controller.stopReason, .eos)
        print("RESAMPLE: \(controller.rewrite)")
        XCTAssertNotEqual(controller.rewrite, first, "no seed is sent, so a rerun is a fresh sample")
        controller.unload()
    }

    func testStopDuringARealRewriteKeepsPartialTextAndDropsTheAudit() async throws {
        guard let dir = ProcessInfo.processInfo.environment["TURBOSPARK_HUMANIZER_INSTALL_DIR"]
        else { throw XCTSkip("set TURBOSPARK_HUMANIZER_INSTALL_DIR to run on the real model") }

        let controller = HumanizeController()
        controller.installedPath = { dir }
        controller.draft = String(repeating: "The committee reviewed 14 proposals and approved 3 of them. ", count: 12)
        controller.run()
        // Wait for the first streamed text, then press Stop.
        for _ in 0..<1200 where controller.rewrite.isEmpty && controller.isBusy {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertFalse(controller.rewrite.isEmpty, "no text streamed before the wait ended")
        controller.cancel()
        for _ in 0..<600 where controller.isBusy {
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertEqual(controller.stopReason, .cancelled)
        XCTAssertNil(controller.audit, "a partial rewrite must not be audited")
        XCTAssertFalse(controller.rewrite.isEmpty)
        XCTAssertTrue(controller.canRun, "Stop must leave the controller reusable")
        controller.unload()
    }
}
