import XCTest
@testable import TurboSparkApp

final class ImageDenoiseProgressTests: XCTestCase {
    func testDenoiseStageReportsCompletedStepAndFraction() throws {
        let progress = try XCTUnwrap(ImageDenoiseProgress(stage: "transformer", completed: 3, total: 8))
        XCTAssertEqual(progress.step, 3)
        XCTAssertEqual(progress.total, 8)
        XCTAssertEqual(progress.fraction, 0.375, accuracy: 1e-9)
    }

    func testUnknownTotalIsNil() {
        XCTAssertNil(ImageDenoiseProgress(stage: "transformer", completed: 0, total: 0))
        XCTAssertNil(ImageDenoiseProgress(stage: "transformer", completed: 2, total: -1))
    }

    /// Other stages carry placeholder counts (0 of 1, or file counts while
    /// downloading), which are not denoise steps.
    func testNonDenoiseStagesAreNil() {
        for stage in ["downloading_model", "loading_model", "text_encoder", "vae_decoder", "png_encode"] {
            XCTAssertNil(ImageDenoiseProgress(stage: stage, completed: 1, total: 4), stage)
        }
        XCTAssertNil(ImageDenoiseProgress(stage: nil, completed: 1, total: 4))
    }

    func testOutOfRangeStepIsClamped() throws {
        let over = try XCTUnwrap(ImageDenoiseProgress(stage: "transformer", completed: 12, total: 8))
        XCTAssertEqual(over.step, 8)
        XCTAssertEqual(over.fraction, 1.0)
        let under = try XCTUnwrap(ImageDenoiseProgress(stage: "transformer", completed: -2, total: 8))
        XCTAssertEqual(under.step, 0)
        XCTAssertEqual(under.fraction, 0.0)
    }
}
