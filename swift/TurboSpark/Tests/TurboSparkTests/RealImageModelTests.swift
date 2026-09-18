import Foundation
import XCTest

@testable import TurboSpark

/// End-to-end image tests against a verified `.image.gturbo` install.
///
/// These are intentionally environment-gated because a real run opens the
/// packed components and executes the Metal pipeline for the full nine-step
/// production request.
final class RealImageModelTests: XCTestCase {
    private func modelPath() throws -> String {
        guard let raw = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_IMAGE_MODEL"],
              !raw.isEmpty
        else {
            throw XCTSkip(
                "set TURBOSPARK_TEST_IMAGE_MODEL to a verified .image.gturbo install to run this")
        }
        return raw
    }

    func testGeneratesAPngAndReturnsRuntimeMetadata() async throws {
        let session = try await TurboSparkImageSession(modelPath: try modelPath())
        let options = ImageGenerateOptions(prompt: "a red kite over a green field", seed: 42)
        var stages: [String] = []
        var result: ImageGenerationResult?

        for try await event in session.generate(options) {
            switch event {
            case let .stage(name, _, _): stages.append(name)
            case let .finished(value): result = value
            case .cancelled: XCTFail("a normal image generation should not cancel")
            }
        }

        let image = try XCTUnwrap(result)
        XCTAssertEqual(image.metadata.prompt, options.prompt)
        XCTAssertEqual(image.metadata.seed, options.seed)
        XCTAssertEqual(image.metadata.schedulerSteps, options.steps)
        XCTAssertEqual(Array(image.png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
        XCTAssertTrue(stages.contains("text_encoder"))
        XCTAssertTrue(stages.contains("transformer"))
        XCTAssertTrue(stages.contains("vae_decoder"))
        XCTAssertTrue(stages.contains("png_encode"))
    }

    func testCancellingARealImageJobReturnsCancelledWithoutAPng() async throws {
        let session = try await TurboSparkImageSession(modelPath: try modelPath())
        let options = ImageGenerateOptions(prompt: "a quiet lake at dusk", seed: 43)
        var cancelled = false
        var finished = false

        for try await event in session.generate(options) {
            switch event {
            case .stage:
                session.cancel()
            case .cancelled:
                cancelled = true
            case .finished:
                finished = true
            }
        }

        XCTAssertTrue(cancelled, "cancellation must surface as an image event")
        XCTAssertFalse(finished, "a cancelled image job must not publish a PNG")
    }
}
