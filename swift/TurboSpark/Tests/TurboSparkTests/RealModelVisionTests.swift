import Foundation
import XCTest

@testable import TurboSpark

/// End-to-end vision tests against a real install with a vision tower.
final class RealModelVisionTests: RealModelTestCase {

    /// **THE END-TO-END VISION ARM, AND ITS LOAD-BEARING ASSERTION IS THE
    /// TOKEN COUNT RATHER THAN THE TEXT.**
    ///
    /// A dropped image is the failure mode this whole path has, and it does
    /// not error: the prompt length agrees, nothing warns, and the model
    /// answers fluently about a picture it was never shown. That is measured
    /// history rather than caution -- `crates/cli` Gotcha 13's first
    /// end-to-end `--image` run transcribed a page it had not seen, with
    /// every shape and length in agreement.
    ///
    /// So this compares the SAME prompt with and without the picture. The
    /// template renders one `<|image_pad|>` marker either way, and the
    /// engine's splice expands that marker to the page's merged-token count
    /// -- hundreds of positions. If the image were silently dropped, the two
    /// prompts would be within a token or two of each other. Nothing about
    /// the page's content has to be known for that to discriminate, which is
    /// what makes it immune to where the transcription happens to stop (the
    /// trap that cost `vision_memory_oracle` two marker designs).
    func testAnImageReachesTheModelAndLengthensThePrompt() async throws {
        let image = try testImagePath()
        let session = try await TurboSparkSession(modelPath: try modelPath())

        // CHECKED, not assumed: a text-only install passes every other line
        // of this test while proving nothing, which is the fixture-must-
        // discriminate rule applied to an env var (Gotcha 11).
        guard session.info.vision.active else {
            return XCTFail(
                "TURBOSPARK_TEST_MODEL points at an install that cannot serve images"
                    + (session.info.vision.reason.map { " (\($0))" } ?? "")
                    + "; point it at one with a vision tower, e.g. qwen38-27b-vision.gturbo")
        }
        XCTAssertNotNil(
            session.info.vision.imageTokenId,
            "an active tower reports the marker id it splices at")

        var options = GenerateOptions()
        options.maxNewTokens = 60
        options.seed = 20260721
        let question = "Transcribe the first line of this page."

        func run(_ message: ChatMessage) async throws -> GenerationResult {
            var result: GenerationResult?
            for try await event in session.generate([message], options: options) {
                if case .finished(let r) = event { result = r }
            }
            return try XCTUnwrap(result)
        }

        let withImage = try await run(
            ChatMessage(role: .user, content: question, images: [.path(image)]))
        let textOnly = try await run(ChatMessage(role: .user, content: question))

        // The whole assertion: a page is worth hundreds of positions, so a
        // dropped image cannot pass this however plausible its answer reads.
        XCTAssertGreaterThan(
            withImage.promptTokens, textOnly.promptTokens + 100,
            "the image prompt is \(withImage.promptTokens) tokens against "
                + "\(textOnly.promptTokens) text-only; the splice did not expand the marker, "
                + "so the picture never reached the model")
        XCTAssertFalse(withImage.content.isEmpty)

        // Printed so a run can be READ. A vision feature needs an arm whose
        // output a person can check against the page: shapes and lengths all
        // agreed in the bug this test exists for.
        print("vision: \(withImage.promptTokens) prompt tokens (text-only \(textOnly.promptTokens))")
        print("vision transcription: \(withImage.content.prefix(300))")
    }

    /// An image sent to an install that cannot serve one is refused BY NAME
    /// rather than dropped, which is the difference between a user seeing a
    /// message and a user seeing a confident answer about nothing.
    func testAnImageIsRefusedByNameWhenTheInstallHasNoTower() async throws {
        let session = try await TurboSparkSession(modelPath: try blockedModelPath())
        guard !session.info.vision.active else {
            throw XCTSkip(
                "TURBOSPARK_TEST_MODEL_NO_SPECULATION happens to carry a vision tower; "
                    + "this case needs an install without one")
        }
        var options = GenerateOptions()
        options.maxNewTokens = 8
        do {
            for try await _ in session.generate(
                [ChatMessage(role: .user, content: "What is this?", images: [.path("/nope.png")])],
                options: options)
            {}
            XCTFail("an image on a tower-less install should be refused")
        } catch {
            let message = "\(error)".lowercased()
            XCTAssertTrue(
                message.contains("vision") || message.contains("image"),
                "the refusal should name what was wrong, got: \(error)")
        }
    }
}
