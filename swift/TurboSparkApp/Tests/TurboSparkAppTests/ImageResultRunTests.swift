import Foundation
import XCTest
import TurboSpark
@testable import TurboSparkApp

final class ImageResultRunTests: XCTestCase {
    // MARK: - Vary seed

    func testVarySeedUsesTheFreshRandomSeed() {
        XCTAssertEqual(ImageVariation.seed(replacing: 42, random: { 7 }), 7)
    }

    func testVarySeedNeverRepeatsTheSavedSeed() {
        // A repeat would make Vary render exactly what Regenerate renders.
        XCTAssertEqual(ImageVariation.seed(replacing: 42, random: { 42 }), 43)
        XCTAssertEqual(ImageVariation.seed(replacing: UInt64.max, random: { UInt64.max }), 0)
    }

    func testVaryRequestKeepsEverythingButTheSeed() {
        let saved = AppImageRequest(options: ImageGenerateOptions(
            prompt: "kite", seed: 42, width: 768, height: 512, steps: 7))
        let varied = ImageVariation.options(from: saved, random: { 9 })
        XCTAssertEqual(varied.seed, 9)
        XCTAssertEqual(varied.prompt, "kite")
        XCTAssertEqual(varied.width, 768)
        XCTAssertEqual(varied.height, 512)
        XCTAssertEqual(varied.steps, 7)
    }

    // MARK: - Latest run

    private let chat = UUID()
    private let start = Date(timeIntervalSince1970: 1_000_000)

    /// Mirrors `startImageGeneration`: one artifact per request, seeds from
    /// `ImageGenerationSequence`, saved a little apart. Returned newest
    /// first, like `savedImageArtifacts`.
    private func batch(
        prompt: String = "kite", seed: UInt64 = 10, count: Int,
        at offset: TimeInterval = 0, chatID: UUID? = nil
    ) -> [AppArtifact] {
        let options = ImageGenerateOptions(prompt: prompt, seed: seed, width: 512, height: 512, steps: 4)
        return ImageGenerationSequence.requests(options: options, count: count).enumerated().map { index, request in
            let when = start.addingTimeInterval(offset + Double(index) * 30)
            return AppArtifact(
                chatID: chatID ?? chat, path: "/tmp/\(UUID()).png", title: "Generated image",
                origin: .imageGeneration, createdAt: when, updatedAt: when,
                imageRequest: AppImageRequest(options: request))
        }.reversed()
    }

    func testLatestRunGroupsOneBatchInRunOrder() throws {
        let images = batch(count: 3)
        let run = try XCTUnwrap(ImageResultRun.latest(in: images))
        XCTAssertEqual(run.map(\.imageRequest?.seed), [10, 11, 12])
    }

    func testSingleImageIsNotARun() {
        XCTAssertNil(ImageResultRun.latest(in: batch(count: 1)))
        XCTAssertNil(ImageResultRun.latest(in: []))
    }

    func testLatestRunStopsAtAnEarlierRunOfTheSamePrompt() throws {
        // Two runs of the same prompt with a fixed seed: the newer one alone.
        let older = batch(seed: 10, count: 2)
        let newer = batch(seed: 10, count: 2, at: 120)
        let run = try XCTUnwrap(ImageResultRun.latest(in: newer + older))
        XCTAssertEqual(run.map(\.id), newer.reversed().map(\.id))
    }

    func testRegenerateOrVaryAfterABatchIsNotARun() {
        let images = batch(count: 2)
        // Regenerate reuses the newest seed (12 would chain; 11 does not).
        let regenerated = batch(seed: 11, count: 1, at: 300)
        XCTAssertNil(ImageResultRun.latest(in: regenerated + images))
        let varied = batch(seed: 999, count: 1, at: 300)
        XCTAssertNil(ImageResultRun.latest(in: varied + images))
    }

    func testRunNeedsMatchingPromptSizeChatAndTime() {
        let a = batch(prompt: "kite", seed: 10, count: 1)
        XCTAssertNil(ImageResultRun.latest(in: batch(prompt: "boat", seed: 11, count: 1, at: 30) + a))
        XCTAssertNil(ImageResultRun.latest(in: batch(seed: 11, count: 1, at: 30, chatID: UUID()) + a))
        // Consecutive seeds an hour apart are two separate generations.
        XCTAssertNil(ImageResultRun.latest(in: batch(seed: 11, count: 1, at: 3600) + a))
    }

    func testRunWithoutRequestMetadataIsNotARun() {
        let bare = AppArtifact(chatID: chat, path: "/tmp/a.png", title: "Image", origin: .imageGeneration)
        XCTAssertNil(ImageResultRun.latest(in: [bare] + batch(count: 2)))
    }

    func testRunIsCappedAtTheBatchLimit() throws {
        // Seeds 10...17 saved back to back can only be two runs of four.
        let images = batch(seed: 10, count: 4, at: 0)
        let next = batch(seed: 14, count: 4, at: 120)
        let run = try XCTUnwrap(ImageResultRun.latest(in: next + images))
        XCTAssertEqual(run.count, ImageGenerationSequence.maxCount)
        XCTAssertEqual(run.map(\.imageRequest?.seed), [14, 15, 16, 17])
    }
}
