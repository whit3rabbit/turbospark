import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The trait-extraction pipeline's pure half: progress parsing, the CLI
/// locator's candidate order, output naming, and preset assembly. The
/// process half (the run itself) is the CLI's own real-install gate, the
/// same split `AppSteeringPolicy` draws for the same reason: assertions
/// against a live model never run in CI.
final class SteeringTraitExtractionTests: XCTestCase {
    // MARK: progress parsing

    func testProgressParserReadsBothSidesAndIgnoresEverythingElse() {
        let positive = SteeringTraitProgress.parse(
            "  [positive] capturing prompt 9/24")
        XCTAssertEqual(
            positive,
            SteeringTraitProgress.Snapshot(side: .positive, done: 9, total: 24))
        let negative = SteeringTraitProgress.parse(
            "[negative] capturing prompt 24/24")
        XCTAssertEqual(
            negative,
            SteeringTraitProgress.Snapshot(side: .negative, done: 24, total: 24))

        // The extractor's own non-progress lines must not parse.
        XCTAssertNil(SteeringTraitProgress.parse("extracting 'refusals' direction"))
        XCTAssertNil(SteeringTraitProgress.parse("[unknown] capturing prompt 1/4"))
        XCTAssertNil(SteeringTraitProgress.parse("[positive] wrote 24 snapshots"))
        XCTAssertNil(SteeringTraitProgress.parse("[positive] capturing prompt x/24"))
        XCTAssertNil(SteeringTraitProgress.parse(""))
    }

    func testOverallFractionAccumulatesAcrossSidesFromLatestSnapshots() throws {
        // Both sides complete at the same count: 1.0 exactly.
        XCTAssertEqual(
            try XCTUnwrap(SteeringTraitProgress.overallFraction(
                positive: .init(side: .positive, done: 24, total: 24),
                negative: .init(side: .negative, done: 24, total: 24))),
            1.0, accuracy: 1e-9)
        // Halfway through the first side only: half of the whole run.
        XCTAssertEqual(
            try XCTUnwrap(SteeringTraitProgress.overallFraction(
                positive: .init(side: .positive, done: 12, total: 24),
                negative: nil)),
            0.25, accuracy: 1e-9)
        // A side overshooting its own total clamps.
        XCTAssertEqual(
            try XCTUnwrap(SteeringTraitProgress.overallFraction(
                positive: .init(side: .positive, done: 30, total: 24),
                negative: .init(side: .negative, done: 24, total: 24))),
            1.0, accuracy: 1e-9)
        // Nothing known: nil, which is the indeterminate-bar signal.
        XCTAssertNil(SteeringTraitProgress.overallFraction(positive: nil, negative: nil))
        // A zero-total snapshot cannot divide.
        XCTAssertNil(
            SteeringTraitProgress.overallFraction(
                positive: .init(side: .positive, done: 0, total: 0), negative: nil))
    }

    // MARK: CLI locator

    func testCandidateURLsOrderBundleFirstThenPathThenInstallPrefixes() {
        let urls = SteeringTraitTool.candidateURLs(
            bundleDirectory: "/Applications/TurboSpark.app/Contents/MacOS",
            path: "/usr/bin:/opt/homebrew/bin",
            home: "/Users/tester")
        let paths = urls.map(\.path)
        XCTAssertEqual(paths.count, 7)
        XCTAssertEqual(paths[0], "/Applications/TurboSpark.app/Contents/MacOS/turbospark-steer")
        // PATH entries in order.
        XCTAssertEqual(paths[1], "/usr/bin/turbospark-steer")
        XCTAssertEqual(paths[2], "/opt/homebrew/bin/turbospark-steer")
        // The two make-install prefixes, then the two Homebrew prefixes a
        // GUI app's minimal launchd PATH omits.
        XCTAssertEqual(paths[3], "/Users/tester/.cargo/bin/turbospark-steer")
        XCTAssertEqual(paths[4], "/Users/tester/.local/bin/turbospark-steer")
        XCTAssertEqual(paths[5], "/opt/homebrew/bin/turbospark-steer")
        XCTAssertEqual(paths[6], "/usr/local/bin/turbospark-steer")
    }

    func testCandidateURLsWithoutBundleOrPathStillNameFallbacks() {
        let urls = SteeringTraitTool.candidateURLs(
            bundleDirectory: nil, path: nil, home: "/Users/tester")
        XCTAssertEqual(urls.count, 4)
        XCTAssertEqual(urls.first?.path, "/Users/tester/.cargo/bin/turbospark-steer")
    }

    // MARK: output naming

    func testOutputURLIsDeterministicAndUnderTheProfileDirectory() {
        let a = SteeringTraitPresetFactory.outputURL(modelAlias: "tinyllama", traitID: "refusals")
        let b = SteeringTraitPresetFactory.outputURL(modelAlias: "tinyllama", traitID: "refusals")
        XCTAssertEqual(a, b, "re-extraction overwrites its own file")
        XCTAssertEqual(a.lastPathComponent, "tinyllama-refusals.gguf")
        XCTAssertTrue(a.path.contains("steering-vectors"), a.path)
    }

    func testSanitizationCollapsesSeparatorsAndFallsBack() {
        XCTAssertEqual(SteeringTraitPresetFactory.sanitized("Qwen 3.8 27B"), "qwen-3-8-27b")
        XCTAssertEqual(SteeringTraitPresetFactory.sanitized("///"), "model")
        XCTAssertEqual(SteeringTraitPresetFactory.sanitized("MIXTRAL"), "mixtral")
        XCTAssertEqual(SteeringTraitPresetFactory.sanitized("--a--b--"), "a-b")
    }

    // MARK: pack report decoding

    func testPackReportDecodesCLIJSONAndDefaultsUnknownModes() throws {
        let json = Data("""
        [
          {"id": "refusals", "title": "Refusal suppression",
           "summary": "s", "positivePrompts": 24, "negativePrompts": 24,
           "evalPrompts": 8, "recommendedMode": "ablate",
           "recommendedScale": 0.3},
          {"id": "future", "title": "Future pack", "summary": "s",
           "positivePrompts": 4, "negativePrompts": 4, "evalPrompts": 3,
           "recommendedMode": "teleport", "recommendedScale": 0.5}
        ]
        """.utf8)
        let reports = try JSONDecoder().decode([SteeringTraitPackReport].self, from: json)
        XCTAssertEqual(reports.count, 2)
        XCTAssertEqual(reports[0].presetMode, .ablate)
        // An unknown mode spelling falls back to the engine's own default
        // rather than failing the sheet for a pack it cannot steer anyway.
        XCTAssertEqual(reports[1].presetMode, .ablate)
        XCTAssertEqual(reports[1].recommendedScale, 0.5)
    }

    // MARK: preset assembly

    /// The factory must reuse an existing preset that already points at the
    /// same output file, so re-extracting a trait updates the row the user
    /// had instead of stacking duplicates. The shape read underneath hits a
    /// nonexistent file here and must degrade to "unknown shape", not throw.
    @MainActor
    func testMakePresetReusesThePresetAlreadyPointingAtTheOutputFile() throws {
        let output = SteeringTraitPresetFactory.outputURL(
            modelAlias: "tinyllama", traitID: "refusals")
        let existing = AppSteeringPreset(
            name: "old name",
            vectorPath: output.path,
            mode: .add,
            scale: 0.9)
        let report = SteeringTraitPackReport(
            id: "refusals", title: "Refusal suppression", summary: "s",
            positivePrompts: 24, negativePrompts: 24, evalPrompts: 8,
            recommendedMode: "ablate", recommendedScale: 0.3)
        let preset = SteeringTraitPresetFactory.makePreset(
            traitID: "refusals",
            traitName: "Refusal suppression",
            report: report,
            scale: 0.3,
            outputURL: output,
            modelAlias: "tinyllama",
            existing: [existing])
        XCTAssertEqual(preset.id, existing.id, "same file, same row")
        XCTAssertEqual(preset.mode, .ablate, "pack recommendation wins")
        XCTAssertEqual(preset.scale, 0.3)
        XCTAssertEqual(preset.vectorPath, output.path)
        // The shape read failed (no file in the test sandbox): cleared to
        // unknown, never zero-width, per AppSteeringPreset's nil contract.
        XCTAssertNil(preset.vectorHidden)
        XCTAssertNil(preset.vectorSpannedLayers)
        XCTAssertTrue(preset.notes.contains("tinyllama"), preset.notes)
    }

    @MainActor
    func testMakePresetCreatesANewRowForANewFile() throws {
        let output = SteeringTraitPresetFactory.outputURL(
            modelAlias: "tinyllama", traitID: "slop")
        let unrelated = AppSteeringPreset(
            name: "other",
            vectorPath: "/elsewhere/vector.gguf",
            mode: .ablate,
            scale: 0.3)
        let report = SteeringTraitPackReport(
            id: "slop", title: "Slop reduction", summary: "s",
            positivePrompts: 14, negativePrompts: 14, evalPrompts: 8,
            recommendedMode: "ablate", recommendedScale: 0.3)
        let preset = SteeringTraitPresetFactory.makePreset(
            traitID: "slop",
            traitName: "Slop reduction",
            report: report,
            scale: 0.3,
            outputURL: output,
            modelAlias: "tinyllama",
            existing: [unrelated])
        XCTAssertNotEqual(preset.id, unrelated.id)
        XCTAssertEqual(preset.name, "Slop reduction")
    }
}
