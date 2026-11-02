import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The trait-extraction pipeline: progress parsing, the CLI locator's
/// candidate order, output naming, preset assembly, and the child-process
/// lifecycle (failure text, fast exit, cancellation) driven by stub shell
/// scripts. The real extraction is the CLI's own real-install gate: no
/// assertion against a live model runs in CI.
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

    // MARK: composer menu

    /// A trait already extracted for this install is a Directions row, so
    /// the Extract section must not offer it again; other installs' vectors
    /// do not hide it.
    func testUnextractedTraitsSkipTheOnesThisInstallAlreadyHas() {
        let mine = AppSteeringPreset(
            name: "Refusal suppression",
            vectorPath: SteeringTraitPresetFactory.outputURL(
                modelAlias: "tinyllama", traitID: "refusals").path,
            mode: .ablate,
            scale: 0.3)
        let other = AppSteeringPreset(
            name: "Slop reduction",
            vectorPath: SteeringTraitPresetFactory.outputURL(
                modelAlias: "qwen38-27b", traitID: "slop").path,
            mode: .ablate,
            scale: 0.3)
        let traits = SteeringComposerMenuModel.unextractedTraits(
            presets: [mine, other], alias: "tinyllama")
        XCTAssertEqual(traits, [.slop, .humor, .sycophancy, .emotions])
    }

    /// Rows keep the registered order, check exactly the active preset only
    /// while steering is on, and leave a preset whose shape was never read
    /// pickable (the engine refuses at open with the numbers).
    func testPresetRowsMarkOnlyTheActivePresetWhileEnabled() {
        let a = AppSteeringPreset(name: "A", vectorPath: "/v/a.gguf", mode: .ablate, scale: 0.3)
        let b = AppSteeringPreset(name: "B", vectorPath: "/v/b.gguf", mode: .ablate, scale: 0.3)
        let on = SteeringComposerMenuModel.presetRows(
            presets: [a, b], activeID: b.id, enabled: true, target: nil)
        XCTAssertEqual(on.map(\.name), ["A", "B"])
        XCTAssertEqual(on.map(\.isActive), [false, true])
        XCTAssertTrue(on.allSatisfy(\.canEnable))
        let off = SteeringComposerMenuModel.presetRows(
            presets: [a, b], activeID: b.id, enabled: false, target: nil)
        XCTAssertFalse(off.contains(where: \.isActive))
    }

    /// A preset with no vector file steers nothing and cannot be picked.
    func testPresetWithoutAVectorCannotBeEnabled() {
        let empty = AppSteeringPreset(name: "Empty", vectorPath: "", mode: .ablate, scale: 0.3)
        let rows = SteeringComposerMenuModel.presetRows(
            presets: [empty], activeID: nil, enabled: false, target: nil)
        XCTAssertEqual(rows.first?.canEnable, false)
        XCTAssertNotNil(rows.first?.blockReason)
    }

    // MARK: process lifecycle

    /// A stub tool: a shell script that ignores the extract arguments.
    private func stubTool(_ body: String) throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("steer-stub-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("turbospark-steer")
        try Data("#!/bin/sh\n\(body)\n".utf8).write(to: url)
        try FileManager.default.setAttributes(
            [.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private func runExtract(
        _ controller: SteeringTraitExtractionController, tool: URL
    ) async throws {
        try await controller.extract(
            cliURL: tool,
            modelPath: "/nonexistent/model",
            traitID: "refusals",
            outputURL: FileManager.default.temporaryDirectory
                .appendingPathComponent("steer-stub-out-\(UUID().uuidString).gguf"))
    }

    /// `consume` drops every complete line from its partial-line buffer, so
    /// the failure tail must come from the kept lines, and progress lines
    /// must not crowd the refusal out of it.
    func testTailKeepsTheRefusalAfterItsLineWasConsumed() {
        let buffer = ProgressLineBuffer { _ in }
        buffer.consume(Data("  [positive] capturing prompt 1/24\n".utf8))
        buffer.consume(Data("error: family Foo does not dispatch steering\n".utf8))
        let tail = buffer.tail(lines: 6)
        XCTAssertTrue(tail.contains("does not dispatch steering"), tail)
        XCTAssertFalse(tail.contains("capturing prompt"), tail)
    }

    /// A tool that refuses at once must surface ITS message, and the wait
    /// must not hang on an exit that beat the termination handler.
    @MainActor
    func testFastFailingToolReportsItsOwnErrorText() async throws {
        let tool = try stubTool("echo 'error: refused for a named reason' >&2\nexit 3")
        let controller = SteeringTraitExtractionController()
        do {
            try await runExtract(controller, tool: tool)
            XCTFail("a non-zero exit must throw")
        } catch let SteeringTraitError.toolFailed(status, message) {
            XCTAssertEqual(status, 3)
            XCTAssertTrue(message.contains("refused for a named reason"), message)
        }
    }

    /// Cancelling mid-run terminates the child and throws
    /// `CancellationError` instead of waiting the run out.
    @MainActor
    func testCancellingMidRunTerminatesTheTool() async throws {
        let tool = try stubTool("exec sleep 30")
        let controller = SteeringTraitExtractionController()
        let started = Date()
        let task = Task { @MainActor in
            try await self.runExtract(controller, tool: tool)
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()
        do {
            try await task.value
            XCTFail("a cancelled run must throw")
        } catch is CancellationError {
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
    }

    /// A task cancelled before `extract` runs launches nothing and does not
    /// trip the unlaunched-process exception in the cancel handler.
    @MainActor
    func testPreCancelledTaskLaunchesNothing() async throws {
        let marker = FileManager.default.temporaryDirectory
            .appendingPathComponent("steer-stub-marker-\(UUID().uuidString)")
        let tool = try stubTool("touch '\(marker.path)'")
        let controller = SteeringTraitExtractionController()
        let task = Task { @MainActor in
            withUnsafeCurrentTask { $0?.cancel() }
            try await self.runExtract(controller, tool: tool)
        }
        do {
            try await task.value
            XCTFail("a pre-cancelled run must throw")
        } catch is CancellationError {
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
    }
}
