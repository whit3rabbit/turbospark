import XCTest
import TurboSpark

@testable import TurboSparkApp

/// A scripted engine: replays `events` for each call and records what the
/// controller sent.
private final class StubEngine: HumanizeEngine, @unchecked Sendable {
    var events: [GenerationEvent]
    var failure: Error?
    private(set) var drafts: [String] = []
    private(set) var cancelCount = 0
    private(set) var options: [HumanizeOptions] = []

    init(events: [GenerationEvent] = [], failure: Error? = nil) {
        self.events = events
        self.failure = failure
    }

    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        drafts.append(draft)
        self.options.append(options)
        let events = events, failure = failure
        return AsyncThrowingStream { continuation in
            for event in events { continuation.yield(event) }
            continuation.finish(throwing: failure)
        }
    }

    func cancel() { cancelCount += 1 }
}

private func result(
    content: String, stop: String = "eos", audit: String? = nil
) throws -> GenerationResult {
    let auditJSON = audit.map { ",\"audit\":\($0)" } ?? ""
    let json = """
        {"promptTokens":10,"newTokens":5,"prefillSeconds":0.1,"decodeSeconds":0.5,
         "stopReason":"\(stop)","content":"\(content)"\(auditJSON)}
        """
    return try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
}

private let cleanAudit = """
    {"droppedNumbers":[],"inventedNumbers":[],"languageDrift":false,
     "reuseRatio":0.1,"reuseFlagged":false,"warnings":[]}
    """

@MainActor
final class HumanizeControllerTests: XCTestCase {

    private func controller(
        installed: Bool = true, engine: StubEngine
    ) -> (HumanizeController, opens: () -> [String]) {
        let controller = HumanizeController()
        var opened: [String] = []
        controller.countDraftTokens = { text, _ in text.count }
        controller.installedPath = { installed ? "/models/humanizer-12b.gturbo" : nil }
        controller.openEngine = { path in
            opened.append(path)
            return engine
        }
        return (controller, { opened })
    }

    /// Runs the controller's task to completion.
    private func settle(_ controller: HumanizeController) async {
        for _ in 0..<200 where controller.isBusy || controller.phase != .idle {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    func testRunStreamsTheRewriteAndKeepsTheAudit() async throws {
        let engine = StubEngine(events: [
            .prefill(done: 5, total: 10),
            .content("Hello "), .content("there."),
            .finished(try result(content: "Hello there.", audit: cleanAudit)),
        ])
        let (controller, opens) = controller(engine: engine)
        controller.draft = "  Greetings.  "
        XCTAssertTrue(controller.canRun)

        controller.run()
        await settle(controller)

        XCTAssertEqual(controller.rewrite, "Hello there.")
        XCTAssertEqual(controller.stopReason, .eos)
        XCTAssertEqual(controller.audit?.isClean, true)
        XCTAssertNil(controller.errorMessage)
        XCTAssertEqual(opens(), ["/models/humanizer-12b.gturbo"])
        // The controller sends the draft as typed; the library trims and wraps.
        XCTAssertEqual(engine.drafts, ["  Greetings.  "])
    }

    func testTheEngineStaysLoadedAcrossRunsAndUnloadReleasesIt() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "x"))])
        let (controller, opens) = controller(engine: engine)
        controller.draft = "a"

        controller.run()
        await settle(controller)
        controller.run()
        await settle(controller)
        XCTAssertEqual(opens().count, 1, "a second run must reuse the resident model")
        XCTAssertTrue(controller.engineLoaded)

        controller.unload()
        XCTAssertFalse(controller.engineLoaded)
        controller.run()
        await settle(controller)
        XCTAssertEqual(opens().count, 2)
    }

    func testAMissingModelBlocksRunInsteadOfOpeningAnything() {
        let (controller, opens) = controller(installed: false, engine: StubEngine())
        controller.draft = "text"
        XCTAssertFalse(controller.isInstalled)
        XCTAssertFalse(controller.canRun)
        controller.run()
        XCTAssertTrue(opens().isEmpty)
    }

    func testASecondClickBeforeTheTaskStartsDoesNotStartASecondRewrite() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "x"))])
        let (controller, opens) = controller(engine: engine)
        controller.draft = "text"

        controller.run()
        // No await between the two calls: the task has not been scheduled yet.
        XCTAssertFalse(controller.canRun, "a started run must refuse a second one at once")
        controller.run()
        await settle(controller)
        XCTAssertEqual(engine.drafts.count, 1)
        XCTAssertEqual(opens().count, 1)
    }

    func testABlankDraftCannotRun() {
        let (controller, _) = controller(engine: StubEngine())
        controller.draft = " \n\t "
        XCTAssertFalse(controller.canRun)
    }

    func testAnEngineErrorIsShownAndLeavesTheControllerIdle() async {
        struct Boom: LocalizedError { var errorDescription: String? { "the draft is too long" } }
        let engine = StubEngine(failure: Boom())
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"

        controller.run()
        await settle(controller)

        XCTAssertEqual(controller.errorMessage, "the draft is too long")
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertNil(controller.audit)
        XCTAssertTrue(controller.canRun, "a failed run must not wedge the button")
    }

    func testALengthStopIsFlaggedAsPossiblyCutOff() async throws {
        let engine = StubEngine(events: [
            .finished(try result(content: "cut", stop: "maxTokens", audit: cleanAudit))
        ])
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertTrue(controller.wasTruncated)
    }

    func testARerunClearsThePreviousResultBeforeStreaming() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "first", audit: cleanAudit))])
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertEqual(controller.rewrite, "first")

        engine.failure = NSError(domain: "t", code: 1)
        engine.events = []
        controller.run()
        await settle(controller)
        XCTAssertEqual(controller.rewrite, "", "stale text must not survive a failed rerun")
        XCTAssertNil(controller.audit)
    }

    func testCancelReachesTheEngineAndKeepsThePartialText() async throws {
        // Never finishes: the stream yields one delta and then stays open.
        final class Hanging: HumanizeEngine, @unchecked Sendable {
            private(set) var cancelled = false
            func humanize(
                _ draft: String, options: HumanizeOptions
            ) -> AsyncThrowingStream<GenerationEvent, Error> {
                AsyncThrowingStream { $0.yield(.content("partial")) }
            }
            func cancel() { cancelled = true }
        }
        let engine = Hanging()
        let controller = HumanizeController()
        controller.installedPath = { "/m" }
        controller.countDraftTokens = { text, _ in text.count }
        controller.openEngine = { _ in engine }
        controller.draft = "text"

        controller.run()
        for _ in 0..<200 where controller.rewrite.isEmpty {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertEqual(controller.rewrite, "partial")
        controller.cancel()
        await settle(controller)

        XCTAssertTrue(engine.cancelled)
        XCTAssertEqual(controller.rewrite, "partial")
        XCTAssertEqual(controller.stopReason, .cancelled)
        XCTAssertEqual(controller.phase, .idle)
    }

    func testLoadDoesNotGenerateAndCancelledLoadCannotResurrectTheEngine() async throws {
        let engine = StubEngine()
        let (controller, opens) = controller(engine: engine)
        controller.load()
        await settle(controller)
        XCTAssertTrue(controller.engineLoaded)
        XCTAssertEqual(opens().count, 1)
        XCTAssertTrue(engine.drafts.isEmpty)
        XCTAssertTrue(controller.usesModel(at: "/models/humanizer-12b.gturbo"))
        controller.unload()

        var finishOpen: CheckedContinuation<HumanizeEngine, Never>?
        controller.openEngine = { _ in await withCheckedContinuation { finishOpen = $0 } }
        controller.load()
        for _ in 0..<100 where finishOpen == nil { await Task.yield() }
        XCTAssertNotNil(finishOpen)
        controller.unload()
        finishOpen?.resume(returning: engine)
        await settle(controller)
        XCTAssertFalse(controller.engineLoaded, "an open returning after Unload must be discarded")
        XCTAssertFalse(controller.usesModel(at: "/models/humanizer-12b.gturbo"))
    }

    func testRunSnapshotsDraftAndOutputLimitBeforeLoading() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "rewritten"))])
        let (controller, _) = controller(engine: engine)
        controller.draft = "first draft"
        controller.outputLimit = 1024
        controller.run()
        controller.draft = "edited while loading"
        controller.outputLimit = 4096
        await settle(controller)
        XCTAssertEqual(engine.drafts, ["first draft"])
        XCTAssertEqual(engine.options.first?.maxNewTokens, 1024)
        XCTAssertEqual(controller.sourceDraft, "first draft")
        XCTAssertNil(controller.currentBudget, "the old count must not describe the edited draft")
    }

    func testContextOverflowBlocksGenerationEvenBeforeDebouncedPreview() async {
        let engine = StubEngine()
        let (controller, opens) = controller(engine: engine)
        controller.countDraftTokens = { _, _ in 6000 }
        controller.draft = "long text"
        controller.run()
        await settle(controller)
        XCTAssertTrue(opens().isEmpty, "refuse before mapping the model")
        XCTAssertTrue(engine.drafts.isEmpty)
        XCTAssertNotNil(controller.errorMessage)
        XCTAssertFalse(controller.canRun)
        XCTAssertEqual(controller.draft, "long text", "never silently truncate the input")
        controller.outputLimit = 1024
        XCTAssertTrue(controller.canRun, "a smaller output reservation may fit")
    }

    func testPreviewIgnoresAStaleTokenCount() async {
        let (controller, _) = controller(engine: StubEngine())
        var finishCount: CheckedContinuation<Int, Never>?
        controller.countDraftTokens = { _, _ in
            if finishCount == nil {
                return await withCheckedContinuation { finishCount = $0 }
            } else {
                return 100
            }
        }
        controller.draft = "old"
        let preview = Task { await controller.refreshContext() }
        for _ in 0..<100 where finishCount == nil { await Task.yield() }
        controller.draft = "new"
        finishCount?.resume(returning: 8000)
        await preview.value
        XCTAssertNil(controller.currentBudget)
    }

    func testAuditLinesNameEachFindingAndStayQuietWhenClean() throws {
        let dirty = try JSONDecoder().decode(
            HumanizeAudit.self,
            from: Data("""
                {"droppedNumbers":["37","2024"],"inventedNumbers":["9"],"languageDrift":true,
                 "reuseRatio":0.4,"reuseFlagged":true,"warnings":[]}
                """.utf8))
        let lines = HumanizeAuditLines.lines(for: dirty)
        XCTAssertEqual(lines.count, 4)
        XCTAssertTrue(lines[0].contains("37, 2024"), "\(lines)")
        XCTAssertTrue(lines[1].contains("9"), "\(lines)")

        let clean = try JSONDecoder().decode(HumanizeAudit.self, from: Data(cleanAudit.utf8))
        XCTAssertTrue(HumanizeAuditLines.lines(for: clean).isEmpty)
    }
}
