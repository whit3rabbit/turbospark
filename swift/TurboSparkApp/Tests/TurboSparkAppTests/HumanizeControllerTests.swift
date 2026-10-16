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

/// Rules persist in defaults, so every controller here gets its own suite.
private func freshDefaults() -> UserDefaults {
    UserDefaults(suiteName: "humanize.tests.\(UUID().uuidString)")!
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
        let controller = HumanizeController(defaults: freshDefaults())
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
        XCTAssertEqual(controller.checks?.isClean, true)
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
        XCTAssertNil(controller.checks)
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
        XCTAssertNil(controller.checks)
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
        let controller = HumanizeController(defaults: freshDefaults())
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

    func testClearRewriteDropsTheResultAndItsNoticesButKeepsTheDraft() async throws {
        // A length stop leaves a notice, and the audit leaves checks and stats:
        // all of it belongs to the rewrite and must go with it.
        let engine = StubEngine(events: [
            .content("cut"),
            .finished(try result(content: "cut", stop: "maxTokens", audit: cleanAudit)),
        ])
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertEqual(controller.rewrite, "cut")
        XCTAssertNotNil(controller.checks)
        XCTAssertNotNil(controller.stats)
        XCTAssertTrue(controller.wasTruncated)
        XCTAssertFalse(controller.segments.isEmpty)

        controller.clearRewrite()

        XCTAssertEqual(controller.rewrite, "")
        XCTAssertEqual(controller.sourceDraft, "")
        XCTAssertTrue(controller.segments.isEmpty)
        XCTAssertNil(controller.checks)
        XCTAssertNil(controller.ruleFindings)
        XCTAssertNil(controller.stats)
        XCTAssertNil(controller.stopReason)
        XCTAssertFalse(controller.wasTruncated)
        XCTAssertEqual(controller.draft, "text", "Clear Rewrite never touches the draft")
        XCTAssertTrue(controller.canRun, "the draft can be rewritten again")
    }

    func testClearRewriteAlsoDropsAnErrorFromTheLastRun() async {
        let engine = StubEngine(failure: NSError(domain: "t", code: 1))
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertNotNil(controller.errorMessage)

        controller.clearRewrite()

        XCTAssertNil(controller.errorMessage)
    }

    func testClearAllEmptiesBothCards() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "first", audit: cleanAudit))])
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertEqual(controller.rewrite, "first")

        controller.clearAll()

        XCTAssertEqual(controller.draft, "")
        XCTAssertEqual(controller.rewrite, "")
        XCTAssertNil(controller.checks)
        XCTAssertFalse(controller.canRun, "an empty draft cannot run")
    }

    func testClearsAreRefusedWhileARewriteRuns() async throws {
        // A running task keeps writing into the rewrite, so a clear would be
        // refilled by the next delta and the draft would vanish mid-run.
        final class Hanging: HumanizeEngine, @unchecked Sendable {
            func humanize(
                _ draft: String, options: HumanizeOptions
            ) -> AsyncThrowingStream<GenerationEvent, Error> {
                AsyncThrowingStream { $0.yield(.content("partial")) }
            }
            func cancel() {}
        }
        let controller = HumanizeController(defaults: freshDefaults())
        controller.installedPath = { "/m" }
        controller.countDraftTokens = { text, _ in text.count }
        controller.openEngine = { _ in Hanging() }
        controller.draft = "text"
        controller.run()
        for _ in 0..<200 where controller.rewrite.isEmpty {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        XCTAssertTrue(controller.isBusy)

        controller.clearRewrite()
        controller.clearAll()

        XCTAssertEqual(controller.rewrite, "partial")
        XCTAssertEqual(controller.draft, "text")

        controller.cancel()
        await settle(controller)
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

    func testRunSnapshotsTheDraftBeforeLoading() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "rewritten"))])
        let (controller, _) = controller(engine: engine)
        controller.draft = "first draft"
        controller.run()
        controller.draft = "edited while loading"
        await settle(controller)
        XCTAssertEqual(engine.drafts, ["first draft"])
        XCTAssertNil(engine.options.first?.maxNewTokens, "parts are sized for the automatic budget")
        XCTAssertEqual(controller.sourceDraft, "first draft")
        XCTAssertNil(controller.currentPlan, "the old count must not describe the edited draft")
    }

    func testASinglePassageTooLongForTheWindowIsRefusedBeforeLoading() async {
        let engine = StubEngine()
        let (controller, opens) = controller(engine: engine)
        // Every paragraph and sentence counts as 6,000 tokens, so no split fits.
        controller.countDraftTokens = { _, _ in 6000 }
        controller.draft = "long text"
        controller.run()
        await settle(controller)
        XCTAssertTrue(opens().isEmpty, "refuse before mapping the model")
        XCTAssertTrue(engine.drafts.isEmpty)
        XCTAssertNotNil(controller.errorMessage)
        XCTAssertFalse(controller.canRun)
        XCTAssertEqual(controller.draft, "long text", "never silently truncate the input")
    }

    func testALongDraftIsRewrittenInPartsAndJoinedWithItsOwnBreaks() async throws {
        let engine = EchoEngine()
        let (controller, _) = controller(engine: StubEngine())
        controller.openEngine = { _ in engine }
        // One token per character: each paragraph fits a part, two do not.
        let paragraphs = ["alpha", "beta", "gamma"].map { String(repeating: "\($0) ", count: 90).trimmingCharacters(in: .whitespaces) }
        controller.draft = paragraphs[0] + "\n\n" + paragraphs[1] + "\n \n\n" + paragraphs[2]
        XCTAssertGreaterThan(controller.draft.count, HumanizePlanner.partTokens)

        controller.run()
        await settle(controller)

        XCTAssertNil(controller.errorMessage)
        XCTAssertEqual(engine.drafts, paragraphs, "each part is sent alone, trimmed of its breaks")
        XCTAssertEqual(controller.rewrite, "ALPHA\n\nBETA\n \n\nGAMMA", "the draft's own breaks are kept verbatim")
        XCTAssertEqual(controller.stopReason, .eos)
        XCTAssertEqual(controller.checks?.droppedNumbers, ["1", "2", "3"], "every part's audit is kept")
        XCTAssertEqual(controller.stats?.tokens, 15)
        XCTAssertNil(controller.partProgress)
        let source = controller.sourceDraft as NSString, output = controller.rewrite as NSString
        XCTAssertEqual(controller.segments.map { source.substring(with: $0.source) }, paragraphs)
        XCTAssertEqual(controller.segments.map { output.substring(with: $0.rewrite) }, ["ALPHA", "BETA", "GAMMA"])
    }

    func testADraftLongerThanTheWindowStillRunsBecauseEachPartFits() async throws {
        let engine = EchoEngine()
        let (controller, _) = controller(engine: StubEngine())
        controller.openEngine = { _ in engine }
        controller.draft = Array(repeating: String(repeating: "word ", count: 140), count: 12)
            .joined(separator: "\n\n")
        XCTAssertGreaterThan(controller.draft.count, HumanizePlanner.window)

        await controller.refreshContext()
        XCTAssertEqual(controller.currentPlan?.parts.count, 12)
        XCTAssertTrue(controller.canRun, "fit is judged per part, not on the whole draft")
        controller.run()
        await settle(controller)
        XCTAssertNil(controller.errorMessage)
        XCTAssertEqual(engine.drafts.count, 12)
    }

    func testAFixedSeedIsSentToEveryPartAndOtherwiseNone() async throws {
        let engine = StubEngine(events: [.finished(try result(content: "x"))])
        let (controller, _) = controller(engine: engine)
        controller.draft = "text"
        controller.run()
        await settle(controller)
        XCTAssertNil(engine.options.last?.seed, "no seed by default, so a rerun is a new sample")

        controller.rules.fixedSeed = true
        controller.rules.seed = 1234
        controller.run()
        await settle(controller)
        XCTAssertEqual(engine.options.last?.seed, 1234)
    }

    func testRulesApplyAfterTheModelAndTheHighlightsFollowTheEditedText() async throws {
        let engine = EchoEngine()
        let (controller, _) = controller(engine: StubEngine())
        controller.openEngine = { _ in engine }
        controller.rules.replacements = "ALPHA -> Omega"
        controller.rules.avoidTerms = "gamma"
        controller.rules.protectedTerms = "beta"
        let paragraphs = ["alpha", "beta", "gamma"].map { String(repeating: "\($0) ", count: 90).trimmingCharacters(in: .whitespaces) }
        controller.draft = paragraphs.joined(separator: "\n\n")

        controller.run()
        await settle(controller)

        XCTAssertEqual(controller.rewrite, "Omega\n\nBETA\n\nGAMMA")
        let output = controller.rewrite as NSString
        XCTAssertEqual(controller.segments.map { output.substring(with: $0.rewrite) }, ["Omega", "BETA", "GAMMA"],
                       "segments describe the replaced text the user copies")
        let findings = try XCTUnwrap(controller.ruleFindings)
        XCTAssertEqual(findings.missingProtected, ["beta"], "protected terms match exactly")
        XCTAssertEqual(findings.avoidFound, ["gamma"], "words to avoid match any case")
        XCTAssertEqual(findings.avoidRanges.map { output.substring(with: $0) }, ["GAMMA"])
    }

    func testARunUsesTheRulesAsTheyWereWhenItStartedAndRulesPersist() async throws {
        let defaults = freshDefaults()
        let engine = StubEngine(events: [.finished(try result(content: "use it"))])
        let controller = HumanizeController(defaults: defaults)
        controller.installedPath = { "/m" }
        controller.countDraftTokens = { text, _ in text.count }
        controller.openEngine = { _ in engine }
        controller.rules.replacements = "use -> apply"
        controller.draft = "text"

        controller.run()
        controller.rules.replacements = ""
        await settle(controller)
        XCTAssertEqual(controller.rewrite, "apply it")

        XCTAssertEqual(HumanizeController(defaults: defaults).rules, controller.rules,
                       "a new controller loads the saved rules")
    }

    func testAStoppedRewriteIsNotCheckedAgainstTheRules() async throws {
        final class Hanging: HumanizeEngine, @unchecked Sendable {
            func humanize(
                _ draft: String, options: HumanizeOptions
            ) -> AsyncThrowingStream<GenerationEvent, Error> {
                AsyncThrowingStream { $0.yield(.content("partial")) }
            }
            func cancel() {}
        }
        let (controller, _) = controller(engine: StubEngine())
        controller.openEngine = { _ in Hanging() }
        controller.rules.protectedTerms = "text"
        controller.draft = "text"
        controller.run()
        for _ in 0..<200 where controller.rewrite.isEmpty {
            try await Task.sleep(nanoseconds: 5_000_000)
        }
        controller.cancel()
        await settle(controller)
        XCTAssertEqual(controller.stopReason, .cancelled)
        XCTAssertNil(controller.ruleFindings, "the unwritten rest would read as missing terms")
    }

    func testAFailedPartKeepsTheFinishedPartsAndSkipsTheAudit() async throws {
        let engine = EchoEngine()
        engine.failOnCall = 2
        let (controller, _) = controller(engine: StubEngine())
        controller.openEngine = { _ in engine }
        controller.draft = ["alpha", "beta"].map { String(repeating: "\($0) ", count: 90) }.joined(separator: "\n\n")

        controller.run()
        await settle(controller)

        XCTAssertNotNil(controller.errorMessage)
        XCTAssertEqual(controller.rewrite, "ALPHA", "the finished part stays on screen")
        XCTAssertEqual(controller.segments.count, 1)
        XCTAssertNil(controller.checks, "a partial rewrite is not audited")
        XCTAssertTrue(controller.canRun)
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
        XCTAssertNil(controller.currentPlan)
    }

    func testAuditLinesNameEachFindingAndStayQuietWhenClean() throws {
        let dirty = try JSONDecoder().decode(
            HumanizeAudit.self,
            from: Data("""
                {"droppedNumbers":["37","2024"],"inventedNumbers":["9"],"languageDrift":true,
                 "reuseRatio":0.4,"reuseFlagged":true,"warnings":[]}
                """.utf8))
        var checks = HumanizeChecks()
        checks.add(dirty)
        let lines = HumanizeAuditLines.lines(for: checks)
        XCTAssertEqual(lines.count, 4)
        XCTAssertTrue(lines[0].contains("37, 2024"), "\(lines)")
        XCTAssertTrue(lines[1].contains("9"), "\(lines)")

        var clean = HumanizeChecks()
        clean.add(try JSONDecoder().decode(HumanizeAudit.self, from: Data(cleanAudit.utf8)))
        XCTAssertTrue(clean.isClean)
        XCTAssertTrue(HumanizeAuditLines.lines(for: clean).isEmpty)
    }
}

/// Rewrites each part to its first word, uppercased, and reports the call's
/// index as a dropped number so a test can see every part's audit.
private final class EchoEngine: HumanizeEngine, @unchecked Sendable {
    private(set) var drafts: [String] = []
    var failOnCall: Int?

    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        drafts.append(draft)
        let call = drafts.count
        let word = draft.split(separator: " ").first.map { $0.uppercased() } ?? ""
        let fails = failOnCall == call
        return AsyncThrowingStream { continuation in
            if fails {
                continuation.finish(throwing: NSError(domain: "echo", code: call))
                return
            }
            do {
                let audit = """
                    {"droppedNumbers":["\(call)"],"inventedNumbers":[],"languageDrift":false,
                     "reuseRatio":0.1,"reuseFlagged":false,"warnings":[]}
                    """
                continuation.yield(.content(word))
                continuation.yield(.finished(try result(content: word, audit: audit)))
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
    }

    func cancel() {}
}
