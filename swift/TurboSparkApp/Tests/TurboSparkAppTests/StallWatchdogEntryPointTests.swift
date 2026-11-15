import XCTest
import TurboSpark

@testable import TurboSparkApp

/// OC-14 entry points beyond the main chat turn: compaction (local summarizer
/// stream) and Humanize (its own session). Every clock is injected and nothing
/// polls, so nothing waits, hangs or relaunches for real.
@MainActor
final class StallWatchdogEntryPointTests: XCTestCase {
    private var now: TimeInterval = 0

    private typealias Stream = AsyncThrowingStream<GenerationEvent, Error>

    private func makeModel() -> AppModel {
        now = 0
        let model = AppModel.withoutChatTools()
        model.stallClock = { [unowned self] in self.now }
        model.stallPollInterval = nil
        return model
    }

    private func waitUntil(_ condition: @MainActor () -> Bool) async {
        for _ in 0..<400 where !condition() {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private func controlledStream() -> (Stream, Stream.Continuation) {
        var captured: Stream.Continuation!
        let stream = Stream { captured = $0 }
        return (stream, captured)
    }

    // MARK: - Compaction

    func testCompactionStreamArmsAndDisarmsOnFinish() async throws {
        let model = makeModel()
        let (stream, continuation) = controlledStream()
        let task = Task { try await model.consumeCompactionStream(stream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isArmed, true)
        continuation.yield(.content("sum"))
        continuation.finish()
        let text = try await task.value
        XCTAssertEqual(text, "sum")
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isArmed, false)
    }

    func testCompactionStallSurfacesTheBannerStateThenActivityClearsIt() async throws {
        let model = makeModel()
        model.isCompacting = true
        let (stream, continuation) = controlledStream()
        let task = Task { try await model.consumeCompactionStream(stream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }
        now = 90
        model.stallWatchdogs[.compaction]?.tick()
        XCTAssertEqual(model.generationStallNotice, GenerationStallNotice(silenceSeconds: 90))
        continuation.yield(.content("x"))
        await waitUntil { model.generationStallNotice == nil }
        XCTAssertNil(model.generationStallNotice)
        continuation.finish()
        _ = try await task.value
    }

    func testCompactionPrefillProgressCountsAsActivity() async throws {
        let model = makeModel()
        model.isCompacting = true
        let (stream, continuation) = controlledStream()
        let task = Task { try await model.consumeCompactionStream(stream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }
        now = 80
        continuation.yield(.prefill(done: 10, total: 100))
        // Wait for the consumer to stamp it: the window then runs from 80.
        await waitUntil { model.livePrefillDone == 10 }
        now = 169
        model.stallWatchdogs[.compaction]?.tick()
        XCTAssertNil(model.generationStallNotice)
        now = 170
        model.stallWatchdogs[.compaction]?.tick()
        XCTAssertNotNil(model.generationStallNotice)
        continuation.finish()
        _ = try await task.value
    }

    func testCompactionCancelDisarmsAndClearsTheNotice() async throws {
        let model = makeModel()
        model.isCompacting = true
        let (stream, _) = controlledStream()
        let task = Task { try await model.consumeCompactionStream(stream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }
        now = 90
        model.stallWatchdogs[.compaction]?.tick()
        XCTAssertNotNil(model.generationStallNotice)
        task.cancel()
        _ = try? await task.value
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isArmed, false)
        XCTAssertNil(model.generationStallNotice)
    }

    func testCompactionErrorDisarms() async throws {
        struct Boom: Error {}
        let model = makeModel()
        let (stream, continuation) = controlledStream()
        let task = Task { try await model.consumeCompactionStream(stream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }
        continuation.finish(throwing: Boom())
        do { _ = try await task.value; XCTFail("expected a throw") } catch is Boom {}
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isArmed, false)
    }

    func testCompactionThatNeverStartsLeavesNothingArmed() async {
        // No session and no overrides: refused before any generate call.
        let model = makeModel()
        let outcome = await model.performCompaction(
            chatID: UUID(), project: nil, focus: nil, trigger: "manual", keepRecent: 2)
        guard case .failed("no model is loaded") = outcome else {
            return XCTFail("expected the no-model refusal, got \(outcome)")
        }
        XCTAssertTrue(model.stallWatchdogs.values.allSatisfy { !$0.isArmed })
    }

    // MARK: - Independence

    func testSourcesKeepIndependentStateOnOneModel() {
        let model = makeModel()
        model.generating = true
        model.isCompacting = true
        model.stallWatchdogBegin(.chat)
        now = 50
        model.stallWatchdogBegin(.compaction)
        now = 90
        model.stallWatchdogs[.chat]?.tick()
        model.stallWatchdogs[.compaction]?.tick()
        // Only the chat monitor's window has elapsed.
        XCTAssertEqual(model.generationStallNotice, GenerationStallNotice(silenceSeconds: 90))
        XCTAssertEqual(model.stallWatchdogs[.chat]?.isShowingStall, true)
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isShowingStall, false)
        // Ending the other source does not clear this source's notice.
        model.stallWatchdogEnd(.compaction)
        XCTAssertNotNil(model.generationStallNotice)
        XCTAssertEqual(model.stallWatchdogs[.chat]?.isArmed, true)
        // Activity on one source does not reset the other.
        model.stallWatchdogBegin(.compaction)
        now = 120
        model.stallWatchdogNoteActivity(.compaction)
        now = 140
        model.stallWatchdogs[.compaction]?.tick()
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isShowingStall, false)
        model.stallWatchdogEnd(.chat)
        XCTAssertNil(model.generationStallNotice)
    }

    // MARK: - Humanize

    private final class ScriptedEngine: HumanizeEngine, @unchecked Sendable {
        let stream: Stream
        private(set) var cancelCount = 0
        init(stream: Stream) { self.stream = stream }
        func humanize(_ draft: String, options: HumanizeOptions) -> Stream { stream }
        func cancel() { cancelCount += 1 }
    }

    private func humanizeController(engine: HumanizeEngine, installed: Bool = true) -> HumanizeController {
        let controller = HumanizeController(
            defaults: UserDefaults(suiteName: "stall.tests.\(UUID().uuidString)")!)
        controller.installedPath = { installed ? "/m" : nil }
        controller.countDraftTokens = { text, _ in text.count }
        controller.openEngine = { _ in engine }
        controller.stallClock = { [unowned self] in self.now }
        controller.stallPollInterval = nil
        controller.draft = "text"
        return controller
    }

    private func settle(_ controller: HumanizeController) async {
        await waitUntil { !controller.isBusy }
    }

    func testHumanizeArmsOnStreamStallShowsNoticeAndStopDisarms() async {
        now = 0
        let (stream, _) = controlledStream()
        let engine = ScriptedEngine(stream: stream)
        let controller = humanizeController(engine: engine)
        controller.run()
        await waitUntil { controller.stallWatchdog?.isArmed == true }
        XCTAssertEqual(controller.stallWatchdog?.isArmed, true)
        now = 90
        controller.stallWatchdog?.tick()
        XCTAssertEqual(controller.stallNotice, GenerationStallNotice(silenceSeconds: 90))
        controller.stopStalledRewrite()
        await settle(controller)
        XCTAssertEqual(controller.stallWatchdog?.isArmed, false)
        XCTAssertNil(controller.stallNotice)
        XCTAssertEqual(engine.cancelCount, 1)
    }

    func testHumanizePrefillProgressIsActivityAndFinishDisarms() async throws {
        now = 0
        let (stream, continuation) = controlledStream()
        let controller = humanizeController(engine: ScriptedEngine(stream: stream))
        controller.run()
        await waitUntil { controller.stallWatchdog?.isArmed == true }
        now = 80
        continuation.yield(.prefill(done: 1, total: 4))
        await waitUntil { controller.prefillProgress != nil }
        now = 169
        controller.stallWatchdog?.tick()
        XCTAssertNil(controller.stallNotice)
        continuation.finish()
        await settle(controller)
        XCTAssertEqual(controller.stallWatchdog?.isArmed, false)
    }

    func testHumanizeErrorAndCancelDisarm() async {
        struct Boom: Error {}
        now = 0
        let (failing, failingContinuation) = controlledStream()
        let errored = humanizeController(engine: ScriptedEngine(stream: failing))
        errored.run()
        await waitUntil { errored.stallWatchdog?.isArmed == true }
        failingContinuation.finish(throwing: Boom())
        await settle(errored)
        XCTAssertEqual(errored.stallWatchdog?.isArmed, false)

        let (open, _) = controlledStream()
        let cancelled = humanizeController(engine: ScriptedEngine(stream: open))
        cancelled.run()
        await waitUntil { cancelled.stallWatchdog?.isArmed == true }
        cancelled.cancel()
        await settle(cancelled)
        XCTAssertEqual(cancelled.stallWatchdog?.isArmed, false)
    }

    func testHumanizeThatNeverStartsLeavesNothingArmed() async {
        let (stream, _) = controlledStream()
        let controller = humanizeController(engine: ScriptedEngine(stream: stream), installed: false)
        controller.run()
        XCTAssertFalse(controller.isBusy)
        XCTAssertNil(controller.stallWatchdog)
    }

    func testHumanizeAndCompactionMonitorsAreIndependent() async throws {
        now = 0
        let model = makeModel()
        model.isCompacting = true
        let (compactStream, compactContinuation) = controlledStream()
        let compaction = Task { try await model.consumeCompactionStream(compactStream) }
        await waitUntil { model.stallWatchdogs[.compaction]?.isArmed == true }

        let (humanizeStream, _) = controlledStream()
        let controller = humanizeController(engine: ScriptedEngine(stream: humanizeStream))
        controller.run()
        await waitUntil { controller.stallWatchdog?.isArmed == true }

        now = 90
        controller.stallWatchdog?.tick()
        XCTAssertNotNil(controller.stallNotice)
        XCTAssertNil(model.generationStallNotice, "a Humanize stall must not raise the chat notice")
        controller.cancel()
        await settle(controller)
        XCTAssertEqual(model.stallWatchdogs[.compaction]?.isArmed, true)
        compactContinuation.finish()
        _ = try await compaction.value
    }
}
