import Foundation
import XCTest
import TurboSpark
@testable import TurboSparkApp

/// OC-08 dedicated decider model. Fakes only: no model, no real clock. Each
/// guard has a test that fails when the guard is removed.

private final class FakeEngine: VoiceCallDeciderEngine, @unchecked Sendable {
    private let lock = NSLock()
    private var _cancels = 0
    private var _runs = 0
    var reply = "REQUEST"
    var hold: Gate?
    var cancels: Int { lock.lock(); defer { lock.unlock() }; return _cancels }
    var runs: Int { lock.lock(); defer { lock.unlock() }; return _runs }
    func complete(_ messages: [ChatMessage], options: GenerateOptions) async throws -> String {
        lock.lock(); _runs += 1; lock.unlock()
        if let hold { await hold.wait() }
        return reply
    }
    func cancel() { lock.lock(); _cancels += 1; lock.unlock() }
}

private final class Gate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Void, Never>?
    private var opened = false
    func wait() async {
        await withCheckedContinuation { c in
            lock.lock()
            if opened { lock.unlock(); c.resume() } else { continuation = c; lock.unlock() }
        }
    }
    func open() {
        lock.lock(); opened = true; let c = continuation; continuation = nil; lock.unlock()
        c?.resume()
    }
}

private final class Opens: @unchecked Sendable {
    private let lock = NSLock()
    private var _paths: [String] = []
    var paths: [String] { lock.lock(); defer { lock.unlock() }; return _paths }
    func add(_ p: String) { lock.lock(); _paths.append(p); lock.unlock() }
}

/// Sleeps a test releases by duration. `fire` wakes only the sleeps waiting
/// NOW, so a later sleep of the same length blocks again.
private final class Clock: @unchecked Sendable {
    private let lock = NSLock()
    private var waiting: [Duration: [CheckedContinuation<Void, Never>]] = [:]
    func sleep(_ d: Duration) async throws {
        await withCheckedContinuation { c in
            lock.lock(); waiting[d, default: []].append(c); lock.unlock()
        }
        try Task.checkCancellation()
    }
    func waiters(_ d: Duration) -> Int { lock.lock(); defer { lock.unlock() }; return waiting[d]?.count ?? 0 }
    func fire(_ d: Duration) {
        lock.lock(); let woken = waiting.removeValue(forKey: d) ?? []; lock.unlock()
        woken.forEach { $0.resume() }
    }
}

private let small = VoiceCallDeciderCandidate(
    installId: "internal:text/qwen-small", alias: "qwen-small", path: "/m/qwen-small", bytes: 2 << 30)

@MainActor
final class VoiceCallDeciderHostTests: XCTestCase {
    private let loadBudget = Duration.seconds(7)
    private let idle = Duration.seconds(11)

    private func host(
        engine: FakeEngine = FakeEngine(), openGate: Gate? = nil, opens: Opens = Opens(),
        clock: Clock = Clock(), physical: UInt64 = 36 << 30
    ) -> (VoiceCallDeciderHost, FakeEngine, Opens, Clock) {
        let h = VoiceCallDeciderHost()
        h.open = { path in
            opens.add(path)
            if let openGate { await openGate.wait() }
            return engine
        }
        h.sleep = { try await clock.sleep($0) }
        h.loadBudget = loadBudget
        h.idleTimeout = idle
        h.physicalMemory = physical
        return (h, engine, opens, clock)
    }

    private func eventually(_ condition: () -> Bool) async -> Bool {
        for _ in 0..<300 {
            if condition() { return true }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        return condition()
    }

    // MARK: Resolution and settings

    func testAnUnknownSavedModelFallsBackToAutomaticWithANote() {
        let gone = VoiceCallDeciderResolution.resolve(
            savedID: "internal:text/removed", available: [small], loadedChatPath: nil)
        XCTAssertEqual(gone, .automatic(note: .savedModelUnavailable))
        XCTAssertNil(gone.dedicatedCandidate, "it must never pick a different model")
        let none = VoiceCallDeciderResolution.resolve(savedID: nil, available: [small], loadedChatPath: nil)
        XCTAssertEqual(none, .automatic(note: nil), "Automatic has no note")
        let ok = VoiceCallDeciderResolution.resolve(savedID: small.installId, available: [small], loadedChatPath: "/other")
        XCTAssertEqual(ok.dedicatedCandidate, small)
    }

    func testTheChosenModelThatIsAlreadyTheChatModelNeedsNoSecondCopy() {
        let same = VoiceCallDeciderResolution.resolve(
            savedID: small.installId, available: [small], loadedChatPath: small.path)
        XCTAssertEqual(same, .automatic(note: nil))
    }

    func testTheChoiceIsAutomaticByDefaultAndToleratesBadValues() throws {
        XCTAssertNil(MacAppSettings().voiceCallDeciderModelID)
        XCTAssertNil(try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8)).voiceCallDeciderModelID)
        XCTAssertNil(
            try JSONDecoder().decode(MacAppSettings.self, from: Data(#"{"voiceCallDeciderModelID": 7}"#.utf8))
                .voiceCallDeciderModelID, "a wrong type must not fail the file")
        var settings = MacAppSettings()
        settings.voiceCallDeciderModelID = small.installId
        let back = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertEqual(back.voiceCallDeciderModelID, small.installId)
    }

    func testTheDecisionBudgetIsPerModelAndNeverShorterThanAutomatic() {
        let gb = UInt64(1) << 30
        XCTAssertEqual(VoiceCallDeciderBudget.decisionTimeout(modelBytes: gb), VoiceCallTuning.decisionTimeout)
        XCTAssertGreaterThan(VoiceCallDeciderBudget.decisionTimeout(modelBytes: 5 * gb), VoiceCallTuning.decisionTimeout)
        XCTAssertGreaterThan(
            VoiceCallDeciderBudget.decisionTimeout(modelBytes: 9 * gb),
            VoiceCallDeciderBudget.decisionTimeout(modelBytes: 5 * gb))
        XCTAssertGreaterThan(VoiceCallDeciderBudget.loadBudget, .seconds(10), "the load has its own, larger budget")
    }

    // MARK: Lazy load

    func testNothingLoadsUntilACallWithAutoSendIsActive() async {
        let (h, _, opens, _) = host()
        h.configure(candidate: small)
        await Task.yield()
        XCTAssertEqual(opens.paths, [], "choosing a model must not load it")
        XCTAssertEqual(h.status, .unloaded)
        h.setWanted(true)
        let loaded = await eventually { h.status == .ready }
        XCTAssertTrue(loaded)
        XCTAssertEqual(opens.paths, [small.path])
    }

    func testAutomaticNeverLoadsAnything() async {
        let (h, _, opens, _) = host()
        h.configure(candidate: nil)
        h.setWanted(true)
        await Task.yield()
        XCTAssertEqual(opens.paths, [])
        XCTAssertEqual(h.status, .unloaded)
    }

    func testAColdLoadNeverCountsAgainstADecision() async throws {
        let openGate = Gate()
        let (h, engine, _, _) = host(openGate: openGate)
        h.configure(candidate: small)
        h.setWanted(true)
        XCTAssertEqual(h.status, .loading)
        // While loading, a decision fails closed AT ONCE and never waits.
        do {
            _ = try await h.complete([], options: GenerateOptions())
            XCTFail("a decision before the load must throw")
        } catch let error as VoiceCallDeciderError {
            XCTAssertEqual(error, .notReady)
        }
        XCTAssertEqual(engine.runs, 0)
        openGate.open()
        let ready = await eventually { h.status == .ready }
        XCTAssertTrue(ready)
        let reply = try await h.complete([], options: GenerateOptions())
        XCTAssertEqual(reply, "REQUEST")
        XCTAssertEqual(engine.runs, 1)
    }

    func testALoadThatOverrunsItsOwnBudgetFailsAndALateOpenIsDropped() async {
        let openGate = Gate()
        let (h, _, _, clock) = host(openGate: openGate)
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { clock.waiters(loadBudget) > 0 }
        clock.fire(loadBudget)
        let failed = await eventually { h.status == .failed(.loadTimedOut) }
        XCTAssertTrue(failed)
        openGate.open()
        try? await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertEqual(h.status, .failed(.loadTimedOut), "a late open must not republish")
        XCTAssertNil(h.heldPath)
    }

    func testAFailedLoadFailsClosedAndIsRetriedByTheNextCall() async throws {
        let h = VoiceCallDeciderHost()
        let attempts = Opens()
        h.open = { path in attempts.add(path); throw VoiceCallDeciderError.notReady }
        h.sleep = { _ in try await Task.sleep(for: .seconds(3_600)) }
        h.configure(candidate: small)
        h.setWanted(true)
        let failed = await eventually { h.status == .failed(.loadFailed) }
        XCTAssertTrue(failed)
        do { _ = try await h.complete([], options: GenerateOptions()); XCTFail() } catch {}
        XCTAssertEqual(attempts.paths.count, 1, "no retry loop inside one call")
        h.setWanted(false)
        h.setWanted(true)
        let retried = await eventually { attempts.paths.count == 2 }
        XCTAssertTrue(retried)
    }

    func testAModelTooLargeBesideTheChatModelIsNotLoaded() async {
        let (h, _, opens, _) = host(physical: 4 << 30) // cap is 1 GB, the model is 2 GB
        h.configure(candidate: small)
        h.setWanted(true)
        XCTAssertEqual(h.status, .failed(.tooLarge))
        XCTAssertEqual(opens.paths, [])
    }

    // MARK: Lanes

    func testADedicatedDeciderNeverRunsWhileATurnIsBusy() async {
        let (h, engine, _, _) = host()
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        h.turnBusy = { true }
        do { _ = try await h.complete([], options: GenerateOptions()); XCTFail() } catch let e as VoiceCallDeciderError {
            XCTAssertEqual(e, .turnBusy)
        } catch { XCTFail("\(error)") }
        XCTAssertEqual(engine.runs, 0, "the engine must not start beside a chat turn")
    }

    func testATurnThatStartsDuringADecisionDiscardsItsAnswer() async {
        let hold = Gate()
        let engine = FakeEngine(); engine.hold = hold
        let (h, _, _, _) = host(engine: engine)
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        var busy = false
        h.turnBusy = { busy }
        let task = Task { try await h.complete([], options: GenerateOptions()) }
        _ = await eventually { engine.runs == 1 }
        busy = true
        h.cancelInFlight()
        XCTAssertEqual(engine.cancels, 1)
        hold.open()
        do { _ = try await task.value; XCTFail("a late answer during a turn must not be returned") }
        catch let e as VoiceCallDeciderError { XCTAssertEqual(e, .turnBusy) }
        catch { XCTFail("\(error)") }
    }

    // MARK: Unload

    func testTheModelUnloadsWhenTheCallEnds() async {
        let (h, engine, _, _) = host()
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        XCTAssertEqual(h.heldPath, small.path)
        h.setWanted(false)
        XCTAssertEqual(h.status, .unloaded)
        XCTAssertNil(h.heldPath)
        XCTAssertEqual(engine.cancels, 1)
        do { _ = try await h.complete([], options: GenerateOptions()); XCTFail() } catch let e as VoiceCallDeciderError {
            XCTAssertEqual(e, .notWanted)
        } catch { XCTFail("\(error)") }
    }

    func testTheModelUnloadsAfterIdleAndTheNextDecisionReloadsFailingClosedOnce() async throws {
        let (h, _, opens, clock) = host()
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        _ = await eventually { clock.waiters(idle) > 0 }
        clock.fire(idle)
        let unloaded = await eventually { h.status == .unloaded }
        XCTAssertTrue(unloaded)
        // The call is still active: this decision fails closed and reloads.
        do { _ = try await h.complete([], options: GenerateOptions()); XCTFail() } catch let e as VoiceCallDeciderError {
            XCTAssertEqual(e, .notReady)
        }
        let reloaded = await eventually { h.status == .ready }
        XCTAssertTrue(reloaded)
        XCTAssertEqual(opens.paths.count, 2)
    }

    func testChangingTheChoiceUnloadsTheOldModel() async {
        let (h, _, _, _) = host()
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        h.configure(candidate: nil)
        XCTAssertEqual(h.status, .unloaded)
        XCTAssertNil(h.heldPath)
    }

    func testReleaseBeforeADeleteOrMoveOnlyLetsGoOfTheHeldPath() async {
        let (h, _, _, _) = host()
        h.configure(candidate: small)
        h.setWanted(true)
        _ = await eventually { h.status == .ready }
        h.release(path: "/m/something-else")
        XCTAssertEqual(h.status, .ready)
        h.release(path: small.path)
        XCTAssertEqual(h.status, .unloaded)
    }

    // MARK: Routing and controller

    func testTheRouterUsesTheDedicatedDeciderOnlyWhenOneIsChosen() async throws {
        struct Fixed: VoiceCallSubmitDeciding {
            let result: VoiceCallSubmitDecision
            func decide(_ utterance: String) async throws -> VoiceCallSubmitDecision { result }
        }
        let flag = Opens() // non-empty means dedicated
        let router = VoiceCallRoutedDecider(
            automatic: Fixed(result: .confirm), dedicated: Fixed(result: .allow),
            usesDedicated: { !flag.paths.isEmpty })
        var decision = try await router.decide("x")
        XCTAssertEqual(decision, .confirm)
        flag.add("on")
        decision = try await router.decide("x")
        XCTAssertEqual(decision, .allow)
    }

    func testTheControllerReadsTheBudgetPerDecision() async {
        final class Seen: @unchecked Sendable {
            let lock = NSLock(); var values: [Duration] = []
            func add(_ d: Duration) { lock.lock(); values.append(d); lock.unlock() }
        }
        let seen = Seen()
        let decider = ClosureDecider { _ in try await Task.sleep(for: .seconds(3_600)); return .allow }
        let h = CallHarness(decider: decider, autoSend: true, sleep: { d in
            seen.add(d)
            try await Task.sleep(for: .seconds(3_600))
        })
        h.controller.decisionTimeoutProvider = { .milliseconds(3_000) }
        await h.startCall()
        await h.speakUtterance("what is the capital of france")
        _ = await h.eventually { !seen.values.isEmpty }
        XCTAssertEqual(seen.values.first, .milliseconds(3_000), "the dedicated model's budget, not the default")
    }

    func testADroppedDecisionStopsTheDedicatedModel() async {
        let cancelled = Opens()
        let gate = Gate()
        let decider = ClosureDecider { _ in await gate.wait(); return .allow }
        let h = CallHarness(decider: decider, autoSend: true)
        h.controller.onDecisionCancelled = { cancelled.add("x") }
        await h.startCall()
        await h.speakUtterance("what is the capital of france")
        await h.settle()
        h.chat.isTurnBusy = true
        h.chat.emit(.turnStarted)
        XCTAssertEqual(cancelled.paths.count, 1, "a turn that starts mid-decision must stop the model")
        gate.open()
    }

    func testTheDecisionRouteHasNoDownloadOrNetworkSurface() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Audio/VoiceCallDeciderHost.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        for banned in ["URLSession", "install(", "Keychain", "ts_hf_token", "URLRequest"] {
            XCTAssertFalse(source.contains(banned), banned)
        }
    }
}

private struct ClosureDecider: VoiceCallSubmitDeciding {
    let body: @Sendable (String) async throws -> VoiceCallSubmitDecision
    func decide(_ utterance: String) async throws -> VoiceCallSubmitDecision { try await body(utterance) }
}
