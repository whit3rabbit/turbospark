import XCTest
@testable import TurboSparkApp

/// OC-08 push to talk. Auto-send may only be considered for an utterance whose
/// capture overlapped an explicit Talk. Fakes only: the talk switch runs on a
/// fake clock, audio is synthetic samples, and the decider is a closure.
/// Mutation checks (recorded in the draft): drop the controller's addressed
/// guard, default addressing to true, drop the gate's addressed guard, and stop
/// clearing Talk on mute each redden at least one test here.

private struct PTTDecider: VoiceCallSubmitDeciding {
    let body: @Sendable (String) async throws -> VoiceCallSubmitDecision
    func decide(_ utterance: String) async throws -> VoiceCallSubmitDecision { try await body(utterance) }
}

private final class PTTCount: @unchecked Sendable {
    private let lock = NSLock()
    private var n = 0
    var value: Int { lock.lock(); defer { lock.unlock() }; return n }
    func bump() { lock.lock(); n += 1; lock.unlock() }
}

private final class PTTGate: @unchecked Sendable {
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

@MainActor
final class VoiceCallPushToTalkTests: XCTestCase {
    static let request = "what is the capital of france"

    private func harness(body: (@Sendable (String) async throws -> VoiceCallSubmitDecision)? = nil)
        -> (CallHarness, PTTCount)
    {
        let calls = PTTCount()
        let decider = PTTDecider { text in
            calls.bump()
            if let body { return try await body(text) }
            return .allow
        }
        return (CallHarness(decider: decider, autoSend: true), calls)
    }

    private func sent(_ h: CallHarness) async -> Bool {
        await h.eventually { !h.chat.autoSubmits.isEmpty }
    }

    /// Time for a send that must NOT happen to have happened.
    private func quiet(_ h: CallHarness) async {
        await h.settle()
        try? await Task.sleep(nanoseconds: 80_000_000)
        await h.settle()
    }

    // MARK: Addressing

    func testAnUtteranceWithoutTalkNeverAutoSendsAndStaysInTheDraft() async {
        let (h, calls) = harness()
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: false)
        await quiet(h)
        XCTAssertEqual(calls.value, 0, "the decider is not even asked")
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
        XCTAssertEqual(h.chat.draft, Self.request, "draft-first, exactly as with auto-send off")
        XCTAssertTrue(h.controller.draftReady)
        XCTAssertNotNil(h.live, "the recognizer reopens at once")
    }

    func testAnUtteranceSaidWhileTalkIsOnSendsAndReportsAddressed() async {
        let (h, calls) = harness()
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: true)
        let ok = await sent(h)
        XCTAssertTrue(ok)
        XCTAssertEqual(calls.value, 1)
        XCTAssertEqual(h.chat.autoSubmits.first?.addressed, true)
        XCTAssertEqual(h.chat.autoSubmits.first?.text, Self.request)
    }

    func testTheDeciderIsStillTheSecondCheck() async {
        // Addressed, but the decider says confirm: nothing sends.
        let (h, calls) = harness(body: { _ in .confirm })
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: true)
        await quiet(h)
        XCTAssertEqual(calls.value, 1)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
        XCTAssertEqual(h.chat.draft, Self.request)
    }

    // MARK: Overlap boundaries

    func testTalkReleasedBeforeTheSilenceThatEndsTheUtteranceStillCounts() async {
        let (h, _) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.talk(1)
        h.silence(0.3)           // still inside the silence window
        h.talkSwitch.toggle()    // released: the utterance had already overlapped
        h.silence(0.6)
        await h.settle()
        let ok = await sent(h)
        XCTAssertTrue(ok)
    }

    func testTalkReleasedAfterTheUtteranceEndedButBeforeTheFinalResultStillCounts() async {
        let (h, _) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.talk(1)
        h.silence(1)             // speech ended; the recognizer is finishing
        h.talkSwitch.toggle()    // released before the final text arrives
        await h.settle()
        let ok = await sent(h)
        XCTAssertTrue(ok)
    }

    func testAnUtteranceThatSpansTalkStartCounts() async {
        let (h, _) = harness()
        await h.startCall()
        h.talk(0.5)              // speech already open, Talk off
        XCTAssertFalse(h.talkSwitch.isTalking)
        h.talkSwitch.toggle()    // Talk starts mid-utterance
        h.talk(0.5)
        h.silence(1)
        await h.settle()
        let ok = await sent(h)
        XCTAssertTrue(ok)
    }

    func testTalkPressedAfterTheUtteranceEndedDoesNotCount() async {
        let (h, calls) = harness()
        await h.startCall()
        h.talk(1)
        h.silence(1)             // speech ended, final text not back yet
        h.talkSwitch.toggle()    // too late: no overlap
        await quiet(h)
        XCTAssertEqual(calls.value, 0)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
        XCTAssertEqual(h.chat.draft, "hello there")
    }

    func testTalkReleasedBeforeSpeechBeganDoesNotCount() async {
        let (h, calls) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.talkSwitch.toggle()
        h.talk(1)
        h.silence(1)
        await quiet(h)
        XCTAssertEqual(calls.value, 0)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
    }

    func testAddressingDoesNotLeakToTheNextUtterance() async {
        let (h, _) = harness(body: { _ in .confirm })
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: true)   // confirm: stays in the draft
        let reopened = await h.eventually { h.live != nil && h.transcribers.count >= 2 }
        XCTAssertTrue(reopened)
        h.chat.draft = ""
        h.transcribers.last?.text = Self.request
        h.talk(1); h.silence(1)                                  // Talk is off now
        await quiet(h)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
    }

    // MARK: Send-time re-check

    func testForcedClearDuringTheDecisionRevokesAddressingSoNothingSends() async {
        let gate = PTTGate()
        let (h, calls) = harness(body: { _ in await gate.wait(); return .allow })
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: true)
        _ = await h.eventually { calls.value == 1 }
        h.talkSwitch.clear()     // e.g. the app lost focus
        gate.open()
        await quiet(h)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty, "addressing flipped between decision and send")
        XCTAssertEqual(h.chat.draft, Self.request)
    }

    func testUnflippedAddressingSendsInTheSameFlow() async {
        // The control for the test above: same flow, no clear, it sends.
        let gate = PTTGate()
        let (h, calls) = harness(body: { _ in await gate.wait(); return .allow })
        await h.startCall()
        await h.speakUtterance(Self.request, addressed: true)
        _ = await h.eventually { calls.value == 1 }
        gate.open()
        let ok = await sent(h)
        XCTAssertTrue(ok)
    }

    func testAForcedClearMidUtteranceDropsTheUtterance() async {
        let (h, calls) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.talk(0.5)
        h.talkSwitch.clear()
        h.talk(0.5)
        h.silence(1)
        await quiet(h)
        XCTAssertEqual(calls.value, 0)
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
    }

    // MARK: Clearing

    func testMuteClearsTalkAndTalkCannotStartWhileMuted() async {
        let (h, calls) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        XCTAssertTrue(h.talkSwitch.isTalking)
        h.controller.mute()
        XCTAssertFalse(h.talkSwitch.isTalking, "mute clears Talk")
        h.talkSwitch.toggle()
        XCTAssertFalse(h.talkSwitch.isTalking, "no Talk while muted")
        await h.controller.unmute()
        await h.settle()
        await h.speakUtterance(Self.request, addressed: false)
        await quiet(h)
        XCTAssertEqual(calls.value, 0, "Talk did not survive the mute")
        XCTAssertTrue(h.chat.autoSubmits.isEmpty)
    }

    func testEndOfCallClearsTalk() async {
        let (h, _) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        XCTAssertTrue(h.talkSwitch.isTalking)
        h.controller.end()
        XCTAssertFalse(h.talkSwitch.isTalking)
        h.talkSwitch.toggle()
        XCTAssertFalse(h.talkSwitch.isTalking, "no Talk with no call")
    }

    func testAnApprovalAndATurnStartClearTalk() async {
        let (h, _) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.chat.hasPendingApproval = true
        h.chat.emit(.approvalRaised)
        XCTAssertFalse(h.talkSwitch.isTalking, "an approval pauses the call and clears Talk")

        let (g, _) = harness()
        await g.startCall()
        g.talkSwitch.toggle()
        g.chat.isTurnBusy = true
        g.chat.emit(.turnStarted)
        XCTAssertFalse(g.talkSwitch.isTalking, "a turn consumed it")
    }

    func testTalkNeedsAutoSendOn() async {
        let h = CallHarness(decider: PTTDecider { _ in .allow }, autoSend: false)
        await h.startCall()
        h.talkSwitch.toggle()
        XCTAssertFalse(h.talkSwitch.isTalking, "Talk only means something for auto-send")
    }

    // MARK: The switch: toggle and hold forms on a fake clock

    private func switchOn(_ clock: Clock) -> VoiceCallTalkSwitch {
        let s = VoiceCallTalkSwitch(now: { clock.now })
        s.canTalk = { true }
        return s
    }

    private final class Clock { var now: TimeInterval = 0 }

    func testToggleFormStartsAndStops() {
        let s = switchOn(Clock())
        s.toggle(); XCTAssertTrue(s.isTalking)
        s.toggle(); XCTAssertFalse(s.isTalking)
    }

    func testPointerHoldTalksOnlyWhileHeld() {
        let c = Clock(); let s = switchOn(c)
        s.pointerDown(); XCTAssertTrue(s.isTalking)
        c.now = VoiceCallTalkSwitch.latchThreshold + 0.5
        s.pointerUp(); XCTAssertFalse(s.isTalking, "a long press is a hold: release stops it")
    }

    func testAPointerTapLatchesAndTheNextTapStops() {
        let c = Clock(); let s = switchOn(c)
        s.pointerDown(); c.now = 0.1; s.pointerUp()
        XCTAssertTrue(s.isTalking, "a tap keeps talking on")
        c.now = 5; s.pointerDown()
        XCTAssertFalse(s.isTalking, "the next press stops it")
        c.now = 5.1; s.pointerUp()
        XCTAssertFalse(s.isTalking, "and its release does not restart it")
        s.pointerDown(); c.now = 5.2; s.pointerUp()
        s.toggle()
        XCTAssertFalse(s.isTalking, "the toggle form also stops a latch")
    }

    func testTheHoldBoundaryIsTheLatchThreshold() {
        let c = Clock(); let s = switchOn(c)
        s.pointerDown(); c.now = VoiceCallTalkSwitch.latchThreshold - 0.01; s.pointerUp()
        XCTAssertTrue(s.isTalking)
        s.toggle()
        c.now = 10; s.pointerDown(); c.now = 10 + VoiceCallTalkSwitch.latchThreshold + 0.001; s.pointerUp()
        XCTAssertFalse(s.isTalking)
    }

    func testAClearDuringAHoldIsNotUndoneByTheLateRelease() {
        let c = Clock(); let s = switchOn(c)
        s.pointerDown()
        s.clear()
        XCTAssertFalse(s.isTalking)
        c.now = 0.1; s.pointerUp()
        XCTAssertFalse(s.isTalking, "a release after a forced clear must not latch it back on")
    }

    // MARK: Idle timeout of a latched Talk

    private func idleSwitch(_ c: Clock) -> VoiceCallTalkSwitch {
        let s = VoiceCallTalkSwitch(now: { c.now }, tick: { _ in try await Task.sleep(for: .seconds(3_600)) })
        s.canTalk = { true }
        return s
    }

    func testALatchedTalkClearsAtTheIdleBoundary() {
        let c = Clock(); let s = idleSwitch(c)
        s.toggle()
        c.now = VoiceCallTalkSwitch.idleTimeout - 0.01; s.checkIdle()
        XCTAssertTrue(s.isTalking, "just before the boundary")
        c.now = VoiceCallTalkSwitch.idleTimeout; s.checkIdle()
        XCTAssertFalse(s.isTalking, "at the boundary")
    }

    func testASpeechOnsetRestartsTheIdleWindow() {
        let c = Clock(); let s = idleSwitch(c)
        s.toggle()
        c.now = 40; s.speechOnset()
        c.now = 80; s.checkIdle()
        XCTAssertTrue(s.isTalking, "40 s after the onset")
        c.now = 85; s.checkIdle()
        XCTAssertFalse(s.isTalking, "45 s after the onset")
    }

    func testATapLatchIsSubjectToTheTimeoutAndAHeldPressIsNot() {
        let c = Clock(); let s = idleSwitch(c)
        s.pointerDown(); c.now = 0.1; s.pointerUp()
        c.now = 0.1 + VoiceCallTalkSwitch.idleTimeout; s.checkIdle()
        XCTAssertFalse(s.isTalking, "a tap latch times out")
        c.now = 100; s.pointerDown()
        c.now = 100 + 10 * VoiceCallTalkSwitch.idleTimeout; s.checkIdle()
        XCTAssertTrue(s.isTalking, "a pointer press that is still held is exempt")
        s.pointerUp()
        XCTAssertFalse(s.isTalking)
    }

    func testMuteAndEndStillClearImmediatelyWithTheTimeoutArmed() async {
        let (h, _) = harness()
        await h.startCall()
        h.talkSwitch.toggle()
        h.controller.mute()
        XCTAssertFalse(h.talkSwitch.isTalking)
        await h.controller.unmute(); await h.settle()
        h.talkSwitch.toggle()
        h.controller.end()
        XCTAssertFalse(h.talkSwitch.isTalking)
    }

    func testTheControllerRestartsTheWindowOnSpeechOnset() async {
        let (h, _) = harness(body: { _ in .confirm })
        await h.startCall()
        h.talkSwitch.toggle()
        h.clock = 40
        h.talk(0.5)              // speech onset at t=40
        h.clock = 80
        h.talkSwitch.checkIdle()
        XCTAssertTrue(h.talkSwitch.isTalking)
        h.clock = 85
        h.talkSwitch.checkIdle()
        XCTAssertFalse(h.talkSwitch.isTalking)
    }

    func testASwitchNobodyOwnsCannotTalk() {
        let s = VoiceCallTalkSwitch(now: { 0 })
        s.toggle(); s.pointerDown()
        XCTAssertFalse(s.isTalking)
    }

    // MARK: Structure

    func testTheGuardsAndTheDesignChoicesArePinnedInSource() throws {
        let audio = VoiceCallSources.directory().appendingPathComponent("Audio")
        let controller = try String(contentsOf: audio.appendingPathComponent("VoiceCallController.swift"), encoding: .utf8)
        XCTAssertTrue(controller.contains("guard addressed, let decider = submitDecider"))
        XCTAssertTrue(controller.contains("private var utteranceAddressed = false"))
        XCTAssertTrue(controller.contains("private var decisionAddressed = false"))
        for site in ["mute()", "pauseForApproval()", "fail(", "resetSession()", "turnStarted()"] {
            XCTAssertTrue(controller.contains(site), site)
        }
        // Every forced-stop path clears Talk.
        XCTAssertGreaterThanOrEqual(controller.components(separatedBy: "talk.clear()").count - 1, 5)
        // Toggle and pointer hold only: no key monitor, because a Command
        // chord's key-up is not reliably delivered.
        for name in ["VoiceCallTalkSwitch.swift", "VoiceCallTalkButton.swift", "VoiceCallPanelView.swift"] {
            let source = try String(contentsOf: audio.appendingPathComponent(name), encoding: .utf8)
            XCTAssertFalse(source.contains("addLocalMonitorForEvents"), name)
        }
        let gate = try String(contentsOf: audio.appendingPathComponent("VoiceCallSubmitDecider.swift"), encoding: .utf8)
        XCTAssertTrue(gate.contains("guard i.addressed else { return false }"))
        XCTAssertTrue(gate.contains("var addressed: Bool = false"))
        XCTAssertNotNil(KeyboardShortcutCatalog.command(for: .toggleVoiceTalk))
    }
}
