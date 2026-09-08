import XCTest

@testable import TurboSparkApp

/// The pure goal decisions: backoff, deferral runs, check-in text,
/// condition clipping, and the evaluator's verdict parsing. Everything
/// here is a value-in/value-out function, so each case pins one rule of
/// `swift/docs/SWIFT_GOALS.md` without a session or a model.
final class GoalPolicyTests: XCTestCase {
    private func goal(
        condition: String = "all tests pass",
        deferredSince: Date? = nil,
        checkinCount: Int = 0,
        lastDeferralPassAt: Date? = nil,
        idleCheckinCount: Int = 0
    ) -> ChatGoalState {
        var state = ChatGoalState(condition: condition)
        state.deferredSince = deferredSince
        state.checkinCount = checkinCount
        state.lastDeferralPassAt = lastDeferralPassAt
        state.idleCheckinCount = idleCheckinCount
        return state
    }

    // MARK: - Backoff

    func testCheckinBackoffDoublesThenCapsAtFourTimesBase() {
        XCTAssertEqual(GoalPolicy.checkinInterval(afterCheckins: 0), 30 * 60)
        XCTAssertEqual(GoalPolicy.checkinInterval(afterCheckins: 1), 60 * 60)
        XCTAssertEqual(GoalPolicy.checkinInterval(afterCheckins: 2), 120 * 60)
        // The published sequence is 30 min -> 1 hr -> every 2 hrs: the cap
        // is 4x the base and every later check-in sits on it.
        XCTAssertEqual(GoalPolicy.checkinInterval(afterCheckins: 3), 120 * 60)
        XCTAssertEqual(GoalPolicy.checkinInterval(afterCheckins: 10), 120 * 60)
    }

    // MARK: - Deferral passes

    func testFirstDeferralPassStartsARunAndIsNotYetDue() {
        let now = Date()
        let pass = GoalPolicy.deferralPass(
            goal: goal(), tasks: [agentTask(startedAt: now)], now: now)
        XCTAssertTrue(pass.isNewRun)
        XCTAssertEqual(pass.deferredSince, now)
        XCTAssertEqual(pass.checkinCount, 0)
        XCTAssertFalse(pass.checkinDue)
    }

    func testCheckinIsDueAfterTheCurrentIntervalElapses() {
        let start = Date(timeIntervalSinceNow: -1800)
        var state = goal()
        state.deferredSince = start
        let pass = GoalPolicy.deferralPass(
            goal: state, tasks: [agentTask(startedAt: start)], now: Date())
        XCTAssertFalse(pass.isNewRun)
        XCTAssertTrue(pass.checkinDue, "30 minutes of deferral is the first due check-in")
    }

    func testBackoffPushesTheNextCheckinOut() {
        let start = Date(timeIntervalSinceNow: -3600)
        var state = goal()
        state.deferredSince = start
        state.checkinCount = 1
        let pass = GoalPolicy.deferralPass(
            goal: state, tasks: [agentTask(startedAt: start)], now: Date())
        XCTAssertTrue(pass.checkinDue, "1 hr elapsed against a 1 hr interval")
        XCTAssertEqual(pass.checkinCount, 1)
    }

    func testWorkStartedAfterTheLastPassBeginsANewRun() {
        let longAgo = Date(timeIntervalSinceNow: -2 * 3600)
        var state = goal()
        state.deferredSince = longAgo
        state.checkinCount = 2
        state.lastDeferralPassAt = Date(timeIntervalSinceNow: -3600)
        // The new task started AFTER the last pass, and the last pass is
        // older than one base interval: the backoff resets.
        let freshTask = agentTask(startedAt: Date(timeIntervalSinceNow: -600))
        let pass = GoalPolicy.deferralPass(goal: state, tasks: [freshTask], now: Date())
        XCTAssertTrue(pass.isNewRun)
        XCTAssertEqual(pass.checkinCount, 0)
        XCTAssertFalse(pass.checkinDue)
    }

    func testFreshWorkSecondsAfterAPassDoesNotResetTheClock() {
        let longAgo = Date(timeIntervalSinceNow: -2 * 3600)
        var state = goal()
        state.deferredSince = longAgo
        state.checkinCount = 2
        state.lastDeferralPassAt = Date(timeIntervalSinceNow: -120)
        // New task, but the last pass was 2 minutes ago: the user has been
        // waiting on this deferral run, so the clock keeps running.
        let freshTask = agentTask(startedAt: Date(timeIntervalSinceNow: -60))
        let pass = GoalPolicy.deferralPass(goal: state, tasks: [freshTask], now: Date())
        XCTAssertFalse(pass.isNewRun)
        XCTAssertEqual(pass.checkinCount, 2)
    }

    func testIdleTimerDelayNeverFallsBelowTheRetryFloor() {
        let start = Date(timeIntervalSinceNow: -7200)
        var state = goal()
        state.deferredSince = start
        state.checkinCount = 2
        let pass = GoalPolicy.deferralPass(
            goal: state, tasks: [agentTask(startedAt: start)], now: Date())
        let delay = GoalPolicy.idleTimerDelay(pass: pass, now: Date())
        XCTAssertGreaterThanOrEqual(delay, GoalPolicy.idleTimerRetryInterval)
    }

    // MARK: - Idle cap

    func testIdleCheckinCapAllowsExactlyThree() {
        let state = goal(idleCheckinCount: 2)
        XCTAssertTrue(state.canIdleCheckin)
        let capped = goal(idleCheckinCount: 3)
        XCTAssertFalse(capped.canIdleCheckin)
    }

    // MARK: - Check-in text

    func testCheckinWithRunningWorkListsEachTask() {
        let text = GoalPolicy.checkinMessage(
            condition: "migrate the config",
            tasks: [
                GoalBackgroundTask(
                    id: "bga_1", kind: .agent, label: "Explorer",
                    detail: "find all call sites", startedAt: Date()),
                GoalBackgroundTask(
                    id: "bg_2", kind: .shell, label: "shell",
                    detail: "npm test", startedAt: Date()),
            ],
            announcingPause: false)
        XCTAssertTrue(text.contains("<goal_checkin>"))
        XCTAssertTrue(text.contains("migrate the config"))
        XCTAssertTrue(text.contains("- bga_1 - agent - Explorer: find all call sites"))
        XCTAssertTrue(text.contains("- bg_2 - shell - shell: npm test"))
        XCTAssertTrue(text.contains("Check on their progress"))
        XCTAssertFalse(text.contains("no longer running"))
    }

    func testCheckinWithDrainedWorkTellsTheModelToContinue() {
        let text = GoalPolicy.checkinMessage(
            condition: "migrate the config", tasks: [], announcingPause: false)
        XCTAssertTrue(text.contains("no longer running"))
        XCTAssertTrue(text.contains("Continue toward the goal"))
    }

    func testThirdIdleCheckinAnnouncesThePause() {
        let text = GoalPolicy.checkinMessage(
            condition: "migrate the config",
            tasks: [agentTask(startedAt: Date())],
            announcingPause: true)
        XCTAssertTrue(text.contains("Idle check-ins are paused until you send a message"))
    }

    // MARK: - Condition clipping

    func testLongConditionsAreClippedForDisplay() {
        let long = String(repeating: "x", count: GoalPolicy.conditionPreviewLength + 10)
        let preview = GoalPolicy.conditionPreview(long)
        XCTAssertEqual(
            preview.count,
            GoalPolicy.conditionPreviewLength + "...".count)
        XCTAssertTrue(preview.hasSuffix("..."))
        let short = "short condition"
        XCTAssertEqual(GoalPolicy.conditionPreview(short), short)
    }

    // MARK: - Verdict parsing

    func testMetVerdictParses() {
        XCTAssertEqual(GoalEvaluator.parseVerdict("{\"ok\": true}"), .met(reason: nil))
        XCTAssertEqual(
            GoalEvaluator.parseVerdict("Sure! {\"ok\": true, \"reason\": \"tests pass\"} done"),
            .met(reason: "tests pass"))
    }

    func testNotMetVerdictParsesWithItsReason() {
        XCTAssertEqual(
            GoalEvaluator.parseVerdict("{\"ok\": false, \"reason\": \"two tests still fail\"}"),
            .notMet(reason: "two tests still fail"))
        // A missing reason gets the honest default rather than an empty string.
        XCTAssertEqual(
            GoalEvaluator.parseVerdict("{\"ok\": false}"),
            .notMet(reason: "the evaluator did not say why it is not met"))
    }

    func testImpossibleVerdictWinsOverOk() {
        XCTAssertEqual(
            GoalEvaluator.parseVerdict(
                "{\"ok\": false, \"impossible\": true, \"reason\": \"the file is gone\"}"),
            .impossible(reason: "the file is gone"))
    }

    func testMalformedAnswersParseAsNoVerdict() {
        XCTAssertNil(GoalEvaluator.parseVerdict("I think it is probably fine."))
        XCTAssertNil(GoalEvaluator.parseVerdict(""))
        XCTAssertNil(GoalEvaluator.parseVerdict("{\"status\": \"ok\"}"))
        // Prose around a well-formed object still parses: the tolerance is
        // first `{` through the LAST `}`.
        XCTAssertEqual(
            GoalEvaluator.parseVerdict("The answer is {\"ok\": true}. Hope that helps!"),
            .met(reason: nil))
    }

    // MARK: - Status rows

    func testStatusRowsNameTheVerdictAndTheIteration() {
        XCTAssertTrue(GoalEvaluator.statusRow(for: .notMet(reason: "lint fails"), iterations: 2)
            .contains("not yet met"))
        XCTAssertTrue(GoalEvaluator.statusRow(for: .notMet(reason: "lint fails"), iterations: 2)
            .contains("lint fails"))
        XCTAssertTrue(GoalEvaluator.statusRow(for: .met(reason: nil), iterations: 2)
            .contains("Goal met after 2 iterations"))
        XCTAssertTrue(GoalEvaluator.statusRow(for: .met(reason: nil), iterations: 1)
            .contains("Goal met after 1 iteration"))
        XCTAssertTrue(
            GoalEvaluator.statusRow(for: .impossible(reason: "impossible: reason"), iterations: 0)
                .contains("impossible"))
        XCTAssertTrue(GoalEvaluator.stallRow.contains("Goal paused"))
    }

    // MARK: - Restore

    func testRestoreKeepsOnlyConditionAndSetTime() {
        var state = goal(condition: "ship the release")
        state.iterations = 7
        state.lastReason = "not done"
        state.deferredSince = Date()
        state.checkinCount = 2
        state.idleCheckinCount = 1
        state.tokensSpent = 9000
        state.isPaused = true
        let restored = state.restoredForRelaunch()
        XCTAssertEqual(restored.condition, "ship the release")
        XCTAssertEqual(restored.setAt, state.setAt)
        XCTAssertEqual(restored.iterations, 0)
        XCTAssertNil(restored.lastReason)
        XCTAssertNil(restored.deferredSince)
        XCTAssertEqual(restored.checkinCount, 0)
        XCTAssertEqual(restored.idleCheckinCount, 0)
        XCTAssertEqual(restored.tokensSpent, 0)
        XCTAssertFalse(restored.isPaused)
    }

    private func agentTask(startedAt: Date) -> GoalBackgroundTask {
        GoalBackgroundTask(
            id: "bga_1", kind: .agent, label: "Explorer", detail: "probe",
            startedAt: startedAt)
    }
}
