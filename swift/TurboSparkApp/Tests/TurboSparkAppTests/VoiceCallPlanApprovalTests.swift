import XCTest
@testable import TurboSparkApp

/// A plan parked in the blocking ExitPlanMode gate waits for the user just
/// like a tool approval or a question card. The voice call must see it as a
/// pending approval: announce it (pause and release the microphone), count
/// the turn as busy, and (through the same predicate) refuse to auto-send.
@MainActor
final class VoiceCallPlanApprovalTests: XCTestCase {
    private var model: AppModel!

    override func setUp() async throws {
        try await super.setUp()
        model = AppModel()
    }

    override func tearDown() async throws {
        // AppModel's init installs static waiters; leave none for other suites.
        PlanModeExecutor.decisionWaiter = nil
        model = nil
        try await super.tearDown()
    }

    private func parkPlan() {
        model.pendingPlanApproval = AppModel.PendingPlanApproval(
            id: UUID(), chatID: model.selectedChatID, toolCallID: UUID(), plan: "1. Read\n2. Edit")
    }

    func testPendingPlanCountsAsPendingVoiceCallApproval() {
        XCTAssertFalse(model.hasPendingVoiceCallApproval)
        parkPlan()
        XCTAssertTrue(model.hasPendingVoiceCallApproval)
        model.pendingPlanApproval = nil
        XCTAssertFalse(model.hasPendingVoiceCallApproval)
    }

    func testPortReportsPlanAsApprovalAndBusy() {
        let port = AppModelVoiceCallPort(model: model)
        XCTAssertFalse(port.hasPendingApproval)
        parkPlan()
        XCTAssertTrue(port.hasPendingApproval)
        XCTAssertTrue(port.isTurnBusy)
    }

    /// The event the controller pauses on. Without it a plan raised while
    /// the call listens would leave the microphone open and the plan unsaid.
    func testParkedPlanRaisesTheApprovalEvent() {
        let port = AppModelVoiceCallPort(model: model)
        var raised = 0
        port.onEvent = { event in
            if case .approvalRaised = event { raised += 1 }
        }
        parkPlan()
        XCTAssertEqual(raised, 1)
    }
}
