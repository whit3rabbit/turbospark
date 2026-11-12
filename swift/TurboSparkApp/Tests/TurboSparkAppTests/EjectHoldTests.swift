import Foundation
import XCTest

@testable import TurboSparkApp

/// Second fix pass, finding 1 of tasks 5.11 and 5.12: killing the background
/// agents of a chat during an eject must not start a new model turn, because
/// that turn holds the session again and the first eject ends in
/// `unloadBlocked` (requirement 6.1 asks for one action, not two).
///
/// Everything that can start a turn on its own goes through
/// `canInjectChatInput`: the completion watcher of a background agent, the
/// queued and scheduled prompts, the parked notifications drained at the tail
/// of a cancelled turn, and goal check-ins. While a release is in progress they
/// PARK. They are not dropped: they drain at the next idle tail after the eject
/// (a later turn's tail, or selecting the chat), exactly as they do after a
/// busy turn.
///
/// A real turn needs a model session, which a unit test cannot build, so
/// `executeGenerationTurn` returns at its guard here. What these tests observe
/// instead is what comes right before it: the notification row appended to the
/// transcript, and the queue entry consumed. Both happen only when the gate
/// lets a turn start.
final class EjectHoldTests: LibraryStateTestCase {
    private typealias Fx = EjectFixtures

    private var holders: SessionHolders!

    override func setUp() async throws {
        try await super.setUp()
        holders = SessionHolders()
    }

    override func tearDown() async throws {
        holders.openAll()
        holders = nil
        try await super.tearDown()
    }

    /// A model with one selected chat, so a notification for it can inject.
    private func makeChatModel() async -> (AppModel, UUID) {
        let model = await makeModel(using: makeState())
        let chat = AppChat(title: "Eject")
        model.chats = [chat]
        model.selectedChatID = chat.id
        return (model, chat.id)
    }

    private func notificationRows(_ model: AppModel, _ chatID: UUID) -> Int {
        model.turnMessages(for: chatID).filter { $0.content.contains("<task-notification>") }.count
    }

    // MARK: - The gate

    func testWhileAReleaseIsInProgressNothingMayInjectIntoTheChat() async {
        let (model, chatID) = await makeChatModel()
        XCTAssertTrue(model.canInjectChatInput(into: chatID), "control: an idle chat takes input")
        XCTAssertTrue(model.canInjectTaskNotification(into: chatID))

        model.beginEjectHold()
        XCTAssertTrue(model.isEjecting)
        XCTAssertFalse(model.canInjectChatInput(into: chatID), "queued and scheduled prompts park")
        XCTAssertFalse(model.canInjectTaskNotification(into: chatID), "and so do notifications")

        model.endEjectHold()
        XCTAssertFalse(model.isEjecting)
        XCTAssertTrue(model.canInjectChatInput(into: chatID))
        XCTAssertTrue(model.canInjectTaskNotification(into: chatID))
    }

    func testTheHoldNestsAndNeverGoesNegative() async {
        let (model, chatID) = await makeChatModel()
        model.beginEjectHold()
        model.beginEjectHold()
        model.endEjectHold()
        XCTAssertFalse(model.canInjectChatInput(into: chatID), "one release is still under way")
        model.endEjectHold()
        XCTAssertTrue(model.canInjectChatInput(into: chatID))
        model.endEjectHold()
        XCTAssertFalse(model.isEjecting)
        model.beginEjectHold()
        XCTAssertTrue(model.isEjecting, "an unbalanced end did not leave a negative count behind")
        model.endEjectHold()
    }

    func testTheLiveSourceHoldsForTheReleaseOfAChatModelOnly() async {
        let (model, _) = await makeChatModel()
        let source = AppModelLoadedModels(model: model, patience: .milliseconds(50))

        source.beginRelease(of: [Fx.server, Fx.audio])
        XCTAssertFalse(model.isEjecting, "another drive's server or audio eject leaves the chat alone")
        source.endRelease(of: [Fx.server, Fx.audio])

        source.beginRelease(of: [Fx.chat, Fx.server])
        XCTAssertTrue(model.isEjecting, "a chat model in the release holds the chat")
        source.endRelease(of: [Fx.chat, Fx.server])
        XCTAssertFalse(model.isEjecting)
    }

    // MARK: - Background agents of the selected chat (the finding)

    /// A background agent whose `chatID` is the SELECTED chat, killed by the
    /// stop step. Its completion watcher would inject a notification and start
    /// a turn; the first eject must instead unload and eject, and the
    /// notification must wait.
    func testKillingTheSelectedChatsBackgroundAgentInjectsNothingAndTheFirstEjectUnloads() async {
        let (model, chatID) = await makeChatModel()
        let owner = ProbeHolder()
        let held = holders.hold(on: model, chatID: chatID, owner: owner)
        let log = EjectLog()
        let source = AppModelLoadedModels(
            model: model, patience: .seconds(3),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))
        let ejector = FakeEjector(log: log)
        let notices = NoticeRecorder(log: log)
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: ejector, locator: FakeVolumeLocator(), present: { notices.add($0) })
        await coordinator.refreshSnapshot()

        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle(timeout: 10)

        XCTAssertEqual(notices.all, [.ejected(models: ["qwen"])], "the first attempt unloads: no unloadBlocked")
        XCTAssertEqual(ejector.calls, [Fx.t7Disk], "and ejects once")
        XCTAssertEqual(model.backgroundAgentRuns["bga_t1"]?.status, "killed")
        XCTAssertEqual(notificationRows(model, chatID), 0, "no notification was injected, so no turn was started")
        XCTAssertEqual(model.pendingTaskNotifications[chatID]?.count, 1, "it is parked")
        XCTAssertNil(held.value, "the session is really gone")
        XCTAssertFalse(model.isEjecting, "the hold ended with the release")
    }

    /// Without the hold the very same stop step injects the notification, so
    /// the assertions above can fail.
    func testWithoutTheHoldTheKilledAgentsNotificationIsInjected() async {
        let (model, chatID) = await makeChatModel()
        _ = holders.hold(on: model, chatID: chatID)
        let source = AppModelLoadedModels(model: model, patience: .seconds(3))

        await source.stopGeneration(in: .chat, models: [Fx.chat])
        let injected = await eventuallyAsync { self.notificationRows(model, chatID) == 1 }
        XCTAssertTrue(injected, "the control: an idle, selected chat takes the notification")
        XCTAssertNil(model.pendingTaskNotifications[chatID])
    }

    func testTheParkedNotificationIsDeliveredAtTheNextIdleTailAfterTheEject() async {
        let (model, chatID) = await makeChatModel()
        let owner = ProbeHolder()
        _ = holders.hold(on: model, chatID: chatID, owner: owner)
        let source = AppModelLoadedModels(
            model: model, patience: .seconds(3),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: FakeEjector(log: EjectLog()), locator: FakeVolumeLocator(),
            present: { _ in })
        await coordinator.refreshSnapshot()
        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle(timeout: 10)
        XCTAssertEqual(notificationRows(model, chatID), 0)

        // The next idle tail (here called directly; a turn's tail and
        // selecting the chat both make this call).
        model.drainPendingTaskNotificationsIfIdle(chatID: chatID)

        XCTAssertNil(model.pendingTaskNotifications[chatID], "not dropped: delivered")
        XCTAssertEqual(notificationRows(model, chatID), 1)
    }

    // MARK: - Queued prompts and parked notifications at a cancelled turn's tail

    func testAQueuedUserPromptParksAtTheTailOfACancelledTurnDuringTheRelease() async {
        let (model, chatID) = await makeChatModel()
        model.pendingUserMessages[chatID] = [QueuedUserPrompt(text: "later")]
        model.generating = true
        model.generationEpoch = 7

        model.beginEjectHold()
        model.finishTurnTail(myEpoch: 7, turnChatID: chatID, session: nil)
        XCTAssertFalse(model.generating, "the turn ended")
        XCTAssertEqual(
            model.pendingUserMessages[chatID]?.map(\.text), ["later"],
            "the queued prompt did not start a turn: generating did not flip back on")
        XCTAssertTrue(model.promptText.isEmpty, "and it was not moved into the composer")

        model.endEjectHold()
        model.drainPendingUserMessagesIfIdle(chatID: chatID)
        XCTAssertNil(model.pendingUserMessages[chatID], "it drains once the release is over")
        XCTAssertEqual(model.promptText, "later")
    }

    func testWithoutTheHoldTheQueuedPromptStartsAtTheTail() async {
        let (model, chatID) = await makeChatModel()
        model.pendingUserMessages[chatID] = [QueuedUserPrompt(text: "later")]
        model.generating = true
        model.generationEpoch = 7

        model.finishTurnTail(myEpoch: 7, turnChatID: chatID, session: nil)
        XCTAssertNil(model.pendingUserMessages[chatID], "the control: the tail consumed it")
    }

    func testAParkedNotificationParksAtTheTailOfACancelledTurnDuringTheRelease() async {
        let (model, chatID) = await makeChatModel()
        model.pendingTaskNotifications[chatID] = [PendingTaskNotification(kind: .taskNotification, note: "n")]
        model.generating = true
        model.generationEpoch = 3

        model.beginEjectHold()
        model.finishTurnTail(myEpoch: 3, turnChatID: chatID, session: nil)
        XCTAssertEqual(model.pendingTaskNotifications[chatID]?.count, 1)
        XCTAssertTrue(model.turnMessages(for: chatID).isEmpty, "nothing was appended or answered")
        model.endEjectHold()

        model.drainPendingTaskNotificationsIfIdle(chatID: chatID)
        XCTAssertNil(model.pendingTaskNotifications[chatID])
        XCTAssertEqual(model.turnMessages(for: chatID).count, 1)
    }

    // MARK: - The stop step runs to quiescence

    /// A turn that ends and is replaced by another in the same hop, which is
    /// what the tail drain of a queued prompt does. The old loop waited for
    /// `generating` to be false once, saw the successor and kept waiting
    /// without cancelling it.
    func testTheChatStopStepCancelsATurnThatIsReplacedByAnotherAtItsTail() async {
        let (model, _) = await makeChatModel()
        let ref = holders.holdByTurn(on: model, restarts: true)
        let source = AppModelLoadedModels(model: model, patience: .seconds(3))

        let started = Date()
        await source.stopGeneration(in: .chat, models: [Fx.chat])

        XCTAssertFalse(model.generating, "the replacement turn was cancelled as well")
        XCTAssertNil(model.runTask)
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "without waiting out the patience")
        let gone = await eventuallyAsync { ref.value == nil }
        XCTAssertTrue(gone)
    }

    func testTheReleaseCancelsAStartedTurnThatStillHoldsTheSession() async {
        let (model, _) = await makeChatModel()
        let owner = ProbeHolder()
        let ref = holders.holdByTurn(on: model, owner: owner)
        let source = AppModelLoadedModels(
            model: model, patience: .seconds(3),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))

        // No stop step first: the release itself must cancel the turn.
        let started = Date()
        await source.unload(.chat, models: [Fx.chat])

        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "the turn was cancelled, not waited out")
        XCTAssertNil(ref.value, "the session is gone")
        XCTAssertTrue(source.loadedModels().isEmpty, "and not listed as retired")
    }
}
