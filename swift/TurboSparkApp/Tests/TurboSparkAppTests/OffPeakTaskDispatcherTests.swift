import Foundation
import XCTest
@testable import TurboSparkApp

/// Records the keep-awake hold the dispatcher takes and releases.
private final class StubActivityPort: OffPeakActivityPort {
    private(set) var begins = 0
    private(set) var ends = 0
    var live: Int { begins - ends }

    func begin(reason: String) -> AnyObject? {
        begins += 1
        return NSObject()
    }

    func end(_ token: AnyObject) { ends += 1 }
}

@MainActor
private final class StubHost: OffPeakHost {
    var existingChats: Set<UUID> = []
    var dispatchableChats: Set<UUID> = []
    var awaitingApproval: Set<UUID> = []
    var acceptDispatch = true
    private(set) var dispatched: [(prompt: String, chatID: UUID)] = []

    func offPeakChatExists(_ chatID: UUID) -> Bool { existingChats.contains(chatID) }
    func offPeakCanDispatch(chatID: UUID) -> Bool { dispatchableChats.contains(chatID) }
    func offPeakDispatch(prompt: String, chatID: UUID) -> Bool {
        guard acceptDispatch else { return false }
        dispatched.append((prompt, chatID))
        return true
    }
    func offPeakIsAwaitingApproval(chatID: UUID) -> Bool { awaitingApproval.contains(chatID) }
}

private func makeTempDirectory() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("offpeak-tests-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

final class OffPeakTaskStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = try makeTempDirectory()
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
    }

    func testClaimsAreFIFO() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        let first = try store.enqueue(prompt: "first", chatID: chat, origin: .user)
        let second = try store.enqueue(prompt: "second", chatID: chat, origin: .user)
        let third = try store.enqueue(prompt: "third", chatID: chat, origin: .user)

        var order: [String] = []
        for _ in 0..<3 {
            let claimed = try XCTUnwrap(store.claimNext { _ in true })
            order.append(claimed.id)
            store.finish(id: claimed.id, state: .completed, outcome: "completed")
        }
        XCTAssertEqual(order, [first.id, second.id, third.id])
    }

    func testOnlyOneTaskRunsAtATime() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        try store.enqueue(prompt: "a", chatID: chat, origin: .user)
        try store.enqueue(prompt: "b", chatID: chat, origin: .user)
        XCTAssertNotNil(store.claimNext { _ in true })
        XCTAssertNil(store.claimNext { _ in true }, "a second claim while one runs must be refused")
    }

    func testClaimIsPersistedAndRecoveredAsInterruptedWhileQueuedStaysQueued() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        let running = try store.enqueue(prompt: "running", chatID: chat, origin: .user)
        let waiting = try store.enqueue(prompt: "waiting", chatID: chat, origin: .user)
        let claimed = try XCTUnwrap(store.claimNext { _ in true })
        XCTAssertEqual(claimed.id, running.id)

        // A crash: no finish, no shutdown. A fresh store reads the file.
        let relaunched = OffPeakTaskStore(directory: directory)
        let recovered = try XCTUnwrap(relaunched.task(id: running.id))
        XCTAssertEqual(recovered.state, .interrupted)
        XCTAssertEqual(recovered.runs.first?.outcome, "crash_recovered")
        XCTAssertNotNil(recovered.runs.first?.endedAt)
        XCTAssertEqual(
            relaunched.task(id: waiting.id)?.state, .queued,
            "a task that never started must not be reported as interrupted")
        XCTAssertNil(relaunched.runningTask())
    }

    func testCapIsEnforcedAndTerminalRowsDoNotCount() throws {
        let store = OffPeakTaskStore(directory: directory, jobCap: 3)
        let chat = UUID()
        for index in 0..<3 {
            try store.enqueue(prompt: "task \(index)", chatID: chat, origin: .user)
        }
        XCTAssertThrowsError(try store.enqueue(prompt: "over", chatID: chat, origin: .user))
        let claimed = try XCTUnwrap(store.claimNext { _ in true })
        store.finish(id: claimed.id, state: .completed, outcome: "completed")
        XCTAssertNoThrow(try store.enqueue(prompt: "fits now", chatID: chat, origin: .user))
    }

    func testCapHoldsUnderConcurrentEnqueue() {
        let store = OffPeakTaskStore(directory: directory, jobCap: 20)
        let chat = UUID()
        let successes = NSLock()
        var accepted = 0
        DispatchQueue.concurrentPerform(iterations: 60) { index in
            if (try? store.enqueue(prompt: "p\(index)", chatID: chat, origin: index % 2 == 0 ? .user : .model)) != nil {
                successes.lock()
                accepted += 1
                successes.unlock()
            }
        }
        XCTAssertEqual(accepted, 20, "check-and-insert must be atomic: no overshoot, no undershoot")
        XCTAssertEqual(store.activeCount, 20)
    }

    func testCancelOnlyQueuedTasks() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        let queued = try store.enqueue(prompt: "queued", chatID: chat, origin: .user)
        let running = try store.enqueue(prompt: "running", chatID: chat, origin: .user)
        try store.cancel(id: queued.id)
        XCTAssertEqual(store.task(id: queued.id)?.state, .cancelled)
        let claimed = try XCTUnwrap(store.claimNext { _ in true })
        XCTAssertEqual(claimed.id, running.id, "a cancelled task is never claimed")
        XCTAssertThrowsError(try store.cancel(id: running.id))
        XCTAssertEqual(store.task(id: running.id)?.state, .running)
    }

    func testRequeueGoesToTheTailAndOnlyForInterrupted() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        let first = try store.enqueue(prompt: "first", chatID: chat, origin: .user)
        let second = try store.enqueue(prompt: "second", chatID: chat, origin: .user)
        XCTAssertThrowsError(try store.requeue(id: second.id), "a queued task is not interrupted")
        let claimed = try XCTUnwrap(store.claimNext { _ in true })
        store.finish(id: claimed.id, state: .interrupted, outcome: "app_quit")
        try store.requeue(id: first.id)

        let next = try XCTUnwrap(store.claimNext { _ in true })
        XCTAssertEqual(next.id, second.id, "the re-run waits behind what was already queued")
    }

    func testModelOriginCannotEnqueueFromAChatRunningAnOffPeakTask() throws {
        let store = OffPeakTaskStore(directory: directory)
        let chat = UUID()
        let other = UUID()
        try store.enqueue(prompt: "seed", chatID: chat, origin: .user)
        _ = try XCTUnwrap(store.claimNext { _ in true })

        XCTAssertThrowsError(try store.enqueue(prompt: "chain", chatID: chat, origin: .model))
        XCTAssertNoThrow(try store.enqueue(prompt: "elsewhere", chatID: other, origin: .model))
        XCTAssertNoThrow(try store.enqueue(prompt: "by the user", chatID: chat, origin: .user))
    }

    func testRejectsEmptyAndOversizedPrompts() {
        let store = OffPeakTaskStore(directory: directory)
        XCTAssertThrowsError(try store.enqueue(prompt: "  \n ", chatID: UUID(), origin: .user))
        let huge = String(repeating: "x", count: OffPeakTaskStore.maxPromptCharacters + 1)
        XCTAssertThrowsError(try store.enqueue(prompt: huge, chatID: UUID(), origin: .user))
    }

    func testUnknownStateDecodesAsInterruptedNeverQueued() throws {
        let json = #"[{"id":"x","prompt":"p","chatID":"\#(UUID().uuidString)","state":"from_the_future"}]"#
        let decoded = try JSONDecoder().decode([AppOffPeakTask].self, from: Data(json.utf8))
        XCTAssertEqual(decoded.first?.state, .interrupted)
    }

    func testToolExecutorsShareTheStoreAndCap() throws {
        let store = OffPeakTaskStore(directory: directory, jobCap: 1)
        let chat = UUID()
        let message = try OffPeakTaskExecutors.executeEnqueue(
            arguments: ["prompt": "reindex"], chatID: chat, store: store)
        XCTAssertTrue(message.contains("queued"))
        XCTAssertEqual(store.list().first?.origin, .model)
        XCTAssertThrowsError(try OffPeakTaskExecutors.executeEnqueue(
            arguments: ["prompt": "second"], chatID: chat, store: store))
        XCTAssertThrowsError(try OffPeakTaskExecutors.executeEnqueue(
            arguments: ["prompt": "ghost"], chatID: UUID(), chatEligible: false, store: store))
        XCTAssertTrue(OffPeakTaskExecutors.executeList(arguments: [:], store: store).contains("reindex"))
        let id = try XCTUnwrap(store.list().first?.id)
        XCTAssertNoThrow(try OffPeakTaskExecutors.executeCancel(arguments: ["id": id], store: store))
        XCTAssertEqual(store.task(id: id)?.state, .cancelled)
    }

    func testOffPeakToolsAreAutomationCategoryAndNotAdvertisedSilently() {
        for name in ["OffPeakEnqueue", "OffPeakList", "OffPeakCancel"] {
            XCTAssertEqual(AppToolCatalog.category(for: name), .automation, name)
            XCTAssertTrue(AppToolRegistry.isImplemented(name), name)
        }
        let readOnly = AppProject(name: "p", permissions: .readOnly)
        let call = AppToolCall(
            name: "OffPeakEnqueue", arguments: ["prompt": "x"],
            category: AppToolCatalog.category(for: "OffPeakEnqueue"))
        if case .deny = AppToolPermissionEngine.evaluate(call: call, project: readOnly) {} else {
            XCTFail("the enqueue tool must be denied in Strict Read-Only like the cron tools")
        }
    }

    func testFeatureIsOffByDefault() throws {
        XCTAssertFalse(MacAppSettings().offPeakEnabled)
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(legacy.offPeakEnabled, "settings written before this feature load as off")
    }
}

@MainActor
final class OffPeakTaskDispatcherTests: XCTestCase {
    private var directory: URL!
    private var store: OffPeakTaskStore!
    private var host: StubHost!
    private var activity: StubActivityPort!
    private var enabled = true
    private var dispatcher: OffPeakTaskDispatcher!
    private let chatA = UUID()
    private let chatB = UUID()

    override func setUpWithError() throws {
        directory = try makeTempDirectory()
        store = OffPeakTaskStore(directory: directory)
        host = StubHost()
        host.existingChats = [chatA, chatB]
        host.dispatchableChats = [chatA, chatB]
        activity = StubActivityPort()
        enabled = true
        dispatcher = OffPeakTaskDispatcher(
            store: store, host: host, activity: activity, isEnabled: { [unowned self] in self.enabled })
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
    }

    func testNothingIsAdmittedWhileTheToggleIsOff() throws {
        enabled = false
        try store.enqueue(prompt: "later", chatID: chatA, origin: .user)
        dispatcher.tick()
        XCTAssertTrue(host.dispatched.isEmpty)
        XCTAssertEqual(activity.begins, 0)
        XCTAssertEqual(store.list().first?.state, .queued)
    }

    func testDispatchTakesTheHoldAndCompletionReleasesIt() throws {
        let task = try store.enqueue(prompt: "summarize", chatID: chatA, origin: .user)
        dispatcher.tick()
        XCTAssertEqual(host.dispatched.map(\.prompt), ["summarize"])
        XCTAssertEqual(host.dispatched.first?.chatID, chatA)
        XCTAssertEqual(store.task(id: task.id)?.state, .running)
        XCTAssertEqual(activity.live, 1)
        XCTAssertTrue(dispatcher.isHoldingActivity)

        dispatcher.turnDidFinish(chatID: chatA, cancelled: false, stillBusy: false)
        let done = try XCTUnwrap(store.task(id: task.id))
        XCTAssertEqual(done.state, .completed)
        XCTAssertNotNil(done.runs.first?.endedAt)
        XCTAssertEqual(activity.live, 0, "the last running task releases the assertion")
        XCTAssertFalse(dispatcher.isHoldingActivity)
    }

    func testOneRunAtATimeAcrossTicks() throws {
        try store.enqueue(prompt: "one", chatID: chatA, origin: .user)
        try store.enqueue(prompt: "two", chatID: chatB, origin: .user)
        dispatcher.tick()
        dispatcher.tick()
        XCTAssertEqual(host.dispatched.count, 1)
        XCTAssertEqual(activity.begins, 1, "one hold for the one running task")
        dispatcher.turnDidFinish(chatID: chatA, cancelled: false, stillBusy: false)
        dispatcher.tick()
        XCTAssertEqual(host.dispatched.map(\.prompt), ["one", "two"])
    }

    func testBusyTurnDoesNotEndTheRun() throws {
        try store.enqueue(prompt: "x", chatID: chatA, origin: .user)
        dispatcher.tick()
        dispatcher.turnDidFinish(chatID: chatA, cancelled: false, stillBusy: true)
        XCTAssertEqual(store.runningTask()?.prompt, "x")
        dispatcher.turnDidFinish(chatID: chatB, cancelled: false, stillBusy: false)
        XCTAssertEqual(store.runningTask()?.prompt, "x", "another chat's turn end is not this run's")
    }

    func testFIFOSkipsAnUndispatchableHeadWithoutReorderingTheRest() throws {
        host.dispatchableChats = [chatB]
        try store.enqueue(prompt: "head, chat not open", chatID: chatA, origin: .user)
        try store.enqueue(prompt: "behind it", chatID: chatB, origin: .user)
        dispatcher.tick()
        XCTAssertEqual(host.dispatched.map(\.prompt), ["behind it"])
        XCTAssertEqual(store.list().first?.state, .queued, "the skipped head stays queued, in place")
    }

    func testRefusedDeliveryReleasesTheClaimAndTheHold() throws {
        host.acceptDispatch = false
        let task = try store.enqueue(prompt: "x", chatID: chatA, origin: .user)
        dispatcher.tick()
        let row = try XCTUnwrap(store.task(id: task.id))
        XCTAssertEqual(row.state, .queued)
        XCTAssertTrue(row.runs.isEmpty, "a delivery that never started leaves no run entry")
        XCTAssertEqual(activity.live, 0)
    }

    func testBlockedApprovalEndsTheRunAndReleasesTheHold() throws {
        let task = try store.enqueue(prompt: "needs a tool", chatID: chatA, origin: .user)
        dispatcher.tick()
        XCTAssertEqual(activity.live, 1)
        host.awaitingApproval = [chatA]
        dispatcher.tick()
        let row = try XCTUnwrap(store.task(id: task.id))
        XCTAssertEqual(row.state, .interrupted)
        XCTAssertEqual(row.runs.first?.outcome, "approval_needed")
        XCTAssertEqual(activity.live, 0)
    }

    func testDeletedChatIsSkippedWhenQueuedAndInterruptedWhenRunning() throws {
        let queued = try store.enqueue(prompt: "orphan", chatID: chatB, origin: .user)
        let live = try store.enqueue(prompt: "live", chatID: chatA, origin: .user)
        host.existingChats = [chatA]
        dispatcher.tick()
        XCTAssertEqual(store.task(id: queued.id)?.state, .skipped)
        XCTAssertEqual(host.dispatched.map(\.prompt), ["live"])

        host.existingChats = []
        dispatcher.tick()
        XCTAssertEqual(store.task(id: live.id)?.state, .interrupted)
        XCTAssertEqual(activity.live, 0)
    }

    func testStopRecordsInterruptedAndShutdownIsRecordedBeforeRelaunch() throws {
        let stopped = try store.enqueue(prompt: "stop me", chatID: chatA, origin: .user)
        dispatcher.tick()
        dispatcher.turnDidFinish(chatID: chatA, cancelled: true, stillBusy: false)
        XCTAssertEqual(store.task(id: stopped.id)?.state, .interrupted)
        XCTAssertEqual(store.task(id: stopped.id)?.runs.first?.outcome, "stopped")

        let quitting = try store.enqueue(prompt: "quit me", chatID: chatA, origin: .user)
        dispatcher.tick()
        XCTAssertEqual(activity.live, 1)
        dispatcher.shutdown()
        XCTAssertEqual(store.task(id: quitting.id)?.state, .interrupted)
        XCTAssertEqual(store.task(id: quitting.id)?.runs.first?.outcome, "app_quit")
        XCTAssertEqual(activity.live, 0, "quit releases the assertion")
    }

    func testTurningTheToggleOffStopsNewAdmissionsButLetsTheRunningTaskFinish() throws {
        try store.enqueue(prompt: "running", chatID: chatA, origin: .user)
        try store.enqueue(prompt: "waiting", chatID: chatB, origin: .user)
        dispatcher.tick()
        enabled = false
        dispatcher.turnDidFinish(chatID: chatA, cancelled: false, stillBusy: false)
        dispatcher.tick()
        XCTAssertEqual(host.dispatched.map(\.prompt), ["running"])
        XCTAssertEqual(store.list().last?.state, .queued)
    }
}
