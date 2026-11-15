import XCTest
@testable import TurboSparkApp

/// OC-07. Gating (who sees the flow), the step list, the flag, and the splash
/// timing rules. The "absent step" and "existing user" rules are the subtle
/// ones; each was mutation-checked.
@MainActor
final class FirstRunCoordinatorTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("FirstRunTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore() -> FirstRunStore {
        FirstRunStore(fileURL: directory.appendingPathComponent("first_run.json"))
    }

    // MARK: Gating

    func testFreshInstallRuns() {
        XCTAssertTrue(FirstRunDecision.shouldRun(
            flagCompleted: false, hasInstalledModel: false, hasChatHistory: false))
    }

    func testCompletedFlagNeverReplays() {
        XCTAssertFalse(FirstRunDecision.shouldRun(
            flagCompleted: true, hasInstalledModel: false, hasChatHistory: false))
    }

    /// An upgrading user has no flag (the flag is new) but has history or a
    /// model: they must not be pushed through the flow.
    func testExistingUserIsSkippedWithoutAFlag() {
        XCTAssertFalse(FirstRunDecision.shouldRun(
            flagCompleted: false, hasInstalledModel: false, hasChatHistory: true))
        XCTAssertFalse(FirstRunDecision.shouldRun(
            flagCompleted: false, hasInstalledModel: true, hasChatHistory: false))
    }

    func testCoordinatorDoesNotStartForAnExistingUser() {
        let coordinator = FirstRunCoordinator(store: makeStore())
        coordinator.begin(hasInstalledModel: false, hasChatHistory: true)
        XCTAssertFalse(coordinator.isActive)
    }

    func testCoordinatorStartsOnAFreshInstallAndOnlyOncePerProcess() {
        let coordinator = FirstRunCoordinator(store: makeStore())
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        XCTAssertTrue(coordinator.isActive)
        XCTAssertEqual(coordinator.current, .welcome)
        coordinator.skipAll()
        XCTAssertFalse(coordinator.isActive)
        // A remount (vault relock) calls begin again: it must stay ended.
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        XCTAssertFalse(coordinator.isActive)
    }

    // MARK: Step list

    func testStepListDropsFirstModelWhenOneIsInstalled() {
        XCTAssertEqual(
            FirstRunFlow.steps(hasInstalledModel: false, importWizardAvailable: false),
            [.welcome, .firstModel, .primer])
        XCTAssertEqual(
            FirstRunFlow.steps(hasInstalledModel: true, importWizardAvailable: false),
            [.welcome, .primer])
    }

    func testImportStepIsAbsentUntilTheWizardExists() {
        XCTAssertFalse(FirstRunFlow.importWizardAvailable)
        XCTAssertFalse(FirstRunFlow.steps(hasInstalledModel: false).contains(.importHandoff))
        XCTAssertEqual(
            FirstRunFlow.steps(hasInstalledModel: false, importWizardAvailable: true),
            [.welcome, .firstModel, .importHandoff, .primer])
    }

    func testModelInstalledMidFlowRemovesTheStepAndKeepsPlace() {
        let coordinator = FirstRunCoordinator(store: makeStore())
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        coordinator.advance()
        XCTAssertEqual(coordinator.current, .firstModel)
        coordinator.installedModelChanged(true)
        XCTAssertEqual(coordinator.steps, [.welcome, .primer])
        // The step on screen went away: the next surviving step shows.
        XCTAssertEqual(coordinator.current, .primer)

        let other = FirstRunCoordinator(store: makeStore())
        other.begin(hasInstalledModel: false, hasChatHistory: false)
        other.installedModelChanged(true)
        XCTAssertEqual(other.current, .welcome)
        XCTAssertEqual(other.steps, [.welcome, .primer])
    }

    // MARK: Movement and the flag

    func testBackAndSkipStepDoNotWriteTheFlag() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        coordinator.skipStep()
        XCTAssertEqual(coordinator.current, .firstModel)
        coordinator.back()
        XCTAssertEqual(coordinator.current, .welcome)
        XCTAssertFalse(coordinator.canGoBack)
        XCTAssertNil(store.load(), "quitting mid-flow must not set the flag")
    }

    func testFinishingSetsTheFlagAndLandsInANewChat() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        while coordinator.isActive { coordinator.advance() }
        XCTAssertEqual(exits, [.newChat])
        XCTAssertEqual(store.load()?.completed, true)
    }

    func testSkippingEveryStepIsAValidFirstRun() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        while coordinator.isActive { coordinator.skipStep() }
        XCTAssertEqual(exits, [.newChat])
        XCTAssertEqual(store.load()?.completed, true)
    }

    func testSkipSetupSetsTheFlagImmediately() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        coordinator.skipAll()
        XCTAssertEqual(exits, [.newChat])
        XCTAssertEqual(store.load()?.completed, true)
    }

    func testBrowseModelsEndsTheFlagAndGoesToDiscover() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        coordinator.advance()
        coordinator.browseModels()
        XCTAssertEqual(exits, [.modelHub])
        XCTAssertEqual(store.load()?.completed, true)
    }

    func testReturningUserNeverSeesTheFlowAfterCompletion() {
        let store = makeStore()
        let first = FirstRunCoordinator(store: store)
        first.begin(hasInstalledModel: false, hasChatHistory: false)
        first.skipAll()
        // A relaunch: a new coordinator, same store, even with no history.
        let relaunch = FirstRunCoordinator(store: store)
        relaunch.begin(hasInstalledModel: false, hasChatHistory: false)
        XCTAssertFalse(relaunch.isActive)
    }

    // MARK: Replay (Help > Show Welcome Tour)

    func testReplayRunsForAnExistingUserAndLeavesTheFlagUntouched() {
        let store = makeStore()
        let coordinator = FirstRunCoordinator(store: store)
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        // An existing user: begin declines.
        coordinator.begin(hasInstalledModel: true, hasChatHistory: true)
        XCTAssertFalse(coordinator.isActive)
        coordinator.replay(hasInstalledModel: true)
        XCTAssertTrue(coordinator.isActive)
        XCTAssertEqual(coordinator.steps, [.welcome, .primer])
        while coordinator.isActive { coordinator.advance() }
        XCTAssertEqual(exits, [.stay])
        XCTAssertNil(store.load(), "a replay must not write the first-run flag")
        XCTAssertFalse(coordinator.isReplay)
    }

    func testReplayDoesNotClearACompletedFlag() {
        let store = makeStore()
        store.markCompleted()
        let coordinator = FirstRunCoordinator(store: store)
        coordinator.replay(hasInstalledModel: false)
        coordinator.skipAll()
        XCTAssertEqual(store.load()?.completed, true)
        let relaunch = FirstRunCoordinator(store: store)
        relaunch.begin(hasInstalledModel: false, hasChatHistory: false)
        XCTAssertFalse(relaunch.isActive)
    }

    func testReplayIsIgnoredWhileAFlowIsOpen() {
        let coordinator = FirstRunCoordinator(store: makeStore())
        coordinator.begin(hasInstalledModel: false, hasChatHistory: false)
        coordinator.advance()
        coordinator.replay(hasInstalledModel: false)
        XCTAssertEqual(coordinator.index, 1)
        XCTAssertFalse(coordinator.isReplay)
    }

    func testReplayBrowseModelsStillGoesToDiscover() {
        let coordinator = FirstRunCoordinator(store: makeStore())
        var exits: [FirstRunExit] = []
        coordinator.onExit = { exits.append($0) }
        coordinator.replay(hasInstalledModel: false)
        coordinator.advance()
        coordinator.browseModels()
        XCTAssertEqual(exits, [.modelHub])
    }

    // MARK: Persistence

    func testCorruptFlagFileIsQuarantinedNotOverwritten() throws {
        let store = makeStore()
        try Data("{ not json".utf8).write(to: store.fileURL)
        XCTAssertNil(store.load())
        XCTAssertFalse(FileManager.default.fileExists(atPath: store.fileURL.path))
        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(
            names.contains { $0.hasPrefix("first_run.corrupt-") },
            "the unreadable file must be kept beside the fresh one: \(names)")
        // The corrupt-flag case replays nothing for a person with history.
        XCTAssertFalse(FirstRunDecision.shouldRun(
            flagCompleted: false, hasInstalledModel: false, hasChatHistory: true))
    }

    func testFlagRoundTrips() {
        let store = makeStore()
        XCTAssertNil(store.load())
        XCTAssertTrue(store.markCompleted(now: Date(timeIntervalSince1970: 1_700_000_000)))
        XCTAssertEqual(store.load()?.completed, true)
        XCTAssertEqual(store.load()?.completedAt, Date(timeIntervalSince1970: 1_700_000_000))
    }

    func testDefaultFlagIsAppLevelNotProfileScoped() {
        XCTAssertEqual(
            FirstRunStore().fileURL.deletingLastPathComponent().standardizedFileURL.path,
            AppStorageRoot.machineRoot.standardizedFileURL.path)
    }
}

final class SplashPolicyTests: XCTestCase {
    func testReadyBeforeTheDelayNeverDraws() {
        XCTAssertFalse(SplashPolicy.shouldDraw(elapsed: 0.1, stateReady: false, reduceMotion: false))
        XCTAssertFalse(SplashPolicy.shouldDraw(elapsed: 1.0, stateReady: true, reduceMotion: false))
        XCTAssertEqual(SplashPolicy.drawDelay(elapsed: 0.1), 0.3, accuracy: 0.0001)
        XCTAssertEqual(SplashPolicy.drawDelay(elapsed: 2), 0)
    }

    func testSlowLoadDrawsAfterTheDelay() {
        XCTAssertTrue(SplashPolicy.shouldDraw(
            elapsed: SplashPolicy.appearDelay, stateReady: false, reduceMotion: false))
    }

    func testReduceMotionNeverDraws() {
        XCTAssertFalse(SplashPolicy.shouldDraw(elapsed: 1.5, stateReady: false, reduceMotion: true))
    }

    func testOnceDrawnItStaysForTheMinimumThenLeaves() {
        XCTAssertEqual(
            SplashPolicy.dismissWait(elapsed: 1.0, shownFor: 0.1, stateReady: true),
            SplashPolicy.minimumOnScreen - 0.1, accuracy: 0.0001)
        XCTAssertEqual(SplashPolicy.dismissWait(elapsed: 1.5, shownFor: 0.5, stateReady: true), 0)
    }

    func testNeverDrawnDismissesAtOnceWhenReady() {
        XCTAssertEqual(SplashPolicy.dismissWait(elapsed: 0.05, shownFor: nil, stateReady: true), 0)
    }

    func testNotReadyWaitsUntilTheCapThenGivesUp() {
        XCTAssertEqual(
            SplashPolicy.dismissWait(elapsed: 1, shownFor: 0.5, stateReady: false),
            SplashPolicy.maximumDisplay - 1, accuracy: 0.0001)
        XCTAssertEqual(
            SplashPolicy.dismissWait(
                elapsed: SplashPolicy.maximumDisplay, shownFor: 2, stateReady: false), 0)
        XCTAssertFalse(SplashPolicy.shouldDraw(
            elapsed: SplashPolicy.maximumDisplay, stateReady: false, reduceMotion: false))
    }
}
