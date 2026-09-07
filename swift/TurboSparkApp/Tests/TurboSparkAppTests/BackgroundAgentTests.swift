import XCTest

@testable import TurboSparkApp

/// The background subagent registry (`AppModel+Subagents.swift`): the
/// notification text, the event routing that keeps a background record
/// alive across `finished` (unlike a foreground one), stopping, dismissal,
/// and the parking rule for notifications. No live session: everything here
/// is state machine and formatting, which is where the contract lives.
@MainActor
final class BackgroundAgentTests: XCTestCase {
    var appModel: AppModel!

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    private func makeChat(_ appModel: AppModel) -> UUID {
        let chat = AppChat(title: "Bg")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return chat.id
    }

    private func result(status: String = "completed", answer: String = "done") -> SubagentRunResult {
        SubagentRunResult(
            agentName: "explore", status: status, finalResponse: answer,
            totalTurns: 2, totalToolCalls: 3, durationSeconds: 4.5, runID: "run-9")
    }

    // MARK: - the notification text

    func testTheNotificationCarriesIdStatusAndResultMarkers() {
        let note = BackgroundAgentNotification.text(
            id: "bga_1", status: "completed", agentName: "explore",
            displayName: "Explore", result: result())
        for marker in ["<task-notification>", "</task-notification>", "task_id: bga_1",
                       "status: completed", "<result>", "</result>", "done",
                       "turns: 2, tool_calls: 3, duration_s: 4.5"] {
            XCTAssertTrue(note.contains(marker), "Missing \(marker) in:\n\(note)")
        }
        XCTAssertTrue(note.hasPrefix("<task-notification>"), "The turn is the notification, nothing before it.")

        let empty = BackgroundAgentNotification.text(
            id: "bga_2", status: "failed", agentName: "explore", displayName: "Explore",
            result: result(status: "failed", answer: "   "))
        XCTAssertTrue(empty.contains("(No output.)"), "An empty answer is named, not pasted as blank.")
    }

    // MARK: - event routing: foreground vs background lifetimes

    func testAForegroundRunIsCreatedLazilyAndRemovedOnFinish() {
        let chatID = makeChat(appModel)
        let key = UUID().uuidString
        appModel.applySubagentEvent(key, .started(
            agentName: "explore", displayName: "Explore", taskDescription: "d",
            promptHead: "p", chatID: chatID))
        XCTAssertEqual(appModel.liveSubagentRuns[key]?.chatID, chatID,
                       "The card binds to the RUN's chat, not the selection.")

        appModel.applySubagentEvent(key, .content("hello"))
        appModel.applySubagentEvent(key, .finished(status: "completed"))
        XCTAssertNil(appModel.liveSubagentRuns[key],
                     "A foreground record dies with its event stream; the tool card takes over.")
    }

    func testABackgroundRecordSurvivesItsOwnFinishEvent() {
        _ = makeChat(appModel)
        let id = "bga_1"
        appModel.backgroundAgentRuns[id] = SubagentRunState(
            id: id, mode: .background, chatID: appModel.selectedChatID)

        appModel.applySubagentEvent(id, .started(
            agentName: "explore", displayName: "Explore", taskDescription: "",
            promptHead: "p", chatID: appModel.selectedChatID))
        appModel.applySubagentEvent(id, .finished(status: "completed"))

        let state = appModel.backgroundAgentRuns[id]
        XCTAssertNotNil(state, "A background card is the run's only record; an event must not remove it.")
        XCTAssertEqual(state?.status, "completed")
        XCTAssertNil(appModel.pendingTaskNotifications[appModel.selectedChatID],
                     "No result was recorded yet, so nothing may inject.")
    }

    func testForegroundKeysNeverLeakIntoTheBackgroundRegistry() {
        _ = makeChat(appModel)
        let key = UUID().uuidString
        appModel.applySubagentEvent(key, .started(
            agentName: "explore", displayName: "Explore", taskDescription: "",
            promptHead: "p", chatID: appModel.selectedChatID))
        appModel.applySubagentEvent(key, .finished(status: "completed"))
        XCTAssertNil(appModel.backgroundAgentRuns[key])
    }

    // MARK: - stopping

    func testStoppingAnUnknownIdThrowsWithTheKnownIds() async {
        appModel.backgroundAgentRuns["bga_7"] = SubagentRunState(
            id: "bga_7", mode: .background, chatID: nil)
        do {
            _ = try await appModel.stopBackgroundAgent("bga_9")
            XCTFail("An unknown id must throw.")
        } catch {
            XCTAssertTrue("\(error)".contains("bga_7"), "The error names an id that DOES resolve.")
        }
    }

    func testStoppingWhenNothingIsRegisteredNamesThat() async {
        do {
            _ = try await appModel.stopBackgroundAgent("bga_9")
            XCTFail("An unknown id must throw.")
        } catch {
            XCTAssertTrue("\(error)".contains("None are registered"))
        }
    }

    func testStoppingAnAlreadyFinishedRunSaysSoInsteadOfThrowing() async {
        let state = SubagentRunState(id: "bga_1", mode: .background, chatID: nil)
        state.apply(.finished(status: "completed"))
        appModel.backgroundAgentRuns["bga_1"] = state
        let message = try? await appModel.stopBackgroundAgent("bga_1")
        XCTAssertTrue(message?.contains("already finished") ?? false)
    }

    func testDismissalRefusesARunningRunAndClearsAFinishedOne() {
        let running = SubagentRunState(id: "bga_1", mode: .background, chatID: nil)
        appModel.backgroundAgentRuns["bga_1"] = running
        appModel.dismissBackgroundAgent("bga_1")
        XCTAssertNotNil(appModel.backgroundAgentRuns["bga_1"],
                        "A live id must keep resolving for stop_agent.")

        running.apply(.finished(status: "completed"))
        appModel.dismissBackgroundAgent("bga_1")
        XCTAssertNil(appModel.backgroundAgentRuns["bga_1"])
    }

    // MARK: - vocabulary drift

    func testTheStopAgentToolIsAdvertisedAndImplementedTogether() {
        for definition in AgentToolDefinitions.all {
            XCTAssertTrue(
                AppToolRegistry.isImplemented(definition.function.name),
                "'\(definition.function.name)' is advertised without an executor (T5).")
        }
        XCTAssertTrue(AppToolRegistry.isImplemented("stop_agent"))
        XCTAssertTrue(AppToolRegistry.isImplemented("agentstop"))
    }

    // MARK: - notification parking and the drain

    func testANotificationParksWhileTheChatIsBusyAndDrainsWhenIdle() {
        let chatID = makeChat(appModel)
        appModel.pendingTaskNotifications[chatID] = ["<task-notification>note</task-notification>"]

        // A chat "busy" for the injection rule: an approval card is up.
        appModel.pendingToolCall = AppToolCall(name: "run_command", arguments: [:], category: .terminal)
        appModel.drainPendingTaskNotificationsIfIdle(chatID: chatID)
        XCTAssertEqual(appModel.chats[0].messages.count, 0, "Busy: nothing injects.")
        XCTAssertEqual(appModel.pendingTaskNotifications[chatID]?.count, 1)

        appModel.pendingToolCall = nil
        appModel.drainPendingTaskNotificationsIfIdle(chatID: chatID)
        XCTAssertEqual(appModel.chats[0].messages.count, 1, "Idle: the parked note becomes a user turn.")
        guard let first = appModel.chats[0].messages.first else { return }
        XCTAssertEqual(first.role, .user)
        XCTAssertTrue(first.content.contains("<task-notification>"))
        XCTAssertNil(appModel.pendingTaskNotifications[chatID])
    }

    func testCanInjectRequiresAnExistingChatAndAClearedCard() {
        let chatID = makeChat(appModel)
        XCTAssertTrue(appModel.canInjectTaskNotification(into: chatID))

        appModel.generating = true
        XCTAssertFalse(appModel.canInjectTaskNotification(into: chatID))
        appModel.generating = false

        appModel.submitting = true
        XCTAssertFalse(appModel.canInjectTaskNotification(into: chatID))
        appModel.submitting = false

        appModel.pendingToolCall = AppToolCall(name: "agent", arguments: [:], category: .automation)
        XCTAssertFalse(appModel.canInjectTaskNotification(into: chatID),
                       "Injecting under an approval card would eat it: approval refuses while generating.")
        appModel.pendingToolCall = nil

        XCTAssertFalse(appModel.canInjectTaskNotification(into: UUID()),
                       "A chat that no longer exists receives nothing.")
    }
}
