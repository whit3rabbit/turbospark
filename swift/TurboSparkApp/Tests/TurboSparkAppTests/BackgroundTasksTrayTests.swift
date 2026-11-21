import XCTest

@testable import TurboSparkApp

/// The status-bar background tasks tray (component audit, background-tasks):
/// one merged list over background subagents and background shells, the
/// chip's summary, and which per-task actions a row offers. The actions are
/// the strips' own, scoped the way the strips scope them, so a tray row in
/// another chat offers Open instead of a Stop the model layer would refuse.
@MainActor
final class BackgroundTasksTrayTests: XCTestCase {

    private let chatA = UUID()
    private let chatB = UUID()

    private func entry(
        _ id: String, kind: BackgroundTaskKind = .shell, running: Bool = true,
        startedAt: TimeInterval, chatID: UUID? = nil, canDismiss: Bool = false
    ) -> BackgroundTaskEntry {
        BackgroundTaskEntry(
            taskID: id, kind: kind, title: id, detail: nil,
            startedAt: Date(timeIntervalSince1970: startedAt),
            status: running ? "running" : "completed", chatID: chatID,
            canDismiss: canDismiss)
    }

    // MARK: - Merge and sort

    func testRunningTasksComeFirstThenMostRecent() {
        let merged = BackgroundTasksTray.sorted([
            entry("old-done", kind: .agent, running: false, startedAt: 10),
            entry("old-run", startedAt: 20),
            entry("new-done", kind: .agent, running: false, startedAt: 90),
            entry("new-run", kind: .agent, startedAt: 80),
        ])
        XCTAssertEqual(merged.map(\.taskID), ["new-run", "old-run", "new-done", "old-done"])
    }

    func testEqualStartTimesBreakTiesByKindThenIDSoOrderIsStable() {
        let merged = BackgroundTasksTray.sorted([
            entry("bash_2", startedAt: 5),
            entry("bga_1", kind: .agent, startedAt: 5),
            entry("bash_1", startedAt: 5),
        ])
        XCTAssertEqual(merged.map(\.taskID), ["bga_1", "bash_1", "bash_2"])
    }

    func testMergeTakesAgentsAndShellsTogether() {
        let run = SubagentRunState(
            id: "bga_1", mode: .background, chatID: chatA,
            displayName: "Explorer", taskDescription: "Map the repo")
        let shell = BackgroundShellSummary(
            id: "bash_1", commandHead: "npm run dev", description: nil,
            startedAt: Date.distantFuture, chatID: nil, elapsedSeconds: 0)
        let merged = BackgroundTasksTray.merge(agents: [run], shells: [shell])
        XCTAssertEqual(merged.map(\.taskID), ["bash_1", "bga_1"])
        XCTAssertEqual(merged.map(\.kind), [.shell, .agent])
    }

    // MARK: - Mapping

    func testAgentTitleFallsBackFromDisplayNameToAgentNameToID() {
        let named = SubagentRunState(
            id: "bga_1", mode: .background, chatID: nil,
            agentName: "explore", displayName: "Explorer", promptHead: "look around")
        XCTAssertEqual(BackgroundTaskEntry(agent: named).title, "Explorer")
        XCTAssertEqual(BackgroundTaskEntry(agent: named).detail, "look around")

        let bare = SubagentRunState(id: "bga_2", mode: .background, chatID: nil, agentName: "explore")
        XCTAssertEqual(BackgroundTaskEntry(agent: bare).title, "explore")
        XCTAssertNil(BackgroundTaskEntry(agent: bare).detail)

        let empty = SubagentRunState(id: "bga_3", mode: .background, chatID: nil)
        XCTAssertEqual(BackgroundTaskEntry(agent: empty).title, "bga_3")
    }

    func testFinishedAgentIsDismissableOnlyOnceCompletionIsRecorded() {
        let run = SubagentRunState(id: "bga_1", mode: .background, chatID: nil)
        XCTAssertTrue(BackgroundTaskEntry(agent: run).isRunning)
        XCTAssertFalse(BackgroundTaskEntry(agent: run).canDismiss)
        run.status = "completed"
        XCTAssertFalse(BackgroundTaskEntry(agent: run).isRunning)
        // The `.finished` event flips status before completion is recorded;
        // dismissing in that window drops the task notification.
        XCTAssertFalse(BackgroundTaskEntry(agent: run).canDismiss)
        run.isRecordedComplete = true
        XCTAssertTrue(BackgroundTaskEntry(agent: run).canDismiss)
    }

    func testShellTitleIsTheDescriptionWithTheCommandAsDetail() {
        let described = BackgroundShellSummary(
            id: "bash_1", commandHead: "npm run dev", description: "Dev server",
            startedAt: Date(), chatID: nil, elapsedSeconds: 0)
        XCTAssertEqual(BackgroundTaskEntry(shell: described).title, "Dev server")
        XCTAssertEqual(BackgroundTaskEntry(shell: described).detail, "npm run dev")
        XCTAssertTrue(BackgroundTaskEntry(shell: described).isRunning)

        let bare = BackgroundShellSummary(
            id: "bash_2", commandHead: "yes", description: "",
            startedAt: Date(), chatID: nil, elapsedSeconds: 0)
        XCTAssertEqual(BackgroundTaskEntry(shell: bare).title, "yes")
        XCTAssertNil(BackgroundTaskEntry(shell: bare).detail)
    }

    // MARK: - Summary

    func testSummaryCountsRunningAndReportsAllFinishedWhenNoneRun() {
        XCTAssertNil(BackgroundTasksTray.summary([]))
        XCTAssertEqual(
            BackgroundTasksTray.summary([
                entry("a", startedAt: 1), entry("b", kind: .agent, startedAt: 2),
                entry("c", kind: .agent, running: false, startedAt: 3),
            ]),
            .running(2))
        XCTAssertEqual(
            BackgroundTasksTray.summary([entry("c", kind: .agent, running: false, startedAt: 3)]),
            .allFinished)
    }

    // MARK: - Actions

    func testTasksInTheSelectedChatOfferTheStripActions() {
        let shell = entry("bash_1", startedAt: 1, chatID: chatA)
        let agent = entry("bga_1", kind: .agent, startedAt: 1, chatID: chatA)
        let done = entry("bga_2", kind: .agent, running: false, startedAt: 1, chatID: chatA, canDismiss: true)
        let pending = entry("bga_3", kind: .agent, running: false, startedAt: 1, chatID: chatA)
        func actions(_ e: BackgroundTaskEntry) -> [BackgroundTaskAction] {
            BackgroundTasksTray.actions(for: e, selectedChatID: chatA, isChatVisible: true)
        }
        XCTAssertEqual(actions(shell), [.kill])
        XCTAssertEqual(actions(agent), [.stop])
        XCTAssertEqual(actions(done), [.dismiss])
        XCTAssertEqual(actions(pending), [])
    }

    func testUnscopedTasksBelongToEveryChat() {
        let shell = entry("bash_1", startedAt: 1, chatID: nil)
        XCTAssertEqual(
            BackgroundTasksTray.actions(for: shell, selectedChatID: chatB, isChatVisible: true),
            [.kill])
    }

    func testAnotherChatsTaskOffersOnlyOpen() {
        // `killBackgroundShell` refuses another chat's shell, so a Kill here
        // would be a dead button; Open takes the user to the chat whose
        // strip can stop it.
        let shell = entry("bash_1", startedAt: 1, chatID: chatA)
        let agent = entry("bga_1", kind: .agent, startedAt: 1, chatID: chatA)
        XCTAssertEqual(
            BackgroundTasksTray.actions(for: shell, selectedChatID: chatB, isChatVisible: true),
            [.open])
        XCTAssertEqual(
            BackgroundTasksTray.actions(for: agent, selectedChatID: chatB, isChatVisible: true),
            [.open])
    }

    func testOpenIsOfferedBesideStopWhenTheChatPaneIsNotShowing() {
        let shell = entry("bash_1", startedAt: 1, chatID: chatA)
        XCTAssertEqual(
            BackgroundTasksTray.actions(for: shell, selectedChatID: chatA, isChatVisible: false),
            [.open, .kill])
    }
}
