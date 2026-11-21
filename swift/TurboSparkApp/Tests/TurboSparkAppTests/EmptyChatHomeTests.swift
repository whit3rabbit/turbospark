import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

final class EmptyChatHomeTests: XCTestCase {
    private let selected = UUID()

    private func chat(
        _ title: String,
        minutesAgo: Double,
        projectID: UUID? = nil,
        lastStopReason: String? = nil,
        draft: String = "",
        goal: ChatGoalState? = nil,
        isGhost: Bool = false,
        isArchived: Bool = false,
        hasMessages: Bool = true
    ) -> AppChat {
        let messages: [AppChatMessage] = hasMessages
            ? [
                AppChatMessage(role: .user, content: "q"),
                AppChatMessage(role: .assistant, content: "a", stopReason: lastStopReason),
            ]
            : []
        return AppChat(
            projectID: projectID, title: title, draft: draft, messages: messages,
            updatedAt: Date(timeIntervalSince1970: 1_000_000 - minutesAgo * 60),
            isGhost: isGhost, goal: goal, isArchived: isArchived)
    }

    private func resume(_ chats: [AppChat], selectedIsGhost: Bool = false, busy: Set<UUID> = [])
        -> [EmptyChatResumeItem]
    {
        EmptyChatHome.resumeItems(
            chats: chats, selectedChatID: selected, selectedIsGhost: selectedIsGhost,
            isBusy: { busy.contains($0) })
    }

    // MARK: Reason

    func testFinishedReplyIsNotUnfinished() {
        XCTAssertNil(EmptyChatHome.resumeReason(for: chat("done", minutesAgo: 1, lastStopReason: "stop")))
        XCTAssertNil(EmptyChatHome.resumeReason(for: chat("plain", minutesAgo: 1)))
    }

    func testIncompleteStopReasonsAreUnfinished() {
        XCTAssertEqual(
            EmptyChatHome.resumeReason(for: chat("e", minutesAgo: 1, lastStopReason: "error")),
            .stopped(.engineError))
        XCTAssertEqual(
            EmptyChatHome.resumeReason(for: chat("c", minutesAgo: 1, lastStopReason: "context_overflow")),
            .stopped(.contextFull))
        XCTAssertEqual(
            EmptyChatHome.resumeReason(for: chat("l", minutesAgo: 1, lastStopReason: GenerationResult.StopReason.maxTokens.rawValue)),
            .stopped(.lengthLimit))
    }

    /// A user stop is a choice, not unfinished work (MessageStopNotice.Severity.neutral).
    func testUserStopIsNotUnfinished() {
        XCTAssertNil(EmptyChatHome.resumeReason(for: chat("u", minutesAgo: 1, lastStopReason: "cancelled")))
    }

    /// Only the LAST row counts: an error earlier in a chat that later
    /// finished is history, not unfinished work.
    func testOnlyTheLastMessageIsRead() {
        var c = chat("old error", minutesAgo: 1, lastStopReason: "error")
        c.messages.append(AppChatMessage(role: .user, content: "again"))
        c.messages.append(AppChatMessage(role: .assistant, content: "ok", stopReason: "stop"))
        XCTAssertNil(EmptyChatHome.resumeReason(for: c))
    }

    func testDraftAndGoalAreUnfinished() {
        XCTAssertEqual(EmptyChatHome.resumeReason(for: chat("d", minutesAgo: 1, draft: "half typed")), .draft)
        XCTAssertNil(EmptyChatHome.resumeReason(for: chat("ws", minutesAgo: 1, draft: "  \n ")))
        XCTAssertEqual(
            EmptyChatHome.resumeReason(for: chat("g", minutesAgo: 1, goal: ChatGoalState(condition: "tests pass"))),
            .goal)
    }

    func testStoppedReplyOutranksGoalAndDraft() {
        let c = chat(
            "all", minutesAgo: 1, lastStopReason: "error", draft: "x",
            goal: ChatGoalState(condition: "y"))
        XCTAssertEqual(EmptyChatHome.resumeReason(for: c), .stopped(.engineError))
        let g = chat("goal+draft", minutesAgo: 1, draft: "x", goal: ChatGoalState(condition: "y"))
        XCTAssertEqual(EmptyChatHome.resumeReason(for: g), .goal)
    }

    // MARK: Selection

    func testResumeIsCappedAtThreeMostRecent() {
        let chats = (0..<5).map { chat("d\($0)", minutesAgo: Double($0), draft: "x") }
        XCTAssertEqual(resume(chats).map(\.title), ["d0", "d1", "d2"])
    }

    func testGhostArchivedSelectedAndBusyChatsAreExcluded() {
        var current = chat("current", minutesAgo: 0, draft: "x")
        current.id = selected
        let busy = chat("busy", minutesAgo: 1, draft: "x")
        let chats = [
            current,
            chat("ghost", minutesAgo: 1, draft: "x", isGhost: true),
            chat("archived", minutesAgo: 1, draft: "x", isArchived: true),
            busy,
            chat("kept", minutesAgo: 5, draft: "x"),
        ]
        XCTAssertEqual(resume(chats, busy: [busy.id]).map(\.title), ["kept"])
    }

    /// Ghost Mode shows no real chats on its empty screen.
    func testGhostSelectionShowsNothing() {
        let chats = [chat("d", minutesAgo: 1, draft: "x")]
        XCTAssertTrue(resume(chats, selectedIsGhost: true).isEmpty)
        XCTAssertTrue(
            EmptyChatHome.projectRecents(
                chats: chats, projectID: UUID(), selectedChatID: selected, selectedIsGhost: true,
                excluding: []
            ).isEmpty)
    }

    // MARK: Project home

    func testProjectRecentsScopeToProjectAndSkipResumeRows() {
        let project = UUID()
        let shown = chat("in resume", minutesAgo: 0, projectID: project, draft: "x")
        let chats = [
            shown,
            chat("p1", minutesAgo: 1, projectID: project),
            chat("other project", minutesAgo: 1, projectID: UUID()),
            chat("no project", minutesAgo: 1),
            chat("empty", minutesAgo: 1, projectID: project, hasMessages: false),
            chat("ghost", minutesAgo: 1, projectID: project, isGhost: true),
            chat("p2", minutesAgo: 2, projectID: project),
            chat("p3", minutesAgo: 3, projectID: project),
            chat("p4", minutesAgo: 4, projectID: project),
        ]
        let recents = EmptyChatHome.projectRecents(
            chats: chats, projectID: project, selectedChatID: selected, selectedIsGhost: false,
            excluding: [shown.id])
        XCTAssertEqual(recents.map(\.title), ["p1", "p2", "p3"])
    }

    func testNoProjectMeansNoProjectHome() {
        let chats = [chat("a", minutesAgo: 1)]
        XCTAssertTrue(
            EmptyChatHome.projectRecents(
                chats: chats, projectID: nil, selectedChatID: selected, selectedIsGhost: false,
                excluding: []
            ).isEmpty)
    }
}
