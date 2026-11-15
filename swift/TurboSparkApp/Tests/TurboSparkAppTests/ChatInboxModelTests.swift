import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// OC-03: the inbox grouping. The pure section function and layout (one
/// section per chat, priority, search, order, retention), the model wiring
/// that feeds them, and the persisted mode.
@MainActor
final class ChatInboxModelTests: XCTestCase {
    private var appModel: AppModel!
    private let t0 = Date(timeIntervalSince1970: 1_700_000_000)

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    private func chat(_ title: String, unread: Bool = false) -> AppChat {
        AppChat(
            title: title,
            messages: unread
                ? [AppChatMessage(
                    role: .assistant, content: "done", createdAt: t0.addingTimeInterval(5))]
                : [],
            lastReadAt: t0)
    }

    private func section(
        _ state: ChatTurnState, unread: Bool = false, selected: Bool = false,
        retained: Bool = false
    ) -> ChatInboxSection? {
        ChatInboxSection.resolve(
            turnState: state, hasUnreadReply: unread, isSelected: selected, isRetained: retained)
    }

    // MARK: - Membership matrix

    func testTurnStatesMapToTheirSections() {
        XCTAssertEqual(section(.awaitingTool), .needsYou)
        XCTAssertEqual(section(.awaitingQuestion), .needsYou)
        XCTAssertEqual(section(.generating), .working)
        XCTAssertEqual(section(.queued), .working)
        XCTAssertEqual(section(.idle, unread: true), .finished)
        XCTAssertNil(section(.idle))
    }

    func testPriorityIsNeedsYouThenWorkingThenFinished() {
        // A chat with an unread reply and a live state sits in the live section.
        XCTAssertEqual(section(.awaitingTool, unread: true), .needsYou)
        XCTAssertEqual(section(.generating, unread: true), .working)
        XCTAssertEqual(section(.queued, unread: true), .working)
        XCTAssertEqual(ChatInboxSection.allCases, [.needsYou, .working, .finished])
    }

    func testSelectedChatStaysInNeedsYouAndWorkingButNeverFinished() {
        XCTAssertEqual(section(.awaitingQuestion, selected: true), .needsYou)
        XCTAssertEqual(section(.generating, selected: true), .working)
        XCTAssertNil(section(.idle, unread: true, selected: true))
    }

    func testRetentionKeepsTheOpenedFinishedRowOnlyWhileSelectedAndIdle() {
        XCTAssertEqual(section(.idle, selected: true, retained: true), .finished)
        XCTAssertNil(
            section(.idle, selected: false, retained: true),
            "A retained id that is no longer selected must not pin the row.")
        XCTAssertEqual(
            section(.generating, selected: true, retained: true), .working,
            "Retention never overrides a live state.")
    }

    // MARK: - Layout

    private func layout(
        _ chats: [AppChat], search: String = "", selected: UUID = UUID(),
        retained: UUID? = nil, states: [UUID: ChatTurnState] = [:]
    ) -> ChatInboxLayout {
        ChatInboxLayout.make(
            chats: chats, searchText: search, selectedID: selected, retainedID: retained,
            turnState: { states[$0] ?? .idle })
    }

    func testCUJ1EachAttentionChatAppearsOnceAndIdleChatsAreAbsent() {
        let asking = chat("asking"), running = chat("running")
        let done = chat("done", unread: true), idle = chat("idle")
        let result = layout(
            [idle, done, running, asking],
            states: [asking.id: .awaitingQuestion, running.id: .generating])
        XCTAssertEqual(result.groups.map(\.section), [.needsYou, .working, .finished])
        XCTAssertEqual(result.groups.map { $0.chats.map(\.title) }, [["asking"], ["running"], ["done"]])
        let all = result.groups.flatMap(\.chats).map(\.id)
        XCTAssertEqual(all.count, Set(all).count, "One section per chat.")
        XCTAssertFalse(all.contains(idle.id))
    }

    func testEmptySectionsAreOmittedAndNothingMeansEmpty() {
        XCTAssertTrue(layout([chat("a"), chat("b")]).isEmpty)
        let running = chat("r")
        let only = layout([running], states: [running.id: .generating])
        XCTAssertEqual(only.groups.map(\.section), [.working])
    }

    func testSectionKeepsTheInputOrder() {
        let a = chat("a", unread: true), b = chat("b", unread: true), c = chat("c", unread: true)
        XCTAssertEqual(layout([b, c, a]).groups[0].chats.map(\.title), ["b", "c", "a"])
    }

    func testSearchFiltersByTitleAndReportsItself() {
        let a = chat("alpha", unread: true), b = chat("beta", unread: true)
        let hit = layout([a, b], search: "  ALP ")
        XCTAssertEqual(hit.groups.flatMap(\.chats).map(\.title), ["alpha"])
        XCTAssertTrue(hit.isSearching)
        XCTAssertTrue(layout([a, b], search: "zzz").isEmpty)
        XCTAssertFalse(layout([a, b], search: "   ").isSearching)
    }

    // MARK: - Model wiring (CUJ 2, retention, scope)

    func testCUJ2ApprovalResolvingMovesTheRowOutOfNeedsYou() {
        let a = chat("A"), b = chat("B")
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        appModel.pendingToolCall = AppToolCall(
            name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
        appModel.pendingToolCallChatID = b.id

        func groups() -> [ChatInboxSection] {
            ChatInboxLayout.make(
                chats: appModel.filteredChats, searchText: "",
                selectedID: appModel.selectedChatID,
                retainedID: appModel.inboxRetainedChatID,
                turnState: { self.appModel.chatTurnState(for: $0) }
            ).groups.map(\.section)
        }
        XCTAssertEqual(groups(), [.needsYou])

        appModel.pendingToolCall = nil
        appModel.pendingToolCallChatID = nil
        XCTAssertEqual(groups(), [], "Resolved with nothing unread: the row leaves.")
    }

    func testOpeningAFinishedChatRetainsItUntilTheSelectionMovesOn() {
        let a = chat("A"), b = chat("B", unread: true)
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        XCTAssertNil(appModel.inboxRetainedChatID)

        appModel.selectChat(id: b.id)
        XCTAssertEqual(appModel.inboxRetainedChatID, b.id)
        XCTAssertFalse(appModel.chats[1].hasUnreadReply, "Opening still marks it read.")
        let open = ChatInboxLayout.make(
            chats: appModel.filteredChats, searchText: "", selectedID: b.id,
            retainedID: appModel.inboxRetainedChatID, turnState: { _ in .idle })
        XCTAssertEqual(open.groups.flatMap(\.chats).map(\.title), ["B"])

        appModel.selectChat(id: a.id)
        XCTAssertNil(appModel.inboxRetainedChatID)
    }

    func testOpeningAnAlreadyReadChatRetainsNothing() {
        let a = chat("A"), b = chat("B")
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        appModel.selectChat(id: b.id)
        XCTAssertNil(appModel.inboxRetainedChatID)
    }

    func testArchivedAndGhostChatsNeverReachTheInbox() {
        var archived = chat("old", unread: true)
        archived.isArchived = true
        var ghost = chat("ghost", unread: true)
        ghost.isGhost = true
        appModel.chats = [archived, ghost, chat("live", unread: true)]
        appModel.selectedChatID = UUID()
        let result = ChatInboxLayout.make(
            chats: appModel.filteredChats, searchText: "", selectedID: appModel.selectedChatID,
            retainedID: nil, turnState: { _ in .idle })
        XCTAssertEqual(result.groups.flatMap(\.chats).map(\.title), ["live"])
    }

    // MARK: - Persisted mode

    func testModeDefaultsToChatsAndUnknownValuesFallBack() throws {
        XCTAssertEqual(appModel.chatSidebarMode, .chats)
        XCTAssertEqual(ChatSidebarMode(storedValue: "bogus"), .chats)
        XCTAssertEqual(ChatSidebarMode(storedValue: "inbox"), .inbox)
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy.chatSidebarMode, "chats")
        let typo = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"chatSidebarMode": 7}"#.utf8))
        XCTAssertEqual(typo.chatSidebarMode, "chats")
    }

    func testModeRoundTripsThroughSettings() throws {
        let data = try JSONEncoder().encode(MacAppSettings(chatSidebarMode: "inbox"))
        let back = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(ChatSidebarMode(storedValue: back.chatSidebarMode), .inbox)
    }
}
