import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// ZP-02: the sidebar row's one status slot. The pure priority function,
/// the derived unread state and its read marker, the tolerant decode of a
/// chat saved before the marker existed, and the pending pill's source.
@MainActor
final class ChatBadgeStateTests: XCTestCase {
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

    private func reply(at date: Date, content: String = "done") -> AppChatMessage {
        AppChatMessage(role: .assistant, content: content, createdAt: date)
    }

    private func chat(_ title: String, read: Date, replies: [Date] = []) -> AppChat {
        AppChat(
            title: title, messages: replies.map { reply(at: $0) }, lastReadAt: read)
    }

    // MARK: - Slot priority

    func testPendingOutranksEverythingOnAnUnselectedRow() {
        for state in [ChatTurnState.awaitingTool, .awaitingQuestion] {
            XCTAssertEqual(
                ChatRowSlot.resolve(turnState: state, isSelected: false, hasUnreadReply: true),
                .pending)
        }
    }

    func testPendingPillIsNotShownOnTheSelectedRow() {
        XCTAssertEqual(
            ChatRowSlot.resolve(
                turnState: .awaitingTool, isSelected: true, hasUnreadReply: false),
            .normal)
    }

    func testRunningOutranksUnreadSoABackgroundStreamShowsTheSpinnerFirst() {
        XCTAssertEqual(
            ChatRowSlot.resolve(turnState: .generating, isSelected: false, hasUnreadReply: true),
            .running)
        XCTAssertEqual(
            ChatRowSlot.resolve(turnState: .idle, isSelected: false, hasUnreadReply: true),
            .unread)
    }

    func testUnreadNeverShowsOnTheSelectedRow() {
        XCTAssertEqual(
            ChatRowSlot.resolve(turnState: .idle, isSelected: true, hasUnreadReply: true),
            .normal)
    }

    func testQueuedAloneIsNotASlotState() {
        XCTAssertEqual(
            ChatRowSlot.resolve(turnState: .queued, isSelected: false, hasUnreadReply: false),
            .normal)
    }

    // MARK: - Unread derivation

    func testReplyNewerThanTheMarkerIsUnreadAndOlderIsNot() {
        XCTAssertTrue(chat("a", read: t0, replies: [t0.addingTimeInterval(5)]).hasUnreadReply)
        XCTAssertFalse(chat("a", read: t0, replies: [t0.addingTimeInterval(-5)]).hasUnreadReply)
        XCTAssertFalse(chat("a", read: t0, replies: [t0]).hasUnreadReply, "Equal is read.")
    }

    func testToolOnlySyntheticAndUserRowsDoNotCountAsAReply() {
        var c = chat("a", read: t0)
        let later = t0.addingTimeInterval(10)
        c.messages = [
            AppChatMessage(role: .assistant, content: "  \n", createdAt: later),
            AppChatMessage(role: .user, content: "hi", createdAt: later),
            AppChatMessage(
                role: .assistant, content: "nudge", createdAt: later, isSynthetic: true),
        ]
        XCTAssertFalse(c.hasUnreadReply)
    }

    // MARK: - Marker lifecycle (CUJ 1)

    func testSelectingAnUnreadChatClearsTheDotAndANewReplyRaisesItAgain() {
        let a = chat("A", read: t0)
        var b = chat("B", read: t0, replies: [t0.addingTimeInterval(5)])
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id

        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .unread)

        appModel.selectChat(id: b.id)
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .normal)
        XCTAssertGreaterThan(appModel.chats[1].lastReadAt, t0.addingTimeInterval(5))

        // Leave B, then a reply completes in B while it is not selected.
        appModel.selectChat(id: a.id)
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .normal)
        b = appModel.chats[1]
        appModel.chats[1].messages.append(reply(at: b.lastReadAt.addingTimeInterval(60)))
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .unread)
    }

    func testAReplyThatLandedWhileSelectedIsReadWhenTheUserLeaves() {
        let a = chat("A", read: t0)
        let b = chat("B", read: t0)
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        // The reply arrives while A is on screen: no dot on the selected row.
        appModel.chats[0].messages.append(reply(at: Date()))
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[0]), .normal)

        appModel.selectedChatID = b.id
        XCTAssertEqual(
            appModel.chatRowSlot(for: appModel.chats[0]), .normal,
            "Leaving a chat records what the user saw there.")
    }

    func testMarkingAnAlreadyReadChatDoesNotWrite() {
        let a = chat("A", read: t0, replies: [t0.addingTimeInterval(-1)])
        appModel.chats = [a]
        XCTAssertFalse(appModel.markChatRead(id: a.id))
        XCTAssertEqual(appModel.chats[0].lastReadAt, t0)
    }

    // MARK: - Pending pill source (CUJ 2)

    func testApprovalAndQuestionRaiseThePillOnTheirChatOnly() {
        let a = chat("A", read: t0)
        let b = chat("B", read: t0)
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id

        appModel.pendingToolCall = AppToolCall(
            name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
        appModel.pendingToolCallChatID = b.id
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .pending)
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[0]), .normal)

        appModel.pendingToolCall = nil
        appModel.pendingToolCallChatID = nil
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .normal)

        appModel.pendingUserQuestions = AppModel.PendingUserQuestions(
            id: UUID(), chatID: b.id, toolCallID: nil, items: [])
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .pending)
        appModel.pendingUserQuestions = AppModel.PendingUserQuestions(
            id: UUID(), chatID: nil, toolCallID: nil, items: [])
        XCTAssertEqual(
            appModel.chatRowSlot(for: appModel.chats[1]), .normal,
            "A question with no chat attributed shows no pill.")
    }

    func testABackgroundGeneratingChatShowsTheSpinnerSlotThenTheDot() {
        let a = chat("A", read: t0)
        var b = chat("B", read: t0, replies: [t0.addingTimeInterval(5)])
        b.lastReadAt = t0
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        appModel.generating = true
        appModel.claimLane(for: b.id)
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .running)

        appModel.generating = false
        appModel.generatingChatID = nil
        appModel.turnCoordinator.releaseHolder()
        XCTAssertEqual(appModel.chatRowSlot(for: appModel.chats[1]), .unread)
    }

    // MARK: - Persistence

    func testMarkerRoundTripsThroughJSON() throws {
        let c = chat("A", read: t0)
        let data = try JSONEncoder().encode(c)
        let back = try JSONDecoder().decode(AppChat.self, from: data)
        XCTAssertEqual(back.lastReadAt.timeIntervalSince1970, t0.timeIntervalSince1970, accuracy: 0.001)
    }

    func testAChatSavedBeforeTheMarkerExistedDecodesAsRead() throws {
        // A reply from last week, no marker key at all.
        let json = """
        {"id":"\(UUID().uuidString)","title":"Old","messages":[
          {"role":"assistant","content":"hello","createdAt":\(Date().addingTimeInterval(-86_400 * 7).timeIntervalSinceReferenceDate)}]}
        """
        let c = try JSONDecoder().decode(AppChat.self, from: Data(json.utf8))
        XCTAssertFalse(c.hasUnreadReply, "Nothing lights up after an upgrade.")
    }

    func testAWrongTypedMarkerCostsTheMarkerNotTheChat() throws {
        let json = """
        {"id":"\(UUID().uuidString)","title":"Keep","lastReadAt":"yesterday-ish","messages":[]}
        """
        let c = try JSONDecoder().decode(AppChat.self, from: Data(json.utf8))
        XCTAssertEqual(c.title, "Keep")
    }

    func testTheSelectedChatsMarkerIsStampedIntoTheArchive() {
        let a = chat("A", read: t0, replies: [t0.addingTimeInterval(5)])
        let b = chat("B", read: t0, replies: [t0.addingTimeInterval(5)])
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        // Selecting stamped A already; rewind it to model a reply that
        // landed while A was on screen and the app then quit.
        appModel.chats[0].lastReadAt = t0
        let archive = appModel.makeChatArchive()
        XCTAssertGreaterThan(archive.chats[0].lastReadAt, t0)
        XCTAssertEqual(archive.chats[1].lastReadAt, t0, "An unselected chat stays unread on disk.")
    }
}
