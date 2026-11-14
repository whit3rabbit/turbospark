import XCTest
@testable import TurboSparkApp

/// OC-05. The stack rules run on the pure value; the model tests drive the
/// real `activeSection` / `selectedChatID` observers and `selectChat`.
final class NavigationHistoryTests: XCTestCase {
    private func entry(_ section: AppModel.AppNavigationSection, _ chat: UUID? = nil) -> NavigationEntry {
        NavigationEntry(section: section, chatID: chat)
    }

    private func always(_: NavigationEntry) -> Bool { true }

    func testBackAndForwardWalkTheStack() {
        var h = NavigationHistory()
        h.record(entry(.chat)); h.record(entry(.images)); h.record(entry(.audio))
        XCTAssertEqual(h.backIndex(isValid: always), 1)
        h.move(to: 1)
        XCTAssertEqual(h.current, entry(.images))
        XCTAssertEqual(h.forwardIndex(isValid: always), 2)
        h.move(to: 0)
        XCTAssertNil(h.backIndex(isValid: always), "nothing before the first entry")
    }

    func testRepeatOfCurrentEntryIsIgnored() {
        var h = NavigationHistory()
        h.record(entry(.images)); h.record(entry(.images))
        XCTAssertEqual(h.entries.count, 1)
    }

    func testNewEntryAfterBackDiscardsForwardStack() {
        var h = NavigationHistory()
        h.record(entry(.chat)); h.record(entry(.images)); h.record(entry(.audio))
        h.move(to: 0)
        h.record(entry(.files))
        XCTAssertEqual(h.entries.map(\.section), [.chat, .files])
        XCTAssertNil(h.forwardIndex(isValid: always))
    }

    func testStackIsBoundedAndTrimsOldestFirst() {
        var h = NavigationHistory()
        let ids = (0..<(NavigationHistory.capacity + 5)).map { _ in UUID() }
        for id in ids { h.record(entry(.chat, id)) }
        XCTAssertEqual(h.entries.count, NavigationHistory.capacity)
        XCTAssertEqual(h.entries.first?.chatID, ids[5])
        XCTAssertEqual(h.current?.chatID, ids.last)
        XCTAssertEqual(h.cursor, NavigationHistory.capacity - 1)
    }

    func testInvalidEntriesAreSkipped() {
        var h = NavigationHistory()
        let dead = UUID()
        h.record(entry(.images)); h.record(entry(.chat, dead)); h.record(entry(.audio))
        let skipDead: (NavigationEntry) -> Bool = { $0.chatID != dead }
        XCTAssertEqual(h.backIndex(isValid: skipDead), 0)
        h.move(to: 0)
        XCTAssertNil(h.forwardIndex(isValid: { $0.section != .audio && $0.chatID != dead }))
    }

    func testChatIDIsDroppedOutsideTheChatSection() {
        XCTAssertNil(entry(.images, UUID()).chatID)
        var h = NavigationHistory()
        h.record(entry(.images, UUID())); h.record(entry(.images, UUID()))
        XCTAssertEqual(h.entries.count, 1)
    }

    func testSeedOnlyAppliesToAnEmptyStack() {
        var h = NavigationHistory()
        h.seedIfEmpty(entry(.files))
        h.seedIfEmpty(entry(.audio))
        XCTAssertEqual(h.entries, [entry(.files)])
    }
}

@MainActor
final class NavigationHistoryModelTests: XCTestCase {
    private func model() -> AppModel {
        let model = AppModel()
        model.chats = []
        model.activeSection = .chat
        model.flushNavigationRecord()
        model.navigationHistory = NavigationHistory()
        model.activeSection = .chat
        model.flushNavigationRecord()
        return model
    }

    private func go(_ model: AppModel, _ section: AppModel.AppNavigationSection) {
        model.activeSection = section
        model.flushNavigationRecord()
    }

    func testSectionSwitchesWalkBackAndForward() {
        let m = model()
        go(m, .images); go(m, .audio)
        XCTAssertTrue(m.canGoBackInHistory)
        XCTAssertFalse(m.canGoForwardInHistory)
        XCTAssertTrue(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .images)
        XCTAssertTrue(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .chat)
        XCTAssertFalse(m.canGoBackInHistory)
        XCTAssertTrue(m.goForwardInHistory())
        XCTAssertEqual(m.activeSection, .images)
        XCTAssertTrue(m.canGoForwardInHistory)
    }

    func testReplayDoesNotWipeTheForwardStack() {
        let m = model()
        go(m, .images); go(m, .audio)
        m.goBackInHistory()
        // The scheduled flush after a replay must see the arrival as a repeat.
        m.flushNavigationRecord()
        XCTAssertTrue(m.canGoForwardInHistory)
        m.goForwardInHistory()
        XCTAssertEqual(m.activeSection, .audio)
    }

    func testSearchStyleChatJumpIsOneStopAndReversible() {
        let m = model()
        var a = AppChat(), b = AppChat()
        a.messages = []; b.messages = []
        m.chats = [a, b]
        m.selectedChatID = a.id
        go(m, .images)
        // selectChat writes activeSection then selectedChatID: one stop.
        m.selectChat(id: b.id)
        m.flushNavigationRecord()
        XCTAssertEqual(m.activeSection, .chat)
        XCTAssertEqual(m.selectedChatID, b.id)
        XCTAssertTrue(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .images, "one step back, not a stop on the old chat")
        XCTAssertTrue(m.goForwardInHistory())
        XCTAssertEqual(m.activeSection, .chat)
        XCTAssertEqual(m.selectedChatID, b.id)
    }

    func testDeletedChatIsSkippedOnBack() {
        let m = model()
        let a = AppChat(), b = AppChat()
        m.chats = [a, b]
        m.selectedChatID = a.id
        go(m, .images)
        m.selectChat(id: b.id); m.flushNavigationRecord()
        go(m, .files)
        // Deleting the open chat reselects another one, as the app does.
        m.chats = [a]
        m.selectedChatID = a.id
        m.flushNavigationRecord()
        // b is gone: back from Files lands on Images, not a ghost chat.
        XCTAssertTrue(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .images)
    }

    func testNewSwitchAfterBackDiscardsForward() {
        let m = model()
        go(m, .images); go(m, .audio)
        m.goBackInHistory()
        go(m, .files)
        XCTAssertFalse(m.canGoForwardInHistory)
    }

    func testRefusedChatSwitchLeavesTheCursorAndForwardStackAlone() {
        let m = model()
        let a = AppChat()
        m.chats = [a]
        m.selectedChatID = a.id
        m.flushNavigationRecord()
        go(m, .images)
        // A submitting turn makes selectChat refuse (laneVerdict) while
        // isRunning is still false, so the walk reaches the refusal.
        m.submitting = true
        XCTAssertFalse(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .images)
        XCTAssertFalse(m.canGoForwardInHistory, "cursor must not move without arriving")
        m.submitting = false
        XCTAssertTrue(m.goBackInHistory())
        XCTAssertEqual(m.activeSection, .chat)
    }
}
