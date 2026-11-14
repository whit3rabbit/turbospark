import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// OC-04: session tabs. The pure list rules (a tab is a chat id, one tab per
/// chat, a bounded closed stack, reopen at the former position), the file
/// store's quarantine, and the AppModel wiring over the existing chat
/// selection paths (the ZP-23 lane verdict decides whether a switch happens,
/// drafts stay on the chat row).
@MainActor
final class SessionTabStoreTests: XCTestCase {
    private func ids(_ n: Int) -> [UUID] { (0..<n).map { _ in UUID() } }

    // MARK: - List rules

    func testOpenSeedsTheCurrentChatAndNeverDuplicates() {
        let (a, b) = (UUID(), UUID())
        var store = SessionTabStore()
        XCTAssertFalse(store.isActive)
        XCTAssertEqual(store.open(b, current: a), .opened)
        XCTAssertEqual(store.tabs, [a, b])
        XCTAssertEqual(store.selectedChatID, b)

        XCTAssertEqual(store.open(a, current: b), .opened)
        XCTAssertEqual(store.tabs, [a, b], "One tab per chat.")
        XCTAssertEqual(store.selectedChatID, a)
    }

    func testOpenWithNothingToSeedKeepsNoOneTabList() {
        var store = SessionTabStore()
        store.open(UUID(), current: nil)
        XCTAssertTrue(store.tabs.isEmpty)
    }

    func testTheCapRefusesWithoutChangingTheList() {
        let all = ids(SessionTabStore.maxTabs + 1)
        var store = SessionTabStore()
        store.open(all[1], current: all[0])
        for id in all[2..<SessionTabStore.maxTabs] { store.open(id, current: nil) }
        XCTAssertEqual(store.tabs.count, SessionTabStore.maxTabs)
        let before = store
        XCTAssertEqual(store.open(all[SessionTabStore.maxTabs], current: nil), .atCapacity)
        XCTAssertEqual(store, before)
    }

    func testNavigationRetargetsTheSelectedTabInPlace() {
        let (a, b, c) = (UUID(), UUID(), UUID())
        var store = SessionTabStore()
        store.open(b, current: a)  // [a, b], b selected
        store.navigate(to: c)
        XCTAssertEqual(store.tabs, [a, c])
        XCTAssertEqual(store.selectedChatID, c)
        store.navigate(to: a)
        XCTAssertEqual(store.tabs, [a, c])
        XCTAssertEqual(store.selectedChatID, a)
        store.navigate(to: nil)  // a ghost chat: list kept, nothing selected
        XCTAssertEqual(store.tabs, [a, c])
        XCTAssertNil(store.selectedChatID)
        store.navigate(to: UUID())  // no selected tab: no tab is hijacked
        XCTAssertEqual(store.tabs, [a, c])
    }

    func testNavigationWhileDormantDoesNothing() {
        var store = SessionTabStore()
        store.navigate(to: UUID())
        XCTAssertEqual(store, SessionTabStore())
    }

    func testClosingPicksTheRightNeighborThenTheLeftAndCollapsesBelowTwo() {
        let (a, b, c) = (UUID(), UUID(), UUID())
        var store = SessionTabStore()
        store.open(b, current: a)
        store.open(c, current: nil)  // [a, b, c], c selected
        XCTAssertEqual(store.neighborAfterClosing(c), b)
        store.navigate(to: a)
        XCTAssertEqual(store.neighborAfterClosing(a), b)
        XCTAssertNil(store.neighborAfterClosing(b), "A background tab has no neighbor.")

        store.close(a, remember: true)
        XCTAssertEqual(store.tabs, [b, c])
        store.close(b, remember: true)
        XCTAssertTrue(store.tabs.isEmpty, "Below two tabs the strip is dormant.")
        XCTAssertEqual(store.closed.map(\.chatID), [a, b], "The closed stack survives dormancy.")
    }

    func testTheClosedStackIsBoundedAndDropsTheOldest() {
        var store = SessionTabStore()
        let keep = UUID()
        let first = UUID()
        store.open(first, current: keep)
        for _ in 0..<(SessionTabStore.closedLimit + 5) {
            let id = UUID()
            store.open(id, current: nil)
            store.close(id, remember: true)
        }
        XCTAssertEqual(store.closed.count, SessionTabStore.closedLimit)
        XCTAssertNotNil(store.closed.last)
    }

    func testReopenTakesTheMostRecentValidEntryAtItsFormerPosition() {
        let (a, b, c) = (UUID(), UUID(), UUID())
        var store = SessionTabStore()
        store.open(b, current: a)
        store.open(c, current: nil)  // [a, b, c]
        store.close(b, remember: true)  // [a, c], b was at 1
        store.close(c, remember: true)  // dormant, c was at 1
        let deleted = c
        let entry = store.takeClosed(isValid: { $0 != deleted })
        XCTAssertEqual(entry, ClosedSessionTab(chatID: b, index: 1), "A deleted chat is skipped.")
        XCTAssertTrue(store.closed.isEmpty, "The skipped entry is dropped, not kept.")

        var open = SessionTabStore()
        open.open(c, current: a)  // [a, c]
        open.open(b, current: nil, at: 1)
        XCTAssertEqual(open.tabs, [a, b, c])
    }

    func testReopenByIDAndClampedPosition() {
        let (a, b) = (UUID(), UUID())
        var store = SessionTabStore()
        store.open(b, current: a)
        store.close(b, remember: true)
        XCTAssertNil(store.takeClosed(UUID(), isValid: { _ in true }))
        let entry = store.takeClosed(b, isValid: { _ in true })
        XCTAssertEqual(entry?.chatID, b)
        var next = SessionTabStore()
        next.open(b, current: a, at: 99)
        XCTAssertEqual(next.tabs, [a, b], "An out-of-range position clamps to the end.")
    }

    func testPruneDropsTabsAndClosedEntriesAndCollapses() {
        let (a, b, c) = (UUID(), UUID(), UUID())
        var store = SessionTabStore()
        store.open(b, current: a)
        store.open(c, current: nil)
        store.close(c, remember: true)  // [a, b]
        store.prune(isValid: { $0 != c })
        XCTAssertTrue(store.closed.isEmpty)
        XCTAssertEqual(store.tabs, [a, b])
        store.prune(isValid: { $0 == a })
        XCTAssertTrue(store.tabs.isEmpty)
    }

    func testCyclingWrapsBothWaysAndNeedsTwoTabs() {
        let (a, b, c) = (UUID(), UUID(), UUID())
        var store = SessionTabStore()
        XCTAssertNil(store.chatID(cycling: 1))
        store.open(b, current: a)
        store.open(c, current: nil)  // c selected, last
        XCTAssertEqual(store.chatID(cycling: 1), a)
        XCTAssertEqual(store.chatID(cycling: -1), b)
    }

    func testRestoreRepairsDuplicatesAndLengthAndHoldsNoClosedStack() {
        let (a, b) = (UUID(), UUID())
        XCTAssertEqual(SessionTabStore(archive: SessionTabArchive(tabs: [a, a, b])).tabs, [a, b])
        XCTAssertTrue(SessionTabStore(archive: SessionTabArchive(tabs: [a])).tabs.isEmpty)
        XCTAssertEqual(
            SessionTabStore(archive: SessionTabArchive(tabs: ids(SessionTabStore.maxTabs + 4)))
                .tabs.count, SessionTabStore.maxTabs)
        XCTAssertTrue(SessionTabStore(archive: SessionTabArchive(tabs: [a, b])).closed.isEmpty)
    }

    // MARK: - File store

    private func scratchURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("oc04-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory.appendingPathComponent("session_tabs.json")
    }

    func testTheArchiveRoundTripsInOrder() throws {
        let url = try scratchURL()
        let (a, b, c) = (UUID(), UUID(), UUID())
        let archive = SessionTabArchive(tabs: [c, a, b])
        XCTAssertTrue(SessionTabFileStore.save(archive, to: url))
        XCTAssertEqual(SessionTabFileStore.load(from: url), archive)
    }

    func testAnAbsentFileIsADormantStartNotAnError() throws {
        let url = try scratchURL()
        let failures = AppJSONStore.readFailureCount
        XCTAssertEqual(SessionTabFileStore.load(from: url), SessionTabArchive())
        XCTAssertEqual(AppJSONStore.readFailureCount, failures)
    }

    func testACorruptFileIsQuarantinedAndTheNextSaveDoesNotOverwriteIt() throws {
        let url = try scratchURL()
        try Data("{ not json".utf8).write(to: url)
        let failures = AppJSONStore.readFailureCount
        XCTAssertEqual(SessionTabFileStore.load(from: url), SessionTabArchive())
        XCTAssertEqual(AppJSONStore.readFailureCount, failures + 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let names = try FileManager.default.contentsOfDirectory(
            atPath: url.deletingLastPathComponent().path)
        XCTAssertTrue(names.contains { $0.contains("corrupt") }, "Kept aside: \(names)")
        XCTAssertTrue(SessionTabFileStore.save(SessionTabArchive(), to: url))
        let after = try FileManager.default.contentsOfDirectory(
            atPath: url.deletingLastPathComponent().path)
        XCTAssertEqual(after.filter { $0.contains("corrupt") }.count, 1)
        AppJSONStore.clearLastReadError()
    }

    // MARK: - Chords

    func testTheTabChordsAreCatalogedAndCollideWithNothing() {
        let ids: [KeyboardCommandID] = [
            .newSessionTab, .closeSessionTab, .reopenSessionTab, .nextSessionTab,
            .previousSessionTab,
        ]
        let commands = KeyboardShortcutCatalog.commands
        for id in ids {
            let command = KeyboardShortcutCatalog.command(for: id)
            XCTAssertNotNil(command, "\(id.rawValue) is missing from the catalog")
            XCTAssertTrue(command?.rebindable ?? false)
            XCTAssertFalse(command?.defaultChord.isReserved ?? true)
            let chord = command?.defaultChord
            let holders = commands.filter { $0.defaultChord == chord || $0.alternate == chord }
            XCTAssertEqual(holders.map(\.id), [id], "\(id.rawValue) shares its chord")
        }
        // Cmd+W stays the window's: the reserved list is why Close Tab is not it.
        XCTAssertTrue(KeyChord(key: "w", modifiers: .command)?.isReserved ?? false)
    }

    // MARK: - AppModel wiring

    private var model: AppModel!

    private func tabsFileURL() -> URL { AppStorageRoot.file(SessionTabFileStore.fileName) }

    override func setUp() async throws {
        try await super.setUp()
        try? FileManager.default.removeItem(at: tabsFileURL())
        model = AppModel()
        model.sessionTabs = SessionTabStore()
    }

    override func tearDown() async throws {
        model = nil
        try? FileManager.default.removeItem(at: tabsFileURL())
        try await super.tearDown()
    }

    private func message(_ text: String) -> AppChatMessage {
        AppChatMessage(role: .user, content: text)
    }

    /// Three chats with messages (so none is a draft), the first selected.
    private func seed() -> (c1: AppChat, c2: AppChat, c3: AppChat) {
        var c1 = AppChat(title: "One")
        var c2 = AppChat(title: "Two")
        var c3 = AppChat(title: "Three")
        for index in 0..<3 {
            let text = ["one", "two", "three"][index]
            if index == 0 { c1.messages = [message(text)] }
            if index == 1 { c2.messages = [message(text)] }
            if index == 2 { c3.messages = [message(text)] }
        }
        model.chats = [c1, c2, c3]
        model.selectedChatID = c1.id
        return (c1, c2, c3)
    }

    private func openAllThree(_ c: (c1: AppChat, c2: AppChat, c3: AppChat)) {
        model.openChatInSessionTab(id: c.c2.id)
        model.openChatInSessionTab(id: c.c3.id)
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c2.id, c.c3.id])
    }

    func testANewTabSeedsTheCurrentChatAndAddsAFocusedDraftTab() {
        let c = seed()
        model.openNewSessionTab()
        XCTAssertEqual(model.sessionTabs.tabs.count, 2)
        XCTAssertEqual(model.sessionTabs.tabs.first, c.c1.id)
        let draftID = model.sessionTabs.tabs[1]
        XCTAssertEqual(model.selectedChatID, draftID)
        XCTAssertEqual(model.sessionTabs.selectedChatID, draftID)
        XCTAssertEqual(model.chats.first { $0.id == draftID }?.messages.count, 0)
        XCTAssertTrue(model.sessionTabsActive)
    }

    func testANewTabReusesAnOpenEmptyDraftInsteadOfPiling() {
        _ = seed()
        model.openNewSessionTab()
        let chatCount = model.chats.count
        let tabs = model.sessionTabs.tabs
        model.openNewSessionTab()
        XCTAssertEqual(model.chats.count, chatCount)
        XCTAssertEqual(model.sessionTabs.tabs, tabs)
    }

    func testSendingInADraftTabPromotesItInPlaceAndTheTextSurvivesSwitches() {
        let c = seed()
        model.openNewSessionTab()
        let draftID = model.selectedChatID
        model.promptText = "unsent words"
        model.activateSessionTab(chatID: c.c1.id)
        XCTAssertEqual(model.selectedChatID, c.c1.id)
        model.activateSessionTab(chatID: draftID)
        XCTAssertEqual(model.promptText, "unsent words", "The draft lives on the chat row.")

        // Sending adds messages to the SAME chat: no tab swap, same position.
        let tabsBefore = model.sessionTabs.tabs
        if let index = model.chats.firstIndex(where: { $0.id == draftID }) {
            model.chats[index].messages = [message("unsent words")]
            model.chats[index].draft = ""
        }
        XCTAssertEqual(model.sessionTabs.tabs, tabsBefore)
        XCTAssertEqual(model.sessionTabs.selectedChatID, draftID)
        XCTAssertEqual(model.selectedChatID, draftID)
    }

    func testClosingAnEmptyDraftDiscardsItsRowAndRemembersNothing() {
        let c = seed()
        model.openNewSessionTab()
        let draftID = model.selectedChatID
        model.closeSessionTab(chatID: draftID)
        XCTAssertFalse(model.chats.contains { $0.id == draftID }, "An empty draft leaves no row.")
        XCTAssertTrue(model.sessionTabs.closed.isEmpty)
        XCTAssertTrue(model.sessionTabs.tabs.isEmpty, "Down to one tab the strip is dormant.")
        XCTAssertEqual(model.selectedChatID, c.c1.id)
    }

    func testClosingADraftWithTextKeepsTheRowAndReopensWithTheTextIntact() {
        let c = seed()
        model.openNewSessionTab()
        let draftID = model.selectedChatID
        model.promptText = "keep me"
        model.closeSessionTab(chatID: draftID)
        XCTAssertTrue(model.chats.contains { $0.id == draftID })
        XCTAssertEqual(model.sessionTabs.closed.map(\.chatID), [draftID])
        XCTAssertEqual(model.selectedChatID, c.c1.id)

        model.reopenClosedSessionTab()
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, draftID])
        XCTAssertEqual(model.selectedChatID, draftID)
        XCTAssertEqual(model.promptText, "keep me")
        XCTAssertTrue(model.sessionTabs.closed.isEmpty)
    }

    func testAClosedTabReopensAtItsFormerPositionAndSelectsTheChat() {
        let c = seed()
        openAllThree(c)
        model.closeSessionTab(chatID: c.c2.id)  // background tab
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c3.id])
        model.reopenClosedSessionTab()
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c2.id, c.c3.id])
        XCTAssertEqual(model.selectedChatID, c.c2.id)
    }

    func testReopenSkipsAChatDeletedWhileClosed() {
        let c = seed()
        openAllThree(c)
        model.closeSessionTab(chatID: c.c2.id)
        model.closeSessionTab(chatID: c.c3.id)  // selected: moves to c1 first
        XCTAssertTrue(model.sessionTabs.tabs.isEmpty)
        model.chats.removeAll { $0.id == c.c3.id }
        model.reopenClosedSessionTab()
        XCTAssertEqual(model.selectedChatID, c.c2.id, "The deleted chat is skipped silently.")
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c2.id])
    }

    func testClosingTheSelectedTabMovesToItsNeighbor() {
        let c = seed()
        openAllThree(c)
        model.activateSessionTab(chatID: c.c2.id)
        model.closeSessionTab(chatID: c.c2.id)
        XCTAssertEqual(model.selectedChatID, c.c3.id, "Right neighbor first.")
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c3.id])
    }

    func testAnApprovalCardPinsTheUserToItsTabAndAClosingMoveIsRefused() {
        let c = seed()
        openAllThree(c)
        model.activateSessionTab(chatID: c.c1.id)
        model.pendingToolCall = AppToolCall(
            name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
        model.pendingToolCallChatID = c.c1.id

        model.activateSessionTab(chatID: c.c2.id)
        XCTAssertEqual(model.selectedChatID, c.c1.id)
        XCTAssertNotNil(model.activeToast)

        let before = model.sessionTabs.tabs
        model.closeSessionTab(chatID: c.c1.id)
        XCTAssertEqual(model.sessionTabs.tabs, before, "A refused move keeps the tab.")
        XCTAssertEqual(model.selectedChatID, c.c1.id)
    }

    func testATabSwitchToAnotherProjectIsRefusedWhileATurnRunsButSameProjectIsNot() {
        let project = AppProject(name: "Alpha")
        model.projects = [project]
        var other = AppChat(projectID: project.id, title: "Other")
        other.messages = [message("x")]
        let c = seed()
        model.chats.append(other)
        model.selectedChatID = c.c1.id
        model.openChatInSessionTab(id: c.c2.id)
        // Seed the cross-project tab directly: opening it is itself a switch.
        model.sessionTabs.open(other.id, current: nil)
        model.activateSessionTab(chatID: c.c1.id)

        model.generating = true
        model.claimLane(for: c.c1.id)
        model.activateSessionTab(chatID: other.id)
        XCTAssertEqual(model.selectedChatID, c.c1.id, "A workspace switch waits for the turn.")
        XCTAssertNotNil(model.activeToast)

        model.activateSessionTab(chatID: c.c2.id)
        XCTAssertEqual(model.selectedChatID, c.c2.id, "ZP-23 lets a background turn run.")

        model.generating = false
        model.activateSessionTab(chatID: other.id)
        XCTAssertEqual(model.selectedChatID, other.id)
        XCTAssertEqual(model.selectedProjectID, project.id)
    }

    func testASidebarClickRetargetsTheSelectedTabAndFeedsHistory() {
        let c = seed()
        model.activeSection = .chat
        model.flushNavigationRecord()  // c1 is the first real stop
        model.openChatInSessionTab(id: c.c2.id)  // [c1, c2], c2 selected
        model.selectChat(id: c.c3.id)  // the sidebar, not the strip
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c3.id])
        XCTAssertEqual(model.sessionTabs.selectedChatID, c.c3.id)

        model.flushNavigationRecord()  // the scheduled flush does not run in a sync test
        model.activateSessionTab(chatID: c.c1.id)
        model.flushNavigationRecord()
        XCTAssertEqual(model.navigationHistory.current?.chatID, c.c1.id)
        XCTAssertTrue(model.canGoBackInHistory, "Tab selection is ordinary history.")
    }

    func testDeletingAChatPrunesItsTabWithoutHijackingAnother() {
        let c = seed()
        openAllThree(c)
        model.activateSessionTab(chatID: c.c2.id)
        model.deleteChat(id: c.c2.id)
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c3.id])
        XCTAssertTrue(model.sessionTabs.closed.isEmpty)
    }

    func testGhostAndArchivedChatsAreNeverTabs() {
        let c = seed()
        model.openChatInSessionTab(id: c.c2.id)
        model.setChatArchived(id: c.c1.id, archived: true)
        XCTAssertTrue(model.sessionTabs.tabs.isEmpty, "Archiving drops the tab; one left is dormant.")

        var ghost = AppChat(title: "Ghost")
        ghost.isGhost = true
        model.chats.append(ghost)
        model.openChatInSessionTab(id: ghost.id)
        XCTAssertFalse(model.sessionTabs.contains(ghost.id))
    }

    func testRestoreDropsDeletedArchivedAndGhostChatsAndAlignsTheSelection() throws {
        let c = seed()
        model.chats[1].isArchived = true
        let url = try scratchURL()
        let gone = UUID()
        SessionTabFileStore.save(
            SessionTabArchive(tabs: [c.c1.id, c.c2.id, gone, c.c3.id]), to: url)
        model.selectedChatID = c.c3.id
        model.loadSessionTabs(from: url)
        XCTAssertEqual(model.sessionTabs.tabs, [c.c1.id, c.c3.id])
        XCTAssertEqual(model.sessionTabs.selectedChatID, c.c3.id)
        XCTAssertTrue(model.sessionTabs.closed.isEmpty, "Recently Closed never survives a relaunch.")
    }

    func testTheCapToastNamesTheLimit() {
        let c = seed()
        model.openChatInSessionTab(id: c.c2.id)
        for index in 0..<SessionTabStore.maxTabs {
            var chat = AppChat(title: "X\(index)")
            chat.messages = [message("m")]
            model.chats.append(chat)
            model.openChatInSessionTab(id: chat.id)
        }
        XCTAssertEqual(model.sessionTabs.tabs.count, SessionTabStore.maxTabs)
        let before = model.sessionTabs
        model.activeToast = nil
        model.openChatInSessionTab(id: c.c3.id)
        XCTAssertEqual(model.sessionTabs, before)
        XCTAssertNotNil(model.activeToast)
    }
}
