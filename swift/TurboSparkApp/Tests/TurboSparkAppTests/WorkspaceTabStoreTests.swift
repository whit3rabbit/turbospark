import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// ZP-21: workspace tabs. The pure list rules, the file store's quarantine,
/// and the AppModel wiring (the tab is a project reference over the existing
/// selection paths, so drafts stay per chat and the ZP-23 project-switch
/// guard decides whether a switch happens).
@MainActor
final class WorkspaceTabStoreTests: XCTestCase {
    private let a = UUID()
    private let b = UUID()
    private let c = UUID()

    private func ids(_ n: Int) -> [UUID] { (0..<n).map { _ in UUID() } }

    // MARK: - List rules

    func testOpenSeedsTheCurrentProjectAndNeverDuplicates() {
        var store = WorkspaceTabStore()
        XCTAssertFalse(store.isActive)
        XCTAssertEqual(store.open(b, current: a), .opened)
        XCTAssertEqual(store.tabs.map(\.projectID), [a, b])
        XCTAssertEqual(store.selectedProjectID, b)
        XCTAssertTrue(store.isActive)

        // Opening an open project only moves the selection.
        XCTAssertEqual(store.open(a, current: b), .opened)
        XCTAssertEqual(store.tabs.map(\.projectID), [a, b])
        XCTAssertEqual(store.selectedProjectID, a)
    }

    func testOpenWithNothingToSeedKeepsNoOneTabList() {
        var store = WorkspaceTabStore()
        XCTAssertEqual(store.open(a, current: nil), .opened)
        XCTAssertTrue(store.tabs.isEmpty, "A single tab is the dormant state.")
        XCTAssertFalse(store.isActive)
    }

    func testTheCapRefusesWithoutChangingTheList() {
        var store = WorkspaceTabStore()
        let all = ids(WorkspaceTabStore.maxTabs + 1)
        store.open(all[1], current: all[0])
        for id in all[2..<WorkspaceTabStore.maxTabs] { store.open(id, current: nil) }
        XCTAssertEqual(store.tabs.count, WorkspaceTabStore.maxTabs)
        let before = store

        XCTAssertEqual(store.open(all[WorkspaceTabStore.maxTabs], current: nil), .atCapacity)
        XCTAssertEqual(store, before)
    }

    func testNavigationRetargetsTheSelectedTabInPlaceAndSelectsAnOpenOne() {
        var store = WorkspaceTabStore()
        store.open(b, current: a)  // [a, b], b selected

        store.navigate(to: c)  // c has no tab: b's tab becomes c
        XCTAssertEqual(store.tabs.map(\.projectID), [a, c])
        XCTAssertEqual(store.selectedProjectID, c)

        store.navigate(to: a)  // a has a tab: selection only
        XCTAssertEqual(store.tabs.map(\.projectID), [a, c])
        XCTAssertEqual(store.selectedProjectID, a)

        store.navigate(to: nil)  // All chats: no tab selected, list kept
        XCTAssertEqual(store.tabs.count, 2)
        XCTAssertNil(store.selectedProjectID)
    }

    func testNavigationWhileDormantDoesNothing() {
        var store = WorkspaceTabStore()
        store.navigate(to: a)
        XCTAssertEqual(store, WorkspaceTabStore())
    }

    func testClosingPicksTheRightNeighborThenTheLeftAndCollapsesBelowTwo() {
        var store = WorkspaceTabStore()
        store.open(b, current: a)
        store.open(c, current: nil)  // [a, b, c], c selected
        XCTAssertEqual(store.neighborAfterClosing(c), b, "Last tab falls back to the left.")
        store.navigate(to: a)
        XCTAssertEqual(store.neighborAfterClosing(a), b)
        XCTAssertNil(store.neighborAfterClosing(b), "Only the selected tab names a neighbor.")

        store.remove(c)
        XCTAssertEqual(store.tabs.map(\.projectID), [a, b])
        store.remove(b)
        XCTAssertTrue(store.tabs.isEmpty, "One tab left is the dormant state.")
        XCTAssertNil(store.selectedProjectID)
    }

    func testPruneDropsDeletedProjects() {
        var store = WorkspaceTabStore()
        store.open(b, current: a)
        store.open(c, current: nil)
        store.prune(validProjectIDs: [a, c])
        XCTAssertEqual(store.tabs.map(\.projectID), [a, c])
        XCTAssertEqual(store.selectedProjectID, c)

        store.prune(validProjectIDs: [a])
        XCTAssertTrue(store.tabs.isEmpty)
    }

    func testCyclingWrapsBothWaysAndNeedsTwoTabs() {
        var store = WorkspaceTabStore()
        XCTAssertNil(store.projectID(cycling: 1))
        store.open(b, current: a)
        store.open(c, current: nil)  // c selected, last
        XCTAssertEqual(store.projectID(cycling: 1), a)
        XCTAssertEqual(store.projectID(cycling: -1), b)
        store.navigate(to: a)
        XCTAssertEqual(store.projectID(cycling: -1), c)
    }

    func testNoteChatOnlyTouchesAnExistingTab() {
        var store = WorkspaceTabStore()
        store.open(b, current: a)
        let chat = UUID()
        store.noteChat(chat, in: a)
        XCTAssertEqual(store.tab(for: a)?.lastChatID, chat)
        let before = store
        store.noteChat(chat, in: c)
        XCTAssertEqual(store, before)
    }

    func testRestoreRepairsDuplicatesSelectionAndLength() {
        let dup = WorkspaceTabArchive(
            tabs: [WorkspaceTab(projectID: a), WorkspaceTab(projectID: a),
                WorkspaceTab(projectID: b)],
            selectedProjectID: c)
        let store = WorkspaceTabStore(archive: dup)
        XCTAssertEqual(store.tabs.map(\.projectID), [a, b])
        XCTAssertNil(store.selectedProjectID, "A selection naming no tab is dropped.")

        let single = WorkspaceTabArchive(tabs: [WorkspaceTab(projectID: a)], selectedProjectID: a)
        XCTAssertTrue(WorkspaceTabStore(archive: single).tabs.isEmpty)

        let many = WorkspaceTabArchive(
            tabs: ids(WorkspaceTabStore.maxTabs + 4).map { WorkspaceTab(projectID: $0) })
        XCTAssertEqual(WorkspaceTabStore(archive: many).tabs.count, WorkspaceTabStore.maxTabs)
    }

    func testBadgeTakesTheStrongestSlotInZP02Order() {
        XCTAssertEqual(WorkspaceTabBadge.resolve(slots: []), .none)
        XCTAssertEqual(WorkspaceTabBadge.resolve(slots: [.normal, .unread]), .unread)
        XCTAssertEqual(WorkspaceTabBadge.resolve(slots: [.unread, .running]), .running)
        XCTAssertEqual(
            WorkspaceTabBadge.resolve(slots: [.running, .unread, .pending, .normal]), .pending)
    }

    // MARK: - File store

    private func scratchURL() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("zp21-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        return directory.appendingPathComponent("workspace_tabs.json")
    }

    func testTheArchiveRoundTripsInOrderWithTheSelection() throws {
        let url = try scratchURL()
        var store = WorkspaceTabStore()
        store.open(b, current: a)
        store.noteChat(UUID(), in: a)
        XCTAssertTrue(WorkspaceTabFileStore.save(store.archive, to: url))

        let restored = WorkspaceTabStore(archive: WorkspaceTabFileStore.load(from: url))
        XCTAssertEqual(restored, store)
    }

    func testAnAbsentFileIsADormantStartNotAnError() throws {
        let url = try scratchURL()
        let failures = AppJSONStore.readFailureCount
        XCTAssertEqual(WorkspaceTabFileStore.load(from: url), WorkspaceTabArchive())
        XCTAssertEqual(AppJSONStore.readFailureCount, failures)
    }

    func testACorruptFileIsQuarantinedAndTheNextSaveDoesNotOverwriteIt() throws {
        let url = try scratchURL()
        try Data("{ not json".utf8).write(to: url)
        let failures = AppJSONStore.readFailureCount

        let archive = WorkspaceTabFileStore.load(from: url)
        XCTAssertEqual(archive, WorkspaceTabArchive(), "Corruption restores no tabs.")
        XCTAssertEqual(AppJSONStore.readFailureCount, failures + 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: url.path))
        let names = try FileManager.default.contentsOfDirectory(
            atPath: url.deletingLastPathComponent().path)
        XCTAssertTrue(names.contains { $0.contains("corrupt") }, "Kept aside: \(names)")

        XCTAssertTrue(WorkspaceTabFileStore.save(WorkspaceTabArchive(), to: url))
        let after = try FileManager.default.contentsOfDirectory(
            atPath: url.deletingLastPathComponent().path)
        XCTAssertEqual(after.filter { $0.contains("corrupt") }.count, 1)
        AppJSONStore.clearLastReadError()
    }

    // MARK: - AppModel wiring

    private var model: AppModel!
    private var projectA: AppProject!
    private var projectB: AppProject!

    private func tabsFileURL() -> URL { AppStorageRoot.file(WorkspaceTabFileStore.fileName) }

    override func setUp() async throws {
        try await super.setUp()
        try? FileManager.default.removeItem(at: tabsFileURL())
        model = AppModel()
        projectA = AppProject(name: "Alpha")
        projectB = AppProject(name: "Beta")
        model.projects = [projectA, projectB]
        model.workspaceTabs = WorkspaceTabStore()
    }

    override func tearDown() async throws {
        model = nil
        try? FileManager.default.removeItem(at: tabsFileURL())
        try await super.tearDown()
    }

    /// Two chats in Alpha (the first is the one last used) and one in Beta,
    /// Alpha selected on its LAST-USED chat.
    private func seedChats() -> (a1: AppChat, a2: AppChat, b1: AppChat) {
        let a1 = AppChat(projectID: projectA.id, title: "A one", draft: "draft in A one")
        let a2 = AppChat(projectID: projectA.id, title: "A two", draft: "draft in A two")
        let b1 = AppChat(projectID: projectB.id, title: "B one", draft: "draft in B one")
        model.chats = [a1, a2, b1]
        model.selectedProjectID = projectA.id
        model.selectedChatID = a1.id
        return (a1, a2, b1)
    }

    func testSwitchingTabsKeepsEachChatsDraftAndRestoresTheLastChat() {
        let (a1, a2, b1) = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.selectedProjectID, projectB.id)
        XCTAssertEqual(model.selectedChatID, b1.id)

        // Type into B, go back to A: A comes back on the chat it was left on.
        model.chats[2].draft = "typed in B"
        model.activateWorkspaceTab(projectID: projectA.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
        XCTAssertEqual(model.selectedChatID, a1.id)
        XCTAssertEqual(model.chats.first { $0.id == a1.id }?.draft, "draft in A one")

        // Move to A's second chat, leave and return: the tab remembers a2.
        model.selectChat(id: a2.id)
        model.activateWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.chats.first { $0.id == b1.id }?.draft, "typed in B")
        model.activateWorkspaceTab(projectID: projectA.id)
        XCTAssertEqual(model.selectedChatID, a2.id, "The tab restores the last chat, not the newest.")
        XCTAssertEqual(model.chats.first { $0.id == a2.id }?.draft, "draft in A two")
    }

    func testAStaleRememberedChatFallsBackToTheProjectsOwnSelection() {
        let (a1, _, _) = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)
        model.activateWorkspaceTab(projectID: projectA.id)
        model.openWorkspaceTab(projectID: projectB.id)
        // A's remembered chat is deleted while its tab is in the background.
        model.chats.removeAll { $0.id == a1.id }
        model.activateWorkspaceTab(projectID: projectA.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
        XCTAssertNotEqual(model.selectedChatID, a1.id)
        XCTAssertEqual(model.chats.first { $0.id == model.selectedChatID }?.projectID, projectA.id)
    }

    func testATabSwitchIsRefusedWhileATurnRunsAndAllowedAfter() {
        let (a1, _, _) = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)
        model.activateWorkspaceTab(projectID: projectA.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id)

        model.generating = true
        model.claimLane(for: a1.id)
        model.activateWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id, "ZP-23 keeps the project guard.")
        XCTAssertEqual(model.workspaceTabs.selectedProjectID, projectA.id)
        XCTAssertNotNil(model.activeToast)

        model.generating = false
        model.activateWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.selectedProjectID, projectB.id)
    }

    func testAnApprovalCardPinsTheUserToItsTab() {
        let (a1, _, _) = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)
        model.activateWorkspaceTab(projectID: projectA.id)
        model.pendingToolCall = AppToolCall(
            name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
        model.pendingToolCallChatID = a1.id
        model.activateWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
    }

    func testOpeningATabWhileATurnRunsLeavesNoOrphanTab() {
        _ = seedChats()
        model.generating = true
        model.openWorkspaceTab(projectID: projectB.id)
        XCTAssertTrue(model.workspaceTabs.tabs.isEmpty)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
    }

    func testClosingTheSelectedTabMovesToItsNeighborAndARefusedMoveKeepsIt() {
        let (a1, _, _) = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)  // [A, B], B selected

        model.generating = true
        model.claimLane(for: model.selectedChatID)
        model.closeWorkspaceTab(projectID: projectB.id)
        XCTAssertTrue(model.workspaceTabs.contains(projectB.id), "Refused move keeps the tab.")
        XCTAssertEqual(model.selectedProjectID, projectB.id)

        model.generating = false
        model.closeWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
        XCTAssertEqual(model.selectedChatID, a1.id)
        XCTAssertTrue(model.workspaceTabs.tabs.isEmpty, "Down to one tab the strip is dormant.")
    }

    func testClosingABackgroundTabNeverMovesTheApp() {
        _ = seedChats()
        let third = AppProject(name: "Gamma")
        model.projects.append(third)
        model.openWorkspaceTab(projectID: projectB.id)
        model.openWorkspaceTab(projectID: third.id)
        model.generating = true  // even mid-turn: no switch is involved
        model.closeWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.workspaceTabs.tabs.map(\.projectID), [projectA.id, third.id])
        XCTAssertEqual(model.selectedProjectID, third.id)
    }

    func testASidebarProjectClickRetargetsTheSelectedTab() {
        _ = seedChats()
        let third = AppProject(name: "Gamma")
        model.projects.append(third)
        model.chats.append(AppChat(projectID: third.id, title: "G one"))
        model.openWorkspaceTab(projectID: projectB.id)  // [A, B]
        model.selectProject(id: third.id)  // the sidebar, not the strip
        XCTAssertEqual(model.workspaceTabs.tabs.map(\.projectID), [projectA.id, third.id])
        XCTAssertEqual(model.workspaceTabs.selectedProjectID, third.id)
    }

    func testDeletingAProjectPrunesItsTab() {
        _ = seedChats()
        let third = AppProject(name: "Gamma")
        model.projects.append(third)
        model.openWorkspaceTab(projectID: projectB.id)
        model.openWorkspaceTab(projectID: third.id)
        model.activateWorkspaceTab(projectID: projectA.id)
        model.deleteProject(id: projectB.id)
        XCTAssertEqual(model.workspaceTabs.tabs.map(\.projectID), [projectA.id, third.id])
    }

    func testTheCapToastNamesTheLimit() {
        _ = seedChats()
        var extra: [AppProject] = []
        for index in 0..<WorkspaceTabStore.maxTabs {
            let project = AppProject(name: "P\(index)")
            extra.append(project)
        }
        model.projects.append(contentsOf: extra)
        for project in [projectB!] + extra.prefix(WorkspaceTabStore.maxTabs - 2) {
            model.chats.append(AppChat(projectID: project.id))
            model.openWorkspaceTab(projectID: project.id)
        }
        XCTAssertEqual(model.workspaceTabs.tabs.count, WorkspaceTabStore.maxTabs)
        let before = model.workspaceTabs
        let selected = model.selectedProjectID
        model.activeToast = nil
        model.openWorkspaceTab(projectID: extra.last!.id)
        XCTAssertEqual(model.workspaceTabs, before)
        XCTAssertEqual(model.selectedProjectID, selected)
        XCTAssertNotNil(model.activeToast)
    }

    func testTabBadgeAggregatesEveryChatOfTheProject() {
        _ = seedChats()
        model.openWorkspaceTab(projectID: projectB.id)
        XCTAssertEqual(model.workspaceTabBadge(for: projectA.id), .none)

        // An unread reply in a NON-last chat of the background project still
        // lights its tab: the badge is not just the tab's active chat.
        let reply = AppChatMessage(
            role: .assistant, content: "done", createdAt: Date().addingTimeInterval(60))
        model.chats[1].messages.append(reply)  // A two, not selected
        XCTAssertEqual(model.workspaceTabBadge(for: projectA.id), .unread)

        model.pendingToolCall = AppToolCall(
            name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
        model.pendingToolCallChatID = model.chats[0].id
        XCTAssertEqual(model.workspaceTabBadge(for: projectA.id), .pending)
    }

    func testRestoreAlignsWithTheRestoredProjectAndPrunesMissingOnes() throws {
        _ = seedChats()
        let ghostProject = UUID()
        var store = WorkspaceTabStore()
        store.open(projectB.id, current: projectA.id)
        store.open(ghostProject, current: nil)
        let url = try scratchURL()
        WorkspaceTabFileStore.save(store.archive, to: url)

        model.selectedProjectID = projectA.id
        model.loadWorkspaceTabs(from: url)
        XCTAssertEqual(model.workspaceTabs.tabs.map(\.projectID), [projectA.id, projectB.id])
        XCTAssertEqual(
            model.workspaceTabs.selectedProjectID, projectA.id,
            "The project archive decides what is on screen.")
    }

    func testCyclingUsesTheSameGuardedPath() {
        _ = seedChats()
        model.interactionMode = .projects
        model.openWorkspaceTab(projectID: projectB.id)
        model.cycleWorkspaceTab(delta: 1)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
        model.generating = true
        model.cycleWorkspaceTab(delta: 1)
        XCTAssertEqual(model.selectedProjectID, projectA.id)
    }
}
