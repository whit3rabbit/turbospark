import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Ghost Mode: temporary chats that exist only in memory.
///
/// The guarantee under test is the ARCHIVE FILTER, not the encryption: every
/// save site funnels through the two archive-construction points, and both
/// exclude `isGhost` rows, so no code path can put a ghost on disk. The
/// vault (AES-GCM under a per-launch key) is defense in depth on top.
@MainActor
final class GhostChatTests: XCTestCase {
    override func setUp() {
        super.setUp()
        // Each case starts from "no chats", so nothing depends on whatever
        // another test in this process left in the shared (test-redirected,
        // per `AppStorageRoot`) storage directory.
        try? FileManager.default.removeItem(at: AppStorageRoot.file("chats_archive.json"))
    }

    private func archiveFileData() -> Data? {
        try? Data(contentsOf: AppStorageRoot.file("chats_archive.json"))
    }

    // MARK: - Persistence exclusion

    /// The core guarantee: a ghost chat carrying content is invisible to a
    /// reload, while sibling normal chats survive untouched -- and the file
    /// on disk never contains the ghost's text at all.
    func testGhostChatNeverReachesTheArchive() {
        let model = AppModel()

        var normal = AppChat(projectID: nil)
        normal.messages.append(AppChatMessage(role: .user, content: "persisted keep me"))
        model.chats.insert(normal, at: 0)

        let ghostID = model.enterGhostChat()
        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "ghost secret turn"))
            payload.draft = "ghost secret draft"
        }

        model.persistChats()

        let reloaded = AppModel()
        XCTAssertTrue(
            reloaded.chats.allSatisfy { !$0.isGhost },
            "a reloaded archive must contain no ghost chat")
        XCTAssertEqual(
            reloaded.chats.first(where: { $0.messages.contains(where: { $0.content == "persisted keep me" }) })?.id,
            normal.id,
            "normal chats in the same archive must survive untouched")
        XCTAssertNil(
            reloaded.chats.first(where: { $0.id == ghostID }),
            "the ghost row must not survive a reload")

        let archiveData = archiveFileData()
        XCTAssertNotNil(archiveData, "the archive must still have been written")
        let archiveText = String(data: archiveData!, encoding: .utf8) ?? ""
        XCTAssertFalse(
            archiveText.contains("ghost secret"),
            "the ghost conversation text must not appear anywhere on disk")
    }

    /// Ghost content lives in the vault; the row keeps nothing. The debug
    /// assert in `makeChatArchive` enforces the same invariant on every
    /// persist; this pins it as a test too.
    func testGhostRowStaysEmptyWhileTheVaultHoldsTheContent() {
        let model = AppModel()
        let ghostID = model.enterGhostChat()

        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "sealed turn"))
            payload.todos = []
            payload.draft = "sealed draft"
        }

        let row = model.chats.first(where: { $0.id == ghostID })
        XCTAssertNotNil(row)
        XCTAssertTrue(row?.messages.isEmpty ?? false)
        XCTAssertTrue(row?.draft.isEmpty ?? false)
        XCTAssertNil(row?.contextSummary)
        XCTAssertNil(row?.skillState)

        let payload = model.ghostPayload(for: ghostID)
        XCTAssertEqual(payload.messages.last?.content, "sealed turn")
        XCTAssertEqual(payload.draft, "sealed draft")

        // And the transcript reads through the vault, not the row.
        XCTAssertEqual(model.turnMessages(for: ghostID).last?.content, "sealed turn")
    }

    /// Ending the ghost chat removes the row, wipes the sealed payload, and
    /// reselects a normal chat.
    func testEndGhostChatRemovesRowAndPayload() {
        let model = AppModel()

        var normal = AppChat(projectID: nil)
        normal.messages.append(AppChatMessage(role: .user, content: "normal turn"))
        model.chats.insert(normal, at: 0)

        let ghostID = model.enterGhostChat()
        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "doomed turn"))
        }
        XCTAssertTrue(model.isInGhostChat)

        model.endGhostChat()

        XCTAssertNil(model.ghostChat, "the ghost row must be gone")
        XCTAssertFalse(model.ghostVault.hasPayload(for: ghostID), "the sealed payload must be wiped")
        XCTAssertFalse(model.isInGhostChat)
        XCTAssertEqual(model.selectedChatID, normal.id, "selection must fall back to a real chat")
    }

    /// Session-scoped by design: leaving the ghost chat for a normal one
    /// keeps it alive in memory until quit or explicit end.
    func testSwitchingAwayKeepsTheGhostChatInMemory() {
        let model = AppModel()

        var normal = AppChat(projectID: nil)
        normal.messages.append(AppChatMessage(role: .user, content: "normal turn"))
        model.chats.insert(normal, at: 0)

        let ghostID = model.enterGhostChat()
        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "kept in memory"))
        }

        model.selectChat(id: normal.id)

        XCTAssertNotNil(model.ghostChat, "the ghost chat must stay for the session")
        XCTAssertTrue(model.ghostChatHasContent)
        XCTAssertEqual(
            model.ghostPayload(for: ghostID).messages.last?.content, "kept in memory")
        XCTAssertEqual(model.ghostChat?.id, ghostID)
    }

    // MARK: - Lifecycle edges

    /// A plain New Chat must not "reuse" the ghost row: its row fields are
    /// empty BY DESIGN, so the emptiness check alone would keep the user in
    /// Ghost Mode after they asked for a persisted chat.
    func testCreateChatDoesNotReuseTheGhostRow() {
        let model = AppModel()
        let ghostID = model.enterGhostChat()
        XCTAssertTrue(model.isInGhostChat)

        let newID = model.createChat()

        XCTAssertNotEqual(newID, ghostID)
        XCTAssertFalse(
            model.chats.first(where: { $0.id == newID })?.isGhost ?? true,
            "the new chat must be a normal persisted chat")
        XCTAssertEqual(model.selectedChatID, newID)
    }

    /// With the setting on, a fresh model opens already inside a temporary
    /// chat -- and only temporary rows exist in memory, none of them
    /// persisted.
    func testAlwaysStartInGhostModeOpensInATemporaryChat() {
        try? FileManager.default.removeItem(at: AppStorageRoot.file("settings.json"))
        let first = AppModel()
        first.alwaysStartInGhostMode = true
        first.persistSettings()

        let second = AppModel()
        XCTAssertTrue(second.alwaysStartInGhostMode)
        XCTAssertTrue(second.isInGhostChat, "the app must open in the temporary chat")
        XCTAssertNotNil(second.ghostChat)
        XCTAssertTrue(
            second.chats.allSatisfy { $0.isGhost },
            "nothing else may have created rows before the ghost took the selection")

        // Cleanup: restore the default so later tests in this process do not
        // inherit the flag from settings.json.
        second.alwaysStartInGhostMode = false
        second.persistSettings()
    }

    // MARK: - Sidebar visibility

    /// The ghost chat shows in the sidebar for the session (with a ghost
    /// badge), including while its transcript exists only in the vault.
    func testSidebarTranscriptCheckReadsTheVault() {
        let model = AppModel()
        let ghostID = model.enterGhostChat()
        let row = model.chats.first(where: { $0.id == ghostID })
        XCTAssertNotNil(row)

        XCTAssertFalse(model.chatHasTranscript(row!), "an empty ghost is not history")

        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "visible turn"))
        }
        XCTAssertTrue(
            model.chatHasTranscript(model.chats.first(where: { $0.id == ghostID })!),
            "a ghost with vaulted messages must list in the sidebar")
    }

    // MARK: - Project and navigation isolation

    /// A ghost chat stamped with project A must not resurface just because
    /// the user switches back to project A: `selectProject`'s replacement
    /// pick used to be an unfiltered list that could land on it.
    func testSwitchingProjectsDoesNotAutoSelectTheGhostChat() {
        let model = AppModel()
        let projectA = model.createProject(name: "A")
        let projectB = model.createProject(name: "B")

        model.selectProject(id: projectA.id)
        let realChatA = model.selectedChatID
        let ghostID = model.enterGhostChat()
        XCTAssertEqual(model.ghostChat?.projectID, projectA.id)

        model.selectProject(id: projectB.id)
        model.selectProject(id: projectA.id)

        XCTAssertNotEqual(
            model.selectedChatID, ghostID,
            "switching back to the ghost's project must not reselect the ghost chat")
        XCTAssertEqual(
            model.selectedChatID, realChatA,
            "switching back to project A must reselect its real chat")
    }

    /// The same list backs Cmd+[ / Cmd+], so a ghost chat must not be a stop
    /// on the ordinary chat-cycling path either.
    func testOrderedChatsExcludesTheGhostChat() {
        let model = AppModel()
        var normal = AppChat(projectID: nil)
        normal.messages.append(AppChatMessage(role: .user, content: "normal"))
        model.chats.insert(normal, at: 0)

        let ghostID = model.enterGhostChat()

        XCTAssertFalse(model.orderedChats.contains { $0.id == ghostID })
        XCTAssertFalse(model.filteredChats.contains { $0.id == ghostID })
    }

    // MARK: - Deletion parity

    /// Deleting a ghost row through the generic path (the sidebar's own
    /// Delete action) must clean up the vault exactly as `endGhostChat()`
    /// does, not just remove the row.
    func testDeletingGhostRowWipesTheVault() {
        let model = AppModel()
        let ghostID = model.enterGhostChat()
        model.mutateGhostPayload(for: ghostID) { payload in
            payload.messages.append(AppChatMessage(role: .user, content: "sealed"))
        }
        XCTAssertTrue(model.ghostVault.hasPayload(for: ghostID))

        model.deleteChat(id: ghostID)

        XCTAssertFalse(
            model.ghostVault.hasPayload(for: ghostID),
            "deleting a ghost row through any path must wipe its sealed payload")
        XCTAssertNil(model.chats.first(where: { $0.id == ghostID }))
    }
}
