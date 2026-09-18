import XCTest

@testable import TurboSparkApp

/// The qwen-code parity state layer: local-command recognition and its
/// dispatcher arms, archive/duplicate/rewind chat operations, the
/// two-stage Esc cancel, and the HTML export. The pure modules behind the
/// commands have their own suite (`QwenParityModulesTests`); this file pins
/// the STATE they move.
@MainActor
final class SlashParityTests: XCTestCase {
    var appModel: AppModel!

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    private func makeChat(title: String = "T", messages: [AppChatMessage] = []) -> AppChat {
        let chat = AppChat(title: title, messages: messages)
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return chat
    }

    // MARK: - Command recognition

    func testEveryLocalCommandSpellingIsRecognized() {
        for name in BuiltInSlashCommand.recognizedLocalNames {
            XCTAssertTrue(BuiltInSlashCommand.isLocalCommand("/\(name)"), name)
            XCTAssertTrue(BuiltInSlashCommand.isLocalCommand("/\(name) argument"), name)
        }
        XCTAssertFalse(BuiltInSlashCommand.isLocalCommand("/copycat"))
        XCTAssertFalse(BuiltInSlashCommand.isLocalCommand("/newish now"))
        XCTAssertFalse(BuiltInSlashCommand.isLocalCommand("copy"))
        // None of the local commands leak into the older gates.
        XCTAssertFalse(BuiltInSlashCommand.isMetaCommand("/copy"))
        XCTAssertFalse(BuiltInSlashCommand.isMemoryCommand("/theme dark"))
        XCTAssertFalse(BuiltInSlashCommand.isLocalMetaCommand("/tasks"))
        XCTAssertFalse(BuiltInSlashCommand.isGoalCommand("/goals"))
    }

    func testRegistryNamesStayUniqueWithTheNewRows() {
        let names = BuiltInSlashCommand.recognizedNames
        XCTAssertEqual(names.count, Set(names).count)
    }

    func testLocalCommandsAreNotAgentCommands() {
        for name in BuiltInSlashCommand.recognizedLocalNames {
            XCTAssertNil(BuiltInSlashCommand.matching(name), name)
        }
    }

    // MARK: - Sheet-opening commands

    func testSheetCommandsSetTheirFlags() {
        _ = makeChat()
        appModel.handleLocalCommand("/context")
        XCTAssertTrue(appModel.showContextSheet)
        appModel.handleLocalCommand("/tasks")
        XCTAssertTrue(appModel.showTasksSheet)
        appModel.handleLocalCommand("/tools")
        XCTAssertTrue(appModel.showToolsSheet)
        appModel.handleLocalCommand("/status")
        XCTAssertTrue(appModel.showStatusSheet)
        appModel.handleLocalCommand("/rewind")
        XCTAssertTrue(appModel.showRewindSheet)
    }

    // MARK: - /rename

    func testRenameCommandRenamesTheSelectedChat() {
        _ = makeChat()
        appModel.handleLocalCommand("/rename Fresh Topic")
        XCTAssertEqual(appModel.selectedChat.title, "Fresh Topic")
    }

    func testRenameWithoutATitleIsANoOp() {
        _ = makeChat(title: "Original")
        appModel.handleLocalCommand("/rename")
        XCTAssertEqual(appModel.selectedChat.title, "Original")
    }

    // MARK: - Archive

    func testArchiveHidesFromSidebarAndUnarchiveRestores() {
        let chat = makeChat()
        appModel.handleLocalCommand("/archive")
        XCTAssertTrue(appModel.chats.first { $0.id == chat.id }?.isArchived == true)
        XCTAssertTrue(appModel.filteredChats.isEmpty, "an archived chat leaves the sidebar")
        XCTAssertEqual(appModel.archivedChats.map(\.id), [chat.id])

        appModel.handleLocalCommand("/unarchive")
        XCTAssertFalse(appModel.chats.first { $0.id == chat.id }?.isArchived == true)
        XCTAssertEqual(appModel.filteredChats.map(\.id), [chat.id])
        XCTAssertTrue(appModel.archivedChats.isEmpty)
    }

    func testGhostChatsRefuseArchive() {
        let ghost = AppChat(title: "Temporary Chat", isGhost: true)
        appModel.chats = [ghost]
        appModel.selectedChatID = ghost.id
        appModel.handleLocalCommand("/archive")
        XCTAssertFalse(appModel.chats.first { $0.id == ghost.id }?.isArchived == true)
        XCTAssertTrue(appModel.archivedChats.isEmpty)
    }

    func testIsArchivedRoundTripsThroughTheArchive() throws {
        var chat = AppChat(title: "kept")
        chat.isArchived = true
        let container = AppChatArchive(selectedChatID: chat.id, chats: [chat])
        let data = try JSONEncoder().encode(container)
        let decoded = try JSONDecoder().decode(AppChatArchive.self, from: data)
        XCTAssertTrue(decoded.chats[0].isArchived)
        // And a chat written BEFORE the field existed decodes as not
        // archived -- the tolerant-decode rule the row lives by.
        let oldChatJSON = """
        {"id":"\(UUID().uuidString)","title":"old","draft":"","messages":[],"todos":[],"artifacts":[],"compactedMessageCount":0,"createdAt":0,"updatedAt":0,"isGhost":false,"isPinned":false}
        """
        let oldArchive = """
        {"selectedChatID":"\(UUID().uuidString)","chats":[\(oldChatJSON)]}
        """
        let decodedOld = try JSONDecoder().decode(AppChatArchive.self, from: Data(oldArchive.utf8))
        XCTAssertFalse(decodedOld.chats[0].isArchived)
    }

    // MARK: - Duplicate

    func testDuplicateCopiesUnderFreshIdentities() {
        let user = AppChatMessage(role: .user, content: "q")
        var reply = AppChatMessage(role: .assistant, content: "a")
        reply.alternates = [AppChatMessage(role: .assistant, content: "old a")]
        let chat = makeChat(title: "Source", messages: [user, reply])

        let copyID = appModel.duplicateChat(id: chat.id)
        XCTAssertEqual(copyID, appModel.selectedChatID)
        let copy = appModel.selectedChat
        XCTAssertEqual(copy.title, "Source (copy)")
        XCTAssertNotEqual(copy.id, chat.id)
        XCTAssertEqual(copy.messages.count, 2)
        XCTAssertNotEqual(copy.messages[0].id, user.id)
        XCTAssertNotEqual(copy.messages[1].id, reply.id)
        XCTAssertEqual(copy.messages[1].alternates.count, 1)
        XCTAssertNotEqual(copy.messages[1].alternates[0].id, reply.alternates[0].id)
        // The original row is untouched and both rows exist.
        XCTAssertEqual(appModel.chats.count, 2)
        XCTAssertEqual(appModel.chats.last?.id, chat.id)
    }

    func testDuplicateRefusesGhostChats() {
        let ghost = AppChat(title: "Temporary Chat", isGhost: true)
        appModel.chats = [ghost]
        appModel.selectedChatID = ghost.id
        XCTAssertNil(appModel.duplicateChat(id: ghost.id))
        XCTAssertEqual(appModel.chats.count, 1)
    }

    // MARK: - /rewind

    func testRewindTargetsArePromptsWithSomethingAfterThem() {
        let u1 = AppChatMessage(role: .user, content: "first prompt")
        let a1 = AppChatMessage(role: .assistant, content: "first answer")
        let u2 = AppChatMessage(role: .user, content: "second prompt")
        _ = makeChat(messages: [u1, a1, u2])

        let targets = appModel.rewindTargets
        // The LAST prompt offers nothing to rewind to, so it is not listed.
        XCTAssertEqual(targets.map(\.id), [u1.id])
        XCTAssertEqual(targets[0].preview, "first prompt")
    }

    func testRewindDropsEverythingAfterTheAnchorPrompt() {
        let u1 = AppChatMessage(role: .user, content: "first prompt")
        let a1 = AppChatMessage(role: .assistant, content: "first answer")
        let u2 = AppChatMessage(role: .user, content: "second prompt")
        let a2 = AppChatMessage(role: .assistant, content: "second answer")
        let chat = makeChat(messages: [u1, a1, u2, a2])
        appModel.chats[0].todos = [TodoItem(content: "left over")]
        _ = chat

        appModel.rewindTo(anchorMessageID: u1.id)
        let messages = appModel.chats[0].messages
        XCTAssertEqual(messages.map(\.id), [u1.id], "the anchor prompt itself survives")
        XCTAssertTrue(appModel.chats[0].todos.isEmpty)
    }

    func testRewindToTheLastPromptDropsItsReply() {
        // The rewind sheet never OFFERS the last prompt, but a rewind to it
        // is still meaningful (redo this reply), so the operation honors it.
        let u = AppChatMessage(role: .user, content: "only")
        let a = AppChatMessage(role: .assistant, content: "reply")
        _ = makeChat(messages: [u, a])
        appModel.rewindTo(anchorMessageID: u.id)
        XCTAssertEqual(appModel.chats[0].messages.map(\.id), [u.id])
    }

    // MARK: - Two-stage Esc cancel

    func testEscArmDisarmCycle() {
        // Without a running turn the intent is refused outright.
        XCTAssertFalse(appModel.handleEscCancelIntent())
        XCTAssertFalse(appModel.isEscCancelArmed)
        appModel.disarmEscCancel()
        XCTAssertFalse(appModel.isEscCancelArmed)
    }

    func testTurnEndClearsCollapsedStateThroughTheModel() {
        // collapseAllTurns only marks turns that would HIDE something.
        let u = AppChatMessage(role: .user, content: "q")
        let a = AppChatMessage(role: .assistant, content: "answer only, nothing hidden")
        _ = makeChat(messages: [u, a])
        appModel.collapseAllTurns()
        XCTAssertTrue(appModel.collapsedTurnAnchors.isEmpty)
        XCTAssertFalse(appModel.canCollapseTurns)

        var toolCall = AppChatMessage(role: .assistant, content: "")
        toolCall.toolCalls = [AppToolCall(name: "Bash", arguments: [:])]
        let final = AppChatMessage(role: .assistant, content: "final")
        appModel.chats[0].messages = [u, toolCall, final]
        XCTAssertTrue(appModel.canCollapseTurns)
        appModel.collapseAllTurns()
        XCTAssertEqual(appModel.collapsedTurnAnchors, [u.id])
    }

    // MARK: - HTML export

    func testHTMLExportCarriesEscapedTitleAndRenderedBlocks() {
        var chat = AppChat(title: "AT&T <test>")
        chat.messages = [
            AppChatMessage(role: .user, content: "say hi"),
            AppChatMessage(role: .assistant, content: "hi **there**\n\n```swift\nlet x = 1\n```\n"),
        ]
        let html = AppChatExport.html(for: chat, modelAlias: "m")
        XCTAssertTrue(html.hasPrefix("<!DOCTYPE html>"))
        XCTAssertTrue(html.contains("<title>AT&amp;T &lt;test&gt;</title>"))
        XCTAssertTrue(html.contains("<h2>User</h2>"))
        XCTAssertTrue(html.contains("<p>say hi</p>"))
        XCTAssertTrue(html.contains("<strong>there</strong>"))
        XCTAssertTrue(html.contains("<pre><code class=\"lang-swift\">let x = 1</code></pre>"))
    }

    func testHTMLExportFormatMetadata() {
        let format = AppChatExportFormat.html
        XCTAssertEqual(format.fileExtension, "html")
        XCTAssertEqual(format.contentType, .html)
    }
}
