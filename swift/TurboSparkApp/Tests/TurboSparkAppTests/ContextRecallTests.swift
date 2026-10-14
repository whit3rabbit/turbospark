import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// The context engine's rules that need no store: what a chat records, how a
/// transcript is reconciled with what was recorded, how much evidence a turn
/// may take, where a chat records, and when the policy and the tools appear
/// (`swift/docs/SWIFT_CONTEXT_ENGINE.md`). The store round trips are
/// `ContextEngineManagerTests`.
@MainActor
final class ContextRecallTests: XCTestCase {
    private typealias Row = ContextRecall.Row

    // MARK: - Rows

    func testRowsKeepWhatThePeopleWroteAndDropToolTrafficAndSyntheticRows() {
        var synthetic = AppChatMessage(role: .user, content: "Stop hook says continue")
        synthetic.isSynthetic = true
        let messages = [
            AppChatMessage(role: .system, content: "SYSTEM"),
            AppChatMessage(role: .user, content: "Where is the rate limiter?"),
            AppChatMessage(role: .assistant, content: ""),
            AppChatMessage(role: .tool, content: "<tool_response>...</tool_response>"),
            synthetic,
            AppChatMessage(role: .assistant, content: "In limiter.rs."),
            AppChatMessage(role: .user, content: "   "),
        ]
        XCTAssertEqual(
            ContextRecall.rows(from: messages, beforeLastQuestion: false),
            [Row(role: .user, text: "Where is the rate limiter?"),
             Row(role: .assistant, text: "In limiter.rs.")])
    }

    func testBeforeLastQuestionLeavesOutTheQuestionAndTheTurnInProgress() {
        let messages = [
            AppChatMessage(role: .user, content: "Q1"),
            AppChatMessage(role: .assistant, content: "A1"),
            AppChatMessage(role: .user, content: "Q2"),
            AppChatMessage(role: .assistant, content: "Looking at the file first."),
        ]
        XCTAssertEqual(
            ContextRecall.rows(from: messages, beforeLastQuestion: true),
            [Row(role: .user, text: "Q1"), Row(role: .assistant, text: "A1")])
        XCTAssertEqual(ContextRecall.rows(from: [], beforeLastQuestion: true), [])
    }

    func testLongRowsAreCapped() {
        let long = String(repeating: "x", count: ContextRecall.maxRowCharacters + 50)
        let rows = ContextRecall.rows(
            from: [AppChatMessage(role: .user, content: long)], beforeLastQuestion: false)
        XCTAssertEqual(rows.first?.text.count, ContextRecall.maxRowCharacters)
    }

    // MARK: - Reconcile plan

    func testPlanAppendsToARecordedPrefixAndRebuildsAnythingElse() {
        let a = Row(role: .user, text: "A")
        let b = Row(role: .assistant, text: "B")
        let c = Row(role: .user, text: "C")
        let edited = Row(role: .user, text: "A, edited")
        XCTAssertEqual(ContextRecall.plan(recorded: [a, b], desired: [a, b]), .upToDate)
        XCTAssertEqual(ContextRecall.plan(recorded: [], desired: []), .upToDate)
        XCTAssertEqual(ContextRecall.plan(recorded: [a], desired: [a, b, c]), .append([b, c]))
        XCTAssertEqual(ContextRecall.plan(recorded: [], desired: [a]), .append([a]))
        // An edited or retried row, a deleted row, a role change: rebuild.
        XCTAssertEqual(ContextRecall.plan(recorded: [a, b], desired: [edited, b]), .rebuild([edited, b]))
        XCTAssertEqual(ContextRecall.plan(recorded: [a, b, c], desired: [a, b]), .rebuild([a, b]))
        XCTAssertEqual(
            ContextRecall.plan(recorded: [a], desired: [Row(role: .assistant, text: "A")]),
            .rebuild([Row(role: .assistant, text: "A")]))
    }

    // MARK: - Kept rows

    func testKeptRowsCountsTheSuffixStillInThePrompt() {
        let rows = [
            Row(role: .user, text: "dropped question"),
            Row(role: .assistant, text: "dropped answer"),
            Row(role: .user, text: "kept question"),
            Row(role: .assistant, text: "kept answer"),
        ]
        let prompt = [
            ChatMessage(role: .system, content: "SYSTEM"),
            // Assembly appends to user rows; the text is still found.
            ChatMessage(role: .user, content: "kept question\n\n<system-reminder>x</system-reminder>"),
            ChatMessage(role: .assistant, content: "kept answer"),
            ChatMessage(role: .user, content: "the new question"),
        ]
        XCTAssertEqual(ContextRecall.keptRows(rows, in: prompt), 2)
        XCTAssertEqual(ContextRecall.keptRows([], in: prompt), 0)
    }

    func testKeptRowsStopsAtTheFirstMissingRow() {
        // An overcount would hide a dropped turn from recall, so a match
        // further back after a miss must not count.
        let rows = [
            Row(role: .user, text: "SYSTEM"),
            Row(role: .assistant, text: "gone"),
            Row(role: .user, text: "kept"),
        ]
        let prompt = [
            ChatMessage(role: .user, content: "SYSTEM"),
            ChatMessage(role: .user, content: "kept"),
        ]
        XCTAssertEqual(ContextRecall.keptRows(rows, in: prompt), 1)
        // The role must match too.
        XCTAssertEqual(
            ContextRecall.keptRows(
                [Row(role: .assistant, text: "kept")], in: prompt), 0)
    }

    // MARK: - Budget

    func testEvidenceBudgetTakesPartOfTheFreeRoomOnly() {
        // 4K window, 1K reply: 3,072 for the prompt, 1,000 used. Free after
        // the margin (128) is 1,944; three quarters is 1,458, under a third
        // of the window (1,365).
        XCTAssertEqual(
            AppModel.contextEvidenceBudget(promptBudget: 3072, measuredTokens: 1000, window: 4096),
            1365)
        XCTAssertEqual(
            AppModel.contextEvidenceBudget(promptBudget: 3072, measuredTokens: 2600, window: 4096),
            (3072 - 2600 - 128) * 3 / 4)
        // Large windows are capped.
        XCTAssertEqual(
            AppModel.contextEvidenceBudget(promptBudget: 100_000, measuredTokens: 1000, window: 131_072),
            AppModel.contextEvidenceMaximumTokens)
        // Too little room: nothing rather than a useless sliver.
        XCTAssertEqual(
            AppModel.contextEvidenceBudget(promptBudget: 3072, measuredTokens: 2900, window: 4096), 0)
        XCTAssertEqual(
            AppModel.contextEvidenceBudget(promptBudget: 3072, measuredTokens: 4000, window: 4096), 0)
    }

    // MARK: - Where a chat records

    private func model(with chat: AppChat, projects: [AppProject] = []) -> AppModel {
        let model = AppModel()
        model.projects = projects
        model.chats = [chat]
        model.contextEngineEnabled = true
        model.conversationRecallEnabled = true
        return model
    }

    func testAChatRecordsIntoTheChatsStoreUnderItsOwnScope() {
        let chat = AppChat(title: "plain")
        let model = model(with: chat)
        let location = model.contextRecallLocation(chatID: chat.id)
        XCTAssertEqual(location?.store, .chats)
        XCTAssertEqual(location?.memoryScope, "app:chat-\(chat.id.uuidString.lowercased())")
    }

    func testAProjectChatRecordsIntoItsProjectStoreOnlyWithTheEngineOn() {
        var project = AppProject(name: "P", rootDirectoryPath: "/tmp/context-recall-project")
        var chat = AppChat(title: "in project")
        chat.projectID = project.id
        let off = model(with: chat, projects: [project])
        XCTAssertEqual(off.contextRecallLocation(chatID: chat.id)?.store, .chats)

        project.contextEngineEnabled = true
        let on = model(with: chat, projects: [project])
        XCTAssertEqual(on.contextRecallLocation(chatID: chat.id)?.store, .project(project.id))
        XCTAssertEqual(
            on.contextRecallLocation(chatID: chat.id)?.memoryScope,
            "project:\(project.id.uuidString.lowercased())")
    }

    func testGhostChatsAndTheSettingsTurnRecordingOff() {
        var ghost = AppChat(title: "Temporary Chat")
        ghost.isGhost = true
        XCTAssertNil(model(with: ghost).contextRecallLocation(chatID: ghost.id))

        let chat = AppChat(title: "plain")
        let recallOff = model(with: chat)
        recallOff.conversationRecallEnabled = false
        XCTAssertNil(recallOff.contextRecallLocation(chatID: chat.id))
        let engineOff = model(with: chat)
        engineOff.contextEngineEnabled = false
        XCTAssertNil(engineOff.contextRecallLocation(chatID: chat.id))
    }

    // MARK: - Policy and tools

    func testThePolicyJoinsTheSystemPromptOnlyForAProjectWithTheEngineOn() {
        let model = AppModel()
        model.contextEngineEnabled = true
        var project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/context-policy-demo")
        XCTAssertFalse(model.buildSystemPromptSections(for: project).contains { $0.section == .contextEvidence })
        XCTAssertFalse(model.buildSystemPromptSections(for: nil).contains { $0.section == .contextEvidence })

        project.contextEngineEnabled = true
        let sections = model.buildSystemPromptSections(for: project)
        XCTAssertEqual(
            sections.first { $0.section == .contextEvidence }?.content,
            AppModel.contextEvidencePolicy)
        XCTAssertFalse(AppModel.contextEvidencePolicy.isEmpty)

        model.contextEngineEnabled = false
        XCTAssertFalse(model.buildSystemPromptSections(for: project).contains { $0.section == .contextEvidence })
    }

    func testContextToolsAreOfferedOnlyForTheActiveContextProject() {
        let saved = AppToolRegistry.contextEngineRoot
        defer { AppToolRegistry.contextEngineRoot = saved }
        let root = URL(fileURLWithPath: "/tmp/context-tools-root", isDirectory: true)
        let other = URL(fileURLWithPath: "/tmp/context-tools-other", isDirectory: true)

        AppToolRegistry.contextEngineRoot = nil
        XCTAssertFalse(AppToolRegistry.isImplemented("context_search", projectURL: root))

        AppToolRegistry.contextEngineRoot = root
        for name in ContextEngineToolDefinitions.names {
            XCTAssertTrue(AppToolRegistry.isImplemented(name, projectURL: root), name)
            XCTAssertFalse(AppToolRegistry.isImplemented(name, projectURL: other), name)
            XCTAssertFalse(AppToolRegistry.isImplemented(name, projectURL: nil), name)
            XCTAssertEqual(AppToolCatalog.category(for: name), .fileRead, name)
        }
        let offered = AppToolCatalog.tools(for: .coder, projectURL: root).map(\.function.name)
        XCTAssertTrue(ContextEngineToolDefinitions.names.isSubset(of: Set(offered)))
        let elsewhere = AppToolCatalog.tools(for: .coder, projectURL: other).map(\.function.name)
        XCTAssertTrue(ContextEngineToolDefinitions.names.isDisjoint(with: Set(elsewhere)))
    }

    // MARK: - Settings

    func testSettingsDecodeDefaultsAndRoundTrip() throws {
        let empty = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertTrue(empty.contextEngineEnabled)
        XCTAssertTrue(empty.conversationRecallEnabled)
        // A wrong type falls back rather than failing the whole file.
        let wrong = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(#"{"contextEngineEnabled": "yes", "conversationRecallEnabled": false}"#.utf8))
        XCTAssertTrue(wrong.contextEngineEnabled)
        XCTAssertFalse(wrong.conversationRecallEnabled)

        var settings = MacAppSettings()
        settings.contextEngineEnabled = false
        settings.conversationRecallEnabled = false
        let back = try JSONDecoder().decode(
            MacAppSettings.self, from: try JSONEncoder().encode(settings))
        XCTAssertFalse(back.contextEngineEnabled)
        XCTAssertFalse(back.conversationRecallEnabled)
    }

    func testProjectOptInDecodesOffAndRoundTrips() throws {
        let legacy = try JSONDecoder().decode(AppProject.self, from: Data(#"{"name": "Old"}"#.utf8))
        XCTAssertFalse(legacy.contextEngineEnabled)
        var project = AppProject(name: "New", rootDirectoryPath: "/tmp/x")
        project.contextEngineEnabled = true
        let back = try JSONDecoder().decode(AppProject.self, from: try JSONEncoder().encode(project))
        XCTAssertTrue(back.contextEngineEnabled)
    }
}
