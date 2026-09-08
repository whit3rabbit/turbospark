import SwiftUI
import XCTest
@testable import TurboSparkApp

/// Tests for the qwen-code web-shell parity features: chat pinning, the
/// usage ledger, conversation export, session stats, input history, the
/// cron scheduler, and the code-block highlighter.
@MainActor
final class QwenParityFeaturesTests: XCTestCase {

    // MARK: - Pinning

    func testPinnedChatsSortFirstAndRecencyBreaksTies() {
        var older = AppChat(title: "older")
        older.updatedAt = Date(timeIntervalSinceNow: -100)
        var newer = AppChat(title: "newer")
        newer.updatedAt = Date(timeIntervalSinceNow: -50)
        var pinnedOld = AppChat(title: "pinned old", isPinned: true)
        pinnedOld.updatedAt = Date(timeIntervalSinceNow: -1000)

        let sorted = AppChat.sortedForSidebar([older, newer, pinnedOld])
        XCTAssertEqual(sorted.map(\.title), ["pinned old", "newer", "older"])
    }

    func testPinnedOrderSurvivesArchiveRoundTrip() throws {
        var chat = AppChat(title: "kept", isPinned: true)
        chat.usage = AppChatUsageLedger()
        chat.usage?.record(promptTokens: 10, outputTokens: 5)
        let encoder = JSONEncoder()
        let data = try encoder.encode(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        let decoded = try JSONDecoder().decode(AppChatArchive.self, from: data)
        XCTAssertEqual(decoded.chats.first?.isPinned, true)
        XCTAssertEqual(decoded.chats.first?.usage?.totalOutputTokens, 5)
    }

    // MARK: - Usage ledger

    func testUsageLedgerAccumulatesAcrossCallsAndDays() {
        var ledger = AppChatUsageLedger()
        let dayOne = Date(timeIntervalSince1970: 1_900_000_000)
        let dayTwo = Date(timeIntervalSince1970: 1_900_000_000 + 86_400)

        ledger.record(promptTokens: 100, outputTokens: 20, on: dayOne)
        ledger.record(promptTokens: 30, outputTokens: -5, on: dayTwo)

        XCTAssertEqual(ledger.totalPromptTokens, 130)
        XCTAssertEqual(ledger.totalOutputTokens, 20, "a negative count records as nothing")
        XCTAssertEqual(ledger.turns, 2)
        XCTAssertEqual(ledger.byDay.count, 2)
        XCTAssertEqual(ledger.byDay[AppChatUsageLedger.dayKey(dayOne)]?.promptTokens, 100)
    }

    func testUsageTotalsMergeChatsAndIgnoreRowsWithoutLedgers() {
        var used = AppChat(title: "used")
        used.usage = AppChatUsageLedger()
        used.usage?.record(promptTokens: 40, outputTokens: 10)
        var ghost = AppChat(title: "ghost", isGhost: true)
        ghost.usage = AppChatUsageLedger()
        ghost.usage?.record(promptTokens: 999, outputTokens: 999)
        let bare = AppChat(title: "bare")

        let totals = AppChatUsageTotals(chats: [used, ghost, bare, used])
        XCTAssertEqual(totals.promptTokens, 80, "only ledgers given are summed")
        XCTAssertEqual(totals.outputTokens, 20)
        XCTAssertEqual(totals.turns, 2)
    }

    // MARK: - Export

    func testMarkdownExportCarriesTitleTurnsAndTools() {
        var chat = AppChat(title: "Export & /special: name")
        chat.messages = [
            AppChatMessage(role: .user, content: "hello there"),
            AppChatMessage(
                role: .assistant,
                content: "Answer.",
                toolCalls: [AppToolCall(name: "read_file", arguments: ["path": "x.rs"])],
                toolResults: []),
        ]
        let markdown = AppChatExport.markdown(for: chat, modelAlias: "gemma4")

        XCTAssertTrue(markdown.contains("# Export & /special: name"))
        XCTAssertTrue(markdown.contains("## User"))
        XCTAssertTrue(markdown.contains("hello there"))
        XCTAssertTrue(markdown.contains("**Tool: read_file**"))
        XCTAssertTrue(markdown.contains("Answer."))
        XCTAssertTrue(markdown.contains("```json"))
    }

    func testJSONExportRoundTripsMessages() throws {
        var chat = AppChat(title: "lossless")
        chat.messages = [AppChatMessage(role: .user, content: "keep me")]
        let data = try AppChatExport.jsonData(for: chat)
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        let decoded = try decoder.decode(AppChat.self, from: data)
        XCTAssertEqual(decoded.messages.first?.content, "keep me")
    }

    // MARK: - Stats

    func testStatsCountsTurnsToolsAndTodos() {
        var chat = AppChat(title: "stats chat")
        let failingCall = AppToolCall(name: "bash")
        chat.messages = [
            AppChatMessage(role: .user, content: "q1"),
            AppChatMessage(role: .assistant, content: "a1"),
            AppChatMessage(
                role: .assistant, content: "", reasoning: "",
                toolCalls: [failingCall],
                toolResults: [AppToolResult(callID: failingCall.id, output: "boom", isError: true)]),
        ]
        chat.todos = [
            TodoItem(content: "done thing", status: "completed"),
            TodoItem(content: "open thing"),
        ]
        chat.compactedMessageCount = 3

        let stats = AppChatStats(chat: chat, modelAlias: "m1")
        XCTAssertEqual(stats.userTurns, 1)
        XCTAssertEqual(stats.assistantTurns, 2)
        XCTAssertEqual(stats.toolCalls, 1)
        XCTAssertEqual(stats.failedToolCalls, 1)
        XCTAssertEqual(stats.todoCompleted, 1)
        XCTAssertEqual(stats.todoTotal, 2)
        XCTAssertEqual(stats.compactedMessages, 3)
        XCTAssertEqual(stats.modelAlias, "m1")
    }

    // MARK: - Input history

    /// A store writing into a throwaway directory, so tests cannot read
    /// what the shared store persisted under the shared test root.
    private func isolatedStore() -> InputHistoryStore {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return InputHistoryStore(directory: dir)
    }

    func testInputHistoryRecordsDeduplicatesConsecutiveAndCaps() {
        let store = isolatedStore()
        let baseline = store.count
        store.record("first")
        store.record("second")
        store.record("second")
        store.record("   ")
        XCTAssertEqual(store.entry(backSteps: 1)?.text, "second")
        XCTAssertEqual(store.entry(backSteps: 2)?.text, "first")
        XCTAssertEqual(store.count, baseline + 2)

        for index in 0..<(InputHistoryStore.capacity + 5) {
            store.record("filler \(index)")
        }
        XCTAssertEqual(store.count, InputHistoryStore.capacity)
    }

    func testHistoryBrowserWalksBackForwardAndParksDraft() {
        let store = isolatedStore()
        store.record("newest")
        store.record("older")

        var browser = ComposerHistoryBrowser()
        XCTAssertTrue(browser.canStepBack(draftText: ""))

        XCTAssertEqual(browser.stepBack(store: store, currentDraft: "parked draft"), "older")
        XCTAssertTrue(browser.isActive)
        XCTAssertEqual(browser.stepBack(store: store, currentDraft: "parked draft"), "newest")
        XCTAssertNil(browser.stepBack(store: store, currentDraft: "parked draft"), "nothing older than the oldest")

        XCTAssertEqual(browser.stepForward(store: store), "older")
        XCTAssertEqual(browser.stepForward(store: store), "parked draft", "past the newest, the parked draft returns")
        XCTAssertFalse(browser.isActive)
        XCTAssertNil(browser.stepForward(store: store))
    }

    func testHistoryBrowserIgnoresUpArrowMidDraftAndExternalEditEndsWalk() {
        let store = isolatedStore()
        store.record("recorded")
        var browser = ComposerHistoryBrowser()
        // Mid-draft (non-empty, not browsing): Up must NOT hijack the caret.
        XCTAssertFalse(browser.canStepBack(draftText: "half-written"))

        _ = browser.stepBack(store: store, currentDraft: "")
        browser.noteExternalEdit()
        XCTAssertFalse(browser.isActive)
        XCTAssertNil(browser.stepForward(store: store), "a dead walk must not hand text back")
    }

    // MARK: - Cron scheduler

    func testCronScheduleParsesAndRejects() {
        XCTAssertNotNil(CronSchedule("*/5 * * * *"))
        XCTAssertNotNil(CronSchedule("0 9 * * 1-5"))
        XCTAssertNotNil(CronSchedule("30 14 1,15 * *"))
        XCTAssertNil(CronSchedule("*/5 * * *"))
        XCTAssertNil(CronSchedule("* * * * * *"))
        XCTAssertNil(CronSchedule("61 * * * *"))
        XCTAssertNil(CronSchedule("* 25 * * *"))
        XCTAssertNil(CronSchedule("a b c d e"))
        XCTAssertNil(CronSchedule("5-2 * * * *"))
        XCTAssertNil(CronSchedule("mon * * * *"), "names are refused, not half-understood")
    }

    func testCronScheduleMatchesAndDayOfWeekNormalization() {
        // 2026-09-07 is a Monday.
        let calendar = Calendar(identifier: .gregorian)
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 7
        components.hour = 9; components.minute = 5
        let mondayMorning = calendar.date(from: components)!

        XCTAssertTrue(CronSchedule("*/5 * * * *")?.matches(mondayMorning) == true)
        XCTAssertTrue(CronSchedule("5 9 * * 1")?.matches(mondayMorning) == true)
        XCTAssertTrue(CronSchedule("5 9 * * 1-5")?.matches(mondayMorning) == true)
        XCTAssertTrue(CronSchedule("5 9 * * *")?.matches(mondayMorning) == true)
        XCTAssertFalse(CronSchedule("5 9 * * 7")?.matches(mondayMorning) == true,
            "7 normalized to Sunday, so it does not match Monday")
    }

    func testCronSevenNormalizesToSundayNotWildcard() {
        let calendar = Calendar(identifier: .gregorian)
        var components = DateComponents()
        components.year = 2026; components.month = 9; components.day = 6
        components.hour = 10; components.minute = 0
        // 2026-09-06 is a Sunday.
        let sunday = calendar.date(from: components)!
        XCTAssertTrue(CronSchedule("0 10 * * 0")?.matches(sunday) == true)
        XCTAssertTrue(CronSchedule("0 10 * * 7")?.matches(sunday) == true, "7 normalizes to Sunday")
        XCTAssertFalse(CronSchedule("0 10 * * 1")?.matches(sunday) == true)
    }

    func testCronNextFireAdvancesAndDescribeNamesCommonShapes() throws {
        let schedule = try XCTUnwrap(CronSchedule("*/15 * * * *"))
        let after = Date(timeIntervalSince1970: 1_900_000_100)
        let next = try XCTUnwrap(schedule.nextFire(after: after))
        XCTAssertGreaterThan(next, after)
        XCTAssertEqual(schedule.describe(), "Every 15 minutes")

        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * *")).describe(), "Daily at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * 1")).describe(), "Weekly on Monday at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * 1-5")).describe(), "Weekdays at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("30 * * * *")).describe(), "Hourly at :30")
    }

    func testCronSchedulerCreateListToggleDeleteRoundTrip() throws {
        let chatID = UUID()
        let job = try CronScheduler.shared.create(
            cron: "*/5 * * * *", prompt: "run the check", recurring: true, chatID: chatID)
        XCTAssertTrue(CronScheduler.shared.list().contains { $0.id == job.id })
        XCTAssertNotNil(job.nextFireAt)

        XCTAssertThrowsError(try CronScheduler.shared.create(
            cron: "not a cron", prompt: "x", recurring: true, chatID: chatID))
        XCTAssertThrowsError(try CronScheduler.shared.create(
            cron: "*/5 * * * *", prompt: "   ", recurring: true, chatID: chatID))

        XCTAssertTrue(CronScheduler.shared.setEnabled(id: job.id, enabled: false))
        let paused = try XCTUnwrap(CronScheduler.shared.list().first { $0.id == job.id })
        XCTAssertNil(paused.nextFireAt)

        XCTAssertTrue(CronScheduler.shared.delete(id: job.id))
        XCTAssertFalse(CronScheduler.shared.delete(id: job.id))
    }

    func testCronExecuteCreateReportsHumanScheduleAndInvalidCronThrows() throws {
        let chatID = UUID()
        let output = try CronScheduler.executeCreate(
            arguments: ["cron": "0 9 * * *", "prompt": "morning digest"], chatID: chatID)
        XCTAssertTrue(output.contains("Daily at 09:00"))
        XCTAssertThrowsError(try CronScheduler.executeCreate(
            arguments: ["cron": "99 * * * *", "prompt": "x"], chatID: chatID))
        XCTAssertThrowsError(try CronScheduler.executeCreate(
            arguments: ["prompt": "no cron at all"], chatID: chatID))
    }

    func testCronOneShotFiresAndRemovesItself() async throws {
        let chatID = UUID()
        // The shared fire handler is nil in tests unless the app installed
        // it; install one that records and reports delivered.
        var delivered: [String] = []
        let previous = CronScheduler.shared.fireHandler
        CronScheduler.shared.fireHandler = { job in
            delivered.append(job.prompt)
            return true
        }
        defer { CronScheduler.shared.fireHandler = previous }

        _ = CronScheduler.shared.createOneShot(
            fireAt: Date().addingTimeInterval(-1), prompt: "wake word", chatID: chatID)
        await CronScheduler.shared.fireDueJobs()
        XCTAssertEqual(delivered, ["wake word"])
        XCTAssertFalse(
            CronScheduler.shared.list().contains { $0.prompt == "wake word" },
            "a fired one-shot removes itself")
    }

    // MARK: - Syntax highlighter

    func testHighlighterKnowsFamiliesAndRefusesUnknown() {
        XCTAssertNotNil(CodeSyntaxHighlighter.family(forLanguage: "swift"))
        XCTAssertNotNil(CodeSyntaxHighlighter.family(forLanguage: "Python"))
        XCTAssertNotNil(CodeSyntaxHighlighter.family(forLanguage: "sql"))
        XCTAssertNotNil(CodeSyntaxHighlighter.family(forLanguage: "html"))
        XCTAssertNil(CodeSyntaxHighlighter.family(forLanguage: "brainfuck"))
        XCTAssertNil(CodeSyntaxHighlighter.family(forLanguage: nil))
        XCTAssertNil(
            CodeSyntaxHighlighter.highlight(String(repeating: "x", count: 20_000), language: "swift"),
            "over the size cap the caller falls back to plain rendering")
    }

    func testHighlighterColorsKeywordsStringsAndComments() throws {
        let code = """
        // a comment
        let count = 5
        let label = "hello"
        """
        let highlighted = try XCTUnwrap(
            CodeSyntaxHighlighter.highlight(code, language: "swift"))
        let text = String(highlighted.characters)
        XCTAssertEqual(text, code, "round-trip: got \(text.debugDescription)")

        // Run-level assertions: find the runs by their characters.
        var runs: [(String, Color?)] = []
        for run in highlighted.runs {
            let range = highlighted[run.range]
            runs.append((String(range.characters), run.foregroundColor))
        }
        XCTAssertTrue(
            runs.contains { $0.0.contains("a comment") && $0.1 == Color.secondary },
            "the // line colors as a comment; runs: \(runs.map { "\($0.0)|\(String(describing: $0.1))" })")
        XCTAssertTrue(
            runs.contains { $0.0.contains("hello") && $0.1 == Color.green },
            "the string literal colors green")
        XCTAssertTrue(
            runs.contains { $0.0 == "let" && $0.1 == Color.purple },
            "the keyword colors purple")
    }

    func testHighlighterBlockCommentSpansLines() throws {
        let code = "/* start\nmiddle\nend */\nafter"
        let highlighted = try XCTUnwrap(
            CodeSyntaxHighlighter.highlight(code, language: "c"))
        var runs: [(String, Color?)] = []
        for run in highlighted.runs {
            runs.append((String(highlighted[run.range].characters), run.foregroundColor))
        }
        let commentRun = runs.first {
            $0.0.contains("middle") && $0.1 == Color.secondary
        }
        XCTAssertNotNil(commentRun, "the multi-line comment stays one comment; runs: \(runs)")
        let plainTail = runs.last { $0.0.contains("after") }
        XCTAssertEqual(plainTail?.1, nil, "text after the comment closes is plain")
    }
}
