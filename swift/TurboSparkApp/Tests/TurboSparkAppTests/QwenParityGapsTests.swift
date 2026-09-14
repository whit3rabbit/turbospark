import XCTest

@testable import TurboSparkApp

/// Cross-concurrency capture box, the same shape the other suites use.
private final class StateBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var _value: T
    init(_ value: T) { self._value = value }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return _value }
        set { lock.lock(); defer { lock.unlock() }; _value = newValue }
    }
}

/// Tests for the second qwen-code parity pass: the git commands, bang
/// commands, split panes, sidebar organization and the interactive
/// AskUserQuestion flow. The pure pieces are asserted directly; the
/// interactive flow is driven through the same executor surface the
/// registry routes.
@MainActor
final class QwenParityGapsTests: XCTestCase {
    var appModel: AppModel!
    private var savedSplitPaneIDs: [String]?

    override func setUp() {
        super.setUp()
        appModel = AppModel()
        savedSplitPaneIDs = UserDefaults.standard.stringArray(forKey: "TurboSpark.splitChatIDs")
        UserDefaults.standard.removeObject(forKey: "TurboSpark.splitChatIDs")
        AskUserQuestionExecutor.answerWaiter = nil
    }

    override func tearDown() {
        appModel = nil
        if let savedSplitPaneIDs {
            UserDefaults.standard.set(savedSplitPaneIDs, forKey: "TurboSpark.splitChatIDs")
        } else {
            UserDefaults.standard.removeObject(forKey: "TurboSpark.splitChatIDs")
        }
        AskUserQuestionExecutor.answerWaiter = nil
        super.tearDown()
    }

    private func makeChat(title: String = "T", messages: [AppChatMessage] = []) -> AppChat {
        let chat = AppChat(title: title, messages: messages)
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return chat
    }

    // MARK: - New local commands

    func testGitAndSessionCommandRowsAreRecognized() {
        for name in ["diff", "log", "prs", "reset", "resume"] {
            XCTAssertTrue(BuiltInSlashCommand.isLocalCommand("/\(name)"), name)
            XCTAssertTrue(
                BuiltInSlashCommand.recognizedLocalNames.contains(name), name)
        }
        XCTAssertNil(BuiltInSlashCommand.matching("diff"))
    }

    func testResetBehavesLikeClear() {
        makeChat(messages: [
            AppChatMessage(role: .user, content: "hi"),
            AppChatMessage(role: .assistant, content: "hello"),
        ])
        appModel.handleLocalCommand("/reset")
        XCTAssertTrue(appModel.selectedChat.messages.isEmpty)
    }

    func testDiffWithoutAProjectRefusesQuietly() {
        // A projectless chat has no root to diff; the sheet must NOT open,
        // which is what distinguishes the refusal from a broken command.
        makeChat()
        appModel.handleLocalCommand("/diff")
        XCTAssertFalse(appModel.showGitSheet)
        appModel.handleLocalCommand("/log")
        XCTAssertFalse(appModel.showGitSheet)
        appModel.handleLocalCommand("/prs")
        XCTAssertFalse(appModel.showGitSheet)
    }

    func testPullRequestRowsParseFromGhJSON() {
        let json = """
        [{"number": 7, "title": "Add gate", "state": "OPEN", "headRefName": "gate", "url": "https://example.test/pr/7"},
         {"number": 6, "title": "Fix ring", "state": "MERGED", "headRefName": "ring", "url": "https://example.test/pr/6"}]
        """
        let rows = AppModel.parsePullRequests(json)
        XCTAssertEqual(rows.count, 2)
        XCTAssertEqual(rows[0].number, 7)
        XCTAssertEqual(rows[0].state, "OPEN")
        XCTAssertEqual(rows[1].headBranch, "ring")
        XCTAssertTrue(AppModel.parsePullRequests("not json").isEmpty)
    }

    func testGitProcessOutputUsesSharedByteCap() async {
        let result = await AppModel.runProcess(
            executable: "/bin/sh",
            arguments: ["-c", "yes x | head -c 2000000"],
            workingDirectory: NSTemporaryDirectory())

        XCTAssertEqual(result.exitCode, 0)
        XCTAssertLessThan(result.stdout.utf8.count, 1_000_100)
        XCTAssertTrue(result.stdout.contains("output truncated at 1000000 bytes"))
    }

    func testDiffBodyIsCappedByLineCount() {
        let body = (0...AppModel.gitDiffLineLimit).map(String.init).joined(separator: "\n")
        let limited = AppModel.limitDiffLines(body)

        XCTAssertTrue(limited.hasSuffix("diff truncated at 10000 lines)"))
        XCTAssertEqual(limited.filter { $0 == "\n" }.count, AppModel.gitDiffLineLimit)
    }

    // MARK: - Bang commands

    func testBangCommandPredicate() {
        XCTAssertTrue(AppModel.isBangCommand("!ls -la"))
        XCTAssertTrue(AppModel.isBangCommand("!  git status"))
        XCTAssertFalse(AppModel.isBangCommand("ls -la"))
        XCTAssertFalse(AppModel.isBangCommand("!"))
        XCTAssertFalse(AppModel.isBangCommand("!  "))
        // `!!` is the shell's own history spelling, not a command.
        XCTAssertFalse(AppModel.isBangCommand("!!"))
        XCTAssertFalse(AppModel.isBangCommand("!!! amazing"))
    }

    func testShellMessageContentRoundTrip() {
        let stored = "!git status\n\n<shell_output>\non branch main\n</shell_output>"
        let parsed = ShellMessageContent.parse(stored)
        XCTAssertEqual(parsed?.command, "git status")
        XCTAssertEqual(parsed?.output, "on branch main")
        // A bare command (interrupted run) still parses, with no output.
        let bare = ShellMessageContent.parse("!ping host")
        XCTAssertEqual(bare?.command, "ping host")
        XCTAssertNil(bare?.output)
        // Ordinary messages are not shell rows, even ones that merely
        // mention a bang mid-text.
        XCTAssertNil(ShellMessageContent.parse("use ! for emphasis"))
    }

    func testBoundedShellOutputClampsHeadAndTail() {
        let small = String(repeating: "x", count: 100)
        XCTAssertEqual(AppModel.boundedShellOutput(small), small)
        let big = String(repeating: "a", count: 6_000) + String(repeating: "b", count: 6_000)
        let bounded = AppModel.boundedShellOutput(big)
        XCTAssertTrue(bounded.contains("characters truncated"))
        XCTAssertTrue(bounded.hasPrefix("aaaa"))
        XCTAssertTrue(bounded.hasSuffix("bbbb"))
        XCTAssertLessThan(bounded.count, big.count)
    }

    // MARK: - Split panes

    func testSplitPaneAddRemoveAndCap() {
        var chats: [AppChat] = []
        for index in 0..<4 {
            chats.append(AppChat(title: "Chat \(index)", messages: []))
        }
        appModel.chats = chats
        appModel.selectedChatID = chats[0].id

        appModel.addSplitPane(chatID: chats[1].id)
        appModel.addSplitPane(chatID: chats[2].id)
        appModel.addSplitPane(chatID: chats[3].id)
        // The selected chat never becomes a pane of itself.
        appModel.addSplitPane(chatID: chats[0].id)
        XCTAssertEqual(appModel.splitChatIDs.count, 3)

        // Past the cap the OLDEST pane drops. Move the selection to a chat
        // already on screen so chats[0] becomes eligible for a pane.
        appModel.selectedChatID = chats[3].id
        appModel.addSplitPane(chatID: chats[0].id)
        XCTAssertFalse(appModel.splitChatIDs.contains(chats[1].id))
        XCTAssertTrue(appModel.splitChatIDs.contains(chats[0].id))

        appModel.closeSplitPane(chatID: chats[0].id)
        XCTAssertFalse(appModel.isSplitPane(chatID: chats[0].id))
    }

    func testGhostChatsCannotBecomePanes() {
        let chat = makeChat()
        appModel.chats[0].isGhost = true
        appModel.addSplitPane(chatID: chat.id)
        XCTAssertTrue(appModel.splitChatIDs.isEmpty)
    }

    // MARK: - Sidebar organization

    func testChatDateBucketDayBoundaries() {
        let calendar = Calendar.current
        let now = calendar.date(from: DateComponents(year: 2026, month: 9, day: 8, hour: 10))!
        let today = calendar.date(byAdding: .hour, value: -1, to: now)!
        let lateYesterday = calendar.date(
            from: DateComponents(year: 2026, month: 9, day: 7, hour: 23, minute: 50))!
        let threeDaysAgo = calendar.date(byAdding: .day, value: -3, to: now)!
        let twoWeeksAgo = calendar.date(byAdding: .day, value: -14, to: now)!

        XCTAssertEqual(ChatDateBucket.bucket(for: today, now: now), .today)
        // A chat updated at 23:50 yesterday is YESTERDAY at 10:00, not
        // "within 24 hours" -- the buckets are day-boundary based.
        XCTAssertEqual(ChatDateBucket.bucket(for: lateYesterday, now: now), .yesterday)
        XCTAssertEqual(ChatDateBucket.bucket(for: threeDaysAgo, now: now), .previous7Days)
        XCTAssertEqual(ChatDateBucket.bucket(for: twoWeeksAgo, now: now), .older)
    }

    func testProjectAccentColorIsDeterministicPerID() {
        let id = UUID()
        let first = ProjectAccentColor.forProject(id: id)
        XCTAssertEqual(first, ProjectAccentColor.forProject(id: id))
        XCTAssertTrue(ProjectAccentColor.allCases.contains(first))
        XCTAssertNotNil(ProjectAccentColor.forProject(id: nil))
    }

    // MARK: - Interactive AskUserQuestion

    func testAwaitingAnswerResolvesWithTheSubmittedReply() async throws {
        let call = AppToolCall(
            name: "ask_user_question",
            arguments: [
                "question": "Proceed?",
                "options": "[\"Yes\", \"No\"]",
            ],
            category: .automation)
        let seenToolCallID = StateBox<UUID?>(nil)
        // The readiness semaphore closes the install-then-submit race: the
        // continuation is installed BEFORE signal, so a submit after the
        // wait can never fall on the floor.
        let ready = DispatchSemaphore(value: 0)
        AskUserQuestionExecutor.answerWaiter = { chatID, _, toolCallID in
            seenToolCallID.value = toolCallID
            return await withCheckedContinuation { continuation in
                AskUserQuestionExecutor.waitForAnswer(chatID: chatID, continuation: continuation)
                ready.signal()
            }
        }
        defer { AskUserQuestionExecutor.answerWaiter = nil }

        async let output = AskUserQuestionExecutor.executeAwaitingAnswer(
            arguments: call.arguments, chatID: nil, toolCallID: call.id)

        XCTAssertEqual(ready.wait(timeout: .now() + 5), .success, "question never parked")
        XCTAssertEqual(seenToolCallID.value, call.id)
        AskUserQuestionExecutor.submitAnswer(chatID: nil, answers: ["Proceed?": "Yes"])
        let result = try await output
        XCTAssertTrue(result.contains("Yes"), result)
        XCTAssertTrue(result.contains("The user answered"), result)
    }

    func testAwaitingAnswerDismissalIsAnAnswerNotAHang() async throws {
        let call = AppToolCall(
            name: "ask_user_question",
            arguments: [
                "question": "Use the cache?",
                "options": "[\"Yes\", \"No\"]",
            ],
            category: .automation)
        let ready = DispatchSemaphore(value: 0)
        AskUserQuestionExecutor.answerWaiter = { chatID, _, _ in
            await withCheckedContinuation { continuation in
                AskUserQuestionExecutor.waitForAnswer(chatID: chatID, continuation: continuation)
                ready.signal()
            }
        }
        defer { AskUserQuestionExecutor.answerWaiter = nil }

        async let output = AskUserQuestionExecutor.executeAwaitingAnswer(
            arguments: call.arguments, chatID: nil, toolCallID: call.id)

        XCTAssertEqual(ready.wait(timeout: .now() + 5), .success, "question never parked")
        AskUserQuestionExecutor.submitAnswer(chatID: nil, answers: [:], dismissed: true)
        let result = try await output
        XCTAssertTrue(result.contains("dismissed"), result)
    }

    func testSubmitWithoutAParkedQuestionIsANoOp() {
        // The idempotence guard: a stray submit (a double-click, a Stop
        // race) must not crash a second resume.
        AskUserQuestionExecutor.submitAnswer(chatID: nil, answers: ["Q": "A"])
        AskUserQuestionExecutor.submitAnswer(chatID: nil, answers: [:], dismissed: true)
    }
}
