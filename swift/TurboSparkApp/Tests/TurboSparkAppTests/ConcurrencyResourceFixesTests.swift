import AVFoundation
import XCTest

@testable import TurboSparkApp

/// Regression tests for the Concurrency and Resources-and-leaks review fixes.
/// Races are exercised through extracted seams or injectable fakes; nothing
/// here relies on a fixed sleep for correctness.
@MainActor
final class ConcurrencyResourceFixesTests: XCTestCase {
    // MARK: - MCP connection test must not revert concurrent edits

    func testDiscoveredToolsMergeIntoTheCurrentServerNotThePreTestCopy() {
        let model = AppModel()
        let original = McpServerConfig(
            name: "fixture", transport: .stdio(command: "npx"), isEnabled: true, autoApprove: true)
        model.globalMcpServers = [original]

        // The user switches the server off and drops auto-approve while the
        // (slow) connection test is still running.
        var edited = original
        edited.isEnabled = false
        edited.autoApprove = false
        model.globalMcpServers = [edited]

        let tool = McpDiscoveredTool(
            name: "t", description: "d", inputSchemaJSON: "{}", serverName: "fixture")
        XCTAssertTrue(model.setGlobalMcpDiscoveredTools(
            id: original.id, tools: [tool], transport: original.transport))

        let stored = model.globalMcpServers[0]
        XCTAssertEqual(stored.discoveredTools, [tool])
        XCTAssertFalse(stored.isEnabled, "the test result must not re-enable the server")
        XCTAssertFalse(stored.autoApprove, "the test result must not restore auto-approve")
    }

    func testDiscoveredToolsAreDroppedWhenTheServerIsGoneOrItsTransportChanged() {
        let model = AppModel()
        let original = McpServerConfig(name: "fixture", transport: .stdio(command: "old"))
        model.globalMcpServers = [original]
        let tool = McpDiscoveredTool(
            name: "t", description: "d", inputSchemaJSON: "{}", serverName: "fixture")

        var retargeted = original
        retargeted.transport = .stdio(command: "new")
        model.globalMcpServers = [retargeted]
        XCTAssertFalse(model.setGlobalMcpDiscoveredTools(
            id: original.id, tools: [tool], transport: original.transport))
        XCTAssertTrue(model.globalMcpServers[0].discoveredTools.isEmpty)

        model.globalMcpServers = []
        XCTAssertFalse(model.setGlobalMcpDiscoveredTools(
            id: original.id, tools: [tool], transport: original.transport))
    }

    // MARK: - Read-aloud: a replaced utterance's late callback

    func testStaleUtteranceCallbackDoesNotClearTheNewMessagesSpeakingState() {
        let speech = AppSpeechSynthesizer()
        let a = UUID()
        let b = UUID()
        speech.speak(text: "first message", messageID: a)
        guard let utteranceA = speech.currentUtterance else {
            return XCTFail("speak must record the active utterance")
        }
        speech.speak(text: "second message", messageID: b)
        XCTAssertEqual(speech.speakingMessageID, b)

        // didCancel(A) is delivered AFTER B started.
        speech.utteranceEnded(utteranceA)

        XCTAssertEqual(speech.speakingMessageID, b)
        XCTAssertTrue(speech.isSpeaking)

        if let utteranceB = speech.currentUtterance {
            speech.utteranceEnded(utteranceB)
        }
        XCTAssertNil(speech.speakingMessageID)
        XCTAssertFalse(speech.isSpeaking)
        speech.stop()
    }

    // MARK: - Goal cleared while the judge runs

    func testGoalIdentityDistinguishesAClearedOrReplacedGoal() {
        let t0 = Date(timeIntervalSince1970: 1_000)
        var goal = ChatGoalState(condition: "all tests pass", setAt: t0)
        XCTAssertTrue(AppModel.goalIsSame(goal, as: goal))
        // Counters change while a judge runs; identity must not.
        var advanced = goal
        advanced.iterations = 3
        advanced.isPaused = true
        XCTAssertTrue(AppModel.goalIsSame(advanced, as: goal))
        // A re-set goal with the same words is a different goal.
        let reset = ChatGoalState(condition: "all tests pass", setAt: t0.addingTimeInterval(5))
        XCTAssertFalse(AppModel.goalIsSame(reset, as: goal))
        goal.condition = "different"
        XCTAssertFalse(AppModel.goalIsSame(goal, as: advanced))
    }

    // MARK: - Double Return during an awaited submission

    func testSecondReturnOnTheInFlightDraftIsNotQueued() {
        XCTAssertTrue(AppModel.isDuplicateOfInFlightSubmission(
            draft: "A", inFlight: "A", submitting: true, generating: false, hasPendingCall: false))
        // A different draft typed meanwhile is a real message and may queue.
        XCTAssertFalse(AppModel.isDuplicateOfInFlightSubmission(
            draft: "B", inFlight: "A", submitting: true, generating: false, hasPendingCall: false))
        // Once the turn is generating, the composer was cleared; same text is new input.
        XCTAssertFalse(AppModel.isDuplicateOfInFlightSubmission(
            draft: "A", inFlight: "A", submitting: true, generating: true, hasPendingCall: false))
        XCTAssertFalse(AppModel.isDuplicateOfInFlightSubmission(
            draft: "A", inFlight: nil, submitting: false, generating: false, hasPendingCall: false))
    }

    // MARK: - Concurrent AskUserQuestion calls in one chat

    func testSecondQuestionCardSurvivesTheFirstCallResuming() async {
        let model = AppModel()
        let chat = UUID()
        let items = [UserQuestionItem(
            question: "q", header: "h",
            options: [UserQuestionOption(label: "x", description: "x")])]

        let first = Task { @MainActor in
            await model.waitForUserAnswers(chatID: chat, items: items, toolCallID: UUID())
        }
        // The first call publishes its card and parks synchronously on the
        // main actor, so one yield is enough for it to be suspended.
        while model.pendingUserQuestions == nil { await Task.yield() }
        await Task.yield()
        let firstCardID = model.pendingUserQuestions?.id

        let second = Task { @MainActor in
            await model.waitForUserAnswers(chatID: chat, items: items, toolCallID: UUID())
        }
        while model.pendingUserQuestions?.id == firstCardID { await Task.yield() }
        let secondCardID = model.pendingUserQuestions?.id
        XCTAssertNotNil(secondCardID)

        // The second ask retired the first continuation (auto-dismiss); wait
        // for the first call to resume and clean up.
        let firstAnswer = await first.value
        XCTAssertTrue(firstAnswer.contains("dismissed"))
        XCTAssertEqual(
            model.pendingUserQuestions?.id, secondCardID,
            "the first call resuming must not erase the second call's card")

        if model.pendingUserQuestions == nil {
            // Regression path: unblock the parked waiter so the test fails
            // instead of hanging.
            AskUserQuestionExecutor.submitAnswer(chatID: chat, answers: ["h": "x"])
        }
        model.submitUserQuestionAnswers(["h": "x"])
        let secondAnswer = await second.value
        XCTAssertTrue(secondAnswer.contains("The user answered"))
        XCTAssertNil(model.pendingUserQuestions)
    }

    // MARK: - Chat install routing for the server

    func testChatInstallRoutingRequiresALiveSessionOnTheSamePath() {
        XCTAssertTrue(AppModel.isChatInstall(modelPath: "/m/a", selectedPath: "/m/a", hasSession: true))
        XCTAssertFalse(AppModel.isChatInstall(modelPath: "/m/a", selectedPath: "/m/a", hasSession: false))
        XCTAssertFalse(AppModel.isChatInstall(modelPath: "/m/a", selectedPath: "/m/b", hasSession: true))
        XCTAssertFalse(AppModel.isChatInstall(modelPath: "/m/a", selectedPath: nil, hasSession: true))
    }

    // MARK: - Memory capture output parsing

    func testFencedAndProseWrappedJSONIsExtractedForMemoryDecoding() throws {
        let fenced = "```json\n{\"claims\": []}\n```"
        let data = try XCTUnwrap(AppModel.memoryJSONObjectData(from: fenced))
        XCTAssertNotNil(try? JSONSerialization.jsonObject(with: data))
        let prose = "Here you go: {\"claims\": [{\"a\": {\"b\": 1}}]} hope that helps"
        let proseData = try XCTUnwrap(AppModel.memoryJSONObjectData(from: prose))
        let object = try JSONSerialization.jsonObject(with: proseData) as? [String: Any]
        XCTAssertNotNil(object?["claims"])
        XCTAssertNil(AppModel.memoryJSONObjectData(from: "no object here"))
    }

    // MARK: - TypeSafe start failure wording

    func testOpaqueServerErrorBecomesAPortInUseHint() {
        struct ServerError: Error {}
        let message = AppModel.typeSafeStartFailureMessage(ServerError(), port: 18080)
        XCTAssertTrue(message.contains("127.0.0.1:18080"), message)
        XCTAssertTrue(message.contains("already be in use"), message)
        struct Other: LocalizedError { var errorDescription: String? { "boom" } }
        XCTAssertEqual(AppModel.typeSafeStartFailureMessage(Other(), port: 1), "boom")
    }

    // MARK: - Bang commands are turns

    func testBangCommandHoldsSubmittingRefusesAConcurrentOneAndAppendsOneRow() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("bang-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        let project = AppProject(name: "Bang", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id
        let chat = AppChat(title: "Bang", messages: [])
        model.chats = [chat]
        model.selectedChatID = chat.id
        let chatID = chat.id

        model.handleBangCommand("!echo bang-ok")
        XCTAssertTrue(model.submitting, "a bang command is a turn: Send must be gated")
        let task = try XCTUnwrap(model.submissionTask, "Stop must be able to reach it")

        // A second one while the first runs is refused, not run concurrently.
        let taskBefore = model.submissionTask
        model.handleBangCommand("!echo second")
        XCTAssertEqual(model.submissionTask == nil, taskBefore == nil)

        await task.value
        XCTAssertFalse(model.submitting)
        XCTAssertNil(model.submissionTask)
        let rows = model.turnMessages(for: chatID).filter { $0.content.hasPrefix("!") }
        XCTAssertEqual(rows.count, 1)
        XCTAssertTrue(rows.first?.content.contains("bang-ok") ?? false)
    }

    func testStopCancelsARunningBangCommandAndAppendsNoRow() async throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("bang-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        let project = AppProject(name: "Bang", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id
        let chat = AppChat(title: "Bang", messages: [])
        model.chats = [chat]
        model.selectedChatID = chat.id
        let chatID = chat.id

        model.handleBangCommand("!sleep 30")
        let task = try XCTUnwrap(model.submissionTask)
        XCTAssertTrue(model.canCancel)
        model.cancel()

        let started = Date()
        await task.value
        XCTAssertLessThan(Date().timeIntervalSince(started), 15, "Stop must end the command, not wait it out")
        XCTAssertTrue(model.turnMessages(for: chatID).filter { $0.content.hasPrefix("!") }.isEmpty)
        XCTAssertFalse(model.isCancellationPending, "the cancel latch must not outlive the command")
    }

    // MARK: - Shell cwd is per chat

    func testShellCwdIsNotSharedBetweenChatsOnTheSameProject() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("cwd-\(UUID().uuidString)", isDirectory: true)
        let sub = root.appendingPathComponent("Tests", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let tracker = ShellCwdTracker()
        let chatA = UUID()
        let chatB = UUID()
        XCTAssertNil(tracker.record(finalDirectory: sub.path, root: root, chatID: chatA))
        XCTAssertEqual(tracker.startDirectory(for: root, chatID: chatA).path, sub.path)
        XCTAssertEqual(
            tracker.startDirectory(for: root, chatID: chatB).path, root.path,
            "chat B never ran cd and must start at the root")
    }
}

/// Browser, REPL, process and shell resource fixes (no model, no main actor).
final class ResourceLeakFixesTests: XCTestCase {
    // MARK: - REPL capture bounds and completion/error truncation

    func testCaptureBoundsMemoryWhileKeepingHeadAndTail() {
        let capture = REPLCallCapture(cap: 100)
        for i in 0..<10_000 {
            capture.append(REPLTextOutputEvent(level: .log, text: "line-\(i)"))
        }
        let snapshot = capture.snapshot()
        XCTAssertTrue(snapshot.dropped)
        XCTAssertLessThan(snapshot.events.count, 100, "a tight loop must not keep every event")
        XCTAssertEqual(snapshot.events.first?.text, "line-0")
        XCTAssertEqual(snapshot.events.last?.text, "line-9999")
    }

    func testCaptureIsSafeUnderConcurrentAppendAndSnapshot() {
        let capture = REPLCallCapture(cap: 1_000)
        DispatchQueue.concurrentPerform(iterations: 2_000) { i in
            if i % 2 == 0 {
                capture.append(REPLTextOutputEvent(level: .log, text: "e\(i)"))
            } else {
                _ = capture.snapshot()
                capture.capturedException = "x\(i)"
            }
        }
        XCTAssertFalse(capture.snapshot().events.isEmpty)
    }

    func testHugeCompletionAndErrorTextAreBoundedWithATruncatedFlag() async {
        let context = REPLWorkerContext(limits: REPLLimits(maximumOutputCharacters: 300))
        let big = await context.evaluate(code: "'x'.repeat(200000)")
        XCTAssertEqual(big.status, .completed)
        XCTAssertLessThan(big.completionText?.count ?? 0, 1_000)
        XCTAssertTrue(big.truncated)

        let loop = await context.evaluate(code: "for (let i = 0; i < 20000; i++) console.log('row' + i); 1")
        XCTAssertTrue(loop.truncated)
        XCTAssertLessThan(loop.consoleText.count, 2_000)
    }

    func testBoundTextKeepsHeadAndTailWithAMarker() {
        let text = String(repeating: "a", count: 500) + String(repeating: "z", count: 500)
        let bounded = REPLWorkerContext.boundText(text, cap: 90)
        XCTAssertTrue(bounded.truncated)
        XCTAssertTrue(bounded.text.hasPrefix("aaaa"))
        XCTAssertTrue(bounded.text.hasSuffix("zzzz"))
        XCTAssertTrue(bounded.text.contains("chars truncated"))
        XCTAssertFalse(REPLWorkerContext.boundText("short", cap: 90).truncated)
    }

    // MARK: - ProcessExecutor: live-child registry and detached-descendant pipe holders

    func testShutdownSweepKillsAForegroundChildAndItsTree() async throws {
        let task = Task {
            try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 60"],
                timeoutSeconds: 120)
        }
        // Poll for registration (bounded) instead of sleeping a fixed time.
        let deadline = Date().addingTimeInterval(10)
        while ProcessExecutor.liveChildCount == 0 && Date() < deadline {
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertGreaterThan(ProcessExecutor.liveChildCount, 0)

        let started = Date()
        ProcessExecutor.killAllLiveChildrenNow()
        let result = try await task.value
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
        XCTAssertNotEqual(result.exitCode, 0)
        XCTAssertEqual(ProcessExecutor.liveChildCount, 0, "a reaped child must leave the registry")
    }

    func testBackgroundedChildHoldingThePipeDoesNotKeepReaderThreadsBlocked() async throws {
        // `sleep 4 &` inherits the pipe write ends and outlives its shell.
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 4 & echo started"],
            timeoutSeconds: 30,
            mergeStreams: true)
        XCTAssertTrue(result.combinedText.contains("started"))
        // The readers must have released their descriptors: poll for the
        // process's fd count to settle instead of asserting at one instant.
        let fdDir = "/dev/fd"
        let baseline = (try? FileManager.default.contentsOfDirectory(atPath: fdDir).count) ?? 0
        let deadline = Date().addingTimeInterval(8)
        var settled = false
        while Date() < deadline {
            let now = (try? FileManager.default.contentsOfDirectory(atPath: fdDir).count) ?? 0
            if now <= baseline { settled = true; break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertTrue(settled)
    }

    // MARK: - Background shells

    private func launch(_ command: String, chatID: UUID?) throws -> BackgroundShellRecord {
        try BackgroundShellManager.shared.launch(
            command: command, startDirectory: FileManager.default.temporaryDirectory,
            environment: ProcessInfo.processInfo.environment, chatID: chatID, description: nil)
    }

    func testKillReturnsBeforeASluggishChildHasExited() async throws {
        let chat = UUID()
        // perl ignores SIGTERM, so the ladder takes ~2 s before SIGKILL.
        let record = try launch(
            "exec /usr/bin/perl -e '$SIG{TERM} = \"IGNORE\"; $| = 1; print \"ready\\n\"; sleep 30'",
            chatID: chat)
        defer { BackgroundShellManager.shared.resetForTests() }
        // Signalling before the handler is installed would just kill it.
        let ready = Date().addingTimeInterval(10)
        while !record.outputBuffer.text.contains("ready") && Date() < ready {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertTrue(record.outputBuffer.text.contains("ready"))

        let started = Date()
        XCTAssertTrue(BackgroundShellManager.shared.kill(record))
        XCTAssertLessThan(
            Date().timeIntervalSince(started), 1.0,
            "the SIGTERM/SIGKILL ladder must not run on the caller's thread")
        XCTAssertEqual(record.state, .killed)

        let deadline = Date().addingTimeInterval(15)
        while record.process.isRunning && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTAssertFalse(record.process.isRunning)
    }

    func testFinishedRecordsReleaseTheirPipeDescriptors() async throws {
        let fdDir = "/dev/fd"
        let baseline = (try? FileManager.default.contentsOfDirectory(atPath: fdDir).count) ?? 0
        let chat = UUID()
        defer { BackgroundShellManager.shared.resetForTests() }
        var records: [BackgroundShellRecord] = []
        for _ in 0..<15 { records.append(try launch("echo hi", chatID: chat)) }
        let deadline = Date().addingTimeInterval(20)
        while records.contains(where: { $0.state == .running }) && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        // 15 records pinned 30 descriptors before the fix; allow slack for
        // unrelated activity but not for the leak.
        var growth = Int.max
        while Date() < deadline {
            let now = (try? FileManager.default.contentsOfDirectory(atPath: fdDir).count) ?? 0
            growth = now - baseline
            if growth < 12 { break }
            try await Task.sleep(nanoseconds: 100_000_000)
        }
        XCTAssertLessThan(growth, 12, "finished background shells must not pin their pipes")
    }

    func testDeletedChatsFinishedRecordsAreForgotten() async throws {
        let chat = UUID()
        let other = UUID()
        defer { BackgroundShellManager.shared.resetForTests() }
        let mine = try launch("echo mine", chatID: chat)
        let theirs = try launch("echo theirs", chatID: other)
        let deadline = Date().addingTimeInterval(15)
        while (mine.state == .running || theirs.state == .running) && Date() < deadline {
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        BackgroundShellManager.shared.forgetFinishedRecords(chatID: chat)
        XCTAssertNil(BackgroundShellManager.shared.record(id: mine.id, chatID: chat))
        XCTAssertNotNil(BackgroundShellManager.shared.record(id: theirs.id, chatID: other))
    }

    // MARK: - HttpRequest body cap

    private final class BigBodyProtocol: URLProtocol, @unchecked Sendable {
        static let size = 64 * 1024
        override class func canInit(with request: URLRequest) -> Bool { request.url?.host == "big.invalid" }
        override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
        override func startLoading() {
            let url = request.url!
            // No Content-Length: forces the streaming cap, not the early refusal.
            let response = HTTPURLResponse(
                url: url, statusCode: 200, httpVersion: "HTTP/1.1",
                headerFields: ["Content-Type": "text/plain"])!
            client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(self, didLoad: Data(repeating: 0x61, count: Self.size))
            client?.urlProtocolDidFinishLoading(self)
        }
        override func stopLoading() {}
    }

    private func bigBodySession() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BigBodyProtocol.self]
        return URLSession(configuration: configuration)
    }

    func testResponseBodyOverTheCapIsRefused() async throws {
        let request = URLRequest(url: URL(string: "https://big.invalid/x")!)
        do {
            _ = try await HttpRequestExecutor.boundedData(
                for: request, session: bigBodySession(), delegate: nil, maxBytes: 1_024)
            XCTFail("a body over the cap must not be buffered and returned")
        } catch let error as NSError {
            XCTAssertEqual(error.domain, "TurboSparkHttpRequest")
            XCTAssertEqual(error.code, 8)
        }
    }

    func testResponseBodyWithinTheCapIsReturnedWhole() async throws {
        let request = URLRequest(url: URL(string: "https://big.invalid/x")!)
        let (data, _) = try await HttpRequestExecutor.boundedData(
            for: request, session: bigBodySession(), delegate: nil,
            maxBytes: BigBodyProtocol.size + 1)
        XCTAssertEqual(data.count, BigBodyProtocol.size)
    }
}
