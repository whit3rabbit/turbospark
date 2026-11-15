import XCTest

@testable import TurboSparkApp

/// The agent's `PushNotification` tool on the shared Allow-only path: opt-in
/// first, never an authorization request, sanitized text only. A fake
/// backend stands in for the system center, so nothing here posts.
@MainActor
final class PushNotificationToolTests: XCTestCase {
    private var savedDeliver: (@Sendable (String, String, UUID?) async -> AgentPushResult)?

    override func setUp() async throws {
        savedDeliver = PushNotificationExecutor.deliver
    }

    override func tearDown() async throws {
        PushNotificationExecutor.deliver = savedDeliver
        PushNotificationExecutor.onNotificationPushed = nil
    }

    // MARK: sanitization

    func testControlAndInvisibleCharactersAreStripped() {
        let hostile = "a\u{1B}[31mb\u{0}c\u{202E}d\u{200B}e\u{7}f\nline2\tx\r\ny"
        let clean = PushNotificationExecutor.sanitize(hostile, limit: 240)
        XCTAssertEqual(clean, "a[31mbcdef line2 x y")
        for scalar in clean.unicodeScalars {
            XCTAssertNotEqual(scalar.properties.generalCategory, .control)
            XCTAssertNotEqual(scalar.properties.generalCategory, .format)
        }
    }

    func testLengthIsCappedAndWhitespaceCollapsed() {
        let long = String(repeating: "word ", count: 200)
        let capped = PushNotificationExecutor.sanitize(long, limit: 40)
        XCTAssertLessThanOrEqual(capped.count, 40)
        XCTAssertTrue(capped.hasSuffix("\u{2026}"))
        XCTAssertEqual(PushNotificationExecutor.sanitize("  a   b  ", limit: 10), "a b")
        XCTAssertEqual(PushNotificationExecutor.sanitize("\n\u{0}\t", limit: 10), "")
    }

    func testOnlySanitizedTitleAndMessageAreHandedOver() async throws {
        final class Box: @unchecked Sendable { var got: (String, String, UUID?)? }
        let box = Box()
        PushNotificationExecutor.deliver = { title, body, chat in
            box.got = (title, body, chat)
            return .posted
        }
        let chat = UUID()
        let output = try await PushNotificationExecutor.execute(
            arguments: [
                "title": "T\u{1B}itle\n", "message": "Done\u{0}!", "status": "proactive",
                "command": "cat ~/.ssh/id_rsa", "path": "/etc/passwd",
            ], chatID: chat)
        XCTAssertEqual(box.got?.0, "Title")
        XCTAssertEqual(box.got?.1, "Done!")
        XCTAssertEqual(box.got?.2, chat)
        XCTAssertFalse(output.contains("id_rsa"))
        XCTAssertFalse(output.contains("passwd"))
        XCTAssertTrue(output.hasPrefix("Notification posted"))
    }

    func testEmptyTitleFallsBackAndEmptyMessageThrows() async throws {
        final class Box: @unchecked Sendable { var title: String? }
        let box = Box()
        PushNotificationExecutor.deliver = { title, _, _ in box.title = title; return .posted }
        _ = try await PushNotificationExecutor.execute(arguments: ["title": "\u{0}", "message": "m"])
        XCTAssertEqual(box.title, "TurboSpark Notice")
        do {
            _ = try await PushNotificationExecutor.execute(arguments: ["message": "\u{1B}\n"])
            XCTFail("an empty message must throw")
        } catch {
            XCTAssertEqual((error as NSError).code, 65)
        }
    }

    // MARK: results are non-errors that tell the agent what happened

    func testEveryNotSentOutcomeIsANonErrorResult() async throws {
        for (result, phrase) in [
            (AgentPushResult.optedOut, "not enabled in the app"),
            (.notAuthorized(.notDetermined), "system notifications are not enabled"),
            (.notAuthorized(.denied), "system notifications are not enabled"),
            (.notAuthorized(.unavailable), "system notifications are not enabled"),
            (.outranked, "pending approval or question"),
            (.failed, "refused"),
            (.ghostChat, "Ghost"),
        ] {
            PushNotificationExecutor.deliver = { _, _, _ in result }
            var toasts = 0
            final class Counter: @unchecked Sendable { var n = 0 }
            let counter = Counter()
            PushNotificationExecutor.onNotificationPushed = { _, _ in counter.n += 1 }
            let output = try await PushNotificationExecutor.execute(arguments: ["message": "m"])
            toasts = counter.n
            XCTAssertTrue(output.contains(phrase), "\(result): \(output)")
            XCTAssertTrue(output.contains("not sent"), output)
            XCTAssertEqual(toasts, 0, "no toast unless it was posted")
        }
    }

    // MARK: through the real wrapper

    private func wired(
        status: AttentionAuthorization, preferences: AttentionPreferences
    ) -> (FakeAttentionBackend, AppNotificationCenter) {
        let backend = FakeAttentionBackend()
        backend.status = status
        let center = AppNotificationCenter(backend: backend, isActive: { true }, playSound: { _ in })
        PushNotificationExecutor.deliver = { title, body, chat in
            await center.pushFromAgent(
                AttentionNotification(chatID: chat ?? UUID(), title: title, body: body),
                preferences: preferences)
        }
        return (backend, center)
    }

    func testOptInOffMeansNoPostAndNoPrompt() async throws {
        let (backend, _) = wired(status: .authorized, preferences: AttentionPreferences())
        let output = try await PushNotificationExecutor.execute(arguments: ["message": "hi"])
        XCTAssertTrue(output.contains("not enabled in the app"), output)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertEqual(backend.requests, 0)
    }

    func testNotAuthorizedMeansNoPostAndNoPrompt() async throws {
        let (backend, _) = wired(status: .notDetermined, preferences: AttentionPreferences(allowAgentPush: true))
        let output = try await PushNotificationExecutor.execute(arguments: ["message": "hi"])
        XCTAssertTrue(output.contains("system notifications are not enabled"), output)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertEqual(backend.requests, 0, "no authorization request at tool-call time")
    }

    func testOptedInAndAuthorizedPostsSanitizedTextOnly() async throws {
        let (backend, _) = wired(status: .authorized, preferences: AttentionPreferences(allowAgentPush: true))
        let chat = UUID()
        let output = try await PushNotificationExecutor.execute(
            arguments: ["title": "Build", "message": "ok\u{1B}[0m\nnext", "extra": "secret"], chatID: chat)
        XCTAssertTrue(output.hasPrefix("Notification posted"), output)
        XCTAssertEqual(backend.posted, [AttentionNotification(chatID: chat, title: "Build", body: "ok[0m next")])
        XCTAssertEqual(backend.requests, 0)
    }

    // MARK: the model path and the ladder

    func testModelRoutesThroughItsCenterAndRefusesGhostChats() async {
        let model = AppModel()
        model.stopCronScheduler()
        let backend = FakeAttentionBackend()
        backend.status = .authorized
        model.attentionCenter = AppNotificationCenter(backend: backend, isActive: { true }, playSound: { _ in })
        let normal = AppChat(title: "Normal")
        let ghost = AppChat(title: "Secret plans", isGhost: true)
        model.chats = [normal, ghost]

        let off = await model.deliverAgentPush(title: "t", body: "b", chatID: normal.id)
        XCTAssertEqual(off, .optedOut, "the switch defaults off")
        model.setAgentPushAllowed(true)
        XCTAssertTrue(model.attentionPreferences.allowAgentPush)
        let ghosted = await model.deliverAgentPush(title: "t", body: "b", chatID: ghost.id)
        XCTAssertEqual(ghosted, .ghostChat)
        XCTAssertTrue(backend.posted.isEmpty)
        let ok = await model.deliverAgentPush(title: "t", body: "b", chatID: normal.id)
        XCTAssertEqual(ok, .posted)
        XCTAssertEqual(backend.posted.map(\.title), ["t"])
        XCTAssertEqual(backend.requests, 0)
    }

    /// The permission ladder position is unchanged: an automation-category
    /// call scores low, as it did before the migration.
    func testLadderPositionIsUnchanged() {
        let assessment = ToolRiskClassifier.assessRisk(
            name: "PushNotification", arguments: ["message": "m"], projectURL: nil)
        XCTAssertEqual(assessment.level, .low)
        XCTAssertEqual(assessment.category, .automation)
    }
}
