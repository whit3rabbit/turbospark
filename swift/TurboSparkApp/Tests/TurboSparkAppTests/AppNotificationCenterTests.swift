import XCTest

@testable import TurboSparkApp

/// OC-02: a recording stand-in for `UNUserNotificationCenter`. Nothing here
/// posts a real notification or shows a real prompt.
@MainActor
final class FakeAttentionBackend: AttentionNotificationBackend {
    var status: AttentionAuthorization = .notDetermined
    var grantOnRequest = true
    var requests = 0
    var posted: [AttentionNotification] = []
    var removed: [[UUID]] = []
    var postError: Error?

    func authorizationStatus() async -> AttentionAuthorization { status }

    func requestAuthorization() async -> Bool {
        requests += 1
        status = grantOnRequest ? .authorized : .denied
        return grantOnRequest
    }

    func post(_ notification: AttentionNotification) async throws {
        if let postError { throw postError }
        posted.append(notification)
    }

    func removeDelivered(chatIDs: [UUID]) { removed.append(chatIDs) }
}

@MainActor
final class AppNotificationCenterTests: XCTestCase {
    private let chat = UUID()
    private var backend: FakeAttentionBackend!
    private var sounds: [String] = []
    private var active = false
    private var center: AppNotificationCenter!

    private var allOn: AttentionPreferences {
        var p = AttentionPreferences()
        for event in AttentionEvent.allCases {
            p[event] = AttentionEventPreference(enabled: true, sound: "Glass")
        }
        return p
    }

    override func setUp() async throws {
        backend = FakeAttentionBackend()
        sounds = []
        active = false
        center = AppNotificationCenter(
            backend: backend, isActive: { [unowned self] in self.active },
            playSound: { [unowned self] in self.sounds.append($0) })
    }

    private func note(_ body: String = "b") -> AttentionNotification {
        AttentionNotification(chatID: chat, title: "T", body: body)
    }

    // MARK: authorization

    func testNothingPostsBeforeTheUserAllows() async {
        // Status is the default (not determined) and events keep coming.
        await center.notify(.needsPermission, notification: note(), preferences: allOn)
        await center.notify(.turnFinished, notification: note(), preferences: allOn)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertEqual(backend.requests, 0, "an event must never start the system prompt")
        XCTAssertTrue(sounds.isEmpty)
        XCTAssertTrue(center.policy.slots.isEmpty, "no slot is claimed without authorization")
    }

    func testRefreshReadsStatusWithoutPrompting() async {
        backend.status = .denied
        await center.refreshAuthorization()
        XCTAssertEqual(center.authorization, .denied)
        XCTAssertEqual(backend.requests, 0)
    }

    func testAllowIsTheOnlyPathToThePromptAndItAsksOnce() async {
        await center.requestAuthorization()
        XCTAssertEqual(backend.requests, 1)
        XCTAssertEqual(center.authorization, .authorized)
        await center.requestAuthorization()
        XCTAssertEqual(backend.requests, 1, "already decided: no second request")
    }

    func testDeclinedAuthorizationPostsNothingAndKeepsTheStateString() async {
        backend.grantOnRequest = false
        await center.requestAuthorization()
        XCTAssertEqual(center.authorization, .denied)
        await center.notify(.error, notification: note(), preferences: allOn)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertTrue(sounds.isEmpty)
        // A denied status never re-prompts.
        await center.requestAuthorization()
        XCTAssertEqual(backend.requests, 1)
    }

    func testUnavailableOutsideTheBundleNeverPosts() async {
        backend.status = .unavailable
        await center.refreshAuthorization()
        await center.requestAuthorization()
        XCTAssertEqual(backend.requests, 0)
        await center.notify(.error, notification: note(), preferences: allOn)
        XCTAssertTrue(backend.posted.isEmpty)
    }

    // MARK: posting

    private func authorize() async {
        backend.status = .authorized
        await center.refreshAuthorization()
    }

    func testAuthorizedUnfocusedPostsWithChatIDAndPlaysTheSound() async {
        await authorize()
        await center.notify(.needsPermission, notification: note("Needs your approval"), preferences: allOn)
        XCTAssertEqual(backend.posted, [note("Needs your approval")])
        XCTAssertEqual(backend.posted.first?.chatID, chat, "the chat id is what a click routes by")
        XCTAssertEqual(sounds, ["Glass"])
    }

    func testFocusedAppPostsNothingAndPlaysNothing() async {
        await authorize()
        active = true
        await center.notify(.needsPermission, notification: note(), preferences: allOn)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertTrue(sounds.isEmpty)
    }

    func testSoundNoneStaysSilent() async {
        await authorize()
        var p = allOn
        p[.turnFinished].sound = "none"
        await center.notify(.turnFinished, notification: note(), preferences: p)
        XCTAssertEqual(backend.posted.count, 1)
        XCTAssertTrue(sounds.isEmpty)
    }

    func testAFailedPostPlaysNoSoundAndFreesTheSlot() async {
        await authorize()
        backend.postError = NSError(domain: "x", code: 1)
        await center.notify(.needsPermission, notification: note(), preferences: allOn)
        XCTAssertTrue(sounds.isEmpty)
        XCTAssertNil(center.policy.slots[chat], "a post that did not happen must not block the next")
        backend.postError = nil
        await center.notify(.turnFinished, notification: note(), preferences: allOn)
        XCTAssertEqual(backend.posted.count, 1)
    }

    func testDedupeKeepsTheBlockingCue() async {
        await authorize()
        await center.notify(.needsPermission, notification: note("approve"), preferences: allOn)
        await center.notify(.turnFinished, notification: note("done"), preferences: allOn)
        XCTAssertEqual(backend.posted.map(\.body), ["approve"])
        XCTAssertEqual(sounds.count, 1, "the outranked event plays no sound either")
    }

    // MARK: release and click

    func testBecomingActiveReleasesEverySlotAndClearsDeliveredBanners() async {
        await authorize()
        await center.notify(.needsPermission, notification: note(), preferences: allOn)
        center.appDidBecomeActive()
        XCTAssertTrue(center.policy.slots.isEmpty)
        XCTAssertEqual(backend.removed.last, [chat])
        await center.notify(.turnFinished, notification: note("done"), preferences: allOn)
        XCTAssertEqual(backend.posted.last?.body, "done")
    }

    func testClickSelectsTheChatAndClearsItsSlot() async {
        await authorize()
        var opened: [UUID] = []
        center.openChat = { opened.append($0) }
        await center.notify(.needsAnswer, notification: note(), preferences: allOn)
        center.handleClick(chatID: chat)
        XCTAssertEqual(opened, [chat])
        XCTAssertNil(center.policy.slots[chat])
    }

    // MARK: the prompt API stays behind the Allow path

    /// `UNUserNotificationCenter.requestAuthorization` may be called only from
    /// the wrapper's backend (and the pre-existing agent PushNotification
    /// tool), and the wrapper's own `requestAuthorization()` only from the
    /// Settings Allow button. A new caller fails here instead of shipping a
    /// prompt nobody asked for.
    func testOnlyTheAllowButtonStartsTheSystemPrompt() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        let files = try XCTUnwrap(FileManager.default.enumerator(at: sources, includingPropertiesForKeys: nil))
            .compactMap { $0 as? URL }.filter { $0.pathExtension == "swift" }
        var systemCallers: Set<String> = []
        var allowCallers: Set<String> = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for line in text.components(separatedBy: "\n") {
                let code = line.trimmingCharacters(in: .whitespaces)
                if code.hasPrefix("//") { continue }
                if code.contains("requestAuthorization(options:") { systemCallers.insert(file.lastPathComponent) }
                if code.contains("center.requestAuthorization()") { allowCallers.insert(file.lastPathComponent) }
            }
        }
        // The agent's PushNotification tool no longer prompts: only the
        // wrapper's backend may call the system API.
        XCTAssertEqual(systemCallers, ["AppNotificationCenter.swift"])
        XCTAssertEqual(allowCallers, ["NotificationsSettingsSection.swift"])
    }

    /// The tool's file must not touch the system center at all: it hands text
    /// to the wrapper's Allow-only path.
    func testThePushNotificationToolNeverImportsOrCallsTheSystemCenter() throws {
        let file = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Tools/Automation/AutomationExecutors.swift")
        let code = try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).hasPrefix("//") }.joined(separator: "\n")
        for forbidden in ["UserNotifications", "UNUserNotificationCenter", "requestAuthorization"] {
            XCTAssertFalse(code.contains(forbidden), forbidden)
        }
    }

    // MARK: agent push (the Allow-only path)

    private var optedIn: AttentionPreferences { AttentionPreferences(allowAgentPush: true) }

    func testAgentPushIsOffByDefaultAndTouchesNothing() async {
        backend.status = .authorized
        let result = await center.pushFromAgent(note("hi"), preferences: AttentionPreferences())
        XCTAssertEqual(result, .optedOut)
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertEqual(backend.requests, 0)
        XCTAssertTrue(center.policy.outstandingChats.isEmpty)
        XCTAssertEqual(AttentionPreferences().allowAgentPush, false)
        // The four events being on does not opt the agent in.
        let result2 = await center.pushFromAgent(note("hi"), preferences: allOn)
        XCTAssertEqual(result2, .optedOut)
        XCTAssertTrue(backend.posted.isEmpty)
    }

    func testAgentPushNeverStartsTheSystemPrompt() async {
        for status in [AttentionAuthorization.notDetermined, .denied, .unavailable] {
            backend.status = status
            let result = await center.pushFromAgent(note("hi"), preferences: optedIn)
            XCTAssertEqual(result, .notAuthorized(status))
        }
        XCTAssertEqual(backend.requests, 0, "a tool call must never request authorization")
        XCTAssertTrue(backend.posted.isEmpty)
        XCTAssertTrue(center.policy.outstandingChats.isEmpty)
    }

    func testAgentPushPostsWhenAllowedEvenWhileTheAppIsActive() async {
        backend.status = .authorized
        active = true
        let result = await center.pushFromAgent(note("done"), preferences: optedIn)
        XCTAssertEqual(result, .posted)
        XCTAssertEqual(backend.posted, [note("done")])
        XCTAssertTrue(sounds.isEmpty, "an agent push is silent")
        XCTAssertEqual(backend.requests, 0)
    }

    func testAgentPushNeverReplacesABlockingCueButReplacesAFinish() async {
        backend.status = .authorized
        await center.refreshAuthorization()
        await center.notify(.needsPermission, notification: note("approve"), preferences: allOn)
        XCTAssertEqual(backend.posted.count, 1)
        var both = allOn
        both.allowAgentPush = true
        let blocked = await center.pushFromAgent(note("x"), preferences: both)
        XCTAssertEqual(blocked, .outranked)
        XCTAssertEqual(backend.posted.count, 1)

        center.handleClick(chatID: chat)  // releases the slot
        await center.notify(.turnFinished, notification: note("fin"), preferences: allOn)
        let replaced = await center.pushFromAgent(note("agent"), preferences: both)
        XCTAssertEqual(replaced, .posted)
        // A later finish does not clobber the agent's banner.
        await center.notify(.turnFinished, notification: note("fin2"), preferences: allOn)
        XCTAssertEqual(backend.posted.map(\.body), ["approve", "fin", "agent"])
    }

    func testAgentPushPostFailureGivesTheSlotBack() async {
        backend.status = .authorized
        backend.postError = NSError(domain: "x", code: 1)
        let failed = await center.pushFromAgent(note("a"), preferences: optedIn)
        XCTAssertEqual(failed, .failed)
        XCTAssertTrue(center.policy.outstandingChats.isEmpty)
        backend.postError = nil
        let ok = await center.pushFromAgent(note("b"), preferences: optedIn)
        XCTAssertEqual(ok, .posted)
    }

    func testActivationClearsAnAgentPushCue() async {
        backend.status = .authorized
        _ = await center.pushFromAgent(note("a"), preferences: optedIn)
        center.appDidBecomeActive()
        XCTAssertTrue(center.policy.outstandingChats.isEmpty)
        XCTAssertEqual(backend.removed.last, [chat])
    }
}
