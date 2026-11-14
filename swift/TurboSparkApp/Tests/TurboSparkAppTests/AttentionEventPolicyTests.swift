import XCTest

@testable import TurboSparkApp

/// OC-02: the delivery matrix. Pure, so no notification is ever posted: toggles,
/// the focus rule, dedupe slots and sound gating.
final class AttentionEventPolicyTests: XCTestCase {
    private let chatA = UUID()
    private let chatB = UUID()

    private func prefs(
        _ events: [AttentionEvent] = AttentionEvent.allCases, sound: AttentionSound = .glass
    ) -> AttentionPreferences {
        var p = AttentionPreferences()
        for event in events { p[event] = AttentionEventPreference(enabled: true, sound: sound.rawValue) }
        return p
    }

    private func policy(active: Bool) -> AttentionEventPolicy {
        AttentionEventPolicy(isActive: { active })
    }

    func testEverythingIsOffByDefault() {
        var p = policy(active: false)
        for event in AttentionEvent.allCases {
            let d = p.decide(event, chat: chatA, preferences: AttentionPreferences())
            XCTAssertEqual(d.outcome, .disabled)
            XCTAssertFalse(d.shouldPost)
        }
        XCTAssertTrue(p.slots.isEmpty, "a disabled event claims no slot")
    }

    func testFocusedAppIsSilent() {
        var p = policy(active: true)
        let d = p.decide(.needsPermission, chat: chatA, preferences: prefs())
        XCTAssertEqual(d.outcome, .appActive)
        XCTAssertNil(d.sound)
        XCTAssertTrue(p.slots.isEmpty, "a suppressed event must not block a later one")
    }

    func testUnfocusedAppDeliversWithTheEventsSound() {
        var p = policy(active: false)
        let d = p.decide(.error, chat: chatA, preferences: prefs(sound: .basso))
        XCTAssertEqual(d.outcome, .deliver)
        XCTAssertEqual(d.sound?.systemName, "Basso")
        XCTAssertEqual(p.slots[chatA], .error)
    }

    func testOnlyTheToggledEventDelivers() {
        var p = policy(active: false)
        let only = prefs([.needsAnswer])
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: only).outcome, .disabled)
        XCTAssertEqual(p.decide(.needsAnswer, chat: chatA, preferences: only).outcome, .deliver)
    }

    func testSoundNoneIsSilentButStillDelivers() {
        var p = policy(active: false)
        let d = p.decide(.turnFinished, chat: chatA, preferences: prefs(sound: .none))
        XCTAssertEqual(d.outcome, .deliver)
        XCTAssertNil(d.sound?.systemName)
    }

    func testUnknownStoredSoundLoadsAsSilence() {
        XCTAssertEqual(AttentionSound(storedValue: "Sosumi-from-the-future"), .none)
        let p = AttentionEventPreference(enabled: true, sound: "garbage")
        XCTAssertEqual(p.soundChoice, .none)
    }

    // MARK: dedupe

    func testNewerEventReplacesAnEqualOrLowerPendingOne() {
        var p = policy(active: false)
        let all = prefs()
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: all).outcome, .deliver)
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: all).outcome, .deliver)
        XCTAssertEqual(p.decide(.error, chat: chatA, preferences: all).outcome, .deliver)
        XCTAssertEqual(p.decide(.needsPermission, chat: chatA, preferences: all).outcome, .deliver)
        XCTAssertEqual(p.decide(.needsAnswer, chat: chatA, preferences: all).outcome, .deliver)
        XCTAssertEqual(p.slots[chatA], .needsAnswer)
    }

    func testABlockingCueIsNotReplacedByAFinishOrAnError() {
        var p = policy(active: false)
        let all = prefs()
        _ = p.decide(.needsPermission, chat: chatA, preferences: all)
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: all).outcome, .outranked)
        XCTAssertEqual(p.decide(.error, chat: chatA, preferences: all).outcome, .outranked)
        XCTAssertEqual(p.slots[chatA], .needsPermission)
    }

    func testAnErrorIsNotReplacedByAFinish() {
        var p = policy(active: false)
        _ = p.decide(.error, chat: chatA, preferences: prefs())
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: prefs()).outcome, .outranked)
    }

    func testSlotsAreIndependentPerChat() {
        var p = policy(active: false)
        _ = p.decide(.needsPermission, chat: chatA, preferences: prefs())
        XCTAssertEqual(p.decide(.turnFinished, chat: chatB, preferences: prefs()).outcome, .deliver)
    }

    func testReleaseAllLetsTheNextEventThrough() {
        var p = policy(active: false)
        _ = p.decide(.needsPermission, chat: chatA, preferences: prefs())
        p.releaseAll()
        XCTAssertEqual(p.decide(.turnFinished, chat: chatA, preferences: prefs()).outcome, .deliver)
    }

    func testReleaseOneChatLeavesTheOthers() {
        var p = policy(active: false)
        _ = p.decide(.needsPermission, chat: chatA, preferences: prefs())
        _ = p.decide(.needsPermission, chat: chatB, preferences: prefs())
        p.release(chat: chatA)
        XCTAssertNil(p.slots[chatA])
        XCTAssertEqual(p.slots[chatB], .needsPermission)
    }

    func testFocusIsReadAtDecisionTime() {
        var active = true
        var p = AttentionEventPolicy(isActive: { active })
        XCTAssertEqual(p.decide(.error, chat: chatA, preferences: prefs()).outcome, .appActive)
        active = false
        XCTAssertEqual(p.decide(.error, chat: chatA, preferences: prefs()).outcome, .deliver)
    }

    func testPreferencesRoundTripAndTolerateJunk() throws {
        var p = AttentionPreferences()
        p[.needsAnswer] = AttentionEventPreference(enabled: true, sound: "Ping")
        let data = try JSONEncoder().encode(p)
        XCTAssertEqual(try JSONDecoder().decode(AttentionPreferences.self, from: data), p)
        XCTAssertTrue(p.anyEnabled)
        XCTAssertFalse(AttentionPreferences().anyEnabled)

        // The whole field decoding as junk must not lose the other settings.
        let settings = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"attentionEvents": 7, "chatSidebarMode": "inbox"}"#.utf8))
        XCTAssertEqual(settings.attentionEvents, AttentionPreferences())
        XCTAssertEqual(settings.chatSidebarMode, "inbox")
        let kept = MacAppSettings(attentionEvents: p)
        let round = try JSONDecoder().decode(MacAppSettings.self, from: JSONEncoder().encode(kept))
        XCTAssertEqual(round.attentionEvents, p)
    }
}
