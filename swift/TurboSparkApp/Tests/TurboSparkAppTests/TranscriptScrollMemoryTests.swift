import AppKit
import SwiftUI
import XCTest

@testable import TurboSparkApp

/// OC-04: per-tab transcript scroll memory. Pure rules first, then the
/// AppModel gate (tabs only, reveal outranks), then a real scroll view for the
/// restore itself.
@MainActor
final class TranscriptScrollMemoryTests: XCTestCase {
    // MARK: - Pure rules

    func testPinnedIsStoredAsFollowLatestNotAnOffset() {
        let a = UUID()
        var memory = TranscriptScrollMemory()
        memory.record(chatID: a, pinned: true, offset: 4200, keeping: [a])
        XCTAssertEqual(memory.restore(for: a, revealPending: false), .followLatest)
    }

    func testUnpinnedIsStoredAsAnOffsetAndNeverNegative() {
        let (a, b) = (UUID(), UUID())
        var memory = TranscriptScrollMemory()
        memory.record(chatID: a, pinned: false, offset: 640, keeping: [a, b])
        memory.record(chatID: b, pinned: false, offset: -30, keeping: [a, b])
        XCTAssertEqual(memory.restore(for: a, revealPending: false), .offset(640))
        XCTAssertEqual(memory.restore(for: b, revealPending: false), .offset(0))
    }

    func testAPendingRevealBeatsTheRememberedPosition() {
        let a = UUID()
        var memory = TranscriptScrollMemory()
        memory.record(chatID: a, pinned: false, offset: 640, keeping: [a])
        XCTAssertNil(memory.restore(for: a, revealPending: true))
        XCTAssertEqual(memory.restore(for: a, revealPending: false), .offset(640))
    }

    func testAnUnknownChatIsANoOp() {
        var memory = TranscriptScrollMemory()
        XCTAssertNil(memory.restore(for: UUID(), revealPending: false))
        memory.forget(UUID())
        XCTAssertEqual(memory, TranscriptScrollMemory())
    }

    func testOnlyChatsWithATabAreRemembered() {
        let (a, gone) = (UUID(), UUID())
        var memory = TranscriptScrollMemory()
        memory.record(chatID: gone, pinned: false, offset: 10, keeping: [gone, a])
        memory.record(chatID: a, pinned: false, offset: 20, keeping: [a])
        XCTAssertNil(memory.restore(for: gone, revealPending: false), "dropped when its tab went away")
        memory.record(chatID: gone, pinned: false, offset: 10, keeping: [a])
        XCTAssertNil(memory.restore(for: gone, revealPending: false))
    }

    func testUnpinnedWithNoReadingKeepsTheEarlierPosition() {
        let a = UUID()
        var memory = TranscriptScrollMemory()
        memory.record(chatID: a, pinned: false, offset: 90, keeping: [a])
        memory.record(chatID: a, pinned: false, offset: nil, keeping: [a])
        XCTAssertEqual(memory.restore(for: a, revealPending: false), .offset(90))
    }

    func testReachableOffsetClampsToTheContent() {
        XCTAssertEqual(TranscriptFollowRules.reachableOffset(target: 500, contentHeight: 2000, viewportHeight: 300), 500)
        XCTAssertEqual(TranscriptFollowRules.reachableOffset(target: 5000, contentHeight: 2000, viewportHeight: 300), 1700)
        XCTAssertEqual(TranscriptFollowRules.reachableOffset(target: -5, contentHeight: 100, viewportHeight: 300), 0)
    }

    // MARK: - AppModel gate

    private func twoTabModel() -> (AppModel, AppChat, AppChat) {
        let model = AppModel()
        var c1 = AppChat(title: "One")
        var c2 = AppChat(title: "Two")
        c1.messages = [AppChatMessage(role: .user, content: "one")]
        c2.messages = [AppChatMessage(role: .user, content: "two")]
        model.chats = [c1, c2]
        model.selectedChatID = c1.id
        var tabs = SessionTabStore()
        tabs.open(c2.id, current: c1.id)
        model.sessionTabs = tabs
        return (model, c1, c2)
    }

    func testModelRoundTripsAnOffsetAndAPinnedReader() {
        let (model, c1, c2) = twoTabModel()
        defer { model.shutdown() }
        model.recordTranscriptScroll(chatID: c1.id, pinned: false, offset: 777)
        model.recordTranscriptScroll(chatID: c2.id, pinned: true, offset: 123)
        XCTAssertEqual(model.transcriptScrollRestore(for: c1.id), .offset(777))
        XCTAssertEqual(model.transcriptScrollRestore(for: c2.id), .followLatest)
    }

    func testModelUnknownChatIsANoOp() {
        let (model, _, _) = twoTabModel()
        defer { model.shutdown() }
        let stranger = UUID()
        model.recordTranscriptScroll(chatID: stranger, pinned: false, offset: 50)
        XCTAssertNil(model.transcriptScrollRestore(for: stranger))
        XCTAssertEqual(model.transcriptScroll, TranscriptScrollMemory())
    }

    func testNothingIsRememberedWhileTabsAreDormant() {
        let (model, c1, _) = twoTabModel()
        defer { model.shutdown() }
        model.sessionTabs = SessionTabStore()
        model.recordTranscriptScroll(chatID: c1.id, pinned: false, offset: 50)
        XCTAssertNil(model.transcriptScrollRestore(for: c1.id))
    }

    func testAPendingRevealForThatChatBeatsTheRestore() throws {
        let (model, c1, c2) = twoTabModel()
        defer { model.shutdown() }
        model.recordTranscriptScroll(chatID: c1.id, pinned: false, offset: 300)
        model.recordTranscriptScroll(chatID: c2.id, pinned: false, offset: 400)
        let target = try XCTUnwrap(c1.messages.first)
        model.toolCardReveal = ToolCardReveal(messageID: target.id, callID: UUID(), token: 1)
        XCTAssertNil(model.transcriptScrollRestore(for: c1.id), "reveal wins")
        XCTAssertEqual(model.transcriptScrollRestore(for: c2.id), .offset(400), "a reveal for another chat does not")
        // Once a card consumed it, the position applies again.
        model.toolCardRevealConsumedToken = 1
        XCTAssertEqual(model.transcriptScrollRestore(for: c1.id), .offset(300))
    }

    // MARK: - The restore on a real scroll view

    private func hostScrollView(follower: TranscriptFollower) async throws -> (NSWindow, NSScrollView) {
        let root = ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(0..<120, id: \.self) { Text(verbatim: "Row \($0)").frame(height: 24) }
            }
            .background(TranscriptScrollObserver(follower: follower))
        }
        .frame(width: 400, height: 300)
        let host = NSHostingView(rootView: root)
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 400, height: 300), styleMask: [.borderless],
            backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        try await Task.sleep(for: .milliseconds(150))
        host.layoutSubtreeIfNeeded()
        func find(_ v: NSView) -> NSScrollView? {
            if let s = v as? NSScrollView { return s }
            for c in v.subviews { if let s = find(c) { return s } }
            return nil
        }
        // The caller keeps the window: a released window detaches the probe.
        return (window, try XCTUnwrap(find(host)))
    }

    func testRestoreLandsOnTheOffsetUnpinnedAndSettles() async throws {
        let follower = TranscriptFollower()
        let (window, scroll) = try await hostScrollView(follower: follower)
        defer { withExtendedLifetime(window) {} }
        XCTAssertNil(follower.memorySnapshot, "not settled yet: nothing to remember")
        await follower.restore(offset: 500)
        XCTAssertEqual(scroll.documentVisibleRect.minY, 500, accuracy: 1)
        XCTAssertFalse(follower.isPinned, "a restored offset is not following the latest")
        XCTAssertNotNil(follower.memorySnapshot)
    }

    func testAReaderScrollCancelsTheRestore() async throws {
        let follower = TranscriptFollower()
        let (window, scroll) = try await hostScrollView(follower: follower)
        defer { withExtendedLifetime(window) {} }
        follower.viewportMoved(distanceFromBottom: 900, userDriven: true)
        await follower.restore(offset: 500)
        XCTAssertNotEqual(scroll.documentVisibleRect.minY, 500, "the reader's own scroll wins")
    }

    func testARevealCancelsTheRestore() async throws {
        let follower = TranscriptFollower()
        let (window, scroll) = try await hostScrollView(follower: follower)
        defer { withExtendedLifetime(window) {} }
        follower.cancelRestore()
        await follower.restore(offset: 500)
        XCTAssertNotEqual(scroll.documentVisibleRect.minY, 500)
    }
}
