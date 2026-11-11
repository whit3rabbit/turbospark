import AppKit
import SwiftUI
import XCTest

@testable import TurboSparkApp

final class TranscriptFollowRulesTests: XCTestCase {
    func testDistanceIsClampedAtZeroForOverscroll() {
        XCTAssertEqual(TranscriptFollowRules.distanceFromBottom(contentHeight: 1000, visibleMaxY: 1000), 0)
        XCTAssertEqual(TranscriptFollowRules.distanceFromBottom(contentHeight: 1000, visibleMaxY: 1040), 0)
        XCTAssertEqual(TranscriptFollowRules.distanceFromBottom(contentHeight: 1000, visibleMaxY: 400), 600)
    }

    func testReachingTheBottomPinsWhoeverMovedIt() {
        for userDriven in [true, false] {
            XCTAssertTrue(TranscriptFollowRules.isPinned(current: false, distanceFromBottom: 0, userDriven: userDriven))
            XCTAssertTrue(TranscriptFollowRules.isPinned(
                current: false, distanceFromBottom: TranscriptFollowRules.bottomThreshold, userDriven: userDriven))
        }
    }

    func testOnlyTheReaderCanLeaveTheBottom() {
        let far = TranscriptFollowRules.bottomThreshold + 1
        XCTAssertFalse(TranscriptFollowRules.isPinned(current: true, distanceFromBottom: far, userDriven: true))
        // A scroll the app issued that landed short keeps following.
        XCTAssertTrue(TranscriptFollowRules.isPinned(current: true, distanceFromBottom: far, userDriven: false))
        // And an unpinned reader stays unpinned when the app moves the view.
        XCTAssertFalse(TranscriptFollowRules.isPinned(current: false, distanceFromBottom: far, userDriven: false))
    }

    func testWhichEventsCountAsTheReaderScrolling() {
        XCTAssertTrue(TranscriptFollowRules.isUserScroll(type: .scrollWheel, keyCode: nil))
        XCTAssertTrue(TranscriptFollowRules.isUserScroll(type: .leftMouseDragged, keyCode: nil))
        XCTAssertTrue(TranscriptFollowRules.isUserScroll(type: .keyDown, keyCode: 116), "Page Up")
        XCTAssertTrue(TranscriptFollowRules.isUserScroll(type: .keyDown, keyCode: 115), "Home")
        // Return sends from the composer and the app scrolls in response.
        XCTAssertFalse(TranscriptFollowRules.isUserScroll(type: .keyDown, keyCode: 36))
        XCTAssertFalse(TranscriptFollowRules.isUserScroll(type: nil, keyCode: nil))
    }

    /// A live run crashed here: `keyCode` raises on a non-key event, and the
    /// current event during layout was a system-defined one.
    func testNonKeyEventsAreClassifiedWithoutReadingTheKeyCode() throws {
        let systemDefined = try XCTUnwrap(NSEvent.otherEvent(
            with: .systemDefined, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
            context: nil, subtype: 8, data1: 68352, data2: -1))
        XCTAssertFalse(TranscriptFollowRules.isUserScroll(event: systemDefined))
        XCTAssertFalse(TranscriptFollowRules.isUserScroll(event: nil))

        let pageUp = try XCTUnwrap(NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
            context: nil, characters: "", charactersIgnoringModifiers: "", isARepeat: false, keyCode: 116))
        XCTAssertTrue(TranscriptFollowRules.isUserScroll(event: pageUp))
        let returnKey = try XCTUnwrap(NSEvent.keyEvent(
            with: .keyDown, location: .zero, modifierFlags: [], timestamp: 0, windowNumber: 0,
            context: nil, characters: "\r", charactersIgnoringModifiers: "\r", isARepeat: false, keyCode: 36))
        XCTAssertFalse(TranscriptFollowRules.isUserScroll(event: returnKey))
    }
}

@MainActor
final class TranscriptFollowerTests: XCTestCase {
    func testStartsPinnedAndUnpinsOnlyOnAUserScrollAway() {
        let follower = TranscriptFollower()
        XCTAssertTrue(follower.isPinned)

        follower.viewportMoved(distanceFromBottom: 500, userDriven: false)
        XCTAssertTrue(follower.isPinned, "an app-driven move must not unpin")

        follower.viewportMoved(distanceFromBottom: 500, userDriven: true)
        XCTAssertFalse(follower.isPinned)

        follower.viewportMoved(distanceFromBottom: 500, userDriven: false)
        XCTAssertFalse(follower.isPinned, "streaming growth must not re-pin a reader who scrolled up")

        follower.viewportMoved(distanceFromBottom: 4, userDriven: true)
        XCTAssertTrue(follower.isPinned, "scrolling back to the bottom resumes following")
    }

    func testPinRestoresFollowing() {
        let follower = TranscriptFollower()
        follower.viewportMoved(distanceFromBottom: 900, userDriven: true)
        XCTAssertFalse(follower.isPinned)
        follower.pin()
        XCTAssertTrue(follower.isPinned)
    }

    /// The probe has to find the `NSScrollView` SwiftUI builds for a
    /// `ScrollView`, and report real geometry from it. This is the part a
    /// future SwiftUI release could break silently.
    func testProbeFindsTheScrollViewAndReportsRealGeometry() async throws {
        let follower = TranscriptFollower()
        let root = ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(0..<120, id: \.self) { index in
                    Text(verbatim: "Row \(index)").frame(height: 24)
                }
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

        let scrollView = try XCTUnwrap(Self.firstScrollView(in: host), "SwiftUI ScrollView is not an NSScrollView")
        let probe = try XCTUnwrap(Self.firstProbe(in: host))
        XCTAssertTrue(probe.attachedScrollView === scrollView)

        // Scrolled to the top of tall content: far from the bottom. A scroll
        // with no user event behind it (this test) must not unpin.
        let document = try XCTUnwrap(scrollView.documentView)
        XCTAssertGreaterThan(document.bounds.height, scrollView.documentVisibleRect.height * 2)
        scrollView.contentView.scroll(to: NSPoint(x: 0, y: 200))
        scrollView.reflectScrolledClipView(scrollView.contentView)
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertTrue(follower.isPinned)

        // Landing on the bottom reports a distance inside the threshold.
        let bottomY = document.bounds.height - scrollView.contentView.bounds.height
        scrollView.contentView.scroll(to: NSPoint(x: 0, y: bottomY))
        scrollView.reflectScrolledClipView(scrollView.contentView)
        try await Task.sleep(for: .milliseconds(50))
        XCTAssertLessThanOrEqual(
            TranscriptFollowRules.distanceFromBottom(
                contentHeight: document.bounds.height, visibleMaxY: scrollView.documentVisibleRect.maxY),
            TranscriptFollowRules.bottomThreshold)
        XCTAssertTrue(follower.isPinned)
    }

    private static func firstScrollView(in view: NSView) -> NSScrollView? {
        if let scroll = view as? NSScrollView { return scroll }
        for child in view.subviews {
            if let scroll = firstScrollView(in: child) { return scroll }
        }
        return nil
    }

    private static func firstProbe(in view: NSView) -> TranscriptScrollProbeView? {
        if let probe = view as? TranscriptScrollProbeView { return probe }
        for child in view.subviews {
            if let probe = firstProbe(in: child) { return probe }
        }
        return nil
    }
}
