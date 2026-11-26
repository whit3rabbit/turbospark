import AppKit
import SwiftUI

/// Pure rules for "follow the stream unless the reader scrolled away".
///
/// Kept free of AppKit so the decision is unit-testable without a window.
enum TranscriptFollowRules {
    /// How close to the end still counts as "at the bottom". Wide enough that
    /// a rubber-band bounce or a half-line of slack does not unpin a reader
    /// who is effectively following along.
    static let bottomThreshold: CGFloat = 32

    static func distanceFromBottom(contentHeight: CGFloat, visibleMaxY: CGFloat) -> CGFloat {
        max(0, contentHeight - visibleMaxY)
    }

    /// Reaching the bottom always pins. Leaving it unpins only when the move
    /// came from the reader: a programmatic scroll into a lazily sized stack
    /// can land short for a frame and must not read as "scrolled away".
    static func isPinned(current: Bool, distanceFromBottom: CGFloat, userDriven: Bool) -> Bool {
        if distanceFromBottom <= bottomThreshold { return true }
        return userDriven ? false : current
    }

    /// The event form of `isUserScroll(type:keyCode:)`. `NSEvent.keyCode`
    /// raises an Objective-C exception on anything but a key event, and the
    /// current event during a layout pass can be a system-defined one (a media
    /// key, say), so the code is read only for `.keyDown`.
    static func isUserScroll(event: NSEvent?) -> Bool {
        guard let event else { return false }
        return isUserScroll(type: event.type, keyCode: event.type == .keyDown ? event.keyCode : nil)
    }

    /// Whether an event is the reader moving the viewport. Only keys that
    /// scroll count: Return from the composer scrolls the transcript too, and
    /// that is the app, not the reader.
    static func isUserScroll(type: NSEvent.EventType?, keyCode: UInt16?) -> Bool {
        switch type {
        case .scrollWheel, .swipe, .leftMouseDragged:
            return true
        case .keyDown:
            // Home, End, Page Up, Page Down, and the arrows.
            guard let keyCode else { return false }
            return [115, 119, 116, 121, 123, 124, 125, 126].contains(keyCode)
        default:
            return false
        }
    }
}

/// Tracks whether a transcript is pinned to its bottom. A pinned transcript
/// follows streamed output; one the reader scrolled up in stays where it is.
@MainActor
final class TranscriptFollower: ObservableObject {
    @Published private(set) var isPinned = true
    /// Last viewport top reported by the probe. Not published: it only feeds
    /// the OC-04 scroll memory.
    private(set) var lastOffset: CGFloat?
    /// Set by the probe: scrolls the viewport to an offset without animation
    /// and returns the offset it actually reached (nil: no scroll view).
    var applyOffset: ((CGFloat) -> CGFloat?)?
    /// True once the initial position (restored or latest) is settled. A
    /// transcript torn down before that has nothing worth remembering.
    private(set) var isSettled = false
    private var restoreCancelled = false

    func pin() {
        if !isPinned { isPinned = true }
    }

    func unpin() {
        if isPinned { isPinned = false }
    }

    /// Ends restoring: a reader's own scroll or a reveal request wins.
    func cancelRestore() { restoreCancelled = true }

    func markSettled() { isSettled = true }

    /// The position to remember for this transcript, nil if it never settled.
    var memorySnapshot: (pinned: Bool, offset: CGFloat?)? {
        isSettled ? (isPinned, lastOffset) : nil
    }

    /// Restores a remembered offset with no animation (Reduce Motion safe).
    /// Lazy content may not be tall enough on the first pass, so it retries a
    /// few times, and gives up the moment the reader scrolls or a reveal wins.
    func restore(offset target: CGFloat, attempts: Int = 40) async {
        unpin()
        for _ in 0..<attempts {
            await Task.yield()
            try? await Task.sleep(for: .milliseconds(25))
            if restoreCancelled { break }
            guard let reached = applyOffset?(target) else { continue }
            if abs(reached - target) < 1 { break }
        }
        markSettled()
    }

    func viewportMoved(distanceFromBottom: CGFloat, userDriven: Bool, offset: CGFloat? = nil) {
        if let offset { lastOffset = offset }
        if userDriven { restoreCancelled = true }
        let next = TranscriptFollowRules.isPinned(
            current: isPinned, distanceFromBottom: distanceFromBottom, userDriven: userDriven)
        if next != isPinned { isPinned = next }
    }
}

/// A zero-size probe that finds its enclosing `NSScrollView` and reports each
/// viewport move to a `TranscriptFollower`. SwiftUI's `onScrollGeometryChange`
/// needs macOS 15 and the floor is 14 (`DESIGN.md` section 8), so this reads
/// the scroll view SwiftUI already builds. Attach it as a background of the
/// scroll view's content, not of a lazy row that can be unloaded.
struct TranscriptScrollObserver: NSViewRepresentable {
    let follower: TranscriptFollower

    func makeNSView(context: Context) -> TranscriptScrollProbeView {
        TranscriptScrollProbeView(follower: follower)
    }

    func updateNSView(_ view: TranscriptScrollProbeView, context: Context) {
        view.follower = follower
    }
}

@MainActor
final class TranscriptScrollProbeView: NSView {
    var follower: TranscriptFollower
    private weak var observedScrollView: NSScrollView?
    private var boundsToken: NSObjectProtocol?

    init(follower: TranscriptFollower) {
        self.follower = follower
        super.init(frame: .zero)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }

    /// The scroll view this probe is watching, for tests.
    var attachedScrollView: NSScrollView? { observedScrollView }

    override func viewDidMoveToWindow() {
        super.viewDidMoveToWindow()
        detach()
        guard window != nil, let scrollView = enclosingScrollView else { return }
        observedScrollView = scrollView
        follower.applyOffset = { [weak self] target in self?.scroll(toOffset: target) }
        let clip = scrollView.contentView
        clip.postsBoundsChangedNotifications = true
        boundsToken = NotificationCenter.default.addObserver(
            forName: NSView.boundsDidChangeNotification, object: clip, queue: .main
        ) { [weak self] _ in
            MainActor.assumeIsolated { self?.report() }
        }
    }

    /// Jumps the viewport to `target`, clamped to what the content allows.
    /// Returns the offset reached.
    private func scroll(toOffset target: CGFloat) -> CGFloat? {
        guard let scrollView = observedScrollView, let document = scrollView.documentView else { return nil }
        let clip = scrollView.contentView
        let y = TranscriptFollowRules.reachableOffset(
            target: target, contentHeight: document.bounds.height, viewportHeight: clip.bounds.height)
        clip.scroll(to: NSPoint(x: clip.bounds.origin.x, y: y))
        scrollView.reflectScrolledClipView(clip)
        return y
    }

    private func detach() {
        if let boundsToken { NotificationCenter.default.removeObserver(boundsToken) }
        boundsToken = nil
        observedScrollView = nil
    }

    private func report() {
        guard let scrollView = observedScrollView, let document = scrollView.documentView else { return }
        follower.viewportMoved(
            distanceFromBottom: TranscriptFollowRules.distanceFromBottom(
                contentHeight: document.bounds.height,
                visibleMaxY: scrollView.documentVisibleRect.maxY),
            userDriven: TranscriptFollowRules.isUserScroll(event: NSApp.currentEvent),
            offset: scrollView.documentVisibleRect.minY)
    }
}

/// The floating "back to the newest output" control shown while the reader is
/// scrolled away from the bottom.
struct JumpToLatestButton: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 5) {
                Image(systemName: "arrow.down")
                    .themedFont(.tiny, weight: .medium)
                    .accessibilityHidden(true)
                Text("Jump to latest", bundle: .module)
                    .themedFont(.small, weight: .medium)
            }
            .lineLimit(1)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .foregroundStyle(.appText)
            .appGlassSurface(.capsule, interactive: true)
            .contentShape(Capsule(style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.96))
        .fixedSize()
        .help(Text("Jump to latest", bundle: .module))
        .accessibilityLabel(Text("Jump to latest", bundle: .module))
    }
}
