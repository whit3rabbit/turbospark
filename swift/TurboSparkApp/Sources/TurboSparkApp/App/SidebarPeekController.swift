import AppKit
import Combine
import Foundation

/// What blocks (or holds open) a sidebar peek, read at decision time.
///
/// Pulled from the window and AppKit when the dwell timer fires rather than
/// pushed in by `RootView`, because a tracking region receives no
/// `mouseEntered` during a drag, so there is no event to push on.
struct SidebarPeekSuppression: Equatable {
    /// A mouse button is down: a drag, a text selection started by mouse, or
    /// a press in progress. Also the hold-open signal for an open peek, so a
    /// click or drag that leaves the panel does not dismiss it mid-gesture.
    var pointerButtonDown = false
    /// A sheet or modal dialog is up on the key window.
    var sheetOrDialogOpen = false
    /// The command center is showing over the content.
    var paletteOpen = false
    /// The app is not frontmost, so a hover must not move windows around.
    var appInactive = false
    /// A context menu or other event-tracking run loop is active. Holds an
    /// open peek so a row's context menu is not torn down under the user.
    var menuTracking = false

    /// Blocks starting a peek.
    var blocksPeek: Bool {
        pointerButtonDown || sheetOrDialogOpen || paletteOpen || appInactive || menuTracking
    }

    /// Delays dismissing an open peek.
    var holdsOpen: Bool { pointerButtonDown || menuTracking }

    /// The live reading. `paletteOpen` is RootView state, so it is passed in.
    @MainActor
    static func current(paletteOpen: Bool) -> SidebarPeekSuppression {
        // NSApp is nil before the app object exists (headless tests): read
        // it as an optional, treating that as "nothing is open".
        let app: NSApplication? = NSApp
        return SidebarPeekSuppression(
            pointerButtonDown: NSEvent.pressedMouseButtons != 0,
            sheetOrDialogOpen: app?.keyWindow?.attachedSheet != nil || app?.modalWindow != nil,
            paletteOpen: paletteOpen,
            appInactive: app.map { !$0.isActive } ?? false,
            menuTracking: RunLoop.main.currentMode == .eventTracking)
    }
}

/// Schedules one cancellable delayed action. Injected so tests drive time.
@MainActor
protocol SidebarPeekScheduler {
    /// Runs `work` after `delay` seconds; the returned closure cancels it.
    func schedule(after delay: TimeInterval, _ work: @escaping @MainActor () -> Void) -> () -> Void
}

@MainActor
struct LiveSidebarPeekScheduler: SidebarPeekScheduler {
    func schedule(after delay: TimeInterval, _ work: @escaping @MainActor () -> Void) -> () -> Void {
        // `.common` modes so the exit timer still fires while a menu tracks.
        let timer = Timer(timeInterval: delay, repeats: false) { _ in
            MainActor.assumeIsolated { work() }
        }
        RunLoop.main.add(timer, forMode: .common)
        return { timer.invalidate() }
    }
}

/// OC-06: the dwell, corridor and suppression state machine behind the
/// collapsed sidebar's hover peek.
///
/// A peek is a transient overlay of the panel over the content. It never
/// touches the docked-state preference. Idle, dwelling (pointer on the edge
/// strip, timer running), peeking (overlay shown), leaving (pointer left the
/// corridor, grace timer running).
///
/// The corridor is the edge strip plus the overlay's frame and a grace band,
/// all feeding `pointerEntered` and `pointerExited`. A count of regions the
/// pointer is inside, not a flag, so the strip-to-panel handoff (exit and
/// enter in either order) cannot read as leaving.
@MainActor
final class SidebarPeekController: ObservableObject {
    /// Initial values, tunable. OpenCode uses 40 ms on its edge and 500 ms on
    /// its toggle; the toggle path is excluded here, so only the edge applies,
    /// and 350 ms is the draft's comfortable middle. Set in review.
    static let dwellDelay: TimeInterval = 0.35
    /// Absorbs boundary flicker and a quick overshoot before docking back.
    static let exitGrace: TimeInterval = 0.2
    /// Width of the leading-edge trigger strip, inside the rail's trailing edge.
    static let triggerStripWidth: CGFloat = 8
    /// Extra hover band past the panel's trailing edge.
    static let corridorGrace: CGFloat = 16

    enum Phase: Equatable { case idle, dwelling, peeking, leaving }

    @Published private(set) var phase: Phase = .idle
    var isPeeking: Bool { phase == .peeking || phase == .leaving }

    /// False when the sidebar is docked or the section has no panel.
    private(set) var isEligible = false

    private let scheduler: SidebarPeekScheduler
    private let suppressionOverride: (() -> SidebarPeekSuppression)?
    /// The command center's visibility, fed by RootView (no AppKit source).
    var paletteOpen = false
    private var cancelTimer: (() -> Void)?
    private var regionsInside = 0

    init(
        scheduler: SidebarPeekScheduler? = nil,
        suppression: (() -> SidebarPeekSuppression)? = nil
    ) {
        self.scheduler = scheduler ?? LiveSidebarPeekScheduler()
        self.suppressionOverride = suppression
    }

    private func suppression() -> SidebarPeekSuppression {
        suppressionOverride?() ?? .current(paletteOpen: paletteOpen)
    }

    /// Eligible only while the panel is collapsed and the section has one.
    func setEligible(_ eligible: Bool) {
        guard eligible != isEligible else { return }
        isEligible = eligible
        if !eligible { cancel() }
    }

    func pointerEntered() {
        regionsInside += 1
        switch phase {
        case .idle:
            guard isEligible, !suppression().blocksPeek else { return }
            phase = .dwelling
            restartTimer(after: Self.dwellDelay) { [weak self] in self?.dwellFired() }
        case .leaving:
            stopTimer()
            phase = .peeking
        case .dwelling, .peeking:
            break
        }
    }

    func pointerExited() {
        regionsInside = max(0, regionsInside - 1)
        guard regionsInside == 0 else { return }
        switch phase {
        case .dwelling:
            stopTimer()
            phase = .idle
        case .peeking:
            phase = .leaving
            restartTimer(after: Self.exitGrace) { [weak self] in self?.leaveFired() }
        case .idle, .leaving:
            break
        }
    }

    /// Drops any peek at once: a toggle click, a chat or section switch,
    /// window deactivation, Escape. The caller then applies its own action.
    func cancel() {
        stopTimer()
        regionsInside = 0
        if phase != .idle { phase = .idle }
    }

    private func dwellFired() {
        cancelTimer = nil
        guard phase == .dwelling else { return }
        // Re-read at fire time: a sheet or a press may have started during
        // the dwell, with no event to say so.
        if isEligible, !suppression().blocksPeek {
            phase = .peeking
        } else {
            phase = .idle
        }
    }

    private func leaveFired() {
        cancelTimer = nil
        guard phase == .leaving else { return }
        if suppression().holdsOpen {
            restartTimer(after: Self.exitGrace) { [weak self] in self?.leaveFired() }
        } else {
            phase = .idle
        }
    }

    private func restartTimer(after delay: TimeInterval, _ work: @escaping @MainActor () -> Void) {
        stopTimer()
        cancelTimer = scheduler.schedule(after: delay, work)
    }

    private func stopTimer() {
        cancelTimer?()
        cancelTimer = nil
    }
}
