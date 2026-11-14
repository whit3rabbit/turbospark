import XCTest

@testable import TurboSparkApp

/// A scheduler whose time only moves when the test says so.
@MainActor
private final class FakeScheduler: SidebarPeekScheduler {
    private(set) var pending: [Int: (delay: TimeInterval, work: @MainActor () -> Void)] = [:]
    private var next = 0

    func schedule(after delay: TimeInterval, _ work: @escaping @MainActor () -> Void) -> () -> Void {
        let id = next
        next += 1
        pending[id] = (delay, work)
        return { [weak self] in self?.pending[id] = nil }
    }

    var pendingCount: Int { pending.count }

    /// Fires every pending timer once.
    func fireAll() {
        let due = pending
        pending = [:]
        for (_, entry) in due { entry.work() }
    }
}

@MainActor
final class SidebarPeekControllerTests: XCTestCase {
    private var scheduler = FakeScheduler()
    private var suppression = SidebarPeekSuppression()

    private func makeController(eligible: Bool = true) -> SidebarPeekController {
        scheduler = FakeScheduler()
        suppression = SidebarPeekSuppression()
        let controller = SidebarPeekController(scheduler: scheduler, suppression: { [unowned self] in
            self.suppression
        })
        controller.setEligible(eligible)
        return controller
    }

    private func peek(_ c: SidebarPeekController) {
        c.pointerEntered()
        scheduler.fireAll()
    }

    func testDwellOpensPeekOnlyAfterTimerFires() {
        let c = makeController()
        c.pointerEntered()
        XCTAssertEqual(c.phase, .dwelling)
        XCTAssertFalse(c.isPeeking)
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .peeking)
    }

    func testLeavingStripBeforeDwellCancelsIt() {
        let c = makeController()
        c.pointerEntered()
        c.pointerExited()
        XCTAssertEqual(c.phase, .idle)
        XCTAssertEqual(scheduler.pendingCount, 0)
    }

    func testNoPeekWhenDocked() {
        let c = makeController(eligible: false)
        c.pointerEntered()
        XCTAssertEqual(c.phase, .idle)
        XCTAssertEqual(scheduler.pendingCount, 0)
    }

    func testDockingWhilePeekingCancelsPeek() {
        let c = makeController()
        peek(c)
        c.setEligible(false)
        XCTAssertEqual(c.phase, .idle)
    }

    func testCorridorExitDocksAfterGrace() {
        let c = makeController()
        peek(c)
        c.pointerExited()
        XCTAssertEqual(c.phase, .leaving)
        XCTAssertTrue(c.isPeeking, "the overlay stays up through the grace period")
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .idle)
    }

    func testReenteringDuringGraceKeepsPeek() {
        let c = makeController()
        peek(c)
        c.pointerExited()
        c.pointerEntered()
        XCTAssertEqual(c.phase, .peeking)
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .peeking)
    }

    func testStripToPanelHandoffInEitherOrderDoesNotLeave() {
        // Panel region entered before the strip region exits.
        let a = makeController()
        a.pointerEntered()  // strip
        scheduler.fireAll()
        a.pointerEntered()  // panel
        a.pointerExited()  // strip
        XCTAssertEqual(a.phase, .peeking)
        // Strip exits before the panel is entered: grace absorbs it.
        let b = makeController()
        b.pointerEntered()
        scheduler.fireAll()
        b.pointerExited()
        b.pointerEntered()
        scheduler.fireAll()
        XCTAssertEqual(b.phase, .peeking)
    }

    func testEachSuppressionBlocksStartingAPeek() {
        let cases: [(String, (inout SidebarPeekSuppression) -> Void)] = [
            ("drag", { $0.pointerButtonDown = true }),
            ("sheet", { $0.sheetOrDialogOpen = true }),
            ("palette", { $0.paletteOpen = true }),
            ("inactive", { $0.appInactive = true }),
            ("menu", { $0.menuTracking = true }),
        ]
        for (name, apply) in cases {
            let c = makeController()
            apply(&suppression)
            c.pointerEntered()
            XCTAssertEqual(c.phase, .idle, "\(name) must block the dwell from starting")
        }
    }

    func testSuppressionStartingDuringDwellBlocksAtFireTime() {
        let c = makeController()
        c.pointerEntered()
        suppression.pointerButtonDown = true  // a drag began mid-dwell
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .idle)
    }

    func testButtonDownHoldsAnOpenPeekPastTheGrace() {
        let c = makeController()
        peek(c)
        c.pointerExited()
        suppression.pointerButtonDown = true
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .leaving, "an in-flight press keeps the peek open")
        suppression.pointerButtonDown = false
        scheduler.fireAll()
        XCTAssertEqual(c.phase, .idle)
    }

    func testCancelDropsPeekAndRegionCount() {
        let c = makeController()
        peek(c)
        c.cancel()
        XCTAssertEqual(c.phase, .idle)
        // A stale region count must not stop the next dwell from starting
        // or the next exit from docking.
        peek(c)
        c.pointerExited()
        XCTAssertEqual(c.phase, .leaving)
    }

    func testLiveSuppressionReadsPaletteFlag() {
        let c = SidebarPeekController(scheduler: FakeScheduler())
        c.setEligible(true)
        c.paletteOpen = true
        c.pointerEntered()
        XCTAssertEqual(c.phase, .idle)
    }
}
