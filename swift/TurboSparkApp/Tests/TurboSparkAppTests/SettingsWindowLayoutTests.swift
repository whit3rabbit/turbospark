import XCTest
@testable import TurboSparkApp

/// The Settings window takes a fixed size from the visible frame (the scene
/// ignored an ideal size and opened at the floor). These pin the clamps: a
/// large display gets a roomy window, a huge one is capped, and a 1280x800
/// display still fits the window and its title bar.
@MainActor
final class SettingsWindowLayoutTests: XCTestCase {
    /// Title bar height above the content frame on a standard window.
    private let titleBar: CGFloat = 28

    func testScalesWithA16InchVisibleFrame() {
        let size = AppSettingsView.windowSize(
            forVisibleFrame: CGRect(x: 0, y: 106, width: 1728, height: 978))
        XCTAssertEqual(size.width, 1728 * 0.76, accuracy: 0.01)
        XCTAssertEqual(size.height, 978 * 0.86, accuracy: 0.01)
        // The old floor is what the window used to open at on this display.
        XCTAssertGreaterThan(size.width, 940 + 300)
        XCTAssertGreaterThan(size.height, 660 + 150)
    }

    func testSmallDisplayClampsToFloorAndStillFits() {
        // 1280x800 with the menu bar and a bottom Dock.
        let visible = CGRect(x: 0, y: 70, width: 1280, height: 705)
        let size = AppSettingsView.windowSize(forVisibleFrame: visible)
        XCTAssertEqual(size.height, 660)
        XCTAssertGreaterThanOrEqual(size.width, 940)
        XCTAssertLessThanOrEqual(size.width, visible.width)
        XCTAssertLessThanOrEqual(size.height + titleBar, visible.height)
    }

    func testLargeDisplayIsCapped() {
        let size = AppSettingsView.windowSize(
            forVisibleFrame: CGRect(x: 0, y: 0, width: 2560, height: 1385))
        XCTAssertEqual(size.width, 1480)
        XCTAssertEqual(size.height, 1000)
    }

    func testNoScreenFallsBackToFixedSize() {
        XCTAssertEqual(
            AppSettingsView.windowSize(forVisibleFrame: nil),
            CGSize(width: 1180, height: 800))
    }
}
