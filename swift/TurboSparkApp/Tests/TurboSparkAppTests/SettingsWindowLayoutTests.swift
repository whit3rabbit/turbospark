import XCTest
@testable import TurboSparkApp

/// The Settings window opens at the main window's own rule: the whole
/// visible frame (`AppWindowDefaultSize`), so Settings is about the same
/// size as the app, and it stays resizable down to `minimumSize`. These pin
/// the rule, the no-screen fallback, and the floor: a 1280x800 display still
/// fits the minimum and its title bar, and the minimum never exceeds the
/// default it shrinks from.
@MainActor
final class SettingsWindowLayoutTests: XCTestCase {
    /// Title bar height above the content frame on a standard window.
    private let titleBar: CGFloat = 28

    func testOpensAtTheWholeVisibleFrame() {
        // A 16-inch panel's visible frame.
        let visible = CGRect(x: 0, y: 106, width: 1728, height: 978)
        XCTAssertEqual(
            AppWindowDefaultSize.size(forVisibleFrame: visible),
            CGSize(width: 1728, height: 978))
    }

    func testSmallDisplayKeepsTheVisibleSizeAndFitsTheMinimum() {
        // 1280x800 with the menu bar and a bottom Dock.
        let visible = CGRect(x: 0, y: 70, width: 1280, height: 705)
        XCTAssertEqual(
            AppWindowDefaultSize.size(forVisibleFrame: visible),
            CGSize(width: 1280, height: 705))
        let minimum = AppSettingsView.minimumSize
        XCTAssertLessThanOrEqual(minimum.width, visible.width)
        XCTAssertLessThanOrEqual(minimum.height + titleBar, visible.height)
    }

    func testMinimumIsBelowEveryDefault() {
        let minimum = AppSettingsView.minimumSize
        let frames: [CGRect?] = [
            CGRect(x: 0, y: 70, width: 1280, height: 705),
            CGRect(x: 0, y: 106, width: 1728, height: 978),
            CGRect(x: 0, y: 0, width: 2560, height: 1385),
            nil,
        ]
        for visible in frames {
            let size = AppWindowDefaultSize.size(forVisibleFrame: visible)
            XCTAssertLessThanOrEqual(minimum.width, size.width)
            XCTAssertLessThanOrEqual(minimum.height, size.height)
        }
    }

    func testSearchMatchesSourceTitleAndKeywords() {
        XCTAssertTrue(AppSettingsView.matches("Keyboard Shortcuts", query: ""))
        XCTAssertTrue(AppSettingsView.matches("Keyboard Shortcuts", query: "shortcut"))
        XCTAssertTrue(AppSettingsView.matches("Engine", keywords: ["sampling"], query: "Sampl"))
        XCTAssertFalse(AppSettingsView.matches("Engine", keywords: ["sampling"], query: "zzz"))
    }

    /// A stray space in the search field must not filter the sidebar down to
    /// the tabs whose titles happen to contain one.
    func testWhitespaceQueryMatchesEveryTab() {
        XCTAssertTrue(AppSettingsView.matches("Appearance", query: " "))
        XCTAssertTrue(AppSettingsView.matches("General", query: "\t "))
        XCTAssertTrue(AppSettingsView.matches("Engine", keywords: ["sampling"], query: " sampl "))
    }

    func testNoScreenFallsBackToFixedSize() {
        XCTAssertEqual(
            AppWindowDefaultSize.size(forVisibleFrame: nil),
            CGSize(width: 1280, height: 760))
        XCTAssertEqual(AppWindowDefaultSize.size(forVisibleFrame: nil), AppWindowDefaultSize.fallback)
    }
}
