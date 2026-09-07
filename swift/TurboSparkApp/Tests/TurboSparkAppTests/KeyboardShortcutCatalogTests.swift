import XCTest
@testable import TurboSparkApp

/// The Keyboard Shortcuts pane is a value now, so it can be held against the
/// app. Every case here reddened under its own mutation and no other.
final class KeyboardShortcutCatalogTests: XCTestCase {
    func testEveryRailSectionHasANavigationRowWithItsOwnKey() {
        let rows = KeyboardShortcutCatalog.navigation.rows
        for section in AppModel.AppNavigationSection.allCases {
            let row = rows.first(where: { $0.label == section.title })
            XCTAssertNotNil(row, "\(section.title) is in the rail and missing from the pane")
            XCTAssertEqual(row?.keys, "\u{2318} \(section.shortcutKey)")
        }
    }

    func testChatSectionListsTheTemporaryChatAndAttachShortcuts() {
        let labels = KeyboardShortcutCatalog.chat.rows.map(\.label)
        XCTAssertTrue(labels.contains("New Temporary Chat"))
        XCTAssertTrue(labels.contains("Add Files or Photos"))
        XCTAssertTrue(labels.contains("Clear Chat History"))
        XCTAssertFalse(labels.contains("Clear History"), "the menu item is Clear Chat History")
    }

    func testRowLabelsAreUniqueAcrossSections() {
        let labels = KeyboardShortcutCatalog.sections.flatMap { $0.rows.map(\.label) }
        XCTAssertEqual(labels.count, Set(labels).count)
    }
}
