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
            // The unsloth-style alternate is derived from the same key as
            // the menu command it mirrors.
            XCTAssertEqual(row?.altKeys, "\u{2303} \(section.shortcutKey)")
        }
    }

    func testTheShortcutsPaneHasItsOwnEntryRow() {
        let rows = KeyboardShortcutCatalog.navigation.rows
        XCTAssertEqual(rows.first(where: { $0.label == "Keyboard Shortcuts" })?.keys, "\u{2318} /")
    }

    func testUnslothAlternateChordsAreListedOnTheirRows() {
        let navigationRows = KeyboardShortcutCatalog.navigation.rows
        XCTAssertEqual(
            navigationRows.first(where: { $0.label == "Toggle Chat Sidebar" })?.altKeys,
            "\u{2318} B")
        let chatRows = KeyboardShortcutCatalog.chat.rows
        XCTAssertEqual(
            chatRows.first(where: { $0.label == "New Chat" })?.altKeys, "\u{21E7} \u{2318} O")
        XCTAssertEqual(
            chatRows.first(where: { $0.label == "Previous Chat" })?.altKeys,
            "\u{21E7} \u{2318} [")
        XCTAssertEqual(
            chatRows.first(where: { $0.label == "Next Chat" })?.altKeys, "\u{21E7} \u{2318} ]")
    }

    func testChatSectionListsTheTemporaryChatAndAttachShortcuts() {
        let labels = KeyboardShortcutCatalog.chat.rows.map(\.label)
        XCTAssertTrue(labels.contains("New Temporary Chat"))
        XCTAssertTrue(labels.contains("Add Files or Photos"))
        XCTAssertTrue(labels.contains("Clear Chat History"))
        XCTAssertFalse(labels.contains("Clear History"), "the menu item is Clear Chat History")
    }

    func testChatSectionListsTheSearchChatsShortcut() {
        let rows = KeyboardShortcutCatalog.chat.rows
        XCTAssertEqual(rows.first(where: { $0.label == "Search Chats" })?.keys, "\u{2318} K")
        // The sibling it must not collide with: Cmd+Shift+K clears history.
        XCTAssertEqual(
            rows.first(where: { $0.label == "Clear Chat History" })?.keys, "\u{21E7} \u{2318} K")
    }

    func testRowLabelsAreUniqueAcrossSections() {
        let labels = KeyboardShortcutCatalog.sections.flatMap { $0.rows.map(\.label) }
        XCTAssertEqual(labels.count, Set(labels).count)
    }
}
