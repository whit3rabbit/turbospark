import Foundation

/// The shortcuts the Keyboard Shortcuts pane lists, as a value.
///
/// A value rather than view code so a test can hold it against the app: the
/// pane used to be a hand-typed list that omitted New Temporary Chat and
/// Add Files, misnamed Clear Chat History, and could not notice that the rail
/// promised a Server shortcut nobody had bound. The navigation rows are
/// DERIVED from `AppNavigationSection`, so a new section appears here the way it
/// appears in the rail (swift/CLAUDE.md Gotcha 17).
///
/// There is no rebinding store in this app. These are fixed, and the pane
/// says so.
public struct KeyboardShortcutRow: Equatable, Identifiable, Sendable {
    public let label: String
    /// Display form, e.g. "\u{2318} 1".
    public let keys: String
    /// A second fixed chord for the same action, or nil. The menus show only
    /// `keys`; the alternates are window-level hidden buttons mounted by
    /// `AlternateShortcutBridge` (swift/CLAUDE.md Gotcha 58).
    public let altKeys: String?
    public var id: String { label }

    public init(label: String, keys: String, altKeys: String? = nil) {
        self.label = label
        self.keys = keys
        self.altKeys = altKeys
    }
}

public struct KeyboardShortcutSection: Equatable, Identifiable, Sendable {
    public let title: String
    public let rows: [KeyboardShortcutRow]
    public var id: String { title }
}

public enum KeyboardShortcutCatalog {
    static let command = "\u{2318}"
    static let shift = "\u{21E7}"
    static let control = "\u{2303}"
    static let returnKey = "\u{21A9}"

    /// One row per `AppSection`, in rail order, plus the two pane toggles and
    /// the Shortcuts pane's own Cmd+/ entry point.
    public static var navigation: KeyboardShortcutSection {
        var rows = AppModel.AppNavigationSection.allCases.map { section in
            KeyboardShortcutRow(
                label: section.title,
                keys: "\(command) \(section.shortcutKey)",
                altKeys: "\(control) \(section.shortcutKey)")
        }
        rows.append(KeyboardShortcutRow(
            label: "Toggle Chat Sidebar",
            keys: "\(control) \(command) S",
            altKeys: "\(command) B"))
        rows.append(KeyboardShortcutRow(label: "Toggle Inspector", keys: "\(shift) \(command) I"))
        rows.append(KeyboardShortcutRow(label: "Keyboard Shortcuts", keys: "\(command) /"))
        return KeyboardShortcutSection(title: "Navigation & View", rows: rows)
    }

    public static let textSize = KeyboardShortcutSection(
        title: "Text Size & Zoom",
        rows: [
            KeyboardShortcutRow(label: "Make Text Bigger", keys: "\(command) +"),
            KeyboardShortcutRow(label: "Make Text Smaller", keys: "\(command) -"),
            KeyboardShortcutRow(label: "Default Text Size", keys: "\(command) 0"),
        ])

    public static let chat = KeyboardShortcutSection(
        title: "Chat & Generation",
        rows: [
            KeyboardShortcutRow(
                label: "New Chat", keys: "\(command) N", altKeys: "\(shift) \(command) O"),
            KeyboardShortcutRow(label: "New Temporary Chat", keys: "\(shift) \(command) N"),
            KeyboardShortcutRow(
                label: "Previous Chat",
                keys: "\(command) [",
                altKeys: "\(shift) \(command) ["),
            KeyboardShortcutRow(
                label: "Next Chat",
                keys: "\(command) ]",
                altKeys: "\(shift) \(command) ]"),
            KeyboardShortcutRow(label: "Focus Prompt", keys: "\(command) L"),
            KeyboardShortcutRow(label: "Search Chats", keys: "\(command) K"),
            KeyboardShortcutRow(label: "Add Files or Photos", keys: "\(command) U"),
            KeyboardShortcutRow(label: "Clear Chat History", keys: "\(shift) \(command) K"),
            KeyboardShortcutRow(label: "Generate Response", keys: "\(command) \(returnKey)"),
            KeyboardShortcutRow(label: "Cancel Generation", keys: "\(command) ."),
        ])

    public static var sections: [KeyboardShortcutSection] {
        [navigation, textSize, chat]
    }
}
