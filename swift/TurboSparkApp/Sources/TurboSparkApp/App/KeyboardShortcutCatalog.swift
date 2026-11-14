import Foundation

/// A stable command id (OC-01): a plain string, so a later feature (OC-04's
/// tab chords) can add ids without touching this type. Ids are persisted in
/// the override store and never renamed; a label may change, an id may not.
public struct KeyboardCommandID: RawRepresentable, Hashable, Sendable, ExpressibleByStringLiteral {
    public let rawValue: String
    public init(rawValue: String) { self.rawValue = rawValue }
    public init(stringLiteral value: String) { self.rawValue = value }

    static func navigation(_ section: AppModel.AppNavigationSection) -> KeyboardCommandID {
        KeyboardCommandID(rawValue: "nav.\(section.rawValue)")
    }

    static let toggleSidebar: KeyboardCommandID = "view.toggleSidebar"
    static let goBack: KeyboardCommandID = "history.back"
    static let goForward: KeyboardCommandID = "history.forward"
    static let toggleInspector: KeyboardCommandID = "view.toggleInspector"
    static let keyboardShortcuts: KeyboardCommandID = "view.keyboardShortcuts"
    static let textBigger: KeyboardCommandID = "text.bigger"
    static let textSmaller: KeyboardCommandID = "text.smaller"
    static let textDefault: KeyboardCommandID = "text.default"
    static let newChat: KeyboardCommandID = "chat.new"
    static let newTemporaryChat: KeyboardCommandID = "chat.newTemporary"
    static let previousChat: KeyboardCommandID = "chat.previous"
    static let nextChat: KeyboardCommandID = "chat.next"
    static let nextWorkspaceTab: KeyboardCommandID = "workspace.nextTab"
    static let previousWorkspaceTab: KeyboardCommandID = "workspace.previousTab"
    static let newSessionTab: KeyboardCommandID = "session.newTab"
    static let closeSessionTab: KeyboardCommandID = "session.closeTab"
    static let reopenSessionTab: KeyboardCommandID = "session.reopenTab"
    static let nextSessionTab: KeyboardCommandID = "session.nextTab"
    static let previousSessionTab: KeyboardCommandID = "session.previousTab"
    static let focusPrompt: KeyboardCommandID = "prompt.focus"
    static let commandCenter: KeyboardCommandID = "palette.open"
    static let runCommand: KeyboardCommandID = "palette.runCommand"
    static let addFiles: KeyboardCommandID = "composer.addFiles"
    static let clearHistory: KeyboardCommandID = "chat.clearHistory"
    static let previousTurn: KeyboardCommandID = "chat.previousTurn"
    static let nextTurn: KeyboardCommandID = "chat.nextTurn"
    static let generate: KeyboardCommandID = "generation.generate"
    static let cancelGeneration: KeyboardCommandID = "generation.cancel"
    static let stopAll: KeyboardCommandID = "generation.stopAll"
    static let startRecording: KeyboardCommandID = "audio.startRecording"
}

/// One command in the inventory: a stable id, the catalog label, its default
/// chord, an optional hidden alternate, and whether the user may rebind it.
///
/// `rebindable` is true only for a command whose chord is carried by exactly
/// one menu item. The rest stay listed (so a conflict check can see them) but
/// fixed: Generate and Cancel share their chords with the Images, Audio and
/// Humanize run and stop buttons (`ownsGenerationShortcuts`), and Add Files
/// and Start Recording are bound on in-window buttons, not in a menu.
/// Rebinding the menu half would leave the buttons on the old chord.
public struct KeyboardCommand: Equatable, Identifiable, Sendable {
    public let id: KeyboardCommandID
    public let label: String
    public let defaultChord: KeyChord
    /// A second fixed chord for the same action, mounted by
    /// `AlternateShortcutBridge`. Never rebindable, always reserved.
    public let alternate: KeyChord?
    public let rebindable: Bool
}

/// The shortcuts the Keyboard Shortcuts pane lists, as a value.
///
/// A value rather than view code so a test can hold it against the app: the
/// pane used to be a hand-typed list that omitted New Temporary Chat and
/// Add Files, misnamed Clear Chat History, and could not notice that the rail
/// promised a Server shortcut nobody had bound. The navigation rows are
/// DERIVED from `AppNavigationSection`, so a new section appears here the way it
/// appears in the rail (swift/CLAUDE.md Gotcha 17).
///
/// This file owns ids and DEFAULTS only and knows nothing of the override
/// store (OC-01). `KeybindMap` (in `KeybindOverrideStore.swift`) layers the
/// user's overrides on top, and the menu, the bridge and the pane all read
/// the resolved chord through it.
///
/// Every `label` and section `title` is a string-catalog key spelled the way
/// the menu item spells it (an ellipsis-bearing menu title such as
/// "Command Center..." drops it here, since the row names the command rather
/// than the item that opens a panel). The pane renders them through the
/// catalog, so a new row needs its key translated.
public struct KeyboardShortcutRow: Equatable, Identifiable, Sendable {
    public let commandID: KeyboardCommandID
    public let label: String
    /// Display form of the RESOLVED chord, e.g. "\u{2318} 1".
    public let keys: String
    /// A second fixed chord for the same action, or nil. The menus show only
    /// `keys`; the alternates are window-level hidden buttons mounted by
    /// `AlternateShortcutBridge` (swift/CLAUDE.md Gotcha 58).
    public let altKeys: String?
    public let chord: KeyChord
    public let rebindable: Bool
    /// True when the resolved chord is a user override.
    public let isCustom: Bool
    public var id: String { commandID.rawValue }
}

public struct KeyboardShortcutSection: Equatable, Identifiable, Sendable {
    public let title: String
    public let rows: [KeyboardShortcutRow]
    public var id: String { title }
}

public enum KeyboardShortcutCatalog {
    private static func chord(
        _ key: String, _ modifiers: KeyChord.Modifiers
    ) -> KeyChord {
        // The literals below are all supported keys; a typo would be a
        // programmer error caught by `KeyboardShortcutCatalogTests`, which
        // lists every command.
        KeyChord(key: key, modifiers: modifiers)
            ?? KeyChord(key: "space", modifiers: modifiers)!
    }

    private static func command(
        _ id: KeyboardCommandID, _ label: String, _ key: String, _ modifiers: KeyChord.Modifiers,
        alt: KeyChord? = nil, rebindable: Bool = true
    ) -> KeyboardCommand {
        KeyboardCommand(
            id: id, label: label, defaultChord: chord(key, modifiers), alternate: alt,
            rebindable: rebindable)
    }

    private typealias Group = (title: String, commands: [KeyboardCommand])

    /// One group per section, in pane order.
    private static let groups: [Group] = makeGroups()

    private static func makeGroups() -> [Group] {
        let cmd: KeyChord.Modifiers = .command
        let shiftCmd: KeyChord.Modifiers = [.shift, .command]
        let ctrl: KeyChord.Modifiers = .control

        // One row per `AppSection`, in rail order, plus the pane toggles and
        // the Shortcuts pane's own Cmd+/ entry point.
        var navigation = AppModel.AppNavigationSection.allCases.map { section -> KeyboardCommand in
            let key = String(section.shortcutKey)
            return command(
                .navigation(section), section.title, key, cmd, alt: chord(key, ctrl))
        }
        navigation += [
            command(
                .toggleSidebar, "Toggle Sidebar", "s", [.control, .command],
                alt: chord("b", cmd)),
            // OC-05. Not Cmd+[ / Cmd+], which are Previous/Next Chat.
            command(.goBack, "Go Back", "leftArrow", [.command, .option]),
            command(.goForward, "Go Forward", "rightArrow", [.command, .option]),
            command(.toggleInspector, "Toggle Inspector", "i", shiftCmd),
            command(.keyboardShortcuts, "Keyboard Shortcuts", "/", cmd),
        ]

        let textSize = [
            command(.textBigger, "Make Text Bigger", "+", cmd),
            command(.textSmaller, "Make Text Smaller", "-", cmd),
            command(.textDefault, "Default Text Size", "0", cmd),
        ]

        let chat = [
            command(.newChat, "New Chat", "n", cmd, alt: chord("o", shiftCmd)),
            command(.newTemporaryChat, "New Temporary Chat", "n", shiftCmd),
            command(.previousChat, "Previous Chat", "[", cmd, alt: chord("[", shiftCmd)),
            command(.nextChat, "Next Chat", "]", cmd, alt: chord("]", shiftCmd)),
            command(.nextWorkspaceTab, "Next Workspace Tab", "tab", ctrl),
            command(.previousWorkspaceTab, "Previous Workspace Tab", "tab", [.control, .shift]),
            // OC-04. Cmd+W is the window's (reserved, OC-01), so Close Tab is
            // Cmd+Shift+W; Option+Arrow is word navigation in the composer,
            // Cmd+[ / ] is the chat cycle and Cmd+Option+Left/Right is
            // history, so the cycle takes Cmd+Option+[ / ].
            command(.newSessionTab, "New Tab", "t", cmd),
            command(.closeSessionTab, "Close Tab", "w", shiftCmd),
            command(.reopenSessionTab, "Reopen Closed Tab", "t", shiftCmd),
            command(.nextSessionTab, "Next Session Tab", "]", [.command, .option]),
            command(.previousSessionTab, "Previous Session Tab", "[", [.command, .option]),
            command(.focusPrompt, "Focus Prompt", "l", cmd),
            command(.commandCenter, "Command Center", "k", cmd),
            command(.runCommand, "Run Command", "p", shiftCmd),
            command(.addFiles, "Add Files or Photos", "u", cmd, rebindable: false),
            command(.clearHistory, "Clear Chat History", "k", shiftCmd),
            command(.previousTurn, "Previous Turn", "upArrow", [.control, .command]),
            command(.nextTurn, "Next Turn", "downArrow", [.control, .command]),
            command(.generate, "Generate Response", "return", cmd, rebindable: false),
            command(.cancelGeneration, "Cancel Generation", ".", cmd, rebindable: false),
            command(.stopAll, "Stop All", ".", shiftCmd),
        ]

        // Bound on the Record page's Start button (`AudioRecordingPanel`), not
        // in a menu, so the pane is the one place it is listed.
        let audio = [
            command(.startRecording, "Start Recording", "r", shiftCmd, rebindable: false),
        ]

        return [
            ("Navigation & View", navigation),
            ("Text Size & Zoom", textSize),
            ("Chat & Generation", chat),
            ("Audio", audio),
        ]
    }

    /// Every command, in pane order.
    public static var commands: [KeyboardCommand] { groups.flatMap(\.commands) }

    public static func command(for id: KeyboardCommandID) -> KeyboardCommand? {
        commands.first { $0.id == id }
    }

    /// The default chord for `id`, or nil for an id this build does not know.
    public static func defaultChord(for id: KeyboardCommandID) -> KeyChord? {
        command(for: id)?.defaultChord
    }

    /// The hidden alternate chord for `id`, mirrored by
    /// `AlternateShortcutBridge`. Read from here so the bridge, the pane and
    /// the conflict check cannot disagree.
    public static func alternate(for id: KeyboardCommandID) -> KeyChord? {
        command(for: id)?.alternate
    }

    /// The sections with every chord resolved through `map`.
    public static func sections(using map: KeybindMap) -> [KeyboardShortcutSection] {
        groups.map { group in
            KeyboardShortcutSection(
                title: group.title,
                rows: group.commands.map { command in
                    let resolved = map.chord(for: command.id) ?? command.defaultChord
                    return KeyboardShortcutRow(
                        commandID: command.id, label: command.label, keys: resolved.display,
                        altKeys: command.alternate?.display, chord: resolved,
                        rebindable: command.rebindable,
                        isCustom: resolved != command.defaultChord)
                })
        }
    }

    private static func section(_ title: String) -> KeyboardShortcutSection {
        sections(using: KeybindMap())[groups.firstIndex { $0.title == title } ?? 0]
    }

    /// The defaults, with no overrides applied.
    public static var navigation: KeyboardShortcutSection { section("Navigation & View") }
    public static var textSize: KeyboardShortcutSection { section("Text Size & Zoom") }
    public static var chat: KeyboardShortcutSection { section("Chat & Generation") }
    public static var audio: KeyboardShortcutSection { section("Audio") }

    public static var sections: [KeyboardShortcutSection] { sections(using: KeybindMap()) }
}
