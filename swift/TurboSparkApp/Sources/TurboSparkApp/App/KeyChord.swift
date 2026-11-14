import AppKit
import SwiftUI

/// A keyboard chord: one key plus a modifier set (OC-01).
///
/// Stored as the CHARACTER SwiftUI's `.keyboardShortcut` takes (or a named
/// key), never a hardware key code. A key code would survive a layout change
/// but cannot be handed to `KeyEquivalent`, so the menu could not carry it.
/// The cost is stated, not hidden: a chord recorded on one keyboard layout
/// names a character, and on another layout that character may sit elsewhere
/// or need Shift. The user re-records it. Shift is part of the stored set, so
/// a recorded Cmd+Shift+] is the character "]" with shift, as the existing
/// defaults are spelled.
///
/// Supported keys: printable characters (letters, digits, punctuation, other
/// scripts), Space, Return, Tab, Delete and the four arrows. Escape is the
/// recorder's cancel key and is never bindable. Function keys are not
/// supported in this version: `KeyEquivalent` has no named function keys.
public struct KeyChord: Hashable, Sendable, Codable {
    public struct Modifiers: OptionSet, Hashable, Sendable {
        public let rawValue: UInt8
        public init(rawValue: UInt8) { self.rawValue = rawValue }
        public static let command = Modifiers(rawValue: 1 << 0)
        public static let control = Modifiers(rawValue: 1 << 1)
        public static let option = Modifiers(rawValue: 1 << 2)
        public static let shift = Modifiers(rawValue: 1 << 3)

        /// Stable persisted spellings, in the order macOS writes them.
        static let names: [(Modifiers, String)] = [
            (.control, "control"), (.option, "option"), (.shift, "shift"), (.command, "command"),
        ]
    }

    /// Canonical key: a lowercase single character, or one of `namedKeys`.
    public let key: String
    public let modifiers: Modifiers

    static let namedKeys: Set<String> = [
        "space", "return", "tab", "delete", "leftArrow", "rightArrow", "upArrow", "downArrow",
    ]

    /// Nil when `key` is not a supported key. Single characters are lowercased
    /// (Shift lives in `modifiers`, so "K" and "k" are one key).
    public init?(key: String, modifiers: Modifiers) {
        guard let canonical = Self.canonicalKey(key) else { return nil }
        self.key = canonical
        self.modifiers = modifiers
    }

    static func canonicalKey(_ raw: String) -> String? {
        if namedKeys.contains(raw) { return raw }
        let lowered = raw.lowercased()
        guard lowered.count == 1, let scalar = lowered.unicodeScalars.first else { return nil }
        // Controls, whitespace (Space is named), and the private-use block
        // AppKit reports function and navigation keys in are not characters a
        // menu can carry.
        if scalar.properties.generalCategory == .control { return nil }
        if scalar.properties.isWhitespace { return nil }
        if (0xE000...0xF8FF).contains(scalar.value) { return nil }
        if scalar.value == 0x7F { return nil }
        return lowered
    }

    // MARK: SwiftUI

    public var keyEquivalent: KeyEquivalent {
        switch key {
        case "space": return .space
        case "return": return .return
        case "tab": return .tab
        case "delete": return .delete
        case "leftArrow": return .leftArrow
        case "rightArrow": return .rightArrow
        case "upArrow": return .upArrow
        case "downArrow": return .downArrow
        default: return KeyEquivalent(key.first ?? " ")
        }
    }

    public var eventModifiers: EventModifiers {
        var result: EventModifiers = []
        if modifiers.contains(.command) { result.insert(.command) }
        if modifiers.contains(.control) { result.insert(.control) }
        if modifiers.contains(.option) { result.insert(.option) }
        if modifiers.contains(.shift) { result.insert(.shift) }
        return result
    }

    // MARK: Display

    /// The glyph form the Shortcuts pane shows, modifiers in the macOS order
    /// (Control, Option, Shift, Command) and the key last: "\u{21E7} \u{2318} P".
    public var display: String {
        var parts: [String] = []
        if modifiers.contains(.control) { parts.append("\u{2303}") }
        if modifiers.contains(.option) { parts.append("\u{2325}") }
        if modifiers.contains(.shift) { parts.append("\u{21E7}") }
        if modifiers.contains(.command) { parts.append("\u{2318}") }
        parts.append(keyDisplay)
        return parts.joined(separator: " ")
    }

    var keyDisplay: String {
        switch key {
        case "space": return AppStrings.localized("Space")
        case "return": return "\u{21A9}"
        case "tab": return "\u{21E5}"
        case "delete": return "\u{232B}"
        case "leftArrow": return "\u{2190}"
        case "rightArrow": return "\u{2192}"
        case "upArrow": return "\u{2191}"
        case "downArrow": return "\u{2193}"
        default: return key.uppercased()
        }
    }

    // MARK: Codable (hand-editable: {"key": "k", "modifiers": ["shift", "command"]})

    private enum CodingKeys: String, CodingKey { case key, modifiers }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let rawKey = try container.decode(String.self, forKey: .key)
        let names = try container.decode([String].self, forKey: .modifiers)
        var set: Modifiers = []
        for name in names {
            // An unknown modifier name is a decode failure for THIS entry, so
            // the store drops the override rather than guessing what it meant.
            guard let match = Modifiers.names.first(where: { $0.1 == name }) else {
                throw DecodingError.dataCorruptedError(
                    forKey: .modifiers, in: container, debugDescription: "unknown modifier \(name)")
            }
            set.insert(match.0)
        }
        guard let chord = KeyChord(key: rawKey, modifiers: set) else {
            throw DecodingError.dataCorruptedError(
                forKey: .key, in: container, debugDescription: "unsupported key")
        }
        self = chord
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(key, forKey: .key)
        try container.encode(
            Modifiers.names.filter { modifiers.contains($0.0) }.map(\.1), forKey: .modifiers)
    }
}

// MARK: - Capture from an NSEvent

extension KeyChord {
    public enum Capture: Equatable, Sendable {
        /// Plain Escape: the recorder's cancel key.
        case cancel
        case chord(KeyChord)
        /// A key or key plus modifiers that cannot be a menu chord (Escape
        /// with modifiers, a function key, a dead key with no character).
        case unsupported
    }

    /// Pure so the mapping is testable without an event. `unmodifiedCharacters`
    /// is `event.characters(byApplyingModifiers: [])`: the character as if no
    /// modifier were held, because `charactersIgnoringModifiers` still applies
    /// Shift and would record Cmd+Shift+] as "}" -- not what the menu spells.
    public static func capture(
        keyCode: UInt16, unmodifiedCharacters: String?, flags: NSEvent.ModifierFlags
    ) -> Capture {
        var modifiers: Modifiers = []
        // Only the four chord modifiers count: Caps Lock, the function flag
        // and the numeric-pad flag ride along on arrow keys.
        if flags.contains(.command) { modifiers.insert(.command) }
        if flags.contains(.control) { modifiers.insert(.control) }
        if flags.contains(.option) { modifiers.insert(.option) }
        if flags.contains(.shift) { modifiers.insert(.shift) }

        let named: String?
        switch keyCode {
        case 53:
            return modifiers.isEmpty ? .cancel : .unsupported
        case 123: named = "leftArrow"
        case 124: named = "rightArrow"
        case 125: named = "downArrow"
        case 126: named = "upArrow"
        case 48: named = "tab"
        case 36, 76: named = "return"
        case 49: named = "space"
        case 51: named = "delete"
        default: named = nil
        }
        let raw = named ?? unmodifiedCharacters ?? ""
        guard let chord = KeyChord(key: raw, modifiers: modifiers) else { return .unsupported }
        return .chord(chord)
    }
}

// MARK: - Reserved chords

extension KeyChord {
    /// Chords the capture control refuses outright. A STATIC list: the text
    /// editing chords every text field answers, the app-level chords macOS
    /// owns (quit, close, hide, minimize, settings, switcher), and the system
    /// shortcuts that are well known and fixed by default (Spotlight, screenshots,
    /// Mission Control arrows, full screen, lock). The user can rebind most of
    /// those system shortcuts in System Settings, and an app cannot read that
    /// table, so a chord this list passes may still be taken by the system;
    /// the chord then never reaches the menu. That limit is documented in the
    /// OC-01 draft rather than hidden.
    public var isReserved: Bool {
        Self.reservedChords.contains(self)
    }

    /// Everything a chord needs besides the key: Command or Control. A chord
    /// of only Shift or Option types a character.
    public var hasCommandOrControl: Bool {
        modifiers.contains(.command) || modifiers.contains(.control)
    }

    private static let reservedChords: Set<KeyChord> = {
        var set = Set<KeyChord>()
        func add(_ key: String, _ modifiers: Modifiers) {
            if let chord = KeyChord(key: key, modifiers: modifiers) { set.insert(chord) }
        }
        let cmd: Modifiers = .command
        // Text editing and window chords every text field and window answer.
        for key in ["c", "v", "x", "a", "z", "q", "w", "h", "m", ",", "`"] { add(key, cmd) }
        add("z", [.command, .shift])
        add("h", [.command, .option])
        add("m", [.command, .option])
        add("w", [.command, .option])
        add("q", [.command, .shift])
        add("tab", cmd)
        add("tab", [.command, .shift])
        for arrow in ["leftArrow", "rightArrow", "upArrow", "downArrow", "delete"] {
            // Option-only arrows are refused earlier (no Command or Control).
            add(arrow, cmd)
            add(arrow, [.command, .shift])
        }
        // System shortcuts.
        add("space", cmd)
        add("space", .control)
        add("space", [.command, .control])
        add("space", [.command, .option])
        add("f", [.command, .control])
        add("q", [.command, .control])
        add("d", [.command, .option])
        for digit in ["3", "4", "5"] { add(digit, [.command, .shift]) }
        for arrow in ["leftArrow", "rightArrow", "upArrow", "downArrow"] {
            add(arrow, .control)
            add(arrow, [.control, .shift])
        }
        return set
    }()
}
