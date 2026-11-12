import Foundation

/// A parsed `computer_key` value: one virtual key plus modifiers.
///
/// Key codes are the ANSI layout's virtual keys (`kVK_*` in Carbon's
/// Events.h). Letters map by position, which is what shortcuts expect; text
/// goes through `computer_type`, which posts Unicode and ignores layout.
public struct ComputerUseKeyCombo: Equatable, Sendable {
    public struct Modifiers: OptionSet, Equatable, Sendable {
        public let rawValue: Int
        public init(rawValue: Int) { self.rawValue = rawValue }
        public static let command = Modifiers(rawValue: 1)
        public static let shift = Modifiers(rawValue: 2)
        public static let option = Modifiers(rawValue: 4)
        public static let control = Modifiers(rawValue: 8)
    }

    public let keyCode: UInt16
    public let modifiers: Modifiers

    public init(keyCode: UInt16, modifiers: Modifiers) {
        self.keyCode = keyCode
        self.modifiers = modifiers
    }

    public static func parse(_ text: String) -> ComputerUseKeyCombo? {
        let parts = text.lowercased().split(separator: "+").map {
            $0.trimmingCharacters(in: .whitespaces)
        }
        guard let keyName = parts.last, !keyName.isEmpty else { return nil }
        var modifiers: Modifiers = []
        for part in parts.dropLast() {
            switch part {
            case "cmd", "command", "meta", "super": modifiers.insert(.command)
            case "shift": modifiers.insert(.shift)
            case "alt", "option", "opt": modifiers.insert(.option)
            case "ctrl", "control": modifiers.insert(.control)
            default: return nil
            }
        }
        guard let code = keyCodes[keyName] else { return nil }
        return ComputerUseKeyCombo(keyCode: code, modifiers: modifiers)
    }

    static let keyCodes: [String: UInt16] = {
        var map: [String: UInt16] = [
            "return": 0x24, "enter": 0x24, "tab": 0x30, "space": 0x31,
            "delete": 0x33, "backspace": 0x33, "forwarddelete": 0x75,
            "escape": 0x35, "esc": 0x35,
            "left": 0x7B, "right": 0x7C, "down": 0x7D, "up": 0x7E,
            "home": 0x73, "end": 0x77, "pageup": 0x74, "pagedown": 0x79,
            "-": 0x1B, "=": 0x18, "[": 0x21, "]": 0x1E, ";": 0x29, "'": 0x27,
            ",": 0x2B, ".": 0x2F, "/": 0x2C, "\\": 0x2A, "`": 0x32,
        ]
        let letters: [(String, UInt16)] = [
            ("a", 0x00), ("s", 0x01), ("d", 0x02), ("f", 0x03), ("h", 0x04), ("g", 0x05),
            ("z", 0x06), ("x", 0x07), ("c", 0x08), ("v", 0x09), ("b", 0x0B), ("q", 0x0C),
            ("w", 0x0D), ("e", 0x0E), ("r", 0x0F), ("y", 0x10), ("t", 0x11), ("o", 0x1F),
            ("u", 0x20), ("i", 0x22), ("p", 0x23), ("l", 0x25), ("j", 0x26), ("k", 0x28),
            ("n", 0x2D), ("m", 0x2E),
            ("1", 0x12), ("2", 0x13), ("3", 0x14), ("4", 0x15), ("6", 0x16), ("5", 0x17),
            ("9", 0x19), ("7", 0x1A), ("8", 0x1C), ("0", 0x1D),
        ]
        for (name, code) in letters { map[name] = code }
        let functionKeys: [UInt16] = [0x7A, 0x78, 0x63, 0x76, 0x60, 0x61, 0x62, 0x64, 0x65, 0x6D, 0x67, 0x6F]
        for (index, code) in functionKeys.enumerated() { map["f\(index + 1)"] = code }
        return map
    }()
}
