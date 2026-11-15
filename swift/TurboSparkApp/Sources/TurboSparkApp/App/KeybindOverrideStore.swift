import Foundation
import SwiftUI

/// Why a chord was refused (OC-01). The pane turns each case into a sentence
/// that names the holder, so a refusal is never just "no".
public enum KeybindRefusal: Equatable, Sendable {
    /// The command is fixed, or this build does not know it.
    case notRebindable
    /// Only Shift or Option: that types a character instead of firing a command.
    case needsCommandOrControl
    /// In the static reserved list (`KeyChord.isReserved`).
    case reserved
    /// Another command, or another command's hidden alternate, already has it.
    case heldBy(KeyboardCommandID, isAlternate: Bool)
}

/// What an assign or restore did.
public enum KeybindOutcome: Equatable, Sendable {
    /// A new override is in the map.
    case assigned
    /// The chord was the command's default, so the override was removed.
    case restoredDefault
    /// Nothing changed (the chord was already in effect).
    case unchanged
    case refused(KeybindRefusal)
}

/// The override layer as a pure value: `id -> chord` for every command the
/// user moved. Resolution order is the one rule the whole feature rests on:
/// an override wins, then the catalog default. There is no unbound state: the
/// clear control restores the default (a command cannot be left without a
/// chord, which keeps every menu command reachable from the keyboard).
///
/// Pure and store-free so the conflict rules, the reserved list and the
/// resolution order are tested without a file or a view.
public struct KeybindMap: Equatable, Sendable {
    public private(set) var overrides: [String: KeyChord]

    public init(overrides: [String: KeyChord] = [:]) {
        self.overrides = overrides
    }

    /// The chord in effect for `id`: the override, else the default. Nil only
    /// for an id this build does not know.
    public func chord(for id: KeyboardCommandID) -> KeyChord? {
        if let override = overrides[id.rawValue] { return override }
        return KeyboardShortcutCatalog.defaultChord(for: id)
    }

    public func isCustom(_ id: KeyboardCommandID) -> Bool {
        overrides[id.rawValue] != nil
    }

    /// Who holds `chord`, other than `id` itself: a command whose resolved
    /// chord it is, or any command's hidden alternate (including `id`'s own,
    /// since two buttons on one chord is undefined in SwiftUI).
    func holder(
        of chord: KeyChord, excluding id: KeyboardCommandID?
    ) -> (id: KeyboardCommandID, isAlternate: Bool)? {
        for command in KeyboardShortcutCatalog.commands {
            if command.alternate == chord { return (command.id, true) }
            if command.id == id { continue }
            let resolved = overrides[command.id.rawValue] ?? command.defaultChord
            if resolved == chord { return (command.id, false) }
        }
        return nil
    }

    /// The refusal `chord` would meet as `id`'s chord, or nil if it is free.
    public func refusal(of chord: KeyChord, for id: KeyboardCommandID) -> KeybindRefusal? {
        guard KeyboardShortcutCatalog.command(for: id)?.rebindable == true else {
            return .notRebindable
        }
        guard chord.hasCommandOrControl else { return .needsCommandOrControl }
        guard !chord.isReserved else { return .reserved }
        if let held = holder(of: chord, excluding: id) {
            return .heldBy(held.id, isAlternate: held.isAlternate)
        }
        return nil
    }

    public mutating func assign(_ chord: KeyChord, to id: KeyboardCommandID) -> KeybindOutcome {
        if chord == self.chord(for: id) { return .unchanged }
        if let refusal = refusal(of: chord, for: id) { return .refused(refusal) }
        if chord == KeyboardShortcutCatalog.defaultChord(for: id) {
            overrides.removeValue(forKey: id.rawValue)
            return .restoredDefault
        }
        overrides[id.rawValue] = chord
        return .assigned
    }

    /// Drops `id`'s override. Refused when another command has since taken
    /// the default chord (it was freed when `id` moved away), naming it.
    public mutating func restoreDefault(_ id: KeyboardCommandID) -> KeybindOutcome {
        guard overrides[id.rawValue] != nil,
            let fallback = KeyboardShortcutCatalog.defaultChord(for: id)
        else { return .unchanged }
        if let held = holder(of: fallback, excluding: id) {
            return .refused(.heldBy(held.id, isAlternate: held.isAlternate))
        }
        overrides.removeValue(forKey: id.rawValue)
        return .restoredDefault
    }

    public mutating func resetAll() {
        overrides.removeAll()
    }

    /// Builds a map from stored entries, dropping what cannot hold: ids this
    /// build does not know (returned as `stale` so a downgrade does not lose
    /// them), fixed commands, a reserved or modifier-less chord, an override
    /// equal to its own default, and a chord two commands claim (the
    /// alphabetically first claimant is dropped, repeatedly, so the result
    /// has no duplicate whatever the file said). A hand-edited file must not
    /// be able to bind two commands to one chord.
    static func sanitized(
        _ stored: [String: KeyChord]
    ) -> (map: KeybindMap, stale: [String: KeyChord]) {
        var kept: [String: KeyChord] = [:]
        var stale: [String: KeyChord] = [:]
        for (rawID, chord) in stored {
            let id = KeyboardCommandID(rawValue: rawID)
            guard let command = KeyboardShortcutCatalog.command(for: id) else {
                stale[rawID] = chord
                continue
            }
            guard command.rebindable, chord != command.defaultChord,
                chord.hasCommandOrControl, !chord.isReserved
            else { continue }
            kept[rawID] = chord
        }
        var map = KeybindMap(overrides: kept)
        var changed = true
        while changed {
            changed = false
            for rawID in map.overrides.keys.sorted() {
                guard let chord = map.overrides[rawID] else { continue }
                let id = KeyboardCommandID(rawValue: rawID)
                if map.holder(of: chord, excluding: id) != nil {
                    map.overrides.removeValue(forKey: rawID)
                    changed = true
                    break
                }
            }
        }
        return (map, stale)
    }
}

/// The persisted shape. `overrides` is required: a file with no such object
/// is not a keybind file, so it is quarantined rather than read as "no
/// overrides". One entry that will not decode (an unknown modifier name, an
/// unsupported key) is dropped alone and does not take the others with it.
struct KeybindArchive: Codable {
    static let currentVersion = 1

    var version: Int
    var overrides: [String: KeyChord]

    private struct Lossy: Decodable {
        let chord: KeyChord?
        init(from decoder: Decoder) throws {
            chord = try? KeyChord(from: decoder)
        }
    }

    private enum CodingKeys: String, CodingKey { case version, overrides }

    init(version: Int = Self.currentVersion, overrides: [String: KeyChord]) {
        self.version = version
        self.overrides = overrides
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        version = (try? container.decodeIfPresent(Int.self, forKey: .version)) ?? Self.currentVersion
        let raw = try container.decode([String: Lossy].self, forKey: .overrides)
        overrides = raw.compactMapValues(\.chord)
    }
}

/// The app-wide, live override store (OC-01). One instance observed by the
/// menu `Commands`, the Shortcuts pane and the help sheet, so a change shows
/// in all three without a relaunch (the `AppearanceManager` pattern).
///
/// UNDER `AppStorageRoot` through `AppJSONStore`, in its own file
/// (`keybind_overrides.json`): not `settings.json`, not `UserDefaults`, and
/// not profile-vault data (shortcuts are a property of this Mac's keyboard
/// habits, not of the profile's private content). A file that fails to
/// decode is quarantined by `AppJSONStore` and defaults apply; the latched
/// read error reaches a toast through `AppModel.surfaceStorageIssues`.
@MainActor
public final class KeybindOverrideStore: ObservableObject {
    public static let shared = KeybindOverrideStore()

    static let storeLabel: String.LocalizationValue = "Keyboard Shortcuts"

    @Published public private(set) var map: KeybindMap
    /// Entries for ids this build does not know, written back untouched.
    private var stale: [String: KeyChord]
    private let fileURL: URL

    public init(fileURL: URL = AppStorageRoot.file("keybind_overrides.json")) {
        self.fileURL = fileURL
        let archive = AppJSONStore.load(KeybindArchive.self, from: fileURL, label: Self.storeLabel)
        let sanitized = KeybindMap.sanitized(archive?.overrides ?? [:])
        map = sanitized.map
        stale = sanitized.stale
    }

    public func chord(for id: KeyboardCommandID) -> KeyChord? { map.chord(for: id) }

    @discardableResult
    public func assign(_ chord: KeyChord, to id: KeyboardCommandID) -> KeybindOutcome {
        var next = map
        let outcome = next.assign(chord, to: id)
        commit(next, outcome)
        return outcome
    }

    @discardableResult
    public func restoreDefault(_ id: KeyboardCommandID) -> KeybindOutcome {
        var next = map
        let outcome = next.restoreDefault(id)
        commit(next, outcome)
        return outcome
    }

    public func resetAll() {
        guard !map.overrides.isEmpty else { return }
        var next = map
        next.resetAll()
        commit(next, .restoredDefault)
    }

    private func commit(_ next: KeybindMap, _ outcome: KeybindOutcome) {
        switch outcome {
        case .assigned, .restoredDefault:
            map = next
            persist()
        case .unchanged, .refused:
            break
        }
    }

    private func persist() {
        var all = stale
        for (id, chord) in map.overrides { all[id] = chord }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        AppJSONStore.save(
            KeybindArchive(overrides: all), to: fileURL, label: Self.storeLabel, encoder: encoder)
    }
}

extension View {
    /// Carries the chord in effect for `id` as this menu item's shortcut.
    /// An id the build does not know carries none rather than a wrong one.
    @ViewBuilder
    func keyboardChord(_ id: KeyboardCommandID, _ store: KeybindOverrideStore) -> some View {
        if let chord = store.chord(for: id) {
            keyboardShortcut(chord.keyEquivalent, modifiers: chord.eventModifiers)
        } else {
            self
        }
    }
}
