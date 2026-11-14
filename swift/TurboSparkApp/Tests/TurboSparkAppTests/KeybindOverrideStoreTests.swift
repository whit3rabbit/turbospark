import AppKit
import XCTest
@testable import TurboSparkApp

/// The static name the menu spells for each rebindable command id, shared by
/// the scan tests. Navigation rows are derived per section and have none.
enum KeybindMenuSpelling {
    static let names: [KeyboardCommandID: String] = [
        .toggleSidebar: ".toggleSidebar", .goBack: ".goBack", .goForward: ".goForward",
        .toggleInspector: ".toggleInspector", .keyboardShortcuts: ".keyboardShortcuts",
        .textBigger: ".textBigger", .textSmaller: ".textSmaller", .textDefault: ".textDefault",
        .newChat: ".newChat", .newTemporaryChat: ".newTemporaryChat",
        .previousChat: ".previousChat", .nextChat: ".nextChat",
        .nextWorkspaceTab: ".nextWorkspaceTab", .previousWorkspaceTab: ".previousWorkspaceTab",
        .focusPrompt: ".focusPrompt", .commandCenter: ".commandCenter",
        .runCommand: ".runCommand", .clearHistory: ".clearHistory",
        .previousTurn: ".previousTurn", .nextTurn: ".nextTurn", .stopAll: ".stopAll",
    ]

    static func id(forName name: String) -> KeyboardCommandID? {
        names.first { $0.value == name }?.key
    }
}

/// OC-01. The pure map carries the rules (resolution order, conflicts,
/// reserved chords, sanitizing), the store carries persistence; each is held
/// here against its own mutation.
final class KeybindMapTests: XCTestCase {
    private func chord(_ key: String, _ modifiers: KeyChord.Modifiers) -> KeyChord {
        KeyChord(key: key, modifiers: modifiers)!
    }

    // MARK: Inventory

    func testEveryCommandResolvesToItsDefaultWithACleanMap() {
        let map = KeybindMap()
        for command in KeyboardShortcutCatalog.commands {
            XCTAssertEqual(map.chord(for: command.id), command.defaultChord, command.label)
        }
    }

    func testCommandIDsAreUniqueAndNoTwoCommandsShareAChord() {
        let commands = KeyboardShortcutCatalog.commands
        XCTAssertEqual(commands.count, Set(commands.map(\.id)).count)
        var seen: [KeyChord: String] = [:]
        for command in commands {
            for chord in [command.defaultChord, command.alternate].compactMap({ $0 }) {
                XCTAssertNil(seen[chord], "\(command.label) shares \(chord.display) with \(seen[chord] ?? "")")
                seen[chord] = command.label
            }
        }
    }

    func testNoDefaultChordIsReservedOrModifierless() {
        // A default the capture control would refuse could never be
        // re-recorded after a reset.
        for command in KeyboardShortcutCatalog.commands {
            XCTAssertFalse(command.defaultChord.isReserved, command.label)
            XCTAssertTrue(command.defaultChord.hasCommandOrControl, command.label)
        }
    }

    // MARK: Resolution order

    func testAnOverrideWinsOverTheDefault() {
        var map = KeybindMap()
        let target = chord("u", [.shift, .command])
        XCTAssertEqual(map.assign(target, to: .keyboardShortcuts), .assigned)
        XCTAssertEqual(map.chord(for: .keyboardShortcuts), target)
        XCTAssertTrue(map.isCustom(.keyboardShortcuts))
        // Untouched commands keep their defaults.
        XCTAssertEqual(
            map.chord(for: .commandCenter), KeyboardShortcutCatalog.defaultChord(for: .commandCenter))
    }

    func testResolvedSectionsReflectTheOverride() {
        var map = KeybindMap()
        _ = map.assign(chord("u", [.shift, .command]), to: .keyboardShortcuts)
        let row = KeyboardShortcutCatalog.sections(using: map)
            .flatMap(\.rows).first { $0.commandID == .keyboardShortcuts }
        XCTAssertEqual(row?.keys, "\u{21E7} \u{2318} U")
        XCTAssertEqual(row?.isCustom, true)
        // The defaults accessors are unaffected.
        XCTAssertEqual(
            KeyboardShortcutCatalog.navigation.rows.first { $0.commandID == .keyboardShortcuts }?.keys,
            "\u{2318} /")
    }

    // MARK: Conflicts

    func testAChordAnotherCommandHoldsIsRefusedWithTheHolderNamed() {
        var map = KeybindMap()
        // Cmd+K is Command Center.
        let outcome = map.assign(chord("k", .command), to: .newChat)
        XCTAssertEqual(outcome, .refused(.heldBy(.commandCenter, isAlternate: false)))
        XCTAssertFalse(map.isCustom(.newChat))
        XCTAssertEqual(
            map.chord(for: .newChat), KeyboardShortcutCatalog.defaultChord(for: .newChat))
    }

    func testAHiddenAlternateChordIsRefused() {
        var map = KeybindMap()
        // Cmd+Shift+O is New Chat's alternate, and the target is another command.
        XCTAssertEqual(
            map.assign(chord("o", [.shift, .command]), to: .keyboardShortcuts),
            .refused(.heldBy(.newChat, isAlternate: true)))
        // Even the command that owns the alternate cannot double-bind it.
        XCTAssertEqual(
            map.assign(chord("o", [.shift, .command]), to: .newChat),
            .refused(.heldBy(.newChat, isAlternate: true)))
    }

    func testAFixedCommandsChordIsRefusedAndAFixedCommandCannotBeRebound() {
        var map = KeybindMap()
        // Cmd+Return is the fixed Generate Response chord.
        XCTAssertEqual(
            map.assign(chord("return", .command), to: .newChat),
            .refused(.heldBy(.generate, isAlternate: false)))
        XCTAssertEqual(
            map.assign(chord("g", [.shift, .command]), to: .generate),
            .refused(.notRebindable))
        XCTAssertEqual(
            map.assign(chord("g", [.shift, .command]), to: "no.such.command"),
            .refused(.notRebindable))
    }

    func testAChordFreedByARebindIsAvailableAndRestoreIsRefusedWhileItIsTaken() {
        var map = KeybindMap()
        let moved = chord("u", [.shift, .command])
        XCTAssertEqual(map.assign(moved, to: .newChat), .assigned)
        // Cmd+N is free now: another command may take it.
        XCTAssertEqual(map.assign(chord("n", .command), to: .goBack), .assigned)
        // Restoring New Chat would put two commands on Cmd+N: refused, with
        // the current holder named.
        XCTAssertEqual(
            map.restoreDefault(.newChat), .refused(.heldBy(.goBack, isAlternate: false)))
        XCTAssertEqual(map.chord(for: .newChat), moved)
        // Free the chord and the restore goes through.
        XCTAssertEqual(map.restoreDefault(.goBack), .restoredDefault)
        XCTAssertEqual(map.restoreDefault(.newChat), .restoredDefault)
        XCTAssertTrue(map.overrides.isEmpty)
    }

    func testAssigningTheDefaultChordRemovesTheOverride() {
        var map = KeybindMap()
        _ = map.assign(chord("u", [.shift, .command]), to: .keyboardShortcuts)
        XCTAssertEqual(map.assign(chord("/", .command), to: .keyboardShortcuts), .restoredDefault)
        XCTAssertTrue(map.overrides.isEmpty)
        XCTAssertEqual(map.assign(chord("/", .command), to: .keyboardShortcuts), .unchanged)
    }

    func testRestoreOfAnUnchangedCommandIsANoOp() {
        var map = KeybindMap()
        XCTAssertEqual(map.restoreDefault(.newChat), .unchanged)
    }

    // MARK: Reserved chords

    func testReservedAndModifierlessChordsAreRefused() {
        var map = KeybindMap()
        for reserved in [
            chord("q", .command), chord("c", .command), chord("v", .command),
            chord(",", .command), chord("tab", .command), chord("space", .command),
            chord("leftArrow", .command), chord("delete", [.command, .shift]),
            chord("rightArrow", .control), chord("4", [.shift, .command]),
        ] {
            XCTAssertEqual(
                map.assign(reserved, to: .keyboardShortcuts), .refused(.reserved), reserved.display)
        }
        for typing in [
            chord("k", .shift), chord("k", .option), chord("k", []), chord("leftArrow", .option),
        ] {
            XCTAssertEqual(
                map.assign(typing, to: .keyboardShortcuts), .refused(.needsCommandOrControl),
                typing.display)
        }
        XCTAssertTrue(map.overrides.isEmpty)
    }

    func testResetAllClearsEveryOverride() {
        var map = KeybindMap()
        _ = map.assign(chord("u", [.shift, .command]), to: .keyboardShortcuts)
        _ = map.assign(chord("y", [.shift, .command]), to: .newChat)
        map.resetAll()
        XCTAssertTrue(map.overrides.isEmpty)
    }

    // MARK: Sanitizing a stored file

    func testSanitizedDropsWhatCannotHoldAndKeepsUnknownIDsAside() {
        let stored: [String: KeyChord] = [
            // Two commands on one chord: exactly one survives.
            "chat.new": chord("u", [.shift, .command]),
            "chat.next": chord("u", [.shift, .command]),
            // Reserved, modifierless, fixed, redundant, and held by a default.
            "chat.previous": chord("q", .command),
            "prompt.focus": chord("k", .shift),
            "generation.generate": chord("g", [.shift, .command]),
            "text.bigger": chord("+", .command),
            "text.smaller": chord("k", .command),
            // An id this build does not know.
            "future.thing": chord("j", [.shift, .command]),
        ]
        let result = KeybindMap.sanitized(stored)
        XCTAssertEqual(result.map.overrides.count, 1)
        XCTAssertNotNil(result.map.overrides["chat.next"], "the alphabetically first claimant is dropped")
        XCTAssertEqual(Array(result.stale.keys), ["future.thing"])
    }

    // MARK: Key capture

    func testCaptureRecordsTheUnshiftedCharacterWithShiftAsAModifier() {
        // Cmd+Shift+] arrives with characters-ignoring-modifiers "}" on a US
        // layout; the recorder asks for the unmodified character instead.
        let capture = KeyChord.capture(keyCode: 30, unmodifiedCharacters: "]", flags: [.command, .shift])
        XCTAssertEqual(capture, .chord(chord("]", [.command, .shift])))
        // Letters are lowercased: shift lives in the modifiers.
        XCTAssertEqual(
            KeyChord.capture(keyCode: 40, unmodifiedCharacters: "K", flags: [.command]),
            .chord(chord("k", .command)))
    }

    func testCaptureNamesSpecialKeysAndIgnoresNoiseFlags() {
        XCTAssertEqual(
            KeyChord.capture(
                keyCode: 123, unmodifiedCharacters: "\u{F702}",
                flags: [.command, .option, .numericPad, .function, .capsLock]),
            .chord(chord("leftArrow", [.command, .option])))
        XCTAssertEqual(
            KeyChord.capture(keyCode: 48, unmodifiedCharacters: "\t", flags: [.control]),
            .chord(chord("tab", .control)))
        XCTAssertEqual(
            KeyChord.capture(keyCode: 36, unmodifiedCharacters: "\r", flags: [.command]),
            .chord(chord("return", .command)))
    }

    func testEscapeCancelsOnlyWithoutModifiersAndFunctionKeysAreUnsupported() {
        XCTAssertEqual(KeyChord.capture(keyCode: 53, unmodifiedCharacters: "\u{1B}", flags: []), .cancel)
        XCTAssertEqual(
            KeyChord.capture(keyCode: 53, unmodifiedCharacters: "\u{1B}", flags: [.command]),
            .unsupported)
        // F1 reports a private-use character.
        XCTAssertEqual(
            KeyChord.capture(keyCode: 122, unmodifiedCharacters: "\u{F704}", flags: [.command]),
            .unsupported)
        XCTAssertEqual(
            KeyChord.capture(keyCode: 122, unmodifiedCharacters: nil, flags: [.command]),
            .unsupported)
    }

    // MARK: Codable

    func testChordRoundTripsAndRejectsAnUnknownModifier() throws {
        let original = chord("]", [.command, .shift])
        let data = try JSONEncoder().encode(original)
        XCTAssertEqual(try JSONDecoder().decode(KeyChord.self, from: data), original)
        let bad = Data(#"{"key":"k","modifiers":["hyper"]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(KeyChord.self, from: bad))
        let unsupported = Data(#"{"key":"escape","modifiers":["command"]}"#.utf8)
        XCTAssertThrowsError(try JSONDecoder().decode(KeyChord.self, from: unsupported))
    }
}

@MainActor
final class KeybindOverrideStoreTests: XCTestCase {
    private var directory: URL!
    private var fileURL: URL { directory.appendingPathComponent("keybind_overrides.json") }

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("KeybindStoreTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        AppJSONStore.clearLastReadError()
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
        AppJSONStore.clearLastReadError()
    }

    private func chord(_ key: String, _ modifiers: KeyChord.Modifiers) -> KeyChord {
        KeyChord(key: key, modifiers: modifiers)!
    }

    func testAnOverrideSurvivesARelaunchAndClearingRemovesItFromTheFile() throws {
        let first = KeybindOverrideStore(fileURL: fileURL)
        let target = chord("u", [.shift, .command])
        XCTAssertEqual(first.assign(target, to: .keyboardShortcuts), .assigned)

        let second = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertEqual(second.chord(for: .keyboardShortcuts), target)

        XCTAssertEqual(second.restoreDefault(.keyboardShortcuts), .restoredDefault)
        let third = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertEqual(
            third.chord(for: .keyboardShortcuts),
            KeyboardShortcutCatalog.defaultChord(for: .keyboardShortcuts))
        XCTAssertTrue(third.map.overrides.isEmpty)
    }

    func testARefusedAssignmentWritesNothing() {
        let store = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertEqual(
            store.assign(chord("k", .command), to: .newChat),
            .refused(.heldBy(.commandCenter, isAlternate: false)))
        XCTAssertFalse(FileManager.default.fileExists(atPath: fileURL.path))
    }

    func testAnUndecodableFileIsQuarantinedAndDefaultsApply() throws {
        try Data("{ this is not json".utf8).write(to: fileURL)
        let readFailuresBefore = AppJSONStore.readFailureCount

        let store = KeybindOverrideStore(fileURL: fileURL)

        XCTAssertTrue(store.map.overrides.isEmpty)
        XCTAssertEqual(
            store.chord(for: .newChat), KeyboardShortcutCatalog.defaultChord(for: .newChat))
        XCTAssertEqual(AppJSONStore.readFailureCount, readFailuresBefore + 1)
        XCTAssertNotNil(AppJSONStore.lastReadError, "the toast latch is set")
        // The bytes are kept beside the store, not overwritten.
        let siblings = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        let kept = siblings.filter { $0.contains("corrupt-") }
        XCTAssertEqual(kept.count, 1)
        XCTAssertFalse(FileManager.default.fileExists(atPath: fileURL.path))
        let keptData = try Data(contentsOf: directory.appendingPathComponent(kept[0]))
        XCTAssertEqual(String(decoding: keptData, as: UTF8.self), "{ this is not json")
    }

    func testAFileWithoutAnOverridesObjectIsQuarantinedToo() throws {
        try Data(#"{"version": 1}"#.utf8).write(to: fileURL)
        let store = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertTrue(store.map.overrides.isEmpty)
        XCTAssertNotNil(AppJSONStore.lastReadError)
    }

    func testOneBadEntryIsDroppedAndTheRestLoad() throws {
        let json = """
            {"version": 1, "overrides": {
              "view.keyboardShortcuts": {"key": "u", "modifiers": ["shift", "command"]},
              "chat.new": {"key": "y", "modifiers": ["hyper"]},
              "chat.next": "not an object"
            }}
            """
        try Data(json.utf8).write(to: fileURL)
        let store = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertEqual(store.chord(for: .keyboardShortcuts), chord("u", [.shift, .command]))
        XCTAssertEqual(store.chord(for: .newChat), KeyboardShortcutCatalog.defaultChord(for: .newChat))
        XCTAssertNil(AppJSONStore.lastReadError, "a bad entry is not a corrupt file")
    }

    func testAHandEditedDuplicateIsDroppedOnLoad() throws {
        let json = """
            {"version": 1, "overrides": {
              "chat.new": {"key": "k", "modifiers": ["command"]}
            }}
            """
        // Cmd+K is Command Center's default, so the stored override cannot hold.
        try Data(json.utf8).write(to: fileURL)
        let store = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertEqual(store.chord(for: .newChat), KeyboardShortcutCatalog.defaultChord(for: .newChat))
        XCTAssertEqual(
            store.chord(for: .commandCenter), KeyboardShortcutCatalog.defaultChord(for: .commandCenter))
    }

    func testAnUnknownIDIsKeptInTheFileAcrossAWrite() throws {
        let json = """
            {"version": 1, "overrides": {
              "future.thing": {"key": "j", "modifiers": ["shift", "command"]}
            }}
            """
        try Data(json.utf8).write(to: fileURL)
        let store = KeybindOverrideStore(fileURL: fileURL)
        XCTAssertTrue(store.map.overrides.isEmpty)
        store.assign(chord("u", [.shift, .command]), to: .keyboardShortcuts)
        let text = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(text.contains("future.thing"))
        XCTAssertTrue(text.contains("view.keyboardShortcuts"))
    }

    func testResetAllEmptiesTheFileAndTheMap() throws {
        let store = KeybindOverrideStore(fileURL: fileURL)
        store.assign(chord("u", [.shift, .command]), to: .keyboardShortcuts)
        store.assign(chord("y", [.shift, .command]), to: .newChat)
        store.resetAll()
        XCTAssertTrue(store.map.overrides.isEmpty)
        XCTAssertTrue(KeybindOverrideStore(fileURL: fileURL).map.overrides.isEmpty)
    }

    // MARK: The menu keeps using the resolution

    func testEveryRebindableCommandIsCarriedThroughTheResolvedChordAndFixedOnesAreNot() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/App")
        let menu = try String(
            contentsOf: sources.appendingPathComponent("TurboSparkApp.swift"), encoding: .utf8)
        // Static names the menu spells for each rebindable id; the navigation
        // rows are derived per section.
        let spelled = KeybindMenuSpelling.names
        for command in KeyboardShortcutCatalog.commands where command.rebindable {
            if command.id.rawValue.hasPrefix("nav.") { continue }
            let name = try XCTUnwrap(spelled[command.id], "\(command.id.rawValue) has no menu site listed")
            XCTAssertTrue(
                menu.contains(".keyboardChord(\(name), keybinds)"),
                "\(command.label) is not carried through the resolved chord")
        }
        XCTAssertTrue(menu.contains(".keyboardChord(.navigation(section), keybinds)"))
        // Only the two fixed Generation chords may be a literal shortcut.
        let literals = menu.components(separatedBy: ".keyboardShortcut(").count - 1
        XCTAssertEqual(literals, 2, "a menu command bypasses the override store")
    }
}
