import XCTest
@testable import TurboSparkApp

/// Pins the behavior fixes from the 2026-10-11 UI audit that are logic rather
/// than layout: which sections the chat Generation commands may act in, how a
/// seeded chat draft treats an existing one, and that `AppStrings` follows
/// the in-app language rather than the system's.
@MainActor
final class UIAuditFixesTests: XCTestCase {
    /// Generate/Queue sends the CHAT draft, so it may only act where the chat
    /// composer is on screen. Enabled elsewhere, Cmd+Return in Audio or
    /// Humanize fell through a disabled workspace run button to the menu and
    /// sent a draft the user could not see.
    func testOnlyTheChatSectionRunsTheChatGenerationCommand() {
        for section in AppModel.AppNavigationSection.allCases {
            XCTAssertEqual(section.runsChatGenerationCommand, section == .chat, "\(section)")
        }
    }

    /// The workspaces with their own run/stop buttons on Cmd+Return and Cmd+.
    /// are exactly the full-bleed ones. A new full-bleed workspace must decide
    /// this explicitly, which the pairing below forces.
    func testWorkspacesWithOwnRunControlsOwnTheGenerationShortcuts() {
        let owners = AppModel.AppNavigationSection.allCases.filter(\.ownsGenerationShortcuts)
        XCTAssertEqual(Set(owners), [.images, .audio, .humanize])
        for section in AppModel.AppNavigationSection.allCases {
            XCTAssertEqual(section.ownsGenerationShortcuts, section.hidesChatPanes, "\(section)")
            XCTAssertFalse(
                section.ownsGenerationShortcuts && section.runsChatGenerationCommand,
                "\(section) cannot both own the chords and run the chat command")
        }
    }

    /// The Keyboard Shortcuts pane lists chords as data while the menus bind
    /// their own, so the two could drift silently: the audit asked that
    /// labels, menu shortcuts and commands come from one definition. This
    /// reads every menu `Button` in `TurboSparkApp.swift`, pairs its labels
    /// with its `.keyboardShortcut`, renders the chord in the pane's
    /// notation, and requires each catalog row naming a menu command to
    /// show exactly that chord.
    func testShortcutsPaneMatchesTheMenuChords() throws {
        let source = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .appendingPathComponent("Sources/TurboSparkApp/App/TurboSparkApp.swift"),
            encoding: .utf8)
        var menuChords: [String: String] = [:]
        for segment in source.components(separatedBy: "Button {").dropFirst() {
            guard let chord = Self.chord(in: segment) else { continue }
            for label in Self.bundledLabels(in: segment) {
                menuChords[label] = chord
            }
        }
        XCTAssertGreaterThan(menuChords.count, 10, "the menu scan found too few chords to mean anything")
        var checked = 0
        for section in KeyboardShortcutCatalog.sections {
            for row in section.rows {
                guard let chord = menuChords[row.label] ?? menuChords[row.label + "..."] else { continue }
                XCTAssertEqual(row.keys, chord, "\(row.label): pane and menu disagree")
                checked += 1
            }
        }
        XCTAssertGreaterThan(checked, 10, "too few pane rows matched a menu command")
    }

    /// `Text("...", bundle: .module)` labels in a menu button's segment.
    private static func bundledLabels(in segment: String) -> [String] {
        let pattern = try! NSRegularExpression(pattern: #"Text\("([^"]+)", bundle: \.module\)"#)
        let text = String(segment)
        let range = NSRange(text.startIndex..., in: text)
        // Only the button's own label closure: stop at its first shortcut.
        let cutoff = text.range(of: ".keyboardShortcut(").map { NSRange($0, in: text).location } ?? range.length
        return pattern.matches(in: text, range: range)
            .filter { $0.range.location < cutoff }
            .compactMap { Range($0.range(at: 1), in: text).map { String(text[$0]) } }
    }

    /// The first `.keyboardShortcut(...)` in a segment, in the pane's
    /// notation: control, shift, command, then the key.
    private static func chord(in segment: String) -> String? {
        let text = String(segment)
        guard let start = text.range(of: ".keyboardShortcut(") else { return nil }
        let tail = text[start.upperBound...]
        guard let end = tail.firstIndex(of: "\n") else { return nil }
        let call = String(tail[..<end])
        let key: String
        if call.hasPrefix("\"") {
            key = String(call.dropFirst().prefix(while: { $0 != "\"" })).uppercased()
        } else if call.hasPrefix(".return") {
            key = "\u{21A9}"
        } else if call.hasPrefix(".upArrow") {
            key = "\u{2191}"
        } else if call.hasPrefix(".downArrow") {
            key = "\u{2193}"
        } else {
            return nil
        }
        var parts: [String] = []
        if call.contains(".control") { parts.append("\u{2303}") }
        if call.contains(".shift") { parts.append("\u{21E7}") }
        if call.contains(".command") { parts.append("\u{2318}") }
        return (parts + [key]).joined(separator: " ")
    }

    /// The Add menu files built-in slash commands under task groups instead
    /// of one 39-row submenu. Every command must still appear exactly once:
    /// a command the group table forgets lands in "More", never nowhere.
    func testSlashCommandGroupsListEveryCommandExactlyOnce() {
        let groups = SlashCommandMenuGroup.grouped(BuiltInSlashCommand.all)
        let listed = groups.flatMap { $0.commands.map(\.name) }
        XCTAssertEqual(listed.count, Set(listed).count, "a command is listed twice")
        XCTAssertEqual(Set(listed), Set(BuiltInSlashCommand.all.map(\.name)))
        XCTAssertFalse(groups.contains { $0.commands.isEmpty }, "empty submenus are not shown")
        XCTAssertGreaterThan(groups.count, 3, "the commands are actually grouped")
    }

    func testComposeInChatKeepsAnExistingDraft() {
        XCTAssertEqual(AppModel.draft(appending: "Schedule a task", to: ""), "Schedule a task")
        XCTAssertEqual(AppModel.draft(appending: "Schedule a task", to: "  \n"), "Schedule a task")
        XCTAssertEqual(
            AppModel.draft(appending: "Schedule a task", to: "half typed"),
            "half typed\n\nSchedule a task")
    }

    /// `String(localized:)` follows the PROCESS language; `AppStrings` must
    /// follow the picker, like every `Text` does. French is resolved from its
    /// compiled table regardless of the system language this test runs under.
    func testAppStringsResolvesThePickedLanguage() {
        XCTAssertEqual(AppStrings.localized("Cancel", language: .french), "Annuler")
        XCTAssertEqual(AppStrings.localized("Cancel", language: .german), "Abbrechen")
        XCTAssertEqual(AppStrings.localized("Cancel", language: .portuguese), "Cancelar")
        XCTAssertNotEqual(AppStrings.bundle(for: .japanese), Bundle.module)
        // System follows the language the window shows, not `.module`, which
        // keeps the launch language (ProcessLocalizationTests).
        XCTAssertEqual(AppStrings.bundle(for: .system), AppStrings.bundle(for: AppLanguage.systemResolved))
    }

    /// Small and mini actions take a label font that fits their bezel;
    /// inheriting the window's base font overflowed it. Regular and larger
    /// keep inheriting, so header actions match the text beside them.
    func testReducedControlSizesGetAFittingLabelFont() {
        XCTAssertEqual(AppMetrics.labelFontStep(for: .mini), .tiny)
        XCTAssertEqual(AppMetrics.labelFontStep(for: .small), .small)
        XCTAssertNil(AppMetrics.labelFontStep(for: .regular))
        XCTAssertNil(AppMetrics.labelFontStep(for: .large))
    }
}
