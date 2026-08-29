import AppKit
import SwiftUI
@testable import TurboSparkApp
import XCTest

@MainActor
final class AppearanceSettingsTests: XCTestCase {
    func testThemePresetsAvailable() {
        XCTAssertFalse(ThemePreset.presets.isEmpty)
        let codex = ThemePreset.presets.first(where: { $0.id == "codex" })
        XCTAssertNotNil(codex)
        XCTAssertEqual(codex?.light.preset, "Codex")
        XCTAssertEqual(codex?.dark.preset, "Codex")

        let emerald = ThemePreset.presets.first(where: { $0.id == "turbospark" })
        XCTAssertNotNil(emerald)
        XCTAssertEqual(emerald?.light.accentName, "Emerald")
    }

    func testColorHexParsingAndFormatting() {
        let hexWhite = "#FFFFFF"
        let colorWhite = Color(hex: hexWhite)
        XCTAssertNotNil(colorWhite)

        let hexBlue = "#2563EB"
        let colorBlue = Color(hex: hexBlue)
        XCTAssertNotNil(colorBlue)

        let reHex = colorBlue?.toHex()
        XCTAssertNotNil(reHex)
        XCTAssertEqual(reHex?.uppercased(), "#2563EB")
    }

    func testAppearanceManagerPresetApplication() {
        let manager = AppearanceManager.shared
        let initialPreset = manager.lightConfig.preset

        if let midnight = ThemePreset.presets.first(where: { $0.id == "midnight" }) {
            manager.applyPreset(midnight, forMode: false)
            XCTAssertEqual(manager.lightConfig.preset, "Midnight Blue")
            XCTAssertEqual(manager.lightConfig.accentName, "Blue")

            // Test Dark mode preset application
            manager.applyPreset(midnight, forMode: true)
            XCTAssertEqual(manager.darkConfig.preset, "Midnight Blue")
        }

        // Restore default
        if let codex = ThemePreset.presets.first(where: { $0.id == "codex" }) {
            manager.applyPreset(codex)
            XCTAssertEqual(manager.lightConfig.preset, "Codex")
            XCTAssertEqual(manager.darkConfig.preset, "Codex")
        }
    }

    func testFontWeightResolution() {
        XCTAssertEqual(Font.Weight.fromName("regular"), .regular)
        XCTAssertEqual(Font.Weight.fromName("medium"), .medium)
        XCTAssertEqual(Font.Weight.fromName("semibold"), .semibold)
        XCTAssertEqual(Font.Weight.fromName("bold"), .bold)
        XCTAssertEqual(Font.Weight.fromName("unknown"), .regular)
    }

    func testReduceMotionResolution() {
        let manager = AppearanceManager.shared
        manager.reduceMotion = .system
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: true))

        manager.reduceMotion = .on
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertTrue(manager.shouldReduceMotion(systemReduceMotion: true))

        manager.reduceMotion = .off
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: false))
        XCTAssertFalse(manager.shouldReduceMotion(systemReduceMotion: true))

        // Reset to system
        manager.reduceMotion = .system
    }

    func testDockIconSelection() {
        let manager = AppearanceManager.shared
        manager.dockIcon = .emeraldSpark
        XCTAssertEqual(manager.dockIcon, .emeraldSpark)
        XCTAssertEqual(manager.dockIcon.label, "TurboSpark")

        manager.dockIcon = .codexDark
        XCTAssertEqual(manager.dockIcon, .codexDark)
        XCTAssertEqual(manager.dockIcon.label, "Codex")

        // Reset
        manager.dockIcon = .emeraldSpark
    }
}
