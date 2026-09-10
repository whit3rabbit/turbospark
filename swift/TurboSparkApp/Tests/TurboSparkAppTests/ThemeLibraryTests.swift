import SwiftUI
import XCTest
@testable import TurboSparkApp

@MainActor
final class ThemeLibraryTests: XCTestCase {
    func testPairedSelectionKeepsTypographyAndDetectsCustomEdits() throws {
        let manager = AppearanceManager()
        manager.setUIFont(family: "Avenir", weight: "Medium")
        manager.uiFontSize = 19
        manager.codeFontSize = 15
        let preset = try XCTUnwrap(ThemePreset.additionalPresets.first)
        manager.applyPreset(preset)
        XCTAssertTrue(manager.lightConfig.matchesPalette(preset.light))
        XCTAssertTrue(manager.darkConfig.matchesPalette(preset.dark))
        XCTAssertEqual(manager.lightConfig.uiFontFamily, "Avenir")
        XCTAssertEqual(manager.darkConfig.uiFontFamily, "Avenir")
        XCTAssertEqual(manager.uiFontSize, 19)
        XCTAssertEqual(manager.codeFontSize, 15)
        XCTAssertEqual(manager.currentThemeID, preset.id)
        manager.darkConfig.foregroundHex = "#AABBCC"
        XCTAssertNil(manager.currentThemeID)
        XCTAssertEqual(manager.currentThemeName, "Custom")
    }

    func testSavedThemeLifecycleAndResetPreservesLibrary() throws {
        let manager = AppearanceManager()
        manager.applyPreset(ThemePreset.additionalPresets[1])
        let first = try manager.saveTheme(named: "Personal")
        let duplicate = try manager.saveTheme(named: "Copy")
        XCTAssertNotEqual(first.id, duplicate.id)
        try manager.renameTheme(id: duplicate.id, name: "Renamed")
        XCTAssertEqual(manager.currentThemeName, "Renamed")
        let light = manager.lightConfig, dark = manager.darkConfig
        manager.deleteTheme(id: duplicate.id)
        XCTAssertEqual(manager.lightConfig, light)
        XCTAssertEqual(manager.darkConfig, dark)
        XCTAssertNil(manager.currentThemeID)
        manager.resetToDefaults()
        XCTAssertTrue(manager.savedThemes.contains { $0.id == first.id })
        XCTAssertEqual(manager.currentThemeID, "sparkblue")
        let reloaded = AppearanceManager()
        XCTAssertTrue(reloaded.savedThemes.contains { $0.id == first.id })
    }

    func testImportFailureChangesNothingAndRoundTripHasFreshIdentity() throws {
        let manager = AppearanceManager()
        let saved = try manager.saveTheme(named: "Round trip")
        let data = try manager.exportTheme()
        let imported = try manager.importTheme(data: data)
        XCTAssertNotEqual(imported.id, saved.id)
        XCTAssertTrue(imported.light.matchesPalette(saved.light))
        let before = manager.savedThemes
        let light = manager.lightConfig, dark = manager.darkConfig
        XCTAssertThrowsError(try manager.importTheme(data: Data("{}".utf8)))
        var malformed = imported
        malformed.dark.accentHex = "garbage"
        XCTAssertThrowsError(try manager.importTheme(data: JSONEncoder().encode(malformed)))
        malformed = imported
        malformed.version = 999
        XCTAssertThrowsError(try manager.importTheme(data: JSONEncoder().encode(malformed)))
        XCTAssertEqual(manager.savedThemes, before)
        XCTAssertEqual(manager.lightConfig, light)
        XCTAssertEqual(manager.darkConfig, dark)
    }

    func testOldArchivePreservesCustomColors() throws {
        let data = Data("{\"appearance\":\"dark\"}".utf8)
        let archive = try JSONDecoder().decode(AppearanceArchive.self, from: data)
        XCTAssertEqual(archive.savedThemes, [])
        XCTAssertEqual(archive.appearance, "dark")
        XCTAssertEqual(archive.lightConfig, .defaultLight)
        XCTAssertEqual(archive.darkConfig, .defaultDark)
    }

    func testAddedPalettesHaveReadableTextAndResolvedSurfacesFollowSelection() {
        let manager = AppearanceManager()
        for preset in ThemePreset.additionalPresets {
            XCTAssertGreaterThanOrEqual(preset.light.textContrastRatio, 4.5)
            XCTAssertGreaterThanOrEqual(preset.dark.textContrastRatio, 4.5)
            manager.applyPreset(preset)
            for scheme in [ColorScheme.light, .dark] {
                manager.appearance = scheme == .dark ? .dark : .light
                let theme = ResolvedAppTheme.resolve(manager: manager, colorScheme: scheme, installedFamilies: [])
                XCTAssertEqual(theme.background, manager.activeBackgroundColor(isDark: scheme == .dark))
                XCTAssertNotEqual(theme.surface, theme.background)
                XCTAssertNotEqual(theme.elevatedSurface, theme.surface)
            }
        }
    }
}
