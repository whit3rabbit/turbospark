import XCTest
@testable import TurboSparkApp

/// Persistence contracts for the settings that were previously unpersisted
/// or stored in the wrong backend: the server's pinned port
/// (`MacAppSettings`/settings.json) and the appearance archive (migrated off
/// UserDefaults into `appearance.json`). The server API key is deliberately
/// untested here: a Keychain round-trip would touch the host user's real
/// login keychain, which a test must not do.
final class ServerAndAppearanceStoreTests: XCTestCase {
    // MARK: - Server port

    /// Tests that a pinned port survives a save/load round-trip through the
    /// settings store (it reset every launch before this field existed).
    func testServerPinnedPortRoundTripsThroughTheSettingsStore() {
        var settings = MacAppSettings()
        settings.serverPinnedPort = 8471
        MacAppSettingsFileStore.save(settings)
        XCTAssertEqual(MacAppSettingsFileStore.load().serverPinnedPort, 8471)
    }

    /// Tests that a settings.json written before the field existed (or by
    /// hand, without it) decodes to 0, the "first free port" automatic.
    func testSettingsJSONWithoutAPortKeyDecodesToAutomatic() throws {
        let url = AppStorageRoot.file("settings.json")
        try Data("{}".utf8).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }
        XCTAssertEqual(MacAppSettingsFileStore.load().serverPinnedPort, 0)
    }

    // MARK: - Appearance migration

    /// Tests the one-way migration: absent appearance.json plus legacy
    /// UserDefaults keys yields an archive built from those keys, and the
    /// keys are removable only after the JSON store has held a save.
    @MainActor
    func testLegacyDefaultsKeysMigrateIntoTheJSONStore() {
        let originalArchive = AppearanceFileStore.load().archive
        let defaults = UserDefaults.standard
        defaults.set("light", forKey: "TurboSpark.appearance")
        defaults.set(15.0, forKey: "TurboSpark.prefs.uiFontSize")
        defer {
            defaults.removeObject(forKey: "TurboSpark.appearance")
            defaults.removeObject(forKey: "TurboSpark.prefs.uiFontSize")
            _ = AppearanceFileStore.save(originalArchive)
        }

        try? FileManager.default.removeItem(at: AppearanceFileStore.fileURL)
        let (archive, migrated) = AppearanceFileStore.load()
        XCTAssertTrue(migrated, "no appearance.json means the archive came from the legacy keys")
        XCTAssertEqual(archive.appearance, "light")
        XCTAssertEqual(archive.uiFontSize, 15.0)

        XCTAssertTrue(AppearanceFileStore.save(archive))
        AppearanceFileStore.removeLegacyKeys()
        XCTAssertNil(defaults.object(forKey: "TurboSpark.appearance"))
        XCTAssertNil(defaults.object(forKey: "TurboSpark.prefs.uiFontSize"))
    }

    /// Tests that a saved archive reads back identically through the store.
    @MainActor
    func testAppearanceArchiveRoundTripsThroughTheJSONStore() {
        let originalArchive = AppearanceFileStore.load().archive
        defer {
            _ = AppearanceFileStore.save(originalArchive)
        }

        var archive = originalArchive
        archive.uiFontSize = 17.0
        archive.usePointerCursors = true
        XCTAssertTrue(AppearanceFileStore.save(archive))

        let (loaded, migrated) = AppearanceFileStore.load()
        XCTAssertFalse(migrated, "a save means the next load reads the JSON, not the legacy keys")
        XCTAssertEqual(loaded.uiFontSize, 17.0)
        XCTAssertTrue(loaded.usePointerCursors)
    }

    /// Tests that a fresh archive decodes or defaults to a 16px base font size and system appearance.
    @MainActor
    func testAppearanceDefaultFontSizeIs16AndSystemTheme() throws {
        let emptyJSON = Data("{}".utf8)
        let decoded = try JSONDecoder().decode(AppearanceArchive.self, from: emptyJSON)
        XCTAssertEqual(decoded.uiFontSize, 16.0)
        XCTAssertEqual(decoded.appearance, AppAppearance.system.rawValue)

        let defaultArchive = AppearanceArchive()
        XCTAssertEqual(defaultArchive.uiFontSize, 16.0)
        XCTAssertEqual(defaultArchive.appearance, AppAppearance.system.rawValue)
    }

    /// Tests that the settings tabs include safety and search keyword matching works.
    func testSettingsTabsIncludeSafetyAndKeywords() {
        XCTAssertTrue(AppSettingsView.SettingsTab.allCases.contains(.safety))
        XCTAssertTrue(AppSettingsView.SettingsTab.safety.keywords.contains("steering"))
        XCTAssertTrue(AppSettingsView.SettingsTab.appearance.keywords.contains("font"))
        XCTAssertTrue(AppSettingsView.SettingsTab.engine.keywords.contains("thermalforge"))
        XCTAssertTrue(AppSettingsView.SettingsTab.engine.keywords.contains("fan"))
    }

    /// `swift/docs/SWIFT_SETTINGS_AUDIT.md`'s settings-search item: the keyword
    /// lists drift from the panes they describe. Each of these names a
    /// control that is really in that pane (checked against the pane's own
    /// source) and was missing from its tab's keywords before this test was
    /// added -- `Command Classifier Veto` is the one added the SAME DAY as
    /// this audit item, in `PermissionsSettingsPaneView`, with nothing
    /// making it reachable by search.
    func testSettingsKeywordsCoverRecentlyAddedControls() {
        XCTAssertTrue(AppSettingsView.SettingsTab.permissions.keywords.contains("advisory veto"))
        XCTAssertTrue(AppSettingsView.SettingsTab.general.keywords.contains("ghost"))
        XCTAssertTrue(AppSettingsView.SettingsTab.engine.keywords.contains("context window"))
        XCTAssertTrue(AppSettingsView.SettingsTab.skills.keywords.contains("triggers"))
    }
}
