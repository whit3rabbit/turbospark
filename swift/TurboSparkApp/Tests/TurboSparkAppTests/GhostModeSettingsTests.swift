import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Persistence for the "Always start in Ghost Mode" preference.
///
/// Same shape as `InteractionModeSettingsTests`: the `MacAppSettings` default
/// and round trip, the legacy-decode fallback, and that the flag actually
/// reaches `settings.json` and back. **Not tested here: `AppModel`'s own
/// `@Published` default** -- `loadSettings()` overwrites it during `init`
/// before any caller could observe it, so only `MacAppSettings`'s default is
/// a reachable level.
final class GhostModeSettingsTests: XCTestCase {
    func testSettingsWrittenBeforeTheFieldExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let data = Data(legacy.utf8)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertFalse(decoded.alwaysStartInGhostMode)
    }

    func testAlwaysStartInGhostModeRoundTrips() throws {
        let settings = MacAppSettings(alwaysStartInGhostMode: true)
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertTrue(decoded.alwaysStartInGhostMode)
    }

    func testTheDefaultIsOff() {
        XCTAssertFalse(MacAppSettings().alwaysStartInGhostMode)
    }

    @MainActor
    func testTheFlagPersistsAcrossANewAppModelInstance() {
        // Start from "no settings file", so this does not depend on whatever
        // another test in this process left behind in the shared (test-
        // redirected, per `AppStorageRoot`) storage directory.
        try? FileManager.default.removeItem(at: AppStorageRoot.file("settings.json"))

        let first = AppModel()
        XCTAssertFalse(first.alwaysStartInGhostMode, "a fresh install must not start in Ghost Mode")

        first.alwaysStartInGhostMode = true
        first.persistSettings()

        // A second instance re-reads `loadSettings()` from the same disk
        // store, the same path a real relaunch takes.
        let second = AppModel()
        XCTAssertTrue(second.alwaysStartInGhostMode)

        second.alwaysStartInGhostMode = false
        second.persistSettings()
        let third = AppModel()
        XCTAssertFalse(third.alwaysStartInGhostMode)
    }
}
