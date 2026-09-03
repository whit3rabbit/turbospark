import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Persistence for the primary Chat/Projects interaction mode.
///
/// `AppModel.interactionMode` used to default to `.projects` and was never
/// written to `MacAppSettings`, so a switch back to Chat mode did not
/// survive a relaunch. These cases pin the fix: the `MacAppSettings`
/// default and round trip, the legacy-decode fallback (`swift/CLAUDE.md`
/// Gotcha 13), and that `setInteractionMode` is what actually reaches disk.
///
/// **Not tested here: `AppModel.interactionMode`'s own `@Published`
/// declaration default.** `AppModel.init()` calls `loadSettings()` as its
/// first statement, which unconditionally overwrites that default from
/// `MacAppSettings` before any caller can observe it (the same is true of
/// `guardrailsMode` and every other persisted enum on this class). A test
/// asserting the declared default would pass or fail on leftover
/// `settings.json` state from whichever other test in this process ran
/// first, not on the line it claims to cover; `MacAppSettings`'s own
/// default (`testTheDefaultIsChat`) is the level that's actually reachable.
final class InteractionModeSettingsTests: XCTestCase {
    func testSettingsWrittenBeforeInteractionModeExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let data = Data(legacy.utf8)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.interactionMode, "chat")
    }

    func testInteractionModeRoundTrips() throws {
        let settings = MacAppSettings(interactionMode: "projects")
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.interactionMode, "projects")
    }

    func testTheDefaultIsChat() {
        XCTAssertEqual(MacAppSettings().interactionMode, "chat")
    }

    @MainActor
    func testSetInteractionModePersistsAcrossANewAppModelInstance() {
        // Start from "no settings file", so this does not depend on
        // whatever another test in this process left behind in the shared
        // (test-redirected, per `AppStorageRoot`) storage directory.
        try? FileManager.default.removeItem(at: AppStorageRoot.file("settings.json"))

        let first = AppModel()
        XCTAssertEqual(first.interactionMode, .chat, "a fresh install with no settings file must come up in Chat")

        first.setInteractionMode(.projects)
        XCTAssertEqual(first.interactionMode, .projects)

        // A second instance re-reads `loadSettings()` from the same disk
        // store `AppModel.init` already reads on every launch, so this is
        // the same path a real relaunch takes.
        let second = AppModel()
        XCTAssertEqual(second.interactionMode, .projects)

        // Switching back must persist too, not just the initial move.
        second.setInteractionMode(.chat)
        let third = AppModel()
        XCTAssertEqual(third.interactionMode, .chat)
    }

    @MainActor
    func testProjectlessChatBuildsEmptySystemPrompt() {
        let model = AppModel()
        // In Chat mode with no project, system prompt must be empty to avoid 7k token injection
        XCTAssertEqual(model.buildSystemPrompt(for: nil), "")

        // When a project is provided, it must still produce the project environment and instructions
        let project = AppProject(name: "Demo", rootDirectoryPath: "/tmp/demo", customInstructions: "Custom")
        let prompt = model.buildSystemPrompt(for: project)
        XCTAssertFalse(prompt.isEmpty)
        XCTAssertTrue(prompt.contains("Demo") || prompt.contains("/tmp/demo") || prompt.contains("Custom"))
    }
}
