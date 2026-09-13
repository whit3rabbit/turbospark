import XCTest

@testable import TurboSparkApp

/// App-wide response-style persistence and prompt assembly.
final class PersonalityTests: XCTestCase {
    func testBuiltInLibraryDefaultsToNoneSelected() {
        let settings = MacAppSettings()
        XCTAssertNil(UUID(uuidString: settings.activePersonalityID))
        XCTAssertEqual(settings.personalities, AppPersonality.builtIns)
        XCTAssertEqual(settings.personalities.map(\.name), [
            "Formal", "Friendly", "Coach", "Creative", "Concise", "Dry Humor"
        ])
    }

    func testPersonalityLibraryAndSelectionRoundTrip() throws {
        let personality = AppPersonality(
            id: UUID(uuidString: "15131CDE-9CAE-4B11-AED8-B7B1E807D39B")!,
            name: "Custom",
            instructions: "Answer clearly.")
        let settings = MacAppSettings(
            personalities: [personality],
            activePersonalityID: personality.id.uuidString)

        let decoded = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))

        XCTAssertEqual(decoded.personalities, [personality])
        XCTAssertEqual(decoded.activePersonalityID, personality.id.uuidString)
    }

    func testSettingsBeforePersonalitiesDecodeToBuiltInsAndNone() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(legacy.utf8))

        XCTAssertEqual(decoded.personalities, AppPersonality.builtIns)
        XCTAssertEqual(decoded.activePersonalityID, "")
    }

    @MainActor
    func testSelectedPersonalityAddsOneSectionAfterTheUserPrompt() {
        let personality = AppPersonality(
            id: UUID(uuidString: "F5C6F07A-AEF1-4551-A2B2-4F4D5D8E74B7")!,
            name: "Custom",
            instructions: "PERSONALITY-TEXT")
        let model = AppModel()
        model.defaultSystemPrompt = ""
        model.personalities = [personality]
        model.selectedPersonalityID = personality.id

        let sections = model.buildSystemPromptSections(for: nil, userPrompt: "USER-PROMPT")

        XCTAssertEqual(sections.map(\.section), [.userPrompt, .personality])
        XCTAssertEqual(sections.map(\.content), ["USER-PROMPT", "PERSONALITY-TEXT"])
        XCTAssertEqual(
            model.buildSystemPrompt(for: nil, userPrompt: "USER-PROMPT"),
            "USER-PROMPT\n\nPERSONALITY-TEXT")
        XCTAssertEqual(model.appWideSystemPrompt, "PERSONALITY-TEXT")
    }

    @MainActor
    func testAStaleSelectionAddsNoPersonalityPrompt() {
        let model = AppModel()
        model.defaultSystemPrompt = ""
        model.personalities = []
        model.selectedPersonalityID = UUID()

        XCTAssertNil(model.selectedPersonality)
        XCTAssertEqual(model.resolvedPersonalityPrompt, "")
        XCTAssertEqual(model.buildSystemPrompt(for: nil), "")
    }
}
