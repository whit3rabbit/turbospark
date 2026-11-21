import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Opt-in follow-up chips under the newest reply (component audit,
/// suggestion-chips): fixed templates, no model call, never sent.
final class FollowUpSuggestionsTests: XCTestCase {
    private func user(_ text: String = "q") -> AppChatMessage {
        AppChatMessage(role: .user, content: text)
    }

    private func assistant(_ text: String) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: text)
    }

    // MARK: - Which row

    func testOffByDefaultShowsNothing() {
        let messages = [user(), assistant("answer")]
        XCTAssertNil(FollowUpSuggestions.rowID(in: messages, isBusy: false, isEnabled: false))
    }

    func testOnShowsUnderTheNewestReply() {
        let old = assistant("first")
        let reply = assistant("second")
        let messages = [user(), old, user(), reply]
        XCTAssertEqual(FollowUpSuggestions.rowID(in: messages, isBusy: false, isEnabled: true), reply.id)
    }

    func testHiddenWhileBusy() {
        XCTAssertNil(FollowUpSuggestions.rowID(in: [user(), assistant("a")], isBusy: true, isEnabled: true))
    }

    func testHiddenWhenTheNewestPromptHasNoReply() {
        // The old reply would sit above the new prompt looking like its answer.
        XCTAssertNil(FollowUpSuggestions.rowID(
            in: [user(), assistant("a"), user()], isBusy: false, isEnabled: true))
    }

    func testHiddenInAnEmptyChat() {
        // The empty state has its own example cards; do not duplicate them.
        XCTAssertNil(FollowUpSuggestions.rowID(in: [], isBusy: false, isEnabled: true))
    }

    // MARK: - Templates

    func testAtMostThreeDistinctTemplates() {
        let templates = FollowUpSuggestions.templates
        XCTAssertFalse(templates.isEmpty)
        XCTAssertLessThanOrEqual(templates.count, 3)
        XCTAssertEqual(Set(templates).count, templates.count)
    }

    func testTemplatesExistInEveryLanguage() throws {
        let catalogURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Localization/Localizable.xcstrings")
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: catalogURL)) as? [String: Any])
        let strings = try XCTUnwrap(root["strings"] as? [String: Any])
        let expected = Set(AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue))
        let keys = FollowUpSuggestions.templates + [
            "Show follow-up suggestions", "Follow-up suggestions",
            "Adds this to your message without sending it.",
        ]
        for key in keys {
            let entry = try XCTUnwrap(strings[key] as? [String: Any], "missing catalog key: \(key)")
            let locs = try XCTUnwrap(entry["localizations"] as? [String: Any])
            XCTAssertEqual(Set(locs.keys), expected, key)
            for (lang, raw) in locs {
                let unit = (raw as? [String: Any])?["stringUnit"] as? [String: Any]
                XCTAssertFalse((unit?["value"] as? String ?? "").isEmpty, "\(lang) \(key)")
            }
        }
    }

    // MARK: - Setting

    func testSettingIsOffByDefaultAndPersists() throws {
        var settings = MacAppSettings()
        XCTAssertFalse(settings.followUpSuggestionsEnabled)
        settings.followUpSuggestionsEnabled = true
        let restored = try JSONDecoder().decode(
            MacAppSettings.self, from: JSONEncoder().encode(settings))
        XCTAssertTrue(restored.followUpSuggestionsEnabled)
    }

    func testOldOrMalformedSettingsLoadAsOff() throws {
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(legacy.followUpSuggestionsEnabled, "settings written before the option stay off")
        let wrongType = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"followUpSuggestionsEnabled":"yes"}"#.utf8))
        XCTAssertFalse(wrongType.followUpSuggestionsEnabled)
    }

    // MARK: - AppModel

    @MainActor
    func testModelDefaultIsOff() {
        XCTAssertFalse(AppModel().followUpSuggestionsEnabled)
    }

    @MainActor
    func testDraftFollowUpFillsAnEmptyComposerAndDoesNotSend() {
        let model = AppModel()
        // AppModel restores a saved draft, so start from a known empty one.
        model.writePromptTextDirectly("")
        let before = model.selectedChat.messages.count
        model.draftFollowUp("Give an example")
        XCTAssertEqual(model.promptText, "Give an example")
        XCTAssertEqual(model.selectedChat.messages.count, before)
        XCTAssertFalse(model.isRunning)
    }

    @MainActor
    func testDraftFollowUpAppendsToAnExistingDraft() {
        let model = AppModel()
        model.writePromptTextDirectly("My own words")
        model.draftFollowUp("Give an example")
        XCTAssertEqual(model.promptText, "My own words\n\nGive an example")
        XCTAssertFalse(model.isRunning)
    }
}
