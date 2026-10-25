import TurboSpark
import XCTest

@testable import TurboSparkApp

final class ReasoningSettingsTests: XCTestCase {
    func testSettingsWrittenBeforeReasoningFieldsExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let data = Data(legacy.utf8)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.reasoning, "off")
        XCTAssertEqual(decoded.modelReasoningDefaults, [:])
    }

    func testReasoningSettingsRoundTrip() throws {
        let settings = MacAppSettings(
            reasoning: "high",
            modelReasoningDefaults: [
                "qwen38-27b": "xhigh",
                "gptoss-120b": "medium",
                "/custom/models/test": "low"
            ]
        )
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.reasoning, "high")
        XCTAssertEqual(decoded.modelReasoningDefaults["qwen38-27b"], "xhigh")
        XCTAssertEqual(decoded.modelReasoningDefaults["gptoss-120b"], "medium")
        XCTAssertEqual(decoded.modelReasoningDefaults["/custom/models/test"], "low")
    }

    @MainActor
    func testAppModelSetReasoningUpdatesModelDefaults() {
        let appModel = AppModel()
        let alias = "qwen-test-\(UUID().uuidString.prefix(8))"
        let testModel = InstalledModel(
            alias: alias,
            repo: "qwen/qwen-test",
            path: "/path/to/\(alias).gturbo",
            family: "qwen38",
            installBytes: 1024
        )
        defer { ModelOrganizationStore.shared.removeMetadata(for: testModel) }
        appModel.selected = testModel
        appModel.setReasoning(.off)
        XCTAssertEqual(appModel.reasoning, .off)

        // **KEYED ON THE INSTALL ID** (task 5.5, requirement 8.1). This case
        // asserted the path key (state#96) and, before that, the alias key.
        // The path is not stable: a move or a drive that reconnects under
        // another name changes it, and the remembered level was lost with it.
        // The level now lives in the organization store next to the model's
        // other metadata; `modelReasoningDefaults` is only the legacy table
        // that `restoreReasoningPreference` still READS and retires.
        appModel.setReasoning(.high)
        XCTAssertEqual(appModel.reasoning, .high)
        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: testModel), "high")
        XCTAssertNil(appModel.modelReasoningDefaults[testModel.path])
        XCTAssertNil(appModel.modelReasoningDefaults[alias])

        appModel.setReasoning(.xhigh)
        XCTAssertEqual(appModel.reasoning, .xhigh)
        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: testModel), "xhigh")
    }

    @MainActor
    func testAppModelRestoresReasoningPreferenceForModel() {
        let appModel = AppModel()
        let qwenModel = InstalledModel(
            alias: "qwen38-\(UUID().uuidString.prefix(8))",
            repo: "qwen/qwen38",
            path: "/path/to/qwen38.gturbo",
            family: "qwen38",
            installBytes: 1024
        )
        let gemmaModel = InstalledModel(
            alias: "gemma4-\(UUID().uuidString.prefix(8))",
            repo: "google/gemma4",
            path: "/path/to/gemma4.gturbo",
            family: "gemma4",
            installBytes: 1024
        )
        defer {
            ModelOrganizationStore.shared.removeMetadata(for: qwenModel)
            ModelOrganizationStore.shared.removeMetadata(for: gemmaModel)
        }
        // The bare-alias spelling is the oldest legacy key; it must still be
        // found, and it is re-keyed to the install id as it is read.
        appModel.modelReasoningDefaults[qwenModel.alias] = "xhigh"
        appModel.modelReasoningDefaults[gemmaModel.alias] = "low"

        appModel.restoreReasoningPreference(for: qwenModel)
        XCTAssertEqual(appModel.reasoning, .xhigh)
        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: qwenModel), "xhigh")

        appModel.restoreReasoningPreference(for: gemmaModel)
        XCTAssertEqual(appModel.reasoning, .low)
    }

    func testReasoningEnumCasesAndLabels() {
        for level in GenerateOptions.Reasoning.allCases {
            XCTAssertFalse(level.label.isEmpty)
            XCTAssertFalse(level.descriptionText.isEmpty)
            XCTAssertEqual(level.id, level.rawValue)
        }
    }
}
