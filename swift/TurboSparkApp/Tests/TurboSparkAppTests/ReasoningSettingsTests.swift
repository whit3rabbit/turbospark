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
        let testModel = InstalledModel(
            alias: "qwen-test",
            repo: "qwen/qwen-test",
            path: "/path/to/qwen-test.gturbo",
            family: "qwen38",
            installBytes: 1024
        )
        appModel.selected = testModel
        appModel.setReasoning(.off)
        XCTAssertEqual(appModel.reasoning, .off)

        appModel.setReasoning(.high)
        XCTAssertEqual(appModel.reasoning, .high)
        XCTAssertEqual(appModel.modelReasoningDefaults["qwen-test"], "high")

        appModel.setReasoning(.xhigh)
        XCTAssertEqual(appModel.reasoning, .xhigh)
        XCTAssertEqual(appModel.modelReasoningDefaults["qwen-test"], "xhigh")
    }

    @MainActor
    func testAppModelRestoresReasoningPreferenceForModel() {
        let appModel = AppModel()
        appModel.modelReasoningDefaults["qwen38"] = "xhigh"
        appModel.modelReasoningDefaults["gemma4"] = "low"

        let qwenModel = InstalledModel(
            alias: "qwen38",
            repo: "qwen/qwen38",
            path: "/path/to/qwen38.gturbo",
            family: "qwen38",
            installBytes: 1024
        )
        appModel.restoreReasoningPreference(for: qwenModel)
        XCTAssertEqual(appModel.reasoning, .xhigh)

        let gemmaModel = InstalledModel(
            alias: "gemma4",
            repo: "google/gemma4",
            path: "/path/to/gemma4.gturbo",
            family: "gemma4",
            installBytes: 1024
        )
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
