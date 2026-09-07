import TurboSpark
import XCTest

@testable import TurboSparkApp

/// `docs/SWIFT_SETTINGS_AUDIT.md`'s subagent-sampling item: every subagent
/// turn ran at a hardcoded `temperature: 0.2` regardless of the user's own
/// Engine settings. `AppModel.samplingOptions()` is the one place both the
/// main chat loop (`executeGenerationTurn`) and every `SubagentRunner.run`
/// call site now read those settings from, so this test is the only thing
/// standing between a future edit and the same drift.
///
/// Every field is set explicitly rather than relied on as a fresh
/// `AppModel()`'s default: `AppModel.init()` calls `loadSettings()` first, so
/// the "default" a fresh instance reports is whatever `settings.json`
/// happens to hold in this process's redirected `AppStorageRoot`
/// (`swift/CLAUDE.md` Gotcha 39), which earlier tests in the same `swift
/// test` invocation can have already written.
@MainActor
final class SamplingOptionsTests: XCTestCase {
    func testSamplingOptionsReflectsEveryEnabledSetting() {
        let appModel = AppModel()
        appModel.temperature = 0.77
        appModel.topKEnabled = true
        appModel.topK = 40
        appModel.topPEnabled = true
        appModel.topP = 0.88
        appModel.repetitionPenaltyEnabled = true
        appModel.repetitionPenalty = 1.15
        appModel.seedEnabled = true
        appModel.seed = 42
        appModel.stopSequences = "###, STOP"

        let options = appModel.samplingOptions()

        XCTAssertEqual(options.temperature, 0.77)
        XCTAssertEqual(options.topK, 40)
        XCTAssertEqual(options.topP, 0.88)
        XCTAssertEqual(options.repetitionPenalty, 1.15)
        XCTAssertEqual(options.seed, 42)
        XCTAssertEqual(options.stop, ["###", "STOP"])
    }

    /// A setting behind a disabled toggle must NOT reach the engine: the
    /// toggle, not the stored number, is what the user turned off.
    func testDisabledTogglesLeaveTheEngineDefault() {
        let appModel = AppModel()
        appModel.topKEnabled = false
        appModel.topK = 999
        appModel.topPEnabled = false
        appModel.topP = 0.01
        appModel.repetitionPenaltyEnabled = false
        appModel.repetitionPenalty = 9.9
        appModel.seedEnabled = false
        appModel.seed = 12345
        appModel.stopSequences = ""

        let options = appModel.samplingOptions()
        let engineDefault = GenerateOptions()

        XCTAssertEqual(options.topK, engineDefault.topK)
        XCTAssertEqual(options.topP, engineDefault.topP)
        XCTAssertEqual(options.repetitionPenalty, engineDefault.repetitionPenalty)
        XCTAssertEqual(options.seed, engineDefault.seed)
        XCTAssertEqual(options.stop, engineDefault.stop)
    }

    /// The provider `AppToolRegistry` (an enum with no `AppModel` to read;
    /// `swift/CLAUDE.md` Gotcha 46) uses to reach these same settings from
    /// the `agent` tool and from a background subagent launch. Cleared
    /// before constructing a fresh `AppModel` so the assertion cannot pass
    /// merely because an earlier test in the same process happened to run
    /// first and install it.
    func testConstructingAnAppModelInstallsTheSubagentSamplingProvider() {
        AppToolRegistry.subagentSamplingOptionsProvider = nil
        let appModel = AppModel()
        appModel.temperature = 0.55

        guard let provider = AppToolRegistry.subagentSamplingOptionsProvider else {
            XCTFail(
                "AppModel.init() must install this provider, or every subagent spawned through "
                    + "the `agent` tool falls back to GenerateOptions()'s bare defaults")
            return
        }
        let options = provider()
        XCTAssertEqual(options.temperature, 0.55, "the provider must read the live setting, not a snapshot")
    }

    // MARK: - Per-chat overrides (`AppModel+Sampling.swift`)

    /// Appends a chat with the given override and returns it, so resolution
    /// by id reaches a real row.
    private func makeChat(_ appModel: AppModel, override: AppSamplingSettings?) -> AppChat {
        var chat = AppChat(title: "sampling")
        chat.samplingOverride = override
        appModel.chats.append(chat)
        return chat
    }

    /// An override must reach the engine WHOLE: every knob, including
    /// `maxNewTokens`, which the turn path used to read off the global
    /// property directly.
    func testASamplingOverrideReplacesTheAppWideSettingsForThatChat() {
        let appModel = AppModel()
        appModel.temperature = 0.77
        appModel.topKEnabled = true
        appModel.topK = 40
        appModel.topPEnabled = true
        appModel.topP = 0.88
        appModel.repetitionPenaltyEnabled = true
        appModel.repetitionPenalty = 1.15
        appModel.seedEnabled = true
        appModel.seed = 42
        appModel.stopSequences = "###, STOP"
        appModel.maxNewTokens = 2048

        let override = AppSamplingSettings(
            temperature: 1.4,
            topKEnabled: false,
            topK: 10,
            topPEnabled: false,
            topP: 0.5,
            maxNewTokens: 512,
            repetitionPenaltyEnabled: false,
            repetitionPenalty: 1.0,
            seedEnabled: true,
            seed: 7,
            stopSequences: "DONE")
        let chat = makeChat(appModel, override: override)

        let options = appModel.samplingOptions(chatID: chat.id)

        XCTAssertEqual(options.temperature, 1.4)
        XCTAssertEqual(options.topK, GenerateOptions().topK, "a disabled toggle must leave the engine default")
        XCTAssertEqual(options.topP, GenerateOptions().topP)
        XCTAssertEqual(options.repetitionPenalty, GenerateOptions().repetitionPenalty)
        XCTAssertEqual(options.seed, 7)
        XCTAssertEqual(options.stop, ["DONE"])
        XCTAssertEqual(options.maxNewTokens, 512)
    }

    func testAChatWithoutAnOverrideRunsOnTheAppWideSettings() {
        let appModel = AppModel()
        appModel.temperature = 0.66
        appModel.topKEnabled = true
        appModel.topK = 33
        appModel.topPEnabled = false
        appModel.repetitionPenaltyEnabled = true
        appModel.repetitionPenalty = 1.2
        appModel.seedEnabled = false
        appModel.seed = 5
        appModel.stopSequences = "HALT"
        appModel.maxNewTokens = 1024
        let chat = makeChat(appModel, override: nil)

        let options = appModel.samplingOptions(chatID: chat.id)

        XCTAssertEqual(options.temperature, 0.66)
        XCTAssertEqual(options.topK, 33)
        XCTAssertEqual(options.topP, GenerateOptions().topP)
        XCTAssertEqual(options.repetitionPenalty, 1.2)
        XCTAssertEqual(options.seed, GenerateOptions().seed)
        XCTAssertEqual(options.stop, ["HALT"])
        XCTAssertEqual(options.maxNewTokens, 1024)
    }

    /// The global builder must be untouched by overrides on ANY chat: the
    /// subagent provider reads it, and subagents deliberately run on the
    /// app-wide settings.
    func testTheGlobalBuilderIgnoresChatOverrides() {
        let appModel = AppModel()
        appModel.temperature = 0.44
        appModel.topKEnabled = true
        appModel.topK = 20
        appModel.maxNewTokens = 2048
        _ = makeChat(
            appModel,
            override: AppSamplingSettings(temperature: 1.9, maxNewTokens: 16384))

        let options = appModel.samplingOptions()

        XCTAssertEqual(options.temperature, 0.44)
        XCTAssertEqual(options.topK, 20)
        XCTAssertEqual(options.maxNewTokens, GenerateOptions().maxNewTokens)
    }

    func testRemovingAnOverrideReturnsTheChatToTheAppWideSettings() {
        let appModel = AppModel()
        appModel.temperature = 0.31
        appModel.topKEnabled = true
        appModel.topK = 90
        appModel.topPEnabled = true
        appModel.topP = 0.9
        appModel.repetitionPenaltyEnabled = false
        appModel.repetitionPenalty = 1.0
        appModel.seedEnabled = false
        appModel.seed = 0
        appModel.stopSequences = ""
        appModel.maxNewTokens = 2048
        let chat = makeChat(
            appModel, override: AppSamplingSettings(temperature: 1.75, maxNewTokens: 8192))

        appModel.removeChatSamplingOverride(id: chat.id)

        XCTAssertNil(appModel.chats.first(where: { $0.id == chat.id })?.samplingOverride)
        let options = appModel.samplingOptions(chatID: chat.id)
        XCTAssertEqual(options.temperature, 0.31)
        XCTAssertEqual(options.topK, 90)
        XCTAssertEqual(options.maxNewTokens, 2048)
    }
}
