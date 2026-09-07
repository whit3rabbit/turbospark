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
}
