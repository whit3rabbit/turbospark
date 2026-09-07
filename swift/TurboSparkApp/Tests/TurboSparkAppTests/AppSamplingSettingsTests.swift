import Foundation
import XCTest

@testable import TurboSparkApp

/// `AppSamplingSettings` and `AppSamplingPreset` (the per-chat sampling
/// override and the preset registry over it): decode tolerance, the
/// decode-side clamps, and the model-side CRUD.
///
/// The clamps are the point of several of these cases: two of the struct's
/// fields become `UInt32` at the use site, and a plain conversion TRAPS
/// rather than erroring (state#35's shape), so a hand-edited archive or
/// settings file must be bounded before it reaches one.
@MainActor
final class AppSamplingSettingsTests: XCTestCase {
    // MARK: - Decode-side clamps

    func testDecodedValuesOutsideTheControlRangesAreClamped() throws {
        let json = """
            {"temperature": 99, "topK": 4294967296, "topP": 0,
             "maxNewTokens": 99999999999, "repetitionPenalty": 42}
            """
        let settings = try JSONDecoder().decode(AppSamplingSettings.self, from: Data(json.utf8))

        XCTAssertEqual(settings.temperature, 2.0)
        XCTAssertEqual(settings.topK, 256)
        XCTAssertEqual(settings.topP, 0.01)
        XCTAssertEqual(settings.maxNewTokens, 16384)
        XCTAssertEqual(settings.repetitionPenalty, 2.0)
    }

    /// `topK: 0` stays legal on purpose: the engine reads 0 as "off", and
    /// forcing the UI minimum would silently turn a hand-set off into k=1,
    /// which is near-greedy.
    func testTopKZeroSurvivesTheClamp() throws {
        let json = #"{"topK": 0}"#
        let settings = try JSONDecoder().decode(AppSamplingSettings.self, from: Data(json.utf8))
        XCTAssertEqual(settings.topK, 0)
    }

    // MARK: - Archive tolerance (`AppChat.samplingOverride`)

    /// An archive written before the field existed must decode, override
    /// nil, everything else intact -- the systemPrompt incident's exact
    /// shape, guarded from the other side.
    func testAnArchiveWithoutTheOverrideKeyDecodesWithNilOverride() throws {
        let chatID = UUID()
        let json = """
            {"selectedChatID":"\(chatID.uuidString)","chats":[{"id":"\(chatID.uuidString)",
              "title":"t","messages":[]}]}
            """
        let archive = try JSONDecoder().decode(AppChatArchive.self, from: Data(json.utf8))

        XCTAssertEqual(archive.chats.first?.title, "t")
        XCTAssertNil(archive.chats.first?.samplingOverride)
    }

    /// A wrong-typed override costs the OVERRIDE, not the chat: a throw
    /// here would fail the row, and `decodeLossyArray` at the archive level
    /// would drop the whole conversation over one hand-edited key.
    func testAWrongTypedOverrideCostsTheOverrideNotTheChat() throws {
        let chatID = UUID()
        let json = """
            {"selectedChatID":"\(chatID.uuidString)","chats":[{"id":"\(chatID.uuidString)",
              "title":"t","messages":[],"samplingOverride": 5}]}
            """
        let archive = try JSONDecoder().decode(AppChatArchive.self, from: Data(json.utf8))

        XCTAssertEqual(archive.chats.count, 1)
        XCTAssertNil(archive.chats.first?.samplingOverride)
    }

    func testAnOverrideRoundTripsThroughTheArchive() throws {
        var chat = AppChat(title: "sampling")
        chat.samplingOverride = AppSamplingSettings(temperature: 1.1, maxNewTokens: 512)
        let archive = AppChatArchive(selectedChatID: chat.id, chats: [chat])

        let data = try JSONEncoder().encode(archive)
        let decoded = try JSONDecoder().decode(AppChatArchive.self, from: data)

        XCTAssertEqual(decoded.chats.first?.samplingOverride?.temperature, 1.1)
        XCTAssertEqual(decoded.chats.first?.samplingOverride?.maxNewTokens, 512)
    }

    // MARK: - Presets

    func testAPresetWithMissingFieldsDecodesToDefaults() throws {
        let json = #"{"id":"\#(UUID().uuidString)"}"#
        let preset = try JSONDecoder().decode(AppSamplingPreset.self, from: Data(json.utf8))

        XCTAssertEqual(preset.name, "Preset")
        XCTAssertEqual(preset.settings, AppSamplingSettings())
    }

    func testSamplingPresetsRoundTripThroughSettings() throws {
        var settings = MacAppSettings()
        settings.samplingPresets = [
            AppSamplingPreset(name: "Creative", settings: AppSamplingSettings(temperature: 1.3))
        ]

        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertEqual(decoded.samplingPresets.first?.name, "Creative")
        XCTAssertEqual(decoded.samplingPresets.first?.settings.temperature, 1.3)
    }

    /// A settings.json written before the field existed decodes to an empty
    /// registry and leaves its neighbours alone.
    func testSettingsWithoutThePresetKeyDecodeToAnEmptyRegistry() throws {
        let json = #"{"temperature": 0.5}"#
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))

        XCTAssertTrue(decoded.samplingPresets.isEmpty)
        XCTAssertEqual(decoded.temperature, 0.5)
    }

    // MARK: - Model CRUD (`AppModel+Sampling.swift`)

    func testUpsertAndDeletePresetRoundTripThroughTheSettingsFile() {
        let appModel = AppModel()
        let preset = AppSamplingPreset(
            name: "Precise", settings: AppSamplingSettings(temperature: 0.05))

        appModel.upsertSamplingPreset(preset)
        XCTAssertEqual(appModel.samplingPresets, [preset])
        // upsert persists immediately, so the file already carries it.
        XCTAssertEqual(MacAppSettingsFileStore.load().samplingPresets, [preset])

        var edited = preset
        edited.settings.temperature = 0.1
        appModel.upsertSamplingPreset(edited)
        XCTAssertEqual(appModel.samplingPresets, [edited], "upsert must key on id, not append")

        appModel.deleteSamplingPreset(edited.id)
        XCTAssertTrue(appModel.samplingPresets.isEmpty)
        XCTAssertTrue(MacAppSettingsFileStore.load().samplingPresets.isEmpty)
    }

    /// The selected chat can be an unmaterialized DRAFT (no row in `chats`);
    /// an explicit sampling edit must promote the row rather than vanish
    /// into the by-id guard.
    func testSettingAnOverrideOnADraftChatMaterializesTheRow() {
        let appModel = AppModel()
        appModel.chats = []
        appModel.selectedChatID = UUID()
        XCTAssertNil(appModel.selectedChatIndex)

        let settings = AppSamplingSettings(temperature: 0.9)
        appModel.setChatSamplingOverride(id: appModel.selectedChatID, settings: settings)

        XCTAssertEqual(appModel.chats.count, 1)
        XCTAssertEqual(appModel.chats.first?.samplingOverride, settings)
    }

    func testSettingAndRemovingAnOverrideOnARealChat() {
        let appModel = AppModel()
        appModel.chats = []
        appModel.selectedChatID = UUID()
        let id = appModel.selectedChatID
        appModel.setChatSamplingOverride(id: id, settings: AppSamplingSettings(temperature: 0.9))
        XCTAssertEqual(appModel.chats.first?.samplingOverride?.temperature, 0.9)

        appModel.removeChatSamplingOverride(id: id)
        XCTAssertNil(appModel.chats.first?.samplingOverride)
    }

    /// A chat with no override, an unknown id, and no id at all must all
    /// resolve identically to the app-wide snapshot: "no override" has one
    /// answer however the question is asked.
    func testEffectiveSamplingFallsBackToTheAppWideSnapshot() {
        let appModel = AppModel()
        appModel.temperature = 0.62
        appModel.chats = []
        appModel.selectedChatID = UUID()

        XCTAssertEqual(
            appModel.effectiveSamplingSettings(chatID: appModel.selectedChatID),
            appModel.globalSamplingSettings())
        XCTAssertEqual(
            appModel.effectiveSamplingSettings(chatID: UUID()),
            appModel.globalSamplingSettings())
        XCTAssertEqual(
            appModel.effectiveSamplingSettings(chatID: nil),
            appModel.globalSamplingSettings())
    }
}
