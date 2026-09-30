import Foundation
import XCTest

@testable import TurboSparkApp

final class ChatRuntimeSettingsPersistenceTests: XCTestCase {
    func testRuntimeSettingsDefaultOnAndRoundTripIndependently() throws {
        var settings = MacAppSettings()

        XCTAssertTrue(settings.microcompactEnabled)
        XCTAssertTrue(settings.autoContinuationEnabled)
        XCTAssertTrue(settings.instructionPinningEnabled)
        XCTAssertEqual(settings.microcompactMinimumSavingsTokens, 512)
        XCTAssertEqual(settings.instructionPinTokenCeiling, 512)

        settings.microcompactEnabled = false
        settings.autoContinuationEnabled = true
        settings.instructionPinningEnabled = false
        settings.microcompactMinimumSavingsTokens = 1_024
        settings.instructionPinTokenCeiling = 128

        let data = try JSONEncoder().encode(settings)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertFalse(restored.microcompactEnabled)
        XCTAssertTrue(restored.autoContinuationEnabled)
        XCTAssertFalse(restored.instructionPinningEnabled)
        XCTAssertEqual(restored.microcompactMinimumSavingsTokens, 1_024)
        XCTAssertEqual(restored.instructionPinTokenCeiling, 128)

        var enabledWithoutInjection = restored
        enabledWithoutInjection.instructionPinningEnabled = true
        enabledWithoutInjection.instructionPinTokenCeiling = 0
        let zeroCeilingData = try JSONEncoder().encode(enabledWithoutInjection)
        let restoredZeroCeiling = try JSONDecoder().decode(
            MacAppSettings.self, from: zeroCeilingData)
        XCTAssertTrue(restoredZeroCeiling.instructionPinningEnabled)
        XCTAssertEqual(restoredZeroCeiling.instructionPinTokenCeiling, 0)
    }

    func testOlderSettingsWithoutChatRuntimeKeysUseDefaults() throws {
        let data = Data(#"{"temperature":0.4}"#.utf8)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertTrue(restored.microcompactEnabled)
        XCTAssertTrue(restored.autoContinuationEnabled)
        XCTAssertTrue(restored.instructionPinningEnabled)
        XCTAssertEqual(restored.microcompactMinimumSavingsTokens, 512)
        XCTAssertEqual(restored.instructionPinTokenCeiling, 512)
        XCTAssertEqual(restored.temperature, 0.4)
    }

    func testLoadingOutOfRangeValuesClampsAndSavesCorrectedSettings() throws {
        let settingsFile = AppStorageRoot.file("settings.json")
        let key = try XCTUnwrap(ProfileRepository.protectedRecordKey(for: settingsFile))
        let previousRecord = try ProfileRepository.shared.rawRecord(key: key)
        let previousData = try? Data(contentsOf: settingsFile)
        try ProfileRepository.shared.deleteRecord(key: key)
        defer {
            restoreSettings(previousData, at: settingsFile)
            if let previousRecord {
                try? ProfileRepository.shared.saveRawRecord(previousRecord, key: key)
            } else {
                try? ProfileRepository.shared.deleteRecord(key: key)
            }
        }

        var seed = MacAppSettings()
        seed.serverAutoStartOnLaunch = false
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(seed)) as? [String: Any])
        object["microcompactMinimumSavingsTokens"] = -10
        object["instructionPinTokenCeiling"] = 9_999
        let invalidData = try JSONSerialization.data(withJSONObject: object)
        try invalidData.write(to: settingsFile, options: .atomic)

        let loaded = MacAppSettingsFileStore.load()
        XCTAssertEqual(loaded.microcompactMinimumSavingsTokens, 64)
        XCTAssertEqual(loaded.instructionPinTokenCeiling, 2_048)
        let normalizedAfterReload = MacAppSettingsFileStore.load()
        XCTAssertEqual(normalizedAfterReload.microcompactMinimumSavingsTokens, 64)
        XCTAssertEqual(normalizedAfterReload.instructionPinTokenCeiling, 2_048)

        var invalidForSave = loaded
        invalidForSave.microcompactMinimumSavingsTokens = 9_999
        invalidForSave.instructionPinTokenCeiling = -10
        MacAppSettingsFileStore.save(invalidForSave)

        let saved = MacAppSettingsFileStore.load()
        XCTAssertEqual(saved.microcompactMinimumSavingsTokens, 4_096)
        XCTAssertEqual(saved.instructionPinTokenCeiling, 0)
        let correctedAfterSave = MacAppSettingsFileStore.load()
        XCTAssertEqual(correctedAfterSave.microcompactMinimumSavingsTokens, 4_096)
        XCTAssertEqual(correctedAfterSave.instructionPinTokenCeiling, 0)
    }

    @MainActor
    func testAppModelLoadsAndPersistsEachRuntimeSetting() throws {
        let settingsFile = AppStorageRoot.file("settings.json")
        let previousData = try? Data(contentsOf: settingsFile)
        defer { restoreSettings(previousData, at: settingsFile) }

        let original = MacAppSettingsFileStore.load()
        defer { MacAppSettingsFileStore.save(original) }
        var seed = original
        seed.serverAutoStartOnLaunch = false
        seed.microcompactEnabled = false
        seed.autoContinuationEnabled = true
        seed.instructionPinningEnabled = false
        seed.microcompactMinimumSavingsTokens = 1_536
        seed.instructionPinTokenCeiling = 0
        MacAppSettingsFileStore.save(seed)

        let model = AppModel()
        XCTAssertFalse(model.microcompactEnabled)
        XCTAssertTrue(model.autoContinuationEnabled)
        XCTAssertFalse(model.instructionPinningEnabled)
        XCTAssertEqual(model.microcompactMinimumSavingsTokens, 1_536)
        XCTAssertEqual(model.instructionPinTokenCeiling, 0)

        model.microcompactEnabled = true
        model.autoContinuationEnabled = false
        model.instructionPinningEnabled = true
        model.microcompactMinimumSavingsTokens = 9_999
        model.instructionPinTokenCeiling = -1
        model.persistSettings()

        XCTAssertEqual(model.microcompactMinimumSavingsTokens, 4_096)
        XCTAssertEqual(model.instructionPinTokenCeiling, 0)
        let restored = MacAppSettingsFileStore.load()
        XCTAssertTrue(restored.microcompactEnabled)
        XCTAssertFalse(restored.autoContinuationEnabled)
        XCTAssertTrue(restored.instructionPinningEnabled)
        XCTAssertEqual(restored.microcompactMinimumSavingsTokens, 4_096)
        XCTAssertEqual(restored.instructionPinTokenCeiling, 0)
    }

    private func restoreSettings(_ data: Data?, at file: URL) {
        if let data {
            try? data.write(to: file, options: .atomic)
        } else {
            try? FileManager.default.removeItem(at: file)
        }
    }
}
