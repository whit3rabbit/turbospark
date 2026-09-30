import Foundation
import XCTest

@testable import TurboSparkApp

final class ChatRuntimeSettingsUITests: XCTestCase {
    @MainActor
    func testControlsReadValuesFromTheAppModelStore() {
        let model = AppModel()
        model.microcompactEnabled = false
        model.autoContinuationEnabled = false
        model.instructionPinningEnabled = false
        model.microcompactMinimumSavingsTokens = 1_536
        model.instructionPinTokenCeiling = 1_024

        let controls = ChatRuntimeSettingsBindings(model: model)

        XCTAssertFalse(controls.microcompactEnabled.wrappedValue)
        XCTAssertFalse(controls.autoContinuationEnabled.wrappedValue)
        XCTAssertFalse(controls.instructionPinningEnabled.wrappedValue)
        XCTAssertEqual(controls.microcompactMinimumSavingsTokens.wrappedValue, 1_536)
        XCTAssertEqual(controls.instructionPinTokenCeiling.wrappedValue, 1_024)
    }

    @MainActor
    func testControlEditsWriteBackToTheAppModelStore() {
        let model = AppModel()
        let controls = ChatRuntimeSettingsBindings(model: model)

        controls.microcompactEnabled.wrappedValue = false
        controls.autoContinuationEnabled.wrappedValue = false
        controls.instructionPinningEnabled.wrappedValue = false
        controls.microcompactMinimumSavingsTokens.wrappedValue = 2_048
        controls.instructionPinTokenCeiling.wrappedValue = 1_536

        XCTAssertFalse(model.microcompactEnabled)
        XCTAssertFalse(model.autoContinuationEnabled)
        XCTAssertFalse(model.instructionPinningEnabled)
        XCTAssertEqual(model.microcompactMinimumSavingsTokens, 2_048)
        XCTAssertEqual(model.instructionPinTokenCeiling, 1_536)
    }

    @MainActor
    func testIntegerControlsClampEditsToTheirSupportedRanges() {
        let model = AppModel()
        let controls = ChatRuntimeSettingsBindings(model: model)

        controls.microcompactMinimumSavingsTokens.wrappedValue = 0
        controls.instructionPinTokenCeiling.wrappedValue = 9_999

        XCTAssertEqual(model.microcompactMinimumSavingsTokens, 64)
        XCTAssertEqual(model.instructionPinTokenCeiling, 2_048)
    }

    func testBoundedControlLabelsExposeTheirRanges() {
        let titles = Set(SettingsControlCatalog.entries.map(\.title))

        XCTAssertTrue(titles.contains("Minimum token savings (64-4,096 tokens)"))
        XCTAssertTrue(titles.contains("Maximum pinned instruction tokens (0-2,048 tokens)"))
    }

    @MainActor
    func testControlsRoundTripLoadedAndEditedValuesThroughPersistedSettings() throws {
        let settingsFile = AppStorageRoot.file("settings.json")
        let key = try XCTUnwrap(ProfileRepository.protectedRecordKey(for: settingsFile))
        let previousRecord = try ProfileRepository.shared.rawRecord(key: key)
        let previousData = try? Data(contentsOf: settingsFile)
        defer {
            if let previousData {
                try? previousData.write(to: settingsFile, options: .atomic)
            } else {
                try? FileManager.default.removeItem(at: settingsFile)
            }
            if let previousRecord {
                try? ProfileRepository.shared.saveRawRecord(previousRecord, key: key)
            } else {
                try? ProfileRepository.shared.deleteRecord(key: key)
            }
        }

        var seed = MacAppSettings()
        seed.microcompactEnabled = false
        seed.autoContinuationEnabled = true
        seed.instructionPinningEnabled = false
        seed.microcompactMinimumSavingsTokens = 1_536
        seed.instructionPinTokenCeiling = 1_024
        MacAppSettingsFileStore.save(seed)

        let loadedModel = AppModel()
        let loadedControls = ChatRuntimeSettingsBindings(model: loadedModel)
        XCTAssertFalse(loadedControls.microcompactEnabled.wrappedValue)
        XCTAssertTrue(loadedControls.autoContinuationEnabled.wrappedValue)
        XCTAssertFalse(loadedControls.instructionPinningEnabled.wrappedValue)
        XCTAssertEqual(loadedControls.microcompactMinimumSavingsTokens.wrappedValue, 1_536)
        XCTAssertEqual(loadedControls.instructionPinTokenCeiling.wrappedValue, 1_024)

        loadedControls.microcompactEnabled.wrappedValue = true
        loadedControls.autoContinuationEnabled.wrappedValue = false
        loadedControls.instructionPinningEnabled.wrappedValue = true
        loadedControls.microcompactMinimumSavingsTokens.wrappedValue = 4_096
        loadedControls.instructionPinTokenCeiling.wrappedValue = 0
        loadedModel.persistSettings()

        let reloadedModel = AppModel()
        let reloadedControls = ChatRuntimeSettingsBindings(model: reloadedModel)
        XCTAssertTrue(reloadedControls.microcompactEnabled.wrappedValue)
        XCTAssertFalse(reloadedControls.autoContinuationEnabled.wrappedValue)
        XCTAssertTrue(reloadedControls.instructionPinningEnabled.wrappedValue)
        XCTAssertEqual(reloadedControls.microcompactMinimumSavingsTokens.wrappedValue, 4_096)
        XCTAssertEqual(reloadedControls.instructionPinTokenCeiling.wrappedValue, 0)
    }
}
