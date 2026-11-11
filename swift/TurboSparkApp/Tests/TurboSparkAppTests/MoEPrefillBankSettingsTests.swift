import Foundation
import XCTest

@testable import TurboSparkApp

final class MoEPrefillBankSettingsTests: XCTestCase {
    func testPreferenceDefaultsOffAndRoundTrips() throws {
        XCTAssertFalse(MacAppSettings().moePrefillBank)
        let legacy = try JSONDecoder().decode(MacAppSettings.self, from: Data("{}".utf8))
        XCTAssertFalse(legacy.moePrefillBank)

        let data = try JSONEncoder().encode(MacAppSettings(moePrefillBank: true))
        XCTAssertTrue(try JSONDecoder().decode(MacAppSettings.self, from: data).moePrefillBank)
    }

    @MainActor
    func testOpenOptionsCarryThePreferenceOnlyForEligibleInstalls() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: dir) }
        let flashNext = dir.appendingPathComponent("flash-next.gturbo")
        let otherMoE = dir.appendingPathComponent("other-moe.gturbo")
        for (install, family) in [(flashNext, "qwen4exp"), (otherMoE, "qwen35moe")] {
            try FileManager.default.createDirectory(at: install, withIntermediateDirectories: true)
            let manifest = Data(#"{"arch":{"family":"\#(family)","numExperts":288}}"#.utf8)
            try manifest.write(to: install.appendingPathComponent("manifest.json"))
        }

        let model = AppModel()
        model.runtimeOptions.moePrefillBank = false
        XCTAssertEqual(model.buildOpenOptions(modelPath: flashNext.path).moePrefillBank, false)
        model.runtimeOptions.moePrefillBank = true
        XCTAssertEqual(model.buildOpenOptions(modelPath: flashNext.path).moePrefillBank, true)
        XCTAssertNil(model.buildOpenOptions(modelPath: otherMoE.path).moePrefillBank)
        XCTAssertNil(model.buildOpenOptions(modelPath: dir.path).moePrefillBank)
    }
}
