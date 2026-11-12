import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

final class AnePrefillSettingsTests: XCTestCase {
    func testAnePrefillSettingsDefaultAndRoundTrip() throws {
        var settings = MacAppSettings()
        XCTAssertEqual(settings.anePrefill, "auto")
        XCTAssertEqual(settings.anePrefillDir, "")

        settings.anePrefill = "on"
        settings.anePrefillDir = "/custom/ane/programs"

        let data = try JSONEncoder().encode(settings)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)

        XCTAssertEqual(restored.anePrefill, "on")
        XCTAssertEqual(restored.anePrefillDir, "/custom/ane/programs")
    }

    func testOlderSettingsWithoutAnePrefillKeysUseDefaults() throws {
        let legacy = Data(#"{"contextTokens":0,"temperature":0.2}"#.utf8)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: legacy)

        XCTAssertEqual(restored.anePrefill, "auto")
        XCTAssertEqual(restored.anePrefillDir, "")
    }

    func testAppAnePrefillOptionMapping() {
        XCTAssertEqual(AppAnePrefillOption(rawValue: "auto"), .auto)
        XCTAssertEqual(AppAnePrefillOption(rawValue: "on"), .on)
        XCTAssertEqual(AppAnePrefillOption(rawValue: "off"), .off)
        XCTAssertNil(AppAnePrefillOption(rawValue: "invalid"))

        XCTAssertEqual(AppAnePrefillOption.auto.anePrefill, .auto)
        XCTAssertEqual(AppAnePrefillOption.on.anePrefill, .split)
        XCTAssertEqual(AppAnePrefillOption.off.anePrefill, .off)
    }

    func testModelFeatureDescriptorExposesAnePrefillSupport() {
        let denseQwen = InstalledModel(
            alias: "qwen35-35b-dense",
            repo: "org/qwen35-35b-dense",
            path: "/path/to/qwen35",
            family: "qwen35"
        )
        let desc = ModelFeatureDescriptor.resolve(installedModel: denseQwen)
        // Supported only if the host hardware is M4+ and family is dense Qwen
        let expected = TurboSparkCapabilities.chipSupportsAnePrefill
        XCTAssertEqual(desc.supportsAnePrefill, expected)

        let nonDenseModel = InstalledModel(
            alias: "llama3-8b",
            repo: "org/llama3-8b",
            path: "/path/to/llama3",
            family: "llama"
        )
        let nonDenseDesc = ModelFeatureDescriptor.resolve(installedModel: nonDenseModel)
        XCTAssertFalse(nonDenseDesc.supportsAnePrefill)
    }
}
