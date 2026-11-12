import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

final class ExpertSlotsWiringSettingsTests: XCTestCase {
    func testWiringSettingDefaultsToAutoAndRoundTrips() throws {
        var settings = MacAppSettings()
        XCTAssertEqual(settings.expertSlotsWiring, "auto")

        settings.expertSlotsWiring = "off"
        let data = try JSONEncoder().encode(settings)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(restored.expertSlotsWiring, "off")
    }

    func testOlderSettingsWithoutTheWiringKeyUseAuto() throws {
        let legacy = Data(#"{"contextTokens":0,"temperature":0.2}"#.utf8)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: legacy)
        XCTAssertEqual(restored.expertSlotsWiring, "auto")
    }

    func testOptionMapsToTheOpenOptionSpelling() throws {
        XCTAssertEqual(AppExpertSlotsWiringOption(rawValue: "auto"), .auto)
        XCTAssertEqual(AppExpertSlotsWiringOption(rawValue: "on"), .on)
        XCTAssertEqual(AppExpertSlotsWiringOption(rawValue: "off"), .off)
        XCTAssertNil(AppExpertSlotsWiringOption(rawValue: "locked"))

        // The engine parses these exact strings (`expertSlotsWiring`).
        for option in AppExpertSlotsWiringOption.allCases {
            let encoded = try JSONEncoder().encode(OpenOptions(expertSlotsWiring: option.expertSlotsWiring))
            let json = try XCTUnwrap(String(data: encoded, encoding: .utf8))
            XCTAssertTrue(json.contains(#""expertSlotsWiring":"\#(option.rawValue)""#), json)
        }
    }

    func testSessionInfoDecodesTheWiringReport() throws {
        let json = Data(#"{"requested":"auto","active":true,"wiredBytes":13421772800,"reason":null}"#.utf8)
        let info = try JSONDecoder().decode(SessionInfo.ExpertSlotsWiring.self, from: json)
        XCTAssertTrue(info.active)
        XCTAssertEqual(info.wiredBytes, 13_421_772_800)
        XCTAssertNil(info.reason)
    }
}
