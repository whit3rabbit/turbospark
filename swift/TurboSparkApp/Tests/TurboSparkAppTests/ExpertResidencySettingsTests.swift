import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

final class ExpertResidencySettingsTests: XCTestCase {
    func testResidencySettingDefaultsToAutoAndRoundTrips() throws {
        var settings = MacAppSettings()
        XCTAssertEqual(settings.expertResidency, "auto")

        settings.expertResidency = "mapped"
        let data = try JSONEncoder().encode(settings)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(restored.expertResidency, "mapped")
    }

    func testOlderSettingsWithoutTheResidencyKeyUseAuto() throws {
        let legacy = Data(#"{"contextTokens":0,"temperature":0.2}"#.utf8)
        let restored = try JSONDecoder().decode(MacAppSettings.self, from: legacy)
        XCTAssertEqual(restored.expertResidency, "auto")
    }

    func testOptionMapsToTheOpenOptionSpelling() throws {
        XCTAssertNil(AppExpertResidencyOption(rawValue: "mmap"))
        // The engine parses these exact strings (`expertResidency`).
        for option in AppExpertResidencyOption.allCases {
            let encoded = try JSONEncoder().encode(OpenOptions(expertResidency: option.requested))
            let json = try XCTUnwrap(String(data: encoded, encoding: .utf8))
            XCTAssertTrue(json.contains(#""expertResidency":"\#(option.rawValue)""#), json)
        }
    }

    private func preflight(applicable: Bool = true, resolved: String?, refusal: String? = nil)
        -> ExpertResidencyPreflight
    {
        ExpertResidencyPreflight(
            applicable: applicable, requested: "mapped", resolved: resolved, refusal: refusal)
    }

    /// Mapped is one global setting, so it must never reach an open that
    /// would refuse it: it is sent only where the install's pre-open answer
    /// resolves mapped, and auto everywhere else.
    func testMappedIsSentOnlyWhereTheOpenWouldTakeIt() {
        let mapped = AppExpertResidencyOption.mapped
        XCTAssertEqual(mapped.openOption(preflight: preflight(resolved: "mapped")), .mapped)
        XCTAssertNil(mapped.fallback(preflight: preflight(resolved: "mapped")))

        let refused = preflight(resolved: nil, refusal: "mapped expert residency is not wired for qwen4exp")
        XCTAssertEqual(mapped.openOption(preflight: refused), .auto)
        XCTAssertEqual(
            mapped.fallback(preflight: refused),
            .refused("mapped expert residency is not wired for qwen4exp"))

        let dense = preflight(applicable: false, resolved: nil)
        XCTAssertEqual(mapped.openOption(preflight: dense), .auto)
        XCTAssertEqual(mapped.fallback(preflight: dense), .noRoutedExperts)

        XCTAssertEqual(mapped.openOption(preflight: nil), .auto)
        XCTAssertEqual(mapped.fallback(preflight: nil), .unreadable)
    }

    func testStreamedAndAutoAreNeverRefused() {
        let streamed = AppExpertResidencyOption.streamed
        XCTAssertEqual(streamed.openOption(preflight: preflight(resolved: "streamed")), .streamed)
        // Unreadable facts still send streamed: the open never refuses it.
        XCTAssertEqual(streamed.openOption(preflight: nil), .streamed)
        XCTAssertEqual(streamed.openOption(preflight: preflight(applicable: false, resolved: nil)), .auto)
        XCTAssertNil(streamed.fallback(preflight: nil))

        XCTAssertEqual(AppExpertResidencyOption.auto.openOption(preflight: nil), .auto)
        XCTAssertNil(AppExpertResidencyOption.auto.fallback(preflight: nil))
    }

    func testPreflightDecodesTheEngineShape() throws {
        let json = Data(#"{"applicable":true,"requested":"mapped","resolved":null,"refusal":"too big"}"#.utf8)
        let decoded = try JSONDecoder().decode(ExpertResidencyPreflight.self, from: json)
        XCTAssertEqual(decoded, preflight(resolved: nil, refusal: "too big"))
    }

    func testAdaptiveSpeculationSendsTheAdaptiveSpelling() throws {
        XCTAssertEqual(AppSpeculationOption(rawValue: "adaptive"), .adaptive)
        let encoded = try JSONEncoder().encode(OpenOptions(speculation: AppSpeculationOption.adaptive.speculation))
        let json = try XCTUnwrap(String(data: encoded, encoding: .utf8))
        XCTAssertTrue(json.contains(#""speculation":"adaptive""#), json)
    }
}
