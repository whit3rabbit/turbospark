import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The memory load guardrails: persistence tolerance, and the mapping onto the
/// binding's own enum.
///
/// See `docs/LOAD_GUARD.md` for the policy. These cases exist for two hazards
/// that are specific to this layer rather than to the engine.
final class LoadGuardSettingsTests: XCTestCase {
    /// **The hazard `swift/CLAUDE.md` Gotcha 13 records, applied to the two
    /// fields this feature added.** A non-optional field on the synthesized
    /// decoder throws `keyNotFound` against a settings file written before it
    /// existed, `load()` swallows the error, and the user's whole settings
    /// object silently reverts to defaults. That has already happened once in
    /// this app, to the chat archive.
    ///
    /// Decoding a payload with NEITHER key must therefore succeed and land on
    /// the documented defaults.
    func testSettingsWrittenBeforeTheseFieldsExistedStillDecode() throws {
        let legacy = #"{"contextTokens":0,"temperature":0.2}"#
        let data = Data(legacy.utf8)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.loadGuard, "relaxed")
        XCTAssertEqual(decoded.loadGuardCustomBytes, 0)
        XCTAssertEqual(decoded.minAutoContextTokens, 0)
    }

    func testAllThreeFieldsRoundTrip() throws {
        let settings = MacAppSettings(
            loadGuard: "strict",
            loadGuardCustomBytes: 8 << 30,
            minAutoContextTokens: 16384
        )
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.loadGuard, "strict")
        XCTAssertEqual(decoded.loadGuardCustomBytes, 8 << 30)
        XCTAssertEqual(decoded.minAutoContextTokens, 16384)
    }

    /// `relaxed` is what shipped before the setting existed, so a fresh
    /// install must behave exactly as the previous release did.
    func testTheDefaultIsRelaxedWithNoFloor() {
        let settings = MacAppSettings()
        XCTAssertEqual(settings.loadGuard, "relaxed")
        XCTAssertEqual(settings.minAutoContextTokens, 0)
        XCTAssertEqual(AppRuntimeOptions().loadGuard, .relaxed)
        XCTAssertEqual(AppRuntimeOptions().minAutoContextTokens, 0)
    }

    /// Every stored spelling maps to a tier. An unrecognized one falls back to
    /// `relaxed` rather than to the strictest tier, which would silently start
    /// refusing models a user had been loading.
    func testEveryStoredSpellingResolves() {
        for option in AppLoadGuardOption.allCases {
            XCTAssertEqual(AppLoadGuardOption(rawValue: option.rawValue), option)
        }
        XCTAssertNil(AppLoadGuardOption(rawValue: "strcit"))
    }

    /// **A custom tier with no ceiling set must not become a ceiling of
    /// nothing**, which would refuse every model. Zero is "not configured",
    /// and the only honest reading of it is the default tier.
    func testACustomTierWithoutACeilingFallsBackRatherThanRefusingEverything() {
        let options = AppRuntimeOptions(loadGuard: .custom, loadGuardCustomBytes: 0)
        let resolved = options.loadGuard.loadGuard(customBytes: options.loadGuardCustomBytes)
        // The enum is not Equatable, so compare through the encoded form the
        // FFI will actually receive -- which is also what the engine parses.
        XCTAssertEqual(try encoded(resolved), "\"relaxed\"")

        let configured = AppLoadGuardOption.custom.loadGuard(customBytes: 4 << 30)
        XCTAssertEqual(try encoded(configured), "4294967296")
    }

    /// The four tier words reach the wire as the spellings the engine's own
    /// parser accepts. A drift here is a silently ignored setting: the FFI
    /// refuses an unknown string, so this is what keeps that refusal from
    /// being the first time anyone finds out.
    func testTierWordsEncodeToTheSpellingsTheEngineParses() throws {
        let expected: [(AppLoadGuardOption, String)] = [
            (.off, "\"off\""),
            (.relaxed, "\"relaxed\""),
            (.balanced, "\"balanced\""),
            (.strict, "\"strict\""),
        ]
        for (option, wire) in expected {
            XCTAssertEqual(try encoded(option.loadGuard(customBytes: 0)), wire)
        }
    }

    private func encoded(_ guard_: OpenOptions.LoadGuard) throws -> String {
        let data = try JSONEncoder().encode(GuardBox(value: guard_))
        let text = String(decoding: data, as: UTF8.self)
        // Unwrap the single-key object the box adds.
        return String(text.dropFirst(#"{"value":"#.count).dropLast())
    }

    private struct GuardBox: Encodable {
        let value: OpenOptions.LoadGuard
    }
}
