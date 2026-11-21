import XCTest
@testable import TurboSparkApp

final class InstructionTokenCaptionTests: XCTestCase {
    func testEmptyOrWhitespaceTextHasNoReading() {
        XCTAssertNil(InstructionTokenCaption.reading(text: "", measuredTokens: nil, windowTokens: nil))
        XCTAssertNil(InstructionTokenCaption.reading(text: " \n\t", measuredTokens: 12, windowTokens: 4096))
    }

    func testMeasuredCountWithWindowIsExactWithPercent() throws {
        let reading = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "Be terse.", measuredTokens: 410, windowTokens: 4096))
        XCTAssertEqual(reading.tokens, 410)
        XCTAssertFalse(reading.isEstimate)
        XCTAssertEqual(reading.percent, 10)
        XCTAssertFalse(reading.exceedsWarning)
        XCTAssertEqual(
            InstructionTokenCaption.text(for: reading, language: .english),
            "410 tokens, 10% of the context window")
    }

    func testNoSessionFallsBackToLabeledEstimateWithoutPercent() throws {
        // 10 characters at four per token, rounded up.
        let reading = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "0123456789", measuredTokens: nil, windowTokens: nil))
        XCTAssertEqual(reading.tokens, 3)
        XCTAssertTrue(reading.isEstimate)
        XCTAssertNil(reading.percent)
        XCTAssertFalse(reading.exceedsWarning)
        XCTAssertEqual(
            InstructionTokenCaption.text(for: reading, language: .english),
            "About 3 tokens (estimate)")
    }

    func testEstimateWithKnownWindowStillSaysEstimate() throws {
        let reading = try XCTUnwrap(
            InstructionTokenCaption.reading(
                text: String(repeating: "a", count: 400), measuredTokens: nil, windowTokens: 1000))
        XCTAssertEqual(reading.tokens, 100)
        XCTAssertEqual(reading.percent, 10)
        XCTAssertEqual(
            InstructionTokenCaption.text(for: reading, language: .english),
            "About 100 tokens (estimate), 10% of the context window")
    }

    func testMeasuredWithoutWindowHasNoPercent() throws {
        let reading = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "x", measuredTokens: 7, windowTokens: 0))
        XCTAssertNil(reading.percent)
        XCTAssertEqual(InstructionTokenCaption.text(for: reading, language: .english), "7 tokens")
    }

    func testWarningOnlyPastAQuarterOfTheWindow() throws {
        let atQuarter = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "x", measuredTokens: 1024, windowTokens: 4096))
        XCTAssertFalse(atQuarter.exceedsWarning)
        let past = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "x", measuredTokens: 1025, windowTokens: 4096))
        XCTAssertTrue(past.exceedsWarning)
        XCTAssertEqual(past.percent, 25)
    }

    func testNegativeMeasurementIsIgnoredForEstimate() throws {
        let reading = try XCTUnwrap(
            InstructionTokenCaption.reading(text: "abcd", measuredTokens: -1, windowTokens: 4096))
        XCTAssertTrue(reading.isEstimate)
        XCTAssertEqual(reading.tokens, 1)
    }

    func testCaptionKeysAreInTheCatalog() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Localization/Localizable.xcstrings")
        let json = try XCTUnwrap(
            try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let strings = try XCTUnwrap(json["strings"] as? [String: Any])
        for key in InstructionTokenCaption.catalogKeys {
            XCTAssertNotNil(strings[key], key)
        }
    }
}
