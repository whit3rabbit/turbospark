import XCTest
@testable import TurboSparkApp

/// Number keys on a focused AskUserQuestion card (component audit,
/// question). Options are numbered 1-9 straight through the card, so one key
/// names exactly one option even when the card holds several questions.
final class InteractiveQuestionKeysTests: XCTestCase {
    private typealias Keys = InteractiveQuestionKeys

    func testDigitsNumberOptionsThroughEveryQuestion() {
        let counts = [2, 3]
        XCTAssertEqual(Keys.target(for: "1", optionCounts: counts), .init(question: 0, option: 0))
        XCTAssertEqual(Keys.target(for: "2", optionCounts: counts), .init(question: 0, option: 1))
        XCTAssertEqual(Keys.target(for: "3", optionCounts: counts), .init(question: 1, option: 0))
        XCTAssertEqual(Keys.target(for: "5", optionCounts: counts), .init(question: 1, option: 2))
    }

    func testKeysPastTheLastOptionPickNothing() {
        XCTAssertNil(Keys.target(for: "6", optionCounts: [2, 3]))
        XCTAssertNil(Keys.target(for: "1", optionCounts: []))
        XCTAssertNil(Keys.target(for: "1", optionCounts: [0]))
    }

    func testOnlyAsciiOneToNineCount() {
        let counts = [9]
        XCTAssertNil(Keys.target(for: "0", optionCounts: counts))
        XCTAssertNil(Keys.target(for: "a", optionCounts: counts))
        XCTAssertNil(Keys.target(for: " ", optionCounts: counts))
        // Full-width and Arabic-Indic digits carry a whole-number value, but
        // the hint glyph on each row is an ASCII digit, so only that key acts.
        XCTAssertNil(Keys.target(for: "\u{FF11}", optionCounts: counts))
        XCTAssertNil(Keys.target(for: "\u{0661}", optionCounts: counts))
        XCTAssertEqual(Keys.target(for: "9", optionCounts: counts), .init(question: 0, option: 8))
    }

    func testAQuestionWithNoOptionsIsSkippedInTheNumbering() {
        XCTAssertEqual(Keys.target(for: "1", optionCounts: [0, 2]), .init(question: 1, option: 0))
        XCTAssertEqual(Keys.number(question: 1, option: 1, optionCounts: [0, 2]), 2)
    }

    func testNumberIsTheInverseAndStopsAtNine() {
        let counts = [4, 4, 4]
        for question in counts.indices {
            for option in 0..<counts[question] {
                let flat = question * 4 + option + 1
                let number = Keys.number(question: question, option: option, optionCounts: counts)
                if flat <= 9 {
                    XCTAssertEqual(number, flat)
                    let key = Character(String(flat))
                    XCTAssertEqual(
                        Keys.target(for: key, optionCounts: counts),
                        .init(question: question, option: option))
                } else {
                    XCTAssertNil(number, "option \(flat) has no key")
                }
            }
        }
        XCTAssertNil(Keys.number(question: 5, option: 0, optionCounts: counts))
        XCTAssertNil(Keys.number(question: 0, option: 7, optionCounts: counts))
    }
}
