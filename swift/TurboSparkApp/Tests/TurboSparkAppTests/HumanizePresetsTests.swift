import XCTest
@testable import TurboSparkApp

final class HumanizePresetsTests: XCTestCase {
    func testAddingKeepsTheUsersOwnLinesAndOrderAndLeavesOtherFieldsAlone() {
        var rules = HumanizeRules()
        rules.avoidTerms = "my word"
        rules.protectedTerms = "Skills Manager"
        rules.fixedSeed = true
        let next = rules.adding(.plainEnglish)
        XCTAssertEqual(HumanizeRules.lines(next.avoidTerms).first, "my word")
        XCTAssertTrue(HumanizeRules.lines(next.avoidTerms).contains("delve"))
        XCTAssertEqual(next.protectedTerms, "Skills Manager")
        XCTAssertTrue(next.fixedSeed)
        XCTAssertEqual(next.seed, rules.seed)
    }

    func testApplyingTwiceChangesNothingAndIsReportedAsContained() {
        for preset in HumanizePreset.allCases {
            let once = HumanizeRules().adding(preset)
            XCTAssertTrue(once.contains(preset))
            XCTAssertEqual(once.adding(preset), once, "\(preset) must be idempotent")
            XCTAssertFalse(HumanizeRules().contains(preset))
        }
    }

    func testPresetsMergeWithoutDroppingEachOthersLines() {
        let both = HumanizeRules().adding(.plainEnglish).adding(.business)
        XCTAssertTrue(both.contains(.plainEnglish) && both.contains(.business))
    }

    func testEveryReplacementLineParsesAndNoPresetFlagsItsOwnOutput() {
        for preset in HumanizePreset.allCases {
            var rules = HumanizeRules()
            rules.replacements = preset.replacements.joined(separator: "\n")
            XCTAssertEqual(rules.replacementPairs.count, preset.replacements.count,
                           "a malformed line would be silently ignored: \(preset)")
            // A rewrite that already follows the preset must pass its own
            // avoid list, or the preset would flag its own replacements.
            let outputs = rules.replacementPairs.map(\.new).joined(separator: " ")
            rules.avoidTerms = preset.avoidTerms.joined(separator: "\n")
            XCTAssertTrue(rules.check(draft: "", rewrite: outputs).avoidFound.isEmpty, "\(preset)")
        }
    }

    func testReplacementsRewriteTheFinishedTextAsDescribed() {
        let casual = HumanizeRules().adding(.casual)
        XCTAssertEqual(casual.applyReplacements(to: "However, we left. Furthermore, it rained."),
                       "But we left. Also, it rained.")
        let plain = HumanizeRules().adding(.plainEnglish)
        XCTAssertEqual(plain.applyReplacements(to: "In order to utilize it, wait."), "To use it, wait.")
    }
}
