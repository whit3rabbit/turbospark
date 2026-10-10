import CryptoKit
import XCTest
@testable import TurboSparkApp

final class HumanizeRulesTests: XCTestCase {
    private func found(_ ranges: [NSRange], in text: String) -> [String] {
        ranges.map { (text as NSString).substring(with: $0) }
    }

    func testLinesAreTrimmedDeduplicatedAndBlankLinesIgnored() {
        XCTAssertEqual(HumanizeRules.lines("  Alpha \n\n\tBeta\nAlpha\n "), ["Alpha", "Beta"])
    }

    func testReplacementsMatchWholeWordsExactlyInLineOrder() {
        var rules = HumanizeRules()
        rules.replacements = "utilize -> use\nnot a pair\n -> orphan\nuse -> apply"
        XCTAssertEqual(rules.replacementPairs.map(\.old), ["utilize", "use"])
        XCTAssertEqual(rules.applyReplacements(to: "We utilize tools. Utilize them. They utilized it."),
                       "We apply tools. Utilize them. They utilized it.",
                       "case-sensitive, whole words, and later pairs see earlier output")

        var literal = HumanizeRules()
        literal.replacements = "a.b -> $1"
        XCTAssertEqual(literal.applyReplacements(to: "a.b axb"), "$1 axb", "regex syntax is escaped both ways")
    }

    func testProtectedTermsMustSurviveExactlyAndOnlyWhenTheDraftHadThem() {
        var rules = HumanizeRules()
        rules.protectedTerms = "Skills Manager\nNorthstar\nAbsent Term"
        let findings = rules.check(
            draft: "Skills Manager ships with Northstar support.",
            rewrite: "The skills manager ships with Northstar support.")
        XCTAssertEqual(findings.missingProtected, ["Skills Manager"], "a changed case did not survive")
        XCTAssertFalse(findings.isEmpty)
    }

    func testWordsToAvoidMatchAnyCaseAsWholeWordsAndReportRanges() {
        var rules = HumanizeRules()
        rules.avoidTerms = "delve\nleverage\nunused"
        let rewrite = "Let us Delve in. We leverage it, delved deeper, and LEVERAGE more."
        let findings = rules.check(draft: "", rewrite: rewrite)
        XCTAssertEqual(findings.avoidFound, ["delve", "leverage"])
        XCTAssertEqual(found(findings.avoidRanges, in: rewrite), ["Delve", "leverage", "LEVERAGE"],
                       "sorted by position, and 'delved' is a different word")
    }

    func testHanTermsMatchWithoutWordBoundaries() {
        var rules = HumanizeRules()
        rules.avoidTerms = "\u{8D4B}\u{80FD}"
        let rewrite = "\u{6211}\u{4EEC}\u{8D4B}\u{80FD}\u{56E2}\u{961F}"
        XCTAssertEqual(found(rules.check(draft: "", rewrite: rewrite).avoidRanges, in: rewrite), ["\u{8D4B}\u{80FD}"])
    }

    func testRulesRoundTripThroughTheStoreAndABadValueStartsClean() throws {
        let defaults = try XCTUnwrap(UserDefaults(suiteName: "humanize.rules.\(UUID().uuidString)"))
        let store = HumanizeRulesStore(defaults: defaults)
        XCTAssertEqual(store.load(), HumanizeRules())
        var rules = HumanizeRules()
        rules.avoidTerms = "delve"
        rules.fixedSeed = true
        rules.seed = 7
        store.save(rules)
        XCTAssertEqual(store.load(), rules)

        defaults.set(Data("not json".utf8), forKey: HumanizeRulesStore.key)
        XCTAssertEqual(store.load(), HumanizeRules())
    }

    /// The panel shows a copy of core's instruction; this hashes it exactly
    /// as core's `wrapper_matches_the_publisher_fingerprint` does, so a typo
    /// or an upstream change cannot leave the display wrong.
    func testTheDisplayedInstructionMatchesCoresPromptFingerprint() {
        let prompt = HumanizeInstruction.text + "\n\n" + "X" + HumanizeInstruction.separator
        let hex = SHA256.hash(data: Data(prompt.utf8)).map { String(format: "%02x", $0) }.joined()
        XCTAssertTrue(hex.hasPrefix(HumanizeInstruction.fingerprint), hex)
    }
}
