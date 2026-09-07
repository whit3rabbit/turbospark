import XCTest

@testable import TurboSparkApp

/// `UnicodeSanitization` -- the invisible-character gate on prompt content.
///
/// The payloads here are the shapes the sanitizer exists for (the tag-char
/// hidden instruction from HackerOne #3086545, bidi overrides, zero-width
/// joiners) plus the two NO-OP cases that matter as much: plain ASCII and
/// ordinary CJK input with full-width punctuation must come back
/// byte-identical, because the NFKC pass behind the probe visibly rewrites
/// compatibility characters and the gate exists to keep it away from text
/// that never carried anything dangerous.
final class UnicodeSanitizationTests: XCTestCase {
    // MARK: - payloads are stripped

    func testTagCharactersCarryingHiddenTextAreStripped() {
        // U+E0042 U+E0055 U+E0052 U+E004E spell "BURN" inside the Unicode
        // tag block: invisible in every renderer, real token content to a
        // model.
        let hidden = "\u{E0042}\u{E0055}\u{E0052}\u{E004E}"
        let prompt = "Please review this file.\u{202B}\(hidden)\u{202C} Thanks!"
        let cleaned = UnicodeSanitization.sanitize(prompt)
        XCTAssertFalse(cleaned.contains(hidden))
        XCTAssertEqual(cleaned, "Please review this file. Thanks!")
    }

    func testBidiOverrideAndDirectionalIsolatesAreStripped() {
        let prompt = "safe\u{202E}gnp\u{202C}\u{2066}text\u{2069}"
        XCTAssertEqual(UnicodeSanitization.sanitize(prompt), "safegnptext")
    }

    func testZeroWidthAndPrivateUseCharactersAreStripped() {
        XCTAssertEqual(UnicodeSanitization.sanitize("a\u{200B}b\u{FEFF}c"), "abc")
        XCTAssertEqual(UnicodeSanitization.sanitize("a\u{E000}b"), "ab")
    }

    func testZeroWidthJoinerIsStrippedOnPromptsByDecision() {
        // U+200D is Cf, so it goes with the rest. On PROMPTS this is the
        // accepted tradeoff (the file's doc explains why tool output is
        // excluded); pinning it here means a future "fix" that spares ZWJ
        // has to argue with this test, not with a comment.
        XCTAssertEqual(UnicodeSanitization.sanitize("a\u{200D}b"), "ab")
    }

    // MARK: - the gate keeps clean text byte-identical

    func testPlainTextIsByteIdentical() {
        let prompt = "The quick brown fox; 123 -- not an em dash!"
        XCTAssertEqual(UnicodeSanitization.sanitize(prompt), prompt)
    }

    func testCJKInputWithFullWidthPunctuationIsByteIdentical() {
        // NFKC would rewrite every one of these (U+FF0C -> U+002C and
        // friends). The probe must keep the whole pipeline off for text
        // like this, which carries no invisible characters at all.
        let prompt = "你好，世界！这是一个测试。"
        XCTAssertEqual(UnicodeSanitization.sanitize(prompt), prompt)
    }

    func testCompatibilityCharactersAloneAreNotNormalized() {
        // A fullwidth Latin A with nothing dangerous present: the NFKC pass
        // never runs, so the character the user chose survives.
        XCTAssertEqual(UnicodeSanitization.sanitize("\u{FF21}"), "\u{FF21}")
    }

    func testNormalizationRunsOnlyWhenInvisibleCharactersArePresent() {
        // The same fullwidth A next to a zero-width space: now the
        // sanitizer is active, NFKC folds the A to ASCII as part of the
        // pass that removes the invisible character.
        let dirty = "\u{FF21}\u{200B}"
        XCTAssertEqual(UnicodeSanitization.sanitize(dirty), "A")
    }

    // MARK: - the probe

    func testProbeSeesWhatTheStripsTarget() {
        XCTAssertTrue("a\u{200B}b".hasInvisibleCharacters)
        XCTAssertTrue("a\u{202E}b".hasInvisibleCharacters)
        XCTAssertTrue("a\u{E0041}b".hasInvisibleCharacters)
        XCTAssertTrue("a\u{FEFF}b".hasInvisibleCharacters)
        XCTAssertFalse("你好，世界！".hasInvisibleCharacters)
        XCTAssertFalse("plain ascii".hasInvisibleCharacters)
    }
}
