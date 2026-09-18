import Foundation
import XCTest

@testable import TurboSparkApp

/// Tests for the typed `@scheme:value` context-reference grammar: which
/// reserved words and scheme prefixes parse, what stays a plain path or
/// plain prose, and the two Hermes fuzz rules (trailing punctuation strips
/// from values; an invalid line range means the full file). Pure functions
/// only -- no model, no I/O.
final class ContextReferenceParserTests: XCTestCase {

    private func parse(_ token: String, quoted: Bool = false) -> ContextReference? {
        ContextReferenceParser.parse(token: token, quoted: quoted)?.reference
    }

    // MARK: - Reserved git words

    func testDiffAndStagedAreReservedWords() {
        XCTAssertEqual(parse("diff"), .diff)
        XCTAssertEqual(parse("staged"), .staged)
        XCTAssertTrue(
            ContextReferenceParser.parse(token: "diff", quoted: false)!.isExplicit,
            "reserved words are explicit scheme references")
    }

    func testTrailingPunctuationDoesNotHideTheReservedWord() {
        // The common prose position: "check @diff, and @staged."
        XCTAssertEqual(parse("diff,"), .diff)
        XCTAssertEqual(parse("diff."), .diff)
        XCTAssertEqual(parse("staged;"), .staged)
    }

    func testQuotedDiffIsAPathNotTheScheme() {
        // Quotes are the escape hatch for a file literally named `diff`.
        let parsed = ContextReferenceParser.parse(token: "diff", quoted: true)
        XCTAssertEqual(parsed?.reference, .path(text: "diff", lines: nil))
        XCTAssertFalse(parsed!.isExplicit, "a quoted token is a path, with the path's silence")
    }

    // MARK: - @git:N

    func testGitCountParsesAndClamps() {
        XCTAssertEqual(parse("git:3"), .git(count: 3))
        XCTAssertEqual(parse("git:15"), .git(count: ContextReferenceParser.gitCommitLimit))
        XCTAssertEqual(parse("git:0"), .git(count: 1))
    }

    func testGitWithoutANumberIsProse() {
        XCTAssertNil(parse("git:"))
        XCTAssertNil(parse("git:abc"))
    }

    // MARK: - @url:

    func testHTTPAndHTTPSURLsParse() throws {
        XCTAssertEqual(parse("url:https://example.com/a"), .url(try XCTUnwrap(URL(string: "https://example.com/a"))))
        XCTAssertEqual(parse("url:http://example.com"), .url(try XCTUnwrap(URL(string: "http://example.com"))))
    }

    func testTrailingPunctuationStripsFromURLValues() throws {
        // "summarize @url:https://example.com." must not send the dot to the
        // fetcher as part of the host.
        XCTAssertEqual(
            parse("url:https://example.com."),
            .url(try XCTUnwrap(URL(string: "https://example.com"))))
    }

    func testNonHTTPSchemesAndEmptyURLsAreProse() {
        XCTAssertNil(parse("url:ftp://example.com"))
        XCTAssertNil(parse("url:"))
        XCTAssertNil(parse("url:not a url"))
    }

    // MARK: - @file: / @folder:

    func testFilePrefixCarriesAnExplicitPath() {
        let parsed = ContextReferenceParser.parse(token: "file:src/main.py", quoted: false)
        XCTAssertEqual(parsed?.reference, .path(text: "src/main.py", lines: nil))
        XCTAssertEqual(parsed?.isExplicit, true)
        XCTAssertNil(parsed?.unslicedPath)
    }

    func testFolderPrefixCarriesAnExplicitPathWithoutRanges() {
        // A folder has no lines: the colon suffix is part of the path, and
        // resolution decides it does not exist.
        let parsed = ContextReferenceParser.parse(token: "folder:src:5", quoted: false)
        XCTAssertEqual(parsed?.reference, .path(text: "src:5", lines: nil))
    }

    // MARK: - Line ranges

    func testLineRangesParseOneIndexedInclusive() {
        XCTAssertEqual(parse("file:main.py:42"), .path(text: "main.py", lines: 42...42))
        XCTAssertEqual(parse("file:main.py:10-25"), .path(text: "main.py", lines: 10...25))
        XCTAssertEqual(parse("main.py:10-25"), .path(text: "main.py", lines: 10...25))
    }

    func testInvalidRangesFallBackToTheFullPath() {
        // Range-shaped but invalid: the range drops, the path before the
        // colon stays (Hermes: invalid ranges silently return the full file).
        XCTAssertEqual(parse("file:main.py:0"), .path(text: "main.py", lines: nil))
        XCTAssertEqual(parse("file:main.py:25-10"), .path(text: "main.py", lines: nil))
        XCTAssertEqual(parse("file:main.py:10-25-30"), .path(text: "main.py", lines: nil))
        XCTAssertEqual(parse("file:main.py:x"), .path(text: "main.py:x", lines: nil),
            "a suffix that does not look like a range at all is part of the path")
    }

    func testARangeStrippedPathKeepsTheRawValueAsFallback() {
        // A file literally named `notes:2024` must still be mentionable:
        // resolution prefers the raw token when the sliced path is absent.
        let parsed = ContextReferenceParser.parse(token: "notes:2024", quoted: false)
        XCTAssertEqual(parsed?.reference, .path(text: "notes", lines: 2024...2024))
        XCTAssertEqual(parsed?.unslicedPath, "notes:2024")
    }

    func testRangeWithTrailingPunctuationStripsFirst() {
        XCTAssertEqual(parse("file:main.py:10-25."), .path(text: "main.py", lines: 10...25))
    }

    // MARK: - Bare paths

    func testBareTokenIsAQuietPath() {
        let parsed = ContextReferenceParser.parse(token: "src/a.rs", quoted: false)
        XCTAssertEqual(parsed?.reference, .path(text: "src/a.rs", lines: nil))
        XCTAssertEqual(parsed?.isExplicit, false)
    }

    func testBarePathPunctuationStrips() {
        XCTAssertEqual(parse("a.txt,"), .path(text: "a.txt", lines: nil))
    }

    func testEmptyTokensParseToNothing() {
        XCTAssertNil(ContextReferenceParser.parse(token: "", quoted: false))
        XCTAssertNil(ContextReferenceParser.parse(token: ",", quoted: false))
    }

    // MARK: - The grammar's own invariants

    func testSplitLineRangeRequiresADigitAfterTheColon() {
        // The discriminator the resolver relies on: `a:b` is a path, `a:1`
        // is a single-line slice.
        XCTAssertEqual(ContextReferenceParser.splitLineRange("a:b").path, "a:b")
        XCTAssertEqual(ContextReferenceParser.splitLineRange("a:1").lines, 1...1)
    }
}
