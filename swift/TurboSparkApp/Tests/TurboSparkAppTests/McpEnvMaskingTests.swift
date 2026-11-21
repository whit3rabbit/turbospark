import XCTest
@testable import TurboSparkApp

final class McpEnvMaskingTests: XCTestCase {
    private let mask = McpEnvMasking.mask

    // MARK: Fixed-width mask

    func testMaskIsFixedWidthWhateverTheValueLength() {
        XCTAssertEqual(McpEnvMasking.maskedValue("abc"), mask)
        XCTAssertEqual(McpEnvMasking.maskedValue(String(repeating: "x", count: 200)), mask)
        XCTAssertFalse(mask.isEmpty)
    }

    func testEmptyValueHasNothingToMask() {
        XCTAssertEqual(McpEnvMasking.maskedValue(""), "")
    }

    // MARK: Scrubbing

    func testScrubReplacesEveryOccurrenceOfAnEnvValue() {
        let text = "npx server --token sk-12345 --again sk-12345"
        XCTAssertEqual(
            McpEnvMasking.scrub(text, secrets: ["sk-12345"]),
            "npx server --token \(mask) --again \(mask)")
    }

    func testShortAndEmptyValuesAreLeftAlone() {
        // "1", "true" and "" would otherwise blank out ordinary words.
        let text = "server exited with 1: true"
        XCTAssertEqual(McpEnvMasking.scrub(text, secrets: ["", "1", "true", "12345"]), text)
    }

    func testSixCharacterValueIsTheThreshold() {
        XCTAssertEqual(McpEnvMasking.scrub("key abcdef", secrets: ["abcdef"]), "key \(mask)")
        XCTAssertEqual(McpEnvMasking.scrub("key abcde", secrets: ["abcde"]), "key abcde")
    }

    func testValueThatIsASubstringOfAnotherLeavesNoTail() {
        // Shorter-first would mask "secret12" and leave "3456" of the longer one.
        let text = "auth=secret123456"
        XCTAssertEqual(
            McpEnvMasking.scrub(text, secrets: ["secret12", "secret123456"]),
            "auth=\(mask)")
        XCTAssertEqual(
            McpEnvMasking.scrub(text, secrets: ["secret123456", "secret12"]),
            "auth=\(mask)")
    }

    func testOverlappingValuesAreMaskedAsOneRun() {
        // "abcdef" and "defghi" overlap on "def": masking one first would
        // break the other's match and leak its tail.
        XCTAssertEqual(
            McpEnvMasking.scrub("x abcdefghi y", secrets: ["abcdef", "defghi"]),
            "x \(mask) y")
    }

    func testSelfOverlappingOccurrencesAreAllCovered() {
        XCTAssertEqual(
            McpEnvMasking.scrub("aaaaaaaa", secrets: ["aaaaaa"]),
            mask)
    }

    func testTextWithoutSecretsIsUnchanged() {
        XCTAssertEqual(McpEnvMasking.scrub("plain text", secrets: []), "plain text")
        XCTAssertEqual(McpEnvMasking.scrub("", secrets: ["abcdefgh"]), "")
    }

    func testNonASCIIValuesAreMatched() {
        XCTAssertEqual(
            McpEnvMasking.scrub("pw: p\u{e4}ssw\u{f6}rt!", secrets: ["p\u{e4}ssw\u{f6}rt"]),
            "pw: \(mask)!")
    }

    func testStdioServerSecretsAreItsEnvValuesOnly() {
        let config = McpServerConfig(
            name: "demo",
            transport: .stdio(
                command: "npx", args: ["--token", "tok-abcdef"],
                env: ["API_KEY": "tok-abcdef", "DEBUG": "1"], cwd: nil, envPassthrough: []))
        XCTAssertEqual(Set(config.envSecretValues), ["tok-abcdef", "1"])
        XCTAssertEqual(
            McpEnvMasking.scrub(config.commandSummary, secrets: config.envSecretValues),
            "npx --token \(mask)")

        let remote = McpServerConfig(
            name: "remote", transport: .sse(url: URL(string: "https://example.com/sse")!))
        XCTAssertEqual(remote.envSecretValues, [])
    }

    // MARK: Editor rows

    func testRowsRoundTripTheEditorText() {
        let text = "API_KEY=abc=def\nDEBUG=1"
        let rows = McpEnvRows.parse(text)
        XCTAssertEqual(rows, [
            McpEnvRows.Row(key: "API_KEY", value: "abc=def"),
            McpEnvRows.Row(key: "DEBUG", value: "1"),
        ])
        XCTAssertEqual(McpEnvRows.serialize(rows), text)
    }

    func testParseSkipsBlankLinesAndKeepsKeyOnlyLines() {
        XCTAssertEqual(McpEnvRows.parse("\n  \nNAME\n"), [McpEnvRows.Row(key: "NAME", value: "")])
    }

    func testValuesKeepSpacesWhileTyping() {
        // Trimming on every keystroke would eat the space before the next word.
        let rows = McpEnvRows.parse("K=a ")
        XCTAssertEqual(rows.first?.value, "a ")
    }

    func testBlankRowSurvivesARoundTrip() {
        // A freshly added row must not vanish before the user types into it.
        let rows = McpEnvRows.parse(McpEnvRows.serialize([McpEnvRows.Row(key: "", value: "")]))
        XCTAssertEqual(rows, [McpEnvRows.Row(key: "", value: "")])
    }

    func testSerializeCannotSmuggleASeparatorThroughAKey() {
        let rows = [McpEnvRows.Row(key: "A=B\nC", value: "v\nw")]
        XCTAssertEqual(McpEnvRows.parse(McpEnvRows.serialize(rows)), [McpEnvRows.Row(key: "ABC", value: "vw")])
    }
}
