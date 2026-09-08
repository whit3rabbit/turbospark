import XCTest
@testable import TurboSparkApp

/// Tests for the qwen-code parity modules: the `/copy` parser
/// (`CopyCommandParser`), the streaming loading phrases, the ANSI SGR
/// parser, turn grouping for transcript collapse, and the interactive
/// markdown table grid.
final class QwenParityModulesTests: XCTestCase {
    // MARK: - CopyCommandParser.parse

    func testParseBareCopyAndCodeTarget() {
        XCTAssertEqual(
            CopyCommandParser.parse("/copy"),
            CopyCommandParser.Request(target: .code(language: nil), index: 1))
        XCTAssertEqual(
            CopyCommandParser.parse("/copy code"),
            CopyCommandParser.Request(target: .code(language: nil), index: 1))
    }

    func testParseLanguageAndOrdinal() {
        XCTAssertEqual(
            CopyCommandParser.parse("/copy python"),
            CopyCommandParser.Request(target: .code(language: "python"), index: 1))
        XCTAssertEqual(
            CopyCommandParser.parse("/copy code python 2"),
            CopyCommandParser.Request(target: .code(language: "python"), index: 2))
        // An ordinal of zero or negative is not an ordinal: it is a file
        // named "0", which parses as the first block of that language.
        XCTAssertEqual(
            CopyCommandParser.parse("/copy rust 0"),
            CopyCommandParser.Request(target: .code(language: "rust"), index: 1))
    }

    func testParseLatexTargets() {
        XCTAssertEqual(
            CopyCommandParser.parse("/copy latex"),
            CopyCommandParser.Request(target: .latex, index: 1))
        XCTAssertEqual(
            CopyCommandParser.parse("/copy latex 3"),
            CopyCommandParser.Request(target: .latex, index: 3))
        XCTAssertEqual(
            CopyCommandParser.parse("/copy inline-latex"),
            CopyCommandParser.Request(target: .inlineLatex, index: 1))
        XCTAssertEqual(
            CopyCommandParser.parse("/copy inline-latex 2"),
            CopyCommandParser.Request(target: .inlineLatex, index: 2))
    }

    func testParseRejectsNonCopyDrafts() {
        XCTAssertNil(CopyCommandParser.parse("/stats"))
        XCTAssertNil(CopyCommandParser.parse("copy this"))
        XCTAssertNil(CopyCommandParser.parse(""))
    }

    // MARK: - CopyCommandParser.fencedCodeBlocks

    func testFencedBlocksCaptureLanguageAndBody() {
        let message = """
        Here is one:

        ```swift
        let a = 1
        ```

        And python:

        ```python
        print("hi")
        print("bye")
        ```
        """
        let blocks = CopyCommandParser.fencedCodeBlocks(in: message)
        XCTAssertEqual(blocks.count, 2)
        XCTAssertEqual(blocks[0].language, "swift")
        XCTAssertEqual(blocks[0].code, "let a = 1\n")
        XCTAssertEqual(blocks[1].language, "python")
        XCTAssertEqual(blocks[1].code, "print(\"hi\")\nprint(\"bye\")\n")
    }

    func testUnterminatedFenceProducesNothing() {
        // A fence the model never closed: no block, never a half-copy. Any
        // later ``` would have closed this one, so "unterminated" means
        // "no exact-fence line anywhere after the opener" -- prose that
        // merely mentions ``` mid-line is not a fence.
        let message = """
        ```python
        never closed
        still inside the fence
        prose mentioning ``` mid-line is not a fence
        """
        XCTAssertTrue(CopyCommandParser.fencedCodeBlocks(in: message).isEmpty)
    }

    func testLongerClosingFenceClosesShorterOpening() {
        let blocks = CopyCommandParser.fencedCodeBlocks(in: "```\nx\n`````\ny\n```\n")
        XCTAssertEqual(blocks.count, 1)
        XCTAssertEqual(blocks[0].code, "x\n")
    }

    // MARK: - CopyCommandParser.resolve

    func testResolveCopiesRequestedBlock() {
        let message = "```swift\nlet a = 1\n```\n```swift\nlet b = 2\n```\n"
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy", in: message),
            .copied("let a = 1\n"))
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy swift 2", in: message),
            .copied("let b = 2\n"))
    }

    func testResolveReportsOutOfRangeAndEmpty() {
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy", in: "no code here"),
            .nothingFound("no code block"))
        let message = "```swift\nlet a = 1\n```\n"
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy swift 5", in: message),
            .indexOutOfRange(requested: 5, available: 1, noun: "code block"))
    }

    func testResolveLatexAndInline() {
        let message = "Euler: $e^{i\\pi} + 1 = 0$ and $$\\int_0^1 x\\,dx = \\tfrac12$$"
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy latex", in: message),
            .copied("\\int_0^1 x\\,dx = \\tfrac12"))
        XCTAssertEqual(
            CopyCommandParser.resolve(draft: "/copy inline-latex", in: message),
            .copied("e^{i\\pi} + 1 = 0"))
    }

    func testInlineLatexSkipsCurrency() {
        // "$5 and $6" is dollars, not math: the space after the opening
        // dollar and before the closing one disqualifies both pairs.
        let spans = CopyCommandParser.inlineLatexSpans(in: "it costs $5 and $6 today")
        XCTAssertTrue(spans.isEmpty)
        let real = CopyCommandParser.inlineLatexSpans(in: "so $x^2$ holds")
        XCTAssertEqual(real, ["x^2"])
    }

    // MARK: - LoadingPhrases

    func testLoadingPhrasesRotateEveryInterval() {
        XCTAssertEqual(
            LoadingPhrases.phrase(elapsed: 0),
            LoadingPhrases.phrase(elapsed: 14.9))
        XCTAssertNotEqual(
            LoadingPhrases.phrase(elapsed: 0),
            LoadingPhrases.phrase(elapsed: LoadingPhrases.rotationInterval))
    }

    func testLoadingPhrasesWrapAroundAndNeverEmpty() {
        let atEnd = LoadingPhrases.phrase(elapsed: LoadingPhrases.rotationInterval * Double(LoadingPhrases.all.count))
        XCTAssertEqual(atEnd, LoadingPhrases.phrase(elapsed: 0))
        for elapsed in stride(from: 0.0, through: 400, by: 7) {
            XCTAssertFalse(LoadingPhrases.phrase(elapsed: elapsed).isEmpty)
        }
    }

    // MARK: - ANSIColorizer

    func testPlainTextPassesThroughAsOneSegment() {
        XCTAssertEqual(ANSIColorizer.segments(in: "hello"), [ANSIColorizer.Segment(text: "hello")])
        XCTAssertEqual(ANSIColorizer.stripped("hello"), "hello")
    }

    func testForegroundColorCodesStyleTheirRange() {
        var red = ANSIColorizer.Segment(text: "bad", foreground: .red)
        red.bold = true
        let segments = ANSIColorizer.segments(in: "\u{1B}[31;1mbad\u{1B}[0m ok")
        XCTAssertEqual(segments.count, 2)
        XCTAssertEqual(segments[0], red)
        XCTAssertEqual(segments[1].text, " ok")
        XCTAssertTrue(segments[1].isPlain)
    }

    func testResetRestoresPlainAndAdjacentRunsMerge() {
        let segments = ANSIColorizer.segments(
            in: "\u{1B}[32mgreen\u{1B}[0mplain1plain2\u{1B}[32mgreen2\u{1B}[0m")
        // green | merged plain | green again
        XCTAssertEqual(segments.count, 3)
        XCTAssertEqual(segments[1].text, "plain1plain2")
        XCTAssertEqual(segments[2].text, "green2")
    }

    func testBrightAndDefaultCodes() {
        let segments = ANSIColorizer.segments(in: "\u{1B}[90mgray\u{1B}[39mback")
        XCTAssertEqual(segments[0].foreground, .brightBlack)
        XCTAssertEqual(segments[1].foreground, .default)
    }

    func test256ColorMapsCubeAndGrayscale() {
        XCTAssertEqual(ANSIColorizer.paletteFor256(1), .red)
        XCTAssertEqual(ANSIColorizer.paletteFor256(196), .brightRed) // full red cube cell
        XCTAssertEqual(ANSIColorizer.paletteFor256(232), .black)
        XCTAssertEqual(ANSIColorizer.paletteFor256(255), .white)
        XCTAssertEqual(ANSIColorizer.paletteFor256(999), .default)
    }

    func testNonSGREscapesAreStrippedNotRendered() {
        let text = "a\u{1B}[2Kb\u{1B}]0;title\u{07}c"
        XCTAssertEqual(ANSIColorizer.stripped(text), "abc")
    }

    func testTruncatedEscapeDropsIntroducer() {
        // A dangling ESC-[ introducer is consumed whole, exactly what the
        // output formatter's own stripper does.
        XCTAssertEqual(ANSIColorizer.stripped("end\u{1B}["), "end")
        XCTAssertEqual(ANSIColorizer.stripped("x\u{1B}"), "x")
    }

    func testStrippedRemovesEverythingStyled() {
        let raw = "\u{1B}[1;33mwarn\u{1B}[0m: \u{1B}[36mhttp://x\u{1B}[0m done"
        XCTAssertEqual(ANSIColorizer.stripped(raw), "warn: http://x done")
    }

    // MARK: - TurnCollapseModel

    private func user(_ content: String) -> AppChatMessage {
        AppChatMessage(role: .user, content: content)
    }

    private func assistant(_ content: String) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: content)
    }

    func testUserMessagesOpenTurnsAndCarryTheirRows() {
        let u1 = user("one"), a1 = assistant("r1"), u2 = user("two"), a2 = assistant("r2")
        let turns = TurnCollapseModel.turns(in: [u1, a1, u2, a2])
        XCTAssertEqual(turns.count, 2)
        XCTAssertEqual(turns[0].anchorID, u1.id)
        XCTAssertEqual(turns[0].messageIDs, [u1.id, a1.id])
        XCTAssertEqual(turns[1].messageIDs, [u2.id, a2.id])
    }

    func testLeadingNonUserRowsFormHeadlessTurn() {
        let a = assistant("synthetic"), u = user("q")
        let turns = TurnCollapseModel.turns(in: [a, u])
        XCTAssertEqual(turns.count, 2)
        XCTAssertNil(turns[0].anchorID)
        XCTAssertFalse(turns[0].isCollapsible)
        XCTAssertEqual(turns[1].anchorID, u.id)
    }

    func testCollapsedTurnShowsPromptAndFinalAnswerOnly() {
        let u = user("q")
        var toolWithCalls = assistant("")
        toolWithCalls.toolCalls = [AppToolCall(name: "Bash", arguments: [:])]
        let partial = assistant("half...")
        let final = assistant("done")
        let messages = [u, toolWithCalls, partial, final]
        let turn = TurnCollapseModel.turns(in: messages)[0]
        let byID = Dictionary(uniqueKeysWithValues: messages.map { ($0.id, $0) })

        XCTAssertEqual(TurnCollapseModel.visibleIDs(collapsedFor: turn, messagesByID: byID), [u.id, final.id])
        XCTAssertEqual(TurnCollapseModel.hiddenCount(for: turn, messagesByID: byID), 2)
    }

    func testSingleMessageTurnCannotHideAnything() {
        let u = user("q")
        let turn = TurnCollapseModel.turns(in: [u])[0]
        XCTAssertEqual(TurnCollapseModel.visibleIDs(collapsedFor: turn, messagesByID: [u.id: u]), [u.id])
        XCTAssertEqual(TurnCollapseModel.hiddenCount(for: turn, messagesByID: [u.id: u]), 0)
    }

    // MARK: - MarkdownTableGrid

    func testSegmentsSplitProseFromTables() {
        let markdown = """
        intro text

        | a | b |
        |---|---|
        | 1 | 2 |
        | 3 | 4 |

        outro
        """
        let segments = MarkdownTableGrid.segments(in: markdown)
        XCTAssertEqual(segments.count, 3)
        guard case .markdown(let prose) = segments[0] else { return XCTFail("expected prose") }
        XCTAssertTrue(prose.contains("intro"))
        guard case .table(let grid) = segments[1] else { return XCTFail("expected table") }
        XCTAssertEqual(grid.header, ["a", "b"])
        XCTAssertEqual(grid.rows, [["1", "2"], ["3", "4"]])
        guard case .markdown(let outro) = segments[2] else { return XCTFail("expected outro") }
        XCTAssertEqual(outro, "\noutro")
    }

    func testEscapedPipeStaysInsideItsCell() {
        let grid = MarkdownTableGrid.parse([
            "| a | b |",
            "|---|---|",
            "| x\\|y | 2 |",
        ])
        XCTAssertEqual(grid?.rows[0], ["x|y", "2"])
    }

    func testRaggedRowsArePaddedNotTruncated() {
        let grid = MarkdownTableGrid.parse([
            "| a | b | c |",
            "|---|---|---|",
            "| 1 | 2 |",
        ])
        XCTAssertEqual(grid?.rows[0].count, 3)
        XCTAssertEqual(grid?.rows[0][2], "")
    }

    func testSortIsNumericWhenPossibleAndStableInHeader() {
        let grid = MarkdownTableGrid(header: ["n", "name"], rows: [
            ["10", "j"], ["9", "k"], ["100", "l"],
        ])
        let ascending = grid.sorted(byColumn: 0, direction: .ascending)
        XCTAssertEqual(ascending.rows.map { $0[0] }, ["9", "10", "100"])
        XCTAssertEqual(ascending.header, grid.header)
        let descending = grid.sorted(byColumn: 1, direction: .descending)
        XCTAssertEqual(descending.rows.map { $0[1] }, ["l", "k", "j"])
    }

    func testOutOfRangeSortColumnIsIdentity() {
        let grid = MarkdownTableGrid(header: ["a"], rows: [["1"], ["2"]])
        XCTAssertEqual(grid.sorted(byColumn: 5, direction: .descending), grid)
    }

    func testSerializers() {
        let grid = MarkdownTableGrid(header: ["a", "b"], rows: [["1", "2"], ["3", "4"]])
        XCTAssertEqual(grid.tsv(), "a\tb\n1\t2\n3\t4")
        XCTAssertEqual(grid.csv(), "a,b\n1,2\n3,4")
        XCTAssertTrue(grid.markdown.hasPrefix("| a | b |"))
    }

    func testCSVQuotesCommasAndEmbeddedQuotes() {
        let grid = MarkdownTableGrid(header: ["x"], rows: [["say, \"hi\""]])
        XCTAssertEqual(grid.csv(), "x\n\"say, \"\"hi\"\"\"")
    }

    func testNonTablePipeLineIsNotATable() {
        // A pipe-delimited line with no separator line under it is prose.
        let segments = MarkdownTableGrid.segments(in: "| just | text |")
        guard case .markdown = segments[0] else { return XCTFail("expected prose") }
    }
}
