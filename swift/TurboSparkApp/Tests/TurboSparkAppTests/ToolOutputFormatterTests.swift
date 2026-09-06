import Foundation
import XCTest
@testable import TurboSparkApp

final class ToolOutputFormatterTests: XCTestCase {
    func testStripAnsiPlainString() {
        let input = "Hello, world! Normal output."
        let output = ToolOutputFormatter.stripAnsi(input)
        XCTAssertEqual(output, input)
    }

    func testStripAnsiColorCodes() {
        let input = "\u{1B}[32m[OK]\u{1B}[0m Build succeeded with \u{1B}[1;34m0 errors\u{1B}[0m."
        let output = ToolOutputFormatter.stripAnsi(input)
        XCTAssertEqual(output, "[OK] Build succeeded with 0 errors.")
    }

    func testStripAnsiExtendedColorCodes() {
        let input = "\u{1B}[38;2;255;100;50mCustom RGB\u{1B}[0m and \u{1B}[48;5;123m256 Color\u{1B}[m"
        let output = ToolOutputFormatter.stripAnsi(input)
        XCTAssertEqual(output, "Custom RGB and 256 Color")
    }

    func testStripAnsiCursorAndScreenCommands() {
        let input = "Compiling...\u{1B}[2K\rFinished [optimized] target(s)\u{1B}[1A"
        let output = ToolOutputFormatter.stripAnsi(input)
        XCTAssertEqual(output, "Compiling...\rFinished [optimized] target(s)")
    }

    func testStripAnsiOscSequences() {
        let input = "\u{1B}]0;Terminal Title\u{07}Command output"
        let output = ToolOutputFormatter.stripAnsi(input)
        XCTAssertEqual(output, "Command output")
    }

    func testTailOutputSmallText() {
        let input = "Line 1\nLine 2\nLine 3"
        let tail = ToolOutputFormatter.tailOutput(input, maxLines: 10, maxChars: 100)
        XCTAssertFalse(tail.isTruncated)
        XCTAssertEqual(tail.hiddenLineCount, 0)
        XCTAssertEqual(tail.hiddenCharCount, 0)
        XCTAssertEqual(tail.visibleText, input)
    }

    func testTailOutputExceedsLines() {
        let lines = (1...20).map { "Line \($0)" }
        let input = lines.joined(separator: "\n")
        let tail = ToolOutputFormatter.tailOutput(input, maxLines: 5, maxChars: 1000)
        XCTAssertTrue(tail.isTruncated)
        XCTAssertEqual(tail.hiddenLineCount, 15)
        XCTAssertEqual(tail.hiddenCharCount, 0)
        XCTAssertEqual(tail.visibleText, "Line 16\nLine 17\nLine 18\nLine 19\nLine 20")
    }

    func testTailOutputExceedsChars() {
        let input = String(repeating: "abcdefghij", count: 10) // 100 chars
        let tail = ToolOutputFormatter.tailOutput(input, maxLines: 10, maxChars: 30)
        XCTAssertTrue(tail.isTruncated)
        XCTAssertEqual(tail.hiddenCharCount, 70)
        XCTAssertEqual(tail.visibleText.count, 30)
    }

    func testTailOutputStripsAnsiFirst() {
        let input = "\u{1B}[32mLine 1\u{1B}[0m\n\u{1B}[31mLine 2\u{1B}[0m"
        let tail = ToolOutputFormatter.tailOutput(input, maxLines: 10, maxChars: 100)
        XCTAssertEqual(tail.visibleText, "Line 1\nLine 2")
    }
}
