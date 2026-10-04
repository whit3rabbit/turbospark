import XCTest
import TurboSpark
@testable import TurboSparkApp

final class AppTitleGenerationTests: XCTestCase {
    func testUsableOutputUsesSmallPlainTextLocalCompletion() async {
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: "Explain how a local session model works.",
            completion: { messages, options in
                XCTAssertEqual(messages.map(\.role), [.system, .user])
                XCTAssertTrue(messages[0].content.localizedCaseInsensitiveContains("plain text"))
                XCTAssertEqual(messages[1].content, "Explain how a local session model works.")
                XCTAssertEqual(options.reasoning, .off)
                XCTAssertEqual(options.temperature, 0.2)
                XCTAssertEqual(options.maxNewTokens, 64)
                return "  Local session models  "
            })

        XCTAssertEqual(title, "Local session models")
    }

    func testEmptyOutputIsRejected() async {
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: "Explain local sessions.",
            completion: { _, _ in " \n\t  " })

        XCTAssertNil(title)
    }

    func testMultilineOutputUsesFirstNonemptyLineAsOneLine() async {
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: "Explain local sessions.",
            completion: { _, _ in " \n  Local   session\tmodels \nExtra explanation" })

        XCTAssertEqual(title, "Local session models")
    }

    func testVerboseOutputIsCappedAtEightyCharacters() async {
        let firstLine = String(repeating: "Title ", count: 20)
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: "Explain local sessions.",
            completion: { _, _ in "\(firstLine)\nIgnore this explanation" })

        XCTAssertEqual(title?.count, 80)
        let normalizedFirstLine = firstLine.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        XCTAssertEqual(title, String(normalizedFirstLine.prefix(80)))
    }

    func testNormalizedWholeMessageEchoIsRejected() async {
        let message = "Explain   local sessions\nwith a short example."
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: message,
            completion: { _, _ in " explain local SESSIONS with a short example. " })

        XCTAssertNil(title)
    }

    func testExactMultilineWholeMessageEchoIsRejected() async {
        let message = "Explain local sessions\nwith a short example."
        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: message,
            completion: { _, _ in message })

        XCTAssertNil(title)
    }

    func testSidecarFailureReturnsNil() async {
        enum SidecarFailure: Error { case unavailable }

        let title = await AppTitleGeneration.generateTitle(
            firstUserMessage: "Explain local sessions.",
            completion: { _, _ in throw SidecarFailure.unavailable })

        XCTAssertNil(title)
    }
}
