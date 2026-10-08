import XCTest

@testable import TurboSpark

/// Runs through the real archive against the committed ChatML fixture, so it
/// pins the header and the Rust side together, with no model install.
final class TokenizerTests: XCTestCase {
    private var fixture: String {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("crates/tokenizer/tests/fixtures/ChatMLTokenizer").path
    }

    func testCountTokenizeAndDetokenizeAgree() throws {
        let tokenizer = try TurboSparkTokenizer(modelPath: fixture)
        let text = "Hello, world! This is a tokenizer test."
        let ids = try tokenizer.tokenize(text)
        XCTAssertFalse(ids.isEmpty)
        XCTAssertEqual(try tokenizer.count(text), ids.count)
        XCTAssertEqual(try tokenizer.detokenize(ids), text)
    }

    func testMissingDirectoryThrowsAndNamesTheProblem() {
        XCTAssertThrowsError(try TurboSparkTokenizer(modelPath: "/definitely/not/a/model")) { error in
            XCTAssertTrue((error as? TurboSparkError)?.message.contains("not a directory") == true)
        }
    }

    func testUseAfterCloseThrowsInsteadOfCrashing() throws {
        let tokenizer = try TurboSparkTokenizer(modelPath: fixture)
        tokenizer.close()
        tokenizer.close()
        XCTAssertThrowsError(try tokenizer.count("x"))
    }
}
