import XCTest

@testable import TurboSpark

/// The live-stream binding's wire shape and handle rules. A real streaming
/// model is exercised by the Rust gate `coreml_parakeet_unified`.
final class AudioStreamTests: XCTestCase {
    func testUpdateDecodesTheNativeSnakeCaseShape() throws {
        let json = """
        {"appended_text":" world","closed_words":[{"text":"hello","start_seconds":0.08,"end_seconds":0.24}],
         "open_word":"world","received_seconds":1.5,"committed_through_seconds":0.4,"is_final":false}
        """
        let update = try audioDecode(AudioStreamUpdate.self, json)
        XCTAssertEqual(update.appendedText, " world")
        XCTAssertEqual(update.closedWords, [AudioStreamWord(text: "hello", startSeconds: 0.08, endSeconds: 0.24)])
        XCTAssertEqual(update.openWord, "world")
        XCTAssertEqual(update.committedThroughSeconds, 0.4)
        XCTAssertFalse(update.isFinal)
        let final = try audioDecode(AudioStreamUpdate.self, #"{"appended_text":"","closed_words":[],"open_word":null,"received_seconds":0,"committed_through_seconds":0,"is_final":true}"#)
        XCTAssertNil(final.openWord)
        XCTAssertTrue(final.isFinal)
    }

    func testMergedUpdatesReadAsOneCall() {
        let first = AudioStreamUpdate(appendedText: "hello", openWord: "hello", receivedSeconds: 2, committedThroughSeconds: 0.9)
        let second = AudioStreamUpdate(
            appendedText: " world", closedWords: [AudioStreamWord(text: "hello", startSeconds: 0, endSeconds: 0.5)],
            openWord: "world", receivedSeconds: 4, committedThroughSeconds: 2.9)
        let merged = first.merged(with: second)
        XCTAssertEqual(merged.appendedText, "hello world")
        XCTAssertEqual(merged.closedWords.map(\.text), ["hello"])
        XCTAssertEqual(merged.openWord, "world", "the later open word is current")
        XCTAssertEqual(merged.receivedSeconds, 4)
        XCTAssertEqual(merged.committedThroughSeconds, 2.9)
        XCTAssertFalse(merged.isFinal)
    }

    func testOpeningSomethingThatIsNotAStreamingInstallFails() {
        XCTAssertThrowsError(try AudioStreamSession(modelPath: "/missing/streaming/model"))
    }
}
