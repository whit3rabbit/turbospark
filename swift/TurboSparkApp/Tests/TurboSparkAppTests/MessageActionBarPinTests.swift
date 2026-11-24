import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Which message keeps its action bar visible without hover (component
/// audit, message-actions): the newest assistant reply, and only when it
/// answers the newest prompt.
final class MessageActionBarPinTests: XCTestCase {
    private func user(_ text: String = "q", synthetic: Bool = false) -> AppChatMessage {
        AppChatMessage(role: .user, content: text, isSynthetic: synthetic)
    }

    private func assistant(_ text: String, tools: Bool = false) -> AppChatMessage {
        AppChatMessage(
            role: .assistant, content: text,
            toolCalls: tools ? [AppToolCall(name: "Read", arguments: ["path": "a"], category: .fileRead)] : [])
    }

    func testPinsTheNewestReply() {
        let old = assistant("first")
        let reply = assistant("second")
        let messages = [user(), old, user(), reply]
        XCTAssertEqual(MessageActionBarPin.pinnedMessageID(in: messages, isRunning: false), reply.id)
    }

    func testSkipsTrailingRowsWithNoText() {
        let reply = assistant("answer")
        let messages = [user(), reply, AppChatMessage(role: .tool, content: "out"), assistant("  \n")]
        XCTAssertEqual(MessageActionBarPin.pinnedMessageID(in: messages, isRunning: false), reply.id)
    }

    func testToolStepWithTextCounts() {
        let step = assistant("Let me check.", tools: true)
        XCTAssertEqual(MessageActionBarPin.pinnedMessageID(in: [user(), step], isRunning: false), step.id)
    }

    func testNothingWhileATurnRuns() {
        XCTAssertNil(MessageActionBarPin.pinnedMessageID(in: [user(), assistant("a")], isRunning: true))
    }

    func testAnUnansweredPromptPinsNothing() {
        // The previous turn's reply must not look like the answer to the
        // newest prompt.
        XCTAssertNil(MessageActionBarPin.pinnedMessageID(
            in: [user(), assistant("a"), user()], isRunning: false))
        XCTAssertNil(MessageActionBarPin.pinnedMessageID(in: [], isRunning: false))
    }

    func testSyntheticPromptsDoNotEndTheTurn() {
        let reply = assistant("a")
        XCTAssertEqual(
            MessageActionBarPin.pinnedMessageID(
                in: [user(), reply, user("goal check", synthetic: true)], isRunning: false),
            reply.id)
    }
}
