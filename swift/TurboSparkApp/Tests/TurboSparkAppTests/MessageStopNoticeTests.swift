import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The transcript's per-message stop notice (`MessageStopNotice.resolve`).
/// A finished reply must draw nothing, so every normal-completion reason the
/// app persists is pinned to nil, and only the reasons that mean "this reply
/// is incomplete" map to a notice.
final class MessageStopNoticeTests: XCTestCase {
    private func assistant(_ reason: String?) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: "partial", stopReason: reason)
    }

    func testNormalCompletionDrawsNothing() {
        let normal: [String?] = [
            nil, "",
            GenerationResult.StopReason.endOfTurn.rawValue,
            GenerationResult.StopReason.eos.rawValue,
            GenerationResult.StopReason.stopString.rawValue,
            GenerationResult.StopReason.toolCalls.rawValue,
            // Intermediate rows of a turn that kept going.
            "tool_use", "guardrail_retry", "goal_check",
            // A finished subagent run.
            "completed",
            // The share export's live-snapshot marker, never a stored row.
            "partial",
        ]
        for reason in normal {
            XCTAssertNil(MessageStopNotice.resolve(assistant(reason)), "reason \(String(describing: reason))")
        }
    }

    func testUnknownReasonDrawsNothingRatherThanGuessing() {
        XCTAssertNil(MessageStopNotice.resolve(assistant(GenerationResult.StopReason.unknown.rawValue)))
        XCTAssertNil(MessageStopNotice.resolve(assistant("some_future_reason")))
    }

    func testIncompleteReasonsMapToTheirNotice() {
        XCTAssertEqual(MessageStopNotice.resolve(assistant("cancelled")), .stoppedByUser)
        XCTAssertEqual(
            MessageStopNotice.resolve(assistant(GenerationResult.StopReason.cancelled.rawValue)),
            .stoppedByUser)
        XCTAssertEqual(MessageStopNotice.resolve(assistant("error")), .engineError)
        // Subagent runs report "failed" for the same outcome.
        XCTAssertEqual(MessageStopNotice.resolve(assistant("failed")), .engineError)
        XCTAssertEqual(MessageStopNotice.resolve(assistant("context_overflow")), .contextFull)
        XCTAssertEqual(
            MessageStopNotice.resolve(assistant(GenerationResult.StopReason.maxTokens.rawValue)),
            .lengthLimit)
        XCTAssertEqual(MessageStopNotice.resolve(assistant("max_turns")), .turnLimit)
    }

    func testOnlyAssistantRowsCarryANotice() {
        let user = AppChatMessage(role: .user, content: "hi", stopReason: "error")
        XCTAssertNil(MessageStopNotice.resolve(user))
    }

    func testSeverityIsDangerOnlyForFailuresAndNeverColorAlone() {
        XCTAssertEqual(MessageStopNotice.stoppedByUser.severity, .neutral)
        XCTAssertEqual(MessageStopNotice.lengthLimit.severity, .warning)
        XCTAssertEqual(MessageStopNotice.turnLimit.severity, .warning)
        XCTAssertEqual(MessageStopNotice.contextFull.severity, .danger)
        XCTAssertEqual(MessageStopNotice.engineError.severity, .danger)
        // Each case pairs its color with a distinct glyph.
        let all: [MessageStopNotice] = [.stoppedByUser, .engineError, .contextFull, .lengthLimit, .turnLimit]
        for notice in all { XCTAssertFalse(notice.symbolName.isEmpty) }
        XCTAssertEqual(MessageStopNotice.engineError.symbolName, "xmark.octagon")
        XCTAssertEqual(MessageStopNotice.lengthLimit.symbolName, "exclamationmark.triangle")
    }

    func testCopyDetailsNamesTheStoredReason() {
        let details = MessageStopNotice.engineError.copyDetails(stopReason: "error")
        XCTAssertTrue(details.hasSuffix("\nstop_reason=error"), details)
        XCTAssertEqual(details.split(separator: "\n").count, 3, details)
    }
}
