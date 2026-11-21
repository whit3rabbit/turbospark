import XCTest
@testable import TurboSparkApp

/// The stored "Thought for N" duration and the folded live preview.
/// Durations come only from measured reasoning-chunk arrival times: a row
/// without a measurement shows no duration, never an estimate.
final class ReasoningTimingTests: XCTestCase {
    private let t0 = Date(timeIntervalSince1970: 1_000_000)

    // MARK: duration text

    func testDurationTextIsNilWithoutAMeasurement() {
        XCTAssertNil(ReasoningDurationText.text(seconds: nil))
        XCTAssertNil(ReasoningDurationText.text(seconds: -1))
        XCTAssertNil(ReasoningDurationText.text(seconds: .nan))
        XCTAssertNil(ReasoningDurationText.text(seconds: .infinity))
    }

    func testDurationTextFormats() {
        XCTAssertEqual(ReasoningDurationText.text(seconds: 0), "<1s")
        XCTAssertEqual(ReasoningDurationText.text(seconds: 0.4), "<1s")
        XCTAssertEqual(ReasoningDurationText.text(seconds: 1.0), "1s")
        XCTAssertEqual(ReasoningDurationText.text(seconds: 12.4), "12s")
        XCTAssertEqual(ReasoningDurationText.text(seconds: 12.6), "13s")
        XCTAssertEqual(ReasoningDurationText.text(seconds: 65), "1m 5s")
    }

    // MARK: clock

    func testClockMeasuresFirstToLastReasoningChunk() {
        var clock = ReasoningClock(chatID: UUID())
        XCTAssertNil(clock.seconds)
        clock.note(at: t0)
        XCTAssertEqual(clock.seconds, 0)
        clock.note(at: t0.addingTimeInterval(3))
        clock.note(at: t0.addingTimeInterval(9.5))
        XCTAssertEqual(clock.seconds, 9.5)
    }

    // MARK: stamping

    private func assistant(_ reasoning: String, at date: Date, seconds: Double? = nil) -> AppChatMessage {
        var message = AppChatMessage(role: .assistant, content: "a", reasoning: reasoning, createdAt: date)
        message.reasoningSeconds = seconds
        return message
    }

    private func measuredClock(_ chatID: UUID) -> ReasoningClock {
        var clock = ReasoningClock(chatID: chatID)
        clock.note(at: t0)
        clock.note(at: t0.addingTimeInterval(7))
        return clock
    }

    func testStampsTheTurnsNewReasoningRowOnce() {
        let chat = UUID()
        var clock = measuredClock(chat)
        var messages = [
            AppChatMessage(role: .user, content: "q", createdAt: t0.addingTimeInterval(-5)),
            assistant("thinking", at: t0.addingTimeInterval(8)),
        ]
        XCTAssertTrue(clock.stamp(&messages, chatID: chat))
        XCTAssertEqual(messages[1].reasoningSeconds, 7)
        // Consumed: a later mutation in the same chat stamps nothing.
        messages.append(assistant("other", at: t0.addingTimeInterval(20)))
        XCTAssertFalse(clock.stamp(&messages, chatID: chat))
        XCTAssertNil(messages[2].reasoningSeconds)
    }

    func testNeverStampsAnOlderRowOrAnotherChat() {
        let chat = UUID()
        var clock = measuredClock(chat)
        // Created before this turn's reasoning began: an old reply, or a
        // variant swapped back in with its own timestamp.
        var messages = [assistant("old", at: t0.addingTimeInterval(-60))]
        XCTAssertFalse(clock.stamp(&messages, chatID: chat))
        XCTAssertNil(messages[0].reasoningSeconds)

        var other = [assistant("new", at: t0.addingTimeInterval(8))]
        XCTAssertFalse(clock.stamp(&other, chatID: UUID()))
        XCTAssertNil(other[0].reasoningSeconds)
    }

    func testDoesNotStampRowsWithoutReasoningOrAlreadyTimed() {
        let chat = UUID()
        var clock = measuredClock(chat)
        var messages = [
            assistant("", at: t0.addingTimeInterval(8)),
            assistant("timed", at: t0.addingTimeInterval(9), seconds: 2),
        ]
        XCTAssertFalse(clock.stamp(&messages, chatID: chat))
        XCTAssertNil(messages[0].reasoningSeconds)
        XCTAssertEqual(messages[1].reasoningSeconds, 2)
    }

    func testUnmeasuredClockStampsNothing() {
        let chat = UUID()
        var clock = ReasoningClock(chatID: chat)
        var messages = [assistant("thinking", at: t0)]
        XCTAssertFalse(clock.stamp(&messages, chatID: chat))
        XCTAssertNil(messages[0].reasoningSeconds)
    }

    // MARK: persistence

    func testOldMessageJSONDecodesWithoutADuration() throws {
        let json = #"{"role":"assistant","content":"hi","reasoning":"r"}"#
        let message = try JSONDecoder().decode(AppChatMessage.self, from: Data(json.utf8))
        XCTAssertEqual(message.reasoning, "r")
        XCTAssertNil(message.reasoningSeconds)
    }

    func testDurationRoundTripsThroughCodable() throws {
        var message = AppChatMessage(role: .assistant, content: "hi", reasoning: "r")
        message.reasoningSeconds = 4.25
        let data = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(AppChatMessage.self, from: data)
        XCTAssertEqual(decoded.reasoningSeconds, 4.25)
    }

    @MainActor
    func testVariantStepCarriesEachVersionsDuration() {
        var older = AppChatMessage(role: .assistant, content: "v1", reasoning: "r1", createdAt: t0)
        older.reasoningSeconds = 3
        var message = AppChatMessage(
            role: .assistant, content: "v2", reasoning: "r2",
            alternates: [older], createdAt: t0.addingTimeInterval(10))
        message.reasoningSeconds = nil
        AppModel.applyVariantStep(&message, delta: -1)
        XCTAssertEqual(message.content, "v1")
        XCTAssertEqual(message.reasoningSeconds, 3)
        AppModel.applyVariantStep(&message, delta: 1)
        XCTAssertEqual(message.content, "v2")
        XCTAssertNil(message.reasoningSeconds)
    }

    // MARK: preview

    func testPreviewTailKeepsShortTextWhole() {
        XCTAssertEqual(ReasoningPreview.tail(of: "short trace", limit: 100), "short trace")
    }

    func testPreviewTailCutsAtALineOrWordBoundary() {
        let text = String(repeating: "alpha beta\n", count: 50) + "last line here"
        let tail = ReasoningPreview.tail(of: text, limit: 40)
        XCTAssertTrue(tail.hasSuffix("last line here"))
        XCTAssertLessThanOrEqual(tail.count, 40)
        XCTAssertFalse(tail.hasPrefix("lpha"), "must not start mid-word")
        XCTAssertFalse(tail.first?.isWhitespace ?? true)
    }
}
