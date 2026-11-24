import XCTest
@testable import TurboSparkApp

/// The transcript's turn rail (`TranscriptScrollRail`): which rows get a
/// tick, where each tick sits, what its preview says, and which tick a
/// pointer or a VoiceOver swipe lands on. The view only draws these answers.
final class TranscriptScrollRailTests: XCTestCase {
    private func user(_ content: String, synthetic: Bool = false) -> AppChatMessage {
        AppChatMessage(role: .user, content: content, isSynthetic: synthetic)
    }

    private func assistant(_ content: String) -> AppChatMessage {
        AppChatMessage(role: .assistant, content: content)
    }

    // MARK: items

    func testFewerThanThreeHumanPromptsHasNoRail() {
        XCTAssertTrue(TranscriptScrollRail.items(in: []).isEmpty)
        let two = [user("a"), assistant("x"), user("b"), assistant("y")]
        XCTAssertTrue(TranscriptScrollRail.items(in: two).isEmpty)
    }

    func testOneTickPerHumanPromptInOrder() {
        let u1 = user("first"), u2 = user("second"), u3 = user("third")
        let items = TranscriptScrollRail.items(in: [u1, assistant("a"), u2, assistant("b"), u3])
        XCTAssertEqual(items.map(\.id), [u1.id, u2.id, u3.id])
        XCTAssertEqual(items.map(\.index), [0, 1, 2])
        XCTAssertEqual(items.map(\.preview), ["first", "second", "third"])
    }

    /// App- or hook-written user rows are not turns the reader took, so they
    /// neither get a tick nor count toward the three-turn floor.
    func testSyntheticUserRowsAreNotTurns() {
        let messages = [user("a"), user("nudge", synthetic: true), user("b"), assistant("x")]
        XCTAssertTrue(TranscriptScrollRail.items(in: messages).isEmpty)
        let more = messages + [user("c")]
        XCTAssertEqual(TranscriptScrollRail.items(in: more).map(\.preview), ["a", "b", "c"])
    }

    // MARK: preview

    func testPreviewIsFirstNonEmptyLineTrimmed() {
        XCTAssertEqual(TranscriptScrollRail.preview(of: "\n  \n  Fix the build  \nmore"), "Fix the build")
        XCTAssertEqual(TranscriptScrollRail.preview(of: "one line"), "one line")
    }

    func testPreviewOfBlankContentIsNil() {
        XCTAssertNil(TranscriptScrollRail.preview(of: ""))
        XCTAssertNil(TranscriptScrollRail.preview(of: " \n\t\n "))
    }

    func testLongPreviewIsTruncatedWithEllipsis() {
        let long = String(repeating: "a", count: 100)
        let preview = TranscriptScrollRail.preview(of: long, limit: 10)
        XCTAssertEqual(preview, "aaaaaaaaaa\u{2026}")
        XCTAssertEqual(TranscriptScrollRail.preview(of: "exactly10!", limit: 10), "exactly10!")
    }

    func testTruncationDropsTrailingSpaceBeforeEllipsis() {
        XCTAssertEqual(TranscriptScrollRail.preview(of: "hello     world", limit: 7), "hello\u{2026}")
    }

    // MARK: layout

    func testTickFractionIsProportionalToTurnIndex() {
        XCTAssertEqual(TranscriptScrollRail.tickFraction(index: 0, count: 3), 0)
        XCTAssertEqual(TranscriptScrollRail.tickFraction(index: 1, count: 3), 0.5)
        XCTAssertEqual(TranscriptScrollRail.tickFraction(index: 2, count: 3), 1)
        XCTAssertEqual(TranscriptScrollRail.tickFraction(index: 1, count: 5), 0.25)
    }

    func testTickFractionOutOfRangeIsNil() {
        XCTAssertNil(TranscriptScrollRail.tickFraction(index: -1, count: 3))
        XCTAssertNil(TranscriptScrollRail.tickFraction(index: 3, count: 3))
        XCTAssertNil(TranscriptScrollRail.tickFraction(index: 0, count: 0))
        XCTAssertEqual(TranscriptScrollRail.tickFraction(index: 0, count: 1), 0.5)
    }

    func testPointerMapsToNearestTick() {
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: 0, count: 5), 0)
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: 0.1, count: 5), 0)
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: 0.2, count: 5), 1)
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: 1, count: 5), 4)
        // Outside the rail clamps to the end ticks.
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: -0.3, count: 5), 0)
        XCTAssertEqual(TranscriptScrollRail.index(atFraction: 1.4, count: 5), 4)
        XCTAssertNil(TranscriptScrollRail.index(atFraction: 0.5, count: 0))
        XCTAssertNil(TranscriptScrollRail.index(atFraction: .nan, count: 5))
    }

    /// The rail must never sit over the reading column, so it hides when the
    /// pane is too narrow to give it a gutter.
    func testRailShowsOnlyWithAGutterBesideTheColumn() {
        XCTAssertTrue(TranscriptScrollRail.fitsBeside(columnWidth: 760, containerWidth: 900))
        XCTAssertFalse(TranscriptScrollRail.fitsBeside(columnWidth: 760, containerWidth: 790))
        XCTAssertFalse(TranscriptScrollRail.fitsBeside(columnWidth: 760, containerWidth: 0))
    }

    // MARK: current and adjust

    func testCurrentIndexFollowsLastJumpTarget() {
        let u1 = user("a"), u2 = user("b"), u3 = user("c")
        let items = TranscriptScrollRail.items(in: [u1, u2, u3])
        XCTAssertEqual(TranscriptScrollRail.currentIndex(of: u2.id, in: items), 1)
        XCTAssertNil(TranscriptScrollRail.currentIndex(of: nil, in: items))
        XCTAssertNil(TranscriptScrollRail.currentIndex(of: UUID(), in: items))
    }

    /// VoiceOver's adjustable action clamps at the ends (a swipe past the last
    /// turn stays there) and, with no current turn, starts from the end the
    /// swipe points at, the same first step `jumpTurn` takes.
    func testAdjustClampsAndStartsFromTheSwipedEnd() {
        XCTAssertEqual(TranscriptScrollRail.adjusted(current: 1, step: 1, count: 3), 2)
        XCTAssertEqual(TranscriptScrollRail.adjusted(current: 2, step: 1, count: 3), 2)
        XCTAssertEqual(TranscriptScrollRail.adjusted(current: 0, step: -1, count: 3), 0)
        XCTAssertEqual(TranscriptScrollRail.adjusted(current: nil, step: -1, count: 3), 2)
        XCTAssertEqual(TranscriptScrollRail.adjusted(current: nil, step: 1, count: 3), 0)
        XCTAssertNil(TranscriptScrollRail.adjusted(current: nil, step: 1, count: 0))
    }
}
