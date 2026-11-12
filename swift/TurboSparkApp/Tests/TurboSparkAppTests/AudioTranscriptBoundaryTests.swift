import XCTest

@testable import TurboSparkApp

/// Transcription windows overlap by one second, so words near a seam are
/// heard twice. `reconcileBoundary` keeps one copy.
final class AudioTranscriptBoundaryTests: XCTestCase {
    private func segment(_ start: Double, _ end: Double, _ text: String) -> AudioLibrarySegment {
        AudioLibrarySegment(start: start, end: end, text: text, timing: "segment")
    }

    @MainActor private func reconcile(_ previous: [AudioLibrarySegment], _ next: [AudioLibrarySegment]) -> [AudioLibrarySegment] {
        var previous = previous, next = next
        AudioWorkspaceController.reconcileBoundary(previous: &previous, next: &next)
        return previous + next
    }

    /// Parakeet Ultra at the 29 s seam of a real take: the first window cut
    /// the sentence and ended it with a period, the second heard all of it.
    @MainActor func testASentenceCutByTheWindowKeepsTheFullCopyFromItsEarlierStart() {
        let previous = [segment(27.68, 28.64, "Perfect."), segment(28.80, 30.00, "Please send the results.")]
        let next = [
            segment(29.00, 33.80, "Please send the results to the whole group, and we can confirm the date on Thursday."),
            segment(33.80, 34.68, "Will do."),
        ]
        let joined = reconcile(previous, next)
        XCTAssertEqual(joined.map(\.text), [
            "Perfect.",
            "Please send the results to the whole group, and we can confirm the date on Thursday.",
            "Will do.",
        ])
        XCTAssertEqual(joined[1].start, 28.80)
        XCTAssertEqual(joined[1].end, 33.80)
        XCTAssertEqual(joined[1].id, next[0].id)
    }

    @MainActor func testOverlappingWordsMatchAcrossPunctuationAndCase() {
        let joined = reconcile([segment(25, 30, "we review the budget, Then")], [segment(29, 33, "the budget then the schedule.")])
        XCTAssertEqual(joined.map(\.text), ["we review the budget, Then", "the schedule."])
    }

    @MainActor func testIdenticalCopiesCollapseToOne() {
        let joined = reconcile([segment(28.5, 30, "Thank you.")], [segment(29, 30.2, "thank you")])
        XCTAssertEqual(joined.map(\.text), ["thank you"])
        XCTAssertEqual(joined.first?.start, 28.5)
    }

    @MainActor func testUncertainBoundariesStay() {
        // One shared word could be chance: both copies stay for the editor.
        XCTAssertEqual(reconcile([segment(26, 30, "over to the")], [segment(29, 32, "the next item")]).count, 2)
        // Different words.
        XCTAssertEqual(reconcile([segment(26, 30, "see you soon")], [segment(29, 32, "next item please")]).count, 2)
        // No time overlap: the same words said twice are both kept.
        XCTAssertEqual(reconcile([segment(20, 25, "thank you")], [segment(29, 31, "thank you")]).count, 2)
    }
}
