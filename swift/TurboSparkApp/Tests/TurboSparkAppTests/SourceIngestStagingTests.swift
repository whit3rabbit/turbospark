import XCTest

@testable import TurboSparkApp

/// The Files section's import staging list: per-file state derived from the
/// importer's own results, failures first, and Retry only where a second
/// attempt could change the answer.
final class SourceIngestStagingTests: XCTestCase {

    private typealias Staging = SourceIngestStaging

    private func url(_ name: String) -> URL {
        URL(fileURLWithPath: "/tmp/ingest/\(name)")
    }

    private func entry(_ name: String, _ state: Staging.State) -> Staging.Entry {
        Staging.Entry(url: url(name), chatID: nil, state: state)
    }

    // MARK: - Classification

    func testSuccessIsAdded() {
        let result = AttachmentImporter.FileResult(url: url("a.md"), failure: nil)
        XCTAssertEqual(Staging.state(for: result), .added)
    }

    func testExtractorRefusalsAreSkippedWithTheirReason() {
        let cases: [(DocumentTextExtractionError, Staging.SkipReason)] = [
            (.unsupportedFormat("x"), .unsupportedType),
            (.documentTooLarge("x"), .tooLarge),
            (.unreadableFile("x"), .unreadable),
            (.noExtractableText("x"), .noText),
        ]
        for (error, reason) in cases {
            let result = AttachmentImporter.FileResult(
                url: url("x"),
                failure: .init(message: "m", extractionError: error))
            XCTAssertEqual(Staging.state(for: result), .skipped(reason), "\(error)")
        }
    }

    func testOtherErrorsFailWithTheImporterMessage() {
        let timedOut = AttachmentImporter.FileResult(
            url: url("big.pdf"),
            failure: .init(message: "Reading big.pdf took too long.",
                           extractionError: .extractionTimedOut("big.pdf")))
        XCTAssertEqual(Staging.state(for: timedOut), .failed("Reading big.pdf took too long."))

        // A store write error carries no extraction error at all.
        let storeError = AttachmentImporter.FileResult(
            url: url("a.md"), failure: .init(message: "disk full", extractionError: nil))
        XCTAssertEqual(Staging.state(for: storeError), .failed("disk full"))
    }

    func testOnlyFailedCanRetry() {
        XCTAssertTrue(Staging.canRetry(.failed("x")))
        for state: Staging.State in [.queued, .reading, .added, .skipped(.tooLarge)] {
            XCTAssertFalse(Staging.canRetry(state), "\(state)")
        }
    }

    // MARK: - Staging and applying

    func testStagingQueuesNewFilesAndReplacesARepeatOfTheSameFile() {
        let chat = UUID()
        let existing = [entry("a.md", .failed("x")), entry("b.md", .added)]
        let staged = Staging.staged(existing, adding: [url("a.md"), url("c.md")], chatID: chat)
        XCTAssertEqual(staged.map(\.url.lastPathComponent), ["b.md", "a.md", "c.md"])
        XCTAssertEqual(staged.map(\.state), [.added, .queued, .queued])
        XCTAssertEqual(staged[1].chatID, chat)
    }

    func testApplyingResultsUpdatesOnlyMatchingEntries() {
        let entries = [entry("a.md", .reading), entry("b.md", .reading), entry("c.md", .added)]
        let results = [
            AttachmentImporter.FileResult(url: url("a.md"), failure: nil),
            AttachmentImporter.FileResult(
                url: url("b.md"),
                failure: .init(message: "m", extractionError: .unsupportedFormat("b.md"))),
        ]
        let applied = Staging.applying(results, to: entries)
        XCTAssertEqual(applied.map(\.state), [.added, .skipped(.unsupportedType), .added])
        XCTAssertEqual(applied.map(\.id), entries.map(\.id))
    }

    func testMarkingReadingTouchesOnlyAQueuedOrFailedEntry() {
        let entries = [entry("a.md", .queued), entry("b.md", .added)]
        XCTAssertEqual(Staging.markingReading(url("a.md"), in: entries).map(\.state), [.reading, .added])
        XCTAssertEqual(Staging.markingReading(url("b.md"), in: entries).map(\.state), [.queued, .added])
    }

    /// A file the importer returned no result for (the call bailed early)
    /// must not stay Reading forever.
    func testFinishingABatchFailsEntriesLeftInFlight() {
        let entries = [entry("a.md", .reading), entry("b.md", .queued), entry("c.md", .added)]
        let finished = Staging.finishing([url("a.md"), url("b.md")], in: entries, fallback: "gone")
        XCTAssertEqual(finished.map(\.state), [.failed("gone"), .failed("gone"), .added])
    }

    // MARK: - Aggregation

    func testOrderingPutsFailuresFirstThenSkippedThenWorkThenDoneStably() {
        let entries = [
            entry("1", .added), entry("2", .queued), entry("3", .failed("x")),
            entry("4", .skipped(.tooLarge)), entry("5", .reading), entry("6", .failed("y")),
            entry("7", .added),
        ]
        XCTAssertEqual(
            Staging.ordered(entries).map(\.url.lastPathComponent),
            ["3", "6", "4", "5", "2", "1", "7"])
    }

    func testSummaryCountsEveryStateAndKnowsWhenWorkIsInFlight() {
        let entries = [
            entry("1", .added), entry("2", .added), entry("3", .failed("x")),
            entry("4", .skipped(.noText)), entry("5", .reading), entry("6", .queued),
        ]
        let summary = Staging.summary(entries)
        XCTAssertEqual(
            summary,
            Staging.Summary(queued: 1, reading: 1, added: 2, skipped: 1, failed: 1))
        XCTAssertTrue(summary.isInFlight)
        XCTAssertEqual(summary.total, 6)
        XCTAssertFalse(Staging.summary([entry("1", .added), entry("2", .failed("x"))]).isInFlight)
        XCTAssertEqual(Staging.summary([]), Staging.Summary())
    }
}
