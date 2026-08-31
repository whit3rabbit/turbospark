import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The arithmetic under the Server pane's charts and console.
///
/// **A STORE RATHER THAN A VIEW MODEL IS WHAT MAKES THIS POSSIBLE.** Every
/// number here would otherwise be checkable only by starting a server,
/// sending it traffic and squinting at a chart -- which is exactly how the
/// address literal and the missing auth row survived in the settings panel
/// for the life of that feature.
final class ServerMetricsStoreTests: XCTestCase {

    private func started(_ id: UInt64, at ms: UInt64 = 0, path: String = "/v1/chat/completions")
        -> ServerEvent
    {
        .requestStarted(id: id, atMs: ms, method: "POST", path: path)
    }

    private func generated(
        _ id: UInt64, model: String = "alpha", prompt: UInt32 = 10, new: UInt32 = 20,
        prefill: Double = 0.5, decode: Double = 1.0
    ) -> ServerEvent {
        .generated(
            id: id, model: model, promptTokens: prompt, newTokens: new,
            prefillSeconds: prefill, decodeSeconds: decode, stopReason: "EndOfTurn")
    }

    private func finished(_ id: UInt64, status: UInt16 = 200, ms: UInt32 = 2_000) -> ServerEvent {
        .requestFinished(id: id, status: status, durationMs: ms)
    }

    // MARK: - Assembling a request from its events

    func testFourEventsBecomeOneRecord() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(.requestRouted(id: 1, requested: "asked", served: "alpha", stream: true))
        store.ingest(generated(1))
        store.ingest(finished(1))

        XCTAssertEqual(store.records.count, 1)
        let record = store.records[0]
        XCTAssertEqual(record.requestedModel, "asked")
        XCTAssertEqual(record.servedModel, "alpha")
        XCTAssertTrue(record.stream)
        XCTAssertEqual(record.newTokens, 20)
        XCTAssertEqual(record.status, 200)
        XCTAssertEqual(store.totalRequests, 1)
        XCTAssertEqual(store.inFlight, 0)
    }

    func testAStartedRequestIsInFlightUntilItFinishes() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        XCTAssertEqual(store.inFlight, 1)
        XCTAssertEqual(store.totalRequests, 0, "unfinished requests are not counted as served")

        store.ingest(finished(1))
        XCTAssertEqual(store.inFlight, 0)
        XCTAssertEqual(store.totalRequests, 1)
    }

    /// **AN EVENT FOR A REQUEST WE NEVER SAW START IS DROPPED, NOT
    /// INVENTED.** That is what an overrun ring produces, and a synthesized
    /// row would carry a method and a path nobody observed.
    func testAnOrphanedEventDoesNotInventARow() {
        var store = ServerMetricsStore()
        store.ingest(generated(99))
        store.ingest(finished(99))
        XCTAssertTrue(store.records.isEmpty)
    }

    // MARK: - The retry case

    /// **THE GUARDRAILS RE-ASK, AND BOTH TURNS REALLY RAN.** Replacing the
    /// counters would under-report exactly the requests that cost the most.
    func testTwoGenerationsAccumulateRatherThanReplace() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(generated(1, prompt: 10, new: 20, prefill: 0.5, decode: 1.0))
        store.ingest(generated(1, prompt: 30, new: 5, prefill: 0.7, decode: 0.25))
        store.ingest(finished(1))

        let record = store.records[0]
        XCTAssertEqual(record.generations, 2)
        XCTAssertEqual(record.promptTokens, 40)
        XCTAssertEqual(record.newTokens, 25)
        XCTAssertEqual(record.prefillSeconds ?? 0, 1.2, accuracy: 0.0001)
        XCTAssertEqual(record.decodeSeconds ?? 0, 1.25, accuracy: 0.0001)
    }

    // MARK: - Derived numbers

    func testTokensPerSecondComesFromTheDecodersOwnCounters() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(generated(1, new: 60, decode: 2.0))
        store.ingest(finished(1))
        XCTAssertEqual(store.records[0].tokensPerSecond ?? 0, 30, accuracy: 0.0001)
    }

    func testAZeroDecodeReportsNoRateRatherThanInfinity() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(generated(1, new: 0, decode: 0))
        store.ingest(finished(1))
        XCTAssertNil(store.records[0].tokensPerSecond)
    }

    /// The queue is the wall clock minus the work, and it is the number with
    /// no other source: there is one runner per model and nothing inside a
    /// generation can see that a second request was waiting.
    func testQueuedSecondsIsTheWallClockMinusTheWork() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(generated(1, prefill: 0.5, decode: 1.0))
        store.ingest(finished(1, ms: 2_000))
        XCTAssertEqual(store.records[0].queuedSeconds ?? 0, 0.5, accuracy: 0.0001)
    }

    /// **AN UNMEASURED WAIT IS NOT A MEASURED ZERO.** A request that never
    /// generated has no prefill or decode to subtract, so there is no queue
    /// figure -- reporting 0 would say it waited nothing.
    func testARequestThatNeverGeneratedReportsNoQueueRatherThanZero() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(finished(1, status: 404, ms: 3))
        XCTAssertNil(store.records[0].queuedSeconds)
    }

    /// **TOTAL TOKENS OVER TOTAL TIME, NOT A MEAN OF RATES.** The mean is the
    /// easy thing to write and it weights a two-token reply the same as a
    /// thousand-token one. Here: 100 tokens in 1s and 2 tokens in 1s is 51
    /// tok/s aggregate, where the mean of the two rates would be 51 as well
    /// -- so the case is built to separate them: 100 in 1s and 2 in 4s is 20.4
    /// aggregate against a mean of 50.25.
    func testAggregateRateIsWeightedByTokensRatherThanByRequest() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(generated(1, new: 100, decode: 1.0))
        store.ingest(finished(1))
        store.ingest(started(2))
        store.ingest(generated(2, new: 2, decode: 4.0))
        store.ingest(finished(2))

        XCTAssertEqual(store.aggregateTokensPerSecond ?? 0, 102.0 / 5.0, accuracy: 0.0001)
        XCTAssertNotEqual(
            store.aggregateTokensPerSecond ?? 0, 50.25, accuracy: 0.01,
            "a mean of per-request rates is the wrong number")
    }

    // MARK: - Percentiles

    func testNearestRankPercentileReturnsARealSample() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        XCTAssertEqual(ServerMetricsStore.percentile(values, 0.5), 5.0)
        XCTAssertEqual(ServerMetricsStore.percentile(values, 0.95), 10.0)
        XCTAssertEqual(ServerMetricsStore.percentile(values, 1.0), 10.0)
        // Nearest-rank, so every answer is a value that was actually
        // observed -- an interpolated p95 invents a number between two real
        // ones and reads as more precise than a handful of samples is.
        XCTAssertTrue(values.contains(ServerMetricsStore.percentile(values, 0.9) ?? 0))
    }

    func testPercentileOfNothingIsNilRatherThanZero() {
        XCTAssertNil(ServerMetricsStore.percentile([], 0.5))
    }

    func testASingleSampleIsEveryPercentile() {
        XCTAssertEqual(ServerMetricsStore.percentile([7.0], 0.5), 7.0)
        XCTAssertEqual(ServerMetricsStore.percentile([7.0], 0.95), 7.0)
    }

    // MARK: - The window

    func testTheWindowIsBoundedAndDropsTheOldest() {
        var store = ServerMetricsStore()
        let overflow = ServerMetricsStore.capacity + 5
        for i in 0..<overflow {
            let id = UInt64(i)
            store.ingest(started(id))
            store.ingest(finished(id))
        }
        XCTAssertEqual(store.records.count, ServerMetricsStore.capacity)
        XCTAssertEqual(store.records.first?.id, 5, "the five oldest went")
    }

    /// **TRIMMING SHIFTS EVERY OPEN REQUEST'S INDEX, and getting that wrong
    /// folds a later event into the WRONG row rather than failing.** That is
    /// the failure worth a test: the counters would attach to somebody
    /// else's request and nothing would look broken.
    ///
    /// The long-running request is deliberately NOT the first one. Records
    /// are appended in order, so the very first is always the first trimmed
    /// -- a case that proves the request is gone rather than that the shift
    /// is right.
    func testAnInFlightRequestSurvivesTrimmingAndItsLateEventsStillLandOnIt() {
        var store = ServerMetricsStore()
        let capacity = ServerMetricsStore.capacity

        // Ten completed requests, then the long-running one, then enough to
        // fill the window exactly.
        for i in 0..<10 {
            store.ingest(started(UInt64(1_000 + i)))
            store.ingest(finished(UInt64(1_000 + i)))
        }
        store.ingest(started(7))
        for i in 0..<(capacity - 11) {
            store.ingest(started(UInt64(2_000 + i)))
            store.ingest(finished(UInt64(2_000 + i)))
        }
        XCTAssertEqual(store.records.count, capacity, "the window should be exactly full")

        // Five more push the five oldest out -- all completed, so request 7
        // survives and its index moves from 10 to 5.
        for i in 0..<5 {
            store.ingest(started(UInt64(3_000 + i)))
            store.ingest(finished(UInt64(3_000 + i)))
        }
        store.ingest(generated(7, new: 42))
        store.ingest(finished(7))

        let record = store.records.first { $0.id == 7 }
        XCTAssertNotNil(record, "a completed request should have been trimmed, not this one")
        XCTAssertEqual(record?.newTokens, 42, "its late events must reach its OWN row")
        // And nothing landed on a neighbour.
        XCTAssertNil(store.records.first { $0.id == 3_000 }?.newTokens)
    }

    /// **A REQUEST THAT OUTLIVES THE WINDOW IS GONE, AND ITS LATER EVENTS
    /// ARE DROPPED RATHER THAN RESURRECTED.** Bounded memory wins over a
    /// complete record of one very old request, and the alternative -- an
    /// index left pointing into a shifted array -- is the corruption the
    /// test above exists to prevent.
    func testAnInFlightRequestOLDERThanTheWindowIsDroppedCleanly() {
        var store = ServerMetricsStore()
        store.ingest(started(0))
        for i in 1...ServerMetricsStore.capacity {
            store.ingest(started(UInt64(i)))
            store.ingest(finished(UInt64(i)))
        }
        store.ingest(generated(0, new: 42))
        store.ingest(finished(0))

        XCTAssertNil(store.records.first { $0.id == 0 })
        XCTAssertEqual(store.records.count, ServerMetricsStore.capacity)
        XCTAssertNil(
            store.records.first?.newTokens,
            "the dropped request's counters must not land on whatever is now at the front")
    }

    // MARK: - Reporting loss

    /// **DROPS ARE SUMMED, BECAUSE THE ENGINE REPORTS PER POLL.** Showing
    /// only the latest poll's number would flash a gap and then hide it.
    func testDroppedEventsAccumulateAcrossPolls() {
        var store = ServerMetricsStore()
        store.ingest(ServerEventBatch(events: [], dropped: 3))
        store.ingest(ServerEventBatch(events: [], dropped: 4))
        XCTAssertEqual(store.droppedEvents, 7)
    }

    // MARK: - Series

    /// Built from what has ANSWERED, never from the attached list: a series
    /// for a model that has served nothing is an empty line a reader has to
    /// work out the meaning of.
    func testServingModelsComesFromTrafficInFirstSeenOrder() {
        var store = ServerMetricsStore()
        for (index, name) in ["beta", "alpha", "beta"].enumerated() {
            let id = UInt64(index)
            store.ingest(started(id))
            store.ingest(.requestRouted(id: id, requested: nil, served: name, stream: false))
            store.ingest(generated(id, model: name))
            store.ingest(finished(id))
        }
        XCTAssertEqual(store.servingModels, ["beta", "alpha"])
    }

    func testChartPointsSkipRequestsThatNeverGenerated() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(.requestRouted(id: 1, requested: nil, served: "alpha", stream: false))
        store.ingest(generated(1))
        store.ingest(finished(1))
        store.ingest(started(2, path: "/v1/models"))
        store.ingest(finished(2))

        XCTAssertEqual(store.points.count, 1)
        XCTAssertEqual(store.points[0].requestID, 1)
    }

    func testRequestRateBucketsByStartTime() {
        var store = ServerMetricsStore()
        // Three at t=1000, one at t=3500, over a 4s span in 1s buckets ending
        // at t=4000: buckets start at 0.
        for id in UInt64(1)...3 {
            store.ingest(started(id, at: 1_000))
            store.ingest(finished(id))
        }
        store.ingest(started(4, at: 3_500))
        store.ingest(finished(4))

        let buckets = store.requestRate(now: 4_000, spanMs: 4_000, bucketMs: 1_000)
        XCTAssertEqual(buckets.map(\.count), [0, 3, 0, 1])
    }

    func testAnErrorIsCountedAndFlagged() {
        var store = ServerMetricsStore()
        store.ingest(started(1))
        store.ingest(finished(1, status: 503))
        XCTAssertTrue(store.records[0].isError)
        XCTAssertEqual(store.totalErrors, 1)
    }
}
