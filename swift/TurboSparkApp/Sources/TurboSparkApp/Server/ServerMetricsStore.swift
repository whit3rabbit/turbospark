import Foundation
import TurboSpark

/// One completed request, assembled from the several events that describe it.
///
/// **THE EVENTS ARRIVE SEPARATELY BECAUSE NO SINGLE EMITTER KNOWS ALL OF
/// THIS** (`crates/server/src/observe.rs`): the HTTP layer knows the method,
/// the path and the status, and the generation path knows the token counts.
/// This is where they are put back together, keyed on the request id.
public struct ServerRequestRecord: Identifiable, Equatable, Sendable {
    public let id: UInt64
    public let startedAtMs: UInt64
    public var method: String
    public var path: String
    /// What the request ASKED for. Differs from `servedModel` on every
    /// single-model fallback, which is the common case.
    public var requestedModel: String?
    public var servedModel: String?
    public var stream: Bool = false
    public var status: UInt16?
    public var durationMs: UInt32?
    public var promptTokens: UInt32?
    public var newTokens: UInt32?
    public var prefillSeconds: Double?
    public var decodeSeconds: Double?
    public var stopReason: String?
    /// How many generations this request ran. Two means the tool-call
    /// guardrails re-asked, which is real work worth seeing rather than a
    /// duplicate row.
    public var generations: Int = 0

    public var isFinished: Bool { status != nil }
    public var isError: Bool { (status ?? 0) >= 400 }

    /// Decode throughput, from the DECODER's own counters.
    ///
    /// **NOT A COUNT OF STREAMED CHUNKS.** `swift/CLAUDE.md` Gotcha 7 is the
    /// worked example of how far apart those two numbers are: special tokens
    /// render to the empty string, the detokenizer withholds partial UTF-8,
    /// and reasoning goes to a different channel entirely. `newTokens` comes
    /// off `RawDecodeResult` and is the number a benchmark would quote.
    public var tokensPerSecond: Double? {
        guard let newTokens, let decodeSeconds, decodeSeconds > 0, newTokens > 0 else {
            return nil
        }
        return Double(newTokens) / decodeSeconds
    }

    /// How long this request spent waiting rather than working: the wall
    /// clock the HTTP layer measured, minus the prefill and decode the
    /// generation reported.
    ///
    /// **THIS IS THE QUEUE, AND IT IS THE NUMBER WITH NO OTHER SOURCE.**
    /// There is one runner per model and a turn holds it for its whole
    /// duration, so a second request waits here and nothing inside a
    /// generation can see that it did. `nil` rather than zero when either
    /// half is missing -- an unmeasured wait is not a measured zero.
    public var queuedSeconds: Double? {
        guard let durationMs, let prefillSeconds, let decodeSeconds else { return nil }
        return max(0, Double(durationMs) / 1000 - prefillSeconds - decodeSeconds)
    }

    public init(id: UInt64, startedAtMs: UInt64, method: String, path: String) {
        self.id = id
        self.startedAtMs = startedAtMs
        self.method = method
        self.path = path
    }
}

/// A point on one of the pane's charts.
public struct ServerMetricPoint: Identifiable, Equatable, Sendable {
    public var id: UInt64 { requestID }
    public let requestID: UInt64
    public let atMs: UInt64
    public let model: String
    public let tokensPerSecond: Double?
    public let prefillSeconds: Double
    public let queuedSeconds: Double?
    public let isError: Bool
}

/// The rolling window the Server pane's charts and console read.
///
/// **PURE, AND DELIBERATELY NOT A VIEW MODEL.** It takes events in and hands
/// series out, with no `TurboSparkServer`, no timer and no SwiftUI in it, so
/// the arithmetic underneath every chart can be tested without a bound
/// socket. The percentile maths and the window trimming are the parts most
/// likely to be quietly wrong.
public struct ServerMetricsStore: Equatable {
    /// How many completed requests to keep. Roughly an hour of steady
    /// interactive use, and bounded because a server left running overnight
    /// must not grow the app's memory without limit.
    public static let capacity = 2_000

    /// Completed and in-flight requests, oldest first.
    public private(set) var records: [ServerRequestRecord] = []
    /// Requests seen but not yet finished, by id.
    private var open: [UInt64: Int] = [:]
    /// Events the engine's ring discarded, totalled over this session.
    ///
    /// **SUMMED HERE BECAUSE THE ENGINE REPORTS PER POLL.** A host that
    /// showed only the latest poll's number would flash a gap and then hide
    /// it, which is worse than not reporting one.
    public private(set) var droppedEvents: UInt64 = 0

    public init() {}

    /// Folds one drained batch in.
    public mutating func ingest(_ batch: ServerEventBatch) {
        droppedEvents += batch.dropped
        for event in batch.events {
            ingest(event)
        }
    }

    public mutating func ingest(_ event: ServerEvent) {
        switch event {
        case let .requestStarted(id, atMs, method, path):
            let record = ServerRequestRecord(
                id: id, startedAtMs: atMs, method: method, path: path)
            open[id] = records.count
            records.append(record)
            // **TRIMMED HERE RATHER THAN ONCE PER BATCH, and that is a
            // correctness fix rather than a tidy-up.** With the trim on the
            // batch path alone, this entry point -- which is public, and is
            // what a test or any other caller reaches for -- grew the window
            // without limit. Found by `testTheWindowIsBoundedAndDropsTheOldest`
            // reading 2,005 records against a 2,000 capacity, and it had
            // ALSO made `testAnInFlightRequestSurvivesTrimmingAndStillUpdates`
            // vacuous: that case passed because no trimming ever happened.
            //
            // This is the only arm that can grow `records`, so it is the only
            // one that needs it.
            trim()

        case let .requestRouted(id, requested, served, stream):
            update(id) {
                $0.requestedModel = requested
                $0.servedModel = served
                $0.stream = stream
            }

        case let .generated(id, model, promptTokens, newTokens, prefill, decode, stopReason):
            update(id) {
                $0.servedModel = model
                $0.generations += 1
                // **A RETRY'S COUNTERS ACCUMULATE RATHER THAN REPLACE.**
                // The guardrails re-ask once, and both generations ran: the
                // machine really did prefill and decode twice, so a chart of
                // work done must show both. Replacing would under-report the
                // cost of exactly the requests that cost the most.
                $0.promptTokens = ($0.promptTokens ?? 0) + promptTokens
                $0.newTokens = ($0.newTokens ?? 0) + newTokens
                $0.prefillSeconds = ($0.prefillSeconds ?? 0) + prefill
                $0.decodeSeconds = ($0.decodeSeconds ?? 0) + decode
                $0.stopReason = stopReason
            }

        case let .requestFinished(id, status, durationMs):
            update(id) {
                $0.status = status
                $0.durationMs = durationMs
            }
            open[id] = nil

        case .modelAttached, .modelDetached, .unknown:
            // Not requests. The console renders these from the event stream
            // directly; nothing here has a row to fold them into.
            break
        }
    }

    private mutating func update(_ id: UInt64, _ body: (inout ServerRequestRecord) -> Void) {
        // **AN EVENT FOR A REQUEST WE NEVER SAW START IS DROPPED, NOT
        // INVENTED.** That happens when the ring overflowed and lost the
        // `requestStarted`, and a synthesized row would carry a method and
        // path nobody observed. `droppedEvents` is what tells the user.
        guard let index = open[id], index < records.count, records[index].id == id else { return }
        body(&records[index])
    }

    /// Keeps the window bounded, and keeps `open`'s indices valid.
    private mutating func trim() {
        guard records.count > Self.capacity else { return }
        let removed = records.count - Self.capacity
        records.removeFirst(removed)
        // Shift every open index by what was dropped, discarding any that
        // fell off the front. Rebuilding from `records` would be O(n) on
        // every poll for a case that fires once every 2,000 requests.
        open = open.compactMapValues { $0 >= removed ? $0 - removed : nil }
    }

    /// Requests currently started and not finished.
    public var inFlight: Int { open.count }

    /// Every completed request that ran a generation, as chart points.
    public var points: [ServerMetricPoint] {
        records.compactMap { record in
            guard record.isFinished, let prefill = record.prefillSeconds,
                let model = record.servedModel
            else { return nil }
            return ServerMetricPoint(
                requestID: record.id,
                atMs: record.startedAtMs,
                model: model,
                tokensPerSecond: record.tokensPerSecond,
                prefillSeconds: prefill,
                queuedSeconds: record.queuedSeconds,
                isError: record.isError)
        }
    }

    /// The models that have actually served something, for a per-series
    /// chart.
    ///
    /// **BUILT FROM WHAT IS PRESENT, NEVER FROM THE ATTACHED LIST**
    /// (`swift/CLAUDE.md` Gotcha 22's rule): a series for a model that has
    /// answered nothing is an empty line a reader has to work out the
    /// meaning of.
    public var servingModels: [String] {
        var seen: [String] = []
        for record in records {
            if let model = record.servedModel, !seen.contains(model) {
                seen.append(model)
            }
        }
        return seen
    }

    public var totalRequests: Int { records.filter(\.isFinished).count }
    public var totalErrors: Int { records.filter(\.isError).count }

    /// Tokens decoded per second across every completed generation in the
    /// window, weighted by tokens rather than averaged over requests.
    ///
    /// **A MEAN OF PER-REQUEST RATES IS THE WRONG NUMBER** and it is the one
    /// that is easy to write: it weights a two-token reply the same as a
    /// thousand-token one. Total tokens over total decode time is what a
    /// throughput figure means.
    public var aggregateTokensPerSecond: Double? {
        var tokens = 0.0
        var seconds = 0.0
        for record in records {
            guard let newTokens = record.newTokens, let decode = record.decodeSeconds else {
                continue
            }
            tokens += Double(newTokens)
            seconds += decode
        }
        guard seconds > 0, tokens > 0 else { return nil }
        return tokens / seconds
    }

    /// The `q`th percentile of `values` by nearest-rank, or `nil` when there
    /// is nothing to rank.
    ///
    /// Nearest-rank rather than interpolated: with the handful of samples a
    /// local server produces, an interpolated p95 invents a value between
    /// two real ones and reads as more precise than the data is.
    public static func percentile(_ values: [Double], _ q: Double) -> Double? {
        guard !values.isEmpty else { return nil }
        let sorted = values.sorted()
        let rank = Int((q * Double(sorted.count)).rounded(.up))
        return sorted[min(max(rank, 1), sorted.count) - 1]
    }

    /// Prefill seconds at the given percentile, over the window.
    public func prefillPercentile(_ q: Double) -> Double? {
        Self.percentile(records.compactMap(\.prefillSeconds), q)
    }

    /// Requests started in each `bucketMs`-wide bucket over the last
    /// `spanMs`, oldest first. `now` is passed in rather than read so this
    /// stays testable.
    public func requestRate(now: UInt64, spanMs: UInt64, bucketMs: UInt64) -> [(at: UInt64, count: Int)] {
        guard bucketMs > 0, spanMs >= bucketMs else { return [] }
        let start = now >= spanMs ? now - spanMs : 0
        let bucketCount = Int(spanMs / bucketMs)
        var counts = [Int](repeating: 0, count: bucketCount)
        for record in records where record.startedAtMs >= start {
            let offset = record.startedAtMs - start
            let index = Int(offset / bucketMs)
            if index < bucketCount {
                counts[index] += 1
            }
        }
        return counts.enumerated().map { (at: start + UInt64($0.offset) * bucketMs, count: $0.element) }
    }
}
