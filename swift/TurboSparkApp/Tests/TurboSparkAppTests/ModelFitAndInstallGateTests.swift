import XCTest
import TurboSpark

@testable import TurboSparkApp

/// The verdict presentation and the install gate: the two decisions that
/// stand between a user and twenty minutes of streaming a checkpoint that
/// cannot open.
final class ModelFitAndInstallGateTests: XCTestCase {

    // MARK: - Presentation

    /// **EVERY VERDICT RENDERS, `.unknown` INCLUDED.** Both pills used to
    /// return `EmptyView()` for it, which is why an unsized row could sit
    /// beside a grid of zeros with nothing saying the zeros were not
    /// measurements (swift Gotcha 23).
    func testEveryVerdictHasAVisiblePresentation() {
        let all: [ModelRecommendation.FitVerdict] = [
            .resident, .streams, .tight, .refused, .unknown,
        ]
        for verdict in all {
            let p = ModelFitPresentation.of(verdict)
            XCTAssertFalse(p.label.isEmpty, "\(verdict) needs a label")
            XCTAssertFalse(p.compactLabel.isEmpty, "\(verdict) needs a compact label")
            XCTAssertFalse(p.help.isEmpty, "\(verdict) needs a tooltip")
        }
    }

    /// Labels are distinct, or a filter built from them collapses two
    /// verdicts into one row.
    func testVerdictLabelsAreDistinct() {
        let all: [ModelRecommendation.FitVerdict] = [
            .resident, .streams, .tight, .refused, .unknown,
        ]
        XCTAssertEqual(Set(all.map { ModelFitPresentation.of($0).label }).count, all.count)
        XCTAssertEqual(Set(all.map { ModelFitPresentation.filterLabel($0) }).count, all.count)
    }

    /// **STREAMING IS NOT A CAUTION.** Reading routed experts from storage is
    /// what this engine is built for, so `.streams` must not share a tone
    /// with `.tight`. Getting this wrong tells a user their MoE model is a
    /// problem when it is the intended configuration.
    func testStreamingReadsAsAFitAndNotAWarning() {
        XCTAssertEqual(ModelFitPresentation.of(.streams).tone, .streaming)
        XCTAssertEqual(ModelFitPresentation.of(.tight).tone, .caution)
        XCTAssertEqual(ModelFitPresentation.of(.refused).tone, .blocked)
        XCTAssertNotEqual(
            ModelFitPresentation.of(.streams).tone,
            ModelFitPresentation.of(.tight).tone)
    }

    /// No tooltip states a byte count. A footprint is only meaningful beside
    /// its context and slot count, and the `.streams` badge used to advertise
    /// a hardcoded "~2-4 GB RAM" for every MoE checkpoint regardless of layer
    /// count or expert stride.
    func testNoVerdictTooltipStatesAByteCount() {
        let all: [ModelRecommendation.FitVerdict] = [
            .resident, .streams, .tight, .refused, .unknown,
        ]
        for verdict in all {
            let help = ModelFitPresentation.of(verdict).help
            for unit in ["GB", "MB", "GiB", "MiB"] {
                XCTAssertFalse(
                    help.contains(unit),
                    "\(verdict) tooltip states a size in \(unit): \(help)")
            }
        }
    }

    // MARK: - Context ladder

    private func rung(
        _ context: UInt32, _ counted: UInt64, _ verdict: String,
        trainedMax: Bool = false, past: Bool = false, largest: Bool = false
    ) throws -> ContextLadderRung {
        let json = """
            {"context":\(context),"kvBytes":\(counted),"counted":\(counted),
             "verdict":"\(verdict)","runs":\(verdict != "refused"),
             "pastTrained":\(past),"isTrainedMax":\(trainedMax),
             "isLargestFitting":\(largest)}
            """
        return try JSONDecoder().decode(ContextLadderRung.self, from: Data(json.utf8))
    }

    /// **THE LADDER IS RENDERED, NEVER DERIVED.** KV is not linear in the
    /// window: a sliding-window layer is a ring capped at
    /// `sliding_window + 128` and stops growing, while a fully-attentive one
    /// grows forever. Measured 4,096 to 131,072 on the shipped baselines,
    /// Mistral 7B grows 32x and Gemma 4 grows 9x, so a view multiplying its
    /// own 4,096 figure is 3.5x high on Gemma. This pins that the app reads
    /// what the engine computed: a nonlinear set of rungs survives intact.
    func testTheLadderCarriesANonlinearCurveUnchanged() throws {
        let gemmaLike = [
            try rung(4_096, 320 * 1024 * 1024, "streams"),
            try rung(32_768, 907 * 1024 * 1024, "streams"),
            try rung(131_072, 2_920 * 1024 * 1024, "tight"),
        ]
        let first = gemmaLike[0].counted
        let last = gemmaLike[2].counted
        let contextRatio = Double(gemmaLike[2].context) / Double(gemmaLike[0].context)
        let byteRatio = Double(last) / Double(first)
        XCTAssertEqual(contextRatio, 32.0, accuracy: 0.01)
        XCTAssertLessThan(
            byteRatio, 12.0,
            "a sliding-window family must not scale with the window; a view that "
                + "multiplied would report \(first * 32) instead of \(last)")
    }

    /// Every verdict on a rung maps to a visible presentation, including
    /// `.refused` at the long end, which is the row a user is looking for.
    func testEveryRungVerdictRenders() throws {
        for v in ["resident", "streams", "tight", "refused", "unknown"] {
            let r = try rung(4_096, 1024, v)
            let p = ModelFitPresentation.of(r.verdict)
            XCTAssertFalse(p.compactLabel.isEmpty, "\(v) needs a compact label")
        }
    }

    /// Past-trained rungs are reported rather than dropped: RoPE extrapolates
    /// rather than failing, and some checkpoints ship YaRN scaling meant to
    /// exceed the trained window.
    func testAPastTrainedRungIsStillARungAndIsMarked() throws {
        let r = try rung(131_072, 8 << 30, "tight", past: true)
        XCTAssertTrue(r.pastTrained)
        XCTAssertTrue(r.runs, "past trained is a warning, not a refusal")
    }

    // MARK: - Throughput

    private func band(_ chip: String, thisChip: Bool) throws -> ThroughputBand {
        let json = """
            {"minTokensPerSecond":33.0,"maxTokensPerSecond":45.6,
             "chip":"\(chip)","measuredOnThisChip":\(thisChip)}
            """
        return try JSONDecoder().decode(ThroughputBand.self, from: Data(json.utf8))
    }

    /// **A BAND FROM OTHER SILICON KEEPS ITS CHIP.** Every frozen row was
    /// taken on one machine, so a chip-matched lookup answers nothing
    /// elsewhere and the row used to vanish. The chip is part of the value,
    /// not provenance: tok/s does not transfer.
    func testAForeignBandIsUsableAndNamesItsChip() throws {
        let foreign = try band("Apple M4 Max", thisChip: false)
        XCTAssertFalse(foreign.measuredOnThisChip)
        XCTAssertFalse(foreign.chip.isEmpty, "an unlabelled foreign band is a false claim")

        let local = try band("Apple M4 Max", thisChip: true)
        XCTAssertTrue(local.measuredOnThisChip)
        XCTAssertEqual(local.minTokensPerSecond, 33.0, accuracy: 0.001)
    }

    // MARK: - The install gate

    private let plentyOfDisk: UInt64 = 900 * 1024 * 1024 * 1024

    func testAnUnprobedFittingRowIsAllowed() {
        let d = ModelInstallGate.decide(
            probeRunnable: nil, refusedBecause: nil, verdict: .streams,
            installBytes: 18_000_000_000, freeDiskBytes: plentyOfDisk)
        XCTAssertEqual(d, .allowed)
    }

    /// A resident fit is as allowed as a streaming one.
    func testAResidentRowIsAllowed() {
        let d = ModelInstallGate.decide(
            probeRunnable: nil, refusedBecause: nil, verdict: .resident,
            installBytes: 4_000_000_000, freeDiskBytes: plentyOfDisk)
        XCTAssertEqual(d, .allowed)
    }

    func testARefusedVerdictBlocks() {
        let d = ModelInstallGate.decide(
            probeRunnable: nil, refusedBecause: nil, verdict: .refused,
            installBytes: 60_000_000_000, freeDiskBytes: plentyOfDisk)
        XCTAssertTrue(d.isBlocked)
    }

    /// **AN UNPORTED ARCHITECTURE BLOCKS IN THE ENGINE'S OWN WORDS, AND
    /// OUTRANKS THE ARITHMETIC.** A model with no decode flow does not "fit"
    /// whatever its footprint would be, and the architecture message is the
    /// one that tells a user something they can act on. `catalog::discover`
    /// orders these the same way.
    func testAnUnrunnableProbeBlocksAndKeepsItsReason() {
        let why = "GGUF architecture \"phi3\" is recognized but has no decode flow here"
        let d = ModelInstallGate.decide(
            probeRunnable: false, refusedBecause: why, verdict: .resident,
            installBytes: 2_000_000_000, freeDiskBytes: plentyOfDisk)
        XCTAssertTrue(d.isBlocked)
        XCTAssertEqual(d.reason, why, "the engine's wording must survive the gate")
    }

    /// A refusal with no message still blocks, and still says something.
    func testAnUnrunnableProbeWithNoReasonStillBlocks() {
        let d = ModelInstallGate.decide(
            probeRunnable: false, refusedBecause: nil, verdict: nil,
            installBytes: nil, freeDiskBytes: plentyOfDisk)
        XCTAssertTrue(d.isBlocked)
        XCTAssertFalse(d.reason?.isEmpty ?? true)
    }

    func testATightFitAsksRatherThanBlocking() {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .tight,
            installBytes: 12_000_000_000, freeDiskBytes: plentyOfDisk)
        XCTAssertEqual(d, .confirm(d.reason ?? ""))
        XCTAssertFalse(d.isBlocked)
    }

    /// **NOT SIZED IS NOT REFUSED.** Refusing what nothing has measured would
    /// block every side-loaded install and every catalog row whose header
    /// nobody has read.
    func testAnUnsizedCandidateIsNotBlocked() {
        for verdict: ModelRecommendation.FitVerdict? in [nil, .unknown] {
            let d = ModelInstallGate.decide(
                probeRunnable: nil, refusedBecause: nil, verdict: verdict,
                installBytes: nil, freeDiskBytes: nil)
            XCTAssertEqual(d, .allowed, "verdict \(String(describing: verdict))")
        }
    }

    /// Low disk asks and explains why temporary scratch remains live until
    /// the final install has passed verification.
    func testLowDiskAsksAndExplainsTheRangeCache() throws {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .streams,
            installBytes: 18_000_000_000,
            freeDiskBytes: 20_000_000_000)  // fits, but not with headroom
        XCTAssertFalse(d.isBlocked)
        let reason = try XCTUnwrap(d.reason)
        XCTAssertTrue(reason.contains("Verified download ranges"), "got \(reason)")
    }

    func testDiskGateCountsInstallAndResumableDownloadScratch() {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .streams,
            installBytes: 12_000_000_000, downloadBytes: 10_000_000_000,
            freeDiskBytes: 25_000_000_000)
        XCTAssertEqual(d, .confirm(d.reason ?? ""))
        XCTAssertTrue(
            d.reason?.contains(MetricFormat.storage(22_000_000_000)) ?? false,
            "got \(d.reason ?? "")")
    }

    /// Disk is checked before memory: it is the failure that wastes the
    /// download rather than the one that fails fast at open.
    func testDiskOutranksATightFit() {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .tight,
            installBytes: 18_000_000_000, freeDiskBytes: 20_000_000_000)
        XCTAssertTrue(d.reason?.contains("free") ?? false, "got \(d.reason ?? "")")
    }

    /// **AN UNKNOWN FREE-SPACE READING IS NOT A FULL DISK.** A failed volume
    /// query must not refuse every install on a machine whose disk the system
    /// declines to describe -- that is Gotcha 23's shape applied to a gate.
    func testAnUnknownDiskReadingDoesNotBlock() {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .streams,
            installBytes: 500_000_000_000, freeDiskBytes: nil)
        XCTAssertEqual(d, .allowed)
    }

    func testZeroFreeDiskAsksRatherThanTreatingAsUnknown() {
        let d = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .streams,
            installBytes: 1_000_000, freeDiskBytes: 0)
        XCTAssertEqual(d, .confirm(d.reason ?? ""))
    }
}
