import XCTest
@testable import TurboSparkApp

/// The resource strip above the composer (`ResourceWarning`): thresholds,
/// which notice wins, that the other attention strips win over it, and that
/// a dismissal holds until the level worsens or the resource recovers.
final class ResourceWarningTests: XCTestCase {
    typealias RW = ResourceWarning

    func testMemoryLevelsFollowTheEngineVocabularyAndIgnoreUnknownWords() {
        XCTAssertNil(RW.memoryLevel("normal"))
        XCTAssertNil(RW.memoryLevel(nil))
        XCTAssertNil(RW.memoryLevel("elevated"))
        XCTAssertEqual(RW.memoryLevel("warn"), .warning)
        XCTAssertEqual(RW.memoryLevel("Critical"), .critical)
    }

    func testThermalWarnsOnlyFromSerious() {
        XCTAssertNil(RW.thermalLevel("nominal"))
        XCTAssertNil(RW.thermalLevel("fair"))
        XCTAssertEqual(RW.thermalLevel("serious"), .warning)
        XCTAssertEqual(RW.thermalLevel("critical"), .critical)
    }

    func testContextThresholdsAndCompactionSilence() {
        XCTAssertNil(RW.contextLevel(fraction: 0.84, isCompacting: false))
        XCTAssertEqual(RW.contextLevel(fraction: 0.85, isCompacting: false), .warning)
        XCTAssertEqual(RW.contextLevel(fraction: 0.95, isCompacting: false), .critical)
        XCTAssertNil(RW.contextLevel(fraction: 0.99, isCompacting: true))
        XCTAssertNil(RW.contextLevel(fraction: nil, isCompacting: false))
        XCTAssertNil(RW.contextLevel(fraction: .nan, isCompacting: false))
    }

    func testNoLevelsWithoutALoadedModel() {
        XCTAssertEqual(RW.levels(
            hasSession: false, memoryPressure: "critical", thermalLevel: "critical",
            contextFraction: 1, isCompacting: false), [:])
        XCTAssertEqual(RW.levels(
            hasSession: true, memoryPressure: "warn", thermalLevel: "nominal",
            contextFraction: 0.9, isCompacting: false), [.memory: .warning, .context: .warning])
    }

    func testHigherLevelWinsThenKindOrder() {
        XCTAssertEqual(
            RW.notice(levels: [.memory: .warning, .thermal: .critical], dismissed: [:], blockedByOtherStrip: false),
            RW.Notice(kind: .thermal, level: .critical))
        XCTAssertEqual(
            RW.notice(levels: [.thermal: .warning, .context: .warning, .memory: .warning], dismissed: [:], blockedByOtherStrip: false),
            RW.Notice(kind: .memory, level: .warning))
        XCTAssertEqual(
            RW.notice(levels: [.thermal: .warning, .context: .warning], dismissed: [:], blockedByOtherStrip: false),
            RW.Notice(kind: .context, level: .warning))
        XCTAssertNil(RW.notice(levels: [:], dismissed: [:], blockedByOtherStrip: false))
    }

    func testLoadFailureStallOrErrorStripWins() {
        XCTAssertNil(RW.notice(
            levels: [.memory: .critical], dismissed: [:], blockedByOtherStrip: true))
    }

    func testDismissHoldsUntilTheLevelWorsens() {
        let warn = RW.Notice(kind: .memory, level: .warning)
        let held = RW.dismissing(warn, from: [:])
        XCTAssertNil(RW.notice(levels: [.memory: .warning], dismissed: held, blockedByOtherStrip: false))
        XCTAssertEqual(
            RW.notice(levels: [.memory: .critical], dismissed: held, blockedByOtherStrip: false),
            RW.Notice(kind: .memory, level: .critical))
    }

    func testDismissingOneResourceRevealsTheNextOne() {
        let levels: [RW.Kind: RW.Level] = [.memory: .warning, .thermal: .warning]
        let held = RW.dismissing(RW.Notice(kind: .memory, level: .warning), from: [:])
        XCTAssertEqual(
            RW.notice(levels: levels, dismissed: held, blockedByOtherStrip: false),
            RW.Notice(kind: .thermal, level: .warning))
    }

    func testEasingKeepsTheHoldAndRecoveryClearsIt() {
        let held = RW.dismissing(RW.Notice(kind: .thermal, level: .critical), from: [:])
        // Critical to serious: still held, the user already saw worse.
        let eased = RW.prunedDismissals(held, levels: [.thermal: .warning])
        XCTAssertEqual(eased, [.thermal: .critical])
        XCTAssertNil(RW.notice(levels: [.thermal: .warning], dismissed: eased, blockedByOtherStrip: false))
        // Back to fine: the hold is forgotten, so the next crossing shows.
        let recovered = RW.prunedDismissals(eased, levels: [:])
        XCTAssertEqual(recovered, [:])
        XCTAssertEqual(
            RW.notice(levels: [.thermal: .warning], dismissed: recovered, blockedByOtherStrip: false),
            RW.Notice(kind: .thermal, level: .warning))
    }

    /// The percent key carries a literal `%` after an interpolation; this
    /// proves the catalog key matches what `AppStrings.localized` looks up.
    @MainActor
    func testStripCopyResolvesFromTheCatalog() {
        let context = RW.Notice(kind: .context, level: .warning)
        let english = ResourceWarningStripView.detail(context, percent: 87, language: .english)
        let german = ResourceWarningStripView.detail(context, percent: 87, language: .german)
        XCTAssertEqual(english, "This chat fills 87% of the context window. Compact it to keep room for replies.")
        XCTAssertTrue(german.contains("87"))
        XCTAssertNotEqual(german, english)
        for kind in RW.Kind.allCases {
            for level in [RW.Level.warning, .critical] {
                let notice = RW.Notice(kind: kind, level: level)
                XCTAssertNotEqual(
                    ResourceWarningStripView.title(notice, language: .german),
                    ResourceWarningStripView.title(notice, language: .english))
            }
        }
    }

    func testDismissingNeverLowersAnExistingHold() {
        let held: [RW.Kind: RW.Level] = [.context: .critical]
        XCTAssertEqual(
            RW.dismissing(RW.Notice(kind: .context, level: .warning), from: held),
            [.context: .critical])
    }
}
