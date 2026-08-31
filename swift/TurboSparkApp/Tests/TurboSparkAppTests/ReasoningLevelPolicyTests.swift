import TurboSpark
import XCTest

@testable import TurboSparkApp

/// What the reasoning picker offers, and what happens to a preference carried
/// between checkpoints that disagree about the spellings.
///
/// These need no model, no Metal device and no install, which is the whole
/// reason `ReasoningLevelPolicy` exists as a value: `AppModel.info` is
/// computed from `session`, so the same assertions written against `AppModel`
/// could only run with a 13 GB install on disk, and so never ran.
final class ReasoningLevelPolicyTests: XCTestCase {
    /// The real sets, and the reason a family table cannot produce them.
    private let qwen38: [GenerateOptions.Reasoning] = [.off, .low, .medium, .xhigh]
    private let gptoss: [GenerateOptions.Reasoning] = [.off, .low, .medium, .high]

    // MARK: - What gets offered

    /// **THE BUG THIS FILE EXISTS FOR.** The menu used to be `allCases` for
    /// every `.level` checkpoint, so Qwen 3.8 offered High -- which its
    /// template REJECTS by name, failing the turn.
    func testALevelCheckpointOffersItsOwnSetAndNotTheUnion() {
        let offered = ReasoningLevelPolicy.offered(support: .level, efforts: qwen38)
        XCTAssertEqual(offered, qwen38)
        XCTAssertFalse(offered.contains(.high), "qwen38 raises on high")

        let other = ReasoningLevelPolicy.offered(support: .level, efforts: gptoss)
        XCTAssertEqual(other, gptoss)
        XCTAssertFalse(other.contains(.xhigh), "gpt-oss has no xhigh")

        // The two sets differ, which is what says this is read rather than
        // assumed. A table producing one answer for both would pass every
        // other case in this file.
        XCTAssertNotEqual(offered, other)
    }

    /// No session means nothing has read the template, and there is no
    /// defensible guess (root Gotcha 56). This used to fall back to a
    /// hardcoded family set and then to `allCases`.
    func testNoSessionOffersOnlyOff() {
        XCTAssertEqual(ReasoningLevelPolicy.offered(support: nil, efforts: []), [.off])
        XCTAssertEqual(
            ReasoningLevelPolicy.offered(support: nil, efforts: qwen38), [.off],
            "levels without a support state are not evidence about a loaded model"
        )
    }

    /// A checkpoint whose template names no reasoning key can express one
    /// state, so it offers one.
    func testACheckpointWithNoReasoningKnobOffersOnlyOff() {
        XCTAssertEqual(ReasoningLevelPolicy.offered(support: SessionInfo.ReasoningSupport.none, efforts: qwen38), [.off])
    }

    /// Toggle-only drops the level, so four on-levels are four names for one
    /// prompt. Offering them is what read as "Extra High" surviving a model
    /// switch it did not apply to.
    func testAToggleOnlyCheckpointOffersOffAndOneOnLevel() {
        let offered = ReasoningLevelPolicy.offered(
            support: .toggleOnly, efforts: [.off, .low, .medium, .high, .xhigh])
        XCTAssertEqual(offered, [.off, .medium])
    }

    /// An empty array is a decoded wire value, and an empty menu would strand
    /// a user with no way to change the setting at all.
    func testAnEmptyLevelSetStillOffersOff() {
        XCTAssertEqual(ReasoningLevelPolicy.offered(support: .level, efforts: []), [.off])
    }

    // MARK: - Carrying a preference between checkpoints

    /// The case the clamp exists for: the two top spellings stand in for each
    /// other. Dropping to `.off` here silently turns thinking off for a user
    /// who explicitly turned it on.
    func testTheTopSpellingsStandInForEachOther() {
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .high, in: qwen38), .xhigh)
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .xhigh, in: gptoss), .high)
    }

    /// An expressible level is left exactly alone, in both directions.
    func testAnExpressibleLevelIsUntouched() {
        for level in qwen38 {
            XCTAssertEqual(ReasoningLevelPolicy.nearest(to: level, in: qwen38), level)
        }
        for level in gptoss {
            XCTAssertEqual(ReasoningLevelPolicy.nearest(to: level, in: gptoss), level)
        }
    }

    /// Off never becomes thinking. The clamp only ever rescues a request TO
    /// think, so it must not manufacture one -- a user who turned reasoning
    /// off and switched models did not ask for the highest available level.
    ///
    /// **THE THIRD CASE IS THE ONLY ONE THAT REACHES THE GUARD, and without
    /// it the guard is untested dead code.** Found by mutation: deleting
    /// `if wanted == .off { return .off }` left all 13 cases green, because
    /// the first two pass sets that CONTAIN `.off` and are answered by the
    /// containment check one line earlier. A set without `.off` is what makes
    /// the guard load-bearing, and it is the exact input on which its absence
    /// clamps a user who turned thinking off UP to the highest level going.
    func testOffIsNeverClampedUpwards() {
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .off, in: qwen38), .off)
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .off, in: [.off, .high]), .off)
        XCTAssertEqual(
            ReasoningLevelPolicy.nearest(to: .off, in: [.low, .high]), .off,
            "a set without off must still not turn thinking on"
        )
    }

    /// The invariant that guard backstops, pinned where it is established:
    /// every offered set contains `.off`, so a user always has a way back to
    /// not thinking whatever the checkpoint.
    func testEveryOfferedSetContainsOff() {
        let supports: [SessionInfo.ReasoningSupport?] = [
            nil, .level, .toggleOnly, SessionInfo.ReasoningSupport.none,
        ]
        for support in supports {
            for efforts in [[], qwen38, gptoss] as [[GenerateOptions.Reasoning]] {
                let offered = ReasoningLevelPolicy.offered(support: support, efforts: efforts)
                XCTAssertTrue(
                    offered.contains(.off),
                    "support \(String(describing: support)) offered \(offered) with no way to turn thinking off"
                )
                XCTAssertEqual(offered.first, .off, "the menu opens at off")
            }
        }
    }

    /// With no on-level available at all, a request to think has nowhere to
    /// go and lands on off.
    func testAThinkingRequestFallsToOffWhenNothingExpressesIt() {
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .high, in: [.off]), .off)
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .medium, in: [.off]), .off)
    }

    /// Anything else keeps the request to think and takes the highest rung
    /// the checkpoint has.
    func testAMidLevelFallsToTheHighestAvailableOnLevel() {
        XCTAssertEqual(ReasoningLevelPolicy.nearest(to: .medium, in: [.off, .low]), .low)
        XCTAssertEqual(
            ReasoningLevelPolicy.nearest(to: .low, in: [.off, .medium, .xhigh]), .xhigh,
            "the highest, not the first: allCases is ascending"
        )
    }

    /// Every offered set survives the clamp, which is the property the app
    /// actually depends on -- `open(_:)` assigns the result straight to
    /// `reasoning`, so a value outside the menu would leave the picker blank.
    func testTheClampAlwaysLandsInsideTheOfferedSet() {
        let sets: [[GenerateOptions.Reasoning]] = [qwen38, gptoss, [.off], [.off, .medium]]
        for offered in sets {
            for wanted in GenerateOptions.Reasoning.allCases {
                let landed = ReasoningLevelPolicy.nearest(to: wanted, in: offered)
                XCTAssertTrue(
                    offered.contains(landed),
                    "\(wanted) landed on \(landed), which \(offered) does not offer"
                )
            }
        }
    }

    // MARK: - Labels

    /// A toggle-only checkpoint has no effort levels, so printing one beside
    /// its on-state promises what the template cannot deliver.
    func testAToggleOnlyOnStateIsLabelledAsASwitch() {
        XCTAssertEqual(ReasoningLevelPolicy.label(for: .medium, support: .toggleOnly), "On")
        XCTAssertEqual(ReasoningLevelPolicy.label(for: .off, support: .toggleOnly), "Off")
        XCTAssertNotEqual(
            ReasoningLevelPolicy.description(for: .medium, support: .toggleOnly),
            GenerateOptions.Reasoning.medium.descriptionText
        )
    }

    /// Every other case prints the level's own name, including with no
    /// session, so the settings default picker reads normally.
    func testEveryOtherSupportStatePrintsTheLevelsOwnName() {
        for support: SessionInfo.ReasoningSupport? in [nil, .level, SessionInfo.ReasoningSupport.none] {
            for level in GenerateOptions.Reasoning.allCases {
                XCTAssertEqual(ReasoningLevelPolicy.label(for: level, support: support), level.label)
                XCTAssertEqual(
                    ReasoningLevelPolicy.description(for: level, support: support),
                    level.descriptionText
                )
            }
        }
    }
}
