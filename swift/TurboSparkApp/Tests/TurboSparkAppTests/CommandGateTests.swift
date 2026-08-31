import XCTest

@testable import TurboSparkApp

/// **THE PORT IS ONLY AS GOOD AS THE ORACLE, AND THE ORACLE IS ONLY AS GOOD AS
/// ITS SPREAD.** A hashed linear model reimplemented in a second language fails
/// silently: it loads, it scores, it returns plausible probabilities, and it is
/// a different model. Two real bugs were caught here and neither was visible
/// any other way.
///
/// - Dropping the `\b` anchors from the word token pattern made
///   `rm -rf ~/Documents` tokenize as `["rm", "-rf", "/documents"]` where
///   sklearn gives `["rm", "rf", "documents"]`. It scored 0.9320 against a true
///   0.9931, which reads as an ordinary model rather than a broken port.
/// - `split(separator: " ")` is not Python's `str.split()`, which splits on any
///   whitespace run.
final class CommandGateTests: XCTestCase {

    private struct OracleCase: Decodable {
        let command: String
        let hazard: Double
        let obfuscation: Double
    }

    /// The trainer applies base64 and hex decode augmentation before hashing.
    /// The Swift side does not implement it, so these two cases are excluded by
    /// name rather than quietly passing on a weaker comparison.
    private static let notPorted: Set<String> = [
        "printf '%s' 'bHMgLWxh' | base64 -d | sh",
        "echo 726d202d7266207e | xxd -r -p | sh",
    ]

    private func loadOracle() throws -> [OracleCase] {
        let url = try XCTUnwrap(
            CommandGate.oracleFixtureURL,
            "oracle fixture missing from the app bundle")
        return try JSONDecoder().decode([OracleCase].self, from: Data(contentsOf: url))
    }

    // MARK: - The fixture has to be able to fail

    /// A port that returns a constant passes a fixture whose cases all score
    /// alike, and a constant is exactly the defect that put the published
    /// models in this state. Check the instrument before trusting it.
    func testTheOracleFixtureDiscriminates() throws {
        let cases = try loadOracle()
        XCTAssertGreaterThanOrEqual(cases.count, 10)
        let hazards = cases.map(\.hazard)
        let span = try XCTUnwrap(hazards.max()) - XCTUnwrap(hazards.min())
        XCTAssertGreaterThan(
            span, 0.5,
            "the fixture spans \(span), so a constant-output port would pass it")
    }

    // MARK: - Parity

    func testSwiftScoringReproducesTheTrainedModel() throws {
        var compared = 0
        for oracle in try loadOracle()
        where !Self.notPorted.contains(oracle.command) && !oracle.command.isEmpty {
            let scores = try XCTUnwrap(
                CommandGate.score(oracle.command),
                "no score for '\(oracle.command)'; is permission-gate.bin in the bundle?")
            compared += 1
            XCTAssertEqual(
                scores.hazard, oracle.hazard, accuracy: 1e-6,
                "hazard drifted on '\(oracle.command)'")
            XCTAssertEqual(
                scores.obfuscation, oracle.obfuscation, accuracy: 1e-6,
                "obfuscation drifted on '\(oracle.command)'")
        }
        XCTAssertGreaterThanOrEqual(
            compared, 10, "too few cases survived the exclusions to prove anything")
    }

    /// The one deliberate divergence from the trainer. Python scores the empty
    /// string like any other row; `CommandGate` returns nil, because "no
    /// opinion" is the honest answer for a command with nothing in it and the
    /// callers both treat nil as "stay quiet". Asserted rather than skipped so
    /// the difference is a decision on the record.
    func testTheEmptyCommandHasNoOpinionByDesign() {
        XCTAssertNil(CommandGate.score(""))
        XCTAssertNil(CommandGate.advisoryReason(for: ""))
        XCTAssertNil(CommandGate.veto(for: ""))
    }

    /// The two landmines, pinned directly so a regression names its own cause
    /// rather than surfacing as a drifted probability.
    func testWordTokenizationKeepsTheWordBoundaryAnchors() throws {
        let pattern = try NSRegularExpression(pattern: "\\b[\\w./:-]+\\b")
        let grams = HashedFeatureVectorizer.wordNgrams(
            "rm -rf ~/documents", 1, 1, pattern: pattern)
        XCTAssertEqual(
            grams, ["rm", "rf", "documents"],
            "leading '-' and '/' must be stripped by the \\b anchors, as sklearn does")
    }

    func testCharNgramsSplitOnAnyWhitespaceRun() {
        var singleSpace: [String] = []
        HashedFeatureVectorizer.charNgrams("ab cd", 3, 3) {
            singleSpace.append(String(decoding: $0, as: UTF8.self))
        }
        var mixedWhitespace: [String] = []
        HashedFeatureVectorizer.charNgrams("ab \t\n cd", 3, 3) {
            mixedWhitespace.append(String(decoding: $0, as: UTF8.self))
        }
        XCTAssertEqual(
            singleSpace, mixedWhitespace,
            "Python's str.split() collapses any whitespace run, so these must agree")
    }

    /// A word shorter than n is emitted once whole, and longer n are then
    /// skipped for that word rather than tried.
    func testShortWordsAreEmittedOnceAndStopTheLadder() {
        var grams: [String] = []
        HashedFeatureVectorizer.charNgrams("ab", 3, 5) {
            grams.append(String(decoding: $0, as: UTF8.self))
        }
        XCTAssertEqual(grams, [" ab", "ab ", " ab "])
    }

    // MARK: - The gate cannot be widened by the model

    /// The veto ships off, so the shipped classifier must be byte-identical to
    /// the allowlist-only behaviour. This is the guard on the DEFAULT, and it
    /// is separate from the corpus tests in `TerminalRiskGateTests`.
    func testTheVetoIsOffByDefault() {
        XCTAssertFalse(
            CommandGate.vetoEnabled,
            "shipping the veto on reddens testOrdinaryDevelopmentCommandsStillRunUnprompted")
    }

    /// Even switched ON, the model must not rescue a single evasion. It runs
    /// after the allowlist, so it can only ever add friction.
    func testEnablingTheVetoCannotPromoteAnyEvasion() {
        let previous = CommandGate.vetoEnabled
        CommandGate.vetoEnabled = true
        defer { CommandGate.vetoEnabled = previous }

        for (command, mechanism) in TerminalRiskGateTests.evasionsThatMustAsk {
            let risk = ToolRiskClassifier.assessTerminalCommand(command)
            XCTAssertTrue(
                risk.isHighRisk,
                "\(mechanism): '\(command)' stopped asking once the model was enabled")
        }
    }

    // MARK: - The setting has to survive a round trip

    /// Every store here writes whole and swallows a decode failure, so a field
    /// that does not round-trip is discarded without an error (swift/CLAUDE.md
    /// Gotcha 13). Both directions are checked because a default of `false`
    /// hides a dropped field perfectly.
    func testTheVetoSettingRoundTripsBothWays() throws {
        for enabled in [true, false] {
            let encoded = try JSONEncoder().encode(
                MacAppSettings(commandAdvisoryVeto: enabled))
            let decoded = try JSONDecoder().decode(MacAppSettings.self, from: encoded)
            XCTAssertEqual(decoded.commandAdvisoryVeto, enabled)
        }
    }

    /// A settings file written before this field existed must still load, and
    /// must land on the safe value rather than throwing the whole file away.
    func testSettingsWrittenBeforeThisFieldStillLoad() throws {
        let legacy = Data(#"{"temperature": 0.2, "guardrailsMode": "select"}"#.utf8)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: legacy)
        XCTAssertFalse(decoded.commandAdvisoryVeto)
    }

    /// The advisory path runs on every `.ask` verdict, so it must never throw,
    /// hang, or return a reason for an empty or exotic command.
    func testAdvisoryIsSafeOnDegenerateInput() {
        XCTAssertNil(CommandGate.score(""))
        for command in ["", "   ", "\n\n", "café naïve über", String(repeating: "a", count: 50_000)] {
            _ = CommandGate.advisoryReason(for: command)
        }
    }
}
