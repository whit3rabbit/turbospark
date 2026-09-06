import XCTest

@testable import TurboSparkApp

/// The plan file's generated name.
///
/// Pure: a seeded generator makes every case deterministic, so nothing here
/// depends on a clock, a directory or a random draw.
final class PlanFileNamerTests: XCTestCase {
    /// SplitMix64, so the same seed gives the same name on every machine.
    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }

    func testTheSameSeedProducesTheSameName() {
        var a = SeededGenerator(state: 42)
        var b = SeededGenerator(state: 42)

        XCTAssertEqual(
            PlanFileNamer.fileName(using: &a),
            PlanFileNamer.fileName(using: &b))
    }

    func testANameIsTwoWordsFromTheTwoListsAndEndsInMarkdown() {
        var generator = SeededGenerator(state: 7)
        let name = PlanFileNamer.fileName(using: &generator)

        XCTAssertTrue(name.hasSuffix(".md"), name)
        let parts = name.replacingOccurrences(of: ".md", with: "").split(separator: "-")
        XCTAssertEqual(parts.count, 3, "plan-<adjective>-<noun>: \(name)")
        XCTAssertEqual(String(parts[0]), "plan")
        XCTAssertTrue(PlanFileNamer.adjectives.contains(String(parts[1])), name)
        XCTAssertTrue(PlanFileNamer.nouns.contains(String(parts[2])), name)
    }

    func testATitleBecomesTheLeadingSlug() {
        var generator = SeededGenerator(state: 3)
        let name = PlanFileNamer.fileName(title: "Add the Artifacts Panel!", using: &generator)

        XCTAssertTrue(name.hasPrefix("add-the-artifacts-panel-"), name)
    }

    func testASlugIsBoundedAndCarriesNoPathSeparators() {
        let slug = PlanFileNamer.slug(String(repeating: "a/b ", count: 60))

        XCTAssertNotNil(slug)
        XCTAssertLessThanOrEqual(slug!.count, 48)
        XCTAssertFalse(slug!.contains("/"), "a title must never become a path")
        XCTAssertFalse(slug!.contains(" "))
    }

    func testATitleWithNothingUsableFallsBackRatherThanProducingABareTag() {
        XCTAssertNil(PlanFileNamer.slug("///"))
        XCTAssertNil(PlanFileNamer.slug(nil))

        var generator = SeededGenerator(state: 11)
        XCTAssertTrue(PlanFileNamer.fileName(title: "///", using: &generator).hasPrefix("plan-"))
    }

    func testACollisionPicksADifferentName() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("plan-namer-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        var probe = SeededGenerator(state: 99)
        let taken = PlanFileNamer.fileName(using: &probe)
        try Data("taken".utf8).write(to: directory.appendingPathComponent(taken))

        var generator = SeededGenerator(state: 99)
        let chosen = PlanFileNamer.uniqueFileName(in: directory, using: &generator)

        XCTAssertNotEqual(chosen, taken, "a plan must never overwrite a plan")
    }
}
