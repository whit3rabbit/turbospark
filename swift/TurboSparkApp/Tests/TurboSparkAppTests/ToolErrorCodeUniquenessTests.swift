import Foundation
import XCTest

/// Guards against two different tools throwing `NSError(domain:code:)` with the
/// same (domain, code) pair. Callers that branch on the code cannot tell the
/// errors apart, and B2 in the issues list was exactly that: PushNotification
/// and Config reused cron's 60-62, and SendFeedback reused Batch's 35.
///
/// Rule: one source file (a type and its `+Extension` files count as one
/// owner) per (domain, code). Several messages with the same code inside one
/// owner are fine: they are the same class of failure from the same tool.
///
/// The scan reads the checked-in sources through `#filePath`, so it does not
/// depend on the code under test.
final class ToolErrorCodeUniquenessTests: XCTestCase {
    /// Cross-owner reuse that predates this guard. Each entry is a known
    /// collision that still needs its own renumbering change; do not add to
    /// this list to silence a new collision, pick an unused code instead.
    /// Keys are "domain:code".
    private static let legacySharedCodes: Set<String> = [
        "McpClientEngine:5", "McpClientEngine:7",
        "TurboSparkMemory:2",
        "TurboSparkSkill:5",
        "TurboSparkTool:4",
        "TurboSparkTool:20", "TurboSparkTool:21", "TurboSparkTool:22",
        "TurboSparkTool:23", "TurboSparkTool:24",
        "TurboSparkTool:26", "TurboSparkTool:27",
        "TurboSparkTool:30", "TurboSparkTool:31", "TurboSparkTool:32",
        "TurboSparkTool:33", "TurboSparkTool:34",
        "TurboSparkTool:40", "TurboSparkTool:41", "TurboSparkTool:42",
        "TurboSparkTool:43", "TurboSparkTool:44", "TurboSparkTool:45",
        "TurboSparkTool:46",
        "TurboSparkTool:50",
        "TurboSparkTool:80",
    ]

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // TurboSparkAppTests/
            .deletingLastPathComponent()  // Tests/
            .deletingLastPathComponent()  // TurboSparkApp/
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    /// Maps "domain:code" to the set of owners that throw it.
    private func scanOwners(in root: URL) throws -> [String: Set<String>] {
        let regex = try NSRegularExpression(
            pattern: #"domain:\s*"([A-Za-z]+)"\s*,\s*code:\s*(\d+)"#)
        var owners: [String: Set<String>] = [:]
        let files = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil)
        while let url = files?.nextObject() as? URL {
            guard url.pathExtension == "swift" else { continue }
            let text = try String(contentsOf: url, encoding: .utf8)
            let range = NSRange(text.startIndex..., in: text)
            let owner = url.deletingPathExtension().lastPathComponent
                .components(separatedBy: "+")[0]
            for match in regex.matches(in: text, range: range) {
                guard let d = Range(match.range(at: 1), in: text),
                      let c = Range(match.range(at: 2), in: text) else { continue }
                owners["\(text[d]):\(text[c])", default: []].insert(owner)
            }
        }
        return owners
    }

    func testNoNewCrossOwnerDuplicateErrorCodes() throws {
        let owners = try scanOwners(in: sourcesRoot)
        XCTAssertGreaterThan(owners.count, 50, "scan found too few codes; the path or regex is broken")

        let collisions = owners.filter { $0.value.count > 1 && !Self.legacySharedCodes.contains($0.key) }
        let report = collisions.keys.sorted().map { "\($0) used by \(owners[$0]!.sorted())" }
        XCTAssertTrue(collisions.isEmpty, "Duplicate error codes across owners: \(report)")
    }

    func testLegacyBaselineHasNoStaleEntries() throws {
        // A fixed collision must leave the baseline, or the list rots into a
        // blanket exemption.
        let owners = try scanOwners(in: sourcesRoot)
        let stale = Self.legacySharedCodes.filter { (owners[$0]?.count ?? 0) < 2 }
        XCTAssertTrue(stale.isEmpty, "Baseline entries that no longer collide: \(stale.sorted())")
    }

    func testB2CodesAreSplitByTool() throws {
        let owners = try scanOwners(in: sourcesRoot)
        XCTAssertEqual(owners["TurboSparkTool:35"], ["BatchToolExecutor"])
        XCTAssertEqual(owners["TurboSparkTool:60"], ["CronScheduler"])
        XCTAssertEqual(owners["TurboSparkTool:61"], ["CronScheduler"])
        XCTAssertEqual(owners["TurboSparkTool:62"], ["CronScheduler"])
    }
}
