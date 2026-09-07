import XCTest

@testable import TurboSparkApp

/// `ShellOutputFormatting.compactWithSpill` and the spill-root path
/// allowlist. Under the cap the function must be EXACTLY the old compaction
/// and must not touch the disk; over the cap the full text must survive on
/// disk and the model-facing text must name where. The allowlist is what
/// makes those files readable through `read_file`, which otherwise refuses
/// every absolute path -- so it must accept a spill path and nothing else.
final class ShellSpillTests: XCTestCase {
    override func tearDown() {
        // Remove only what the tests wrote (top-level .txt files in the
        // real spill root), leaving anything a real session spilled.
        let fm = FileManager.default
        let root = ShellOutputFormatting.spillRootURL
        if let contents = try? fm.contentsOfDirectory(at: root, includingPropertiesForKeys: nil) {
            for url in contents where url.pathExtension == "txt" && url.lastPathComponent.contains("spilltest") {
                try? fm.removeItem(at: url)
            }
        }
        super.tearDown()
    }

    private func bigText(_ chars: Int) -> String {
        String(repeating: "x", count: chars)
    }

    // MARK: - under the cap: compaction only

    func testUnderTheCapNothingIsWrittenAndTextIsUnchanged() {
        let root = ShellOutputFormatting.spillRootURL
        let before = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        let text = bigText(100)
        let out = ShellOutputFormatting.compactWithSpill(text, label: "spilltest small")
        XCTAssertEqual(out, text)
        let after = (try? FileManager.default.contentsOfDirectory(atPath: root.path)) ?? []
        XCTAssertEqual(after, before, "An under-cap output must not write anything.")
    }

    // MARK: - over the cap: spill + note

    func testOverTheCapTheFullTextLandsOnDiskAndIsNamed() throws {
        let text = bigText(ShellOutputFormatting.maxModelOutputChars + 5_000)
        let out = ShellOutputFormatting.compactWithSpill(text, label: "spilltest big")
        // The model still sees head+tail...
        XCTAssertTrue(out.contains("chars truncated"))
        // ...plus a path that resolves to the full text. Asserted before
        // any parsing, so a mutation that drops the note fails here rather
        // than crashing the bundle on a missing array element.
        guard out.contains("saved to ") else {
            XCTFail("The over-cap note must name the spill file. Got: \(out.suffix(300))")
            return
        }
        let path = out.components(separatedBy: "saved to ")[1]
            .components(separatedBy: ". ").first ?? ""
        let url = URL(fileURLWithPath: path)
        XCTAssertTrue(ShellOutputFormatting.isUnderSpillRoot(url), "Path from the note: \(path)")
        let spilled = try String(contentsOf: url, encoding: .utf8)
        XCTAssertEqual(spilled, text, "The spill file must hold the FULL output.")
        try? FileManager.default.removeItem(at: url)
    }

    func testADeterministicSpillNameIsOverwrittenNotAccumulated() throws {
        let text = bigText(ShellOutputFormatting.maxModelOutputChars + 10)
        _ = ShellOutputFormatting.compactWithSpill(text, label: "spilltest", spillName: "spilltest-det")
        _ = ShellOutputFormatting.compactWithSpill(text, label: "spilltest", spillName: "spilltest-det")
        let fm = FileManager.default
        let root = ShellOutputFormatting.spillRootURL
        let matches = try fm.contentsOfDirectory(atPath: root.path)
            .filter { $0.contains("spilltest-det") }
        XCTAssertEqual(matches.count, 1, "One shell, one spill file, whatever the poll count.")
        try? fm.removeItem(at: root.appendingPathComponent(matches[0]))
    }

    // MARK: - pruning

    func testPruneKeepsTheNewestFiles() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("spillprune-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let fm = FileManager.default
        for index in 0..<25 {
            let url = root.appendingPathComponent("f\(String(format: "%02d", index)).txt")
            try "x".write(to: url, atomically: true, encoding: .utf8)
            // Distinct mtimes: oldest first, so the KEPT set is the high
            // indices and what the prune drops is exactly 00..04.
            try fm.setAttributes(
                [.modificationDate: Date(timeInterval: Double(index), since: Date(timeIntervalSince1970: 0))],
                ofItemAtPath: url.path)
        }
        ShellOutputFormatting.pruneSpillFiles(root: root, keeping: 20)
        let survivors = try fm.contentsOfDirectory(atPath: root.path).sorted()
        XCTAssertEqual(survivors.count, 20)
        XCTAssertEqual(survivors.first, "f05.txt", "The five oldest must go.")
        XCTAssertEqual(survivors.last, "f24.txt")
    }

    // MARK: - the read_file allowlist

    func testTheSpillRootIsAcceptedAndEverythingElseAbsoluteIsRefused() throws {
        let inside = ShellOutputFormatting.spillRootURL.appendingPathComponent("some-file.txt")
        let resolved = try AppToolRegistry.resolveSecurePath(
            relPath: inside.path, rootURL: FileManager.default.temporaryDirectory)
        XCTAssertEqual(resolved.standardizedFileURL.resolvingSymlinksInPath().path,
                       inside.standardizedFileURL.resolvingSymlinksInPath().path)

        let outside = FileManager.default.temporaryDirectory
            .appendingPathComponent("not-a-spill-\(UUID().uuidString).txt")
        XCTAssertThrowsError(
            try AppToolRegistry.resolveSecurePath(
                relPath: outside.path, rootURL: FileManager.default.temporaryDirectory)
        ) { error in
            XCTAssertEqual((error as NSError).code, 13)
        }
    }

    func testAPathUnderTheRootButEscapingThroughASubdirectoryIsStillContained() throws {
        // `..` inside an absolute path must not pivot out of the spill root.
        let inside = ShellOutputFormatting.spillRootURL.appendingPathComponent("x/../../../etc/hosts")
        XCTAssertThrowsError(
            try AppToolRegistry.resolveSecurePath(
                relPath: inside.path, rootURL: FileManager.default.temporaryDirectory)
        ) { error in
            XCTAssertEqual((error as NSError).code, 13)
        }
    }
}
