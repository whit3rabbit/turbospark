import XCTest
@testable import TurboSparkApp

/// OC-11: the hardened run. Each guard has a mutation check noted beside it.
final class FormatterRunnerTests: XCTestCase {
    private func run(
        _ fixture: FormatterFixture, _ row: FormatterRow, path: String = "Sources/A.swift",
        timeout: TimeInterval = 10, env: [String: String] = ["PATH": "/usr/bin:/bin"]
    ) async -> FormatterOutcome {
        var runner = FormatterRunner(timeoutSeconds: timeout)
        runner.parentEnvironment = env
        return await runner.format(path: path, row: row, rootURL: fixture.project, searchPath: nil)
    }

    func testFormatsTheFileKeepsItsModeAndLeavesNoTempCopy() async throws {
        let fixture = try FormatterFixture()
        let target = fixture.project.appendingPathComponent("Sources/A.swift")
        try FileManager.default.setAttributes([.posixPermissions: 0o640], ofItemAtPath: target.path)
        let outcome = await run(fixture, try fixture.row(FormatterFixture.upcase))
        guard case .formatted(let change) = outcome else { return XCTFail("\(outcome)") }
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "LET A = 1\n")
        XCTAssertEqual(change.added.first?.line, 1)
        XCTAssertEqual(change.added.first?.text, "LET A = 1")
        let mode = try FileManager.default.attributesOfItem(atPath: target.path)[.posixPermissions] as? Int
        XCTAssertEqual(mode, 0o640)
        XCTAssertEqual(fixture.leftoverTemps(), [])
    }

    /// Only a temporary sibling copy is handed to the formatter, never the real
    /// file. Mutation check: pass `target.path` instead of `temp.path` to the
    /// launch and the logged argument equals the real path.
    func testFormatterReceivesOnlyTheTemporaryCopy() async throws {
        let fixture = try FormatterFixture(files: ["Sources/A.swift": "x\n", "Sources/B.swift": "y\n"])
        let row = try fixture.row(#"echo "$#|$1" > "\#(fixture.log.path)""#)
        _ = await run(fixture, row)
        let parts = fixture.logText.trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "|")
        XCTAssertEqual(parts.first, "1", "exactly one path argument")
        let arg = String(parts.last ?? "")
        XCTAssertNotEqual(arg, fixture.project.appendingPathComponent("Sources/A.swift").path)
        XCTAssertTrue((arg as NSString).lastPathComponent.hasPrefix(FormatterRunner.tempPrefix))
        XCTAssertEqual((arg as NSString).deletingLastPathComponent, fixture.project.appendingPathComponent("Sources").path)
        XCTAssertEqual(try fixture.read("Sources/B.swift"), "y\n")
    }

    func testFailureKeepsTheOriginalAndReportsAnEscapedTail() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(#"echo "bad <token> \#u{1B}[31mred\#u{1B}[0m" >&2; exit 3"#)
        let outcome = await run(fixture, row)
        guard case .failed(let reason) = outcome else { return XCTFail("\(outcome)") }
        XCTAssertTrue(reason.hasPrefix("exit 3"))
        XCTAssertTrue(reason.contains("&lt;token&gt;"))
        XCTAssertFalse(reason.contains("\u{1B}"))
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
        XCTAssertEqual(fixture.leftoverTemps(), [])
    }

    /// Mutation check: replace `outputCapBytes`/`errorTailChars` truncation
    /// with the whole text and the length bound fails.
    func testErrorTailIsBounded() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row(#"head -c 100000 /dev/zero | tr '\0' 'e' >&2; exit 1"#)
        guard case .failed(let reason) = await run(fixture, row) else { return XCTFail("expected failure") }
        XCTAssertLessThan(reason.count, FormatterRunner.errorTailChars + 40)
    }

    /// Mutation check: set the timeout to 120 and this takes 30 s and fails.
    func testHangingFormatterIsKilledAtTheTimeout() async throws {
        let fixture = try FormatterFixture()
        let row = try fixture.row("sleep 30")
        let started = Date()
        let outcome = await run(fixture, row, timeout: 0.5)
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
        guard case .failed(let reason) = outcome else { return XCTFail("\(outcome)") }
        XCTAssertTrue(reason.contains("timed out"))
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
        XCTAssertEqual(fixture.leftoverTemps(), [])
    }

    /// A user save landing while the formatter runs must not be overwritten.
    /// Mutation check: drop the `current == original` guard and the user's
    /// line is lost.
    func testFileChangedDuringTheRunIsNotReplaced() async throws {
        let fixture = try FormatterFixture()
        let real = fixture.project.appendingPathComponent("Sources/A.swift").path
        let row = try fixture.row(#"echo "user line" >> "\#(real)"; "# + FormatterFixture.upcase)
        guard case .failed(let reason) = await run(fixture, row) else { return XCTFail("expected failure") }
        XCTAssertTrue(reason.contains("changed while"))
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\nuser line\n")
    }

    func testUnchangedFileAddsNothing() async throws {
        let fixture = try FormatterFixture()
        let outcome = await run(fixture, try fixture.row("exit 0"))
        XCTAssertEqual(outcome, .unchanged)
    }

    func testEmptyOutputIsRefused() async throws {
        let fixture = try FormatterFixture()
        let outcome = await run(fixture, try fixture.row(#": > "$1""#))
        guard case .failed = outcome else { return XCTFail("\(outcome)") }
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
    }

    // MARK: workspace rooting

    /// Mutation check: replace `resolvedIfContained` with the lexical join and
    /// the escaping symlink and the absolute outside path both run the script.
    func testPathsOutsideTheProjectAndEscapingSymlinksAreSkipped() async throws {
        let fixture = try FormatterFixture()
        let outside = fixture.tools.appendingPathComponent("Outside.swift")
        try "let o = 1\n".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: fixture.project.appendingPathComponent("Sources/Link.swift"), withDestinationURL: outside)
        let row = try fixture.row(#"echo ran >> "\#(fixture.log.path)"; "# + FormatterFixture.upcase)
        for path in [outside.path, "Sources/Link.swift", "../tools/Outside.swift"] {
            let outcome = await run(fixture, row, path: path)
            XCTAssertEqual(outcome, .skipped, path)
        }
        XCTAssertEqual(fixture.logText, "", "the formatter never started")
        XCTAssertEqual(try String(contentsOf: outside, encoding: .utf8), "let o = 1\n")
    }

    func testSymlinkInsideTheProjectFormatsItsTarget() async throws {
        let fixture = try FormatterFixture()
        try FileManager.default.createSymbolicLink(
            at: fixture.project.appendingPathComponent("Alias.swift"),
            withDestinationURL: fixture.project.appendingPathComponent("Sources/A.swift"))
        guard case .formatted = await run(fixture, try fixture.row(FormatterFixture.upcase), path: "Alias.swift") else {
            return XCTFail("expected a format")
        }
        let alias = fixture.project.appendingPathComponent("Alias.swift")
        XCTAssertEqual(try FileManager.default.destinationOfSymbolicLink(atPath: alias.path),
                       fixture.project.appendingPathComponent("Sources/A.swift").path, "the link itself survives")
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "LET A = 1\n")
    }

    /// Mutation check: remove the `isSensitivePath` guards and `.env.swift` formats.
    func testSensitiveAndGitPathsAreSkipped() async throws {
        let fixture = try FormatterFixture(files: [
            ".env.swift": "k = 1\n", ".git/hooks/pre.swift": "h = 1\n", "Sources/A.swift": "a\n"])
        let row = try fixture.row(#"echo ran >> "\#(fixture.log.path)"; "# + FormatterFixture.upcase)
        for path in [".env.swift", ".git/hooks/pre.swift"] {
            let outcome = await run(fixture, row, path: path)
            XCTAssertEqual(outcome, .skipped, path)
        }
        XCTAssertEqual(fixture.logText, "")
    }

    /// The spelling the agent used counts too: a sensitive-looking name that
    /// links to an ordinary file is still refused. Mutation check: drop the
    /// first `isSensitivePath` guard in `format` and this runs the script.
    func testSensitiveSpellingOfAnOrdinaryFileIsSkipped() async throws {
        let fixture = try FormatterFixture()
        try FileManager.default.createSymbolicLink(
            at: fixture.project.appendingPathComponent(".env.swift"),
            withDestinationURL: fixture.project.appendingPathComponent("Sources/A.swift"))
        let row = try fixture.row(#"echo ran >> "\#(fixture.log.path)""#)
        let outcome = await run(fixture, row, path: ".env.swift")
        XCTAssertEqual(outcome, .skipped)
        XCTAssertEqual(fixture.logText, "")
    }

    func testOversizedAndBinaryFilesAreSkipped() async throws {
        let fixture = try FormatterFixture()
        let big = fixture.project.appendingPathComponent("Big.swift")
        try Data(repeating: 0x61, count: FormatterRunner.maximumFileBytes + 1).write(to: big)
        try Data([0xFF, 0xFE, 0x00]).write(to: fixture.project.appendingPathComponent("Bin.swift"))
        let row = try fixture.row(#"echo ran >> "\#(fixture.log.path)""#)
        let big1 = await run(fixture, row, path: "Big.swift")
        let bin1 = await run(fixture, row, path: "Bin.swift")
        XCTAssertEqual(big1, .skipped)
        XCTAssertEqual(bin1, .skipped)
        XCTAssertEqual(fixture.logText, "")
    }

    // MARK: launch policy

    /// Mutation check: delete the `isContained` check in the policy and the
    /// in-project binary runs.
    func testBinaryInsideTheWorkspaceAndRelativeCommandsAreRefused() async throws {
        let fixture = try FormatterFixture()
        let inProject = fixture.project.appendingPathComponent("fmt")
        try "#!/bin/bash\necho ran >> \"\(fixture.log.path)\"\n".write(to: inProject, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: inProject.path)
        var row = try fixture.row("exit 0")
        row.command = inProject.path
        XCTAssertThrowsError(try FormatterLaunchPolicy.resolve(
            row: row, rootURL: fixture.project, searchPath: nil, filePath: "x")) {
            XCTAssertEqual($0 as? FormatterLaunchError, .insideWorkspace(inProject.path))
        }
        let outcome = await run(fixture, row)
        XCTAssertEqual(outcome, .skipped)
        row.command = "./fmt"
        XCTAssertThrowsError(try FormatterLaunchPolicy.resolve(
            row: row, rootURL: fixture.project, searchPath: nil, filePath: "x")) {
            XCTAssertEqual($0 as? FormatterLaunchError, .relativeCommand("./fmt"))
        }
        row.command = "node_modules/.bin/prettier"
        XCTAssertThrowsError(try FormatterLaunchPolicy.resolve(
            row: row, rootURL: fixture.project, searchPath: nil, filePath: "x"))
        XCTAssertEqual(fixture.logText, "")
    }

    func testSymlinkedBinaryResolvingIntoTheWorkspaceIsRefused() throws {
        let fixture = try FormatterFixture()
        let real = fixture.project.appendingPathComponent("fmt")
        try "#!/bin/bash\n".write(to: real, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: real.path)
        let link = fixture.tools.appendingPathComponent("linked-fmt")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: real)
        var row = try fixture.row("exit 0")
        row.command = link.path
        XCTAssertThrowsError(try FormatterLaunchPolicy.resolve(
            row: row, rootURL: fixture.project, searchPath: nil, filePath: "x"))
    }

    func testMissingBinaryIsASilentSkip() async throws {
        let fixture = try FormatterFixture()
        var row = try fixture.row("exit 0")
        row.command = fixture.tools.appendingPathComponent("not-installed").path
        let outcome = await run(fixture, row)
        XCTAssertEqual(outcome, .skipped)
        XCTAssertEqual(try fixture.read("Sources/A.swift"), "let a = 1\n")
    }

    /// Mutation check: pass the parent environment through (`environment: nil`)
    /// and the secret reaches the child.
    func testChildEnvironmentIsTheAllowlistAndCwdIsTheProjectRoot() async throws {
        let fixture = try FormatterFixture()
        let dump = fixture.tools.appendingPathComponent("env.txt")
        let row = try fixture.row(#"{ echo "cwd=$(pwd -P)"; env | sed 's/^/env:/'; } > "\#(dump.path)""#)
        _ = await run(fixture, row, env: [
            "PATH": "/usr/bin:/bin", "HOME": "/Users/x", "LANG": "en_US.UTF-8", "TMPDIR": "/tmp/",
            "OPENAI_API_KEY": "sk-secret", "HTTPS_PROXY": "http://proxy", "AWS_SECRET_ACCESS_KEY": "s"])
        let text = try String(contentsOf: dump, encoding: .utf8)
        let cwd = text.split(separator: "\n").first { $0.hasPrefix("cwd=") }.map { String($0.dropFirst(4)) } ?? ""
        XCTAssertEqual(PathContainment.canonical(URL(fileURLWithPath: cwd)).path, fixture.project.path, text)
        XCTAssertFalse(text.contains("sk-secret"))
        XCTAssertFalse(text.contains("PROXY"))
        XCTAssertFalse(text.contains("AWS_"))
        XCTAssertTrue(text.contains("env:HOME=/Users/x"))
        let allowed: Set<String> = ["PATH", "HOME", "LANG", "TMPDIR", "PWD", "SHLVL", "_", "OLDPWD", "__CF_USER_TEXT_ENCODING"]
        let keys = text.split(separator: "\n").filter { $0.hasPrefix("env:") }
            .map { String($0.dropFirst(4).split(separator: "=", maxSplits: 1).first ?? "") }
        XCTAssertTrue(Set(keys).isSubset(of: allowed), "unexpected variables: \(Set(keys).subtracting(allowed))")
    }

    // MARK: edit-tool snapshot

    /// Mutation check: delete the `refreshAfterFormat` calls and the next edit
    /// sees the file as stale.
    func testSnapshotIsRefreshedSoTheNextEditIsNotStale() async throws {
        let fixture = try FormatterFixture()
        let url = fixture.project.appendingPathComponent("Sources/A.swift")
        let store = FileSnapshotStore.shared
        await store.recordSnapshot(url: url, content: "let a = 1\n")
        await store.recordBackup(url: url, content: "old\n")
        await store.recordSnapshot(url: url, content: "let a = 1\n")
        guard case .formatted = await run(fixture, try fixture.row(FormatterFixture.upcase)) else {
            return XCTFail("expected a format")
        }
        let stale = await store.isStale(url: url)
        let undoApplies = await store.backupMatchesCurrent(url: url)
        let backup = await store.restoreBackup(url: url)
        XCTAssertFalse(stale)
        XCTAssertTrue(undoApplies, "undo still applies to the formatted file")
        XCTAssertEqual(backup, "old\n", "the undo backup keeps the pre-edit content")
        await store.removeSnapshot(url: url)
        await store.discardBackup(url: url)
    }

    func testAnUntrackedFileStaysUntrackedAfterAFormat() async throws {
        let fixture = try FormatterFixture()
        let url = fixture.project.appendingPathComponent("Sources/A.swift")
        _ = await run(fixture, try fixture.row(FormatterFixture.upcase))
        let tracked = await FileSnapshotStore.shared.isTracked(url: url)
        XCTAssertFalse(tracked)
    }
}
