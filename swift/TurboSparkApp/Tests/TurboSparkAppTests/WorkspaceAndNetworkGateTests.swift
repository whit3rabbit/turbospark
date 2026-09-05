import XCTest

@testable import TurboSparkApp

/// The non-terminal halves of the tool boundary: which workspace a tool runs
/// against, which permission preset a new project gets, which hosts count as
/// private, and what an unimplemented transport reports.
final class WorkspaceAndNetworkGateTests: XCTestCase {

    // MARK: - S1: one default, not two

    /// **Every path that creates a project must agree on the preset.**
    ///
    /// `AppProject.init`, its tolerant `init(from:)` fallback and
    /// `AppModel.createProject` all default to `.standard`. The new-project
    /// sheet seeded `.auto` instead, which is `terminal: .allow` under
    /// `mode: .auto` -- so a project made the only way users make one ran
    /// model-proposed shell commands with no prompt, while the default
    /// everything else used asked first.
    ///
    /// A SwiftUI view has no headless entry point, so the sheet's `@State`
    /// cannot be read from a test directly. What CAN be pinned is that all
    /// four sites read one named constant and that the constant asks: the
    /// sheet's line is `AppProjectPermissions.newProjectDefault`, and a
    /// regression would have to reintroduce a second spelling to get past
    /// this.
    func testTheDefaultPresetAsksBeforeShellAndFileWrites() {
        let fresh = AppProjectPermissions.newProjectDefault
        XCTAssertEqual(fresh.terminal, .ask, "a new project must not run shell commands silently")
        XCTAssertEqual(fresh.fileWrite, .ask)
        XCTAssertEqual(fresh.mcp, .ask)
        XCTAssertEqual(fresh.fileRead, .allow, "reading inside the workspace stays unprompted")
    }

    func testEveryProjectCreationPathDefaultsToTheSamePreset() throws {
        // The memberwise default.
        XCTAssertEqual(AppProject(name: "A").permissions, .newProjectDefault)

        // The decode fallback, for a row written before the field existed.
        let legacy = Data(#"{"name":"B"}"#.utf8)
        let decoded = try JSONDecoder().decode(AppProject.self, from: legacy)
        XCTAssertEqual(decoded.permissions, .newProjectDefault)

        // And the bare initializer of the permission struct itself, which is
        // what a caller spelling its own fields starts from.
        XCTAssertEqual(AppProjectPermissions().terminal, .ask)
    }

    /// The specific regression: the default must NOT be the permissive
    /// preset. Stated as its own inequality because the two tests above would
    /// both pass if `newProjectDefault` were redefined as `.auto` AND
    /// `.auto` were quietly narrowed.
    func testTheDefaultPresetIsNotThePermissiveOne() {
        XCTAssertNotEqual(
            AppProjectPermissions.newProjectDefault, AppProjectPermissions.auto,
            "the new-project sheet seeded `.auto` (terminal: .allow) for every project a user "
                + "ever made through the UI")
        XCTAssertNotEqual(AppProjectPermissions.newProjectDefault, AppProjectPermissions.permissive)
    }

    /// The `.auto` preset is still available and still means what it says.
    /// It is a choice a user can make, not a default they are given.
    func testTheAutoPresetIsStillPermissiveWhenExplicitlyChosen() {
        XCTAssertEqual(AppProjectPermissions.auto.terminal, .allow)
        XCTAssertEqual(AppProjectPermissions.auto.mode, .auto)
    }

    // MARK: - S3: no project, no root

    /// **A projectless chat has no workspace, and the tools that need one are
    /// refused rather than pointed at the home directory.**
    ///
    /// The old fallback rooted them at `~`, where `resolveSecurePath`'s
    /// containment check passes for `Library/Application Support`, browser
    /// profiles, shell history and every token on disk.
    func testPathAndShellToolsAreRefusedWithoutAProject() async {
        let rooted: [(String, [String: String])] = [
            ("read_file", ["path": "Library/Application Support/anything.json"]),
            ("write_file", ["path": "x.txt", "content": "y"]),
            ("edit_file", ["path": "x.txt", "old_string": "a"]),
            ("list_directory", ["path": "."]),
            ("search_code", ["pattern": "token"]),
            ("run_command", ["command": "ls"]),
            ("apply_patch", ["patch_text": "*** Begin Patch\n*** End Patch"]),
        ]

        for (name, arguments) in rooted {
            let call = AppToolCall(
                name: name, arguments: arguments,
                category: AppToolRegistry.category(for: name))
            let result = await AppToolRegistry.execute(call: call, in: nil)
            XCTAssertTrue(result.isError, "'\(name)' must be refused with no project")
            XCTAssertTrue(
                result.output.contains("needs a project workspace"),
                "'\(name)' must say WHY it was refused, got: \(result.output)")
        }
    }

    /// The tools that need no root still work, which is the case the old
    /// fallback was really reaching for. Refusing these too would have made a
    /// projectless chat useless.
    func testRootlessToolsStillWorkWithoutAProject() async {
        for name in ["todowrite"] {
            let call = AppToolCall(
                name: name, arguments: [:],
                category: AppToolRegistry.category(for: name))
            let result = await AppToolRegistry.execute(call: call, in: nil)
            XCTAssertFalse(
                result.isError, "'\(name)' needs no workspace and must still run: \(result.output)")
        }
    }

    /// Containment itself, unchanged by the above and worth pinning beside
    /// it: with a project, a path that climbs out is still refused.
    func testPathsThatEscapeTheProjectRootAreStillRefused() throws {
        let root = URL(fileURLWithPath: "/tmp")
        for escape in ["../../../../etc/passwd", "/etc/passwd", "~/.ssh/id_rsa"] {
            XCTAssertThrowsError(
                try AppToolRegistry.resolveSecurePath(relPath: escape, rootURL: root),
                "'\(escape)' must not resolve inside the root")
        }
        XCTAssertNoThrow(try AppToolRegistry.resolveSecurePath(relPath: "sub/file.txt", rootURL: root))
    }

    // MARK: - S4: what counts as private

    /// **The private-network check knew six literals and every other spelling
    /// of the same address read as an ordinary outbound fetch.**
    ///
    /// The decimal and octal forms are the ones worth staring at:
    /// `http://2130706433/` and `http://0177.0.0.1/` both reach 127.0.0.1
    /// through URLSession, and a `hasPrefix("127.")` test sees neither.
    func testPrivateAndMetadataHostsAreRecognizedInEverySpelling() {
        let privateHosts = [
            "localhost", "127.0.0.1", "127.1", "0.0.0.0",
            "2130706433",  // 127.0.0.1 as a single decimal
            "0177.0.0.1",  // 127.0.0.1 with an octal first octet
            "0x7f.0.0.1",  // and with a hex one
            "169.254.169.254", "metadata.google.internal", "metadata",
            "10.0.0.5", "192.168.1.1",
            "172.16.0.1", "172.20.10.1", "172.31.255.254",  // the whole /12
            "100.64.0.1",  // CGNAT
            "::1", "[::1]", "fd00::1", "fe80::1",
            "printer.local", "db.internal", "app.localhost",
        ]
        for host in privateHosts {
            XCTAssertTrue(
                AppToolSandbox.isPrivateOrMetadataHost(host),
                "'\(host)' resolves to this machine or a private network")
        }
    }

    /// The other direction, or the check is just "return true". `172.32.x` is
    /// deliberately here: it is one octet outside the private /12 and a
    /// prefix test on "172." would swallow it.
    func testPublicHostsAreNotFlaggedAsPrivate() {
        for host in [
            "example.com", "api.anthropic.com", "8.8.8.8", "1.1.1.1",
            "172.32.0.1", "172.15.0.1", "11.0.0.1", "192.169.0.1", "9.9.9.9",
        ] {
            XCTAssertFalse(
                AppToolSandbox.isPrivateOrMetadataHost(host), "'\(host)' is a public host")
        }
    }

    /// The classifier reads that check, which is the wiring `validateDomain`
    /// never had: `SandboxConfig`'s three network fields configured a
    /// function with no callers anywhere in the tree.
    func testTheWebRiskArmUsesThePrivateHostCheck() {
        let ssrf = ToolRiskClassifier.assessRisk(
            name: "WebFetch", arguments: ["url": "http://2130706433/latest/meta-data/"])
        XCTAssertTrue(ssrf.isHighRisk, "a decimal-encoded loopback URL must ask")

        let ordinary = ToolRiskClassifier.assessRisk(
            name: "WebFetch", arguments: ["url": "https://example.com/docs"])
        XCTAssertFalse(ordinary.isHighRisk)

        let malformed = ToolRiskClassifier.assessRisk(
            name: "WebFetch", arguments: ["url": "not a url at all"])
        XCTAssertTrue(malformed.isHighRisk, "a URL that cannot be parsed cannot be classified")
    }

    // MARK: - S5: an unimplemented transport must not report success

    /// **The SSE arm returned "SSE remote tool execution completed." for
    /// every call without issuing a request.** The model was told an external
    /// action had happened and the transcript showed a green result. Throwing
    /// is not a smaller feature than that; it is the only honest state.
    func testTheSseTransportRefusesInsteadOfFabricatingSuccess() async {
        let config = McpServerConfig(
            name: "remote",
            transport: .sse(url: URL(string: "https://example.com/sse")!, headers: [:])
        )
        do {
            let output = try await McpClientEngine.shared.callTool(
                config: config, toolName: "delete_everything", arguments: [:])
            XCTFail("an unimplemented transport must throw, got: \(output)")
        } catch {
            let message = error.localizedDescription
            XCTAssertTrue(
                message.contains("SSE"), "the error must name the transport: \(message)")
            XCTAssertTrue(
                message.contains("NOT"), "the error must say the call did not run: \(message)")
        }
    }

    // MARK: - M6/M7: bounded reads, bounded tracking

    /// A model-supplied path is an unbounded read until something bounds it.
    func testAnOversizedFileIsRefusedRatherThanLoaded() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-limits-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let big = root.appendingPathComponent("big.txt")
        // Sparse-ish write: one byte at the far end makes a file of the right
        // SIZE without allocating it, which is what the check reads.
        FileManager.default.createFile(atPath: big.path, contents: nil)
        let handle = try FileHandle(forWritingTo: big)
        try handle.truncate(atOffset: UInt64(AppFileReadLimits.maximumBytes + 1))
        try handle.close()

        XCTAssertThrowsError(
            try AppFileReadLimits.readTextFile(at: big, describing: "big.txt")
        ) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("limit"),
                "the refusal must name the limit: \(error.localizedDescription)")
        }

        let small = root.appendingPathComponent("small.txt")
        try "hello".write(to: small, atomically: true, encoding: .utf8)
        XCTAssertEqual(try AppFileReadLimits.readTextFile(at: small, describing: "small.txt"), "hello")
    }

    /// The snapshot map used to grow for the life of the process: every file
    /// the model read added an entry and `reset()` had no callers at all.
    func testTheSnapshotStoreIsBounded() async {
        let store = FileSnapshotStore()
        for i in 0..<(FileSnapshotStore.maximumTrackedFiles + 50) {
            await store.recordSnapshot(
                url: URL(fileURLWithPath: "/tmp/turbospark-snapshot-\(i).txt"),
                content: "content \(i)")
        }
        let count = await store.trackedFileCount
        XCTAssertLessThanOrEqual(count, FileSnapshotStore.maximumTrackedFiles)

        await store.reset()
        let afterReset = await store.trackedFileCount
        XCTAssertEqual(afterReset, 0)
    }

    /// Hashing the content the caller already holds must agree with hashing
    /// the file, or the cheaper overload is a different check wearing the
    /// same name.
    func testHashingSuppliedContentAgreesWithHashingTheFile() async throws {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-hash-\(UUID().uuidString).txt")
        defer { try? FileManager.default.removeItem(at: url) }
        try "the same bytes".write(to: url, atomically: true, encoding: .utf8)

        let fromFile = FileSnapshotStore()
        await fromFile.recordSnapshot(url: url)
        let fileStale = await fromFile.isStale(url: url)
        XCTAssertFalse(fileStale)

        let fromContent = FileSnapshotStore()
        await fromContent.recordSnapshot(url: url, content: "the same bytes")
        let contentStale = await fromContent.isStale(url: url)
        XCTAssertFalse(
            contentStale,
            "a snapshot taken from supplied content must match the file on disk")

        let wrong = FileSnapshotStore()
        await wrong.recordSnapshot(url: url, content: "different bytes")
        let wrongStale = await wrong.isStale(url: url)
        XCTAssertTrue(wrongStale)
    }

    // MARK: - M8: the archive stays bounded

    /// A 1 MB tool result (`ProcessExecutor`'s own cap) must not reach the
    /// archive whole: the whole file is re-encoded on every mutation.
    func testALargeToolResultIsTruncatedOnTheWayToDisk() throws {
        let huge = String(repeating: "x", count: 2_000_000)
        let result = AppToolResult(callID: UUID(), output: huge)
        XCTAssertEqual(result.output.count, 2_000_000, "the in-memory value stays whole")

        let encoded = try JSONEncoder().encode(result)
        XCTAssertLessThan(
            encoded.count, AppToolResult.maximumPersistedOutputBytes * 2,
            "the encoded form must be bounded by the persist cap")

        let decoded = try JSONDecoder().decode(AppToolResult.self, from: encoded)
        XCTAssertTrue(decoded.output.contains("truncated"), "truncation must be visible, not silent")
        XCTAssertEqual(decoded.callID, result.callID, "every other field survives the round trip")
    }
}
