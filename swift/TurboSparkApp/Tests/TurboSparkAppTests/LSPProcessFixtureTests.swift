import XCTest
@testable import TurboSparkApp

/// OC-10: the real spawn path (`LSPProcessTransport`, launch policy, kill on
/// stop) against a tiny bash fixture that speaks just enough LSP. It is a
/// scripted stand-in: no real language server is ever started by tests.
final class LSPProcessFixtureTests: XCTestCase {
    /// Replies to initialize and shutdown, publishes one error per didOpen or
    /// didChange, and records its cwd, pid and environment in `$1`.
    private static let script = #"""
    #!/bin/bash
    { echo "cwd=$(pwd -P)"; echo "pid=$$"; env | sed 's/^/env:/'; } > "$1"
    send() { printf 'Content-Length: %d\r\n\r\n%s' "${#1}" "$1"; }
    len=""
    while IFS= read -r line; do
      line=${line%$'\r'}
      case "$line" in
        Content-Length:*) len=${line#Content-Length: } ;;
        "")
          [ -z "$len" ] && continue
          body=$(dd bs=1 count="$len" 2>/dev/null)
          len=""
          id=$(printf '%s' "$body" | sed -nE 's/.*"id":([0-9]+).*/\1/p')
          case "$body" in
            *'"method":"initialize"'*) send '{"jsonrpc":"2.0","id":'"$id"',"result":{"capabilities":{},"serverInfo":{"name":"bash-fixture"}}}' ;;
            *'"method":"shutdown"'*) send '{"jsonrpc":"2.0","id":'"$id"',"result":null}' ;;
            *'"method":"exit"'*) exit 0 ;;
            *'"method":"textDocument/didOpen"'*|*'"method":"textDocument/didChange"'*)
              uri=$(printf '%s' "$body" | sed -nE 's/.*"uri":"([^"]*)".*/\1/p')
              send '{"jsonrpc":"2.0","method":"textDocument/publishDiagnostics","params":{"uri":"'"$uri"'","diagnostics":[{"range":{"start":{"line":0,"character":1},"end":{"line":0,"character":4}},"severity":1,"message":"fixture error"}]}}' ;;
          esac ;;
      esac
    done
    """#

    private func fixture(in folder: URL) throws -> (script: URL, info: URL) {
        let script = folder.appendingPathComponent("fixture-ls.sh")
        try Self.script.write(to: script, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)
        return (script, folder.appendingPathComponent("info.txt"))
    }

    /// The login-shell PATH probe is skipped: it runs the user's shell and is
    /// slow on a loaded host. The policy and environment are the real ones.
    private static let factory: LanguageServerManager.TransportFactory = { config, root in
        try LSPProductionTransport.make(config: config, root: root, searchPath: nil)
    }

    private func isAlive(_ pid: Int32) -> Bool { kill(pid, 0) == 0 }

    private func waitUntil(_ timeout: TimeInterval = 5, _ condition: () -> Bool) async -> Bool {
        let end = Date().addingTimeInterval(timeout)
        while Date() < end { if condition() { return true }; try? await Task.sleep(nanoseconds: 50_000_000) }
        return condition()
    }

    func testRealSpawnIsWorkspaceRootedScrubbedAndKilledOnStop() async throws {
        setenv("TS_LSP_SECRET_PROBE", "leak-me", 1)
        defer { unsetenv("TS_LSP_SECRET_PROBE") }
        let project = try LSPTestProject()
        // The fixture lives OUTSIDE the project: a binary inside it is refused.
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("lsp-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        let (script, info) = try fixture(in: PathContainment.canonical(outside))
        let config = LSPServerConfig(
            id: "fixture", name: "bash-fixture", extensions: ["swift"], languageID: "swift",
            command: script.path, args: [info.path], enabled: true, isDiscovered: false)
        let lsp = LanguageServerManager(
            rootURL: project.root, settings: { [config] }, idleTimeout: 0, transportFactory: Self.factory)

        let reports = await lsp.postEdit(paths: ["Sources/A.swift"], budget: 8)
        guard case .diagnostics(let list)? = reports?.first?.status else {
            return XCTFail("expected diagnostics from the fixture, got \(String(describing: reports))")
        }
        XCTAssertEqual(list.first?.message, "fixture error")
        XCTAssertEqual(reports?.first?.server, "bash-fixture")

        let text = try String(contentsOf: info, encoding: .utf8)
        let lines = text.components(separatedBy: "\n")
        let realRoot = URL(fileURLWithPath: project.root.path).resolvingSymlinksInPath().path
        let privateRoot = realRoot.hasPrefix("/private/") ? realRoot : "/private" + realRoot
        XCTAssertTrue(lines.contains("cwd=\(privateRoot)") || lines.contains("cwd=\(realRoot)"), "workspace rooted: \(lines.first ?? "")")
        let keys = Set(lines.filter { $0.hasPrefix("env:") }.map { String($0.dropFirst(4).prefix { $0 != "=" }) })
        XCTAssertFalse(keys.contains("TS_LSP_SECRET_PROBE"), "the scrubbed environment leaks nothing")
        XCTAssertTrue(keys.isSubset(of: ["PATH", "HOME", "LANG", "TMPDIR", "PWD", "SHLVL", "_", "OLDPWD", "__CF_USER_TEXT_ENCODING"]), "\(keys)")
        let pidText = try XCTUnwrap(lines.first { $0.hasPrefix("pid=") })
        let pid = try XCTUnwrap(Int32(pidText.dropFirst(4)))
        XCTAssertTrue(isAlive(pid))

        await lsp.stopAll()
        let gone = await waitUntil { !self.isAlive(pid) }
        XCTAssertTrue(gone, "the server process is killed on project close")
    }

    func testAServerBinaryInsideTheProjectIsNeverSpawned() async throws {
        let project = try LSPTestProject()
        let (script, _) = try fixture(in: project.root)
        let config = LSPServerConfig(
            id: "inside", name: "inside", extensions: ["swift"], languageID: "swift",
            command: script.path, args: [], enabled: true, isDiscovered: false)
        let lsp = LanguageServerManager(
            rootURL: project.root, settings: { [config] }, idleTimeout: 0, transportFactory: Self.factory)
        let reports = await lsp.postEdit(paths: ["Sources/A.swift"], budget: 4)
        guard case .unavailable(let reason)? = reports?.first?.status else {
            return XCTFail("expected unavailable, got \(String(describing: reports))")
        }
        XCTAssertTrue(reason.contains("inside the project"), reason)
        XCTAssertFalse(FileManager.default.fileExists(atPath: project.root.appendingPathComponent("info.txt").path))
    }

    func testSettingsTestActionStartsAndStopsInAnEmptyFolder() async throws {
        let outside = FileManager.default.temporaryDirectory.appendingPathComponent("lsp-fixture-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: outside) }
        let (script, info) = try fixture(in: PathContainment.canonical(outside))
        let config = LSPServerConfig(
            id: "t", name: "t", extensions: ["x"], languageID: "x",
            command: script.path, args: [info.path], enabled: true, isDiscovered: false)
        let result = await LSPServerProbe.test(config, transportFactory: Self.factory)
        guard case .success(let name) = result else { return XCTFail("probe failed: \(result)") }
        XCTAssertEqual(name, "bash-fixture")
        let text = try String(contentsOf: info, encoding: .utf8)
        XCTAssertFalse(text.contains("cwd=\(PathContainment.canonical(outside).path)\n"), "the probe runs in its own empty folder")
        XCTAssertEqual(LSPProcessRegistry.count, 0)
    }
}
