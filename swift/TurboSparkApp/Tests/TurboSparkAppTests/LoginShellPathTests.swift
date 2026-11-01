import XCTest
@testable import TurboSparkApp

/// The login-shell PATH probe and how MCP spawn resolution uses it.
final class LoginShellPathTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoginShellPathTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    private func executable(_ relativePath: String, body: String = "#!/bin/sh\nexit 0\n") throws -> URL {
        let url = scratch.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try body.write(to: url, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
        return url
    }

    private var begin: String { LoginShellPath.beginMarker }
    private var end: String { LoginShellPath.endMarker }

    // MARK: parse

    /// rc files print around the command; only the marked text is PATH.
    func testParseIgnoresRcNoiseOnBothSides() {
        let output = "Welcome!\nnvm: using v22\n\(begin)/a/bin:/b/bin\n\(end)\nbye\n"
        XCTAssertEqual(LoginShellPath.parse(output), "/a/bin:/b/bin")
    }

    func testParseTakesTheLastMarkerPair() {
        let output = "\(begin)/stale\(end) noise \(begin)/real/bin\(end)"
        XCTAssertEqual(LoginShellPath.parse(output), "/real/bin")
    }

    func testParseWithoutAnEndMarkerIsNil() {
        XCTAssertNil(LoginShellPath.parse("\(begin)/a/bin"))
    }

    func testParseOfAnEmptyValueIsNil() {
        XCTAssertNil(LoginShellPath.parse("\(begin)\n\(end)"))
    }

    /// A relative entry would resolve against the project root in the child,
    /// so a cloned repository's own `node` could run.
    func testParseDropsRelativeEntries() {
        let output = "\(begin).:/a/bin:node_modules/.bin::./bin:/b/bin\n\(end)"
        XCTAssertEqual(LoginShellPath.parse(output), "/a/bin:/b/bin")
        XCTAssertNil(LoginShellPath.parse("\(begin).:bin\(end)"), "Nothing absolute means no PATH")
    }

    // MARK: defaultShell

    private let shells: Set<String> = ["/bin/zsh", "/bin/bash", "/opt/homebrew/bin/fish"]

    /// `launchctl setenv SHELL` must not choose what this app runs.
    func testAnUnlistedShellVariableIsIgnored() {
        XCTAssertEqual(
            LoginShellPath.defaultShell(
                parent: ["SHELL": "/tmp/evil"], passwdShell: "/bin/bash", allowedShells: shells),
            "/bin/bash")
        XCTAssertEqual(
            LoginShellPath.defaultShell(
                parent: ["SHELL": "/tmp/evil"], passwdShell: nil, allowedShells: shells),
            "/bin/zsh")
    }

    func testThePasswdShellBeatsAListedShellVariable() {
        XCTAssertEqual(
            LoginShellPath.defaultShell(
                parent: ["SHELL": "/bin/bash"], passwdShell: "/opt/homebrew/bin/fish", allowedShells: shells),
            "/opt/homebrew/bin/fish")
    }

    func testAnUnlistedPasswdShellFallsBackToAListedShellVariable() {
        XCTAssertEqual(
            LoginShellPath.defaultShell(
                parent: ["SHELL": "/bin/bash"], passwdShell: "/usr/local/bin/unlisted", allowedShells: shells),
            "/bin/bash")
    }

    func testSystemShellsSkipsCommentsAndBlanks() {
        let contents = "# List of acceptable shells\n\n/bin/bash\n  /bin/zsh  \nrelative/sh\n"
        XCTAssertEqual(LoginShellPath.systemShells(contents: contents), ["/bin/bash", "/bin/zsh"])
        XCTAssertEqual(LoginShellPath.systemShells(contents: nil), [])
    }

    func testNushellIsNotProbed() {
        XCTAssertNil(LoginShellPath.shellInvocation(shell: "/opt/homebrew/bin/nu"))
        XCTAssertEqual(LoginShellPath.shellInvocation(shell: "/bin/zsh")?.1.first, "-ilc")
        XCTAssertEqual(LoginShellPath.shellInvocation(shell: "/opt/homebrew/bin/fish")?.1.first, "-lc")
    }

    // MARK: capture

    func testCaptureReturnsTheMarkedPath() async throws {
        let shell = try executable(
            "shell-ok", body: "#!/bin/sh\necho banner\nprintf '%s' '\(begin)/x/bin:/y/bin\(end)'\necho trailer\n")
        let path = await LoginShellPath.capture(shell: shell.path, parent: ["HOME": scratch.path])
        XCTAssertEqual(path, "/x/bin:/y/bin")
    }

    /// The probe env is built from scratch: a secret in this app's env must
    /// not reach the user's rc files, and the rc runs in HOME, not a project.
    func testCaptureDoesNotForwardUnlistedVariablesAndRunsInHome() async throws {
        let shell = try executable(
            "shell-env",
            // The leading `/probe|` keeps the payload one absolute "entry", so
            // parse's relative-entry filter lets it through.
            body: "#!/bin/sh\nprintf '%s' '\(begin)/probe|'\nprintf '%s|' \"${FAKE_SECRET:-absent}\" \"$PWD\" \"$TERM\"\nprintf '%s' '\(end)'\n")
        let path = await LoginShellPath.capture(
            shell: shell.path,
            parent: ["HOME": scratch.path, "FAKE_SECRET": "sk-secret", "ANTHROPIC_API_KEY": "sk-ant"])
        let fields = Array(try XCTUnwrap(path).split(separator: "|").map(String.init).dropFirst())
        XCTAssertEqual(fields.first, "absent", "An unlisted variable must not reach the probe shell")
        XCTAssertEqual(
            URL(fileURLWithPath: fields[1]).resolvingSymlinksInPath().path,
            scratch.resolvingSymlinksInPath().path)
        XCTAssertEqual(fields[2], "dumb")
    }

    func testCaptureOfANonZeroExitIsNil() async throws {
        let shell = try executable("shell-fail", body: "#!/bin/sh\nprintf '%s' '\(begin)/x\(end)'\nexit 3\n")
        let path = await LoginShellPath.capture(shell: shell.path, parent: ["HOME": scratch.path])
        XCTAssertNil(path)
    }

    /// A hung rc must cost the timeout, not hang the first MCP spawn.
    func testCaptureOfAHungShellTimesOutToNil() async throws {
        let shell = try executable("shell-hang", body: "#!/bin/sh\nsleep 30\n")
        let started = Date()
        let path = await LoginShellPath.capture(
            shell: shell.path, parent: ["HOME": scratch.path], timeoutSeconds: 0.5)
        XCTAssertNil(path)
        XCTAssertLessThan(Date().timeIntervalSince(started), 10)
    }

    func testCaptureOfAMissingShellIsNil() async {
        let path = await LoginShellPath.capture(
            shell: scratch.appendingPathComponent("nope").path, parent: ["HOME": scratch.path])
        XCTAssertNil(path)
    }

    // MARK: McpClientEngine.childEnvironment

    func testTheLoginShellPathGoesAfterHomebrewAndBeforeTheInheritedPath() {
        let env = McpClientEngine.childEnvironment(
            parent: ["PATH": "/usr/bin", "ANTHROPIC_API_KEY": "sk-secret"],
            passthrough: [], declared: [:],
            executableDirectory: "/opt/tools/bin",
            searchPath: "/Users/me/.nvm/versions/node/v22.1.0/bin:/usr/bin")

        XCTAssertEqual(
            env["PATH"],
            "/opt/tools/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin"
                + ":/Users/me/.nvm/versions/node/v22.1.0/bin:/usr/bin")
        XCTAssertNil(env["ANTHROPIC_API_KEY"], "searchPath must widen PATH only")
    }

    /// Covers the inherited PATH too: a terminal launch hands the app the
    /// user's PATH verbatim, `.` included.
    func testRelativeEntriesNeverReachTheChildPath() {
        let env = McpClientEngine.childEnvironment(
            parent: ["PATH": ".:/usr/bin:node_modules/.bin"],
            passthrough: [], declared: [:],
            executableDirectory: "/opt/tools/bin",
            searchPath: "./bin:/login/bin")
        XCTAssertEqual(
            env["PATH"],
            "/opt/tools/bin:/opt/homebrew/bin:/opt/homebrew/sbin:/usr/local/bin:/login/bin:/usr/bin")
    }

    /// A relative search entry resolves against this process's cwd, never a
    /// place the user meant, so it is skipped.
    func testARelativeSearchEntryDoesNotResolve() throws {
        let name = "ts-mcp-fixture-\(UUID().uuidString.prefix(8))"
        _ = try executable("rel/\(name)")
        let previous = FileManager.default.currentDirectoryPath
        FileManager.default.changeCurrentDirectoryPath(scratch.path)
        defer { FileManager.default.changeCurrentDirectoryPath(previous) }
        XCTAssertNil(McpClientEngine.resolveExecutablePath(name, searchPath: "rel"))
    }

    func testADeclaredPathStillBeatsTheLoginShellPath() {
        let env = McpClientEngine.childEnvironment(
            parent: ["PATH": "/usr/bin"], passthrough: [],
            declared: ["PATH": "/custom/bin"],
            executableDirectory: "/opt/tools/bin",
            searchPath: "/login/bin")
        XCTAssertEqual(env["PATH"], "/custom/bin")
    }

    // MARK: McpClientEngine.resolveExecutablePath

    func testABareNameResolvesFromTheLoginShellPath() throws {
        let name = "ts-mcp-fixture-\(UUID().uuidString.prefix(8))"
        let tool = try executable("login/bin/\(name)")
        XCTAssertNil(McpClientEngine.resolveExecutablePath(name), "Precondition: not on any fixed path")
        XCTAssertEqual(
            McpClientEngine.resolveExecutablePath(
                name, searchPath: "/nonexistent:\(tool.deletingLastPathComponent().path)"),
            tool.path)
    }

    /// The terminal's order is the user's choice, so the first hit wins.
    func testTheFirstLoginShellPathEntryWins() throws {
        let name = "ts-mcp-fixture-\(UUID().uuidString.prefix(8))"
        let first = try executable("first/\(name)")
        let second = try executable("second/\(name)")
        let searchPath = [first, second].map { $0.deletingLastPathComponent().path }.joined(separator: ":")
        XCTAssertEqual(McpClientEngine.resolveExecutablePath(name, searchPath: searchPath), first.path)
    }

    // MARK: McpClientEngine.nvmNodeBinDirectory

    private func makeNvm(versions: [String], alias: String?) throws -> URL {
        let home = scratch.appendingPathComponent("home-\(UUID().uuidString.prefix(6))", isDirectory: true)
        for version in versions {
            try FileManager.default.createDirectory(
                at: home.appendingPathComponent(".nvm/versions/node/\(version)/bin"),
                withIntermediateDirectories: true)
        }
        if let alias {
            let file = home.appendingPathComponent(".nvm/alias/default")
            try FileManager.default.createDirectory(
                at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            try "\(alias)\n".write(to: file, atomically: true, encoding: .utf8)
        }
        return home
    }

    private func nvmVersion(_ home: URL) -> String? {
        McpClientEngine.nvmNodeBinDirectory(home: home)
            .map { URL(fileURLWithPath: $0).deletingLastPathComponent().lastPathComponent }
    }

    func testNvmAliasPrefixPicksTheHighestMatchNumerically() throws {
        let home = try makeNvm(versions: ["v18.1.0", "v22.1.0", "v22.9.0", "v22.10.0", "v24.0.0"], alias: "22")
        XCTAssertEqual(nvmVersion(home), "v22.10.0")
    }

    func testNvmAliasNodeOrMissingPicksTheHighestInstalled() throws {
        XCTAssertEqual(nvmVersion(try makeNvm(versions: ["v18.1.0", "v22.10.0"], alias: "node")), "v22.10.0")
        XCTAssertEqual(nvmVersion(try makeNvm(versions: ["v18.1.0", "v22.10.0"], alias: nil)), "v22.10.0")
        XCTAssertEqual(nvmVersion(try makeNvm(versions: ["v18.1.0", "v22.10.0"], alias: "lts/*")), "v22.10.0")
    }

    func testNvmAliasWithNoMatchFallsBackToTheHighestInstalled() throws {
        XCTAssertEqual(nvmVersion(try makeNvm(versions: ["v18.1.0", "v20.2.0"], alias: "16")), "v20.2.0")
    }

    func testNvmWithNoVersionsIsNil() throws {
        XCTAssertNil(nvmVersion(try makeNvm(versions: [], alias: "22")))
        XCTAssertNil(McpClientEngine.nvmNodeBinDirectory(home: scratch.appendingPathComponent("no-home")))
    }
}
