import XCTest
@testable import TurboSparkApp

/// OC-10 deferred items: the per-project language server toggle (it can only
/// NARROW the app-wide `lsp.enabled` switch) and per-row environment
/// passthrough behind a deny list. Fake servers only; no real language server.
final class LSPProjectToggleTests: XCTestCase {
    // MARK: persistence

    private func decodeArchive(_ json: String) throws -> AppProjectArchive {
        try JSONDecoder().decode(AppProjectArchive.self, from: Data(json.utf8))
    }

    /// Mutation check: make the decode fall back to `.off` and this fails.
    func testArchiveWrittenBeforeTheFieldDecodesAsInherit() throws {
        let archive = try decodeArchive("""
        {"projects": [{"name": "Legacy", "maxAutonomousSteps": 7}]}
        """)
        XCTAssertEqual(archive.projects[0].lspMode, .inherit)
        XCTAssertEqual(archive.projects[0].maxAutonomousSteps, 7, "the rest of the row still decodes")
    }

    func testUnknownOrWrongTypedValueDecodesAsInherit() throws {
        for raw in ["\"on\"", "\"sometimes\"", "7", "null", "true", "[]"] {
            let archive = try decodeArchive("""
            {"projects": [{"name": "P", "lspMode": \(raw)}]}
            """)
            XCTAssertEqual(archive.projects.first?.lspMode, .inherit, "value \(raw)")
        }
    }

    /// Mutation check: drop `lspMode` from `CodingKeys` and this fails.
    func testOffSurvivesARoundTripAndIsIndependentOfTheFormatterField() throws {
        var project = AppProject(name: "P")
        project.lspMode = .off
        let data = try JSONEncoder().encode(AppProjectArchive(projects: [project, AppProject(name: "Q")]))
        let back = try JSONDecoder().decode(AppProjectArchive.self, from: data)
        XCTAssertEqual(back.projects.map(\.lspMode), [.off, .inherit])
        XCTAssertEqual(back.projects.map(\.formatterMode), [.inherit, .inherit])
    }

    // MARK: precedence

    /// Mutation check: change `&&` to `||` and the first two assertions fail.
    func testBothSwitchesMustAllowServers() {
        XCTAssertFalse(ProjectLSPMode.allowsServers(appWideEnabled: true, project: .off), "project off beats app on")
        XCTAssertFalse(ProjectLSPMode.allowsServers(appWideEnabled: false, project: .inherit), "app off beats project inherit")
        XCTAssertFalse(ProjectLSPMode.allowsServers(appWideEnabled: false, project: .off))
        XCTAssertTrue(ProjectLSPMode.allowsServers(appWideEnabled: true, project: .inherit))
        XCTAssertEqual(Set(ProjectLSPMode.allCases.map(\.rawValue)), ["inherit", "off"], "no widening case exists")
    }

    func testOnlyAFlipToOffIsANarrowing() {
        XCTAssertTrue(ProjectLSPMode.isNarrowing(from: .inherit, to: .off))
        XCTAssertFalse(ProjectLSPMode.isNarrowing(from: .off, to: .off))
        XCTAssertFalse(ProjectLSPMode.isNarrowing(from: .off, to: .inherit))
        XCTAssertFalse(ProjectLSPMode.isNarrowing(from: .inherit, to: .inherit))
    }

    // MARK: the post-edit step

    private func manager(root: URL, server: ScriptedLSPServer, starts: Counter? = nil) -> LanguageServerManager {
        LanguageServerManager(rootURL: root, settings: { [.fake()] }, idleTimeout: 0) { _, _ in
            starts?.bump()
            return server
        }
    }

    private func context(_ project: LSPTestProject, mode: ProjectLSPMode) -> PostEditContext {
        PostEditContext(toolName: "write_file", relativePaths: ["Sources/A.swift"], rootURL: project.root,
                        chatID: UUID(), lspMode: mode)
    }

    /// Mutation check: drop the `context.lspMode` test in the step and the
    /// project-off case starts a server.
    func testProjectOffStartsNoServerWhileAppIsOn() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        let step = LSPDiagnosticsStep(isEnabled: { true }, managerProvider: { _ in lsp })
        let note = await step.run(context(project, mode: .off))
        XCTAssertNil(note)
        XCTAssertEqual(starts.value, 0)
    }

    func testAppOffStartsNoServerWhileProjectInherits() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        let step = LSPDiagnosticsStep(isEnabled: { false }, managerProvider: { _ in lsp })
        let note = await step.run(context(project, mode: .inherit))
        XCTAssertNil(note)
        XCTAssertEqual(starts.value, 0)
    }

    /// The control: with both allowing it the same fixture does start a
    /// server, so the two refusals above are the toggles and not a broken
    /// fixture.
    func testBothAllowingStartsTheServer() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        let step = LSPDiagnosticsStep(isEnabled: { true }, managerProvider: { _ in lsp })
        _ = await step.run(context(project, mode: .inherit))
        XCTAssertEqual(starts.value, 1)
        await lsp.stopAll()
    }

    /// Mutation check: build the context without `lspMode:` in
    /// `PostEditSeam.note` and the project-off case starts a server.
    func testSeamCarriesTheProjectModeToTheStep() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        let step = LSPDiagnosticsStep(isEnabled: { true }, managerProvider: { _ in lsp })
        var record = AppProject(name: "p", rootDirectoryPath: project.root.path)
        record.lspMode = .off
        let edit = AppToolCall(
            name: "write_file", arguments: ["path": "Sources/A.swift", "content": "x"], category: .fileWrite)
        _ = await PostEditSeam.note(call: edit, isError: false, project: record, chatID: UUID(), steps: [step])
        XCTAssertEqual(starts.value, 0)
        await lsp.stopAll()
    }

    // MARK: the tools and the manager

    /// Mutation check: delete the `projectMode` guard in the executor and the
    /// three tools reach the server.
    func testToolsRefuseForAProjectSetToOffWithoutStartingAServer() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        for (name, arguments) in [("project_diagnostics", ["file": "Sources/A.swift"]),
                                  ("document_symbols", ["file": "Sources/A.swift"]),
                                  ("workspace_symbols", ["query": "A"])] {
            do {
                _ = try await LSPToolExecutor.execute(
                    name: name, arguments: arguments, rootURL: project.root, isEnabled: true,
                    projectMode: .off, manager: lsp)
                XCTFail("\(name) should refuse")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("off for this project"), "\(name): \(error.localizedDescription)")
            }
        }
        XCTAssertEqual(starts.value, 0)
    }

    func testAppOffStillWinsOverAProjectThatInherits() async throws {
        let project = try LSPTestProject()
        let lsp = manager(root: project.root, server: ScriptedLSPServer())
        do {
            _ = try await LSPToolExecutor.execute(
                name: "document_symbols", arguments: ["file": "Sources/A.swift"], rootURL: project.root,
                isEnabled: false, projectMode: .inherit, manager: lsp)
            XCTFail("expected a refusal")
        } catch { XCTAssertTrue(error.localizedDescription.contains("turned off")) }
    }

    /// Mutation check: delete the guard in `client(for:)` and the direct call
    /// below starts a server.
    func testTheManagerItselfNeverStartsAServerWhileOff() async throws {
        let project = try LSPTestProject()
        let starts = Counter()
        let lsp = manager(root: project.root, server: ScriptedLSPServer(), starts: starts)
        await lsp.setProjectMode(.off)
        do {
            _ = try await lsp.client(for: .fake())
            XCTFail("expected a refusal")
        } catch { XCTAssertEqual(error as? LSPManagerError, .offForProject) }
        XCTAssertEqual(starts.value, 0)
        let reports = await lsp.postEdit(paths: ["Sources/A.swift"], budget: 1)
        guard case .skipped? = reports?.first?.status else { return XCTFail("expected skipped, got \(String(describing: reports))") }
        XCTAssertEqual(starts.value, 0)
    }

    /// Mutation check: make `setProjectMode(.off)` only record the mode and
    /// the running server is never killed.
    func testFlippingToOffStopsTheRunningServerAndBackOnAllowsARestart() async throws {
        let project = try LSPTestProject()
        let server = ScriptedLSPServer()
        let lsp = manager(root: project.root, server: server)
        _ = await lsp.postEdit(paths: ["Sources/A.swift"], budget: 2)
        XCTAssertFalse(server.killed)
        await lsp.setProjectMode(.off)
        XCTAssertTrue(server.killed)
        XCTAssertTrue(server.receivedMethods().contains("exit"))
        await lsp.setProjectMode(.inherit)
        let again = manager(root: project.root, server: ScriptedLSPServer())
        _ = try await again.client(for: .fake())
        await again.stopAll()
    }

    /// The registry path the app takes on a save that narrows the project.
    func testRegistryStopOnNarrowingKillsTheServer() async throws {
        let project = try LSPTestProject()
        let server = ScriptedLSPServer()
        let lsp = manager(root: project.root, server: server)
        await LanguageServerRegistry.shared.install(lsp, for: project.root)
        _ = await lsp.postEdit(paths: ["Sources/A.swift"], budget: 2)
        if ProjectLSPMode.isNarrowing(from: .inherit, to: .off) {
            await LanguageServerRegistry.shared.stop(root: project.root)
        }
        XCTAssertTrue(server.killed)
    }

    // MARK: passthrough deny list

    func testSecretLikeAndInjectionNamesAreRefused() {
        let refused = [
            "GITHUB_TOKEN", "my_api_key", "DB_PASSWORD", "CLIENT_SECRET", "AWS_PROFILE", "ANTHROPIC_BASE_URL",
            "OPENAI_ORG", "SSH_AUTH_SOCK", "GIT_SSH_COMMAND", "DYLD_INSERT_LIBRARIES", "LD_PRELOAD",
            "SERVICE_CREDENTIALS", "HTTPS_PROXY", "NODE_OPTIONS", "PYTHONPATH", "BASH_ENV",
            "PATH", "HOME", "1BAD", "has space", "a=b", "",
        ]
        for name in refused {
            XCTAssertNotNil(LSPEnvironmentPolicy.refusal(for: name), "\(name) must be refused")
        }
        for name in ["SDKROOT", "JAVA_HOME", "CLANGD_FLAGS", "RUSTUP_HOME", "_PLAIN"] {
            XCTAssertNil(LSPEnvironmentPolicy.refusal(for: name), "\(name) must be accepted")
        }
    }

    func testParseSeparatesAcceptedFromRefusedAndBoundsTheList() {
        let parsed = LSPEnvironmentPolicy.parse("JAVA_HOME, GITHUB_TOKEN  SDKROOT\nJAVA_HOME,LD_PRELOAD")
        XCTAssertEqual(parsed.accepted, ["JAVA_HOME", "SDKROOT"])
        XCTAssertEqual(parsed.refused, ["GITHUB_TOKEN", "LD_PRELOAD"])
        let many = (0..<20).map { "VAR\($0)" }.joined(separator: ",")
        XCTAssertEqual(LSPEnvironmentPolicy.parse(many).accepted.count, LSPEnvironmentPolicy.maximumNames)
    }

    // MARK: spawn environment

    private let parent = [
        "PATH": "/usr/bin", "HOME": "/Users/x", "LANG": "en_US.UTF-8", "TMPDIR": "/tmp/",
        "JAVA_HOME": "/jdk", "SDKROOT": "/sdk", "GITHUB_TOKEN": "ghp_x", "LD_PRELOAD": "/evil.so",
        "NODE_OPTIONS": "--require /evil.js",
    ]

    /// The scrub allowlist is unchanged for a row without passthrough.
    /// Mutation check: pass the whole parent environment and this fails.
    func testRowWithoutPassthroughGetsExactlyTheAllowlist() {
        let env = LSPLaunchPolicy.environment(parent: parent, executable: "/usr/bin/ls", searchPath: nil)
        XCTAssertEqual(Set(env.keys), ["PATH", "HOME", "LANG", "TMPDIR"])
    }

    func testPassthroughAddsOnlyNamedAcceptedVariablesPresentInTheApp() {
        let env = LSPLaunchPolicy.environment(
            parent: parent, executable: "/usr/bin/ls", searchPath: nil,
            passthrough: ["JAVA_HOME", "ABSENT_VAR"])
        XCTAssertEqual(Set(env.keys), ["PATH", "HOME", "LANG", "TMPDIR", "JAVA_HOME"])
        XCTAssertEqual(env["JAVA_HOME"], "/jdk")
    }

    /// The deny list is re-applied at spawn, so a hand-edited stored row
    /// cannot skip the settings field's check. Mutation check: pass
    /// `passthrough` through without `accepted(...)` and this fails.
    func testDeniedNamesNeverReachTheChildEvenFromAStoredRow() {
        let env = LSPLaunchPolicy.environment(
            parent: parent, executable: "/usr/bin/ls", searchPath: nil,
            passthrough: ["GITHUB_TOKEN", "LD_PRELOAD", "NODE_OPTIONS", "SDKROOT"])
        XCTAssertEqual(Set(env.keys), ["PATH", "HOME", "LANG", "TMPDIR", "SDKROOT"])
    }

    func testProductionTransportIsBuiltWithTheRowsPassthroughOnly() throws {
        var config = LSPServerConfig.fake()
        config.command = "/bin/sh"
        config.envPassthrough = ["SDKROOT"]
        let project = try LSPTestProject()
        // Resolving succeeds and the transport is created without spawning.
        XCTAssertNoThrow(try LSPProductionTransport.make(config: config, root: project.root, searchPath: nil))
    }

    // MARK: stored rows

    func testRowStoredBeforeThePassthroughFieldStillDecodes() throws {
        let json = """
        {"id":"custom-1","name":"clangd","extensions":["c"],"languageID":"c","command":"/usr/bin/clangd","args":[],"enabled":true,"isDiscovered":false}
        """
        let row = try JSONDecoder().decode(LSPServerConfig.self, from: Data(json.utf8))
        XCTAssertEqual(row.envPassthrough, [])
        XCTAssertEqual(row.command, "/usr/bin/clangd")
    }

    func testLoadingSanitizesStoredPassthroughAndArgumentsAndNeverTrustsTheDiscoveredRow() throws {
        let suite = "lsp-toggle-\(UUID().uuidString)"
        let defaults = try XCTUnwrap(UserDefaults(suiteName: suite))
        defer { defaults.removePersistentDomain(forName: suite) }
        var custom = LSPServerConfig.fake(id: "custom-1")
        custom.envPassthrough = ["SDKROOT", "GITHUB_TOKEN"]
        custom.args = ["--ok", String(repeating: "x", count: 600)]
        var discovered = LSPServerConfig.sourcekitDefault
        discovered.envPassthrough = ["SDKROOT"]
        discovered.args = ["--evil"]
        LSPSettings.save([discovered, custom], defaults: defaults)
        let loaded = LSPSettings.servers(defaults: defaults)
        XCTAssertEqual(loaded.first { $0.id == "custom-1" }?.envPassthrough, ["SDKROOT"])
        XCTAssertEqual(loaded.first { $0.id == "custom-1" }?.args, ["--ok"])
        XCTAssertEqual(loaded.first { $0.id == LSPServerConfig.sourcekitID }?.envPassthrough, [])
        XCTAssertEqual(loaded.first { $0.id == LSPServerConfig.sourcekitID }?.args, [])
    }

    func testArgumentsAreAnArrayOnePerLineNeverShellSplit() {
        XCTAssertEqual(LSPSettings.parseArguments("--flag=a b\n\n  -x  \n$(rm -rf ~)"),
                       ["--flag=a b", "-x", "$(rm -rf ~)"])
    }

    /// A workspace-relative executable is still refused with arguments set.
    func testRelativeExecutableStillRefusedWithPassthroughAndArgs() throws {
        var config = LSPServerConfig.fake()
        config.command = "./tools/server"
        config.args = ["--stdio"]
        config.envPassthrough = ["SDKROOT"]
        let project = try LSPTestProject()
        XCTAssertThrowsError(try LSPLaunchPolicy.resolve(config: config, rootURL: project.root, searchPath: nil)) {
            XCTAssertEqual($0 as? LSPLaunchError, .relativeCommand("./tools/server"))
        }
    }
}

private final class Counter: @unchecked Sendable {
    private let lock = NSLock()
    private var count = 0
    func bump() { lock.lock(); count += 1; lock.unlock() }
    var value: Int { lock.lock(); defer { lock.unlock() }; return count }
}
