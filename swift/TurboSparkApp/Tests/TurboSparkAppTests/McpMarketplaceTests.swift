import XCTest

@testable import TurboSparkApp

/// Covers the MCP marketplace: what a catalog decodes to, and what an entry is
/// allowed to become.
final class McpMarketplaceTests: XCTestCase {

    // MARK: - Manifest decoding

    func testAManifestDecodesItsServers() throws {
        let json = """
            {
              "name": "Example Catalog",
              "description": "A few servers",
              "owner": "example",
              "servers": [
                {
                  "name": "memory",
                  "description": "Knowledge graph",
                  "version": "1.2.0",
                  "category": "Storage",
                  "transport": {
                    "type": "stdio",
                    "command": "npx",
                    "args": ["-y", "@example/memory"]
                  }
                }
              ]
            }
            """
        let manifest = try JSONDecoder().decode(
            McpMarketplaceManifest.self, from: Data(json.utf8))

        XCTAssertEqual(manifest.name, "Example Catalog")
        XCTAssertEqual(manifest.owner, "example")
        XCTAssertEqual(manifest.servers.count, 1)
        XCTAssertEqual(manifest.servers.first?.name, "memory")
        XCTAssertEqual(manifest.servers.first?.version, "1.2.0")
        XCTAssertEqual(manifest.servers.first?.commandSummary, "npx -y @example/memory")
    }

    /// One unreadable entry is one dropped row, never a rejected catalog. The
    /// entry below has no `transport`, which is the one field with no decode
    /// tolerance, so it cannot be launched and has nothing to fall back to.
    func testAManifestWithOneUnreadableEntryKeepsTheRest() throws {
        let json = """
            {
              "name": "Mixed",
              "servers": [
                {"name": "good", "description": "", "transport":
                  {"type": "stdio", "command": "echo"}},
                {"name": "broken", "description": "no transport at all"},
                {"name": "also-good", "description": "", "transport":
                  {"type": "stdio", "command": "cat"}}
              ]
            }
            """
        let manifest = try JSONDecoder().decode(
            McpMarketplaceManifest.self, from: Data(json.utf8))

        XCTAssertEqual(manifest.servers.map(\.name), ["good", "also-good"])
    }

    /// Container-level tolerance would be the bug: a `servers` key that is not
    /// an array must throw, so the file is quarantined rather than silently
    /// read as an empty catalog.
    func testAManifestWhoseServersKeyIsNotAnArrayThrows() {
        let json = """
            {"name": "Broken", "servers": "this used to be an array"}
            """
        XCTAssertThrowsError(
            try JSONDecoder().decode(McpMarketplaceManifest.self, from: Data(json.utf8)))
    }

    // MARK: - Install rules

    private func entry(
        name: String, command: String = "/bin/echo", args: [String] = []
    ) -> McpMarketplaceEntry {
        McpMarketplaceEntry(
            name: name,
            entryDescription: "fixture",
            transport: .stdio(command: command, args: args))
    }

    /// A catalog is a file in somebody else's repository naming a binary this
    /// app will spawn. It arrives switched off and not auto-approved.
    func testAnInstalledEntryIsDisabledAndNotAutoApproved() throws {
        let config = try McpMarketplaceManager().makeServerConfig(
            from: entry(name: "memory"), marketplaceName: "Example", existingNames: [])

        XCTAssertFalse(config.isEnabled)
        XCTAssertFalse(config.autoApprove)
        XCTAssertEqual(config.name, "memory")
        XCTAssertEqual(config.sourcePath, "marketplace:Example")
    }

    /// Name is the identity key both resolvers use, so a duplicate is not a
    /// cosmetic problem: the second server can never be dialled.
    func testAnEntryCollidingWithAnExistingNameIsRefused() {
        XCTAssertThrowsError(
            try McpMarketplaceManager().makeServerConfig(
                from: entry(name: "memory"),
                marketplaceName: "Example",
                existingNames: ["memory"])
        ) { error in
            XCTAssertEqual(
                error as? McpMarketplaceManager.MarketplaceError, .nameCollision("memory"))
        }
    }

    /// Both resolvers lowercase, so the collision check must too.
    func testTheCollisionCheckIsCaseInsensitive() {
        XCTAssertThrowsError(
            try McpMarketplaceManager().makeServerConfig(
                from: entry(name: "Memory"),
                marketplaceName: "Example",
                existingNames: ["  memory  "])
        )
    }

    func testAnEntryWithNoNameIsRefused() {
        XCTAssertThrowsError(
            try McpMarketplaceManager().makeServerConfig(
                from: entry(name: "   "), marketplaceName: "Example", existingNames: [])
        ) { error in
            XCTAssertEqual(error as? McpMarketplaceManager.MarketplaceError, .emptyName)
        }
    }

    /// Validate BEFORE the row exists (state#107's rule). An entry naming a
    /// command that resolves nowhere would otherwise sit in the list failing
    /// every call, with the failure arriving at the first tool call rather than
    /// at install time.
    func testAnEntryWhoseCommandResolvesNowhereIsRefused() {
        let missing = "turbospark-no-such-command-\(UUID().uuidString)"
        XCTAssertThrowsError(
            try McpMarketplaceManager().makeServerConfig(
                from: entry(name: "ghost", command: missing),
                marketplaceName: "Example",
                existingNames: [])
        ) { error in
            XCTAssertEqual(
                error as? McpMarketplaceManager.MarketplaceError, .commandNotFound(missing))
        }
    }

    // MARK: - Name rule

    func testNameIsTakenIgnoresCaseAndSurroundingSpace() {
        XCTAssertTrue(McpServerConfig.nameIsTaken(" Memory ", among: ["memory"]))
        XCTAssertFalse(McpServerConfig.nameIsTaken("memory", among: ["memories"]))
    }

    /// An empty candidate is not "taken" -- the empty-name refusal is a
    /// separate rule with its own message, and folding them together would
    /// report a collision that does not exist.
    func testAnEmptyNameIsNotReportedAsTaken() {
        XCTAssertFalse(McpServerConfig.nameIsTaken("   ", among: ["", "memory"]))
    }

    /// This is what lets an EDIT keep its own name.
    func testTheConfigOverloadExcludesTheRowBeingEdited() {
        let existing = McpServerConfig(name: "memory", transport: .stdio(command: "echo"))
        XCTAssertFalse(
            McpServerConfig.nameIsTaken(
                "memory", among: [existing], excludingID: existing.id))
        XCTAssertTrue(
            McpServerConfig.nameIsTaken("memory", among: [existing], excludingID: UUID()))
    }

    // MARK: - Git cache naming

    func testACacheDirectoryNameCarriesNoPathSeparators() {
        let name = MarketplaceGit.cacheDirectoryName(
            for: "git@github.com:owner/repo.git")
        XCTAssertFalse(name.contains("/"))
        XCTAssertFalse(name.contains(":"))
        XCTAssertFalse(name.isEmpty)
    }

    /// Two different repositories must not share one cache directory.
    func testTwoRepositoriesGetDifferentCacheDirectories() {
        XCTAssertNotEqual(
            MarketplaceGit.cacheDirectoryName(for: "https://example.com/a.git"),
            MarketplaceGit.cacheDirectoryName(for: "https://example.com/b.git"))
    }

    // MARK: - Reading a local catalog

    func testALocalFolderCatalogIsRead() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-catalog-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let json = """
            {"name": "Local", "servers": [
              {"name": "echo", "description": "", "transport":
                {"type": "stdio", "command": "echo"}}]}
            """
        try Data(json.utf8).write(
            to: directory.appendingPathComponent("mcp-marketplace.json"), options: .atomic)

        let manifest = try await McpMarketplaceManager(cacheRoot: directory)
            .fetchMarketplace(source: .directory(path: directory.path))

        XCTAssertEqual(manifest.name, "Local")
        XCTAssertEqual(manifest.servers.first?.name, "echo")
    }

    func testAFolderWithNoManifestReportsThePathItLookedAt() async {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-empty-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await McpMarketplaceManager(cacheRoot: directory)
                .fetchMarketplace(source: .directory(path: directory.path))
            XCTFail("Expected a missing-manifest error")
        } catch {
            XCTAssertTrue(
                error.localizedDescription.contains("mcp-marketplace.json"),
                "Error should name the file it looked for: \(error.localizedDescription)")
        }
    }

    // MARK: - Editor form validation

    private func form(
        name: String = "memory",
        transport: McpServerFormValidation.Transport = .stdio,
        command: String = "npx",
        url: String = "",
        existing: [String] = []
    ) -> McpServerFormValidation {
        McpServerFormValidation(
            name: name, transport: transport, command: command,
            endpointURLText: url, existingNames: existing)
    }

    func testAWellFormedStdioFormIsValid() {
        XCTAssertNil(form().message)
    }

    func testAnEmptyNameBlocksSave() {
        XCTAssertEqual(form(name: "  ").message, "A server name is required.")
    }

    func testAnEmptyCommandBlocksSave() {
        XCTAssertEqual(form(command: " ").message, "A command is required.")
    }

    /// The editor half of the identity rule. Two servers sharing a name means
    /// the second can never be dialled.
    func testADuplicateNameBlocksSave() {
        XCTAssertEqual(
            form(name: "Memory", existing: ["memory"]).message,
            "A server named 'Memory' already exists.")
    }

    /// `existingNames` arrives with the edited row removed, so an edit that
    /// changes nothing but the command still saves.
    func testEditingAServerCanKeepItsOwnName() {
        XCTAssertNil(form(name: "memory", existing: ["other"]).message)
    }

    /// **THE SILENT FALLBACK THIS REPLACED.** `buildConfig` used to write
    /// `URL(string: text) ?? URL(string: "http://localhost:8000/sse")!`, so a
    /// URL that did not parse was accepted and saved pointing somewhere the
    /// user never typed.
    func testAnUnparseableEndpointURLBlocksSaveRatherThanBeingSubstituted() {
        let invalid = form(transport: .sse, url: "not a url at all")
        XCTAssertNil(invalid.parsedEndpointURL)
        XCTAssertEqual(invalid.message, "A valid server URL is required.")
    }

    /// A scheme-less string parses as a relative URL and is not an endpoint.
    func testASchemelessEndpointIsRefused() {
        XCTAssertNil(form(transport: .sse, url: "example.com/sse").parsedEndpointURL)
    }

    func testAValidEndpointURLIsAccepted() {
        let valid = form(transport: .sse, url: " https://example.com/sse ")
        XCTAssertEqual(valid.parsedEndpointURL?.absoluteString, "https://example.com/sse")
        XCTAssertNil(valid.message)
    }

    /// The command field belongs to the stdio arm alone, so an empty one must
    /// not block an endpoint the user did fill in.
    func testAnEmptyCommandDoesNotBlockTheEndpointArm() {
        XCTAssertNil(form(transport: .sse, command: "", url: "https://example.com/sse").message)
    }

    // MARK: - Fetch then install, over one catalog

    /// The three outcomes a real catalog produces, on one manifest: an entry
    /// that installs, an entry refused before it reaches the store, and an
    /// entry dropped at decode. Self-contained, so it needs no network and no
    /// checked-in fixture.
    func testFetchThenInstallOverACatalogWithGoodAndBadEntries() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-mixed-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let missing = "turbospark-no-such-command-\(UUID().uuidString)"
        let json = """
            {
              "name": "Mixed Catalog",
              "owner": "fixture",
              "servers": [
                {"name": "echo-server", "description": "Echoes.",
                 "transport": {"type": "stdio", "command": "/bin/echo", "args": ["hello"],
                               "cwd": "/tmp", "envPassthrough": ["LANG"]}},
                {"name": "missing-binary", "description": "Not installed here.",
                 "transport": {"type": "stdio", "command": "\(missing)"}},
                {"name": "no-transport", "description": "Dropped at decode."}
              ]
            }
            """
        try Data(json.utf8).write(
            to: directory.appendingPathComponent("mcp-marketplace.json"), options: .atomic)

        let manager = McpMarketplaceManager(cacheRoot: directory)
        let manifest = try await manager.fetchMarketplace(source: .directory(path: directory.path))

        XCTAssertEqual(
            manifest.servers.map(\.name), ["echo-server", "missing-binary"],
            "The entry with no transport is dropped, the other two survive")

        let installed = try manager.makeServerConfig(
            from: manifest.servers[0], marketplaceName: manifest.name, existingNames: [])
        XCTAssertFalse(installed.isEnabled)
        XCTAssertFalse(installed.autoApprove)
        XCTAssertEqual(installed.sourcePath, "marketplace:Mixed Catalog")
        XCTAssertEqual(
            installed.commandSummary, "/bin/echo hello",
            "The command line is what the sheet shows before the Install button")

        guard case .stdio(_, _, _, let cwd, let passthrough) = installed.transport else {
            return XCTFail("Expected stdio transport")
        }
        XCTAssertEqual(cwd, "/tmp")
        XCTAssertEqual(passthrough, ["LANG"])

        XCTAssertThrowsError(
            try manager.makeServerConfig(
                from: manifest.servers[1], marketplaceName: manifest.name, existingNames: []),
            "An entry naming a command that resolves nowhere is refused before install")
    }
}
