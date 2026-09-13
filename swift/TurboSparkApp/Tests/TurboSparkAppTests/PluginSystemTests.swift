import XCTest
@testable import TurboSparkApp

/// Manifest parsing and validation: the shapes Claude Code's plugin.json
/// accepts, what this port drops with a diagnostic, and what fails the
/// plugin alone.
final class PluginManifestTests: XCTestCase {

    private func parse(_ json: String) throws -> PluginManifest {
        try PluginManifestParser.parseManifest(
            data: Data(json.utf8), fallbackName: "fallback", sourceDescription: "test")
    }

    func testFullManifestParsesEverySurface() throws {
        let manifest = try parse("""
        {
            "name": "review-tools",
            "version": "1.2.3",
            "description": "Code review helpers",
            "author": {"name": "Ada", "email": "a@example.com"},
            "keywords": ["review", "lint"],
            "commands": {
                "about": {"source": "./commands/about.md", "description": "About"},
                "inline": {"content": "Do the thing", "argumentHint": "<what>"}
            },
            "agents": ["agents/explorer.md"],
            "skills": "./skills",
            "hooks": ["./hooks/extra.json"],
            "mcpServers": "./servers.json",
            "userConfig": {
                "api_key": {"type": "string", "title": "API Key", "sensitive": true},
                "depth": {"type": "number", "title": "Depth", "default": 5}
            }
        }
        """)
        XCTAssertEqual(manifest.name, "review-tools")
        XCTAssertEqual(manifest.version, "1.2.3")
        XCTAssertEqual(manifest.author?.name, "Ada")
        // Object-map fields are keyed dictionaries, so iteration order is
        // unspecified: look entries up by KEY, never by position.
        XCTAssertEqual(manifest.commandSpecs.count, 2)
        let about = manifest.commandSpecs.first { $0.name == "about" }
        let inline = manifest.commandSpecs.first { $0.name == "inline" }
        XCTAssertEqual(about?.sourcePath, "./commands/about.md")
        XCTAssertNil(about?.inlineContent)
        XCTAssertEqual(inline?.inlineContent, "Do the thing")
        XCTAssertEqual(manifest.agentPaths, ["agents/explorer.md"])
        XCTAssertEqual(manifest.skillPaths, ["./skills"])
        XCTAssertEqual(manifest.hookFilePaths, ["./hooks/extra.json"])
        XCTAssertEqual(manifest.mcpServerFilePaths, ["./servers.json"])
        XCTAssertEqual(manifest.userConfig.count, 2)
        XCTAssertEqual(
            manifest.userConfig.first { $0.key == "api_key" }?.optionSpec.isSensitive, true)
        XCTAssertEqual(
            manifest.userConfig.first { $0.key == "depth" }?.optionSpec.defaultValue, "5")
    }

    func testUnknownTopLevelKeysAreIgnoredNotFatal() throws {
        let manifest = try parse("""
        {"name": "x", "futureField": {"anything": true}}
        """)
        XCTAssertEqual(manifest.name, "x")
    }

    func testCommandEntryWithBothSourceAndContentKeepsSourceWithDiagnostic() throws {
        let manifest = try parse("""
        {"name": "x", "commands": {"a": {"source": "./a.md", "content": "text"}}}
        """)
        XCTAssertEqual(manifest.commandSpecs.first?.sourcePath, "./a.md")
        XCTAssertNil(manifest.commandSpecs.first?.inlineContent)
        XCTAssertTrue(manifest.unsupportedNotes.contains { $0.contains("both source and content") })
    }

    func testCommandEntryWithNeitherSourceNorContentIsDropped() throws {
        let manifest = try parse("""
        {"name": "x", "commands": {"a": {"description": "orphan"}}}
        """)
        XCTAssertTrue(manifest.commandSpecs.isEmpty)
        XCTAssertTrue(manifest.unsupportedNotes.contains { $0.contains("needs one of source or content") })
    }

    func testUnsupportedSurfacesAreRecordedNotApplied() throws {
        let manifest = try parse("""
        {"name": "x", "lspServers": {}, "outputStyles": "./s.md", "channels": [], "settings": {"agent": {}}}
        """)
        XCTAssertEqual(manifest.unsupportedNotes.count, 4)
        XCTAssertTrue(manifest.unsupportedNotes.allSatisfy { $0.contains("ignored") })
    }

    func testUnparseableJSONFailsThePluginAlone() {
        XCTAssertThrowsError(try parse("{not json")) { error in
            XCTAssertTrue(error is PluginLoadError)
        }
    }

    func testReservedNamesAreRejected() {
        XCTAssertThrowsError(try parse(#"{"name": "inline"}"#))
        XCTAssertThrowsError(try parse(#"{"name": "builtin"}"#))
        XCTAssertThrowsError(try parse(#"{"name": "has space"}"#))
    }

    func testMarketplaceNameValidationRefusesTraversalAndNonASCII() {
        XCTAssertNotNil(PluginManifestParser.validateMarketplaceName("a/b"))
        XCTAssertNotNil(PluginManifestParser.validateMarketplaceName(".."))
        XCTAssertNotNil(PluginManifestParser.validateMarketplaceName("a b"))
        XCTAssertEqual(PluginManifestParser.validateMarketplaceName("caf\u{e9}"), "marketplace name 'caf\u{e9}' must be ASCII")
        XCTAssertNil(PluginManifestParser.validateMarketplaceName("good-name_1"))
    }

    func testPathsThatClimbOutAreDropped() throws {
        let manifest = try parse("""
        {"name": "x", "agents": ["../escape.md", "ok.md", "/absolute.md"], "hooks": "./../../hooks.json", "mcpServers": "./../mcp.json"}
        """)
        XCTAssertEqual(manifest.agentPaths, ["ok.md"])
        XCTAssertTrue(manifest.hookFilePaths.isEmpty)
        XCTAssertTrue(manifest.mcpServerFilePaths.isEmpty)
        XCTAssertEqual(manifest.unsupportedNotes.filter { $0.contains("outside the plugin") }.count, 4)
    }

    func testSynthesizedManifestNamesTheSource() {
        let manifest = PluginManifestParser.synthesizedManifest(
            name: "dir-plugin", sourceDescription: "/tmp/plugins/dir-plugin")
        XCTAssertEqual(manifest.name, "dir-plugin")
        XCTAssertEqual(manifest.descriptionText, "Plugin from /tmp/plugins/dir-plugin")
    }

    func testStrictMergeFillsGapsFromTheEntry() throws {
        let base = try parse("""
        {"name": "x", "version": "0.1.0", "commands": {"local": {"source": "./a.md"}}}
        """)
        let merged = try PluginManifestParser.merging(
            pluginManifest: base,
            entryDict: [
                "name": "x",
                "version": "2.0.0",
                "description": "From the marketplace",
                "skills": "./skills"
            ],
            pluginName: "x", sourceDescription: "test")
        XCTAssertEqual(merged.version, "0.1.0", "the plugin's own manifest wins a filled field")
        XCTAssertEqual(merged.descriptionText, "From the marketplace", "the entry fills a gap")
        XCTAssertEqual(merged.skillPaths, ["./skills"])
        XCTAssertEqual(merged.commandSpecs.count, 1)
    }

    func testStrictMergeConflictsWhenBothSidesDefineTheSameSurface() throws {
        let base = try parse("""
        {"name": "x", "commands": {"local": {"source": "./a.md"}}}
        """)
        XCTAssertThrowsError(try PluginManifestParser.merging(
            pluginManifest: base,
            entryDict: ["name": "x", "commands": ["remote": ["source": "./b.md"]]],
            pluginName: "x", sourceDescription: "test")) { error in
            guard let loadError = error as? PluginLoadError else {
                return XCTFail("expected PluginLoadError")
            }
            XCTAssertTrue(loadError.reason.contains("both the plugin manifest and the marketplace entry"))
        }
    }

    func testNonStrictEntrySynthesizesWhenThePluginHasNoManifest() throws {
        let merged = try PluginManifestParser.merging(
            pluginManifest: nil,
            entryDict: ["name": "loose", "skills": "./skills", "strict": false],
            pluginName: "loose", sourceDescription: "test")
        XCTAssertEqual(merged.name, "loose")
        XCTAssertEqual(merged.skillPaths, ["./skills"])
    }

    func testStrictEntryWithoutAPluginManifestIsRefused() {
        XCTAssertThrowsError(try PluginManifestParser.merging(
            pluginManifest: nil,
            entryDict: ["name": "strict-one"],
            pluginName: "strict-one", sourceDescription: "test"))
    }
}

/// Variable substitution: the `${CLAUDE_PLUGIN_ROOT}` / `${CLAUDE_PLUGIN_DATA}`
/// / `${user_config.KEY}` contract.
final class PluginVariableExpanderTests: XCTestCase {

    private func expand(
        _ text: String,
        sensitive: Set<String> = [],
        preserve: Bool = false,
        values: [String: String] = [:]
    ) -> String {
        PluginVariableExpander.expand(
            text, pluginRoot: "/plugins/root", pluginData: "/plugins/data",
            optionValue: { values[$0] }, sensitiveKeys: sensitive, preserveSensitive: preserve)
    }

    func testRootAndDataSubstitute() {
        XCTAssertEqual(
            expand("run ${CLAUDE_PLUGIN_ROOT}/bin.sh and log to ${CLAUDE_PLUGIN_DATA}"),
            "run /plugins/root/bin.sh and log to /plugins/data")
    }

    func testOptionValuesSubstitute() {
        XCTAssertEqual(expand("key=${user_config.api_key}", values: ["api_key": "abc"]), "key=abc")
    }

    func testSensitiveValueIsMaskedInProseAndPreservedForEnvironments() {
        let prose = expand(
            "key=${user_config.token}", sensitive: ["token"], values: ["token": "sekret"])
        XCTAssertTrue(prose.contains(PluginVariableExpander.sensitivePlaceholder))
        XCTAssertFalse(prose.contains("sekret"))

        let env = expand(
            "key=${user_config.token}", sensitive: ["token"],
            preserve: true, values: ["token": "sekret"])
        XCTAssertEqual(env, "key=sekret")
    }

    func testUnknownKeyStaysLiteral() {
        XCTAssertEqual(expand("${user_config.missing}"), "${user_config.missing}")
    }

    func testReferencedKeysAreFoundInOrder() {
        XCTAssertEqual(
            PluginVariableExpander.referencedOptionKeys(in: "a ${user_config.one} b ${user_config.two}"),
            ["one", "two"])
    }
}
