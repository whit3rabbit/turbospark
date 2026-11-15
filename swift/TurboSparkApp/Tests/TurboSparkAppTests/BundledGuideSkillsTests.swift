import Foundation
import XCTest
@testable import TurboSparkApp

/// ZP-20: the bundled `turbospark-guide` and `visualize` skills.
///
/// Skill text guides the agent and enforces nothing, so these tests pin what
/// a code path can check: both ship in the resource bundle and resolve at the
/// lowest precedence, a user switch-off hides them, every store file name the
/// guide cites is still a literal in the app sources (the docs-drift guard),
/// every Settings pane it names is a real pane, and the visualize contract
/// keeps its load-bearing clauses.
final class BundledGuideSkillsTests: XCTestCase {
    private let guideName = "turbospark-guide"
    private let visualizeName = "visualize"

    override func tearDown() {
        for name in [guideName, visualizeName] {
            SkillManager.shared.setSkillEnabled(true, scope: .bundled, name: name)
        }
        SkillManager.shared.invalidateResolutionCache()
        super.tearDown()
    }

    // MARK: - Discovery

    private func skill(_ name: String) throws -> AppSkill {
        try XCTUnwrap(
            SkillManager.shared.discoverBundledSkills().first { $0.name == name },
            "\(name) must ship in TurboSparkAppResources/BundledSkills")
    }

    private func text(_ name: String) throws -> String {
        try String(contentsOf: skill(name).sourceURL, encoding: .utf8)
    }

    func testBothSkillsShipAsBundledAndAreOfferedToTheSkillTool() throws {
        for name in [guideName, visualizeName] {
            let found = try skill(name)
            XCTAssertEqual(found.scope, .bundled)
            XCTAssertFalse(found.skillDescription.isEmpty, name)
            XCTAssertFalse(SkillManager.shared.ownsSkill(found), "bundled skills are read-only")
        }
        let offered = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
        let listing = SkillManager.shared.formatSkillsWithinBudget(offered, contextWindowTokens: nil)
        XCTAssertTrue(listing.contains("\(guideName) [Bundled]"), listing)
        XCTAssertTrue(listing.contains("\(visualizeName) [Bundled]"), listing)
    }

    /// Discovery reads one folder level, so a nested `diagnose-*/SKILL.md`
    /// would never be offered (review finding M5). The pack is therefore one
    /// skill per top-level folder, and no skill hides below another.
    func testNoSkillIsNestedBelowAnotherSkillFolder() throws {
        let root = try XCTUnwrap(SkillManager.shared.bundledSkillsDirectory)
        let fm = FileManager.default
        for entry in try fm.contentsOfDirectory(atPath: root.path) {
            let dir = root.appendingPathComponent(entry)
            let nested = (try? fm.subpathsOfDirectory(atPath: dir.path)) ?? []
            let stray = nested.filter { $0.hasSuffix("SKILL.md") && $0 != "SKILL.md" }
            XCTAssertTrue(stray.isEmpty, "\(entry) nests skills that discovery cannot see: \(stray)")
        }
    }

    func testUserSwitchOffHidesABundledSkillFromTheSkillTool() throws {
        SkillManager.shared.setSkillEnabled(false, scope: .bundled, name: guideName)
        let offered = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
        let listing = SkillManager.shared.formatSkillsWithinBudget(offered, contextWindowTokens: nil)
        XCTAssertFalse(listing.contains("\(guideName) ["), listing)
        XCTAssertTrue(listing.contains("\(visualizeName) [Bundled]"), "only the switched-off skill goes")
        XCTAssertFalse(try skill(guideName).isEnabled)
    }

    func testSkillsDoNotDependOnAPythonOrNodeRuntime() throws {
        for name in [guideName, visualizeName] {
            let body = try text(name).lowercased()
            for banned in ["python", "pip install", "npm install", "node -e", "node script", "pnpm ", "yarn "] {
                XCTAssertFalse(body.contains(banned), "\(name) mentions \(banned)")
            }
        }
    }

    // MARK: - Guide drift guard

    /// Every store file the guide names must still be written or read by the
    /// app. This checks FILE NAMES only: it does not prove the prose around
    /// them is right, and renaming a pane label is covered by the pane test.
    func testEveryStoreFileTheGuideCitesIsALiteralInTheAppSources() throws {
        let names = Self.citedStoreFileNames(in: try text(guideName))
        // A floor, so an extraction rule that silently matches nothing fails.
        for expected in [
            "settings.json", "projects_archive.json", "global_mcp_servers.json",
            "disabled_items.json", "trusted_hook_hashes.json", "custom_hooks.json",
            "installed_plugins.json", "local_plugins.json", "known_marketplaces.json",
            "hooks.json", "plugin.json", ".mcp.json",
        ] {
            XCTAssertTrue(names.contains(expected), "extraction lost \(expected): \(names)")
        }
        let sources = try Self.appSourceText()
        XCTAssertEqual(Self.unresolved(names, in: sources), [], "the guide cites a file the app no longer names")
    }

    /// Mutation check for the guard above: a renamed store file must be
    /// reported, and a present one must not be.
    func testDriftGuardReportsARenamedStoreFile() throws {
        let sources = try Self.appSourceText()
        let mutated = "Read `global_mcp_servers_renamed.json` and `settings.json`."
        let names = Self.citedStoreFileNames(in: mutated)
        XCTAssertEqual(Self.unresolved(names, in: sources), ["global_mcp_servers_renamed.json"])
    }

    func testEverySettingsPaneTheGuideNamesExists() throws {
        let body = try text(guideName)
        let panes = Set(AppSettingsView.SettingsTab.allCases.map(\.title))
        // "Settings > X" references, and the one sentence that lists them all.
        let referenced = Self.settingsPaneReferences(in: body)
        XCTAssertGreaterThanOrEqual(referenced.count, 5, "extraction found too few panes: \(referenced)")
        XCTAssertEqual(referenced.subtracting(panes), [], "the guide names a pane that does not exist")
        let flat = body.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for pane in panes {
            XCTAssertTrue(flat.contains(pane), "the pane list lost \(pane)")
        }
    }

    func testGuideCoversEveryDiagnosisAndRedactsSecrets() throws {
        let body = try text(guideName)
        for heading in [
            "## Diagnose hooks", "## Diagnose commands", "## Diagnose skills",
            "## Diagnose MCP", "## Diagnose plugins",
        ] {
            XCTAssertTrue(body.contains(heading), heading)
        }
        XCTAssertTrue(body.contains("<redacted>"), "the guide must tell the agent to redact secrets")
    }

    /// The MCP section states the child environment allowlist. Pin it to the
    /// engine's own constant so the prose cannot drift from the code.
    func testGuideMcpEnvironmentClaimMatchesTheEngine() throws {
        let body = try text(guideName)
        for key in McpClientEngine.baselineEnvironmentKeys {
            XCTAssertTrue(body.contains("`\(key)`"), "guide lost baseline env key \(key)")
        }
    }

    // MARK: - Visualize contract

    func testVisualizeKeepsItsLoadBearingClauses() throws {
        // Collapse line wraps so a phrase may span lines.
        let body = try text(visualizeName).split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for needle in [
            "1 MB", "No external loads", "CDN", "@import", "fetch(", "<!doctype html>",
            "Network is off for this preview", "Save a copy", "prefers-color-scheme",
            "advisory", "No charting library",
        ] {
            XCTAssertTrue(body.contains(needle), "visualize lost: \(needle)")
        }
    }

    /// The contract leans on three behaviours of the real panel; pin each to
    /// the code instead of to prose.
    func testVisualizeRelianceOnTheArtifactPanelHolds() throws {
        // An .html file written by a file write tool becomes an artifact.
        XCTAssertTrue(AppArtifactRenderKind.htmlExtensions.contains("html"))
        XCTAssertTrue(AppArtifactRenderKind.htmlExtensions.contains("htm"))
        let page = ArtifactRegistrar.ProducedFile(
            url: URL(fileURLWithPath: "/tmp/chart.html"), toolCallID: UUID(),
            toolName: "write_file", origin: .fileWrite)
        XCTAssertEqual(ArtifactRegistrar.artifactCandidates(from: [page]), [page])
        // Offline: a page with network authority is never a file document.
        let file = ArtifactWebDocument.file(
            page: URL(fileURLWithPath: "/tmp/a/chart.html"), readAccessFolder: URL(fileURLWithPath: "/tmp/a"))
        XCTAssertNotNil(file.readAccessFolder)
        XCTAssertFalse(ArtifactWebView.allowsFileNavigation(
            to: URL(fileURLWithPath: "/tmp/a/x.js"), readFolder: file.readAccessFolder, networkAllowed: true))
        XCTAssertTrue(ArtifactWebView.allowsFileNavigation(
            to: URL(fileURLWithPath: "/tmp/a/x.js"), readFolder: file.readAccessFolder, networkAllowed: false))
        // The panel strings the skill quotes still exist in the panel source.
        let panel = try Self.sourceText("Generation/ArtifactPanelView.swift")
        XCTAssertTrue(panel.contains("Network is off for this preview"))
        XCTAssertTrue(panel.contains("Save a copy"))
    }

    // MARK: - Helpers

    private static var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp", isDirectory: true)
    }

    private static func sourceText(_ relative: String) throws -> String {
        try String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8)
    }

    /// Every Swift source under `Sources/TurboSparkApp`, joined. Scanned for
    /// the file-name literals the app uses to open its stores.
    private static func appSourceText() throws -> String {
        let fm = FileManager.default
        let enumerator = try XCTUnwrap(fm.enumerator(at: sourcesRoot, includingPropertiesForKeys: nil))
        var out = ""
        for case let url as URL in enumerator where url.pathExtension == "swift" {
            if let text = try? String(contentsOf: url, encoding: .utf8) { out += text }
        }
        return out
    }

    /// Extraction rule: the final path component of every inline code span
    /// that ends in `.json`. Example fragments with `<...>` placeholders and
    /// spans that are not a bare file or path are ignored.
    static func citedStoreFileNames(in text: String) -> Set<String> {
        var names = Set<String>()
        let parts = text.components(separatedBy: "`")
        // Odd indices are inside a code span.
        for index in stride(from: 1, to: parts.count, by: 2) {
            let span = parts[index]
            guard span.hasSuffix(".json"), !span.contains(" ") else { continue }
            let last = span.split(separator: "/").last.map(String.init) ?? span
            if !last.contains("<") { names.insert(last) }
        }
        return names
    }

    static func unresolved(_ names: Set<String>, in sources: String) -> [String] {
        // A bare literal ("hooks.json") or the tail of a path literal
        // ("~/.turbospark/hooks.json"), never a mere substring.
        names.filter { !sources.contains("\"\($0)\"") && !sources.contains("/\($0)\"") }.sorted()
    }

    /// `Settings > Pane Name` references, the pane ending at punctuation, a
    /// backtick or a line wrap (a pane name never contains `.` or `,`).
    static func settingsPaneReferences(in text: String) -> Set<String> {
        let flat = text.split(whereSeparator: \.isWhitespace).joined(separator: " ")
        let panes = Set(AppSettingsView.SettingsTab.allCases.map(\.title))
        var found = Set<String>()
        var remaining = Substring(flat)
        while let range = remaining.range(of: "Settings > ") {
            remaining = remaining[range.upperBound...]
            // Longest pane title that prefixes the remainder.
            if let pane = panes.filter({ remaining.hasPrefix($0) }).max(by: { $0.count < $1.count }) {
                found.insert(pane)
            } else {
                found.insert(String(remaining.prefix(24)))
            }
        }
        return found
    }
}
