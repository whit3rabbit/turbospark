import Foundation
import XCTest
@testable import TurboSparkApp

/// ZP-07: the bundled `plugin-creator` skill, its discovery, and the
/// "Create Plugin" draft. The skill text guides the agent and enforces
/// nothing, so these tests pin the parts a code path can check: it ships in
/// the resource bundle, it resolves with the lowest precedence, it cannot be
/// edited, and the manifest templates it hands the agent are accepted by the
/// real parsers.
final class PluginCreatorSkillTests: XCTestCase {
    private var tempDir: URL!
    private var savedBundledRoot: URL?

    override func setUp() {
        super.setUp()
        tempDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ts_zp07_\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
        savedBundledRoot = SkillManager.shared.bundledSkillsDirectory
    }

    override func tearDown() {
        SkillManager.shared.bundledSkillsDirectory = savedBundledRoot
        SkillManager.shared.invalidateResolutionCache()
        try? FileManager.default.removeItem(at: tempDir)
        super.tearDown()
    }

    private func bundledSkill() throws -> AppSkill {
        try XCTUnwrap(
            SkillManager.shared.discoverBundledSkills().first { $0.name == "plugin-creator" },
            "plugin-creator must ship in TurboSparkAppResources/BundledSkills")
    }

    private func skillText() throws -> String {
        try String(contentsOf: bundledSkill().sourceURL, encoding: .utf8)
    }

    // MARK: - Discovery

    func testBundledRootIsDiscoveredFromTheRealResourceBundle() throws {
        let skill = try bundledSkill()
        XCTAssertEqual(skill.scope, .bundled)
        XCTAssertFalse(skill.skillDescription.isEmpty)
        XCTAssertEqual(skill.sourceURL.lastPathComponent, "SKILL.md")
    }

    func testBundledSkillIsOfferedToTheSkillTool() throws {
        let offered = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
        XCTAssertTrue(offered.contains { $0.name == "plugin-creator" && $0.scope == .bundled })
        let listing = SkillManager.shared.formatSkillsWithinBudget(offered, contextWindowTokens: nil)
        XCTAssertTrue(listing.contains("plugin-creator [Bundled]"), listing)
    }

    func testBundledSkillCannotBeEditedOrDeleted() throws {
        let skill = try bundledSkill()
        XCTAssertFalse(SkillManager.shared.ownsSkill(skill))
        XCTAssertThrowsError(try SkillManager.shared.requireOwnedSkill(skill))
    }

    func testProjectSkillOfTheSameNameOverridesTheBundledOne() throws {
        let bundledRoot = tempDir.appendingPathComponent("bundled", isDirectory: true)
        try writeSkill(named: "zp07-fixture", body: "bundled body", under: bundledRoot)
        let project = tempDir.appendingPathComponent("project", isDirectory: true)
        try writeSkill(
            named: "zp07-fixture", body: "project body",
            under: project.appendingPathComponent(".turbospark/skills", isDirectory: true))

        SkillManager.shared.bundledSkillsDirectory = bundledRoot
        SkillManager.shared.invalidateResolutionCache()

        let withoutProject = SkillManager.shared.resolveEffectiveSkills(projectURL: nil)
            .filter { $0.name == "zp07-fixture" }
        XCTAssertEqual(withoutProject.map(\.scope), [.bundled])

        SkillManager.shared.invalidateResolutionCache()
        let withProject = SkillManager.shared.resolveEffectiveSkills(projectURL: project)
            .filter { $0.name == "zp07-fixture" }
        XCTAssertEqual(withProject.count, 1)
        XCTAssertEqual(withProject.first?.scope, .projectLocal(projectPath: project.path))
        XCTAssertTrue(withProject.first?.content.contains("project body") ?? false)
    }

    func testMissingBundledRootDiscoversNothing() {
        SkillManager.shared.bundledSkillsDirectory = tempDir.appendingPathComponent("absent")
        XCTAssertTrue(SkillManager.shared.discoverBundledSkills().isEmpty)
        SkillManager.shared.bundledSkillsDirectory = nil
        XCTAssertTrue(SkillManager.shared.discoverBundledSkills().isEmpty)
    }

    // MARK: - Skill text

    func testSkillTextCarriesTheContainmentRulesAndTheNamedError() throws {
        // Collapse line wraps so a phrase may span lines.
        let text = try skillText().split(whereSeparator: \.isWhitespace).joined(separator: " ")
        for needle in [
            "PLUGIN_PATH_ESCAPE", "No `..` segment", "No absolute path", "Do not create symlinks",
            "Do not overwrite an existing file", ".claude-plugin/plugin.json",
            ".claude-plugin/marketplace.json", "Add Local Folder", "Manage sources",
            "nothing has been installed",
        ] {
            XCTAssertTrue(text.contains(needle), "skill text lost: \(needle)")
        }
    }

    /// The JSON templates are the contract between the skill and the app:
    /// fill the placeholders the way the agent would and the real parsers
    /// must accept them without a diagnostic.
    func testSkillTemplatesAreAcceptedByTheRealParsers() throws {
        let blocks = try jsonBlocks(in: skillText())
        let manifestTemplate = try XCTUnwrap(blocks.first { $0.contains("\"version\"") })
        let marketplaceTemplate = try XCTUnwrap(blocks.first { $0.contains("\"owner\"") })
        let mcpTemplate = try XCTUnwrap(blocks.first { $0.contains("\"mcpServers\"") })

        let manifest = try PluginManifestParser.parseManifest(
            data: Data(fill(manifestTemplate).utf8), fallbackName: "demo",
            sourceDescription: "template")
        XCTAssertEqual(manifest.name, "demo")
        XCTAssertNil(PluginManifestParser.validatePluginName(manifest.name))
        XCTAssertTrue(manifest.unsupportedNotes.isEmpty, "\(manifest.unsupportedNotes)")

        let marketplace = try PluginManifestParser.parseMarketplace(
            data: Data(fill(marketplaceTemplate).utf8), sourceDescription: "template")
        XCTAssertEqual(marketplace.name, "demo-dev")
        XCTAssertEqual(marketplace.entries.map(\.name), ["demo"])
        XCTAssertEqual(marketplace.entries.first?.sourceValue as? String, "./plugins/demo")

        let mcp = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(fill(mcpTemplate).utf8)) as? [String: Any])
        XCTAssertNotNil((mcp["mcpServers"] as? [String: Any])?["server"])
    }

    /// The skill promises that an escaping manifest path is refused by the
    /// app's own parser too; pin that promise to the parser, not to prose.
    func testParserRefusesEscapingManifestPathsTheSkillForbids() throws {
        let json = #"{"name":"demo","skills":"../outside","agents":["/abs/agents"],"commands":"./commands"}"#
        let manifest = try PluginManifestParser.parseManifest(
            data: Data(json.utf8), fallbackName: "demo", sourceDescription: "t")
        XCTAssertTrue(manifest.skillPaths.isEmpty)
        XCTAssertTrue(manifest.agentPaths.isEmpty)
        XCTAssertFalse(manifest.unsupportedNotes.isEmpty)
    }

    // MARK: - Create Plugin draft

    @MainActor func testDraftNamesTheBundledSkillAndIsAppendedNotSent() throws {
        let skill = try bundledSkill()
        let draft = PluginSettingsPaneView.createPluginDraft
        XCTAssertTrue(draft.contains(skill.name), draft)
        XCTAssertEqual(AppModel.draft(appending: draft, to: ""), draft)
        XCTAssertEqual(AppModel.draft(appending: draft, to: "keep me"), "keep me\n\n" + draft)
    }

    @MainActor func testDraftAndMenuStringsExistInEveryLanguageAndKeepTheSkillName() throws {
        let catalogURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Localization/Localizable.xcstrings")
        let root = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: catalogURL)) as? [String: Any])
        let strings = try XCTUnwrap(root["strings"] as? [String: Any])
        let draftKey = PluginSettingsPaneView.createPluginDraft
        for key in ["Create Plugin...", draftKey] {
            let entry = try XCTUnwrap(strings[key] as? [String: Any], "missing catalog key: \(key)")
            let locs = try XCTUnwrap(entry["localizations"] as? [String: Any])
            let expected = Set(AppLanguage.allCases.filter { $0 != .system }.map(\.rawValue))
            XCTAssertEqual(Set(locs.keys), expected)
            for (lang, raw) in locs {
                let unit = (raw as? [String: Any])?["stringUnit"] as? [String: Any]
                let value = unit?["value"] as? String ?? ""
                XCTAssertFalse(value.isEmpty, "\(lang) \(key)")
                if key == draftKey {
                    // The agent must still be pointed at the skill by its id,
                    // and the four fill-in lines must survive translation.
                    XCTAssertTrue(value.contains("plugin-creator"), lang)
                    XCTAssertEqual(value.filter { $0 == "\n" }.count, 4, lang)
                }
            }
        }
    }

    // MARK: - Helpers

    private func writeSkill(named name: String, body: String, under root: URL) throws {
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let text = "---\nname: \(name)\ndescription: fixture\n---\n\(body)\n"
        try text.write(to: dir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)
    }

    private func jsonBlocks(in text: String) -> [String] {
        var blocks: [String] = []
        var current: [String]?
        for line in text.split(separator: "\n", omittingEmptySubsequences: false) {
            if line.hasPrefix("```json") { current = []; continue }
            if line.hasPrefix("```"), let done = current { blocks.append(done.joined(separator: "\n")); current = nil; continue }
            current?.append(String(line))
        }
        return blocks
    }

    /// Fills the template placeholders the way the agent would.
    private func fill(_ template: String) -> String {
        template
            .replacingOccurrences(of: "<name>", with: "demo")
            .replacingOccurrences(of: "<one line>", with: "A demo")
            .replacingOccurrences(of: "<author or the user>", with: "Ann")
            .replacingOccurrences(of: "<author>", with: "Ann")
            .replacingOccurrences(of: "<server>", with: "server")
            .replacingOccurrences(of: "<executable>", with: "/usr/bin/true")
    }
}
