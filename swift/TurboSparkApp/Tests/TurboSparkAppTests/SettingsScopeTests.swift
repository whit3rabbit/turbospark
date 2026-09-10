import XCTest
@testable import TurboSparkApp

final class SettingsScopeTests: XCTestCase {
    func testProjectSkillKeysIsolateIdenticalNamesAndPreserveLegacyDefaults() {
        let saved = DisabledItemStore.names(for: .skills)
        defer { DisabledItemStore.setNames(saved, for: .skills) }
        DisabledItemStore.setNames(["project:deploy"], for: .skills)
        let manager = SkillManager()
        let first = SkillScope.projectLocal(projectPath: "/tmp/settings-project-one")
        let second = SkillScope.projectLocal(projectPath: "/tmp/settings-project-two")
        XCTAssertTrue(manager.isSkillDisabled(scope: first, name: "deploy"))
        manager.setSkillEnabled(true, scope: first, name: "deploy")
        XCTAssertFalse(manager.isSkillDisabled(scope: first, name: "deploy"))
        XCTAssertTrue(manager.isSkillDisabled(scope: second, name: "deploy"))
        manager.setSkillEnabled(true, scope: second, name: "deploy")
        manager.setSkillEnabled(false, scope: first, name: "deploy")
        XCTAssertFalse(manager.isSkillDisabled(scope: second, name: "deploy"))
        XCTAssertTrue(manager.isSkillDisabled(scope: first, name: "deploy"))
    }

    @MainActor
    func testPluginPreferenceTargetsCapturedProjectAndPreservesUserDefault() {
        let model = AppModel()
        let first = AppProject(name: "One", rootDirectoryPath: "/tmp/settings-one")
        let second = AppProject(name: "Two", rootDirectoryPath: "/tmp/settings-two")
        model.projects = [first, second]
        model.pluginEnableState["sample@test"] = true
        model.setPluginPreference(id: "sample@test", enabled: false, scope: .project(first))
        XCTAssertEqual(model.projects[0].enabledPlugins["sample@test"], false)
        XCTAssertNil(model.projects[1].enabledPlugins["sample@test"])
        XCTAssertEqual(model.pluginEnableState["sample@test"], true)
        model.setPluginPreference(id: "sample@test", enabled: nil, scope: .project(first))
        XCTAssertNil(model.projects[0].enabledPlugins["sample@test"])
    }

    @MainActor
    func testUninstallActionRemovesOnlyRequestedScope() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = PluginMarketplaceManager(root: root)
        let ledger = PluginLedgerStore(root: root)
        let cache = manager.installCacheDirectory.appendingPathComponent("test/sample/1")
        try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
        let first = AppProject(name: "One", rootDirectoryPath: "/tmp/settings-one")
        let second = AppProject(name: "Two", rootDirectoryPath: "/tmp/settings-two")
        let records = [InstalledPluginRecord(scope: "user", installPath: cache.path, version: "1"),
            InstalledPluginRecord(scope: "project", projectPath: first.rootDirectoryPath, installPath: cache.path, version: "1"),
            InstalledPluginRecord(scope: "project", projectPath: second.rootDirectoryPath, installPath: cache.path, version: "1")]
        try ledger.saveChecked(InstalledPluginLedger(plugins: ["sample@test": records]))
        let persisted = ledger.load().plugins["sample@test"]!
        let model = AppModel()
        model.projects = [first, second]
        model.pluginEnableState["sample@test"] = false
        XCTAssertTrue(model.uninstallPluginID("sample@test", scope: .project(first), manager: manager))
        XCTAssertEqual(ledger.load().plugins["sample@test"], [persisted[0], persisted[2]])
        XCTAssertTrue(FileManager.default.fileExists(atPath: cache.path))
        XCTAssertEqual(model.pluginEnableState["sample@test"], false)
    }

    func testExternalSkillCannotBeSavedOrDeleted() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let source = root.appendingPathComponent("SKILL.md")
        try "original".write(to: source, atomically: true, encoding: .utf8)
        let skill = AppSkill(manifest: SkillManifest(name: "external", description: "test"), content: "replacement",
            sourceURL: source, skillDirectoryURL: root, scope: .userGlobal, agentOrigin: .claude, isEnabled: true)
        let manager = SkillManager()
        XCTAssertThrowsError(try manager.saveSkill(skill))
        XCTAssertThrowsError(try manager.deleteSkill(skill))
        XCTAssertEqual(try String(contentsOf: source), "original")
    }
}

extension SettingsScopeTests {
    func testMarketplaceInheritanceHidingAndProfileSeparation() throws {
        let user: [String: MarketplaceSource] = ["shared": .directory(path: "/tmp/catalog")]
        var first = ProjectMarketplaces()
        first.sources["plugins"] = ["local": .directory(path: "/tmp/local-catalog")]
        first.hidden["plugins"] = ["shared"]
        let second = ProjectMarketplaces()
        XCTAssertEqual(Set(first.resolved(user, kind: .plugins).keys), ["local"])
        XCTAssertEqual(second.resolved(user, kind: .plugins), user)
        XCTAssertEqual(first.resolved(user, kind: .mcp), user)
        XCTAssertEqual(first.resolved([:], kind: .plugins).keys.sorted(), ["local"])
        XCTAssertEqual(try JSONDecoder().decode(ProjectMarketplaces.self, from: JSONEncoder().encode(first)), first)
    }

    func testSearchCatalogHasUniqueTargetsAndSpecificControls() {
        let entries = SettingsControlCatalog.entries
        XCTAssertEqual(Set(entries.map(\.id)).count, entries.count)
        XCTAssertTrue(entries.contains { $0.pane == .engine && $0.title == "Temperature" })
        XCTAssertTrue(entries.contains { $0.pane == .appearance && $0.title == "UI font" })
        XCTAssertTrue(entries.contains { $0.pane == .appearance && $0.title == "Theme gallery" })
    }
}

extension SettingsScopeTests {
    func testProjectOnlyPluginNeverContributesOutsideItsInstallation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let cache = root.appendingPathComponent("cache/test/sample/1")
        try FileManager.default.createDirectory(at: cache.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try Data("{\"name\":\"sample\",\"version\":\"1\"}".utf8)
            .write(to: cache.appendingPathComponent(".claude-plugin/plugin.json"))
        let ledger = PluginLedgerStore(root: root)
        try ledger.saveChecked(InstalledPluginLedger(plugins: ["sample@test": [
            InstalledPluginRecord(scope: "project", projectPath: "/tmp/project-one", installPath: cache.path, version: "1")]]))
        let manager = PluginManager(turboSparkRoot: root, claudeRoot: root.appendingPathComponent("absent"),
            userEnableProvider: { [:] }, projectEnableProvider: { _ in [:] }, claudeEnableProvider: { [:] })
        XCTAssertTrue(manager.resolve(projectURL: nil).plugins.isEmpty)
        XCTAssertTrue(manager.resolve(projectURL: URL(fileURLWithPath: "/tmp/project-two")).plugins.isEmpty)
        XCTAssertEqual(manager.resolve(projectURL: URL(fileURLWithPath: "/tmp/project-one")).plugins.map(\.id), ["sample@test"])
    }
}
