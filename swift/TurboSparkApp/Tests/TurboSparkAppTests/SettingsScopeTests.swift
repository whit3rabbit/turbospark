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
        let owned = root.appendingPathComponent("project/.turbospark/skills/alias")
        try FileManager.default.createDirectory(at: owned, withIntermediateDirectories: true)
        let link = owned.appendingPathComponent("SKILL.md")
        try FileManager.default.createSymbolicLink(at: link, withDestinationURL: source)
        let alias = AppSkill(manifest: skill.manifest, content: "replacement", sourceURL: link,
            skillDirectoryURL: owned, scope: .projectLocal(projectPath: root.appendingPathComponent("project").path),
            agentOrigin: .custom, isEnabled: true)
        XCTAssertThrowsError(try manager.saveSkill(alias))
        XCTAssertThrowsError(try manager.deleteSkill(alias))
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

extension SettingsScopeTests {
    @MainActor
    func testMcpOverrideActionIsIsolatedAndInheritsAgain() {
        let model = AppModel()
        let server = McpServerConfig(name: "scope-test", transport: .stdio(command: "/usr/bin/true", args: [], env: [:]))
        let one = AppProject(name: "One", rootDirectoryPath: "/tmp/mcp-one")
        let two = AppProject(name: "Two", rootDirectoryPath: "/tmp/mcp-two")
        model.projects = [one, two]
        model.setMcpOverride(serverID: server.id, enabled: false, projectID: one.id)
        XCTAssertFalse(AppToolCatalogMcp.resolvedServers(global: [server], project: model.projects[0])[0].isEnabled)
        XCTAssertTrue(AppToolCatalogMcp.resolvedServers(global: [server], project: model.projects[1])[0].isEnabled)
        model.setMcpOverride(serverID: server.id, enabled: nil, projectID: one.id)
        XCTAssertTrue(AppToolCatalogMcp.resolvedServers(global: [server], project: model.projects[0])[0].isEnabled)
    }

    func testFinalUninstallRemovesAllCachedVersionsButKeepsAnotherProfile() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = PluginMarketplaceManager(root: root.appendingPathComponent("one"))
        let second = PluginMarketplaceManager(root: root.appendingPathComponent("two"))
        for manager in [first, second] {
            let family = manager.installCacheDirectory.appendingPathComponent("test/sample")
            for version in ["1", "2"] {
                try FileManager.default.createDirectory(at: family.appendingPathComponent(version), withIntermediateDirectories: true)
            }
            try PluginLedgerStore(root: manager.root).saveChecked(InstalledPluginLedger(plugins: ["sample@test": [
                InstalledPluginRecord(scope: "user", installPath: family.appendingPathComponent("2").path, version: "2")
            ]]))
        }
        try first.uninstall(pluginID: "sample@test", scope: "user")
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.installCacheDirectory.appendingPathComponent("test/sample").path))
        XCTAssertTrue(FileManager.default.fileExists(atPath: second.installCacheDirectory.appendingPathComponent("test/sample/2").path))
        XCTAssertNotNil(PluginLedgerStore(root: second.root).load().plugins["sample@test"])
    }

    func testSourceRemovalReportsUnreadableArchiveWithoutDestroyingIt() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let manager = PluginMarketplaceManager(root: root)
        try FileManager.default.createDirectory(at: manager.knownMarketplacesURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        let original = Data("invalid archive".utf8)
        try original.write(to: manager.knownMarketplacesURL)
        XCTAssertThrowsError(try manager.removeKnownMarketplace(name: "test"))
        XCTAssertEqual(try Data(contentsOf: manager.knownMarketplacesURL), original)
    }
}

extension SettingsScopeTests {
    @MainActor
    func testFailedUninstallActionRetainsRecordCacheAndPreference() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer {
            try? fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: root.path)
            try? fm.removeItem(at: root)
        }
        let manager = PluginMarketplaceManager(root: root)
        let cache = manager.installCacheDirectory.appendingPathComponent("test/sample/1")
        try fm.createDirectory(at: cache, withIntermediateDirectories: true)
        let ledger = PluginLedgerStore(root: root)
        try ledger.saveChecked(InstalledPluginLedger(plugins: ["sample@test": [
            InstalledPluginRecord(scope: "user", installPath: cache.path, version: "1")
        ]]))
        let before = ledger.load()
        let model = AppModel()
        model.pluginEnableState["sample@test"] = false
        // The ledger remains readable, but quarantine creation cannot succeed.
        try fm.setAttributes([.posixPermissions: 0o555], ofItemAtPath: root.path)
        XCTAssertFalse(model.uninstallPluginID("sample@test", scope: .user, manager: manager))
        XCTAssertEqual(ledger.load(), before)
        XCTAssertTrue(fm.fileExists(atPath: cache.path))
        XCTAssertEqual(model.pluginEnableState["sample@test"], false)
    }
}

extension SettingsScopeTests {
    func testSkillInstallFailurePreservesExistingProjectCopyAndRejectsTraversal() async throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? fm.removeItem(at: root) }
        let project = root.appendingPathComponent("project")
        let destination = project.appendingPathComponent(".turbospark/skills/deploy")
        let source = root.appendingPathComponent("source")
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)
        try fm.createDirectory(at: source, withIntermediateDirectories: true)
        let old = Data("---\nname: deploy\ndescription: Original\n---\nKeep me".utf8)
        try old.write(to: destination.appendingPathComponent("SKILL.md"))
        try Data("partial download".utf8).write(to: source.appendingPathComponent("README.md"))
        let manager = SkillMarketplaceManager()
        for (name, code) in [("deploy", 5), ("../escape", 7)] {
            let entry = MarketplaceSkillEntry(name: name, description: "test", source: .directory(path: source.path))
            do {
                _ = try await manager.installSkill(entry: entry, targetScope: .projectLocal(projectPath: project.path))
                XCTFail("Invalid installation must fail")
            } catch {
                XCTAssertEqual((error as NSError).domain, "SkillMarketplace")
                XCTAssertEqual((error as NSError).code, code)
            }
        }
        XCTAssertEqual(try Data(contentsOf: destination.appendingPathComponent("SKILL.md")), old)
        XCTAssertFalse(fm.fileExists(atPath: destination.appendingPathComponent("README.md").path))
        XCTAssertFalse(fm.fileExists(atPath: project.appendingPathComponent(".turbospark/escape").path))
    }
}

extension SettingsScopeTests {
    func testDeletingOwnedSkillRetainsOtherInstallationRecords() throws {
        let fm = FileManager.default
        let root = fm.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let manager = SkillMarketplaceManager()
        let previousLedger = try? Data(contentsOf: manager.installedLedgerURL)
        defer {
            try? fm.removeItem(at: root)
            if let previousLedger { try? previousLedger.write(to: manager.installedLedgerURL) }
            else { try? fm.removeItem(at: manager.installedLedgerURL) }
        }
        let project = root.appendingPathComponent("one")
        let folder = project.appendingPathComponent(".turbospark/skills/shared")
        try fm.createDirectory(at: folder, withIntermediateDirectories: true)
        let file = folder.appendingPathComponent("SKILL.md")
        try Data("owned".utf8).write(to: file)
        let records = [InstalledSkillRecord(skillName: "shared", scope: "project", projectPath: project.path, installPath: folder.path),
            InstalledSkillRecord(skillName: "shared", scope: "project", projectPath: root.appendingPathComponent("two").path, installPath: root.appendingPathComponent("two/.turbospark/skills/shared").path),
            InstalledSkillRecord(skillName: "shared", scope: "user", installPath: root.appendingPathComponent("user/shared").path)]
        try fm.createDirectory(at: manager.installedLedgerURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try JSONEncoder().encode(["shared": records]).write(to: manager.installedLedgerURL)
        let skill = AppSkill(manifest: SkillManifest(name: "shared", description: "test"), content: "owned",
            sourceURL: file, skillDirectoryURL: folder, scope: .projectLocal(projectPath: project.path), agentOrigin: .custom, isEnabled: true)
        try SkillManager().deleteSkill(skill)
        let remaining = try JSONDecoder().decode([String: [InstalledSkillRecord]].self, from: Data(contentsOf: manager.installedLedgerURL))
        XCTAssertEqual(remaining["shared"]?.map(\.installPath), Array(records.dropFirst()).map(\.installPath))
        XCTAssertFalse(fm.fileExists(atPath: folder.path))
    }
}
