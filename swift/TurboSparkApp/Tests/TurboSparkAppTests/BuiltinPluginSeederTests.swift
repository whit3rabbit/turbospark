import XCTest
@testable import TurboSparkApp

/// ZP-05 builtin seeding over a fixture bundle and a temp plugin root: the
/// real installer, ledger and state file, no network, nothing under `~`.
final class BuiltinPluginSeederTests: XCTestCase {
    var root: URL!
    var bundle: URL!
    var manager: PluginMarketplaceManager!

    override func setUpWithError() throws {
        let base = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-seed-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("plugins", isDirectory: true)
        bundle = base.appendingPathComponent("BuiltinPlugins", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: bundle, withIntermediateDirectories: true)
        manager = PluginMarketplaceManager(root: root)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
    }

    private func seeder(lockTimeout: TimeInterval = 10) -> BuiltinPluginSeeder {
        BuiltinPluginSeeder(bundleDirectory: bundle, manager: manager, lockTimeout: lockTimeout)
    }

    /// Writes the bundled marketplace: one folder per (name, version).
    private func writeBundle(_ plugins: [(String, String)], sources: [String: String] = [:]) throws {
        var entries: [String] = []
        for (name, version) in plugins {
            let dir = bundle.appendingPathComponent(name, isDirectory: true)
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
            try Data("{\"name\": \"\(name)\", \"version\": \"\(version)\"}".utf8).write(
                to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent("skills/hi"), withIntermediateDirectories: true)
            try Data("---\ndescription: Hi\n---\nhello".utf8).write(
                to: dir.appendingPathComponent("skills/hi/SKILL.md"))
            entries.append("{\"name\": \"\(name)\", \"source\": \"\(sources[name] ?? "./\(name)")\"}")
        }
        try FileManager.default.createDirectory(
            at: bundle.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try Data("""
        {"name": "turbospark-plugins-builtin", "owner": {"name": "TurboSpark"}, "plugins": [\(entries.joined(separator: ","))]}
        """.utf8).write(to: bundle.appendingPathComponent(".claude-plugin/marketplace.json"))
    }

    private func userRecords(_ name: String) -> [InstalledPluginRecord] {
        (manager.ledger.load().plugins["\(name)@turbospark-plugins-builtin"] ?? [])
            .filter { $0.scope == "user" }
    }

    func testSeedInstallsThroughTheLedgerAndIsIdempotent() async throws {
        try writeBundle([("guide", "1.0.0")])
        let first = try await seeder().seed()
        XCTAssertEqual(first.seeded, ["guide"])
        let record = try XCTUnwrap(userRecords("guide").first)
        XCTAssertEqual(record.version, "1.0.0")
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: record.installPath + "/skills/hi/SKILL.md"))

        let second = try await seeder().seed()
        XCTAssertEqual(second.seeded, [], "an installed, current builtin is not copied again")
        XCTAssertEqual(userRecords("guide").first?.installedAt, record.installedAt)
    }

    func testSuppressionSurvivesRestartAndRestoreClearsIt() async throws {
        try writeBundle([("guide", "1.0.0")])
        _ = try await seeder().seed()
        try manager.uninstall(pluginID: "guide@turbospark-plugins-builtin", scope: "user")
        try seeder().recordUninstalled(pluginName: "guide")

        let afterRestart = try await seeder().seed()
        XCTAssertEqual(afterRestart.seeded, [])
        XCTAssertEqual(afterRestart.suppressed, ["guide"])
        XCTAssertTrue(userRecords("guide").isEmpty, "a restart must not resurrect it")

        // Restore: the row's Install path installs and then clears the record.
        let entry = try XCTUnwrap(seeder().entries().first)
        let outcome = try await manager.install(
            entry: entry, marketplaceName: BuiltinPluginSeeder.marketplaceName,
            checkoutDirectory: bundle, marketplaceSource: .directory(path: bundle.path),
            scope: "user")
        try seeder().recordInstalled(pluginName: "guide", version: outcome.version)
        XCTAssertFalse(seeder().isSuppressed("guide"))
        let again = try await seeder().seed()
        XCTAssertEqual(again.seeded, [], "restored and current: nothing to do")
        XCTAssertEqual(userRecords("guide").count, 1)
    }

    func testRemovalOutsideTheUninstallPathBecomesSuppression() async throws {
        try writeBundle([("guide", "1.0.0")])
        _ = try await seeder().seed()
        // Ledger row gone without the app's uninstall hook running.
        try manager.uninstall(pluginID: "guide@turbospark-plugins-builtin", scope: "user")
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, [])
        XCTAssertEqual(seeder().record(for: "guide")?.state, .suppressed)
    }

    func testBundleVersionBumpReachesAnInstalledBuiltin() async throws {
        try writeBundle([("guide", "1.0.0")])
        _ = try await seeder().seed()
        try writeBundle([("guide", "1.1.0")])
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, ["guide"])
        XCTAssertEqual(userRecords("guide").first?.version, "1.1.0")
        XCTAssertEqual(seeder().record(for: "guide")?.version, "1.1.0")
    }

    func testInstalledButUnrecordedIsAdoptedNotReinstalled() async throws {
        try writeBundle([("guide", "1.0.0")])
        let entry = try XCTUnwrap(seeder().entries().first)
        _ = try await manager.install(
            entry: entry, marketplaceName: BuiltinPluginSeeder.marketplaceName,
            checkoutDirectory: bundle, marketplaceSource: .directory(path: bundle.path),
            scope: "user")
        let stamp = userRecords("guide").first?.installedAt
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, [])
        XCTAssertEqual(seeder().record(for: "guide")?.state, .seeded)
        XCTAssertEqual(userRecords("guide").first?.installedAt, stamp)
    }

    func testCorruptStateFailsClosed() async throws {
        try writeBundle([("guide", "1.0.0")])
        try Data("not json".utf8).write(to: seeder().stateURL)
        do {
            _ = try await seeder().seed()
            XCTFail("expected corruptState")
        } catch {
            XCTAssertEqual(error as? BuiltinPluginSeeder.SeedError, .corruptState)
        }
        XCTAssertTrue(userRecords("guide").isEmpty)
        XCTAssertEqual(
            try String(contentsOf: seeder().stateURL, encoding: .utf8), "not json",
            "the unreadable file is left for the user, not overwritten")
    }

    func testSeedLockSerializesSeeders() async throws {
        try writeBundle([("guide", "1.0.0")])
        let held = try seeder().acquireLock()
        do {
            _ = try await seeder(lockTimeout: 0.2).seed()
            XCTFail("a second seeder must wait on the held lock")
        } catch {
            XCTAssertEqual(error as? BuiltinPluginSeeder.SeedError, .lockTimeout)
        }
        XCTAssertTrue(userRecords("guide").isEmpty)
        held.release()
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, ["guide"])
    }

    // MARK: - Source refusals

    func testSymlinkInsideAPluginIsRefused() async throws {
        try writeBundle([("guide", "1.0.0")])
        try FileManager.default.createSymbolicLink(
            atPath: bundle.appendingPathComponent("guide/leak").path, withDestinationPath: "/etc")
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, [])
        XCTAssertNotNil(report.refused["guide"])
        XCTAssertTrue(userRecords("guide").isEmpty)
        XCTAssertNil(seeder().record(for: "guide"))
    }

    func testEscapingAndNonRelativeSourcesAreRefused() async throws {
        try writeBundle(
            [("up", "1.0.0"), ("abs", "1.0.0"), ("root", "1.0.0")],
            sources: ["up": "./../outside", "abs": "/etc", "root": "./"])
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, [])
        XCTAssertEqual(Set(report.refused.keys), ["up", "abs", "root"])
    }

    func testOversizedPluginIsRefusedAndOthersStillSeed() async throws {
        try writeBundle([("big", "1.0.0"), ("ok", "1.0.0")])
        let dir = bundle.appendingPathComponent("big/many", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for index in 0...BuiltinPluginSeeder.maxFilesPerPlugin {
            FileManager.default.createFile(atPath: dir.appendingPathComponent("f\(index)").path, contents: Data())
        }
        let report = try await seeder().seed()
        XCTAssertEqual(report.seeded, ["ok"])
        XCTAssertNotNil(report.refused["big"])
    }

    func testNoBundleMeansNoBuiltinsAndNoSideEffects() async throws {
        let empty = BuiltinPluginSeeder(
            bundleDirectory: bundle.appendingPathComponent("missing"), manager: manager)
        XCTAssertFalse(empty.hasBuiltins)
        let report = try await empty.seed()
        XCTAssertEqual(report, BuiltinPluginSeeder.Report())
        XCTAssertFalse(FileManager.default.fileExists(atPath: empty.stateURL.path))
        let none = BuiltinPluginSeeder(bundleDirectory: nil, manager: manager)
        XCTAssertFalse(none.hasBuiltins)
    }

    @MainActor func testBuiltinPluginNameParsing() {
        XCTAssertEqual(AppModel.builtinPluginName(of: "guide@turbospark-plugins-builtin"), "guide")
        XCTAssertNil(AppModel.builtinPluginName(of: "guide@other"))
        XCTAssertNil(AppModel.builtinPluginName(of: "@turbospark-plugins-builtin"))
    }
}
