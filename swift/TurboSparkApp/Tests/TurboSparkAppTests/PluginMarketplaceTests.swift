import XCTest
@testable import TurboSparkApp

/// The plugin marketplace: manifest parsing, directory-source installs into
/// the versioned cache, the v2 ledger, and uninstall cleanup. All offline
/// (directory and relative sources only); git and github sources share
/// `MarketplaceGit` with the two marketplaces that already exercise it.
final class PluginMarketplaceTests: XCTestCase {
    var root: URL!
    var marketplaceManager: PluginMarketplaceManager!
    var ledger: PluginLedgerStore!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-market-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        marketplaceManager = PluginMarketplaceManager(root: root)
        ledger = PluginLedgerStore(root: root)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func makeLocalPlugin(name: String, version: String?) throws -> URL {
        let dir = root.appendingPathComponent("source/\(name)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("skills/greet"), withIntermediateDirectories: true)
        try Data("---\ndescription: Greets\n---\nhello".utf8).write(
            to: dir.appendingPathComponent("skills/greet/SKILL.md"))
        var manifest = "{\"name\": \"\(name)\""
        if let version {
            manifest += ", \"version\": \"\(version)\""
        }
        manifest += "}"
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try Data(manifest.utf8).write(
            to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
        return dir
    }

    private func makeCheckout(entriesJSON: String) throws -> URL {
        let dir = root.appendingPathComponent("checkout", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try Data("""
        {"name": "test-market", "owner": {"name": "Tester"}, "plugins": [\(entriesJSON)]}
        """.utf8).write(
            to: dir.appendingPathComponent(".claude-plugin/marketplace.json"))
        return dir
    }

    // MARK: - Manifest parsing

    func testMarketplaceManifestParses() throws {
        let dir = try makeCheckout(entriesJSON: """
        {"name": "greet", "source": {"type": "directory", "path": "/tmp/x"}, "description": "A greeter"}
        """)
        let data = try Data(contentsOf: dir.appendingPathComponent(".claude-plugin/marketplace.json"))
        let manifest = try PluginManifestParser.parseMarketplace(
            data: data, sourceDescription: "test")
        XCTAssertEqual(manifest.name, "test-market")
        XCTAssertEqual(manifest.ownerName, "Tester")
        XCTAssertEqual(manifest.entries.count, 1)
        XCTAssertEqual(manifest.entries.first?.name, "greet")
        XCTAssertEqual(manifest.entries.first?.strict, true, "strict is the default")
        XCTAssertEqual(manifest.entries.first?.descriptionText, "A greeter")
    }

    func testMarketplaceEntryWithoutANameIsStrippedNotFatal() throws {
        let dir = try makeCheckout(entriesJSON: """
        {"description": "no name here"}
        """)
        let data = try Data(contentsOf: dir.appendingPathComponent(".claude-plugin/marketplace.json"))
        let manifest = try PluginManifestParser.parseMarketplace(data: data, sourceDescription: "t")
        XCTAssertEqual(manifest.entries.count, 0, "one bad entry must not take the marketplace down")
    }

    // MARK: - Install and uninstall

    func testDirectorySourceInstallLandsInVersionedCacheWithLedgerRecord() async throws {
        let pluginDir = try makeLocalPlugin(name: "greet", version: "2.5.0")
        let entry = PluginManifestParser.MarketplaceEntry(
            name: "greet",
            sourceValue: ["type": "directory", "path": pluginDir.path],
            strict: true, raw: ["name": "greet"])

        let outcome = try await marketplaceManager.install(
            entry: entry, marketplaceName: "test-market",
            checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "user")

        XCTAssertEqual(outcome.pluginID, "greet@test-market")
        XCTAssertEqual(outcome.version, "2.5.0", "the plugin manifest version wins")
        XCTAssertTrue(
            outcome.installPath.hasPrefix(
                root.appendingPathComponent("cache/test-market/greet/2.5.0").path))
        XCTAssertTrue(
            FileManager.default.fileExists(
                atPath: outcome.installPath + "/skills/greet/SKILL.md"),
            "the plugin contents were copied, not referenced")

        let records = ledger.load().plugins["greet@test-market"] ?? []
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first?.scope, "user")
        XCTAssertEqual(records.first?.version, "2.5.0")
        XCTAssertEqual(records.first?.installPath, outcome.installPath)
    }

    func testEntryVersionAndUnknownFillTheVersionGap() async throws {
        let noVersion = try makeLocalPlugin(name: "noversion", version: nil)
        let outcome = try await marketplaceManager.install(
            entry: PluginManifestParser.MarketplaceEntry(
                name: "noversion",
                sourceValue: ["type": "directory", "path": noVersion.path],
                strict: true, raw: ["name": "noversion"]),
            marketplaceName: "test-market", checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "user")
        XCTAssertEqual(outcome.version, "unknown")

        let versioned = try makeLocalPlugin(name: "entryv", version: nil)
        let outcome2 = try await marketplaceManager.install(
            entry: PluginManifestParser.MarketplaceEntry(
                name: "entryv",
                sourceValue: ["type": "directory", "path": versioned.path],
                strict: true, raw: ["name": "entryv", "version": "7.7.7"]),
            marketplaceName: "test-market", checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "user")
        XCTAssertEqual(outcome2.version, "7.7.7", "the entry version is the second source")
    }

    func testRelativeSourceInstallsFromTheCheckout() async throws {
        let checkout = try makeCheckout(entriesJSON: "")
        // The plugin lives INSIDE the checkout; the entry source is relative.
        try FileManager.default.createDirectory(
            at: checkout.appendingPathComponent("./plugins/greet/.claude-plugin"),
            withIntermediateDirectories: true)
        try Data(#"{"name": "greet", "version": "1.0.0"}"#.utf8).write(
            to: checkout.appendingPathComponent("./plugins/greet/.claude-plugin/plugin.json"))
        try FileManager.default.createDirectory(
            at: checkout.appendingPathComponent("./plugins/greet/skills/x"),
            withIntermediateDirectories: true)

        let outcome = try await marketplaceManager.install(
            entry: PluginManifestParser.MarketplaceEntry(
                name: "greet", sourceValue: "./plugins/greet",
                strict: true, raw: ["name": "greet"]),
            marketplaceName: "test-market",
            checkoutDirectory: checkout, marketplaceSource: .url(url: "https://example.com/market.json", headers: nil), scope: "user")
        XCTAssertEqual(outcome.version, "1.0.0")
        XCTAssertTrue(FileManager.default.fileExists(atPath: outcome.installPath))
    }

    func testRelativeSourceCannotEscapeCheckoutThroughTraversalOrPluginRoot() async throws {
        let checkout = try makeCheckout(entriesJSON: "")
        let outside = root.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)

        for (relative, pluginRoot) in [("./../outside", nil), ("./", "../outside")] {
            if let pluginRoot {
                try Data("""
                {"name":"test-market","metadata":{"pluginRoot":"\(pluginRoot)"},"plugins":[]}
                """.utf8).write(
                    to: checkout.appendingPathComponent(".claude-plugin/marketplace.json"))
            }
            do {
                _ = try await marketplaceManager.install(
                    entry: PluginManifestParser.MarketplaceEntry(
                        name: "escape", sourceValue: relative,
                        strict: true, raw: ["name": "escape"]),
                    marketplaceName: "test-market", checkoutDirectory: checkout,
                    marketplaceSource: .url(url: "https://example.com/market.json", headers: nil),
                    scope: "user")
                XCTFail("A marketplace source outside its checkout must be rejected")
            } catch let error as PluginLoadError {
                XCTAssertTrue(error.reason.contains("inside the marketplace checkout"))
            }
        }
    }

    func testConfinedSourceRejectsSymlinkEscape() throws {
        let checkout = try makeCheckout(entriesJSON: "")
        let outside = root.appendingPathComponent("outside", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try FileManager.default.createSymbolicLink(
            at: checkout.appendingPathComponent("linked"), withDestinationURL: outside)

        XCTAssertThrowsError(try PluginMarketplaceManager.confinedSource(
            checkout.appendingPathComponent("linked"), within: checkout, pluginName: "escape"))
    }

    func testRemoteMarketplaceCannotInstallLocalDirectorySource() async throws {
        let pluginDir = try makeLocalPlugin(name: "escape", version: "1")
        do {
            _ = try await marketplaceManager.install(
                entry: PluginManifestParser.MarketplaceEntry(
                    name: "escape",
                    sourceValue: ["type": "directory", "path": pluginDir.path],
                    strict: true, raw: ["name": "escape"]),
                marketplaceName: "remote", checkoutDirectory: nil,
                marketplaceSource: .url(url: "https://example.com/market.json", headers: nil),
                scope: "user")
            XCTFail("A remote marketplace must not select an arbitrary local directory")
        } catch let error as PluginLoadError {
            XCTAssertTrue(error.reason.contains("cannot install a local directory source"))
        }
    }

    func testUninstallRemovesTheCacheDirectoryOnlyAfterTheLastScope() async throws {
        let pluginDir = try makeLocalPlugin(name: "greet", version: "1.0.0")
        let entry = PluginManifestParser.MarketplaceEntry(
            name: "greet",
            sourceValue: ["type": "directory", "path": pluginDir.path],
            strict: true, raw: ["name": "greet"])

        _ = try await marketplaceManager.install(
            entry: entry, marketplaceName: "test-market", checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "user")
        let installPath = (ledger.load().plugins["greet@test-market"] ?? []).first!.installPath

        // The project scope lands as a second record.
        _ = try await marketplaceManager.install(
            entry: entry, marketplaceName: "test-market",
            checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "project",
            projectRootURL: URL(fileURLWithPath: "/tmp/proj"))

        try marketplaceManager.uninstall(pluginID: "greet@test-market", scope: "user")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: installPath),
            "the project install still references the cache directory")

        try marketplaceManager.uninstall(
            pluginID: "greet@test-market", scope: "project",
            projectRootURL: URL(fileURLWithPath: "/tmp/proj"))
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: installPath),
            "the last uninstall removes the version cache directory")
        XCTAssertNil(ledger.load().plugins["greet@test-market"])
    }

    func testUninstallRefusesToDeleteOutsideTheCacheRoot() async throws {
        let outside = root.appendingPathComponent("precious", isDirectory: true)
        try FileManager.default.createDirectory(at: outside, withIntermediateDirectories: true)
        try Data("keep me".utf8).write(to: outside.appendingPathComponent("data.txt"))
        // A hand-edited ledger row pointing OUTSIDE the cache root.
        ledger.upsertRecord(
            InstalledPluginRecord(
                scope: "user", installPath: outside.path, version: "1.0.0"),
            for: "sneaky@evil")

        try marketplaceManager.uninstall(pluginID: "sneaky@evil", scope: "user")
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: outside.appendingPathComponent("data.txt").path),
            "a ledger row pointing outside the cache root is never deleted")
        XCTAssertNil(
            ledger.load().plugins["sneaky@evil"],
            "the ledger record itself is still removed")
    }

    // MARK: - Paths

    func testSanitizedComponentsContainNoMetacharacters() {
        XCTAssertEqual(PluginMarketplaceManager.sanitizedComponent("../../etc"), "------etc")
        XCTAssertEqual(PluginMarketplaceManager.sanitizedVersion("1.2.3+build"), "1.2.3-build")
    }

    // MARK: - Ledger store

    func testLedgerUpsertReplacesSameScopeAndProject() throws {
        ledger.upsertRecord(
            InstalledPluginRecord(scope: "user", installPath: "/a", version: "1"), for: "p@m")
        ledger.upsertRecord(
            InstalledPluginRecord(scope: "user", installPath: "/b", version: "2"), for: "p@m")
        let records = ledger.load().plugins["p@m"] ?? []
        XCTAssertEqual(records.count, 1)
        XCTAssertEqual(records.first?.installPath, "/b")
    }
}

extension PluginMarketplaceTests {
    func testLedgerWriteFailureRollsBackNewAndReplacementCaches() async throws {
        let source = try makeLocalPlugin(name: "rollback", version: "1")
        let entry = PluginManifestParser.MarketplaceEntry(name: "rollback",
            sourceValue: ["type": "directory", "path": source.path], strict: true, raw: ["name": "rollback"])
        // A directory at the ledger path makes the atomic file write fail
        // without relying on host identity or permission escalation.
        try FileManager.default.createDirectory(at: ledger.ledgerURL, withIntermediateDirectories: true)
        let cache = marketplaceManager.installCacheDirectory.appendingPathComponent("test/rollback/1")
        for replacing in [false, true] {
            if replacing {
                try FileManager.default.createDirectory(at: cache, withIntermediateDirectories: true)
                try Data("previous install".utf8).write(to: cache.appendingPathComponent("marker"))
            }
            do {
                _ = try await marketplaceManager.install(entry: entry, marketplaceName: "test",
                    checkoutDirectory: nil, marketplaceSource: .directory(path: root.path), scope: "project", projectRootURL: root.appendingPathComponent("project"))
                XCTFail("An installation without a persisted scope must fail")
            } catch {
                XCTAssertEqual((error as NSError).domain, NSCocoaErrorDomain)
            }
            if replacing {
                XCTAssertEqual(try Data(contentsOf: cache.appendingPathComponent("marker")), Data("previous install".utf8))
                XCTAssertFalse(FileManager.default.fileExists(atPath: cache.appendingPathComponent("skills").path))
            } else {
                XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path))
            }
        }
    }
}
