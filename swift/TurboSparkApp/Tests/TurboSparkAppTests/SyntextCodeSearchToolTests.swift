import XCTest
@testable import TurboSparkApp

final class SyntextCodeSearchToolTests: XCTestCase {
    private func createTestWorkspace() throws -> (AppProject, URL, URL) {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("syntext_test_\(UUID().uuidString)", isDirectory: true)
        let indexDir = FileManager.default.temporaryDirectory
            .appendingPathComponent("syntext_idx_\(UUID().uuidString).syntext", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

        let swiftFile = root.appendingPathComponent("Main.swift")
        let rustFile = root.appendingPathComponent("lib.rs")
        let docFile = root.appendingPathComponent("Notes.txt")

        try "func calculateTax(amount: Double) -> Double {\n    return amount * 0.15\n}\n"
            .write(to: swiftFile, atomically: true, encoding: .utf8)
        try "pub fn process_order(id: u64) -> bool {\n    println!(\"processing\");\n    true\n}\n"
            .write(to: rustFile, atomically: true, encoding: .utf8)
        try "Project overview notes and documentation.\n"
            .write(to: docFile, atomically: true, encoding: .utf8)

        let project = AppProject(
            name: "Syntext Test Project",
            rootDirectoryPath: root.path,
            syntextIndexEnabled: true
        )
        return (project, root, indexDir)
    }

    func testSyntextToolBuildAndSearch() async throws {
        let (_, root, indexDir) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: indexDir)
        }

        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        let isIndexedInitial = await tool.isIndexed
        XCTAssertFalse(isIndexedInitial)

        let stats = try await tool.buildIndex()
        XCTAssertGreaterThan(stats.totalDocuments, 0)
        let isIndexedAfter = await tool.isIndexed
        XCTAssertTrue(isIndexedAfter)

        let searchResult = try await tool.grep(query: "calculateTax")
        XCTAssertTrue(searchResult.contains("Main.swift"), "Expected Main.swift in search results")
        XCTAssertTrue(searchResult.contains("calculateTax"), "Expected calculateTax match line")

        let swiftFiltered = try await tool.grep(query: "process_order", fileTypes: ["swift"])
        XCTAssertTrue(swiftFiltered.contains("No matches found"), "process_order should not be in swift files")

        let rustSearch = try await tool.grep(query: "process_order", fileTypes: ["rs"])
        XCTAssertTrue(rustSearch.contains("lib.rs"), "Expected lib.rs match for rust search")
    }

    func testConcurrentEnsureIndexUsesSingleBuild() async throws {
        let (_, root, indexDir) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: indexDir)
        }

        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for _ in 0..<8 {
                group.addTask { try await tool.ensureIndex() }
            }
            try await group.waitForAll()
        }

        let stats = try await tool.stats()
        let isLoaded = await tool.isLoaded
        XCTAssertGreaterThan(stats.totalDocuments, 0)
        XCTAssertTrue(isLoaded)
    }

    func testManagerUnloadsInactiveProjectHandle() async throws {
        let (_, firstRoot, _) = try createTestWorkspace()
        let (_, secondRoot, _) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: firstRoot)
            try? FileManager.default.removeItem(at: secondRoot)
        }

        let manager = SyntextIndexManager.shared
        await manager.deactivateAll()
        let firstTool = await manager.tool(for: firstRoot)
        try await firstTool.ensureIndex()
        let wasLoaded = await firstTool.isLoaded
        XCTAssertTrue(wasLoaded)

        _ = await manager.tool(for: secondRoot)
        let remainsLoaded = await firstTool.isLoaded
        let remainsIndexed = await firstTool.isIndexed
        XCTAssertFalse(remainsLoaded)
        XCTAssertTrue(remainsIndexed, "unloading must preserve the reusable disk index")

        try await manager.deleteIndex(for: firstRoot)
        try await manager.deleteIndex(for: secondRoot)
        await manager.deactivateAll()
    }

    func testSyntextToolFileModification() async throws {
        let (_, root, indexDir) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: indexDir)
        }

        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent(".git").path))
        try await tool.buildIndex()

        let modifiedFile = root.appendingPathComponent("Main.swift")
        try "func calculateTax(amount: Double) -> Double {\n    let discountFactor = 0.9\n    return amount * 0.15 * discountFactor\n}\n"
            .write(to: modifiedFile, atomically: true, encoding: .utf8)

        try await tool.fileDidChange(at: modifiedFile)
        await tool.syncQuietly()

        let updatedMatch = try await tool.grep(query: "discountFactor")
        XCTAssertTrue(updatedMatch.contains("discountFactor"), "Updated content should match after notifyChange")
    }

    func testAppToolRegistryGrepSearchExecution() async throws {
        let (project, root, _) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
        }

        // Build index for the project root
        _ = try await SyntextIndexManager.shared.buildIndex(for: root)

        let call = AppToolCall(
            name: "grep_search",
            arguments: [
                "query": "calculateTax",
                "literal_search": "true"
            ],
            category: .fileRead
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "grep_search execution should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("calculateTax"), "Output should contain search hit: \(result.output)")
        try await SyntextIndexManager.shared.deleteIndex(for: root)
    }

    func testGrepSearchFallsBackWithoutProjectOptIn() async throws {
        let (_, root, _) = try createTestWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }
        let project = AppProject(
            name: "Unindexed Project",
            rootDirectoryPath: root.path,
            syntextIndexEnabled: false
        )
        await SyntextIndexManager.shared.deactivateAll()
        AppToolRegistry.syntextIndexingEnabled = true

        let call = AppToolCall(
            name: "grep_search",
            arguments: ["query": "calculateTax", "literal_search": "true"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertFalse(result.isError, "unindexed fallback should still search: \(result.output)")
        XCTAssertTrue(result.output.contains("calculateTax"))
        let wasIndexed = await SyntextIndexManager.shared.isIndexed(for: root)
        XCTAssertFalse(wasIndexed)
    }

    func testAppProjectSyntextIndexEnabledCoding() throws {
        let original = AppProject(
            name: "Test Coding Project",
            rootDirectoryPath: "/path/to/repo",
            syntextIndexEnabled: true
        )

        let encoded = try JSONEncoder().encode(original)
        let decoded = try JSONDecoder().decode(AppProject.self, from: encoded)

        XCTAssertEqual(decoded.syntextIndexEnabled, true)
        XCTAssertEqual(decoded.name, "Test Coding Project")

        // Test backward compatibility when syntextIndexEnabled is absent
        let legacyJSON = "{\"name\":\"Legacy Project\"}".data(using: .utf8)!
        let legacyDecoded = try JSONDecoder().decode(AppProject.self, from: legacyJSON)
        XCTAssertEqual(legacyDecoded.syntextIndexEnabled, false)
    }

    func testDeleteIndexRemovesDirectoryAndResetsState() async throws {
        let (_, root, indexDir) = try createTestWorkspace()
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: indexDir)
        }

        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        _ = try await tool.buildIndex()
        let isIndexedBefore = await tool.isIndexed
        XCTAssertTrue(isIndexedBefore)
        XCTAssertTrue(FileManager.default.fileExists(atPath: indexDir.path))

        try await tool.deleteIndex()
        let isIndexedAfter = await tool.isIndexed
        XCTAssertFalse(isIndexedAfter)
        XCTAssertFalse(FileManager.default.fileExists(atPath: indexDir.path))
    }

    func testMacAppSettingsSyntextIndexingEnabledPersistence() throws {
        let defaultSettings = MacAppSettings()
        XCTAssertTrue(defaultSettings.syntextIndexingEnabled)

        let disabledSettings = MacAppSettings(syntextIndexingEnabled: false)
        XCTAssertFalse(disabledSettings.syntextIndexingEnabled)

        let encoded = try JSONEncoder().encode(disabledSettings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: encoded)
        XCTAssertFalse(decoded.syntextIndexingEnabled)

        // Test backward compatibility: absent key decodes to true
        let legacyJSON = "{\"temperature\":0.7}".data(using: .utf8)!
        let legacyDecoded = try JSONDecoder().decode(MacAppSettings.self, from: legacyJSON)
        XCTAssertTrue(legacyDecoded.syntextIndexingEnabled)
    }
}
