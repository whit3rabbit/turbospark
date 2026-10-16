import Foundation
import XCTest
@testable import TurboSparkApp

/// What indexed code search may return, and how it is bounded.
final class SyntextGrepSafetyTests: XCTestCase {
    private var root: URL!
    private var indexDir: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        let base = FileManager.default.temporaryDirectory
            .appendingPathComponent("grep-safety-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("repo", isDirectory: true)
        indexDir = base.appendingPathComponent("index.syntext", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root.deletingLastPathComponent())
        AppToolRegistry.syntextIndexingEnabled = true
        super.tearDown()
    }

    private func write(_ relative: String, _ text: String) throws {
        let url = root.appendingPathComponent(relative)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    }

    private func builtTool() async throws -> SyntextCodeSearchTool {
        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        try await tool.buildIndex()
        return tool
    }

    func testLiteralSearchMatchesMetacharactersExactly() async throws {
        try write("a.php", "echo $var;\n$value = 1;\n")
        try write("b.js", "const x = array[0];\nconst y = array0;\n")
        let tool = try await builtTool()

        let dollar = try await tool.grep(query: "$var", literalSearch: true, contextLines: 0)
        XCTAssertTrue(dollar.contains("a.php:1:"), dollar)
        XCTAssertFalse(dollar.contains("a.php:2:"), dollar)

        let bracket = try await tool.grep(query: "array[0]", literalSearch: true, contextLines: 0)
        XCTAssertTrue(bracket.contains("b.js:1:"), bracket)
        XCTAssertFalse(bracket.contains("b.js:2:"), "regex [0] would match array0: \(bracket)")
    }

    func testIndexedOutputCarriesContextLines() async throws {
        try write("ctx.swift", "line one\nline two\nNEEDLE here\nline four\nline five\n")
        let tool = try await builtTool()
        let output = try await tool.grep(query: "NEEDLE", contextLines: 1)
        XCTAssertTrue(output.contains("ctx.swift-2-line two"), output)
        XCTAssertTrue(output.contains("ctx.swift:3:NEEDLE here"), output)
        XCTAssertTrue(output.contains("ctx.swift-4-line four"), output)
    }

    private func writeSecrets() throws {
        try write("Main.swift", "let token = \"TOPSECRETVALUE\"\n")
        try write(".env", "TOPSECRETVALUE=1\n")
        try write(".git/config", "url = https://user:TOPSECRETVALUE@host/repo\n")
        try write("server.pem", "TOPSECRETVALUE\n")
        try write("credentials", "TOPSECRETVALUE\n")
        try write("node_modules/pkg/index.js", "TOPSECRETVALUE\n")
        try write(".hidden/notes.txt", "TOPSECRETVALUE\n")
    }

    func testIndexedSearchHidesSensitiveHiddenAndVendorFiles() async throws {
        try writeSecrets()
        let tool = try await builtTool()
        let output = try await tool.grep(query: "TOPSECRETVALUE", contextLines: 0)
        XCTAssertTrue(output.contains("Main.swift:1:"), output)
        for hidden in [".env", ".git/config", "server.pem", "credentials", "node_modules/", ".hidden/"] {
            XCTAssertFalse(output.contains(hidden), "\(hidden) leaked: \(output)")
        }
    }

    func testUnindexedSearchHidesTheSameFiles() throws {
        try writeSecrets()
        let output = try AppToolRegistry.searchCode(pattern: "TOPSECRETVALUE", relPath: ".", rootURL: root)
        XCTAssertTrue(output.contains("Main.swift:1:"), output)
        for hidden in [".env", ".git/config", "server.pem", "credentials", "node_modules/", ".hidden/"] {
            XCTAssertFalse(output.contains(hidden), "\(hidden) leaked: \(output)")
        }
    }

    func testPathScopeHoldsWithAPathFilter() async throws {
        try write("src/a.swift", "NEEDLE\n")
        try write("other/src/x.swift", "NEEDLE\n")
        let tool = try await builtTool()
        let scope = try AppToolRegistry.syntextPathScope(relPath: "src", rootURL: root)
        XCTAssertEqual(scope, "src/")
        let output = try await tool.grep(query: "NEEDLE", pathFilter: "*.swift", scope: scope, contextLines: 0)
        XCTAssertTrue(output.contains("src/a.swift"), output)
        XCTAssertFalse(output.contains("other/src/x.swift"), output)
    }

    func testEscapingPathIsAnErrorNotAWholeRepoSearch() {
        XCTAssertThrowsError(try AppToolRegistry.syntextPathScope(relPath: "../outside", rootURL: root))
        XCTAssertThrowsError(try AppToolRegistry.syntextPathScope(relPath: "/etc", rootURL: root))
    }

    func testResultsAreCapped() async throws {
        try write("many.txt", String(repeating: "HIT\n", count: 900))
        let tool = try await builtTool()
        // No context: a match's context lines can be later matches, printed
        // with `:` too, which would blur the count.
        let output = try await tool.grep(query: "HIT", contextLines: 0, maxResults: 100_000)
        let matchLines = output.split(separator: "\n").filter { $0.hasPrefix("many.txt:") }
        XCTAssertEqual(matchLines.count, SyntextCodeSearchTool.maxResultsCap)
        XCTAssertTrue(output.contains("Showing the first 500 matches"), String(output.suffix(200)))
    }

    func testSearchSeesANewFileNotifiedJustBefore() async throws {
        // Enough base documents that one new file fits syntext's overlay.
        for index in 0..<30 { try write("base\(index).swift", "let base\(index) = \(index)\n") }
        let tool = try await builtTool()
        // A file created after the build. Syntext re-reads edited files it
        // already indexes at query time, but a new file is invisible until
        // its notification is committed.
        try write("created.swift", "let newFileMarker = 2\n")
        try await tool.fileDidChange(at: root.appendingPathComponent("created.swift"))
        // No quiet-period wait: the search itself must commit the pending change.
        let output = try await tool.grep(query: "newFileMarker", caseSensitive: true, contextLines: 0)
        XCTAssertTrue(output.contains("created.swift:1:"), output)
    }

    func testOverlayFullCommitRebuildsInsteadOfStayingStale() async throws {
        // One base document: syntext refuses the commit of a second
        // ("overlay too large"). That used to be swallowed, and the new file
        // stayed invisible until a manual Reindex.
        try write("only.swift", "let only = 1\n")
        let tool = try await builtTool()
        try write("created.swift", "let overflowMarker = 2\n")
        try await tool.fileDidChange(at: root.appendingPathComponent("created.swift"))
        do {
            let stale = try await tool.grep(query: "overflowMarker", caseSensitive: true, contextLines: 0)
            XCTFail("a stale handle answered instead of the fallback: \(stale)")
        } catch is SyntextIndexNotReady {
        }
        try await tool.ensureIndex()
        let rebuilt = try await tool.grep(query: "overflowMarker", caseSensitive: true, contextLines: 0)
        XCTAssertTrue(rebuilt.contains("created.swift:1:"), rebuilt)
    }

    func testDeletedFileLeavesTheIndex() async throws {
        try write("gone.swift", "let deletedMarker = 1\n")
        let tool = try await builtTool()
        try FileManager.default.removeItem(at: root.appendingPathComponent("gone.swift"))
        try await tool.fileDidDelete(at: root.appendingPathComponent("gone.swift"))
        let output = try await tool.grep(query: "deletedMarker", contextLines: 0)
        XCTAssertTrue(output.contains("No matches found"), output)
    }

    func testSearchDoesNotWaitForAFirstBuild() async throws {
        try write("a.swift", "let x = 1\n")
        let tool = SyntextCodeSearchTool(repoRoot: root, indexDir: indexDir)
        do {
            _ = try await tool.grep(query: "x")
            XCTFail("a first search must not wait for the build")
        } catch is SyntextIndexNotReady {
        }
        // The build it started finishes in the background.
        try await tool.ensureIndex()
        let output = try await tool.grep(query: "let x", contextLines: 0)
        XCTAssertTrue(output.contains("a.swift:1:"), output)
    }

    func testDefaultIndexDirectoryIsAFixedLengthNameUnderTheTestRoot() {
        let long = URL(fileURLWithPath: "/" + String(repeating: "deep-folder-name/", count: 30))
        let dir = SyntextCodeSearchTool.defaultIndexDirectory(for: long)
        XCTAssertEqual(dir.lastPathComponent.count, 32 + ".syntext".count)
        XCTAssertTrue(dir.path.hasPrefix(AppStorageRoot.machineRoot.path), dir.path)
        XCTAssertEqual(
            dir, SyntextCodeSearchTool.defaultIndexDirectory(for: URL(fileURLWithPath: long.path + "/")))
    }

    func testGrepSearchOnASensitivePathIsHighRisk() {
        let assessment = ToolRiskClassifier.assessRisk(name: "grep_search", arguments: ["query": "x", "path": ".env"])
        XCTAssertEqual(assessment.level, .high)
    }

    func testRegistryUsesTheIndexWhenOptedIn() async throws {
        try write("Main.swift", "func calculateTax() {}\n")
        let project = AppProject(name: "Indexed", rootDirectoryPath: root.path, syntextIndexEnabled: true)
        _ = try await SyntextIndexManager.shared.buildIndex(for: root)
        AppToolRegistry.syntextIndexingEnabled = true
        let call = AppToolCall(name: "grep_search", arguments: ["query": "calculateTax"], category: .fileRead)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, result.output)
        // The unindexed scan prefixes "Found N matches"; the index prints ripgrep lines.
        XCTAssertFalse(result.output.hasPrefix("Found"), "answered by the fallback: \(result.output)")
        XCTAssertTrue(result.output.contains("Main.swift:1:"), result.output)
        try await SyntextIndexManager.shared.deleteIndex(for: root)
    }
}
