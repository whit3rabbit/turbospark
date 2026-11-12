import XCTest
@testable import TurboSparkApp

/// The tool vocabularies are still separate tables (execute switch,
/// supportedToolNames, builtInCategory, workspaceRootedToolNames). Until they
/// collapse into one, these assertions make drift between them a test failure
/// instead of a permission-category bug.
final class ToolVocabularyParityTests: XCTestCase {
    /// Names that deliberately take `category(for:)`'s `.automation` default.
    /// A new executor alias must either get a category or be added here on
    /// purpose; silently landing on the default is the bug this pins.
    private static let intentionallyDefaulted: Set<String> = [
        "cron_create", "cron_delete", "cron_list", "croncreate", "crondelete", "cronlist",
        "schedule_wakeup", "schedulewakeup", "skill",
    ]

    func testEverySupportedToolNameResolvesToABuiltInCategoryOrIsListedAsDefaulted() {
        let missing = AppToolRegistry.supportedToolNames
            .filter { AppToolCatalog.builtInCategory(for: $0) == nil }
            .subtracting(Self.intentionallyDefaulted)
            .sorted()
        XCTAssertTrue(
            missing.isEmpty,
            "Executor-supported names with no permission category fall to the default gate: \(missing)")
        for name in Self.intentionallyDefaulted {
            XCTAssertEqual(AppToolCatalog.category(for: name), .automation, name)
        }
    }

    func testWorkspaceRootedNamesAreAllExecutable() {
        let unknown = AppToolRegistry.workspaceRootedToolNames
            .subtracting(AppToolRegistry.supportedToolNames)
            // codemode is dispatched before the static switch.
            .subtracting(["codemode"])
            .sorted()
        XCTAssertTrue(unknown.isEmpty, "Rooted names the executor does not implement: \(unknown)")
    }

    /// Every string literal `execute` dispatches on, read from its source so a
    /// new `case` cannot be added without landing here.
    private static func executeCaseNames() throws -> Set<String> {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Tools/Registry/AppToolRegistry.swift")
        let source = try String(contentsOf: url, encoding: .utf8)
        let start = try XCTUnwrap(source.range(of: "switch call.name.lowercased() {"))
        let end = try XCTUnwrap(source.range(of: "            default:\n", range: start.upperBound..<source.endIndex))
        let region = String(source[start.upperBound..<end.lowerBound])
        let caseLine = try NSRegularExpression(
            pattern: "(?m)^ {12}case ((?:\"[^\"]+\"|[\\s,])+):")
        var names: Set<String> = []
        let ns = region as NSString
        for m in caseLine.matches(in: region, range: NSRange(location: 0, length: ns.length)) {
            let list = ns.substring(with: m.range(at: 1))
            for lit in list.split(separator: "\"", omittingEmptySubsequences: false).enumerated()
            where lit.offset % 2 == 1 {
                names.insert(String(lit.element))
            }
        }
        return names
    }

    /// `codemode` and `repl` are the two named exceptions. `codemode` has an
    /// `execute` case but is gated by `CodemodeSettings.isEnabled` in
    /// `isImplemented`, so it is deliberately absent from the static
    /// `supportedToolNames` (it is opt-in). `repl` has a category but no
    /// `execute` case and is not advertised: the REPL subsystem is not
    /// reachable from the tool path yet (issue C3), and that must stay true
    /// until it is wired on purpose.
    func testEveryExecuteCaseIsSupportedAndCategorized() throws {
        let cases = try Self.executeCaseNames()
        XCTAssertGreaterThan(cases.count, 100, "case parser found too few names; the source shape changed")
        let notSupported = cases
            .subtracting(AppToolRegistry.supportedToolNames)
            .subtracting(["codemode"])
            .sorted()
        XCTAssertTrue(notSupported.isEmpty, "execute cases missing from supportedToolNames: \(notSupported)")
        let uncategorized = cases
            .filter { AppToolCatalog.builtInCategory(for: $0) == nil }
            .subtracting(Self.intentionallyDefaulted)
            .sorted()
        XCTAssertTrue(uncategorized.isEmpty, "execute cases with no builtInCategory: \(uncategorized)")
    }

    func testCodemodeAndReplExceptionsStayAsDocumented() throws {
        let cases = try Self.executeCaseNames()
        XCTAssertTrue(cases.contains("codemode"))
        XCTAssertFalse(AppToolRegistry.supportedToolNames.contains("codemode"))
        XCTAssertTrue(AppToolRegistry.workspaceRootedToolNames.contains("codemode"))
        XCTAssertEqual(AppToolCatalog.builtInCategory(for: "codemode"), .mcp)
        // repl: classified but neither executed nor supported (C3).
        XCTAssertFalse(cases.contains("repl"))
        XCTAssertFalse(AppToolRegistry.supportedToolNames.contains("repl"))
        XCTAssertEqual(AppToolCatalog.builtInCategory(for: "repl"), .terminal)
    }

    func testMemoryReadToolsAreReadOnlyAndWorkWithoutAProject() {
        for name in ["memory_search", "memory_explain"] {
            XCTAssertEqual(AppToolCatalog.category(for: name), .fileRead, name)
            XCTAssertFalse(AppToolRegistry.workspaceRootedToolNames.contains(name), name)
        }
        XCTAssertEqual(AppToolCatalog.category(for: "stopagent"), .automation)
        XCTAssertTrue(AppToolRegistry.supportedToolNames.contains("stopagent"))
    }

    func testFileAliasSetsAreSupported() {
        let all = AppToolRegistry.readFileAliases
            .union(AppToolRegistry.writeFileAliases)
            .union(AppToolRegistry.editFileAliases)
        XCTAssertTrue(all.isSubset(of: AppToolRegistry.supportedToolNames),
                      "\(all.subtracting(AppToolRegistry.supportedToolNames).sorted())")
    }
}
