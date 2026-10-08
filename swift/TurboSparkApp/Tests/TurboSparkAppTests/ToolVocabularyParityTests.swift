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
        "schedule_wakeup", "schedulewakeup", "skill", "memory_explain", "memory_search",
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

    func testFileAliasSetsAreSupported() {
        let all = AppToolRegistry.readFileAliases
            .union(AppToolRegistry.writeFileAliases)
            .union(AppToolRegistry.editFileAliases)
        XCTAssertTrue(all.isSubset(of: AppToolRegistry.supportedToolNames),
                      "\(all.subtracting(AppToolRegistry.supportedToolNames).sorted())")
    }
}
