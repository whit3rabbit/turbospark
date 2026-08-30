import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Regression tests for state#14/U5/U6: alias de-duplication was one level
/// deep (`if existingAliases.contains(alias) { alias += " (LM Studio)" }`,
/// no re-check), so a THIRD model sharing a base alias collided silently --
/// `Set.insert` on an already-present value is a no-op, so two distinct
/// `InstalledModel` rows ended up with the identical alias, which is the
/// field `selectModel`/`deleteModel` used to key off of.
@MainActor
final class ModelIdentityTests: XCTestCase {
    func testUniqueAliasLoopsPastMultipleCollisions() {
        let appModel = AppModel()
        var existing: Set<String> = ["gemma4"]

        let first = appModel.uniqueAlias(base: "gemma4", suffix: "LM Studio", existing: existing)
        XCTAssertEqual(first, "gemma4 (LM Studio)")
        existing.insert(first)

        // A THIRD model with the same base alias: the one-deep version
        // would try "gemma4 (LM Studio)" again, find it already taken, and
        // insert the duplicate anyway.
        let second = appModel.uniqueAlias(base: "gemma4", suffix: "LM Studio", existing: existing)
        XCTAssertNotEqual(second, first, "A second collision on the same suffix must not silently duplicate the first.")
        XCTAssertFalse(existing.contains(second))
        existing.insert(second)

        let third = appModel.uniqueAlias(base: "gemma4", suffix: "LM Studio", existing: existing)
        XCTAssertFalse(existing.contains(third))
        XCTAssertNotEqual(third, first)
        XCTAssertNotEqual(third, second)
    }

    func testSelectModelKeysOnPathNotAlias() {
        let appModel = AppModel()
        // Two distinct models sharing an alias (the exact shape the old
        // uniquing bug could produce, or a pre-existing corrupt install
        // list) must still be distinguishable by path.
        let modelA = InstalledModel(alias: "shared", repo: "a/a", path: "/models/a.gturbo", family: "gemma4")
        let modelB = InstalledModel(alias: "shared", repo: "b/b", path: "/models/b.gturbo", family: "gemma4")
        appModel.installed = [modelA, modelB]
        appModel.selected = modelA

        appModel.selectModel(modelB)
        XCTAssertEqual(appModel.selected?.path, modelB.path, "selectModel must switch based on path even when aliases collide.")
    }
}
