import TurboSpark
import XCTest

@testable import TurboSparkApp

/// E1-E6: the model install and delete lifecycle.
@MainActor
final class ModelInstallLifecycleTests: XCTestCase {

    // MARK: - E1: Cancel stops watching, and says so

    func testCancellingAnInstallRefusesToStartTheSameOneAgain() {
        let appModel = AppModel()
        appModel.installingAlias = "gemma4"
        appModel.isInstallingModel = true

        appModel.cancelInstall()

        // The engine has no install-cancel call: dropping the consumer ends
        // DELIVERY while `ts_install` keeps streaming that checkpoint to the
        // same directory. A second install of the same alias would be a
        // second writer on it.
        XCTAssertTrue(appModel.abandonedInstallAliases.contains("gemma4"))
        XCTAssertFalse(
            appModel.canInstall(alias: "gemma4"),
            "Re-installing an alias whose walk cannot be stopped must be refused.")
        XCTAssertTrue(
            appModel.canInstall(alias: "some-other-model"),
            "A DIFFERENT model writes a different directory and is still installable.")
    }

    func testAnAbandonedInstallCannotBeRestartedThroughInstallModel() {
        let appModel = AppModel()
        appModel.abandonedInstallAliases.insert("gemma4")

        appModel.installModel(alias: "gemma4")

        XCTAssertFalse(
            appModel.isInstallingModel,
            "`installModel` must consult the abandoned set, not just `isInstallingModel`.")
        XCTAssertNotNil(appModel.activeToast, "The refusal must say why.")
    }

    // MARK: - E6: delete matches on path, never on a colliding alias

    private func model(alias: String, path: String) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "", revision: "", path: path, family: "gemma4",
            installBytes: 0, installedOn: "")
    }

    func testAScannedRowWithACollidingAliasIsNotTreatedAsCatalogTracked() {
        // The predicate is tested against a FIXTURE rather than through
        // `deleteModel`: that path reads the real `~/.turbospark/installed.json`,
        // which is not redirected by `AppStorageRoot`, so a test driven
        // through it can only discriminate on machines that happen to have a
        // colliding alias installed.
        let catalogRows = [model(alias: "gemma4", path: "/Users/x/.turbospark/models/gemma4.gturbo")]
        let scanned = model(alias: "gemma4", path: "/Volumes/External/lmstudio/gemma4.gturbo")

        XCTAssertFalse(
            AppModel.isCatalogTracked(model: scanned, in: catalogRows),
            "An alias collision must not make a scanned row look installed -- the delete that "
                + "follows removes the CATALOG's copy, at a different path.")

        XCTAssertTrue(
            AppModel.isCatalogTracked(model: catalogRows[0], in: catalogRows),
            "A genuine catalog row must still be deletable.")
    }

    func testACatalogRowIsMatchedThroughPathNormalisation() {
        let catalogRows = [model(alias: "a", path: "/tmp/models/../models/a.gturbo")]
        XCTAssertTrue(
            AppModel.isCatalogTracked(
                model: model(alias: "different-alias", path: "/tmp/models/a.gturbo"),
                in: catalogRows),
            "The same file under a different spelling is the same file, and the alias is not "
                + "what identifies it.")
    }

    // MARK: - E5: model metadata is encrypted, not in UserDefaults

    func testModelMetadataIsWrittenIntoTheProfileVaultRatherThanUserDefaults() throws {
        // swift/CLAUDE.md Gotcha 37's class, in the one store the original fix
        // missed because it is not a path at all. A test that deletes a model
        // calls `removeMetadata`, which mutated the developer's real
        // nicknames and favorites.
        let alias = "metadata-test-\(UUID().uuidString.prefix(8))"
        ModelOrganizationStore.shared.setNickname("A Nickname", for: alias, path: "/tmp/\(alias)")

        let file = AppStorageRoot.file("model_organization.json")
        XCTAssertFalse(FileManager.default.fileExists(atPath: file.path))
        let key = try XCTUnwrap(ProfileRepository.protectedRecordKey(for: file))
        let contents = try XCTUnwrap(ProfileRepository.shared.rawRecord(key: key))
        XCTAssertNotNil(contents.range(of: Data("A Nickname".utf8)))

        ModelOrganizationStore.shared.removeMetadata(for: alias, path: "/tmp/\(alias)")
    }
}
