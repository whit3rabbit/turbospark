import TurboSpark
import XCTest

@testable import TurboSparkApp

/// E1-E6: the model install and delete lifecycle.
@MainActor
final class ModelInstallLifecycleTests: XCTestCase {

    // MARK: - E1: Cancel retains ownership until the native stream closes

    func testCancellationWaitsForTheWriterBeforeAllowingRetry() async {
        let appModel = AppModel()
        defer { appModel.stopCronScheduler() }
        let (events, continuation) = AsyncThrowingStream<InstallEvent, Error>.makeStream()
        appModel.installModel(alias: "cancel-fixture", stream: { _ in events })
        let task = appModel.installTask
        await Task.yield()

        appModel.cancelInstall()
        XCTAssertTrue(appModel.isInstallingModel)
        XCTAssertTrue(appModel.isCancellingModelInstall)
        XCTAssertFalse(task?.isCancelled ?? true, "the consumer must wait for native completion")
        XCTAssertFalse(appModel.canInstall(alias: "cancel-fixture"))
        XCTAssertTrue(
            appModel.canInstall(alias: "another-model"),
            "a different model may queue while the native writer is stopping")
        appModel.installModel(alias: "cancel-fixture", stream: { _ in
            XCTFail("a second writer must not start during cancellation")
            return events
        })

        continuation.yield(.stage("a late installer message"))
        continuation.finish(throwing: NSError(domain: "install cancelled", code: 1))
        await task?.value
        XCTAssertFalse(appModel.isInstallingModel)
        XCTAssertFalse(appModel.isCancellingModelInstall)
        XCTAssertEqual(appModel.modelDownloads.first?.status, .cancelled)
        XCTAssertNil(appModel.error, "an acknowledged cancellation is not an install failure")
        XCTAssertTrue(appModel.canInstall(alias: "cancel-fixture"))
    }

    func testCompletedInstallWinsCancelRaceWithoutAutoLoading() async {
        let appModel = AppModel()
        defer { appModel.stopCronScheduler() }
        let (events, continuation) = AsyncThrowingStream<InstallEvent, Error>.makeStream()
        appModel.installModel(alias: "completed-fixture", stream: { _ in events })
        let task = appModel.installTask
        await Task.yield()
        appModel.cancelInstall()
        continuation.yield(.finished(model(alias: "completed-fixture", path: "/tmp/completed-fixture")))
        continuation.finish()
        await task?.value
        XCTAssertEqual(appModel.modelDownloads.first?.status, .completed)
        XCTAssertNil(appModel.session, "Cancel must prevent automatic loading")
        XCTAssertFalse(appModel.isInstallingModel)
        XCTAssertNil(appModel.error)
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
