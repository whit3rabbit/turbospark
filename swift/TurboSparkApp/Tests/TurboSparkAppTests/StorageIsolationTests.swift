import Foundation
@testable import TurboSparkApp
import XCTest

/// Guards the seam that keeps the test suite out of the user's real data.
///
/// **THIS IS A REGRESSION TEST FOR ACTUAL DATA LOSS.** Seven stores each
/// computed `applicationSupportDirectory + "TurboSpark"` for themselves with
/// no test seam, so an `AppModel()` built in a test wrote the real archive.
/// `AppChatFileStore.save` writes the file WHOLE, so a one-chat fixture
/// replaced a user's entire chat history, on every `swift test`, silently.
/// It was found only because the fixture chat kept reappearing in the UI
/// after being deleted.
///
/// The case that carries this file is
/// `testSavingAChatDoesNotTouchTheRealArchive`: the others check the wiring,
/// but only that one reproduces the original failure.
final class StorageIsolationTests: XCTestCase {
    /// Where the app would write if nothing redirected it.
    private var realStoreDirectory: URL {
        FileManager.default
            .urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
            .appendingPathComponent("TurboSpark", isDirectory: true)
    }

    private var realArchiveURL: URL {
        realStoreDirectory.appendingPathComponent("chats_archive.json")
    }

    func testTheSuiteIsDetectedAsATestHost() {
        XCTAssertTrue(
            AppStorageRoot.isRunningTests,
            "the redirect keys on this; if it is false the suite writes real user data")
    }

    func testTheStoreRootIsNotTheUsersApplicationSupportDirectory() {
        XCTAssertNotEqual(
            AppStorageRoot.directory.standardizedFileURL,
            realStoreDirectory.standardizedFileURL)

        // Not merely different, but outside it: a subdirectory would still put
        // fixtures where a user browsing their own data would find them.
        XCTAssertFalse(
            AppStorageRoot.directory.standardizedFileURL.path
                .hasPrefix(realStoreDirectory.standardizedFileURL.path))
    }

    /// The original failure, reproduced end to end.
    func testSavingAChatDoesNotTouchTheRealArchive() throws {
        let fileManager = FileManager.default
        let before = try? Data(contentsOf: realArchiveURL)
        let existedBefore = fileManager.fileExists(atPath: realArchiveURL.path)

        var archive = AppChatArchive.empty()
        archive.chats = [AppChat(title: "storage isolation fixture")]
        AppChatFileStore.save(archive)

        let after = try? Data(contentsOf: realArchiveURL)
        XCTAssertEqual(
            fileManager.fileExists(atPath: realArchiveURL.path), existedBefore,
            "saving a chat under test created or removed the user's real archive")
        XCTAssertEqual(
            before, after,
            "saving a chat under test overwrote the user's real chat archive")

        // And it did land somewhere: a save that silently wrote nothing would
        // pass both assertions above while proving nothing.
        let reloaded = AppChatFileStore.load()
        XCTAssertEqual(reloaded.chats.first?.title, "storage isolation fixture")
    }

    func testEveryStoreDerivesFromTheOneRoot() {
        // A store spelling its own path is how this broke; each of these
        // resolves through `AppStorageRoot`, so all of them move together.
        let root = AppStorageRoot.directory.standardizedFileURL.path
        XCTAssertTrue(AppStorageRoot.file("chats_archive.json").path.hasPrefix(root))
        XCTAssertTrue(AppStorageRoot.subdirectory("Hooks").path.hasPrefix(root))
        XCTAssertTrue(AppStorageRoot.subdirectory("tools").path.hasPrefix(root))
    }
}
