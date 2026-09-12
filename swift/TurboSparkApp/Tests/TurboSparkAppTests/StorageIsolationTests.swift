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

    /// The test root must be fresh per LAUNCH, not merely per process id.
    ///
    /// Process ids are reused by the OS, and a run whose id matches an
    /// earlier run's inherits that run's stores: a cross-run channel for
    /// every file under the root. It was the first suspect in the cron flake
    /// and turned out not to be the cause there, but the channel is real
    /// either way. The launch token cannot be observed changing from inside
    /// one process, so what this pins is the shape that makes per-launch
    /// roots what they are: a token no reused pid can reproduce, beside a
    /// pid that a leaked directory can still be traced by.
    func testTheTestRootCarriesAPerLaunchTokenBeyondThePid() {
        let name = AppStorageRoot.machineRoot.lastPathComponent
        XCTAssertTrue(
            name.hasPrefix("TurboSparkTests-"),
            "the test root left the TurboSparkTests- prefix; update this test with intent")

        let remainder = name.dropFirst("TurboSparkTests-".count)
        guard let separator = remainder.firstIndex(of: "-") else {
            XCTFail("test root \(name) carries no launch token beyond the pid")
            return
        }
        let pid = String(remainder[..<separator])
        let token = String(remainder[remainder.index(after: separator)...])
        XCTAssertNotNil(
            Int(pid),
            "test root \(name) lost its pid component, which traces a leaked directory")
        XCTAssertNotNil(
            UUID(uuidString: token),
            "test root \(name) carries no per-launch token, so a reused pid inherits an old run's files")

        // The root stays under the temp directory, where per-launch
        // accumulation is purgeable and nothing can be mistaken for data.
        let temporary = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .standardizedFileURL.path
        XCTAssertTrue(
            AppStorageRoot.machineRoot.standardizedFileURL.path.hasPrefix(temporary),
            "test root \(AppStorageRoot.machineRoot.path) left the temp directory")
    }
}
