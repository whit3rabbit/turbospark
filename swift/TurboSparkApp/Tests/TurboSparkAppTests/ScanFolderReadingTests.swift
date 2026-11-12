import Combine
import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Records what a scan was asked to do, from whatever thread asks.
final class ScanRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recorded: [String] = []
    private var mainThread: [Bool] = []

    func note(_ entry: String) {
        lock.lock(); defer { lock.unlock() }
        recorded.append(entry)
        mainThread.append(Thread.isMainThread)
    }

    var entries: [String] { lock.lock(); defer { lock.unlock() }; return recorded }
    var ranOnMainThread: [Bool] { lock.lock(); defer { lock.unlock() }; return mainThread }
}

/// A status a test changes between refreshes, readable from the scan's thread.
final class FolderStatusBox: @unchecked Sendable {
    private let lock = NSLock()
    private var current: ScanFolderStatus = .connected
    var value: ScanFolderStatus {
        get { lock.lock(); defer { lock.unlock() }; return current }
        set { lock.lock(); defer { lock.unlock() }; current = newValue }
    }
}

/// Task 6.12 (requirement 3.9): reading the additional scan folders.
///
/// Every case uses a temporary directory as the volumes root, so a drive being
/// plugged in is a folder appearing and no test reads a real volume.
final class ScanFolderReadingTests: XCTestCase {
    private var base: URL!
    private var root: URL!

    override func setUpWithError() throws {
        base = FileManager.default.temporaryDirectory
            .appendingPathComponent("scan-folder-reading-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("Volumes", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    private var status: (String) -> ScanFolderStatus {
        let volumes = root.path
        return { ModelStorageManager.scanFolderStatus($0, volumesRoot: volumes) }
    }

    /// A folder under the fake volumes root, with `models` `.gguf` files in it.
    @discardableResult
    private func plugIn(_ volume: String, models: Int = 0) throws -> String {
        let folder = root.appendingPathComponent("\(volume)/models", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        for index in 0..<models {
            try Data("x".utf8).write(to: folder.appendingPathComponent("model-\(index).gguf"))
        }
        return folder.path
    }

    private func unplug(_ volume: String) throws {
        try FileManager.default.removeItem(at: root.appendingPathComponent(volume))
    }

    private let countModels: (String) -> Int = {
        ModelStorageManager.scanModels(in: $0, sourceTag: "Custom").count
    }

    // MARK: - Reading

    func testAFolderUnderAMissingVolumeIsNotConnectedAndIsNeverCounted() throws {
        let live = try plugIn("T7", models: 2)
        let dead = root.path + "/Gone/models"
        let counted = ScanRecorder()

        let readings = CustomModelFoldersSectionView.readFolders(
            [live, dead], status: status,
            count: { counted.note($0); return 2 })

        XCTAssertEqual(readings[dead], ScanFolderReading(status: .notConnected(volumeName: "Gone"), modelCount: 0))
        XCTAssertEqual(readings[live], ScanFolderReading(status: .connected, modelCount: 2))
        XCTAssertEqual(counted.entries, [live], "the dead path must not be walked")
    }

    func testAFolderThatIsGoneFromAMountedVolumeIsNotFoundAndNotCounted() throws {
        try plugIn("T7")
        let absent = root.path + "/T7/vanished"
        let counted = ScanRecorder()

        let readings = CustomModelFoldersSectionView.readFolders(
            [absent], status: status, count: { counted.note($0); return 9 })

        XCTAssertEqual(readings[absent], ScanFolderReading(status: .notFound, modelCount: 0))
        XCTAssertEqual(counted.entries, [])
    }

    func testARepeatedPathIsReadOnce() throws {
        let live = try plugIn("T7")
        let asked = ScanRecorder()
        let readings = CustomModelFoldersSectionView.readFolders(
            [live, live, live], status: { asked.note($0); return .connected }, count: { _ in 1 })
        XCTAssertEqual(readings.count, 1)
        XCTAssertEqual(asked.entries, [live])
    }

    func testTheCountsOfConnectedFoldersStayTheExistingFolderCounts() {
        // `folderCounts` is the walk the readings build on; its contract
        // (a repeated path counted once, in order) is unchanged.
        let counted = ScanRecorder()
        let counts = CustomModelFoldersSectionView.folderCounts(for: ["/a", "/b", "/a"]) { path in
            counted.note(path)
            return path == "/a" ? 3 : 5
        }
        XCTAssertEqual(counts, ["/a": 3, "/b": 5])
        XCTAssertEqual(counted.entries, ["/a", "/b"])
    }

    // MARK: - The observable completion

    func testAFolderUnderAMissingFakeVolumeShowsNotConnectedInTheRow() throws {
        let dead = root.path + "/T7/models"

        let readings = CustomModelFoldersSectionView.readFolders(
            [dead], status: status, count: countModels)
        let row = ScanFolderRowModel(path: dead, reading: readings[dead], language: .english)

        XCTAssertEqual(row.statusText, "Not connected")
        XCTAssertEqual(row.statusGlyph, LibraryRowStatus.notConnected.glyph)
        XCTAssertNil(row.countText)
        XCTAssertFalse(row.reveal.isEnabled)
        XCTAssertTrue(row.remove.isEnabled)
    }

    func testPluggingTheDriveInFlipsTheRowToConnectedAndCountsItsModels() throws {
        let folder = root.path + "/T7/models"
        func rendered() -> ScanFolderRowModel {
            let readings = CustomModelFoldersSectionView.readFolders(
                [folder], status: status, count: countModels)
            return ScanFolderRowModel(path: folder, reading: readings[folder], language: .english)
        }

        let before = rendered()
        XCTAssertEqual(before.statusText, "Not connected")
        XCTAssertNil(before.countText)

        try plugIn("T7", models: 2)
        let during = rendered()
        XCTAssertNil(during.statusText)
        XCTAssertEqual(during.countText, "Models: 2")
        XCTAssertTrue(during.reveal.isEnabled)

        try unplug("T7")
        let after = rendered()
        XCTAssertEqual(after.statusText, "Not connected")
        XCTAssertNil(after.countText)
    }

    // MARK: - Off the main thread

    @MainActor
    func testTheReadingRunsOffTheMainThread() async throws {
        let live = try plugIn("T7", models: 1)
        let dead = root.path + "/Gone/models"
        let seen = ScanRecorder()
        let volumes = root.path

        let readings = await CustomModelFoldersSectionView.readOffMainActor(
            [live, dead],
            status: {
                seen.note("status \($0)")
                return ModelStorageManager.scanFolderStatus($0, volumesRoot: volumes)
            },
            count: {
                seen.note("count \($0)")
                return 1
            })

        XCTAssertEqual(readings?[live]?.status, .connected)
        XCTAssertEqual(readings?[dead]?.status, .notConnected(volumeName: "Gone"))
        XCTAssertEqual(seen.entries.count, 3)
        XCTAssertFalse(seen.ranOnMainThread.contains(true), "the status check and the walk are file system work")
    }

    func testACancelledReadDeliversNothing() async {
        let task = Task.detached { () -> [String: ScanFolderReading]? in
            while !Task.isCancelled { await Task.yield() }
            return await CustomModelFoldersSectionView.readOffMainActor(
                ["/a", "/b"], status: { _ in .connected }, count: { _ in 1 })
        }
        task.cancel()
        let result = await task.value
        XCTAssertNil(result, "a superseded reading must not be applied")
    }

    // MARK: - Where the view reads it

    private var sourcesRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
    }

    private func code(_ relative: String) throws -> String {
        MoveStoreRetirementTests.codeWithoutComments(
            try String(contentsOf: sourcesRoot.appendingPathComponent(relative), encoding: .utf8))
    }

    func testTheViewDrawsTheRowModelAndNoRawCount() throws {
        let view = try code("Components/CustomModelFoldersSectionView.swift")
        XCTAssertTrue(view.contains("ScanFolderRowModel("), "the row is drawn from the model")
        XCTAssertFalse(view.contains("scannedCounts"), "a missing count must not default to zero")
        XCTAssertFalse(view.contains("Text(\"Models:"), "the count text comes from the row model")
    }

    func testTheViewRereadsWhenAVolumeComesOrGoes() throws {
        let view = try code("Components/CustomModelFoldersSectionView.swift")
        let task = try XCTUnwrap(view.range(of: ".task(id:"))
        let window = String(view[task.lowerBound...].prefix(260))
        XCTAssertTrue(window.contains("volumeChangeCount"), window)
        XCTAssertTrue(window.contains("customModelDirectories"), window)
    }

    func testTheViewAddsNoSecondVolumeObserver() throws {
        for file in [
            "Components/CustomModelFoldersSectionView.swift", "Components/ScanFolderRowView.swift",
            "State/ScanFolderRowModel.swift",
        ] {
            let text = try code(file)
            XCTAssertFalse(text.contains("NSWorkspace.shared.notificationCenter"), file)
            XCTAssertFalse(text.contains("didMountNotification"), file)
            XCTAssertFalse(text.contains("VolumeMonitor"), file)
        }
    }

    func testNoTrailOfTheRetiredUnmountedStoreVolumeHelper() throws {
        for file in ["State/ModelStorageManager.swift", "Components/CustomModelFoldersSectionView.swift"] {
            XCTAssertFalse(try code(file).contains("unmountedStoreVolume"), file)
        }
    }
}

/// The refresh that scans the folders, and the trigger that re-reads them.
@MainActor
final class ScanFolderRefreshTests: LibraryStateTestCase {
    private func folderWithModel(named name: String) throws -> String {
        let folder = directory.appendingPathComponent("scan-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: folder.appendingPathComponent("\(name).gguf"))
        return folder.path
    }

    private func quietModel() async -> AppModel {
        let model = await makeModel(using: makeState())
        model.enableLMStudioDetection = false
        model.catalogInstalledReader = { [] }
        model.imageInstalledReader = { [] }
        model.error = nil
        model.activeToast = nil
        return model
    }

    func testRefreshingWithAnOfflineScanFolderRaisesNoErrorAndNoToast() async throws {
        let live = try folderWithModel(named: "reachable")
        // A volume that cannot be mounted: only its absence is looked at.
        let dead = "/Volumes/TurboSpark-offline-\(UUID().uuidString)/models"
        XCTAssertEqual(ModelStorageManager.scanFolderStatus(dead).isNotConnected, true, "the fixture needs a missing volume")
        let model = await quietModel()
        model.customModelDirectories = [dead, live]

        model.refreshModels()
        await model.modelScanTask?.value

        XCTAssertNil(model.error, "an unplugged drive is not an error (requirement 3.4)")
        XCTAssertNil(model.activeToast, "an unplugged drive is not a notification (requirement 3.4)")
        XCTAssertEqual(model.installed.map(\.alias), ["reachable"], "the reachable folder is still scanned")
    }

    func testRefreshingWithOnlyAnOfflineScanFolderListsNothingAndSaysNothing() async throws {
        let dead = "/Volumes/TurboSpark-offline-\(UUID().uuidString)/models"
        let model = await quietModel()
        model.customModelDirectories = [dead]

        model.refreshModels()
        await model.modelScanTask?.value

        XCTAssertNil(model.error)
        XCTAssertNil(model.activeToast)
        XCTAssertTrue(model.installed.isEmpty)
        XCTAssertEqual(model.imageModels.count, 0)
    }

    func testAnUnpluggedScanFolderIsNotWalkedAndItsModelsAreNotListed() async throws {
        // A real folder with a real model, declared to be on an unplugged drive:
        // if the refresh walked it, "hidden" would be listed.
        let folder = try folderWithModel(named: "hidden")
        let model = await quietModel()
        model.scanFolderStatusReader = { $0 == folder ? .notConnected(volumeName: "T7") : .connected }
        model.customModelDirectories = [folder]

        model.refreshModels()
        await model.modelScanTask?.value

        XCTAssertTrue(model.installed.isEmpty, "the dead path was walked: \(model.installed.map(\.alias))")
        XCTAssertNil(model.error)
        XCTAssertNil(model.activeToast)
    }

    func testAFolderWhoseDriveIsBackIsWalkedAgain() async throws {
        let folder = try folderWithModel(named: "back")
        let model = await quietModel()
        let box = FolderStatusBox()
        model.scanFolderStatusReader = { _ in box.value }
        model.customModelDirectories = [folder]

        box.value = .notConnected(volumeName: "T7")
        model.refreshModels()
        await model.modelScanTask?.value
        XCTAssertTrue(model.installed.isEmpty)

        box.value = .connected
        model.refreshModels()
        await model.modelScanTask?.value
        XCTAssertEqual(model.installed.map(\.alias), ["back"])
    }

    func testAFolderThatIsNotFoundIsNotWalkedEither() async throws {
        let folder = try folderWithModel(named: "stray")
        let model = await quietModel()
        model.scanFolderStatusReader = { _ in .notFound }
        model.customModelDirectories = [folder]

        model.refreshModels()
        await model.modelScanTask?.value

        XCTAssertTrue(model.installed.isEmpty)
        XCTAssertNil(model.error)
    }

    func testTheTextAndTheImageScanGetTheSameReachableList() throws {
        let discovery = MoveStoreRetirementTests.codeWithoutComments(
            try String(
                contentsOf: URL(fileURLWithPath: #filePath)
                    .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                    .appendingPathComponent("Sources/TurboSparkApp/State/AppModel+ModelDiscovery.swift"),
                encoding: .utf8))
        XCTAssertTrue(discovery.contains("customPaths: reachable"))
        XCTAssertTrue(discovery.contains("scanImageInstalls(in: reachable)"))
        XCTAssertFalse(discovery.contains("customPaths: customDirs"))
        XCTAssertFalse(discovery.contains("scanImageInstalls(in: customDirs)"))
    }

    // MARK: - The trigger

    func testAMountBumpsTheVolumeChangeCount() async {
        let state = makeState()
        XCTAssertEqual(state.volumeChangeCount, 0)
        await state.handle(.mounted(LibraryFixtures.volume("Backup")))
        XCTAssertEqual(state.volumeChangeCount, 1)
    }

    func testAnUnmountAndARenameBumpIt() async {
        let state = makeState()
        await state.handle(.didUnmount(LibraryFixtures.volume("T7")))
        XCTAssertEqual(state.volumeChangeCount, 1)
        await state.handle(.renamed(old: LibraryFixtures.volume("T7"), new: LibraryFixtures.volume("T7-Fast")))
        XCTAssertEqual(state.volumeChangeCount, 2)
    }

    func testAnAttemptedUnmountDoesNotBumpIt() async {
        // `willUnmount` fires for an eject the user then cancels: nothing left.
        let state = makeState()
        await state.handle(.willUnmount(LibraryFixtures.volume("T7")))
        XCTAssertEqual(state.volumeChangeCount, 0)
    }

    func testTheCountIsPublishedSoAViewObservingTheStateIsRedrawn() async {
        let state = makeState()
        var changes = 0
        let subscription = state.objectWillChange.sink { changes += 1 }
        await state.handle(.mounted(LibraryFixtures.volume("Backup")))
        XCTAssertGreaterThanOrEqual(changes, 1)
        subscription.cancel()
    }
}

extension ScanFolderStatus {
    fileprivate var isNotConnected: Bool {
        if case .notConnected = self { return true }
        return false
    }
}
