import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A Humanize engine that does nothing, so the workspace can hold a model.
private final class IdleHumanizeEngine: HumanizeEngine, @unchecked Sendable {
    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

/// Task 6.4, the BINDING delete rewiring from the 4.6 review: every engine row
/// is deleted through its install id off the main actor, the raw
/// `removeItem(model.path)` fallback is gone, the resident session is kept until
/// the engine has answered, and a not-connected row offers neither Delete nor
/// "Remove from list" (the engine forgets only with the library connected).
///
/// **THE "NEVER TOUCHES THE FILE SYSTEM" CHECKS USE A REAL FOLDER.** The row's
/// recorded path is a folder in a temp directory with a file in it. The fake
/// engine deletes nothing, so any app code that removes that path itself (the
/// old fallback) is visible as a missing file.
@MainActor
final class ModelDeleteRoutingTests: LibraryStateTestCase {
    private var recordedFolder: URL!
    private var weightsFile: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        recordedFolder = directory.appendingPathComponent("text/qwen.gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: recordedFolder, withIntermediateDirectories: true)
        weightsFile = recordedFolder.appendingPathComponent("model_weights.bin")
        try Data("weights".utf8).write(to: weightsFile)
    }

    // MARK: Fixtures

    private func libraryRow(
        alias: String = "qwen", availability: LibraryAvailability = .available
    ) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: recordedFolder.path, family: "qwen",
            installBytes: 4_000_000_000, library: t7.id, libraryName: t7.name,
            installId: "\(t7.id):text/\(alias)", availability: availability, locationKind: .library,
            connectName: availability == .notConnected ? t7.name : nil)
    }

    private func thisMacRow(alias: String = "qwen") -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: recordedFolder.path, family: "qwen",
            installBytes: 4_000_000_000, library: "internal", libraryName: "This Mac",
            installId: "internal:text/\(alias)", availability: .available, locationKind: .thisMac)
    }

    /// The memberwise default the app's own scanners build for another app's
    /// folder: no library, This Mac kind, a placeholder id.
    private func scannedRow(alias: String = "lmstudio-model") -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "lm studio/\(alias)", path: recordedFolder.path, family: "qwen",
            installBytes: 1_000)
    }

    /// A model wired to a fake engine that applies writes, with `rows` listed.
    private func makeDeletingModel(rows: [InstalledModel]) async -> (AppModel, LibraryState) {
        service.with {
            $0.appliesWrites = true
            $0.installed = rows
        }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        return (model, state)
    }

    private func assertFolderUntouched(
        _ message: String = "the app must not remove a recorded path itself",
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertTrue(
            FileManager.default.fileExists(atPath: weightsFile.path), message, file: file, line: line)
    }

    private func start(_ model: AppModel, delete row: InstalledModel) async {
        await model.startModelDelete(row)?.value
    }

    // MARK: Text rows

    func testALibraryRowDeletesThroughItsInstallIdAndNeverTouchesTheFileSystem() async {
        let row = libraryRow()
        let (model, state) = await makeDeletingModel(rows: [row])
        ModelOrganizationStore.shared.toggleFavorite(row)
        defer { ModelOrganizationStore.shared.removeMetadata(for: row) }
        XCTAssertTrue(ModelOrganizationStore.shared.isFavorite(row))

        await start(model, delete: row)

        XCTAssertEqual(
            service.deleteCalls,
            [FakeModelLibraryService.DeleteCall(kind: .text, installId: row.installId, onMainThread: false)])
        assertFolderUntouched()
        XCTAssertFalse(
            state.allInstalled.contains { $0.installId == row.installId },
            "the library list is read again, so the row leaves the list")
        XCTAssertEqual(model.activeToast?.style, .info)
        XCTAssertEqual(model.activeToast?.message, AppStrings.localized("Deleted model 'qwen'"))
        XCTAssertFalse(
            ModelOrganizationStore.shared.isFavorite(row), "a deleted model's notes and tags go with it")
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
    }

    func testAThisMacEngineRowAlsoDeletesThroughItsInstallId() async {
        let row = thisMacRow()
        let (model, _) = await makeDeletingModel(rows: [row])

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls.map(\.installId), ["internal:text/qwen"])
        XCTAssertEqual(service.deleteCalls.map(\.kind), [.text])
        assertFolderUntouched()
    }

    func testAScannedRowNeverDeletesAFileAndNeverReachesTheEngine() async {
        let row = scannedRow()
        let (model, _) = await makeDeletingModel(rows: [])
        model.selected = row

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [], "a row the app did not install is not the engine's")
        assertFolderUntouched("a scanned folder is never deleted")
        XCTAssertTrue(
            ModelOrganizationStore.shared.isExcludedFromScan(path: row.path),
            "it leaves the list and stays out of the next scan")
        // The refresh that follows selects the first listed model, so the check
        // is that the removed row is not the one left selected.
        XCTAssertNotEqual(model.selected?.path, row.path, "a removed row is not left selected")
        XCTAssertEqual(model.activeToast?.style, .info)
        ModelOrganizationStore.shared.restoreToScan(path: row.path)
    }

    func testAMissingRowCanBeDeletedAndTheEngineTreatsItAsForgetOnly() async {
        let row = libraryRow(availability: .missing)
        let (model, state) = await makeDeletingModel(rows: [row])

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls.map(\.installId), [row.installId])
        XCTAssertEqual(service.deleteCalls.map(\.kind), [.text])
        assertFolderUntouched()
        XCTAssertTrue(state.allInstalled.isEmpty)
    }

    // MARK: Image rows

    func testAnImageRowDeletesThroughTheImageInstallId() async {
        let image = ImageInstalledModel(
            alias: "z-image", modelID: "owner/z", revision: "r", path: recordedFolder.path,
            width: 1024, height: 1024, schedulerSteps: 9, library: "internal",
            libraryName: "This Mac", installId: "internal:image/z-image")
        let (model, _) = await makeDeletingModel(rows: [])
        model.managedImageModels = [image]
        model.imageModels = [image]

        await model.startImageModelDelete(image)?.value

        XCTAssertEqual(
            service.deleteCalls,
            [FakeModelLibraryService.DeleteCall(
                kind: .image, installId: "internal:image/z-image", onMainThread: false)])
        assertFolderUntouched()
    }

    func testAScannedImageFolderIsOnlyExcludedAndNeverDeleted() async {
        let image = ImageInstalledModel(
            alias: "scanned", modelID: "owner/z", revision: "r", path: recordedFolder.path,
            width: 512, height: 512, schedulerSteps: 4)
        let (model, _) = await makeDeletingModel(rows: [])

        await model.startImageModelDelete(image)?.value

        XCTAssertEqual(service.deleteCalls, [])
        assertFolderUntouched("a scanned image folder is not the app's to delete")
        ModelOrganizationStore.shared.restoreToScan(path: image.path)
    }

    // MARK: Order and threading

    func testTheModelStaysSelectedWhileTheEngineDeleteRunsOffTheMainActor() async {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        let gate = DispatchSemaphore(value: 0)
        service.with { $0.deleteGate = gate }
        defer { gate.signal() }
        model.selected = row
        model.modelPathText = row.path

        let task = model.startModelDelete(row)
        let parked = await eventually { self.service.deleteCalls.count == 1 }

        XCTAssertTrue(parked, "the engine call must start without blocking the main actor")
        XCTAssertEqual(model.selected?.installId, row.installId, "the session is kept until the engine answers")
        XCTAssertEqual(model.modelPathText, row.path)
        XCTAssertTrue(model.modelsBeingChanged.contains(row.installId))
        XCTAssertEqual(service.deleteCalls.first?.onMainThread, false)

        let second = model.startModelDelete(row)
        XCTAssertNil(second, "a second delete of a row being deleted is refused")
        XCTAssertEqual(service.deleteCalls.count, 1)

        gate.signal()
        await task?.value
        XCTAssertNotEqual(
            model.selected?.installId, row.installId, "success clears the deleted model's selection")
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
    }

    // MARK: Failures keep the model

    func testBusyKeepsTheModelSelectedAndPostsTheBusySentenceAsAWarning() async throws {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        service.with {
            $0.deleteError = TurboSparkError(code: .busy, message: "busy: a model is installing")
        }
        model.selected = row
        model.modelPathText = row.path
        ModelOrganizationStore.shared.toggleFavorite(row)
        defer { ModelOrganizationStore.shared.removeMetadata(for: row) }

        await start(model, delete: row)

        XCTAssertEqual(model.selected?.installId, row.installId, "a refused delete leaves the model loaded")
        XCTAssertEqual(model.modelPathText, row.path)
        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertEqual(
            model.activeToast?.message,
            ModelDeleteText.failure(
                TurboSparkError(code: .busy, message: "busy: x"), alias: "qwen", libraryName: "T7"))
        XCTAssertNil(model.error, "a refusal is a toast, not an availability banner")
        XCTAssertTrue(ModelOrganizationStore.shared.isFavorite(row), "its notes and tags stay")
        assertFolderUntouched()
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
    }

    func testEveryEngineCodeHasItsOwnWarningSentence() async {
        let row = libraryRow()
        let cases: [(TurboSparkError, String)] = [
            (TurboSparkError(code: .busy, message: "busy: an install runs"), "busy"),
            (TurboSparkError(code: .invalidArgument, message: "use_install_id: use the id"), "generic"),
            (TurboSparkError(code: .invalidArgument, message: "not_connected: T7 is away"), "notConnected"),
            (TurboSparkError(code: .invalidArgument, message: "unknown_install: no such install"), "unknown"),
            (TurboSparkError(code: .invalidArgument, message: "io: disk full"), "io"),
            (TurboSparkError(code: .invalidArgument, message: "library_read_only: newer"), "readOnly"),
            // The audio device's busy status has no library prefix: it is not
            // the library `busy`, and it keeps the engine's own reason.
            (TurboSparkError(code: .busy, message: "the audio device is busy"), "uncoded"),
        ]
        var seen: [String: String] = [:]
        for (error, label) in cases {
            let (model, _) = await makeDeletingModel(rows: [row])
            service.with { $0.deleteError = error }
            await start(model, delete: row)
            let message = model.activeToast?.message ?? "none"
            XCTAssertEqual(model.activeToast?.style, .warning, label)
            XCTAssertEqual(
                message, ModelDeleteText.failure(error, alias: "qwen", libraryName: "T7"), label)
            XCTAssertNil(model.error, label)
            if let earlier = seen[label] { XCTAssertEqual(earlier, message, label) }
            seen[label] = message
        }
        XCTAssertEqual(
            Set(seen.values).count, Set(seen.keys).count,
            "each distinct cause reads differently: \(seen)")
        XCTAssertTrue(seen["uncoded"]?.contains("audio device is busy") ?? false)
        XCTAssertFalse(
            seen["uncoded"] == seen["busy"],
            "the audio device's busy status is not the library busy")
    }

    // MARK: Not-connected rows

    func testANotConnectedRowCannotBeDeletedAndTheRowOffersNoRemoval() async {
        let row = libraryRow(availability: .notConnected)
        let (model, state) = await makeDeletingModel(rows: [row])
        model.selected = row

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [], "Delete is never offered for a drive that is away")
        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertEqual(
            model.activeToast?.message, ModelDeleteText.notConnected(alias: "qwen", libraryName: "T7"))
        XCTAssertEqual(model.selected?.installId, row.installId)
        assertFolderUntouched()

        // The engine forgets a row only with its library connected (its marker
        // would keep listing the model), so the row offers no removal either.
        let rowModel = InstalledModelRowModel(
            model: row, library: state.library(for: row), connection: nil, measuredMBps: nil)
        XCTAssertFalse(rowModel.offersDelete)
        XCTAssertFalse(rowModel.offersRemoveFromList)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "the row stays listed")
    }

    func testRemoveFromListSaysWhyWhenTheEngineRefusesAnOfflineLibrary() async {
        let row = libraryRow(availability: .notConnected)
        let (model, state) = await makeDeletingModel(rows: [row])
        service.with {
            $0.forgetError = TurboSparkError(
                code: .invalidArgument, message: "not_connected: the library T7 is not connected")
        }

        await model.startRemoveFromList(row)?.value

        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertEqual(
            model.activeToast?.message, ModelDeleteText.notConnected(alias: "qwen", libraryName: "T7"))
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "the row stays")
    }

    func testAMissingRowCanBeRemovedFromTheList() async {
        let row = libraryRow(availability: .missing)
        let (model, state) = await makeDeletingModel(rows: [row])

        await model.startRemoveFromList(row)?.value

        XCTAssertEqual(service.deleteCalls.map(\.kind), [.forget])
        XCTAssertTrue(state.allInstalled.isEmpty)
        assertFolderUntouched()
    }

    func testAScannedRowHasNothingToForget() async {
        let row = scannedRow()
        let (model, _) = await makeDeletingModel(rows: [])

        let task = model.startRemoveFromList(row)
        await task?.value

        XCTAssertEqual(service.deleteCalls, [], "a placeholder id names no registry install")
    }

    // MARK: The guards before the delete

    func testATurnInFlightRefusesTheDeleteBeforeTheEngine() async {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        model.generating = true

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [])
        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertTrue(model.activeToast?.message.contains("while a turn is running") ?? false)
    }

    func testALoadInFlightRefusesTheDeleteBeforeTheEngine() async {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        model.opening = true
        defer { model.opening = false }

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [])
        XCTAssertTrue(model.activeToast?.message.contains("while a model is loading") ?? false)
    }

    func testAServedModelRefusesTheDeleteBeforeTheEngine() async throws {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        let json = """
            {"port":9090,"host":"::1","modelId":"","models":["qwen"],"imageModels":[],
             "audioModels":[],"authEnabled":false,"uptimeSeconds":1}
            """
        model.serverInfo = try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [])
        XCTAssertTrue(model.activeToast?.message.contains("attached to the running server") ?? false)
    }

    func testABusyHumanizeWorkspaceRefusesTheDeleteBeforeTheEngine() async {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        let workspace = model.humanizeWorkspace
        workspace.installedPath = { row.path }
        workspace.openEngine = { _ in
            try await Task.sleep(nanoseconds: 60_000_000_000)
            return IdleHumanizeEngine()
        }
        workspace.load()
        XCTAssertTrue(workspace.isBusy)
        defer { workspace.unload() }

        await start(model, delete: row)

        XCTAssertEqual(service.deleteCalls, [])
        XCTAssertEqual(
            model.activeToast?.message, AppStrings.localized("Stop Humanize before deleting its model."))
    }

    func testHumanizeKeepsItsModelUntilTheDeleteSucceeds() async {
        let row = libraryRow()
        let (model, _) = await makeDeletingModel(rows: [row])
        let workspace = model.humanizeWorkspace
        workspace.installedPath = { row.path }
        workspace.openEngine = { _ in IdleHumanizeEngine() }
        workspace.load()
        let loaded = await eventually { workspace.engineLoaded }
        XCTAssertTrue(loaded)

        service.with {
            $0.deleteError = TurboSparkError(code: .busy, message: "busy: an install runs")
        }
        await start(model, delete: row)
        XCTAssertTrue(workspace.engineLoaded, "a refused delete leaves the Humanize model loaded")

        service.with { $0.deleteError = nil }
        await start(model, delete: row)
        XCTAssertFalse(workspace.engineLoaded, "a successful delete releases it")
    }
}
