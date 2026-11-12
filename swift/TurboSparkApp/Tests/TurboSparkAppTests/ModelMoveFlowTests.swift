import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A Humanize engine that does nothing, so the workspace can hold a model.
private final class MoveIdleHumanizeEngine: HumanizeEngine, @unchecked Sendable {
    func humanize(
        _ draft: String, options: HumanizeOptions
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { $0.finish() }
    }
    func cancel() {}
}

/// Task 6.11 (requirements 7.1 to 7.5 and 8.1): moving a model between libraries
/// from the detail pane, against the fake library service.
///
/// **THE APP SUITES CANNOT MOVE A REAL MODEL.** They link the release engine, where
/// the location overrides the Rust tests use to build fixture libraries do not
/// exist, so every case drives `FakeModelLibraryService.move`. The fake records the
/// call (install id, destination, the free-space figure, the thread) and a script
/// answers it, so what these cases pin is what the APP does around the engine: what
/// it refuses before calling, what it hands over, what it keeps when the engine
/// says no, and what it rewrites when the engine says yes.
///
/// **THE APP NEVER TOUCHES THE FILES.** The recorded folders are real, and a case
/// that would have the app move or remove one shows as a missing file.
@MainActor
final class ModelMoveFlowTests: LibraryStateTestCase {
    private let gb: UInt64 = 1_000_000_000
    private let spare = LibraryFixtures.external("spare-id", name: "Spare", volume: "Spare")

    // MARK: Fixtures

    private func textRow(
        _ alias: String, library: ModelLibrary? = nil, bytes: UInt64 = 4_000_000_000
    ) -> InstalledModel {
        let id = library?.id ?? "internal"
        let isInternal = library?.isInternal ?? true
        let root = isInternal ? "/Users/test/.turbospark/models" : (library?.root ?? "/Volumes/\(id)/TurboSpark")
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "\(root)/text/\(alias).gturbo", family: "qwen",
            installBytes: bytes, library: id, libraryName: library?.name ?? "This Mac",
            installId: "\(id):text/\(alias)", availability: .available,
            locationKind: isInternal ? .thisMac : .library)
    }

    /// A model on the fake engine that applies moves, with `rows` listed, the
    /// installed list reading from the fake the way the engine reads its registry,
    /// and no folder scans.
    private func makeMover(
        rows: [InstalledModel], extraLibraries: [ModelLibrary] = []
    ) async -> (AppModel, LibraryState) {
        service.with {
            $0.libraries.append(contentsOf: extraLibraries)
            $0.installed = rows
            $0.appliesMoves = true
        }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        model.enableLMStudioDetection = false
        model.customModelDirectories = []
        model.downloadHistoryWritable = false
        model.modelDownloads = []
        let service = self.service!
        model.catalogInstalledReader = { service.with { $0.installed.filter { $0.availability == .available } } }
        model.thisMacFreeSpaceReader = { 100 * 1_000_000_000 }
        model.moveProgressInterval = 0
        model.refreshModels()
        return (model, state)
    }

    private func subject(_ row: InstalledModel) throws -> MoveSubject {
        try XCTUnwrap(MoveSubject.make(text: row))
    }

    private func openSheet(
        _ model: AppModel, _ row: InstalledModel, to destination: String? = "t7-id"
    ) throws -> ModelMoveSheetModel {
        let sheet = try XCTUnwrap(model.presentMoveSheet(for: subject(row)))
        sheet.selection = destination
        return sheet
    }

    private func finishing(
        _ kind: String = "moved", stale: String? = nil, path: String? = nil
    ) -> FakeModelLibraryService.MoveScript {
        { call, progress in
            progress(0, 100)
            progress(100, 100)
            let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
            return try LibraryFixtures.moveOutcome(
                kind, installId: "\(call.libraryId):text/\(alias)", staleMarker: stale, path: path)
        }
    }

    private func failing(_ code: String) -> FakeModelLibraryService.MoveScript {
        { _, progress in
            progress(0, 100)
            throw LibraryFixtures.moveError(code)
        }
    }

    private func forget(_ rows: InstalledModel...) {
        for row in rows { ModelOrganizationStore.shared.removeMetadata(for: row) }
    }

    // MARK: Success

    private func assertSuccess(
        kind: String, stale: String? = nil, path: String? = nil, file: StaticString = #filePath,
        line: UInt = #line
    ) async throws {
        let alias = "mv-ok-\(kind)"
        let row = textRow(alias)
        let moved = LibraryFixtures.row(alias, library: t7)
        defer { forget(row, moved) }
        let (model, state) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing(kind, stale: stale, path: path) }

        let result = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        guard case .moved(let text) = result else {
            return XCTFail("a move the engine finished is never a failure: \(result)", file: file, line: line)
        }
        XCTAssertEqual(text.headline, "Moved \(alias) to T7.", file: file, line: line)
        XCTAssertEqual(text.detail == nil, kind == "moved", file: file, line: line)

        // The install is in the destination, under the id the engine sent.
        let listed = try XCTUnwrap(
            state.allInstalled.first { $0.alias == alias }, "the row is still listed", file: file, line: line)
        XCTAssertEqual(listed.installId, "t7-id:text/\(alias)", file: file, line: line)
        XCTAssertEqual(listed.library, "t7-id", file: file, line: line)
        XCTAssertEqual(listed.locationKind, .library, file: file, line: line)
        XCTAssertEqual(listed.libraryName, "T7", file: file, line: line)
        XCTAssertEqual(state.allInstalled.filter { $0.alias == alias }.count, 1, "one row, not two", file: file, line: line)
        XCTAssertEqual(
            model.loadableModels.first { $0.alias == alias }?.installId, "t7-id:text/\(alias)",
            "the pickers follow it", file: file, line: line)

        // The move is over in every way the app tracks it.
        XCTAssertNil(model.activeMove, file: file, line: line)
        XCTAssertTrue(model.modelsBeingChanged.isEmpty, file: file, line: line)
        XCTAssertNil(model.error, "a finished move is not an app error", file: file, line: line)
        XCTAssertEqual(model.activeToast?.style, .info, file: file, line: line)
        XCTAssertEqual(model.activeToast?.message, text.spoken, file: file, line: line)
        XCTAssertFalse(model.isInstallOrMoveRunning, file: file, line: line)

        XCTAssertEqual(service.moveCalls.count, 1, file: file, line: line)
        XCTAssertEqual(service.moveCalls.first?.installId, row.installId, file: file, line: line)
        XCTAssertEqual(service.moveCalls.first?.libraryId, "t7-id", file: file, line: line)
        XCTAssertEqual(service.moveCalls.first?.freeBytes, Int64(900 * gb), "the destination's measured free space", file: file, line: line)
        XCTAssertEqual(service.moveCalls.first?.onMainThread, false, "the engine call runs off the main actor", file: file, line: line)
        XCTAssertEqual(model.lastMovedModel?.oldInstallId, row.installId, file: file, line: line)
        XCTAssertEqual(model.lastMovedModel?.newInstallId, listed.installId, file: file, line: line)
        XCTAssertEqual(model.lastMovedModel?.newPath, listed.path, file: file, line: line)
    }

    func testAMovedOutcomePutsTheInstallInTheDestination() async throws {
        try await assertSuccess(kind: "moved")
    }

    func testAStaleSourceMarkerIsAMoveThatSucceededNotAFailure() async throws {
        try await assertSuccess(kind: "source_marker_stale", stale: "Old Drive")
    }

    func testAFailedSourceCleanupIsAMoveThatSucceededNotAFailure() async throws {
        try await assertSuccess(
            kind: "source_cleanup_failed", stale: "Old Drive", path: "/Users/test/.turbospark/models/text/left-over.gturbo")
    }

    func testTheSheetSaysWhatHappenedAndStaysOpenOnSuccess() async throws {
        let row = textRow("mv-sheet-ok")
        defer { forget(row, LibraryFixtures.row("mv-sheet-ok", library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        service.with {
            $0.moveScript = finishing("source_cleanup_failed", stale: "Old", path: "/Users/test/.turbospark/models/text/x")
        }
        let sheet = try openSheet(model, row)

        await sheet.move()?.value

        guard case .finished(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(text.headline, "Moved mv-sheet-ok to T7.")
        XCTAssertTrue(text.detail?.contains("/Users/test/.turbospark/models/text/x") ?? false)
        XCTAssertTrue(sheet.canDismiss)
        XCTAssertTrue(model.moveSheet === sheet, "the sheet stays up so the user reads it")
    }

    // MARK: Identity (requirement 8.1)

    func testMetadataStillUnderTheOldPathKeyIsCarriedWithThePreMovePath() async throws {
        let alias = "mv-legacy"
        let row = textRow(alias)
        let newRow = LibraryFixtures.row(alias, library: t7)
        defer { forget(row, newRow) }
        let store = ModelOrganizationStore.shared
        // An entry no view has read since an older build wrote it: it still sits
        // under `alias::path`, which is how the path before the move spells it.
        let legacyKey = ModelIdentityKey.metadataPathKey(alias: alias, path: row.path)
        store.setNickname("Daily driver", for: alias, path: row.path)
        store.addTag("fast", for: alias, path: row.path)
        XCTAssertNotNil(store.metadataByModelKey[legacyKey])
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        let carried = try XCTUnwrap(store.metadataByModelKey["t7-id:text/\(alias)"], "moved to the new id")
        XCTAssertEqual(carried.nickname, "Daily driver")
        XCTAssertEqual(carried.tags, ["fast"])
        XCTAssertNil(store.metadataByModelKey[legacyKey], "and the old key is gone")
        XCTAssertEqual(store.nickname(newRow), "Daily driver")
    }

    func testMetadataUnderTheOldInstallIdMovesToTheNewOne() async throws {
        let alias = "mv-ided"
        let row = textRow(alias)
        let newRow = LibraryFixtures.row(alias, library: t7)
        defer { forget(row, newRow) }
        let store = ModelOrganizationStore.shared
        store.setFavorite(true, for: row)
        store.setNotes("good at code", for: row)
        store.setNickname("Coder", for: row)
        store.addTag("work", for: row)
        store.setReasoningDefault("high", for: row)
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertNil(store.metadataByModelKey[row.installId], "nothing is left under the old id")
        let carried = try XCTUnwrap(store.metadataByModelKey[newRow.installId])
        XCTAssertTrue(carried.isFavorite)
        XCTAssertEqual(carried.notes, "good at code")
        XCTAssertEqual(carried.nickname, "Coder")
        XCTAssertEqual(carried.tags, ["work"])
        XCTAssertEqual(carried.reasoningDefault, "high")
    }

    func testTheReKeyIsGivenTheOldIdTheNewIdAndThePreMovePath() async throws {
        let alias = "mv-spy"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        struct Call: Equatable {
            let old: String
            let new: String
            let legacyPath: String?
        }
        var calls: [Call] = []
        model.metadataRekey = { old, new, legacyPath in calls.append(Call(old: old, new: new, legacyPath: legacyPath)) }
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        // The path BEFORE the move: an entry no view has read since an older build wrote
        // it sits under `alias::oldPath`, and the store finds it only by that path.
        XCTAssertEqual(
            calls, [Call(old: "internal:text/\(alias)", new: "t7-id:text/\(alias)", legacyPath: row.path)])
        XCTAssertNotEqual(calls.first?.legacyPath, model.loadableModels.first { $0.alias == alias }?.path)

        // A move that fails or is cancelled hands the store nothing.
        calls = []
        service.with { $0.moveScript = failing("io") }
        let moved = try XCTUnwrap(model.loadableModels.first { $0.alias == alias })
        _ = await model.runMove(
            subject: try XCTUnwrap(MoveSubject.make(text: moved)), destinationId: "internal", size: .known(4 * gb))
        XCTAssertEqual(calls, [])
    }

    func testAReasoningDefaultStillInTheSettingsFileFollowsTheModel() async throws {
        let alias = "mv-reasoning"
        let row = textRow(alias)
        let newRow = LibraryFixtures.row(alias, library: t7)
        defer { forget(row, newRow) }
        let (model, _) = await makeMover(rows: [row])
        // The table the reasoning level lived in before it joined the record, keyed
        // by the raw path: after a move that key names a folder that is gone.
        model.modelReasoningDefaults[row.path] = "low"
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(ModelOrganizationStore.shared.reasoningDefault(for: newRow), "low")
        XCTAssertNil(model.modelReasoningDefaults[row.path], "the legacy entry is retired, not left behind")
    }

    func testAFailedMoveLeavesTheMetadataAlone() async throws {
        let alias = "mv-keep"
        let row = textRow(alias)
        defer { forget(row) }
        let store = ModelOrganizationStore.shared
        store.setFavorite(true, for: row)
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = failing("io") }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertNotNil(store.metadataByModelKey[row.installId])
        XCTAssertTrue(store.isFavorite(row))
        XCTAssertNil(store.metadataByModelKey["t7-id:text/\(alias)"])
        XCTAssertNil(model.lastMovedModel)
    }

    func testAServerFavoriteThatNamedTheModelStillResolvesToIt() async throws {
        let alias = "mv-fav"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, state) = await makeMover(rows: [row])
        let byId = ServerFavorite(
            name: "by id", host: "127.0.0.1", port: 8080, modelPaths: [row.path],
            modelInstallIds: [row.installId], contextTokens: 4096)
        // A favorite saved before identities existed: a path and no id.
        let byPath = ServerFavorite(
            name: "by path", host: "127.0.0.1", port: 8081, modelPaths: ["/other.gturbo", row.path],
            contextTokens: 4096)
        let unrelated = ServerFavorite(
            name: "other", host: "127.0.0.1", port: 8082, modelPaths: ["/other.gturbo"],
            modelInstallIds: ["internal:text/other"], contextTokens: 4096)
        model.serverFavorites = [byId, byPath, unrelated]
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        let newId = "t7-id:text/\(alias)"
        XCTAssertEqual(model.serverFavorites[0].modelInstallIds, [newId], "the stored id follows the model")
        XCTAssertEqual(model.serverFavorites[1].modelInstallIds, ["", newId], "a path-only entry gains the id, aligned")
        XCTAssertEqual(model.serverFavorites[2], unrelated, "another model's favorite is not touched")
        let rows = model.serverFavoriteRows()
        for favorite in model.serverFavorites.prefix(2) {
            let resolved = favorite.resolve(in: rows)
            let found = resolved.compactMap { outcome -> InstalledModel? in
                if case .available(let model) = outcome { return model }
                return nil
            }
            XCTAssertTrue(found.contains { $0.installId == newId }, "\(favorite.name) still resolves: \(resolved)")
        }
        _ = state
    }

    func testAFavoriteThatIsNotRekeyedStillResolvesThroughTheLibraryIndependentPart() async throws {
        // The fallback 5.9 added: even with the stored id untouched, the same
        // `<modality>/<alias>` in another library is the model that moved.
        let moved = LibraryFixtures.row("mv-part", library: t7)
        let favorite = ServerFavorite(
            name: "f", host: "h", port: 1, modelPaths: ["/old/path"], modelInstallIds: ["internal:text/mv-part"],
            contextTokens: 1)
        guard case .available(let found)? = favorite.resolve(in: [moved]).first else {
            return XCTFail("did not resolve")
        }
        XCTAssertEqual(found.installId, moved.installId)
    }

    func testTheSelectedRowFollowsTheMoveByItsNewInstallId() async throws {
        let alias = "mv-selected"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let other = textRow("mv-other")
        // The moving row is SECOND: when the row the selection held is gone,
        // `refreshModels` moves the selection to the first loadable row, which here is
        // another model. Only the explicit re-selection puts it back on the moved one.
        let (model, _) = await makeMover(rows: [other, row])
        model.selected = row
        model.modelPathText = row.path
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(model.selected?.installId, "t7-id:text/\(alias)", "it is selected at its new place")
        XCTAssertEqual(model.selected?.library, "t7-id")
        XCTAssertEqual(model.modelPathText, model.selected?.path)
        XCTAssertNotEqual(model.selected?.path, row.path)
        XCTAssertNotEqual(model.selected?.installId, other.installId, "not the first model that happens to be listed")
    }

    func testAMoveNeverLoadsOrOpensAModel() async throws {
        let alias = "mv-no-load"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        model.selected = row
        var loads: [String] = []
        model.loadObserver = { loads.append($0.alias) }
        service.with { $0.moveScript = finishing() }
        let sheet = try openSheet(model, row)

        await sheet.move()?.value

        XCTAssertEqual(loads, [], "moving the selected model does not open it")
        XCTAssertNil(model.session)
        XCTAssertFalse(model.opening)
        XCTAssertNil(model.pendingLoadConfirmation)
    }

    func testAnUnselectedRowLeavesTheSelectionAlone() async throws {
        let row = textRow("mv-moving")
        let other = textRow("mv-stays")
        defer { forget(row, LibraryFixtures.row("mv-moving", library: t7)) }
        let (model, _) = await makeMover(rows: [row, other])
        model.selected = other
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(model.selected?.installId, other.installId)
    }

    // MARK: Progress

    func testProgressStartsIndeterminateAndReachesTheSheet() async throws {
        let alias = "mv-progress"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        let start = MoveGate()
        let middle = MoveGate()
        let end = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                await start.wait()
                progress(0, 200)
                await middle.wait()
                progress(100, 200)
                await end.wait()
                progress(200, 200)
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let sheet = try openSheet(model, row)

        let task = sheet.move()
        let started = await eventually { model.activeMove != nil && self.service.moveCalls.count == 1 }
        XCTAssertTrue(started)
        let first = try XCTUnwrap(model.activeMove)
        XCTAssertEqual(first.installId, row.installId)
        XCTAssertEqual(first.alias, alias)
        XCTAssertEqual(first.sourceName, "This Mac")
        XCTAssertEqual(first.destinationName, "T7")
        XCTAssertEqual(first.phase, .starting, "indeterminate until the first callback")
        XCTAssertNil(first.fraction)
        XCTAssertEqual(sheet.stage, .moving)
        XCTAssertTrue(model.modelsBeingChanged.contains(row.installId))
        XCTAssertTrue(model.isInstallOrMoveRunning)

        start.open()
        let zero = await eventually { model.activeMove?.hasProgress == true }
        XCTAssertTrue(zero)
        XCTAssertEqual(model.activeMove?.total, 200)
        XCTAssertEqual(model.activeMove?.done, 0)
        XCTAssertEqual(model.activeMove?.phase, .copying)

        middle.open()
        let half = await eventually { model.activeMove?.done == 100 }
        XCTAssertTrue(half)
        XCTAssertEqual(model.activeMove?.fraction, 0.5)
        XCTAssertEqual(sheet.progress, model.activeMove, "the sheet draws the app's progress")

        end.open()
        await task?.value
        XCTAssertNil(model.activeMove)
        XCTAssertNil(sheet.progress)
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
    }

    func testAFloodOfCallbacksIsThrottledButTheLastFigureLands() async throws {
        let alias = "mv-flood"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        model.moveProgressInterval = 3_600
        let observed = Observed()
        service.with {
            $0.moveScript = { call, progress in
                for done in stride(from: UInt64(0), through: 1_000, by: 1) { progress(done, 1_000) }
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let cancellable = model.$activeMove.sink { if let move = $0 { observed.add(move.done) } }
        defer { cancellable.cancel() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertLessThan(observed.values.count, 20, "1001 callbacks must not become 1001 redraws")
        XCTAssertTrue(observed.values.contains(0), "the first figure starts the bar")
        XCTAssertTrue(observed.values.contains(1_000), "the last figure is never dropped")
    }

    final class Observed: @unchecked Sendable {
        private let lock = NSLock()
        private var seen: [UInt64] = []
        func add(_ value: UInt64) { lock.lock(); seen.append(value); lock.unlock() }
        var values: [UInt64] { lock.lock(); defer { lock.unlock() }; return seen }
    }

    // MARK: Cancel

    func testCancelAsksTheEngineFromAnotherThreadAndLeavesTheOriginal() async throws {
        let alias = "mv-cancel"
        let row = textRow(alias)
        defer { forget(row) }
        let (model, state) = await makeMover(rows: [row])
        model.selected = row
        let parked = MoveGate()
        service.with {
            $0.moveScript = { _, progress in
                progress(0, 100)
                await parked.wait()
                throw LibraryFixtures.moveError("cancelled")
            }
            $0.cancelHandler = { parked.open(); return true }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        let running = await eventually { model.activeMove?.hasProgress == true }
        XCTAssertTrue(running)

        sheet.cancel()

        XCTAssertEqual(model.activeMove?.cancelRequested, true, "Cancel says so at once")
        XCTAssertEqual(model.activeMove?.phase, .cancelling)
        XCTAssertEqual(service.cancelCalls.count, 1, "the engine is asked, not just the screen")
        let cancel = try XCTUnwrap(service.cancelCalls.first)
        let move = try XCTUnwrap(service.moveCalls.first)
        XCTAssertFalse(move.onMainThread)
        XCTAssertNotEqual(cancel.threadID, move.threadID, "the cancel is made from a thread other than the move's")
        XCTAssertEqual(sheet.stage, .moving, "the sheet waits for the engine's answer")
        XCTAssertFalse(sheet.canDismiss)

        // The fake engine answers a cancel by letting the move go. Letting it go here
        // too means a cancel that never reached the seam fails above instead of hanging.
        parked.open()
        await task?.value

        guard case .failed(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(text.kind, .cancelled)
        XCTAssertEqual(text.text, "Move cancelled. The original was not changed.")
        XCTAssertNil(model.activeMove)
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "the original is where it was")
        XCTAssertEqual(model.selected?.installId, row.installId, "and still selected")
        XCTAssertNil(model.error)
        XCTAssertNil(model.lastMovedModel)
        XCTAssertTrue(sheet.canDismiss)
    }

    func testACancelBeforeTheEngineRegisteredTheMoveIsRepeatedOnTheFirstProgress() async throws {
        let alias = "mv-early-cancel"
        let row = textRow(alias)
        defer { forget(row) }
        let (model, _) = await makeMover(rows: [row])
        let enter = MoveGate()
        let parked = MoveGate()
        service.with {
            $0.cancelAnswer = false  // "no move is running yet"
            $0.moveScript = { _, progress in
                await enter.wait()
                progress(0, 100)
                await parked.wait()
                throw LibraryFixtures.moveError("cancelled")
            }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        let entered = await eventually { self.service.moveCalls.count == 1 }
        XCTAssertTrue(entered)

        sheet.cancel()
        XCTAssertEqual(service.cancelCalls.count, 1)
        service.with { $0.cancelHandler = { parked.open(); return true } }
        enter.open()

        let repeated = await eventually { self.service.cancelCalls.count == 2 }
        XCTAssertTrue(repeated, "the first callback proves the engine is running, so the cancel is sent again")
        parked.open()
        await task?.value
        guard case .failed(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(text.kind, .cancelled)
    }

    func testACancelThatArrivesTooLateIsAMoveThatSucceeded() async throws {
        let alias = "mv-late-cancel"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, state) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
            $0.cancelHandler = { parked.open(); return true }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove?.hasProgress == true }
        sheet.cancel()
        parked.open()
        await task?.value

        guard case .finished = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(state.allInstalled.first?.installId, "t7-id:text/\(alias)")
    }

    // MARK: Failures

    func testEveryFailureCodeIsOneSentenceInsideTheSheetAndTheOriginalStays() async throws {
        var seen: [String: String] = [:]
        for code in ModelMoveTextTests.failureCodes + ["cancelled", "from_the_future"] {
            let alias = "mv-fail"
            let row = textRow(alias)
            let (model, state) = await makeMover(rows: [row])
            service.with {
                $0.moveScript = failing(code)
                $0.moveCalls = []
            }
            model.selected = row
            let listBefore = service.listCalls
            let sheet = try openSheet(model, row)

            await sheet.move()?.value

            guard case .failed(let text) = sheet.stage else { return XCTFail("\(code): \(sheet.stage)") }
            let kept = ModelMoveText.originalKept(language: AppStrings.currentLanguage)
            if code == "cancelled" {
                XCTAssertEqual(text.kind, .cancelled, code)
            } else {
                XCTAssertEqual(text.kind, .failed, code)
                XCTAssertTrue(text.text.hasSuffix(kept), "\(code): ends with the reassurance -> \(text.text)")
            }
            XCTAssertNil(model.error, "\(code): a failed move is not an app error")
            XCTAssertNil(model.activeToast, "\(code): it is said inside the sheet, not as a banner or a toast")
            XCTAssertNil(model.activeMove, code)
            XCTAssertTrue(model.modelsBeingChanged.isEmpty, code)
            XCTAssertTrue(model.moveSheet === sheet, "\(code): the sheet stays open to retry or close")
            XCTAssertTrue(sheet.canDismiss, code)
            XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "\(code): the original is listed")
            XCTAssertEqual(model.loadableModels.map(\.installId), [row.installId], "\(code): and loadable")
            XCTAssertEqual(model.selected?.installId, row.installId, "\(code): and still selected")
            XCTAssertGreaterThan(service.listCalls, listBefore, "\(code): the library state is read again")
            XCTAssertNil(model.lastMovedModel, code)
            if let earlier = seen[text.text] { XCTFail("\(code) reads like \(earlier)") }
            seen[text.text] = code
        }
    }

    func testAFailedMoveCanBeRetried() async throws {
        let alias = "mv-retry"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, state) = await makeMover(rows: [row])
        service.with { $0.moveScript = failing("io") }
        let sheet = try openSheet(model, row)
        await sheet.move()?.value
        guard case .failed = sheet.stage else { return XCTFail("\(sheet.stage)") }

        sheet.retry()
        XCTAssertEqual(sheet.stage, .choosing)
        service.with { $0.moveScript = finishing() }
        await sheet.move()?.value

        guard case .finished = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(state.allInstalled.first?.installId, "t7-id:text/\(alias)")
        XCTAssertEqual(service.moveCalls.count, 2)
    }

    func testADriveLostMidMoveNamesTheOneThatLeft() async throws {
        let row = textRow("mv-unplug")
        let (model, _) = await makeMover(rows: [row])
        let service = self.service!
        let away = LibraryFixtures.external("t7-id", name: "T7", available: false)
        service.with {
            $0.moveScript = { _, progress in
                progress(0, 100)
                // The drive is gone by the time the engine says so.
                service.with { $0.libraries = [LibraryFixtures.internalLibrary, away] }
                throw LibraryFixtures.moveError("not_connected")
            }
        }
        let sheet = try openSheet(model, row)

        await sheet.move()?.value

        guard case .failed(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(text.text, "T7 was disconnected during the move. The original was kept.")
    }

    func testAHungLibraryListingDoesNotHoldTheFailureBack() async throws {
        let row = textRow("mv-hung")
        let (model, _) = await makeMover(rows: [row])
        model.moveRefreshPatience = .milliseconds(80)
        service.with { $0.moveScript = failing("not_connected") }
        // From now on a listing hangs, as one does on a drive that was just pulled.
        let hung = DispatchSemaphore(value: 0)
        service.with { $0.listGate = hung }
        defer { hung.signal() }
        let sheet = try openSheet(model, row)

        let started = Date()
        await sheet.move()?.value

        XCTAssertLessThan(Date().timeIntervalSince(started), 2.5, "the answer does not wait on the listing")
        guard case .failed(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(
            text.text, "A drive was disconnected during the move. The original was kept.",
            "with no listing to say which drive, the sentence names none")
        XCTAssertNil(model.activeMove)
    }

    func testTheEnginesInsufficientSpaceNeverQuotesFiguresThatContradictIt() async throws {
        // The app saw 900 GB free and a 4 GB model and let the move start; the engine
        // measured the folder itself and said there is not enough. Quoting "T7 has only
        // 900 GB free, but this model needs 4 GB" would be a sentence that disproves
        // itself, so the figures are left out and the plain sentence is said.
        let row = textRow("mv-space")
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = failing("insufficient_space") }
        let sheet = try openSheet(model, row)

        await sheet.move()?.value

        guard case .failed(let text) = sheet.stage else { return XCTFail("\(sheet.stage)") }
        XCTAssertEqual(text.text, "T7 does not have enough free space for this model. The original was kept.")
        XCTAssertFalse(text.text.contains(MetricFormat.storage(900 * gb)), text.text)
        XCTAssertFalse(text.text.contains(MetricFormat.storage(4 * gb)), text.text)
    }

    // MARK: Refusals before copying

    private func assertRefused(
        _ expected: MoveRefusal, row: InstalledModel, prepare: (AppModel) -> Void,
        file: StaticString = #filePath, line: UInt = #line
    ) async throws {
        let (model, state) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }
        prepare(model)

        let result = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(result, .refused(expected), file: file, line: line)
        XCTAssertEqual(service.moveCalls, [], "a refused move never reaches the engine", file: file, line: line)
        XCTAssertNil(model.activeMove, file: file, line: line)
        XCTAssertTrue(model.modelsBeingChanged.isEmpty, file: file, line: line)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], file: file, line: line)
        XCTAssertNil(model.error, file: file, line: line)
    }

    func testAModelLoadedInChatIsNeverMoved() async throws {
        let row = textRow("mv-loaded")
        try await assertRefused(.loadedInChat, row: row) { model in
            model.moveHolderProbe.loadedChatPath = { _ in row.path }
        }
    }

    func testTheSelectedModelWithALiveSessionIsNeverMoved() async throws {
        let row = textRow("mv-selected-live")
        try await assertRefused(.loadedInChat, row: row) { model in
            model.selected = row
            model.moveHolderProbe.hasChatSession = { _ in true }
        }
    }

    func testASelectedModelWithNoSessionMayBeMoved() async throws {
        let alias = "mv-selected-idle"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        model.selected = row
        service.with { $0.moveScript = finishing() }

        let result = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        guard case .moved = result else { return XCTFail("\(result)") }
    }

    func testAModelAttachedToTheRunningServerIsNeverMoved() async throws {
        let row = textRow("mv-served")
        try await assertRefused(.attachedToServer, row: row) { model in
            model.moveHolderProbe.servedIDs = { _ in [row.alias] }
        }
        // By the path the server was given, too.
        try await assertRefused(.attachedToServer, row: row) { model in
            model.moveHolderProbe.servedIDs = { _ in [row.path] }
        }
    }

    func testAModelHumanizeIsUsingIsNeverMoved() async throws {
        let row = textRow("mv-humanize")
        try await assertRefused(.usedByHumanize, row: row) { model in
            model.moveHolderProbe.humanizeUses = { _, path in path == row.path }
        }
    }

    /// The probe's live readers, with nothing replaced: the ids the server serves
    /// (read from `serverInfo`) and a Humanize engine that holds the model.
    func testTheLiveServerAndHumanizeReadersRefuseTheMove() async throws {
        let served = textRow("mv-served-live")
        let (model, _) = await makeMover(rows: [served])
        service.with { $0.moveScript = finishing() }
        let json = """
            {"port":9090,"host":"::1","modelId":"","models":["mv-served-live"],"imageModels":[],
             "audioModels":[],"authEnabled":false,"uptimeSeconds":1}
            """
        model.serverInfo = try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
        var result = await model.runMove(subject: try subject(served), destinationId: t7.id, size: .known(4 * gb))
        XCTAssertEqual(result, .refused(.attachedToServer))
        model.serverInfo = nil

        let held = textRow("mv-humanize-live")
        service.with { $0.installed.append(held) }
        let workspace = model.humanizeWorkspace
        workspace.installedPath = { held.path }
        workspace.openEngine = { _ in MoveIdleHumanizeEngine() }
        workspace.load()
        let loaded = await eventually { workspace.engineLoaded }
        XCTAssertTrue(loaded)
        defer { workspace.unload() }
        result = await model.runMove(subject: try subject(held), destinationId: t7.id, size: .known(4 * gb))
        XCTAssertEqual(result, .refused(.usedByHumanize))
        XCTAssertEqual(service.moveCalls, [])
    }

    func testARunningTurnALoadingModelAndAnInstallEachRefuseTheMove() async throws {
        try await assertRefused(.turnRunning, row: textRow("mv-turn")) { $0.generating = true }
        try await assertRefused(.turnRunning, row: textRow("mv-submit")) { $0.submitting = true }
        try await assertRefused(.modelOpening, row: textRow("mv-open")) { $0.opening = true }
        try await assertRefused(.installRunning, row: textRow("mv-install")) { $0.isInstallingModel = true }
        try await assertRefused(.installRunning, row: textRow("mv-image-install")) { $0.isInstallingImageModel = true }
    }

    func testAnUnmeasurableDestinationIsRefusedAndNeverPassedAsANegativeNumber() async throws {
        let row = textRow("mv-unmeasured")
        let (model, state) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }
        // T7's capacity cannot be read: the library state has no figure for it.
        let unreadable = makeState(freeSpace: { url in url.path == "/Users/test/.turbospark/models" ? 100 * 1_000_000_000 : nil })
        await unreadable.refresh()
        model.libraryState = unreadable
        XCTAssertNil(unreadable.freeBytes["t7-id"])

        let result = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(result, .refused(.freeSpaceUnknown(name: "T7")))
        XCTAssertEqual(service.moveCalls, [], "no call, so no -1 either")
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId])
    }

    func testTooLittleFreeSpaceIsRefusedBeforeCopyingWithBothFigures() async throws {
        let row = textRow("mv-tight")
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }
        let tight = makeState(freeSpace: { url in url.path == "/Volumes/T7/TurboSpark" ? 3_000_000_000 : 100 * 1_000_000_000 })
        await tight.refresh()
        model.libraryState = tight
        let sheet = try openSheet(model, row)

        let plan = sheet.plan(language: .english)
        XCTAssertEqual(plan.refusal, .insufficientSpace(required: 4 * gb, available: 3 * gb, name: "T7"))
        XCTAssertEqual(plan.requiredText, MetricFormat.storage(4 * gb))
        XCTAssertEqual(plan.availableText, MetricFormat.storage(3 * gb))
        XCTAssertFalse(plan.canMove)

        let task = sheet.move()
        XCTAssertNil(task, "Move does nothing while the plan is refused")
        XCTAssertEqual(service.moveCalls, [])
    }

    func testADestinationThatLeftAfterItWasChosenIsRefusedByName() async throws {
        let row = textRow("mv-left")
        let (model, _) = await makeMover(rows: [row], extraLibraries: [spare])
        service.with { $0.moveScript = finishing() }
        let sheet = try openSheet(model, row, to: spare.id)
        // Spare is unplugged after it was chosen.
        service.with { state in
            state.libraries = state.libraries.map { $0.id == "spare-id" ? LibraryFixtures.external("spare-id", name: "Spare", available: false) : $0 }
        }
        await model.libraryState.refresh()

        let result = await model.runMove(subject: try subject(row), destinationId: spare.id, size: .known(4 * gb))

        XCTAssertEqual(result, .refused(.destinationNotConnected(name: "Spare")))
        XCTAssertEqual(service.moveCalls, [])
        XCTAssertEqual(sheet.plan(language: .english).refusal, .destinationNotConnected(name: "Spare"))
    }

    func testTheSameLibraryIsRefusedAndNotCalled() async throws {
        let row = textRow("mv-same")
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }

        let result = await model.runMove(subject: try subject(row), destinationId: "internal", size: .known(4 * gb))

        XCTAssertEqual(result, .refused(.sameLibrary))
        XCTAssertEqual(service.moveCalls, [])
    }

    func testASecondMoveIsRefusedWhileOneRuns() async throws {
        let first = textRow("mv-first")
        let second = textRow("mv-second")
        defer { forget(first, LibraryFixtures.row("mv-first", library: t7)) }
        let (model, _) = await makeMover(rows: [first, second])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let sheet = try openSheet(model, first)
        let task = sheet.move()
        _ = await eventually { model.activeMove?.hasProgress == true }

        let result = await model.runMove(subject: try subject(second), destinationId: t7.id, size: .known(4 * gb))

        XCTAssertEqual(result, .refused(.moveRunning))
        XCTAssertEqual(service.moveCalls.count, 1, "the engine was asked once")
        XCTAssertEqual(model.activeMove?.installId, first.installId, "and the running move is untouched")
        parked.open()
        await task?.value
    }

    // MARK: What a running move blocks

    func testAMoveIsAnInstallOrMoveForEveryControlThatAsks() async throws {
        let alias = "mv-busy"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, state) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        XCTAssertFalse(model.isInstallOrMoveRunning)
        let libraries = ModelLibrariesController(libraryState: state, model: model)
        let offers = LibraryOfferController(libraryState: state, isBusy: { model.isInstallOrMoveRunning })
        XCTAssertFalse(libraries.isChangeBlocked)
        XCTAssertFalse(offers.isChangeBlocked)

        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove != nil }

        XCTAssertTrue(model.isInstallOrMoveRunning, "a move is running")
        XCTAssertTrue(libraries.isChangeBlocked, "the libraries section disables every library change")
        XCTAssertTrue(offers.isChangeBlocked, "and so does the offer banner")
        await libraries.add(path: "/Volumes/Anywhere")
        XCTAssertEqual(libraries.message?.text, LibraryActionText.busy(language: AppStrings.currentLanguage))
        XCTAssertEqual(service.addCalls, [], "the engine is not asked to add while a move runs")

        parked.open()
        await task?.value
        XCTAssertFalse(model.isInstallOrMoveRunning)
        XCTAssertFalse(libraries.isChangeBlocked)
        XCTAssertFalse(offers.isChangeBlocked)
    }

    func testDeleteAndRemoveFromListRefuseOnTheMovingRow() async throws {
        let alias = "mv-guard"
        let row = textRow(alias)
        let other = textRow("mv-guard-other")
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row, other])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove != nil }

        XCTAssertTrue(model.modelsBeingChanged.contains(row.installId))
        XCTAssertNil(model.startModelDelete(row), "Delete is refused on the row that is moving")
        XCTAssertNil(model.startRemoveFromList(row), "and so is Remove from list")
        XCTAssertEqual(service.deleteCalls, [], "neither reached the engine")

        parked.open()
        await task?.value
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
    }

    func testInstallsRefuseWhileAMoveRunsWithTheBusySentence() async throws {
        let alias = "mv-installs"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        var started: [ModelDownload.Request] = []
        model.modelDownloadStartOverride = { started.append($0) }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove != nil }

        let request = ModelDownload.Request.catalog(alias: "some-model")
        XCTAssertFalse(model.canQueueModelDownload(request), "the Download buttons are disabled")
        XCTAssertFalse(model.enqueueModelDownload(request))
        XCTAssertEqual(model.activeToast?.style, .warning)
        XCTAssertEqual(model.activeToast?.message, LibraryActionText.busy(language: AppStrings.currentLanguage))
        XCTAssertTrue(model.modelDownloads.isEmpty, "nothing was queued")
        XCTAssertEqual(service.installCalls, [])

        // A row that was queued before the move waits for it.
        model.modelDownloads = [ModelDownload(request: .catalog(alias: "queued-before"), status: .queued)]
        model.startNextModelDownloadIfPossible()
        XCTAssertTrue(started.isEmpty, "the queue does not start an install under a move")

        parked.open()
        await task?.value
        XCTAssertEqual(started, [.catalog(alias: "queued-before")], "and starts when the move ends")
    }

    // MARK: Quit, vault lock, profile switch

    func testShutdownDuringAMoveCancelsItAndClearsTheActiveMove() async throws {
        let alias = "mv-quit"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, state) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { _, progress in
                progress(0, 100)
                await parked.wait()
                throw LibraryFixtures.moveError("cancelled")
            }
            $0.cancelHandler = { parked.open(); return true }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove?.hasProgress == true }

        model.shutdown()

        XCTAssertEqual(service.cancelCalls.count, 1, "the engine is told to stop")
        XCTAssertNil(model.activeMove, "a quit never leaves a move on screen")
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
        XCTAssertFalse(model.isInstallOrMoveRunning)
        parked.open()
        await task?.value
        XCTAssertNil(model.activeMove)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "the original stays")
        XCTAssertNil(model.activeToast, "a model that was shut down says nothing more")
        XCTAssertNil(ModelOrganizationStore.shared.metadataByModelKey["t7-id:text/\(alias)"])
    }

    func testAShutdownCancelThatFoundNoMoveIsSentAgainOnTheFirstFigure() async throws {
        // The quit arrives between the engine call and the engine registering the move:
        // the cancel answers "no move is running". The shutdown bumped the generation, so
        // the first figure of the move is a stale one, and it must still send the cancel
        // again or the copy runs to its end after the app was told to stop it.
        let alias = "mv-quit-early"
        let row = textRow(alias)
        defer { forget(row) }
        let (model, state) = await makeMover(rows: [row])
        let enter = MoveGate()
        let parked = MoveGate()
        service.with {
            $0.cancelAnswer = false  // "no move is running yet"
            $0.moveScript = { _, progress in
                await enter.wait()
                progress(0, 100)
                await parked.wait()
                throw LibraryFixtures.moveError("cancelled")
            }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        let entered = await eventually { self.service.moveCalls.count == 1 }
        XCTAssertTrue(entered)

        model.shutdown()
        XCTAssertEqual(service.cancelCalls.count, 1, "the shutdown asked, and found no move")
        XCTAssertNil(model.activeMove)
        service.with { $0.cancelHandler = { parked.open(); return true } }
        enter.open()

        let repeated = await eventually { self.service.cancelCalls.count == 2 }
        XCTAssertTrue(repeated, "the first figure proves the engine is running, so the shutdown's cancel is sent again")
        parked.open()
        await task?.value
        XCTAssertEqual(service.cancelCalls.count, 2, "once it was heard, it is not sent a third time")
        XCTAssertNil(model.activeMove)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId], "the original stays")
        XCTAssertNil(model.activeToast, "a model that was shut down says nothing more")
    }

    func testAShutdownCancelIsNotRepeatedOnceTheEngineCallHasEnded() async throws {
        // The engine runs one move at a time for the whole process. A late figure of a
        // move that has already ended must not cancel whichever move starts after it.
        let row = textRow("mv-quit-ended")
        defer { forget(row) }
        let (model, _) = await makeMover(rows: [row])
        let enter = MoveGate()
        service.with {
            $0.cancelAnswer = false
            $0.moveScript = { _, _ in
                await enter.wait()
                throw LibraryFixtures.moveError("cancelled")
            }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        let entered = await eventually { self.service.moveCalls.count == 1 }
        XCTAssertTrue(entered)
        model.shutdown()
        XCTAssertEqual(service.cancelCalls.count, 1)
        let staleGeneration = model.moveDriver.generation - 1

        enter.open()
        await task?.value
        model.applyMoveProgress(done: 5, total: 10, generation: staleGeneration)

        XCTAssertEqual(service.cancelCalls.count, 1, "the call has ended: nothing is left to cancel")
    }

    func testAMoveClaimedBeforeAShutdownNeverReachesTheEngine() async throws {
        let row = textRow("mv-claimed")
        let (model, state) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }
        guard
            case .started(let ticket) = model.beginMove(
                subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))
        else { return XCTFail("the move was refused") }
        XCTAssertNotNil(model.activeMove, "the move is claimed before the engine is called")
        XCTAssertTrue(model.isInstallOrMoveRunning)
        XCTAssertTrue(model.modelsBeingChanged.contains(row.installId))
        XCTAssertEqual(service.moveCalls, [])

        model.shutdown()
        XCTAssertNil(model.activeMove)
        let result = await model.performMove(ticket)

        XCTAssertEqual(result, .abandoned)
        XCTAssertEqual(service.moveCalls, [], "a move started after the quit could not be cancelled")
        XCTAssertTrue(model.modelsBeingChanged.isEmpty)
        XCTAssertEqual(state.allInstalled.map(\.installId), [row.installId])
    }

    func testAMoveThatFinishesAfterShutdownDoesNotRewriteTheNextProfilesState() async throws {
        let alias = "mv-quit-late"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let store = ModelOrganizationStore.shared
        store.setFavorite(true, for: row)
        let (model, _) = await makeMover(rows: [row])
        model.serverFavorites = [
            ServerFavorite(name: "f", host: "h", port: 1, modelPaths: [row.path], modelInstallIds: [row.installId], contextTokens: 1)
        ]
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let sheet = try openSheet(model, row)
        let task = sheet.move()
        _ = await eventually { model.activeMove?.hasProgress == true }

        model.shutdown()
        parked.open()
        await task?.value

        XCTAssertNotNil(store.metadataByModelKey[row.installId], "the process-wide store is left to the next model")
        XCTAssertEqual(model.serverFavorites.first?.modelInstallIds, [row.installId])
        XCTAssertNil(model.activeMove)
        XCTAssertNil(model.activeToast)
    }

    // MARK: The sheet cannot be dismissed while it moves

    func testTheSheetCannotBeDismissedWhileMoving() async throws {
        let alias = "mv-dismiss"
        let row = textRow(alias)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                await parked.wait()
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        let sheet = try openSheet(model, row)
        XCTAssertTrue(sheet.canDismiss, "choosing a destination can be abandoned")

        let task = sheet.move()
        _ = await eventually { model.activeMove != nil }

        XCTAssertFalse(sheet.canDismiss)
        XCTAssertFalse(sheet.escapeCloses, "Escape does not close it, and does not cancel the move either")
        XCTAssertFalse(model.dismissMoveSheet(), "no code path dismisses it mid-move")
        XCTAssertFalse(sheet.close())
        XCTAssertTrue(model.moveSheet === sheet)
        XCTAssertNotNil(model.activeMove)

        parked.open()
        await task?.value
        XCTAssertTrue(sheet.canDismiss)
        XCTAssertTrue(sheet.close())
        XCTAssertNil(model.moveSheet)
    }

    func testAnIdleSheetClosesAndForgetsItsSubject() async throws {
        let row = textRow("mv-close")
        let (model, _) = await makeMover(rows: [row])
        let sheet = try openSheet(model, row)

        XCTAssertTrue(sheet.escapeCloses)
        XCTAssertTrue(model.dismissMoveSheet())
        XCTAssertNil(model.moveSheet)
    }

    func testOpeningTheSheetForTheSameModelTwiceKeepsTheOneSheet() async throws {
        let row = textRow("mv-twice")
        let (model, _) = await makeMover(rows: [row])
        let first = try openSheet(model, row)
        let second = model.presentMoveSheet(for: try subject(row))
        XCTAssertTrue(first === second)
    }

    // MARK: Sizes that are not recorded

    func testASizeThatIsNotRecordedIsMeasuredOffTheMainActor() async throws {
        let row = textRow("mv-measure", bytes: 0)
        let (model, _) = await makeMover(rows: [row])
        let measured = Observed()
        model.modelSizeReader = { path in
            measured.add(Thread.isMainThread ? 1 : 0)
            return path == row.path ? 5_000_000_000 : 0
        }
        let sheet = try openSheet(model, row)
        XCTAssertEqual(sheet.size, .measuring)
        XCTAssertFalse(sheet.plan(language: .english).canMove, "Move waits for the size")

        await sheet.measuringTask?.value

        XCTAssertEqual(sheet.size, .known(5 * gb))
        XCTAssertEqual(measured.values, [0], "one read, off the main thread")
        XCTAssertTrue(sheet.plan(language: .english).canMove)
    }

    func testASizeThatCannotBeMeasuredIsRefusedNotTreatedAsZero() async throws {
        let row = textRow("mv-unmeasurable", bytes: 0)
        let (model, _) = await makeMover(rows: [row])
        model.modelSizeReader = { _ in 0 }
        let sheet = try openSheet(model, row)

        await sheet.measuringTask?.value

        XCTAssertEqual(sheet.size, .unavailable)
        XCTAssertEqual(sheet.plan(language: .english).refusal, .sizeUnavailable)
        XCTAssertNil(sheet.move())
        XCTAssertEqual(service.moveCalls, [])
    }

    // MARK: Image and audio

    func testAnImageMoveFollowsTheSelectedImageModelToItsNewPath() async throws {
        let image = ImageInstalledModel(
            alias: "z-image", modelID: "owner/z", revision: "r",
            path: "/Users/test/.turbospark/models/image/z-image", width: 1024, height: 1024, schedulerSteps: 9,
            library: "internal", libraryName: "This Mac", installId: "internal:image/z-image")
        let moved = ImageInstalledModel(
            alias: "z-image", modelID: "owner/z", revision: "r",
            path: "/Volumes/T7/TurboSpark/image/z-image", width: 1024, height: 1024, schedulerSteps: 9,
            library: "t7-id", libraryName: "T7", installId: "t7-id:image/z-image")
        let (model, _) = await makeMover(rows: [])
        model.imageInstalledReader = { [moved] in [moved] }
        model.managedImageModels = [image]
        model.imageModels = [image]
        model.imageModelPathText = image.path
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                return try LibraryFixtures.moveOutcome("moved", installId: "t7-id:image/z-image")
            }
        }
        let subject = try XCTUnwrap(MoveSubject.make(image: image, isManaged: true))

        let result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(2 * gb))

        guard case .moved = result else { return XCTFail("\(result)") }
        XCTAssertEqual(service.moveCalls.first?.installId, "internal:image/z-image")
        XCTAssertEqual(model.imageModelPath, moved.path, "the selected image model is selected at its new place")
        XCTAssertEqual(model.lastMovedModel?.newPath, moved.path)
    }

    func testAnImageLoadedInTheImagesWorkspaceOrServedIsNeverMoved() async throws {
        let image = ImageInstalledModel(
            alias: "z-image", modelID: "owner/z", revision: "r",
            path: "/Users/test/.turbospark/models/image/z-image", width: 1, height: 1, schedulerSteps: 1,
            library: "internal", libraryName: "This Mac", installId: "internal:image/z-image")
        let subject = try XCTUnwrap(MoveSubject.make(image: image, isManaged: true))
        let (model, _) = await makeMover(rows: [])
        service.with { $0.moveScript = finishing() }

        model.moveHolderProbe.loadedImagePath = { _ in image.path }
        var result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(gb))
        XCTAssertEqual(result, .refused(.loadedInImages))

        model.moveHolderProbe.loadedImagePath = { _ in nil }
        model.serverImageAttachedModel = image
        result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(gb))
        XCTAssertEqual(result, .refused(.attachedToServer))
        XCTAssertEqual(service.moveCalls, [])
    }

    func testAnAudioModelResidentInAudioOrServedIsNeverMoved() async throws {
        let record = try JSONDecoder().decode(
            AudioInstalledRecord.self,
            from: Data(
                """
                {"alias":"kokoro","path":"/Users/test/.turbospark/models/audio/kokoro","family":"kokoro",
                 "status":"ready","installBytes":300000000}
                """.utf8))
        let subject = try XCTUnwrap(MoveSubject.make(audio: record))
        let (model, _) = await makeMover(rows: [])
        service.with { $0.moveScript = finishing() }

        model.moveHolderProbe.audioResidentPath = { _ in record.path }
        var result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(gb))
        XCTAssertEqual(result, .refused(.residentInAudio))

        model.moveHolderProbe.audioResidentPath = { _ in nil }
        model.serverSpeechAttachedModel = record
        result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(gb))
        XCTAssertEqual(result, .refused(.attachedToServer))
        XCTAssertEqual(service.moveCalls, [])
    }

    func testAnAudioRowInALibraryNamesItsLibraryFromTheLibraryList() async throws {
        // An audio report row carries its library by id only.
        let record = try JSONDecoder().decode(
            AudioInstalledRecord.self,
            from: Data(
                """
                {"library":"t7-id","alias":"kokoro","path":"/Volumes/T7/TurboSpark/audio/kokoro",
                 "family":"kokoro","status":"ready","installBytes":300000000}
                """.utf8))
        let subject = try XCTUnwrap(MoveSubject.make(audio: record))
        XCTAssertNil(subject.libraryName)
        let (model, _) = await makeMover(rows: [])

        let sheet = try XCTUnwrap(model.presentMoveSheet(for: subject))

        XCTAssertEqual(sheet.subject.libraryName, "T7")
        XCTAssertEqual(sheet.plan(language: .english).sourceName, "T7")
        XCTAssertEqual(sheet.plan(language: .english).choices.map(\.id), ["internal"], "This Mac is the other place")
    }

    func testAnAudioMoveAsksTheEngineForTheAudioInstallId() async throws {
        let record = try JSONDecoder().decode(
            AudioInstalledRecord.self,
            from: Data(
                """
                {"alias":"kokoro","path":"/Users/test/.turbospark/models/audio/kokoro","family":"kokoro",
                 "status":"ready","installBytes":300000000}
                """.utf8))
        let subject = try XCTUnwrap(MoveSubject.make(audio: record))
        let (model, _) = await makeMover(rows: [])
        service.with {
            $0.moveScript = { _, progress in
                progress(0, 100)
                return try LibraryFixtures.moveOutcome("moved", installId: "t7-id:audio/kokoro")
            }
        }

        let result = await model.runMove(subject: subject, destinationId: t7.id, size: .known(300_000_000))

        guard case .moved(let text) = result else { return XCTFail("\(result)") }
        XCTAssertEqual(text.headline, "Moved kokoro to T7.")
        XCTAssertEqual(service.moveCalls.first?.installId, "internal:audio/kokoro")
    }

    // MARK: The App Never Touches Files

    func testNoFileIsMovedCopiedOrRemovedByTheApp() async throws {
        let alias = "mv-files"
        let folder = directory.appendingPathComponent("text/\(alias).gturbo", isDirectory: true)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let weights = folder.appendingPathComponent("weights.bin")
        try Data("weights".utf8).write(to: weights)
        let row = InstalledModel(
            alias: alias, repo: "o/\(alias)", path: folder.path, family: "qwen", installBytes: 4_000_000_000,
            library: "internal", libraryName: "This Mac", installId: "internal:text/\(alias)",
            availability: .available, locationKind: .thisMac)
        defer { forget(row, LibraryFixtures.row(alias, library: t7)) }
        let (model, _) = await makeMover(rows: [row])
        service.with { $0.moveScript = finishing() }

        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))
        XCTAssertTrue(FileManager.default.fileExists(atPath: weights.path), "only the engine moves or removes a model")

        service.with { $0.moveScript = failing("io") }
        _ = await model.runMove(subject: try subject(row), destinationId: t7.id, size: .known(4 * gb))
        XCTAssertTrue(FileManager.default.fileExists(atPath: weights.path))
    }
}
