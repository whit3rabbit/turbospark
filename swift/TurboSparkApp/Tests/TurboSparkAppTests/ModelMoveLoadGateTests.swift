import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.11, review round 1: a model that is being moved is never loaded.
///
/// **THE HAZARD** (the state#43 class). The engine deletes the original folder once
/// the copy is verified. `moveAppState` refuses a move for a model that is loaded
/// WHEN THE MOVE STARTS, and nothing else looked at the move afterwards, so a load
/// started during the copy mapped a folder the engine was about to remove.
///
/// Every case here parks the engine inside `move` (a `MoveGate`), so the move is in
/// flight for the whole assertion, and then asks for a load through one of the
/// doors: the interactive choke point, the menu predicate, the net under `open`, the
/// server attach, a chosen folder and a reload. The load is observed where it begins
/// (`AppModel.loadObserver`), which fires before the engine open that fails on these
/// fixtures' folders.
@MainActor
final class ModelMoveLoadGateTests: LibraryStateTestCase {
    private let gb: UInt64 = 1_000_000_000
    private let movingSentence = "This model is being moved. Try again when the move finishes."

    final class Loads {
        var aliases: [String] = []
    }

    // MARK: Fixtures

    private func textRow(_ alias: String) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "/Users/test/.turbospark/models/text/\(alias).gturbo",
            family: "qwen", installBytes: 4_000_000_000, library: "internal", libraryName: "This Mac",
            installId: "internal:text/\(alias)", availability: .available, locationKind: .thisMac)
    }

    private func makeMover(rows: [InstalledModel]) async -> (AppModel, LibraryState, Loads) {
        service.with {
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
        let loads = Loads()
        model.loadObserver = { loads.aliases.append($0.alias) }
        return (model, state, loads)
    }

    private func parkedScript(_ parked: MoveGate, outcome: String = "moved") -> FakeModelLibraryService.MoveScript {
        { call, progress in
            progress(0, 100)
            await parked.wait()
            if outcome != "moved" { throw LibraryFixtures.moveError(outcome) }
            let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
            return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
        }
    }

    private func startMove(
        _ model: AppModel, _ row: InstalledModel
    ) async throws -> (sheet: ModelMoveSheetModel, task: Task<Void, Never>) {
        let sheet = try XCTUnwrap(model.presentMoveSheet(for: XCTUnwrap(MoveSubject.make(text: row))))
        sheet.selection = "t7-id"
        let task = try XCTUnwrap(sheet.move())
        let running = await eventually { model.activeMove?.hasProgress == true }
        XCTAssertTrue(running, "the engine is inside the move")
        return (sheet, task)
    }

    private func forget(_ rows: InstalledModel...) {
        for row in rows { ModelOrganizationStore.shared.removeMetadata(for: row) }
    }

    // MARK: The interactive choke point

    func testARequestToLoadTheMovingModelIsBusyAndLoadsNothing() async throws {
        let row = textRow("lg-busy")
        let other = textRow("lg-busy-other")
        defer { forget(row, other, LibraryFixtures.row("lg-busy", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row, other])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        XCTAssertEqual(model.requestLoad(row), .busy)
        XCTAssertEqual(model.requestLoad(row, purpose: .chat(switchToChat: true)), .busy)
        XCTAssertEqual(model.requestLoad(row, confirmed: true), .busy, "Load anyway does not override a move")
        XCTAssertEqual(model.requestLoad(row, bringWindowForward: true), .busy, "nor does the menu bar")
        await letTasksStart()

        XCTAssertEqual(loads.aliases, [], "the engine open never began")
        XCTAssertFalse(model.opening)
        XCTAssertNil(model.session)
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertEqual(model.activeToast?.message, movingSentence, "a click is answered, not ignored")
        XCTAssertNil(model.error)

        parked.open()
        await task.value
    }

    func testALoadOfADifferentModelStillWorksDuringAMove() async throws {
        let row = textRow("lg-diff")
        let other = textRow("lg-diff-other")
        defer { forget(row, other, LibraryFixtures.row("lg-diff", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row, other])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        XCTAssertEqual(model.requestLoad(other), .started)
        let began = await eventually { loads.aliases == ["lg-diff-other"] }

        XCTAssertTrue(began, "\(loads.aliases)")
        parked.open()
        await task.value
    }

    func testTheWindowBetweenTheClaimAndTheEngineCallRefusesLoadsToo() async throws {
        let row = textRow("lg-claim")
        defer { forget(row, LibraryFixtures.row("lg-claim", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        service.with {
            $0.moveScript = { call, progress in
                progress(0, 100)
                let alias = ModelIdentityKey.alias(inInstallId: call.installId) ?? ""
                return try LibraryFixtures.moveOutcome("moved", installId: "\(call.libraryId):text/\(alias)")
            }
        }
        guard
            case .started(let ticket) = model.beginMove(
                subject: try XCTUnwrap(MoveSubject.make(text: row)), destinationId: t7.id, size: .known(4 * gb))
        else { return XCTFail("the move was refused") }
        XCTAssertEqual(service.moveCalls, [], "the engine has not been called yet")

        XCTAssertEqual(model.requestLoad(row), .busy)
        await letTasksStart()
        XCTAssertEqual(loads.aliases, [])

        _ = await model.performMove(ticket)
    }

    // MARK: The menu predicate

    func testTheLoadChatModelMenuItemIsOffForTheMovingSelectedModel() async throws {
        let row = textRow("lg-menu")
        defer { forget(row, LibraryFixtures.row("lg-menu", library: t7)) }
        let (model, _, _) = await makeMover(rows: [row])
        model.selected = row
        XCTAssertTrue(model.canLoadModel, "the other terms allow a load, so the move is the reason below")
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        XCTAssertFalse(model.canLoadModel)

        parked.open()
        await task.value
    }

    func testTheMenuItemStaysOnForAnotherSelectedModelDuringAMove() async throws {
        let row = textRow("lg-menu-a")
        let other = textRow("lg-menu-b")
        defer { forget(row, other, LibraryFixtures.row("lg-menu-a", library: t7)) }
        let (model, _, _) = await makeMover(rows: [row, other])
        model.selected = other
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        XCTAssertTrue(model.canLoadModel)

        parked.open()
        await task.value
    }

    func testLoadModelFromTheMenuLoadsNothingWhileTheSelectedModelMoves() async throws {
        let row = textRow("lg-menu-load")
        defer { forget(row, LibraryFixtures.row("lg-menu-load", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        model.selected = row
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        model.loadModel()
        await letTasksStart()

        XCTAssertEqual(loads.aliases, [])
        parked.open()
        await task.value
    }

    // MARK: The net under every path that cannot ask

    func testOpenRefusesTheMovingModelWithAToastAndNoSession() async throws {
        let row = textRow("lg-open")
        defer { forget(row, LibraryFixtures.row("lg-open", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        await model.open(row)

        XCTAssertEqual(loads.aliases, [], "the post-install load and a favorite open through here")
        XCTAssertFalse(model.opening)
        XCTAssertNil(model.session)
        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertNil(model.error)

        parked.open()
        await task.value
    }

    func testReloadOfTheMovingSelectedModelLoadsNothing() async throws {
        let row = textRow("lg-reload")
        defer { forget(row, LibraryFixtures.row("lg-reload", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        model.selected = row
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        model.reloadModel()
        await letTasksStart()

        XCTAssertEqual(loads.aliases, [])
        XCTAssertFalse(model.opening)
        parked.open()
        await task.value
    }

    func testTheServerAttachIsRefusedForTheMovingModel() async throws {
        let row = textRow("lg-attach")
        defer { forget(row, LibraryFixtures.row("lg-attach", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        // No server runs here, so the generic busy answer would also be `.busy`: the
        // toast is what tells the move's refusal from that one.
        XCTAssertEqual(model.requestLoad(row, purpose: .serverAttach), .busy)
        XCTAssertEqual(model.activeToast?.message, movingSentence)
        XCTAssertFalse(model.canOpenNow(row), "the net under performServerAttach and a favorite's auto-start")
        await letTasksStart()
        XCTAssertEqual(loads.aliases, [])

        parked.open()
        await task.value
    }

    func testAChosenFolderThatIsTheMovingModelIsRefusedToo() async throws {
        let row = textRow("lg-folder")
        defer { forget(row, LibraryFixtures.row("lg-folder", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)

        model.setModelURL(URL(fileURLWithPath: row.path))
        await letTasksStart()

        XCTAssertEqual(loads.aliases, [])
        XCTAssertFalse(model.opening)
        parked.open()
        await task.value
    }

    // MARK: A load works again when the move ends

    func testAfterASuccessfulMoveTheMovedModelLoads() async throws {
        let row = textRow("lg-after-ok")
        defer { forget(row, LibraryFixtures.row("lg-after-ok", library: t7)) }
        let (model, state, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked) }
        let (_, task) = try await startMove(model, row)
        XCTAssertEqual(model.requestLoad(row), .busy)

        parked.open()
        await task.value

        let moved = try XCTUnwrap(state.allInstalled.first { $0.alias == "lg-after-ok" })
        XCTAssertEqual(moved.installId, "t7-id:text/lg-after-ok")
        XCTAssertEqual(model.requestLoad(moved), .started)
        let began = await eventually { loads.aliases == ["lg-after-ok"] }
        XCTAssertTrue(began, "\(loads.aliases)")
    }

    func testAfterAFailedMoveTheOriginalLoads() async throws {
        let row = textRow("lg-after-fail")
        defer { forget(row, LibraryFixtures.row("lg-after-fail", library: t7)) }
        let (model, _, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with { $0.moveScript = parkedScript(parked, outcome: "io") }
        let (_, task) = try await startMove(model, row)
        XCTAssertEqual(model.requestLoad(row), .busy)

        parked.open()
        await task.value

        XCTAssertNil(model.activeMove)
        XCTAssertEqual(model.requestLoad(row), .started)
        let began = await eventually { loads.aliases == ["lg-after-fail"] }
        XCTAssertTrue(began, "\(loads.aliases)")
    }

    func testAfterACancelledMoveTheOriginalLoads() async throws {
        let row = textRow("lg-after-cancel")
        defer { forget(row) }
        let (model, _, loads) = await makeMover(rows: [row])
        let parked = MoveGate()
        service.with {
            $0.moveScript = parkedScript(parked, outcome: "cancelled")
            $0.cancelHandler = { parked.open(); return true }
        }
        let (sheet, task) = try await startMove(model, row)
        sheet.cancel()
        await task.value

        XCTAssertEqual(model.requestLoad(row), .started)
        let began = await eventually { loads.aliases == ["lg-after-cancel"] }
        XCTAssertTrue(began, "\(loads.aliases)")
    }
}
