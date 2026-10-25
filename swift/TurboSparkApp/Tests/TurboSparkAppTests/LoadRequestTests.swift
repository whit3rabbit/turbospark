import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.7 (requirements 3.2, 5.4, 5.5): the one choke point every interactive
/// load goes through.
///
/// A warned model can be CHOSEN in a picker (it is available), so choosing it must
/// not silently load: `AppModel.requestLoad` asks once, in one shared
/// confirmation, and loads only when the user says "Load anyway". A model that is
/// not available is never loaded by any path. The loads that cannot ask (launch
/// restoration, a favorite's auto-start, Humanize, the load that follows an
/// install) keep loading as they did, minus the unavailable ones.
///
/// "Load started" is observed through `AppModel.loadObserver`, which fires where
/// the engine open begins. The open itself fails on the fixtures' folders, which
/// do not exist, and that failure is not what these cases are about.
final class LoadRequestTests: LibraryStateTestCase {
    private let gib: Int = 1 << 30

    private var slowDrive: ModelLibrary { LibraryFixtures.external("slow-id", name: "Slow") }
    private var midDrive: ModelLibrary { LibraryFixtures.external("mid-id", name: "Mid") }
    private var fastDrive: ModelLibrary { LibraryFixtures.external("fast-id", name: "Fast") }
    private var awayDrive: ModelLibrary { LibraryFixtures.external("away-id", name: "Away", available: false) }

    private func libraryRow(
        _ alias: String, in lib: ModelLibrary, bytes: UInt64 = 7_000_000_000, family: String = "qwen",
        availability: LibraryAvailability = .available
    ) -> InstalledModel {
        let root = lib.root ?? "/Volumes/\(lib.name)/TurboSpark"
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "\(root)/text/\(alias).gturbo", family: family,
            installBytes: bytes, library: lib.id, libraryName: lib.name,
            installId: "\(lib.id):text/\(alias)", availability: availability, locationKind: .library,
            connectName: availability == .notConnected ? lib.name : nil)
    }

    private var local: InstalledModel { LibraryFixtures.row("local") }
    /// A dense model on a USB 2.0 drive: the slow-load warning.
    private var dense: InstalledModel { libraryRow("dense", in: slowDrive) }
    /// A mixture-of-experts model far above memory on a 5 Gb/s drive: the streaming warning.
    private var streaming: InstalledModel {
        libraryRow("streaming", in: midDrive, bytes: 10_000 * UInt64(gib), family: "gemma4")
    }
    private var quick: InstalledModel { libraryRow("quick", in: fastDrive) }
    private var away: InstalledModel { libraryRow("away", in: awayDrive, availability: .notConnected) }
    private var gone: InstalledModel { libraryRow("gone", in: fastDrive, availability: .missing) }

    final class Loads {
        var aliases: [String] = []
    }

    /// A model over a state that knows five libraries, with the picker list as
    /// the engine would hand it over: only what is available.
    private func makeLoadModel() async -> (model: AppModel, loads: Loads) {
        probe.setProvider { url in
            switch url.standardizedFileURL.pathComponents.dropFirst(2).first {
            case "Slow": return DriveConnection(kind: .usb, bitsPerSecond: 480_000_000, volumeUUID: "s", isLocal: true)
            case "Mid": return DriveConnection(kind: .usb, bitsPerSecond: 5_000_000_000, volumeUUID: "m", isLocal: true)
            case "test": return DriveConnection(kind: .internalDrive, bitsPerSecond: nil, volumeUUID: "i", isLocal: true)
            default: return DriveConnection(kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "f", isLocal: true)
            }
        }
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.slowDrive, self.midDrive, self.fastDrive, self.awayDrive]
            $0.installed = [self.local, self.dense, self.streaming, self.quick, self.away, self.gone]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        model.enableLMStudioDetection = false
        model.customModelDirectories = []
        model.installed = [local, dense, streaming, quick]
        let loads = Loads()
        model.loadObserver = { loads.aliases.append($0.alias) }
        return (model, loads)
    }

    private func settle() async { await letTasksStart() }

    // MARK: - Open

    func testAnOpenModelLoadsAtOnce() async {
        let (model, loads) = await makeLoadModel()

        let outcome = model.requestLoad(quick)
        let started = await eventually { loads.aliases == ["quick"] }

        XCTAssertEqual(outcome, .started)
        XCTAssertTrue(started, "\(loads.aliases)")
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertEqual(model.selected?.alias, "quick")
    }

    func testAThisMacModelLoadsAtOnce() async {
        let (model, loads) = await makeLoadModel()

        XCTAssertEqual(model.requestLoad(local), .started)
        let started = await eventually { loads.aliases == ["local"] }
        XCTAssertTrue(started)
        XCTAssertNil(model.pendingLoadConfirmation)
    }

    // MARK: - Warned

    func testAWarnedModelAsksFirstAndDoesNotLoad() async {
        let (model, loads) = await makeLoadModel()

        let outcome = model.requestLoad(dense)
        await settle()

        XCTAssertEqual(outcome, .needsConfirmation)
        XCTAssertEqual(loads.aliases, [], "choosing a warned model must not silently load it")
        XCTAssertFalse(model.opening)
        XCTAssertNotEqual(model.selected?.alias, "dense", "the selection waits for the answer too")
        let pending = model.pendingLoadConfirmation
        XCTAssertEqual(pending?.model.alias, "dense")
        XCTAssertEqual(pending?.warning, .slowDrive(estimateSeconds: 175))
        XCTAssertNil(model.error)
    }

    func testTheStreamingWarningIsItsOwnPrompt() async {
        let (model, loads) = await makeLoadModel()

        XCTAssertEqual(model.requestLoad(streaming), .needsConfirmation)
        await settle()

        XCTAssertEqual(model.pendingLoadConfirmation?.warning, .streamsFromDrive)
        XCTAssertEqual(loads.aliases, [])
    }

    func testConfirmingLoadsTheModelOnce() async {
        let (model, loads) = await makeLoadModel()
        model.requestLoad(dense)

        model.confirmPendingLoad()
        let started = await eventually { loads.aliases == ["dense"] }
        await settle()

        XCTAssertTrue(started)
        XCTAssertEqual(loads.aliases, ["dense"], "one load, and no second prompt")
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertEqual(model.selected?.alias, "dense")
    }

    func testCancellingLoadsNothingAndAsksNoMore() async {
        let (model, loads) = await makeLoadModel()
        model.requestLoad(dense)

        model.cancelPendingLoad()
        await settle()

        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertEqual(loads.aliases, [])
        XCTAssertNotEqual(model.selected?.alias, "dense")
        // A second confirm with nothing pending is nothing.
        model.confirmPendingLoad()
        await settle()
        XCTAssertEqual(loads.aliases, [])
    }

    func testLoadAnywayRunsTheSameLoadAsLoadAndNeverAsks() async {
        let (model, loads) = await makeLoadModel()

        // The normal Load of an open model...
        XCTAssertEqual(model.requestLoad(quick), .started)
        _ = await eventually { loads.aliases == ["quick"] }
        let normalSelection = model.selected?.alias

        // ...and Load anyway on a warned one: the same funnel, no prompt.
        model.loadAnyway(dense)
        let started = await eventually { loads.aliases == ["quick", "dense"] }

        XCTAssertTrue(started, "\(loads.aliases)")
        XCTAssertEqual(normalSelection, "quick")
        XCTAssertEqual(model.selected?.alias, "dense", "it selects the model the way Load does")
        XCTAssertNil(model.pendingLoadConfirmation)
    }

    func testSelectModelReportsNotStartedAndPromptsForAWarnedModel() async {
        let (model, loads) = await makeLoadModel()

        XCTAssertFalse(model.selectModel(dense), "no load was started")
        await settle()

        XCTAssertNotNil(model.pendingLoadConfirmation)
        XCTAssertEqual(loads.aliases, [])
    }

    func testTheLoadButtonOfASelectedWarnedModelAsksToo() async {
        let (model, loads) = await makeLoadModel()
        model.selected = dense

        model.loadModel()
        await settle()

        XCTAssertEqual(model.pendingLoadConfirmation?.model.alias, "dense")
        XCTAssertEqual(loads.aliases, [])
    }

    func testOpeningTheChatForAWarnedModelSwitchesPanesOnlyOnceItLoads() async {
        let (model, loads) = await makeLoadModel()
        model.activeSection = .modelHub

        model.openChatWithModel(dense)
        await settle()
        XCTAssertEqual(model.activeSection, .modelHub, "the answer comes first")
        XCTAssertEqual(loads.aliases, [])

        model.confirmPendingLoad()
        _ = await eventually { loads.aliases == ["dense"] }
        XCTAssertEqual(model.activeSection, .chat)
    }

    func testOpeningTheChatForAnOpenModelSwitchesPanesAtOnce() async {
        let (model, loads) = await makeLoadModel()
        model.activeSection = .modelHub

        model.openChatWithModel(quick)
        _ = await eventually { loads.aliases == ["quick"] }

        XCTAssertEqual(model.activeSection, .chat)
    }

    func testANewerRequestReplacesTheOlderPrompt() async {
        let (model, loads) = await makeLoadModel()
        model.requestLoad(dense)
        model.requestLoad(streaming)
        XCTAssertEqual(model.pendingLoadConfirmation?.model.alias, "streaming")

        // And a request that loads at once clears a prompt nobody answered.
        model.requestLoad(quick)
        _ = await eventually { loads.aliases == ["quick"] }
        XCTAssertNil(model.pendingLoadConfirmation)
    }

    func testAChosenFolderInsideAWarnedLibraryAsksToo() async {
        let (model, loads) = await makeLoadModel()

        model.setModelURL(URL(fileURLWithPath: dense.path))
        await settle()

        XCTAssertEqual(model.pendingLoadConfirmation?.model.alias, "dense")
        XCTAssertFalse(model.opening)
        XCTAssertEqual(loads.aliases, [])

        model.confirmPendingLoad()
        XCTAssertTrue(model.opening, "the folder load starts synchronously, as it always did")
        _ = await eventually { loads.aliases == ["dense"] }
    }

    func testAFolderThatIsNotAnInstalledModelLoadsWithoutAsking() async {
        let (model, loads) = await makeLoadModel()

        model.setModelURL(URL(fileURLWithPath: "/tmp/some-folder-nobody-installed"))
        _ = await eventually { loads.aliases == ["some-folder-nobody-installed"] }

        XCTAssertNil(model.pendingLoadConfirmation)
    }

    func testABusyModelIsNotPromptedOrLoaded() async {
        let (model, loads) = await makeLoadModel()
        model.generating = true

        XCTAssertEqual(model.requestLoad(dense), .busy)
        XCTAssertEqual(model.requestLoad(quick), .busy)
        await settle()

        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertEqual(loads.aliases, [])
    }

    // MARK: - Not available

    func testANotConnectedModelIsRefusedWithAStatementNotAnError() async {
        let (model, loads) = await makeLoadModel()

        let outcome = model.requestLoad(away)
        await settle()

        XCTAssertEqual(outcome, .refused(.notConnected(libraryName: "Away")))
        XCTAssertEqual(loads.aliases, [])
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertNil(model.error, "an unplugged drive is never an availability banner (requirement 3.4)")
        XCTAssertEqual(model.activeToast?.message, "Connect Away to load")
    }

    func testAMissingModelIsRefusedToo() async {
        let (model, loads) = await makeLoadModel()

        XCTAssertEqual(model.requestLoad(gone), .refused(.missing))
        await settle()

        XCTAssertEqual(loads.aliases, [])
        XCTAssertNil(model.error)
        XCTAssertEqual(model.activeToast?.message, "The files for this model are missing.")
    }

    func testConfirmingNeverOverridesAModelThatIsNoLongerThere() async {
        let (model, loads) = await makeLoadModel()
        XCTAssertEqual(model.requestLoad(dense), .needsConfirmation)

        // The drive leaves while the question is on screen.
        let leaving = LibraryFixtures.external("slow-id", name: "Slow", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, leaving]
            $0.installed = [self.libraryRow("dense", in: leaving, availability: .notConnected)]
        }
        await model.libraryState.refresh()
        model.confirmPendingLoad()
        await settle()

        XCTAssertEqual(loads.aliases, [], "Load anyway is advice overridden, not a missing drive overridden")
        XCTAssertNil(model.error)
    }

    func testTheGateReadsLiveAvailabilityNotTheRowThePickerHolds() async {
        let (model, loads) = await makeLoadModel()
        // The picker still holds the row as available (the list is stale for a
        // moment after an unplug), the library list says the drive is away.
        let leaving = LibraryFixtures.external("fast-id", name: "Fast", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, leaving]
            $0.installed = [self.libraryRow("quick", in: leaving, availability: .notConnected)]
        }
        await model.libraryState.refresh()
        XCTAssertEqual(model.installed.map(\.alias), ["local", "dense", "streaming", "quick"], "stale on purpose")

        XCTAssertEqual(model.requestLoad(quick), .refused(.notConnected(libraryName: "Fast")))
        await settle()
        XCTAssertEqual(loads.aliases, [])
    }

    // MARK: - Loads that cannot ask

    func testAnEngineOpenOfAnUnavailableModelNeverStarts() async {
        let (model, loads) = await makeLoadModel()

        await model.open(away)
        await model.open(gone)

        XCTAssertEqual(loads.aliases, [], "the funnel under every path refuses a model that is not there")
        XCTAssertFalse(model.opening)
        XCTAssertNil(model.error)
    }

    func testAnEngineOpenOfAWarnedModelLoadsAsItDidWithoutAPrompt() async {
        // The load that follows an install cannot ask: nobody is looking at a
        // dialog. It loads as it does today.
        let (model, loads) = await makeLoadModel()

        await model.open(dense)

        XCTAssertEqual(loads.aliases, ["dense"])
        XCTAssertNil(model.pendingLoadConfirmation)
    }

    func testRestoringTheSelectionNeverLoadsAndNeverAsks() async {
        let (model, loads) = await makeLoadModel()
        let warnedFirst = [dense, local]
        model.catalogInstalledReader = { warnedFirst }
        model.selected = nil

        model.refreshModels()
        await settle()

        XCTAssertEqual(model.selected?.alias, "dense", "restoring a selection picks a row")
        XCTAssertEqual(loads.aliases, [], "and loads nothing")
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertNil(model.activeToast, "no prompt, no toast")
    }

    func testRefreshingWhileADriveIsAwayNeverLoadsAnything() async {
        let (model, loads) = await makeLoadModel()
        model.selected = away
        model.refreshModels()
        await settle()

        XCTAssertEqual(loads.aliases, [])
        XCTAssertNil(model.pendingLoadConfirmation)
        XCTAssertNil(model.error)
        XCTAssertNotEqual(model.selected?.alias, "away")
    }

    // MARK: - The confirmation's words

    func testTheConfirmationSaysTheWarningAndOffersLoadAnyway() async {
        let (model, _) = await makeLoadModel()
        model.requestLoad(dense)
        let pending = try! XCTUnwrap(model.pendingLoadConfirmation)

        XCTAssertEqual(pending.title(language: .english), "Load dense anyway?")
        XCTAssertEqual(
            pending.message(language: .english),
            LoadGateText.text(for: .slowDrive(estimateSeconds: 175), language: .english))
        XCTAssertEqual(pending.confirmTitle(language: .english), "Load anyway")
    }

    func testTheDialogIsMountedOnceAtTheRootAndPresentsThePendingRequest() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        let rootView = RawModelDeleteScanTests.code(
            of: try String(contentsOf: root.appendingPathComponent("App/RootView.swift"), encoding: .utf8))
            .joined(separator: "\n")
        XCTAssertTrue(rootView.contains("LoadAnywayConfirmation(model: model)"))

        // And it is the only place that presents it.
        var presenters: [String] = []
        if let walker = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil) {
            for case let url as URL in walker where url.pathExtension == "swift" {
                let code = RawModelDeleteScanTests.code(of: try String(contentsOf: url, encoding: .utf8))
                    .joined(separator: "\n")
                if code.contains("pendingLoadConfirmation"), code.contains("confirmationDialog") {
                    presenters.append(url.lastPathComponent)
                }
            }
        }
        XCTAssertEqual(presenters, ["LoadAnywayConfirmation.swift"])
    }
}
