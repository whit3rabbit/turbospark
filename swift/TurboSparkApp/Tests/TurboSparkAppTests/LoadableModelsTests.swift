import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.8 (requirement 3.3): one loadable list.
///
/// `AppModel.installed` is the engine's available-only list at the time of the
/// last `refreshModels()`, plus the rows the app scanned from other apps'
/// folders. It goes stale when a drive is unplugged, so a picker reading it can
/// offer a model whose files are gone. The fix is at the root, not per view:
/// `installed` follows `libraryState.allInstalled`, and `loadableModels` is the
/// one list every picker and every code path that starts a load reads.
///
/// Every case runs against `FakeModelLibraryService`; the app suites link the
/// release engine, which cannot build fixture libraries.
final class LoadableModelsTests: LibraryStateTestCase {
    private var spare: ModelLibrary {
        LibraryFixtures.external("spare-id", name: "Spare", available: false)
    }

    /// A row the app built itself for another app's folder: no library, a
    /// placeholder id (`internal:text/<alias>`), always loadable.
    private func scanned(_ alias: String) -> InstalledModel {
        InstalledModel(alias: alias, repo: "lm studio/\(alias)", path: "/lm/\(alias).gguf", family: "qwen")
    }

    // MARK: - The pure list

    func testTheListDropsOfflineAndMissingRowsAndKeepsTheOrderOfTheRest() {
        let live = LibraryFixtures.row("on-t7", library: t7)
        let away = LibraryFixtures.row("on-spare", library: spare, availability: .notConnected)
        let gone = LibraryFixtures.row("gone", library: t7, availability: .missing)
        let local = LibraryFixtures.row("local")

        // `installed` still holds the rows the engine listed before the drive
        // left: the list answers from the LIVE availability, not from them.
        let loadable = LoadableModelList.filter(
            installed: [local, LibraryFixtures.row("on-spare", library: spare), live, LibraryFixtures.row("gone", library: t7)],
            live: [local, live, away, gone])

        XCTAssertEqual(loadable.map(\.alias), ["local", "on-t7"])
    }

    func testAScannedRowStaysAndIsNeverMatchedByItsPlaceholderId() {
        // The placeholder id of a scanned row can equal the id of a This Mac row
        // whose folder is gone: they are two models, and the scanned one is fine.
        let missingThisMac = LibraryFixtures.row("shared-name", availability: .missing)
        let lookalike = scanned("shared-name")
        XCTAssertEqual(lookalike.installId, missingThisMac.installId, "the fixture needs the ids to collide")

        let loadable = LoadableModelList.filter(installed: [lookalike], live: [missingThisMac])

        XCTAssertEqual(loadable.map(\.path), ["/lm/shared-name.gguf"])
    }

    func testARowTheLiveListDoesNotKnowCountsAsAvailable() {
        // Before the first listing has landed, and for a custom folder the engine
        // does not list, the row's own word stands.
        let row = LibraryFixtures.row("local")
        XCTAssertEqual(LoadableModelList.filter(installed: [row], live: []).map(\.alias), ["local"])
        XCTAssertEqual(LoadableModelList.availability(of: row, live: []), .available)
    }

    func testTheLiveRowFollowsTheInstallIdNotTheAlias() {
        // Two libraries cannot share an install id, but a scanned row and a
        // library row can share an alias: only the id decides.
        let onSpare = LibraryFixtures.row("qwen", library: spare, availability: .notConnected)
        let onT7 = LibraryFixtures.row("qwen", library: t7)
        let loadable = LoadableModelList.filter(installed: [onT7, scanned("qwen")], live: [onSpare, onT7])
        XCTAssertEqual(loadable.map(\.path), [onT7.path, "/lm/qwen.gguf"])
    }

    // MARK: - On the model

    /// A model wired to a state, with the folder scans off so `installed` holds
    /// only what the test puts there, and the engine's own list replaced by what
    /// the fake service says is available (as the engine does).
    private func makeFollowingModel(
        _ state: LibraryState, reads: ReadCounter = ReadCounter()
    ) async -> AppModel {
        let model = await makeModel(using: state)
        model.enableLMStudioDetection = false
        model.customModelDirectories = []
        let service = self.service!
        model.catalogInstalledReader = {
            reads.count += 1
            return service.with { $0.installed.filter { $0.availability == .available } }
        }
        return model
    }

    final class ReadCounter { var count = 0 }

    func testLoadableModelsLeavesOutAnOfflineLibraryWhileInstalledIsStale() async {
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7, self.spare]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-spare", library: self.spare, availability: .notConnected),
            ]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        // The picker list as it was before the drive left.
        model.installed = [LibraryFixtures.row("local"), LibraryFixtures.row("on-spare", library: spare)]

        XCTAssertEqual(model.installed.count, 2, "the stale list still holds the offline row")
        XCTAssertEqual(model.loadableModels.map(\.alias), ["local"], "no picker may offer it")
    }

    func testInstalledFollowsMountsUnmountsAndRenamesWithoutARefreshLoop() async {
        let state = makeState()
        await state.refresh()
        let reads = ReadCounter()
        let model = await makeFollowingModel(state, reads: reads)
        model.startInstalledListFollowing()
        model.refreshModels()
        XCTAssertEqual(model.installed.map(\.alias), ["local", "on-t7"])
        let baselineReads = reads.count
        let baselineListings = service.installedCalls

        // Unplug: the engine's list loses the row, the library list marks it away.
        let away = LibraryFixtures.external("t7-id", name: "T7", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, away]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: away, availability: .notConnected),
            ]
        }
        await state.refresh()
        let followed = await eventually { model.installed.map(\.alias) == ["local"] }
        XCTAssertTrue(followed, "installed did not follow the unplug: \(model.installed.map(\.alias))")
        XCTAssertEqual(model.loadableModels.map(\.alias), ["local"])

        // Replug.
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7]
            $0.installed = [LibraryFixtures.row("local"), LibraryFixtures.row("on-t7", library: self.t7)]
        }
        await state.refresh()
        let returned = await eventually { model.installed.map(\.alias) == ["local", "on-t7"] }
        XCTAssertTrue(returned, "installed did not follow the replug: \(model.installed.map(\.alias))")
        XCTAssertEqual(model.loadableModels.map(\.alias), ["local", "on-t7"])

        // One reload per change of the library list, and no listing the test did
        // not ask for: reloading the pickers must not refresh the libraries, or
        // the two would feed each other.
        await letTasksStart()
        XCTAssertEqual(reads.count - baselineReads, 2, "one reload per change")
        XCTAssertEqual(service.installedCalls - baselineListings, 2, "only the test's own two refreshes")
        await letTasksStart()
        XCTAssertEqual(reads.count - baselineReads, 2, "nothing runs on its own afterwards")
        XCTAssertEqual(service.installedCalls - baselineListings, 2)
    }

    func testABurstOfChangesIsOneReload() async {
        let state = makeState()
        await state.refresh()
        let reads = ReadCounter()
        let model = await makeFollowingModel(state, reads: reads)
        model.startInstalledListFollowing()
        model.refreshModels()
        let baseline = reads.count

        // Three different listings land before the main actor runs anything else.
        for names in [["a"], ["a", "b"], ["a", "b", "c"]] {
            service.with { $0.installed = names.map { LibraryFixtures.row($0) } }
            await state.refresh()
        }
        _ = await eventually { model.installed.map(\.alias) == ["a", "b", "c"] }
        await letTasksStart()

        XCTAssertEqual(model.installed.map(\.alias), ["a", "b", "c"])
        XCTAssertLessThanOrEqual(reads.count - baseline, 3, "at most one reload per change")
    }

    func testAnUnchangedListingDoesNotReloadThePickers() async {
        let state = makeState()
        await state.refresh()
        let reads = ReadCounter()
        let model = await makeFollowingModel(state, reads: reads)
        model.startInstalledListFollowing()
        model.refreshModels()
        let baseline = reads.count

        await state.refresh()
        await state.refresh()
        await letTasksStart()

        XCTAssertEqual(reads.count, baseline, "a listing that changed nothing is not a reason to reload")
    }

    func testTheModelStopsFollowingWhenItIsShutDown() async {
        let state = makeState()
        await state.refresh()
        let reads = ReadCounter()
        let model = await makeFollowingModel(state, reads: reads)
        model.startInstalledListFollowing()
        model.refreshModels()
        let baseline = reads.count

        model.stopInstalledListFollowing()
        service.with { $0.installed = [LibraryFixtures.row("late")] }
        await state.refresh()
        await letTasksStart()

        XCTAssertEqual(reads.count, baseline)
    }

    func testAnAddedLibrarysModelsReachThePickers() async throws {
        // A successful Add does not call `refreshModels()` (task 6.6), so the
        // follow-up is what makes the added library's models loadable.
        let added = LibraryFixtures.external("added-id", name: "Added")
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary]
            $0.installed = [LibraryFixtures.row("local")]
            $0.markers["/Volumes/Added"] = try! LibraryFixtures.marker("added-id", name: "Added")
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.startInstalledListFollowing()
        model.refreshModels()
        XCTAssertEqual(model.loadableModels.map(\.alias), ["local"])

        service.with { $0.installed.append(LibraryFixtures.row("on-added", library: added)) }
        _ = try await state.addLibrary(path: "/Volumes/Added")

        let reached = await eventually { model.loadableModels.map(\.alias).contains("on-added") }
        XCTAssertTrue(reached, "\(model.loadableModels.map(\.alias))")
    }

    // MARK: - Selection

    func testASelectionOnAnOfflineRowMovesToALoadableOne() async {
        let state = makeState()
        await state.refresh()
        let reads = ReadCounter()
        let model = await makeFollowingModel(state, reads: reads)
        model.refreshModels()
        model.selected = model.installed.first { $0.alias == "on-t7" }
        XCTAssertEqual(model.selected?.alias, "on-t7")

        // The drive leaves. The reader is stale on purpose: it still returns the
        // row, as `installed` does between an unplug and the next reload.
        let away = LibraryFixtures.external("t7-id", name: "T7", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, away]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: away, availability: .notConnected),
            ]
        }
        await state.refresh()
        model.catalogInstalledReader = {
            [LibraryFixtures.row("local"), LibraryFixtures.row("on-t7", library: self.t7)]
        }
        model.refreshModels()

        XCTAssertEqual(model.selected?.alias, "local", "a selection must never rest on a model that cannot load")
    }

    func testNothingLoadableLeavesNothingSelected() async {
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.spare]
            $0.installed = [LibraryFixtures.row("on-spare", library: self.spare, availability: .notConnected)]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.selected = LibraryFixtures.row("on-spare", library: spare)
        model.refreshModels()

        XCTAssertNil(model.selected)
    }

    // MARK: - The other consumers

    func testTheServerOffersOnlyLoadableModelsToAttach() async {
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7, self.spare]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: self.t7),
                LibraryFixtures.row("on-spare", library: self.spare, availability: .notConnected),
            ]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.installed = [
            LibraryFixtures.row("local"), LibraryFixtures.row("on-t7", library: t7),
            LibraryFixtures.row("on-spare", library: spare),
        ]

        XCTAssertEqual(model.serverAttachableModels.map(\.alias), ["local", "on-t7"])
    }

    func testHumanizeDoesNotSeeAnOfflineModel() async {
        let humanizer = HumanizeController.modelAlias
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.spare]
            $0.installed = [LibraryFixtures.row(humanizer, library: self.spare, availability: .notConnected)]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.installed = [LibraryFixtures.row(humanizer, library: spare)]

        XCTAssertNil(model.humanizeWorkspace.installedPath(), "the Humanizer's drive is away: it cannot be loaded")
        XCTAssertFalse(model.humanizeWorkspace.isInstalled)

        // The drive returns.
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7]
            $0.installed = [LibraryFixtures.row(humanizer, library: self.t7)]
        }
        await state.refresh()
        model.installed = [LibraryFixtures.row(humanizer, library: t7)]
        XCTAssertEqual(
            model.humanizeWorkspace.installedPath(), LibraryFixtures.row(humanizer, library: t7).path)
    }

    // MARK: - First run (the setup screen)

    private func offlineOnlyState() async -> LibraryState {
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.spare]
            $0.installed = [LibraryFixtures.row("on-spare", library: self.spare, availability: .notConnected)]
        }
        let state = makeState()
        await state.refresh()
        return state
    }

    func testAUserWhoseOnlyModelsAreOnAnUnpluggedDriveIsNotToldToInstallOne() async {
        let model = await makeFollowingModel(await offlineOnlyState())
        model.installed = []

        XCTAssertFalse(model.requiresModelInstallation)
        XCTAssertFalse(model.showsModelSetupScreen, "the first-run screen says nothing is installed")
        XCTAssertFalse(model.showsNoModelInstalledBanner, "an unplugged drive is a statement, not a warning banner")
        XCTAssertFalse(model.isModelInstalled, "nothing can be loaded now")
        XCTAssertEqual(model.notConnectedStatement, "Connect Spare to load")
    }

    func testAUserWithNoModelAnywhereStillGetsTheSetupScreen() async {
        let state = makeState()
        service.with { $0.installed = [] }
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.installed = []

        XCTAssertTrue(model.requiresModelInstallation)
        // The routing is asked with an empty transcript on purpose: a fresh
        // `AppModel` reads the saved chat history, and another suite in the same
        // process (ImageWorkspaceTests) can leave a transcript there, which says
        // nothing about whether this user has a model.
        XCTAssertTrue(
            AppModel.showsModelSetupScreenTerms(
                requiresInstall: model.requiresModelInstallation, installing: model.isInstallingModel,
                hasTranscript: false))
        XCTAssertTrue(model.showsNoModelInstalledBanner)
        XCTAssertNil(model.notConnectedStatement)
    }

    func testAModelWhoseFilesAreGoneDoesNotCountAsOneTheUserCouldConnect() async {
        // A missing row has nothing to plug in: installing is the right prompt.
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7]
            $0.installed = [LibraryFixtures.row("gone", library: self.t7, availability: .missing)]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.installed = []

        XCTAssertTrue(model.requiresModelInstallation)
        XCTAssertNil(model.notConnectedStatement)
    }

    func testALoadableModelMeansNoSetupScreenWhateverElseIsAway() async {
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.spare]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-spare", library: self.spare, availability: .notConnected),
            ]
        }
        let state = makeState()
        await state.refresh()
        let model = await makeFollowingModel(state)
        model.installed = [LibraryFixtures.row("local")]

        XCTAssertFalse(model.requiresModelInstallation)
        XCTAssertTrue(model.isModelInstalled)
        XCTAssertFalse(model.showsModelSetupScreen)
    }
}
