import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.9 (requirements 8.1 and 8.2): a server favorite follows its models
/// by install identity, falls back to the legacy path, is re-keyed on the first
/// load, and says "not connected" for a model whose drive is away instead of
/// "no longer installed".
///
/// Everything here runs over `FakeModelLibraryService`: the app suites link the
/// release engine, which has no location overrides to build real libraries.
final class ServerFavoriteIdentityTests: LibraryStateTestCase {
    private var models: [AppModel] = []

    override func tearDownWithError() throws {
        // A model that persisted settings leaves its favorites in the
        // per-process settings file, where the next `AppModel()` would read
        // them as its own.
        for model in models {
            model.settingsPersistDebounceTask?.cancel()
            model.settingsPersistDebounceTask = nil
            model.serverFavorites = []
            model.stopServer()
        }
        if !models.isEmpty { MacAppSettingsFileStore.save(MacAppSettings()) }
        models = []
        try super.tearDownWithError()
    }

    // MARK: - Fixtures

    private func libraryRow(
        _ alias: String, library: String = "t7-id", libraryName: String = "T7",
        path: String? = nil, availability: LibraryAvailability = .available
    ) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: path ?? "/Volumes/\(libraryName)/TurboSpark/text/\(alias).gturbo",
            family: "qwen", library: library, libraryName: libraryName,
            installId: "\(library):text/\(alias)", availability: availability,
            locationKind: .library, connectName: availability == .notConnected ? libraryName : nil)
    }

    private func thisMacRow(_ alias: String, availability: LibraryAvailability = .available) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: "/Users/test/.turbospark/models/text/\(alias).gturbo",
            family: "qwen", library: "internal", libraryName: "This Mac",
            installId: "internal:text/\(alias)", availability: availability, locationKind: .thisMac)
    }

    /// A row the app built itself for another app's folder: the memberwise
    /// defaults, so `library` is nil and the id is the placeholder.
    private func placeholderRow(_ alias: String = "lm-model") -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "lmstudio/\(alias)", revision: "local",
            path: "/Users/test/.lmstudio/models/\(alias).gguf", family: "qwen")
    }

    private func favorite(paths: [String] = [], ids: [String] = []) -> ServerFavorite {
        ServerFavorite(
            name: "Saved", host: "::1", port: 0, modelPaths: paths, modelInstallIds: ids,
            contextTokens: 4096)
    }

    private func served(_ alias: String...) throws -> ServerInfo {
        let names = alias.map { "\"\($0)\"" }.joined(separator: ",")
        let json = """
            {"port":9090,"host":"::1","modelId":"","models":[\(names)],"imageModels":[],
             "audioModels":[],"authEnabled":false,"uptimeSeconds":1}
            """
        return try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
    }

    private func makeFavoriteModel(
        libraries: [ModelLibrary]? = nil, rows: [InstalledModel], installed: [InstalledModel] = []
    ) async -> AppModel {
        service.with {
            if let libraries { $0.libraries = libraries }
            $0.installed = rows
        }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        model.installed = installed
        models.append(model)
        return model
    }

    // MARK: - Resolution by identity (requirement 8.1)

    func testAnInstallIdFindsTheRowAtItsCurrentPathWhenTheDriveRemountedUnderAnotherName() {
        let moved = libraryRow("qwen", libraryName: "T7 1")
        let saved = favorite(
            paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])

        XCTAssertEqual(saved.resolve(in: [moved]), [.available(moved)])
    }

    func testTheInstallIdIsResolvedBeforeTheLegacyPath() {
        // The old mount point now holds a different drive's model, so the saved
        // path matches a row that is NOT the one the favorite is about.
        let current = libraryRow("qwen", libraryName: "T7 1")
        let squatter = libraryRow(
            "other", library: "usb2", libraryName: "USB2",
            path: "/Volumes/T7/TurboSpark/text/qwen.gturbo")
        let saved = favorite(
            paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])

        XCTAssertEqual(saved.resolve(in: [squatter, current]), [.available(current)])
    }

    func testAModelMovedToAnotherLibraryIsFoundByItsAliasWhenItIsTheOnlyCandidate() {
        let moved = thisMacRow("qwen")
        let saved = favorite(
            paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])

        XCTAssertEqual(saved.resolve(in: [libraryRow("llama"), moved]), [.available(moved)])
    }

    func testTwoCandidatesForOneAliasAreNeverGuessed() {
        let saved = favorite(ids: ["t7-id:text/qwen"])
        let rows = [thisMacRow("qwen"), libraryRow("qwen", library: "usb2", libraryName: "USB2")]

        XCTAssertEqual(saved.resolve(in: rows), [.missing("qwen")])
    }

    func testTheAliasMatchNeverCrossesModality() {
        // `t7-id:text/qwen` must not adopt an image install with the same alias.
        let image = InstalledModel(
            alias: "qwen", repo: "owner/qwen", path: "/Users/test/.turbospark/models/image/qwen",
            family: "qwen", modality: "image", library: "internal", libraryName: "This Mac",
            installId: "internal:image/qwen", locationKind: .thisMac)

        XCTAssertEqual(favorite(ids: ["t7-id:text/qwen"]).resolve(in: [image]), [.missing("qwen")])
    }

    func testAPlaceholderIsNeverMatchedThroughAnInstallId() {
        // The placeholder's default id looks like a This Mac row's.
        let placeholder = placeholderRow("qwen")
        XCTAssertEqual(placeholder.installId, "internal:text/qwen")

        XCTAssertEqual(favorite(ids: ["internal:text/qwen"]).resolve(in: [placeholder]), [.missing("qwen")])
    }

    func testAnIdWithNoRowFallsBackToTheLegacyPath() {
        // The alias differs, so the cross-library match finds nothing and the
        // path route is the only one that can answer.
        let row = libraryRow("qwen2", path: "/Volumes/T7/TurboSpark/text/qwen.gturbo")
        let saved = favorite(paths: [row.path], ids: ["gone-id:text/qwen"])

        XCTAssertEqual(saved.resolve(in: [row]), [.available(row)])
    }

    // MARK: - Resolution by legacy path

    func testAPathOnlyFavoriteResolvesByPathAgainstLibraryAndPlaceholderRows() {
        let library = libraryRow("qwen")
        let placeholder = placeholderRow()
        let saved = favorite(paths: [library.path, placeholder.path])

        XCTAssertEqual(saved.resolve(in: [library, placeholder]), [.available(library), .available(placeholder)])
    }

    func testAnUnknownPathIsMissingAndNamesThePath() {
        XCTAssertEqual(favorite(paths: ["/nowhere/x.gturbo"]).resolve(in: [libraryRow("qwen")]), [.missing("/nowhere/x.gturbo")])
    }

    func testAFolderThatIsGoneOnAConnectedDriveIsMissingNotNotConnected() {
        let gone = libraryRow("qwen", availability: .missing)
        let outcomes = favorite(paths: [gone.path], ids: [gone.installId]).resolve(in: [gone])

        XCTAssertEqual(outcomes, [.missing(gone.path)])
    }

    // MARK: - Not connected (requirement 8.2)

    func testAnOfflineModelReportsItsDriveByIdentity() {
        let offline = libraryRow("qwen", availability: .notConnected)

        XCTAssertEqual(
            favorite(paths: [offline.path], ids: [offline.installId]).resolve(in: [offline]),
            [.notConnected(model: "qwen", libraryName: "T7")])
    }

    func testAnOfflineModelReportsItsDriveByLegacyPath() {
        let offline = libraryRow("qwen", availability: .notConnected)

        XCTAssertEqual(
            favorite(paths: [offline.path]).resolve(in: [offline]),
            [.notConnected(model: "qwen", libraryName: "T7")])
    }

    func testTheDriveToConnectComesFromTheRowsConnectName() {
        let custom = InstalledModel(
            alias: "qwen", repo: "owner/qwen", path: "/Volumes/Spare/models/qwen", family: "qwen",
            library: nil, libraryName: "custom location", installId: "internal:text/qwen",
            availability: .notConnected, locationKind: .custom, connectName: "Spare")

        XCTAssertEqual(
            favorite(paths: [custom.path]).resolve(in: [custom]),
            [.notConnected(model: "qwen", libraryName: "Spare")])
    }

    // MARK: - Re-keying (requirement 8.1)

    func testRekeyingAPathOnlyFavoriteWritesTheIdsAndKeepsThePaths() throws {
        let t7 = libraryRow("qwen")
        let local = thisMacRow("llama")
        let saved = favorite(paths: [t7.path, local.path])

        let rekeyed = try XCTUnwrap(saved.rekeyed(in: [t7, local]))

        XCTAssertEqual(rekeyed.modelInstallIds, [t7.installId, local.installId])
        XCTAssertEqual(rekeyed.modelPaths, saved.modelPaths, "the legacy paths stay for an older app")
        XCTAssertEqual(rekeyed.id, saved.id)
    }

    func testRekeyingNeverWritesAPlaceholderIdAndKeepsTheIdsAlignedWithThePaths() throws {
        let placeholder = placeholderRow()
        let t7 = libraryRow("qwen")
        let saved = favorite(paths: [placeholder.path, t7.path])

        let rekeyed = try XCTUnwrap(saved.rekeyed(in: [placeholder, t7]))

        XCTAssertEqual(rekeyed.modelInstallIds, ["", t7.installId])
        XCTAssertFalse(rekeyed.modelInstallIds.contains(placeholder.installId))
        XCTAssertEqual(rekeyed.modelPaths, [placeholder.path, t7.path])
    }

    func testAFavoriteOfPlaceholdersOnlyHasNothingToRekey() {
        let placeholder = placeholderRow()

        XCTAssertNil(favorite(paths: [placeholder.path]).rekeyed(in: [placeholder]))
    }

    func testAnAlreadyKeyedFavoriteHasNothingToRekey() {
        let t7 = libraryRow("qwen")

        XCTAssertNil(favorite(paths: [t7.path], ids: [t7.installId]).rekeyed(in: [t7]))
    }

    func testAMovedModelsStaleIdIsReplacedByItsCurrentId() throws {
        let moved = thisMacRow("qwen")
        let saved = favorite(paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])

        let rekeyed = try XCTUnwrap(saved.rekeyed(in: [moved]))

        XCTAssertEqual(rekeyed.modelInstallIds, ["internal:text/qwen"])
    }

    func testAnOfflineModelIsRekeyedBecauseItsIdentityIsKnown() throws {
        let offline = libraryRow("qwen", availability: .notConnected)

        let rekeyed = try XCTUnwrap(favorite(paths: [offline.path]).rekeyed(in: [offline]))

        XCTAssertEqual(rekeyed.modelInstallIds, [offline.installId])
    }

    func testAnEntryThatResolvesToNothingKeepsItsStoredId() {
        let saved = favorite(paths: ["/Volumes/Gone/qwen.gturbo"], ids: ["gone-id:text/qwen"])

        XCTAssertNil(saved.rekeyed(in: []), "a model that is not listed keeps the identity it had")
    }

    // MARK: - Storage shape

    func testTheInstallIdsRoundTripAndThePathsAreStillWritten() throws {
        let saved = favorite(paths: ["/a.gturbo", "/b.gturbo"], ids: ["", "t7-id:text/b"])

        let data = try JSONEncoder().encode(saved)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])

        XCTAssertEqual(object["modelInstallIds"] as? [String], ["", "t7-id:text/b"])
        XCTAssertEqual(object["modelPaths"] as? [String], ["/a.gturbo", "/b.gturbo"])
        XCTAssertEqual(try JSONDecoder().decode(ServerFavorite.self, from: data), saved)
    }

    func testAnOldFavoriteWithoutInstallIdsDecodesWithNone() throws {
        let json = """
            {"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","name":"Old","host":"127.0.0.1",
             "port":8080,"modelPaths":["/Volumes/T7/TurboSpark/text/qwen.gturbo"],"contextTokens":4096}
            """

        let decoded = try JSONDecoder().decode(ServerFavorite.self, from: Data(json.utf8))

        XCTAssertEqual(decoded.modelInstallIds, [])
        XCTAssertEqual(decoded.modelPaths, ["/Volumes/T7/TurboSpark/text/qwen.gturbo"])
        XCTAssertEqual(decoded.name, "Old")
    }

    func testAnOlderSettingsFileStillLoadsItsFavoritesAndEverythingElse() throws {
        // `serverFavorites` decodes leniently, to an EMPTY list on any failure,
        // so a favorite that stopped decoding would silently vanish.
        let json = """
            {"serverHost":"::1","serverPinnedPort":9191,"temperature":0.3,
             "serverFavorites":[
               {"id":"6F9619FF-8B86-D011-B42D-00C04FC964FF","name":"Old","host":"127.0.0.1",
                "port":8080,"modelPaths":["/a.gturbo","/b.gturbo"],"contextTokens":4096}]}
            """

        let settings = try JSONDecoder().decode(MacAppSettings.self, from: Data(json.utf8))

        XCTAssertEqual(settings.serverFavorites.count, 1)
        XCTAssertEqual(settings.serverFavorites.first?.modelInstallIds, [])
        XCTAssertEqual(settings.serverFavorites.first?.modelPaths, ["/a.gturbo", "/b.gturbo"])
        XCTAssertEqual(settings.serverHost, "::1")
        XCTAssertEqual(settings.serverPinnedPort, 9191)
        XCTAssertEqual(settings.temperature, 0.3, accuracy: 0.0001)
    }

    func testTheModelCountIsTheLargerOfTheTwoLists() {
        XCTAssertEqual(favorite(paths: ["/a", "/b"]).modelCount, 2)
        XCTAssertEqual(favorite(ids: ["t7-id:text/a", "t7-id:text/b", "t7-id:text/c"]).modelCount, 3)
        XCTAssertEqual(favorite(paths: ["/a", "/b"], ids: ["", "t7-id:text/b"]).modelCount, 2)
        XCTAssertEqual(favorite().modelCount, 0)
    }

    // MARK: - Saving

    func testSavingAFavoriteWritesInstallIdsAndPathsForLibraryRows() async throws {
        let t7 = libraryRow("qwen")
        let local = thisMacRow("llama")
        let model = await makeFavoriteModel(rows: [t7, local], installed: [t7, local])
        model.serverInfo = try served("qwen.gturbo", "llama.gturbo")

        model.saveServerFavorite(name: "  Both  ")

        let saved = try XCTUnwrap(model.serverFavorites.first)
        XCTAssertEqual(saved.name, "Both")
        XCTAssertEqual(saved.modelInstallIds, [t7.installId, local.installId])
        XCTAssertEqual(saved.modelPaths, [t7.path, local.path])
    }

    func testSavingAPlaceholderRowWritesThePathOnly() async throws {
        let placeholder = placeholderRow()
        let t7 = libraryRow("qwen")
        let model = await makeFavoriteModel(rows: [t7], installed: [placeholder, t7])
        model.serverInfo = try served("lm-model.gguf", "qwen.gturbo")

        model.saveServerFavorite(name: "Mixed")

        let saved = try XCTUnwrap(model.serverFavorites.first)
        XCTAssertEqual(saved.modelPaths, [placeholder.path, t7.path])
        XCTAssertEqual(saved.modelInstallIds, ["", t7.installId], "the placeholder's id is never an identity")
        XCTAssertFalse(saved.modelInstallIds.contains(placeholder.installId))
    }

    func testSavingOnlyPlaceholderRowsWritesNoIdentityAtAll() async throws {
        let placeholder = placeholderRow()
        let model = await makeFavoriteModel(rows: [], installed: [placeholder])
        model.serverInfo = try served("lm-model.gguf")

        model.saveServerFavorite(name: "Other app")

        let saved = try XCTUnwrap(model.serverFavorites.first)
        XCTAssertEqual(saved.modelPaths, [placeholder.path])
        XCTAssertEqual(saved.modelInstallIds, [])
    }

    // MARK: - Loading through the model

    func testALegacyFavoriteIsRekeyedOnItsFirstLoadAndPersistedOnce() async throws {
        let t7 = libraryRow("qwen")
        let placeholder = placeholderRow()
        let model = await makeFavoriteModel(rows: [t7], installed: [placeholder])
        let legacy = favorite(paths: [t7.path, placeholder.path])
        model.serverFavorites = [legacy]
        model.settingsPersistDebounceTask?.cancel()
        model.settingsPersistDebounceTask = nil

        let outcomes = model.resolveServerFavoriteModels(legacy)

        XCTAssertEqual(outcomes, [.available(t7), .available(placeholder)])
        let rekeyed = try XCTUnwrap(model.serverFavorites.first)
        XCTAssertEqual(rekeyed.modelInstallIds, [t7.installId])
        XCTAssertEqual(rekeyed.modelPaths, legacy.modelPaths, "the paths stay")
        XCTAssertEqual(rekeyed.id, legacy.id)
        let pending = try XCTUnwrap(model.settingsPersistDebounceTask, "the re-key must be persisted")
        await pending.value
        let onDisk = try XCTUnwrap(MacAppSettingsFileStore.load().serverFavorites.first)
        XCTAssertEqual(onDisk.modelInstallIds, [t7.installId])
        XCTAssertEqual(onDisk.modelPaths, legacy.modelPaths)

        // The second load has nothing left to write.
        _ = model.resolveServerFavoriteModels(rekeyed)
        XCTAssertNil(model.settingsPersistDebounceTask, "a favorite that is already keyed is not written again")
    }

    func testAMovedModelResolvesThroughItsAliasAndTheFavoriteFollowsIt() async throws {
        let moved = thisMacRow("qwen")
        let model = await makeFavoriteModel(rows: [moved])
        let saved = favorite(paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])
        model.serverFavorites = [saved]

        XCTAssertEqual(model.resolveServerFavoriteModels(saved), [.available(moved)])
        XCTAssertEqual(model.serverFavorites.first?.modelInstallIds, ["internal:text/qwen"])
    }

    func testADriveRemountedUnderAnotherNameIsFoundAtItsCurrentPath() async throws {
        let remounted = libraryRow("qwen", libraryName: "T7 1")
        let model = await makeFavoriteModel(rows: [remounted])
        let saved = favorite(paths: ["/Volumes/T7/TurboSpark/text/qwen.gturbo"], ids: ["t7-id:text/qwen"])
        model.serverFavorites = [saved]

        model.settingsPersistDebounceTask?.cancel()
        model.settingsPersistDebounceTask = nil

        XCTAssertEqual(model.resolveServerFavoriteModels(saved), [.available(remounted)])
        XCTAssertEqual(model.serverFavorites.first?.modelInstallIds, ["t7-id:text/qwen"])
        XCTAssertNil(model.settingsPersistDebounceTask, "the identity did not change, so nothing is written")
    }

    func testTheRowsAreTheLibraryRowsPlusTheAvailableOnesTheEngineDoesNotList() async {
        let t7 = libraryRow("qwen", availability: .notConnected)
        let placeholder = placeholderRow()
        // The same model in both lists must appear once.
        let model = await makeFavoriteModel(rows: [t7], installed: [t7, placeholder])

        XCTAssertEqual(model.serverFavoriteRows(), [t7, placeholder])
    }

    func testAFavoriteThatWasDeletedMeanwhileIsNotBroughtBack() async {
        let t7 = libraryRow("qwen")
        let model = await makeFavoriteModel(rows: [t7])
        let deleted = favorite(paths: [t7.path])
        model.serverFavorites = []

        XCTAssertEqual(model.resolveServerFavoriteModels(deleted), [.available(t7)])
        XCTAssertEqual(model.serverFavorites, [])
    }

    // MARK: - Loading starts the server and reports what it could not attach

    private func loadAndWaitForToast(_ model: AppModel, _ saved: ServerFavorite) async -> AppToast? {
        model.loadServerFavorite(saved)
        let arrived = await eventually(timeout: 15) { model.activeToast != nil && !model.serverBusy }
        return arrived ? model.activeToast : nil
    }

    func testAnOfflineFavoriteStartsTheServerAttachesNothingAndNamesTheDrive() async throws {
        let offline = libraryRow("qwen", availability: .notConnected)
        let model = await makeFavoriteModel(rows: [offline])
        let toast = await loadAndWaitForToast(
            model, favorite(paths: [offline.path], ids: [offline.installId]))

        let shown = try XCTUnwrap(toast, "no toast was shown")
        XCTAssertEqual(shown.style, .warning)
        XCTAssertEqual(shown.message, ServerFavoriteModel.notConnectedMessage(model: "qwen", library: "T7"))
        XCTAssertFalse(shown.message.contains("no longer installed"))
        XCTAssertNotNil(model.server, "an offline favorite still starts the server")
        XCTAssertEqual(model.serverInfo?.models, [], "nothing is attached")
    }

    func testAFavoriteWhoseFolderIsGoneKeepsTheNoLongerInstalledToast() async throws {
        let gone = libraryRow("qwen", availability: .missing)
        let model = await makeFavoriteModel(rows: [gone])
        let toast = await loadAndWaitForToast(
            model, favorite(paths: [gone.path], ids: [gone.installId]))

        let shown = try XCTUnwrap(toast, "no toast was shown")
        XCTAssertEqual(shown.style, .warning)
        XCTAssertEqual(shown.message, "Model is no longer installed: \(gone.path)")
        XCTAssertEqual(model.serverInfo?.models, [])
    }

    func testAnUnknownModelKeepsTheNoLongerInstalledToast() async throws {
        let model = await makeFavoriteModel(rows: [libraryRow("llama")])
        let toast = await loadAndWaitForToast(model, favorite(paths: ["/nowhere/qwen.gturbo"]))

        XCTAssertEqual(try XCTUnwrap(toast).message, "Model is no longer installed: /nowhere/qwen.gturbo")
    }

    func testTheDriveAndModelNamesAreShownEscaped() {
        // A name from a drive can carry controls and bidirectional marks.
        let message = ServerFavoriteModel.notConnectedMessage(
            model: "qwen\u{202E}gnp", library: "T7\nSpare", language: .english)

        XCTAssertFalse(message.contains("\u{202E}"))
        XCTAssertFalse(message.contains("\n"))
    }

    // MARK: - The new string, in every language

    func testTheNotConnectedSentenceIsTranslatedIntoEveryLanguage() {
        let english = ServerFavoriteModel.notConnectedMessage(
            model: "qwen", library: "T7", language: .english)
        XCTAssertEqual(english, "qwen is on T7, which is not connected. Connect the drive and try again.")

        var untranslated: [String] = []
        for language in AppLanguage.allCases where language != .system && language != .english {
            let text = ServerFavoriteModel.notConnectedMessage(
                model: "qwen", library: "T7", language: language)
            if text == english { untranslated.append(language.rawValue) }
            XCTAssertTrue(text.contains("qwen"), "\(language.rawValue): the model name is missing")
            XCTAssertTrue(text.contains("T7"), "\(language.rawValue): the drive name is missing")
            XCTAssertFalse(text.contains("%@"), "\(language.rawValue): a placeholder was left unfilled")
            let modelAt = text.range(of: "qwen")?.lowerBound
            let driveAt = text.range(of: "T7")?.lowerBound
            if let modelAt, let driveAt {
                XCTAssertLessThan(modelAt, driveAt, "\(language.rawValue): the arguments are in English order")
            }
        }
        XCTAssertEqual(untranslated, [], "these languages fell back to English")
    }
}
