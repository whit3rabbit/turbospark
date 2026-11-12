import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.10 (requirements 2.1, 2.3, 2.4 and 3.1): an audio install on a drive
/// that is away is a state to show, never an error and never "incompatible", it
/// cannot be run, and its Download button is replaced by the statement.
///
/// The destination choice and the free-space check for audio installs are covered
/// with the other modalities in `InstallDestinationRoutingTests`.
@MainActor
final class AudioOfflineInstallTests: XCTestCase {
    // MARK: Fixtures

    private func decoder() -> JSONDecoder {
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return decoder
    }

    /// The report the engine sends for one install in a library that is away,
    /// through the package's decoder.
    private func report(
        offlineAlias: String?, libraryName: String = "T7", incompatibleAlias: String? = nil
    ) throws -> AudioInstalledReport {
        func record(_ alias: String) -> String {
            """
            {"alias":"\(alias)","path":"/Volumes/T7/TurboSpark/audio/\(alias).gturbo","family":"kokoro",
             "status":"runs","install_bytes":10}
            """
        }
        func identity(_ alias: String) -> String {
            """
            {"task":"text_to_speech","alias":"\(alias)","repository":"owner/audio","revision":"pinned",
             "asset_fingerprint":"assets"}
            """
        }
        let offline = offlineAlias.map {
            """
            {"record":\(record($0)),"identity":\(identity($0)),"library_name":"\(libraryName)"}
            """
        }
        let incompatible = incompatibleAlias.map {
            "{\"record\":\(record($0)),\"identity\":\(identity($0)),\"error\":\"checksum mismatch\"}"
        }
        let json = """
            {"installed":[],"needs_adoption":[],"incompatible":[\(incompatible ?? "")],
             "other_audio":[],"not_connected":[\(offline ?? "")]}
            """
        return try decoder().decode(AudioInstalledReport.self, from: Data(json.utf8))
    }

    private func profile(alias: String = "kokoro-fixture", repository: String = "owner/audio") -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .textToSpeech, alias: alias, repository: repository, revision: "pinned",
                assetFingerprint: "assets"),
            family: "kokoro", displayName: "Kokoro Fixture",
            capabilities: AudioCapabilities(operations: ["synthesize"], backend: "metal", cancellation: "step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), downloadBytes: 2_000_000_000,
            readiness: "qualified")
    }

    private func makeController(
        report: AudioInstalledReport, profiles: [AudioProfile]
    ) throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "offline-audio" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let snapshot = AudioCatalogSnapshot(profiles: profiles, installed: report, capabilities: [])
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
            assets: ManagedAssetStore(vault: vault), catalogReader: { snapshot })
        return (controller, root)
    }

    private func settle(_ controller: AudioWorkspaceController) async {
        _ = await eventually { !controller.profiles.isEmpty && !controller.isRefreshingModels }
    }

    // MARK: The report keeps the bucket (requirement 3.1)

    func testTheEnginesNotConnectedBucketDecodesAndIsNotIncompatible() throws {
        let decoded = try report(offlineAlias: "kokoro-fixture")
        XCTAssertEqual(decoded.notConnected.map(\.record.alias), ["kokoro-fixture"])
        XCTAssertEqual(decoded.notConnected.first?.libraryName, "T7")
        XCTAssertTrue(decoded.incompatible.isEmpty, "an unplugged drive is not a broken install")
    }

    // MARK: The pure status

    func testAnOfflineInstallIsNotConnectedNotNotInstalled() throws {
        let offline = try report(offlineAlias: "kokoro-fixture").notConnected
        let status = AudioInstallStatus.resolve(
            alias: "kokoro-fixture", installedPaths: [:], localModels: [], notConnected: offline)
        XCTAssertEqual(status, .notConnected(libraryName: "T7"))
        XCTAssertFalse(status.offersDownload, "Download is replaced by the statement")
        XCTAssertFalse(status.isInstalled)
    }

    func testAnInstallThatIsHereIsInstalledEvenIfAnOfflineRowSharesTheAlias() throws {
        let offline = try report(offlineAlias: "kokoro-fixture").notConnected
        let status = AudioInstallStatus.resolve(
            alias: "kokoro-fixture", installedPaths: ["kokoro-fixture": "/x"], localModels: [],
            notConnected: offline)
        XCTAssertEqual(status, .installed)
    }

    func testALocalFolderChoiceIsUsableWhateverTheDriveSays() throws {
        let offline = try report(offlineAlias: "kokoro-fixture").notConnected
        let status = AudioInstallStatus.resolve(
            alias: "kokoro-fixture", installedPaths: [:], localModels: ["kokoro-fixture"], notConnected: offline)
        XCTAssertEqual(status, .installed, "a checkpoint folder the user chose does not live on the away drive")
    }

    func testAProfileWithNothingInstalledOffersDownload() {
        let status = AudioInstallStatus.resolve(
            alias: "kokoro-fixture", installedPaths: [:], localModels: [], notConnected: [])
        XCTAssertEqual(status, .notInstalled)
        XCTAssertTrue(status.offersDownload)
    }

    func testAnotherProfilesOfflineRowDoesNotMatter() throws {
        let offline = try report(offlineAlias: "other-model").notConnected
        XCTAssertEqual(
            AudioInstallStatus.resolve(
                alias: "kokoro-fixture", installedPaths: [:], localModels: [], notConnected: offline),
            .notInstalled)
    }

    // MARK: How the picker draws it

    func testTheOfflineRowReadsNotConnectedWithAGlyphAndNeverIncompatible() {
        let label = AudioOfflineInstalls.pickerLabel(
            displayName: "Kokoro Fixture", status: .notConnected(libraryName: "T7"), language: .english)
        XCTAssertEqual(label.glyph, LibraryRowStatus.notConnected.glyph, "a shape, not color alone")
        XCTAssertTrue(label.title.contains("Kokoro Fixture"))
        XCTAssertTrue(label.title.contains("Not connected"), label.title)
        XCTAssertFalse(label.title.localizedCaseInsensitiveContains("incompatible"))
        XCTAssertFalse(label.title.localizedCaseInsensitiveContains("missing"))

        let plain = AudioOfflineInstalls.pickerLabel(
            displayName: "Kokoro Fixture", status: .installed, language: .english)
        XCTAssertEqual(plain.title, "Kokoro Fixture")
        XCTAssertNil(plain.glyph)
    }

    func testTheOfflineLabelIsTranslated() {
        for language in AppLanguage.allCases where language != .system {
            let label = AudioOfflineInstalls.pickerLabel(
                displayName: "Kokoro Fixture", status: .notConnected(libraryName: "T7"), language: language)
            XCTAssertTrue(label.title.contains("Kokoro Fixture"), language.rawValue)
            XCTAssertTrue(
                label.title.contains(AppStrings.localized("Not connected", language: language)), language.rawValue)
        }
    }

    func testTheStatementReplacesTheInstallButtonAndNamesTheDrive() {
        let statement = AudioOfflineInstalls.statement(
            for: .notConnected(libraryName: "T7"), language: .english)
        XCTAssertEqual(statement, "Installed on T7, which is not connected. Connect the drive to use it.")
        XCTAssertNil(AudioOfflineInstalls.statement(for: .installed, language: .english))
        XCTAssertNil(AudioOfflineInstalls.statement(for: .notInstalled, language: .english))
    }

    func testADrivesNameIsEscapedInTheStatement() throws {
        let statement = try XCTUnwrap(
            AudioOfflineInstalls.statement(
                for: .notConnected(libraryName: "T7\u{202E}gnp\n"), language: .english))
        XCTAssertFalse(statement.contains("\u{202E}"))
        XCTAssertFalse(statement.contains("\n"))
    }

    // MARK: The workspace

    func testARefreshKeepsTheBucketAndDoesNotOfferTheModelForUse() async throws {
        let target = profile()
        let (controller, root) = try makeController(
            report: report(offlineAlias: target.identity.alias), profiles: [target])
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        await settle(controller)

        controller.selectPage(.voiceover)
        controller.recipe.modelID = target.identity.alias

        XCTAssertEqual(controller.notConnectedInstalls.map(\.record.alias), [target.identity.alias])
        XCTAssertTrue(controller.incompatibleInstalls.isEmpty, "never shown as incompatible")
        XCTAssertEqual(controller.installStatus(for: target), .notConnected(libraryName: "T7"))
        XCTAssertFalse(controller.modelInstalled, "it cannot be run")
        XCTAssertFalse(controller.canRun)
        XCTAssertNil(controller.installedPaths[target.identity.alias], "it is not in the usable list")
    }

    func testInstallingAnOfflineProfileIsRefusedBeforeItQueues() async throws {
        let target = profile()
        let (controller, root) = try makeController(
            report: report(offlineAlias: target.identity.alias), profiles: [target])
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        await settle(controller)
        controller.selectPage(.voiceover)
        controller.recipe.modelID = target.identity.alias
        var queued = 0
        controller.installHandler = { _, _, _ in queued += 1; return true }

        controller.installSelectedModel(library: nil)

        XCTAssertEqual(queued, 0, "alias_elsewhere would refuse it; the app does not ask")
    }

    func testTheChosenLibraryReachesTheInstallHandler() async throws {
        let target = profile()
        let (controller, root) = try makeController(report: report(offlineAlias: nil), profiles: [target])
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        await settle(controller)
        controller.selectPage(.voiceover)
        controller.recipe.modelID = target.identity.alias
        var received: [(alias: String, acknowledged: [String], library: String?)] = []
        controller.installHandler = { profile, acknowledged, library in
            received.append((profile.identity.alias, acknowledged, library))
            return true
        }

        controller.installSelectedModel(acknowledged: ["terms-v1"], library: "t7-id")

        XCTAssertEqual(received.map(\.alias), [target.identity.alias])
        XCTAssertEqual(received.first?.acknowledged, ["terms-v1"], "the acknowledgement flow is untouched")
        XCTAssertEqual(received.first?.library, "t7-id")
    }

    func testTheApiPaneListsTheBucketAndOffersNothingFromIt() throws {
        let decoded = try report(offlineAlias: "kokoro-fixture")
        let rows = AudioOfflineInstalls.offlineSpeechRows(in: decoded)
        XCTAssertEqual(rows.map(\.alias), ["kokoro-fixture"])
        XCTAssertEqual(rows.first?.libraryName, "T7")
        XCTAssertEqual(rows.first?.glyph, LibraryRowStatus.notConnected.glyph)
        XCTAssertTrue(
            rows.first?.statement(language: .english).contains("T7") ?? false)
        // Nothing in the report's usable list comes from the bucket.
        XCTAssertTrue(decoded.installed.isEmpty)
    }

    func testTheApiPaneRowsKeepOnlySpeechFamilies() throws {
        let json = """
            {"installed":[],"needs_adoption":[],"incompatible":[],"other_audio":[],
             "not_connected":[{"record":{"alias":"music-a","path":"/p","family":"minimax_music3","status":"runs","install_bytes":1},
                               "identity":null,"library_name":"T7"}]}
            """
        let decoded = try decoder().decode(AudioInstalledReport.self, from: Data(json.utf8))
        XCTAssertEqual(decoded.notConnected.count, 1)
        XCTAssertTrue(AudioOfflineInstalls.offlineSpeechRows(in: decoded).isEmpty)
    }

    func testTheAudioViewsReadTheBucketAndGateTheirInstallButton() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        func code(_ file: String) throws -> String {
            MoveStoreRetirementTests.codeWithoutComments(
                try String(contentsOf: sources.appendingPathComponent(file), encoding: .utf8))
        }
        let picker = try code("Audio/AudioModelPickerView.swift")
        XCTAssertTrue(picker.contains("installStatus(for:"), "the picker asks for the profile's status")
        XCTAssertTrue(picker.contains("InstallDestinationPicker("), "the audio install shows the destination menu")
        let api = try code("Server/AudioAPIPaneView.swift")
        XCTAssertTrue(api.contains("offlineSpeechRows"), "the API pane shows the bucket")
        let controller = try code("Audio/AudioWorkspaceController.swift")
        XCTAssertTrue(controller.contains("notConnected"), "the refresh keeps the bucket")
    }
}
