import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.11: "Move to..." in the audio picker is offered for an installed audio
/// model with a receipt, and for nothing else.
///
/// The engine refuses to move an install that has no receipt into a library ("adopt
/// it on This Mac first"), so the picker never offers it: an unadopted install has
/// its own Adopt action, a checkpoint folder the user chose is not an install, and a
/// profile with nothing installed has nothing to move.
@MainActor
final class ModelMoveAudioTests: XCTestCase {
    private func report(
        installed: [(alias: String, library: String?)] = [], needsAdoption: [String] = []
    ) throws -> AudioInstalledReport {
        func identity(_ alias: String) -> String {
            """
            {"task":"text_to_speech","alias":"\(alias)","repository":"owner/audio","revision":"pinned",
             "asset_fingerprint":"assets"}
            """
        }
        func record(_ alias: String, library: String?) -> String {
            let lib = library.map { "\"library\":\"\($0)\"," } ?? ""
            return """
                {\(lib)"alias":"\(alias)","path":"/Volumes/T7/TurboSpark/audio/\(alias)","family":"kokoro",
                 "status":"runs","install_bytes":300000000}
                """
        }
        let rows = installed.map { record($0.alias, library: $0.library) }.joined(separator: ",")
        let legacy = needsAdoption.map {
            "{\"record\":\(record($0, library: nil)),\"identity\":\(identity($0))}"
        }.joined(separator: ",")
        let json = """
            {"installed":[\(rows)],"needs_adoption":[\(legacy)],"incompatible":[],"other_audio":[],
             "not_connected":[]}
            """
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(AudioInstalledReport.self, from: Data(json.utf8))
    }

    private func profile(_ alias: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .textToSpeech, alias: alias, repository: "owner/audio", revision: "pinned",
                assetFingerprint: "assets"),
            family: "kokoro", displayName: "Kokoro Fixture",
            capabilities: AudioCapabilities(operations: ["synthesize"], backend: "metal", cancellation: "step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), downloadBytes: 2_000_000_000,
            readiness: "qualified")
    }

    private func makeController(report: AudioInstalledReport, profiles: [AudioProfile]) throws -> AudioWorkspaceController {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "move-audio" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let snapshot = AudioCatalogSnapshot(profiles: profiles, installed: report, capabilities: [])
        addTeardownBlock { try? FileManager.default.removeItem(at: root) }
        return AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
            assets: ManagedAssetStore(vault: vault), catalogReader: { snapshot })
    }

    private func settle(_ controller: AudioWorkspaceController) async {
        _ = await eventually { !controller.profiles.isEmpty && !controller.isRefreshingModels }
    }

    func testAnInstalledModelWithAReceiptOffersMoveWithItsLibraryAndSize() async throws {
        let controller = try makeController(
            report: report(installed: [("kokoro-fixture", "t7-id")]), profiles: [profile("kokoro-fixture")])
        await settle(controller)

        let subject = try XCTUnwrap(controller.moveSubject(for: profile("kokoro-fixture")))

        XCTAssertEqual(subject.kind, .audio)
        XCTAssertEqual(subject.installId, "t7-id:audio/kokoro-fixture")
        XCTAssertEqual(subject.libraryId, "t7-id")
        XCTAssertEqual(subject.knownBytes, 300_000_000)
        XCTAssertFalse(subject.isThisMac)
    }

    func testAnAudioInstallOnThisMacIsIdentifiedAsInternal() async throws {
        let controller = try makeController(
            report: report(installed: [("kokoro-fixture", nil)]), profiles: [profile("kokoro-fixture")])
        await settle(controller)

        let subject = try XCTUnwrap(controller.moveSubject(for: profile("kokoro-fixture")))

        XCTAssertEqual(subject.installId, "internal:audio/kokoro-fixture")
        XCTAssertTrue(subject.isThisMac)
    }

    func testAnInstallWithNoReceiptOffersNoMove() async throws {
        let controller = try makeController(
            report: report(needsAdoption: ["kokoro-fixture"]), profiles: [profile("kokoro-fixture")])
        await settle(controller)

        XCTAssertEqual(controller.needsAdoption.map(\.record.alias), ["kokoro-fixture"], "the fixture is an unadopted install")
        XCTAssertNil(
            controller.moveSubject(for: profile("kokoro-fixture")),
            "the engine refuses an unadopted install into a library: Adopt comes first")
    }

    func testAProfileWithNothingInstalledOffersNoMove() async throws {
        let controller = try makeController(report: report(), profiles: [profile("kokoro-fixture")])
        await settle(controller)

        XCTAssertNil(controller.moveSubject(for: profile("kokoro-fixture")))
    }

    func testAnotherProfilesInstallIsNotThisProfilesMove() async throws {
        let controller = try makeController(
            report: report(installed: [("other-model", "t7-id")]),
            profiles: [profile("kokoro-fixture"), profile("other-model")])
        await settle(controller)

        XCTAssertNil(controller.moveSubject(for: profile("kokoro-fixture")))
        XCTAssertNotNil(controller.moveSubject(for: profile("other-model")))
    }
}
