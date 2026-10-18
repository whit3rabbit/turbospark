import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Installed > Audio groups rows as the catalog's installed report does,
/// adds local folders, and offers only pinned models this Mac can run.
@MainActor
final class AudioInstalledCatalogTests: XCTestCase {
    private func profile(_ alias: String, task: AudioTask = .speechToText, revision: String = "abcdef0123") -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: alias, repository: "o/\(alias)", revision: revision, assetFingerprint: "f"),
            family: "whisper", displayName: alias,
            capabilities: AudioCapabilities(operations: ["transcribe"], backend: "metal", cancellation: "window", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "qualified")
    }

    /// The engine's records have no public initializer; the app decodes them.
    private func decode<T: Decodable>(_ type: T.Type, _ json: String) throws -> T {
        try JSONDecoder().decode(T.self, from: Data(json.utf8))
    }

    private func record(_ alias: String) -> String {
        #"{"alias":"\#(alias)","path":"/models/\#(alias)","family":"whisper","status":"verified","installBytes":1000}"#
    }

    private func identity(_ alias: String) -> String {
        #"{"task":"speech_to_text","alias":"\#(alias)","repository":"o/\#(alias)","revision":"abcdef0123","assetFingerprint":"f"}"#
    }

    func testRowsFollowTheInstalledReportThenDownloads() throws {
        let profiles = [
            profile("installed-model"), profile("adopt-me"), profile("broken"), profile("downloadable"),
            profile("music-model", task: .music), profile("cannot-run-here"), profile("no-revision", revision: ""),
            profile("enhancer", task: .enhancement),
        ]
        let installed = [try decode(AudioInstalledRecord.self, record("installed-model"))]
        let adoption = [try decode(AudioLegacyInstall.self, #"{"record":\#(record("adopt-me")),"identity":\#(identity("adopt-me"))}"#)]
        let incompatible = [try decode(AudioIncompatibleInstall.self, #"{"record":\#(record("broken")),"identity":null,"error":"bad receipt"}"#)]
        let other = [try decode(AudioInstalledRecord.self, record("retired-model"))]
        let entries = AudioInstalledCatalog.entries(
            profiles: profiles, installed: installed, needsAdoption: adoption, incompatible: incompatible, otherAudio: other,
            localModels: ["downloadable"], canRun: { $0.identity.alias != "cannot-run-here" })

        let groups = AudioInstalledCatalog.grouped(entries).map { ($0.group, $0.entries.map(\.alias)) }
        XCTAssertEqual(groups.map(\.0), [.installed, .local, .needsAdoption, .needsAttention, .otherAudio, .available])
        XCTAssertEqual(groups[0].1, ["installed-model"])
        XCTAssertEqual(groups[1].1, ["downloadable"], "a chosen folder is its own row")
        XCTAssertEqual(groups[2].1, ["adopt-me"])
        XCTAssertEqual(groups[3].1, ["broken"])
        XCTAssertEqual(entries.first { $0.group == .needsAttention }?.error, "bad receipt")
        XCTAssertEqual(groups[4].1, ["retired-model"])
        XCTAssertNil(entries.first { $0.group == .otherAudio }?.profile, "the catalog no longer lists it")
        // Speech-to-Text sorts before music by task order. Models on disk,
        // models this Mac cannot run, unpinned rows and tasks the engine
        // cannot open are never offered for download.
        XCTAssertEqual(groups[5].1, ["downloadable", "music-model"])
        XCTAssertEqual(entries.first { $0.alias == "installed-model" }?.installBytes, 1000)
    }

    func testRowIdentityIsUniqueAcrossGroups() {
        let entries = AudioInstalledCatalog.entries(
            profiles: [profile("downloadable")], installed: [], needsAdoption: [], incompatible: [], otherAudio: [],
            localModels: ["downloadable"], canRun: { _ in true })
        XCTAssertEqual(entries.count, 2)
        XCTAssertEqual(Set(entries.map(\.id)).count, 2, "a model with a local folder and a download is two rows")
    }
}
