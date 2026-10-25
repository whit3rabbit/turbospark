import XCTest
import TurboSpark
@testable import TurboSparkApp

final class AudioDiscoverCatalogTests: XCTestCase {
    private func profile(
        _ alias: String,
        task: AudioTask = .speechToText,
        family: String = "parakeet_tdt",
        operations: [String],
        backend: String = "coreml",
        canRun: Bool = true,
        downloadBytes: UInt64 = 100
    ) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: alias, repository: "FluidInference/\(alias)", revision: "revision", assetFingerprint: "fingerprint"),
            family: family,
            displayName: alias,
            capabilities: AudioCapabilities(operations: operations, backend: backend, cancellation: "window", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1),
            downloadBytes: downloadBytes,
            readiness: "qualified")
    }

    private var profiles: [AudioProfile] {
        [
            profile("parakeet-unified-en-coreml", family: "parakeet_unified", operations: ["stream_transcribe"]),
            profile("parakeet-ultra-coreml", operations: ["transcribe"], downloadBytes: 300),
            profile("parakeet-redux-coreml", operations: ["transcribe"], downloadBytes: 200),
            profile("speaker-diarization-coreml", task: .diarization, family: "pyannote_community1", operations: ["diarize"]),
            profile("parakeet-ultra-unsupported", operations: ["transcribe"], canRun: false),
            profile("parakeet-redux-wrong-operation", operations: ["stream_transcribe"]),
        ]
    }

    func testListsTheFourRunnableProfilesInTheirJobGroups() {
        let entries = AudioDiscoverCatalog.entries(profiles: profiles, canRun: \.capabilities.canRun)
        XCTAssertEqual(entries.map(\.id), [
            "parakeet-unified-en-coreml",
            "parakeet-ultra-coreml",
            "parakeet-redux-coreml",
            "speaker-diarization-coreml",
        ])
        XCTAssertEqual(AudioDiscoverCatalog.grouped(entries).map(\.group), [.liveText, .finalTranscript, .speakers])
        XCTAssertEqual(AudioDiscoverCatalog.grouped(entries).map { $0.entries.map(\.id) }, [
            ["parakeet-unified-en-coreml"],
            ["parakeet-ultra-coreml", "parakeet-redux-coreml"],
            ["speaker-diarization-coreml"],
        ])
        XCTAssertEqual(entries.first?.installedEntry.group, .available)
        XCTAssertEqual(entries.first?.installedEntry.profile?.identity.alias, "parakeet-unified-en-coreml")
    }

    func testSearchMatchesAliasesAndTaskNames() {
        XCTAssertEqual(
            AudioDiscoverCatalog.entries(profiles: profiles, searchText: "redux", canRun: \.capabilities.canRun).map(\.id),
            ["parakeet-redux-coreml"])
        XCTAssertEqual(
            AudioDiscoverCatalog.entries(profiles: profiles, searchText: "speakers", canRun: \.capabilities.canRun).map(\.id),
            ["speaker-diarization-coreml"])
    }

    func testSortBySizeChangesOrderWithoutChangingRecommendations() {
        let entries = AudioDiscoverCatalog.entries(profiles: profiles, sort: .size, canRun: \.capabilities.canRun)
        XCTAssertEqual(entries.map(\.id), [
            "parakeet-unified-en-coreml",
            "speaker-diarization-coreml",
            "parakeet-redux-coreml",
            "parakeet-ultra-coreml",
        ])
    }
}
