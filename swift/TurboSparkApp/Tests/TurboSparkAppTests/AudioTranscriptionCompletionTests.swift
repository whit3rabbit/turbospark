import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The final-pass completion both transcription engines share: speaker
/// labels are tried whenever a diarization model is installed, a label
/// failure is reported without failing the saved transcript, and the take
/// ends completed either way. The FluidAudio path used to skip the labels.
@MainActor
final class AudioTranscriptionCompletionTests: XCTestCase {
    private var root: URL!
    private var controller: AudioWorkspaceController!

    override func setUp() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        self.root = root
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "completion" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        // The init refresh would overwrite profiles and paths a test sets.
        let deadline = Date().addingTimeInterval(20)
        while controller.isRefreshingModels || controller.profiles.isEmpty {
            guard Date() < deadline else { return XCTFail("the audio catalog never finished loading") }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
    }

    override func tearDown() {
        controller?.shutdown()
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    /// A one second recording and its transcription take with one final
    /// revision, as `transcribe` leaves them before completion.
    private func savedTake() throws -> (source: AudioLibraryItem, result: AudioLibraryItem, revision: UUID) {
        let wav = try AudioMediaIO.wav(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000, channels: 1)
        let asset = try controller.assets.store(data: wav, fileName: "take.wav", mimeType: "audio/wav")
        var source = AudioLibraryItem(title: "Take", kind: .recording)
        source.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: 16_000)]
        source.status = .completed
        try controller.save(source)
        var result = AudioLibraryItem(title: "Take - Transcribe", kind: .transcription)
        result.sourceID = source.id
        result.status = .processing
        let revision = result.addTranscript([AudioLibrarySegment(start: 0, end: 1, text: "hello", timing: "none")], source: .final)
        try controller.save(result)
        return (source, result, revision)
    }

    private var labelFailurePrefix: String { AppStrings.localized("Transcript saved. Speakers could not be labeled: \("")") }

    func testCompletesWithoutLabelsWhenNoDiarizationModelIsInstalled() async throws {
        let diarizers = Set(controller.profiles.filter { $0.identity.task == .diarization }.map(\.identity.alias))
        controller.installedPaths = controller.installedPaths.filter { !diarizers.contains($0.key) }
        controller.preferences.localModels = controller.preferences.localModels.filter { !diarizers.contains($0.key) }
        XCTAssertNil(controller.installedDiarizationProfile)
        let (source, result, revision) = try savedTake()

        try await controller.completeFinalTranscription(
            item: source, into: result.id, transcription: .init(revision: revision, words: []), token: controller.epoch)

        XCTAssertEqual(controller.status, AppStrings.localized("Saved in this profile"))
        XCTAssertEqual(try controller.library.item(result.id)?.status, .completed)
    }

    /// With a diarization model installed, completion tries to label and
    /// reports the failure in the status strip rather than dropping it.
    func testTriesSpeakerLabelsAndReportsTheirFailure() async throws {
        let diarizer = AudioProfile(
            identity: AudioProfileIdentity(task: .diarization, alias: "test-diarizer", repository: "x/y",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: "f"),
            family: "pyannote_community1", displayName: "Test diarizer",
            capabilities: AudioCapabilities(maxInputSeconds: 600, operations: ["diarize"], backend: "coreml",
                                            cancellation: "window", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "qualified")
        controller.profiles.insert(diarizer, at: 0)
        // Installed as far as the controller knows, but nothing is there, so
        // opening it fails the way a broken install would.
        controller.installedPaths["test-diarizer"] = root.appendingPathComponent("missing-diarizer").path
        XCTAssertEqual(controller.installedDiarizationProfile?.identity.alias, "test-diarizer")
        let (source, result, revision) = try savedTake()

        try await controller.completeFinalTranscription(
            item: source, into: result.id, transcription: .init(revision: revision, words: []), token: controller.epoch)

        XCTAssertTrue(controller.status.hasPrefix(labelFailurePrefix), controller.status)
        XCTAssertEqual(try controller.library.item(result.id)?.status, .completed, "a label failure never fails the saved transcript")
    }
}
