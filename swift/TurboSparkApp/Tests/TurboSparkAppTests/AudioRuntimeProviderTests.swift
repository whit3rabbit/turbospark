import TurboSpark
import XCTest
@testable import TurboSparkApp

@MainActor
final class AudioRuntimeProviderTests: XCTestCase {
    private func profile(_ alias: String, family: String, backend: String,
                         task: AudioTask = .textToSpeech, canRun: Bool = false) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: alias, repository: "x/y",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: "f"),
            family: family, displayName: alias,
            capabilities: AudioCapabilities(operations: ["synthesize"], backend: backend,
                                            cancellation: "chunk", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "runtime_pending")
    }

    /// For every row the catalog ships, the provider gives the answer the
    /// family checks it replaced gave: Qwen3-TTS runs in Swift MLX, every
    /// Rust row in the Rust session with the Rust verdict. A FluidAudio row
    /// stays hidden until this build carries its adapter.
    func testCatalogRowsKeepTheirRuntimeAndVerdict() throws {
        let profiles = try AudioCatalog.profiles()
        XCTAssertFalse(profiles.isEmpty)
        XCTAssertTrue(profiles.contains { $0.family == "qwen3_tts" }, "the check must see a Swift MLX row")
        XCTAssertTrue(profiles.contains { $0.capabilities.backend == "swift_coreml" },
                      "the check must see a FluidAudio row")
        for p in profiles {
            let provider = AudioRuntimeProvider(p)
            if p.capabilities.backend == "swift_coreml" {
                let family = try XCTUnwrap(FluidAudioFamily(rawValue: p.family), p.identity.alias)
                let wired = family.supportedTasks.contains(p.identity.task)
                XCTAssertEqual(provider, wired ? .fluidAudio(family) : .unsupported, p.identity.alias)
                continue
            }
            XCTAssertEqual(provider == .qwenMLX, p.family == "qwen3_tts", p.identity.alias)
            XCTAssertNotEqual(provider, .unsupported, p.identity.alias)
            XCTAssertEqual(provider.canRun(p), p.family == "qwen3_tts" || p.capabilities.canRun, p.identity.alias)
        }
    }

    /// A Swift-hosted row reaches a runtime only through a known family with a
    /// job for the row's task. Its Rust `canRun`, false by design, never
    /// decides.
    func testSwiftHostedRowsRouteOnlyToAShippedAdapter() {
        XCTAssertEqual(AudioRuntimeProvider(profile("a", family: "other_tts", backend: "swift_mlx")), .unsupported)
        XCTAssertEqual(AudioRuntimeProvider(profile("b", family: "fluidaudio_unknown", backend: "swift_coreml")),
                       .unsupported)
        for family in FluidAudioFamily.allCases {
            let task = family.supportedTasks.first ?? .textToSpeech
            let p = profile(family.rawValue, family: family.rawValue, backend: "swift_coreml", task: task)
            let provider = AudioRuntimeProvider(p)
            XCTAssertEqual(provider, family.hasAdapter ? .fluidAudio(family) : .unsupported, family.rawValue)
            XCTAssertEqual(provider.canRun(p), family.hasAdapter, family.rawValue)
        }
    }

    /// A family routes only the tasks it has a job for. A row for another
    /// task is unsupported, so the picker hides it instead of letting
    /// `beginRun` refuse it after the user pressed Run.
    func testFluidAudioRoutesOnlyTasksTheFamilySupports() {
        let ja = FluidAudioFamily.parakeetJa.rawValue
        XCTAssertEqual(AudioRuntimeProvider(profile("ja-stt", family: ja, backend: "swift_coreml", task: .speechToText)),
                       .fluidAudio(.parakeetJa))
        XCTAssertEqual(AudioRuntimeProvider(profile("ja-tts", family: ja, backend: "swift_coreml", task: .textToSpeech)),
                       .unsupported)
        XCTAssertEqual(AudioRuntimeProvider(profile("kokoro", family: FluidAudioFamily.kokoroANE.rawValue,
                                                    backend: "swift_coreml", task: .textToSpeech)), .unsupported)
        XCTAssertEqual(AudioRuntimeProvider(profile("vad", family: FluidAudioFamily.sileroVAD.rawValue,
                                                    backend: "swift_coreml", task: .speechDetection)), .unsupported)
    }

    /// The supported tasks name exactly the jobs `beginRun` runs. Adding a
    /// task is the deliberate step that puts a family's rows in the picker,
    /// so it comes with the job and residency that run it, and with this
    /// test's update.
    func testSupportedTasksNameOnlyWiredJobs() {
        for family in FluidAudioFamily.allCases {
            switch family {
            case .parakeetJa, .paraformerZh:
                XCTAssertEqual(family.supportedTasks, [.speechToText], family.rawValue)
            case .sileroVAD, .kokoroANE, .pocketTTS, .senseVoice:
                XCTAssertTrue(family.supportedTasks.isEmpty, family.rawValue)
            }
            XCTAssertEqual(family.hasAdapter, !family.supportedTasks.isEmpty, family.rawValue)
        }
    }

    /// The picker never offers a row this build cannot run, so a catalog
    /// that is newer than the app cannot put an empty choice in front of the
    /// user. Shipped rows on the same page stay listed.
    func testPickerHidesRowsWithoutARuntime() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-ui" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let shipped = try AudioCatalog.profiles()
        let hidden = [
            profile("fluid-tts", family: FluidAudioFamily.pocketTTS.rawValue, backend: "swift_coreml"),
            profile("other-swift", family: "other_tts", backend: "swift_mlx"),
        ]
        XCTAssertTrue(hidden.allSatisfy { AudioRuntimeProvider($0) == .unsupported })
        controller.profiles = shipped + hidden
        controller.selectPage(.voiceover)
        let offered = Set(controller.availableProfiles.map(\.identity.alias))
        XCTAssertFalse(offered.isEmpty, "the page must still offer the shipped voices")
        XCTAssertTrue(offered.isDisjoint(with: hidden.map(\.identity.alias)))
    }
}
