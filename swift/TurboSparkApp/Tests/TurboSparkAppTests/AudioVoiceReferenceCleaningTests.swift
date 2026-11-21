import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Clean reference on the Voice Cloning page: speech enhancement over the
/// reference region only, saved as a new `.enhancement` take that becomes the
/// reference. The source take is never changed, and consent is cleared
/// because the cleaned take is a new reference (swift/AGENTS.md).
final class AudioVoiceReferenceCleaningTests: XCTestCase {
    private var root: URL!

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    private func profile(task: AudioTask, alias: String, family: String, backend: String = "metal",
                         canRun: Bool = true, operations: [String] = ["enhance"]) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: alias, repository: "someone/" + family + "-Base-",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: alias),
            family: family, displayName: alias,
            capabilities: AudioCapabilities(languages: ["en"], operations: operations, backend: backend,
                                            cancellation: "chunk", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 48_000, channels: 1), readiness: "implemented_unqualified")
    }

    private var isolator: AudioProfile { profile(task: .enhancement, alias: "voice-isolator-v1", family: "voice_isolator") }
    private var realtime: AudioProfile { profile(task: .enhancement, alias: "voice-isolator-v1-realtime", family: "voice_isolator") }
    private var mossformer: AudioProfile { profile(task: .enhancement, alias: "mossformer2-se-48k-fp16", family: "mossformer2_se") }

    /// MegaTTS3 selected on Voice Cloning, so the speech settings clone. The
    /// alias must be a curated one (`curatedCloneAliases`): the page lists
    /// nothing else, and a made-up alias leaves the mode at custom voice.
    @MainActor private func makeController(extra: [AudioProfile]) async throws -> AudioWorkspaceController {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { self.root }, profileIDProvider: { "voice-reference-cleaning" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        controller.liveTranscription = false
        while controller.isRefreshingModels { try await Task.sleep(for: .milliseconds(10)) }
        let mega = profile(task: .textToSpeech, alias: "megatts3-f32", family: "megatts3", backend: "portable_cpu",
                           operations: ["synthesize", "voice_clone"])
        controller.profiles = [mega] + extra
        controller.selectPage(.voiceClone)
        controller.selectModel(mega.identity.alias)
        XCTAssertEqual(controller.speechSettings.mode, .clone)
        return controller
    }

    /// A 4 s imported take at 24 kHz, set as the reference trimmed to 1.5 s
    /// from 1 s, with consent given for that region.
    @MainActor private func consentedSource(_ controller: AudioWorkspaceController) throws -> AudioLibraryItem {
        let samples = [Float](repeating: 0.1, count: 96_000)
        let asset = try controller.assets.store(data: try AudioMediaIO.wav(samples: samples, sampleRate: 24_000, channels: 1), fileName: "source.wav")
        var source = AudioLibraryItem(title: "Noisy voice", kind: .imported)
        source.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 24_000, channels: 1, frameCount: Int64(samples.count))]
        try controller.library.save(source)
        controller.publish(source)
        controller.setVoiceReference(source)
        controller.setVoiceReferenceTrim(\.start, to: 1)
        controller.setVoiceReferenceTrim(\.duration, to: 1.5)
        controller.speechSettings.referenceText = "hello there"
        controller.setCloneConsent(true)
        XCTAssertTrue(controller.cloneConsentGiven)
        return source
    }

    private func wire(_ request: AudioRequest) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
    }

    // MARK: Request

    func testPlanReadsTheRegionAndSendsTheEnhancementRate() throws {
        let clip = AudioLibraryClip(assetReference: "a", sampleRate: 48_000, channels: 1, frameCount: 480_000)
        let reference = AudioVoiceReference(sourceID: UUID(), title: "t", clips: [clip], start: 2.5, duration: 4)
        let plan = AudioWorkspaceController.referenceCleaningPlan(for: reference)
        XCTAssertEqual(plan.request.task, .enhancement)
        XCTAssertEqual(plan.request.inputSampleRate, 48_000)
        XCTAssertEqual(plan.start, 2.5, "the region, not the start of the take")
        XCTAssertEqual(plan.duration, 4, "the region, not the whole 10 s take")
        XCTAssertEqual(plan.clips, [clip])
        let fields = try wire(plan.request)
        XCTAssertEqual(fields["task"] as? String, "enhancement")
        XCTAssertEqual(fields["input_sample_rate"] as? Int, 48_000)
        XCTAssertEqual(fields["consent"] as? Bool, false, "enhancement never carries clone consent")
        XCTAssertNil(fields["reference_sample_rate"])
    }

    // MARK: Visibility

    func testProfilePreferenceAndGating() {
        let all = [mossformer, realtime, isolator]
        func pick(_ profiles: [AudioProfile], installed: Set<String>, runnable: Bool = true) -> String? {
            AudioWorkspaceController.referenceCleaningProfile(in: profiles, runnable: runnable) {
                installed.contains($0.identity.alias)
            }?.identity.alias
        }
        XCTAssertNil(pick(all, installed: []), "nothing installed hides the action")
        XCTAssertEqual(pick(all, installed: Set(all.map(\.identity.alias))), "voice-isolator-v1")
        XCTAssertEqual(pick(all, installed: ["voice-isolator-v1-realtime", "mossformer2-se-48k-fp16"]), "voice-isolator-v1-realtime")
        XCTAssertEqual(pick(all, installed: ["mossformer2-se-48k-fp16"]), "mossformer2-se-48k-fp16")
        XCTAssertNil(pick(all, installed: Set(all.map(\.identity.alias)), runnable: false), "a runtime without enhancement hides it")
        let blocked = profile(task: .enhancement, alias: "voice-isolator-v1", family: "voice_isolator", canRun: false)
        XCTAssertNil(pick([blocked], installed: ["voice-isolator-v1"]), "a model this Mac cannot run hides it")
        let swiftHosted = profile(task: .enhancement, alias: "voice-isolator-v1", family: "voice_isolator", backend: "swift_coreml")
        XCTAssertNil(pick([swiftHosted], installed: ["voice-isolator-v1"]), "only a Rust session runs it")
        let other = profile(task: .enhancement, alias: "some-other-denoiser", family: "mossformer2_se")
        XCTAssertNil(pick([other], installed: ["some-other-denoiser"]), "only the listed installs")
    }

    @MainActor func testActionIsHiddenUntilAnEnhancementModelIsInstalled() async throws {
        let controller = try await makeController(extra: [isolator])
        defer { controller.shutdown() }
        _ = try consentedSource(controller)
        XCTAssertFalse(controller.offersReferenceCleaning)
        XCTAssertFalse(controller.canCleanVoiceReference)
        controller.cleanVoiceReference()
        XCTAssertNil(controller.jobTask, "nothing runs while hidden")
        controller.installedPaths[isolator.identity.alias] = root.path
        XCTAssertTrue(controller.offersReferenceCleaning)
        XCTAssertTrue(controller.canCleanVoiceReference)
    }

    // MARK: Cleaning

    @MainActor func testCleaningSavesADerivedTakeUsesItAndClearsConsent() async throws {
        let controller = try await makeController(extra: [mossformer, isolator])
        defer { controller.shutdown() }
        controller.installedPaths[isolator.identity.alias] = root.path
        controller.installedPaths[mossformer.identity.alias] = root.path
        let source = try consentedSource(controller)
        let original = try XCTUnwrap(controller.speechSettings.reference)
        var sent: (alias: String, request: AudioRequest, samples: Int)?
        controller.referenceCleaningRunner = { profile, request, pcm, _ in
            sent = (profile.identity.alias, request, pcm.count)
            return AudioResult(pcm: AudioPCM(format: AudioPCMFormat(sampleRate: 48_000, channels: 1), samples: pcm.map { $0 * 0.5 }))
        }
        controller.cleanVoiceReference()
        XCTAssertTrue(controller.isBusy)
        XCTAssertTrue(controller.isCleaningVoiceReference)
        await controller.jobTask?.value

        XCTAssertNil(controller.error)
        XCTAssertFalse(controller.isBusy)
        XCTAssertFalse(controller.isCleaningVoiceReference)
        XCTAssertEqual(sent?.alias, "voice-isolator-v1", "the best installed enhancer")
        XCTAssertEqual(sent?.request.task, .enhancement)
        XCTAssertEqual(sent?.request.inputSampleRate, 48_000)
        XCTAssertEqual(Double(sent?.samples ?? 0), 1.5 * 48_000, accuracy: 48, "only the 1.5 s region, read at 48 kHz")

        let cleaned = try XCTUnwrap(controller.items.first { $0.kind == .enhancement })
        XCTAssertNotEqual(cleaned.id, source.id)
        XCTAssertEqual(cleaned.sourceID, source.id)
        XCTAssertEqual(cleaned.status, .completed)
        XCTAssertEqual(cleaned.duration, 1.5, accuracy: 0.01)
        XCTAssertEqual(controller.items.first { $0.id == source.id }, source, "the source take is untouched")
        let saved = try XCTUnwrap(try controller.library.loadReporting().items.first { $0.id == source.id })
        XCTAssertEqual(saved.clips, source.clips, "and so is its saved record")
        XCTAssertEqual(saved.kind, .imported)
        XCTAssertEqual(saved.title, source.title)

        let reference = try XCTUnwrap(controller.speechSettings.reference)
        XCTAssertNotEqual(reference, original)
        XCTAssertEqual(reference.sourceID, cleaned.id, "the cleaned take is the reference")
        XCTAssertEqual(reference.clips, cleaned.clips)
        XCTAssertEqual(reference.start, 0)
        XCTAssertTrue(reference.isValid)
        XCTAssertEqual(controller.speechSettings.referenceText, "hello there", "the same words as the region")
        XCTAssertNil(controller.cloneOptions.consentedReference, "consent never carries over to a new reference")
        XCTAssertFalse(controller.cloneConsentGiven)
    }

    @MainActor func testCancelLeavesTheReferenceAndConsent() async throws {
        let controller = try await makeController(extra: [mossformer])
        defer { controller.shutdown() }
        controller.installedPaths[mossformer.identity.alias] = root.path
        _ = try consentedSource(controller)
        let original = controller.speechSettings
        let count = controller.items.count
        let (entered, signal) = AsyncStream<Void>.makeStream()
        controller.referenceCleaningRunner = { _, _, _, _ in
            signal.yield()
            try await Task.sleep(for: .seconds(30))
            return AudioResult()
        }
        controller.cleanVoiceReference()
        for await _ in entered { break }
        controller.cancel()
        await controller.jobTask?.value

        XCTAssertFalse(controller.isBusy)
        XCTAssertFalse(controller.isCleaningVoiceReference)
        XCTAssertNil(controller.error, "a cancel is not an error")
        XCTAssertEqual(controller.status, AppStrings.localized("Interrupted"))
        XCTAssertEqual(controller.speechSettings, original, "the old reference, transcript and consent stay")
        XCTAssertTrue(controller.cloneConsentGiven)
        XCTAssertEqual(controller.items.count, count, "no take is saved")
    }

    @MainActor func testFailureSurfacesTheErrorAndKeepsTheReference() async throws {
        let controller = try await makeController(extra: [mossformer])
        defer { controller.shutdown() }
        controller.installedPaths[mossformer.identity.alias] = root.path
        _ = try consentedSource(controller)
        let original = controller.speechSettings
        controller.referenceCleaningRunner = { _, _, _, _ in throw AudioWorkspaceError.message("engine refused") }
        controller.cleanVoiceReference()
        await controller.jobTask?.value
        XCTAssertEqual(controller.error, "engine refused")
        XCTAssertEqual(controller.status, AppStrings.localized("Failed"))
        XCTAssertEqual(controller.speechSettings, original)
        XCTAssertFalse(controller.items.contains { $0.kind == .enhancement })
    }
}
