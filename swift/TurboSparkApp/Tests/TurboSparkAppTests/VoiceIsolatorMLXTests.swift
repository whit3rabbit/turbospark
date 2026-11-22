import Foundation
import MLX
import Metal
import TurboSpark
import VoiceIsolatorMLX
import XCTest

@testable import TurboSparkApp

/// The offline Voice Isolator runs on the vendored MLX port; every other
/// enhancement row, and an MLX load that fails, keeps the Rust session.
@MainActor
final class VoiceIsolatorMLXTests: XCTestCase {
    private func profile(_ alias: String, family: String = "voice_isolator") -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .enhancement, alias: alias, repository: "inworld/voice-isolator",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: "f"),
            family: family, displayName: alias,
            capabilities: AudioCapabilities(operations: ["enhance"], backend: "metal", cancellation: "token", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 48_000, channels: 1), readiness: "qualified")
    }

    private func fixture() throws -> (AudioWorkspaceController, model: URL, root: URL) {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "MLX needs a Metal device")
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "voice-isolator-mlx" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let model = root.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        return (controller, model, root)
    }

    /// Of the catalog's rows, only the offline Voice Isolator moves to MLX.
    /// Its provider stays Rust, so pickers, Clean reference's choice and the
    /// CLI-facing catalog verdict are unchanged.
    func testOnlyTheOfflineVoiceIsolatorRunsOnMLX() throws {
        let profiles = try AudioCatalog.profiles()
        let enhancement = profiles.filter { $0.identity.task == .enhancement }
        XCTAssertTrue(enhancement.contains { $0.identity.alias == "voice-isolator-v1-realtime" }, "the check must see the realtime row")
        XCTAssertTrue(enhancement.contains { $0.family != "voice_isolator" }, "the check must see another enhancement family")
        let mlx = profiles.filter(VoiceIsolatorMLXModel.handles).map(\.identity.alias)
        XCTAssertEqual(mlx, ["voice-isolator-v1"])
        for p in profiles where VoiceIsolatorMLXModel.handles(p) {
            XCTAssertEqual(AudioRuntimeProvider(p), .rust, p.identity.alias)
        }
    }

    /// A profile this engine does not run never reaches its loader.
    func testOtherEnhancementProfilesNeverLoadMLX() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in XCTFail("must not verify"); return model.path }
        controller.voiceIsolatorLoadDependencies.load = { _ in XCTFail("must not load"); throw CancellationError() }
        for p in [profile("voice-isolator-v1-realtime"), profile("mossformer2-se-48k-fp16", family: "mossformer2_se")] {
            controller.installedPaths[p.identity.alias] = model.path
            let isolator = try await controller.offlineVoiceIsolator(for: p)
            XCTAssertNil(isolator, p.identity.alias)
        }
    }

    /// An MLX failure falls back: nil, nothing resident, no opening left.
    func testAFailedMLXLoadFallsBackToTheRustSession() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        struct Broken: Error {}
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in model.path }
        controller.voiceIsolatorLoadDependencies.load = { _ in throw Broken() }
        controller.installedPaths[VoiceIsolatorMLXModel.alias] = model.path
        let isolator = try await controller.offlineVoiceIsolator(for: profile(VoiceIsolatorMLXModel.alias))
        XCTAssertNil(isolator)
        XCTAssertNil(controller.voiceIsolatorModel)
        XCTAssertNil(controller.opening)
        XCTAssertNil(controller.loadedModelAlias)
    }

    /// A failed verification is not an MLX failure: it propagates instead of
    /// quietly opening the Rust session on the same unverified bytes.
    func testAVerificationFailureIsNotMaskedByTheFallback() async throws {
        let (controller, model, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        struct Corrupt: Error {}
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in throw Corrupt() }
        controller.voiceIsolatorLoadDependencies.load = { _ in XCTFail("an unverified install must not load"); throw CancellationError() }
        controller.installedPaths[VoiceIsolatorMLXModel.alias] = model.path
        do {
            _ = try await controller.offlineVoiceIsolator(for: profile(VoiceIsolatorMLXModel.alias))
            XCTFail("a verification failure must propagate")
        } catch {
            XCTAssertTrue(error is Corrupt, "\(error)")
        }
        XCTAssertNil(controller.voiceIsolatorModel)
    }

    // MARK: Real checkpoint

    /// `TURBOSPARK_VOICE_ISOLATOR_MODEL` names the folder the Rust parity
    /// test uses: `denoiser.safetensors` plus `real_clip.npy` (10 s of speech
    /// in white noise, 48 kHz) and `real_voice.npy`, the torch CPU output for
    /// it (`crates/audio/tools/gen_voice_isolator_real_reference.py`).
    private func realFolder() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_VOICE_ISOLATOR_MODEL"], !path.isEmpty else {
            throw XCTSkip("TURBOSPARK_VOICE_ISOLATOR_MODEL is not set")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    /// Reads a 1-D little-endian float32 `.npy` (format 1.0).
    private func readNpy(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.prefix(6), Data([0x93]) + Data("NUMPY".utf8))
        let headerLength = Int(data[8]) | Int(data[9]) << 8
        let header = String(decoding: data[10 ..< 10 + headerLength], as: UTF8.self)
        XCTAssertTrue(header.contains("'<f4'") && header.contains("False"), header)
        return data.dropFirst(10 + headerLength).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// The library at float32 against the torch CPU reference, and the app's
    /// float16 engine (through the controller's real load) against both.
    func testRealCheckpointMatchesTheTorchReference() async throws {
        let folder = try realFolder()
        let clip = try readNpy(folder.appendingPathComponent("real_clip.npy"))
        let reference = try readNpy(folder.appendingPathComponent("real_voice.npy"))
        XCTAssertEqual(clip.count, reference.count)

        let fp32 = try VoiceIsolator(contentsOf: folder.appendingPathComponent(VoiceIsolatorMLXModel.weightFile), dtype: .float32)
        let y32 = try fp32.isolate(clip, sampleRate: 48_000)
        let maxAbs32 = zip(y32, reference).map { abs($0 - $1) }.max() ?? .infinity
        print("voice isolator MLX fp32 vs torch CPU: max abs \(maxAbs32)")
        XCTAssertLessThan(maxAbs32, 2e-6)

        let (controller, _, root) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        // The folder is not a managed install, so stand in for the catalog's
        // byte check; the MLX load itself is the real one.
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in folder.path }
        controller.installedPaths[VoiceIsolatorMLXModel.alias] = folder.path
        let engine = try await controller.offlineVoiceIsolator(for: profile(VoiceIsolatorMLXModel.alias))
        let isolator = try XCTUnwrap(engine, "the real checkpoint must load on MLX")
        XCTAssertEqual(controller.loadedModelAlias, VoiceIsolatorMLXModel.alias)

        final class Seen: @unchecked Sendable { var last = 0; let lock = NSLock() }
        let seen = Seen()
        var request = AudioRequest(task: .enhancement)
        request.inputSampleRate = 48_000
        let result = try await isolator.enhance(request, pcm: clip) { update in
            seen.lock.lock(); seen.last = max(seen.last, update.completed); seen.lock.unlock()
        }
        let pcm = try XCTUnwrap(result.pcm)
        XCTAssertEqual(pcm.format, AudioPCMFormat(sampleRate: 48_000, channels: 1))
        XCTAssertEqual(pcm.samples.count, clip.count)
        XCTAssertEqual(seen.last, 1000)
        var signal = 0.0, error = 0.0
        for (a, b) in zip(reference, pcm.samples) {
            signal += Double(a) * Double(a); error += Double(a - b) * Double(a - b)
        }
        let snr = 10 * log10(signal / error)
        print("voice isolator MLX fp16 vs torch CPU: SNR \(snr) dB")
        // Upstream's own fp16 gate (ModelTests) is 40 dB against fp32.
        XCTAssertGreaterThan(snr, 40)

        // Refuses a rate the callers never send, rather than mislabel it.
        request.inputSampleRate = 16_000
        do { _ = try await isolator.enhance(request, pcm: clip) { _ in }; XCTFail("a 16 kHz job must be refused") } catch {}
        controller.unloadModel()
        XCTAssertNil(controller.voiceIsolatorModel)
    }
}
