import TurboSpark
import XCTest
@testable import TurboSparkApp

final class AudioEngineLabelTests: XCTestCase {
    private func profile(backend: String, compute: String? = nil, httpApi: Bool? = nil) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: "x", repository: "", revision: "", assetFingerprint: ""),
            family: "f", displayName: "X",
            capabilities: AudioCapabilities(operations: ["transcribe"], backend: backend, cancellation: "window",
                                            canRun: false, compute: compute, httpApi: httpApi),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "runtime_pending")
    }

    func testBackendAndMeasuredPlacementNameTheEngine() {
        let cases: [(String, String?, AudioEngineLabel)] = [
            ("coreml", "ane", .coreMLNeuralEngine), ("coreml", "gpu", .coreMLGPU),
            ("swift_coreml", "cpu", .coreMLCPU), ("swift_coreml", "mixed", .coreMLMixed),
            ("coreml", nil, .coreML), ("swift_coreml", "unknown", .coreML),
            ("swift_mlx", nil, .mlxGPU), ("metal", nil, .metalGPU), ("metal", "ane", .metalGPU),
            ("portable_cpu", nil, .cpu), ("experimental_metal_or_portable", nil, .experimental),
            ("experimental_metal_or_portable_cpu", nil, .experimental),
            ("unavailable", nil, .unavailable), ("", nil, .unavailable),
        ]
        for (backend, compute, expected) in cases {
            XCTAssertEqual(AudioEngineLabel(backend: backend, compute: compute), expected,
                           "\(backend) \(compute ?? "nil")")
        }
    }

    /// No shipped backend falls through to "No runtime" for want of a
    /// mapping: every profile, every discovered family, and the local
    /// Qwen3-ASR choice name a real engine.
    @MainActor
    func testEveryShippedBackendNamesAnEngine() throws {
        var backends = Set(try AudioCatalog.profiles().map(\.capabilities.backend))
        backends.formUnion(try AudioCatalog.capabilities().map(\.backend))
        backends.insert(AudioWorkspaceController.localQwenProfile.capabilities.backend)
        XCTAssertTrue(backends.contains("coreml") && backends.contains("metal"), "\(backends)")
        for backend in backends where backend != "unavailable" {
            XCTAssertNotEqual(AudioEngineLabel(backend: backend, compute: nil), .unavailable, backend)
        }
    }

    /// The catalog's measured placement crosses the native boundary into the
    /// label: Redux Parakeet's encoder runs on the GPU.
    func testCatalogPlacementReachesTheLabel() throws {
        let parakeet = try XCTUnwrap(AudioCatalog.profiles().first { $0.identity.alias == "parakeet-redux-coreml" })
        XCTAssertEqual(parakeet.capabilities.compute, "gpu")
        XCTAssertEqual(AudioEngineLabel(parakeet), .coreMLGPU)
    }

    func testProviderAndAppOnlyFollowTheCatalog() throws {
        let fluid = profile(backend: "swift_coreml", compute: "ane", httpApi: false)
        XCTAssertEqual(AudioEngineLabel.provider(for: fluid), "FluidAudio")
        XCTAssertTrue(AudioEngineLabel.isAppOnly(fluid))
        let rust = profile(backend: "metal", httpApi: true)
        XCTAssertEqual(AudioEngineLabel.provider(for: rust), "TurboSpark")
        XCTAssertFalse(AudioEngineLabel.isAppOnly(rust))
        // A payload from before the flag is never shown as app-only.
        XCTAssertFalse(AudioEngineLabel.isAppOnly(profile(backend: "metal")))
        // Shipped rows are app-only exactly when the app is their only runtime.
        for p in try AudioCatalog.profiles() {
            XCTAssertEqual(AudioEngineLabel.isAppOnly(p), p.capabilities.backend == "swift_coreml", p.identity.alias)
        }
    }
}
