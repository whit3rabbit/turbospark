import FluidAudio
import Foundation

/// The only owner of FluidAudio in the app. Left alone, FluidAudio downloads
/// from Hugging Face `main` and deletes a model folder after a failed load.
/// Installs here come from the pinned Rust audio catalog instead, so this
/// type turns every network path off before FluidAudio does any work:
///
/// - `ModelHub.offlineMode` stops `ModelHub` downloads, and with them the
///   delete-and-download-again on a failed load.
/// - `ModelRegistry.baseURL` points `AssetDownloader`, which ignores offline
///   mode (Kokoro voice packs, PocketTTS voices), at a scheme URLSession
///   refuses, so a missing file fails at once instead of fetching `main`.
///
/// Only files under `Audio/FluidAudio/` import FluidAudio
/// (`FluidAudioIsolationTests`), and every use goes through this type.
/// docs/FLUIDAUDIO.md records why each switch exists.
actor FluidAudioEngine {
    static let shared = FluidAudioEngine()

    /// A registry base URL no session can fetch from.
    static let blockedRegistryURL = "turbospark-offline:"

    /// Runs once per process, on first use, from whichever thread gets there.
    private static let offline: Void = {
        ModelHub.offlineMode = true
        ModelRegistry.baseURL = blockedRegistryURL
    }()

    /// Sets the offline switches if they are not set yet. Every entry point
    /// calls it before FluidAudio does any work.
    nonisolated static func enforceOffline() { _ = offline }

    /// Whether FluidAudio would refuse to reach the network now.
    nonisolated static var isOfflineEnforced: Bool {
        ModelHub.offlineMode && ModelRegistry.baseURL == blockedRegistryURL
    }

    private init() { Self.enforceOffline() }
}
