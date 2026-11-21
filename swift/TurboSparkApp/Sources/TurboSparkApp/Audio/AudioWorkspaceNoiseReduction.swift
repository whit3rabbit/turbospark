import Foundation
import TurboSpark

/// "Reduce background noise" for live capture: an opt-in, off by default,
/// that cleans the microphone before live text hears it
/// (`AudioLiveVoiceCleaner`). The saved recording is never cleaned.
extension AudioWorkspaceController {
    /// The realtime voice isolator install live cleaning opens.
    static let liveNoiseReductionAlias = AudioEnhanceStream.realtimeAlias

    var liveNoiseReductionProfile: AudioProfile? {
        profiles.first { $0.identity.alias == Self.liveNoiseReductionAlias }
    }

    /// The realtime install's folder, when it is installed and can run here.
    var liveNoiseReductionPath: String? {
        guard let profile = liveNoiseReductionProfile, profile.capabilities.canRun else { return nil }
        return installedPaths[profile.identity.alias]
    }

    /// Whether the setting is shown at all.
    var offersLiveNoiseReduction: Bool { liveNoiseReductionPath != nil }

    /// The saved choice. Optional in the store so older profiles decode; nil is off.
    var reduceLiveNoise: Bool {
        get { preferences.reduceLiveNoise ?? false }
        set { preferences.reduceLiveNoise = newValue; persistPreferences() }
    }

    /// Whether the next take cleans the microphone for live text.
    var reducesLiveNoise: Bool { reduceLiveNoise && offersLiveNoiseReduction }
}
