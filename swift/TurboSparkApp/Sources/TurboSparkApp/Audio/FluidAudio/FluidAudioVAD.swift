import CoreML
import FluidAudio
import Foundation

/// One stretch of speech, in seconds from the start of the audio.
struct FluidAudioSpeechSegment: Sendable, Equatable {
    let start: Double
    let end: Double
}

/// A loaded FluidAudio voice activity detector. Built only from a verified
/// install whose model passed its signature check. Input is mono 16 kHz.
protocol FluidAudioSpeechDetector: Sendable {
    /// Where speech is in `samples`: ordered, within the audio, and no longer
    /// than `maxSegmentSeconds` each. Silence and empty input give none.
    func speechSegments(in samples: [Float], maxSegmentSeconds: Double) async throws -> [FluidAudioSpeechSegment]
}

extension FluidAudioSpeechDetector {
    /// FluidAudio's default cap, sized for a 15 s recognizer window.
    static var defaultMaxSegmentSeconds: Double { 14 }
}

extension FluidAudioEngine {
    /// Loads the detector for a speech-detection family from a verified
    /// install. Not cached: residency belongs to the workspace.
    func speechDetector(for family: FluidAudioFamily, installRoot: URL) async throws -> any FluidAudioSpeechDetector {
        Self.enforceOffline()
        switch family {
        case .sileroVAD: return try SileroSpeechDetector(root: installRoot)
        case .kokoroANE, .pocketTTS, .senseVoice, .paraformerZh, .parakeetJa:
            throw FluidAudioModelError.notDetector(family)
        }
    }
}

/// Silero VAD 6.2.1, unified 256 ms export. FluidAudio asks for the Neural
/// Engine and Core ML runs the model on the CPU (docs/FLUIDAUDIO_PHASE0.md),
/// so this loads it for the CPU and skips a Neural Engine compile that
/// places nothing there.
///
/// The model goes in through `VadManager(config:vadModel:)`, never the
/// initializers that take a folder: those go through `ModelHub`.
struct SileroSpeechDetector: FluidAudioSpeechDetector {
    private let manager: VadManager

    init(root: URL) throws {
        let model = try FluidAudioModelLoader.load(FluidAudioSignatures.SileroVad.model, in: root, units: .cpuOnly)
        manager = VadManager(config: VadConfig(computeUnits: .cpuOnly), vadModel: model)
    }

    func speechSegments(in samples: [Float], maxSegmentSeconds: Double) async throws -> [FluidAudioSpeechSegment] {
        guard !samples.isEmpty else { return [] }
        // FluidAudio's defaults for everything but the cap: 0.15 s minimum
        // speech, 0.75 s of silence to end a segment, 0.1 s padding.
        let config = VadSegmentationConfig(maxSpeechDuration: maxSegmentSeconds)
        return try await manager.segmentSpeech(samples, config: config).map {
            FluidAudioSpeechSegment(start: $0.startTime, end: $0.endTime)
        }
    }
}
