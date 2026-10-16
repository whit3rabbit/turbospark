import CoreML
import FluidAudio
import Foundation

/// The text of one window and, when the model reports them, token timings in
/// seconds from the window's start.
struct FluidAudioTranscript: Sendable, Equatable {
    struct Token: Sendable, Equatable {
        let text: String
        let start: Double
        let end: Double
    }

    let text: String
    let tokens: [Token]
}

/// A loaded FluidAudio speech recognizer. Built only from a verified install
/// whose models passed their signature check. Input is mono 16 kHz.
protocol FluidAudioTranscriber: Sendable {
    /// The longest window `transcribe` accepts, in samples. Nil when the
    /// recognizer splits longer input itself.
    var maxWindowSamples: Int? { get }
    func transcribe(_ samples: [Float]) async throws -> FluidAudioTranscript
}

extension FluidAudioEngine {
    /// Loads the recognizer for a speech family from a verified install. Not
    /// cached: residency belongs to the workspace, which owns load and unload.
    func transcriber(for family: FluidAudioFamily, installRoot: URL) async throws -> any FluidAudioTranscriber {
        Self.enforceOffline()
        switch family {
        case .parakeetJa: return try ParakeetJaTranscriber(root: installRoot)
        case .paraformerZh: return try ParaformerTranscriber(root: installRoot)
        case .sileroVAD, .diarizer, .kokoroANE, .pocketTTS, .senseVoice:
            throw FluidAudioModelError.notSpeech(family)
        }
    }
}

/// Parakeet TDT 0.6B Japanese. Placement measured: the encoder on the Neural
/// Engine, decoder and joint on the CPU. The preprocessor runs on the CPU, as
/// FluidAudio loads it.
struct ParakeetJaTranscriber: FluidAudioTranscriber {
    private let manager: AsrManager

    init(root: URL) throws {
        let ane = MLComputeUnits.cpuAndNeuralEngine
        let models = AsrModels(
            encoder: try FluidAudioModelLoader.load(FluidAudioSignatures.ParakeetJa.encoder, in: root, units: ane),
            preprocessor: try FluidAudioModelLoader.load(
                FluidAudioSignatures.ParakeetJa.preprocessor, in: root, units: .cpuOnly),
            decoder: try FluidAudioModelLoader.load(FluidAudioSignatures.ParakeetJa.decoder, in: root, units: ane),
            joint: try FluidAudioModelLoader.load(FluidAudioSignatures.ParakeetJa.joint, in: root, units: ane),
            configuration: AsrModels.defaultConfiguration(),
            vocabulary: try FluidAudioModelLoader.vocabulary(in: root),
            version: .tdtJa)
        manager = AsrManager(config: .default, models: models)
    }

    /// FluidAudio splits input over its 15 s model window into overlapping
    /// chunks and merges them.
    var maxWindowSamples: Int? { nil }

    func transcribe(_ samples: [Float]) async throws -> FluidAudioTranscript {
        // A fresh decoder state per call: each window stands alone, and the
        // workspace reconciles window boundaries itself.
        var state = TdtDecoderState.make(decoderLayers: AsrModelVersion.tdtJa.decoderLayers)
        let result = try await manager.transcribe(samples, decoderState: &state)
        let tokens = (result.tokenTimings ?? []).map {
            FluidAudioTranscript.Token(text: $0.token, start: $0.startTime, end: $0.endTime)
        }
        return FluidAudioTranscript(text: result.text, tokens: tokens)
    }
}

/// Paraformer large (Mandarin), int8 encoder and decoder on the Neural
/// Engine, waveform preprocessor on the CPU. Text only: no token timings.
struct ParaformerTranscriber: FluidAudioTranscriber {
    private let manager: ParaformerManager

    init(root: URL) throws {
        let ane = MLComputeUnits.cpuAndNeuralEngine
        let models = ParaformerModels(
            preprocessor: try FluidAudioModelLoader.load(
                FluidAudioSignatures.Paraformer.preprocessor, in: root, units: .cpuOnly),
            encoder: try FluidAudioModelLoader.load(FluidAudioSignatures.Paraformer.encoder, in: root, units: ane),
            cifAlphas: try FluidAudioModelLoader.load(FluidAudioSignatures.Paraformer.cifAlphas, in: root, units: ane),
            decoder: try FluidAudioModelLoader.load(FluidAudioSignatures.Paraformer.decoder, in: root, units: ane),
            vocabulary: try FluidAudioModelLoader.vocabulary(in: root))
        manager = ParaformerManager(models: models)
    }

    /// FluidAudio cuts Paraformer input past 512 encoder frames (about 30.7 s)
    /// and output past 128 tokens, logging but not throwing, so a long window
    /// loses its end. 15 s leaves room for 8.5 characters a second. Longer
    /// windows are refused, never shortened.
    static let windowSamples = 15 * 16_000
    var maxWindowSamples: Int? { Self.windowSamples }

    func transcribe(_ samples: [Float]) async throws -> FluidAudioTranscript {
        guard samples.count <= Self.windowSamples else {
            throw FluidAudioModelError.windowTooLong(samples: samples.count, limit: Self.windowSamples)
        }
        return FluidAudioTranscript(text: try await manager.transcribe(audio: samples), tokens: [])
    }
}
