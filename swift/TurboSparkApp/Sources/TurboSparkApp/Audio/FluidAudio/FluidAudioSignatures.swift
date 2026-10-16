import CoreML

/// The signature of every model each speech family loads, read from the
/// pinned exports' compiled descriptions (docs/FLUIDAUDIO.md, Families).
/// A pin bump re-reads them: an export whose inputs moved fails here, not
/// inside FluidAudio.
enum FluidAudioSignatures {
    private static let f32 = MLMultiArrayDataType.float32
    private static let f16 = MLMultiArrayDataType.float16
    private static let i32 = MLMultiArrayDataType.int32
    private static let one = [[1]]

    /// `parakeet-0.6b-ja-coreml@2952296f`: 15 s windows, TDT decoder.
    enum ParakeetJa {
        static let preprocessor = FluidAudioModelSignature(
            file: "Preprocessor.mlmodelc",
            inputs: [FluidAudioFeature("audio_signal", f32, [[1, 240_000]]), FluidAudioFeature("audio_length", i32, one)],
            outputs: [FluidAudioFeature("mel_features", f32), FluidAudioFeature("mel_length", i32)])
        static let encoder = FluidAudioModelSignature(
            file: "Encoder.mlmodelc",
            inputs: [FluidAudioFeature("mel_features", f32, [[1, 80, 1501]]), FluidAudioFeature("mel_length", i32, one)],
            outputs: [FluidAudioFeature("encoder", f32), FluidAudioFeature("encoder_length", i32)])
        static let decoder = FluidAudioModelSignature(
            file: "Decoderv2.mlmodelc",
            inputs: [
                FluidAudioFeature("targets", i32, [[1, 1]]), FluidAudioFeature("target_length", i32, one),
                FluidAudioFeature("h_in", f32, [[2, 1, 640]]), FluidAudioFeature("c_in", f32, [[2, 1, 640]]),
            ],
            outputs: [FluidAudioFeature("decoder", f32), FluidAudioFeature("h_out", f32), FluidAudioFeature("c_out", f32)])
        static let joint = FluidAudioModelSignature(
            file: "Jointerv2.mlmodelc",
            inputs: [FluidAudioFeature("encoder_step", f32, [[1, 1024, 1]]), FluidAudioFeature("decoder_step", f32, [[1, 640, 1]])],
            outputs: [
                FluidAudioFeature("token_id", i32), FluidAudioFeature("token_prob", f32), FluidAudioFeature("duration", i32),
                FluidAudioFeature("top_k_ids", i32), FluidAudioFeature("top_k_logits", f32),
            ])
    }

    /// The shape buckets the LFR feature encoders accept (post-LFR frames).
    private static func buckets(_ width: Int) -> [[Int]] {
        [128, 256, 512, 1024, 1800].map { [1, $0, width] }
    }

    /// Shared by SenseVoice and Paraformer: waveform in, LFR features out.
    static let lfrPreprocessor = { (file: String) in
        FluidAudioModelSignature(
            file: file, inputs: [FluidAudioFeature("waveform", f32)], outputs: [FluidAudioFeature("features", f32)])
    }

    /// `paraformer-large-zh-coreml@9b4faaa0`, int8 encoder and decoder.
    enum Paraformer {
        static let preprocessor = lfrPreprocessor("ParaformerPreprocessor.mlmodelc")
        static let encoder = FluidAudioModelSignature(
            file: "ParaformerEncoder_int8.mlmodelc",
            inputs: [FluidAudioFeature("speech", f32, buckets(560)), FluidAudioFeature("speech_lengths", i32, one)],
            outputs: [FluidAudioFeature("enc_out", f16)])
        static let cifAlphas = FluidAudioModelSignature(
            file: "ParaformerCifAlphas.mlmodelc",
            inputs: [FluidAudioFeature("enc_out", f32, buckets(512))],
            outputs: [FluidAudioFeature("alphas", f16)])
        static let decoder = FluidAudioModelSignature(
            file: "ParaformerDecoder_int8.mlmodelc",
            inputs: [
                FluidAudioFeature("enc", f32, [[1, 512, 512]]), FluidAudioFeature("elen", i32, one),
                FluidAudioFeature("ac", f32, [[1, 128, 512]]), FluidAudioFeature("tn", i32, one),
            ],
            outputs: [FluidAudioFeature("logits", f16)])
    }
}
