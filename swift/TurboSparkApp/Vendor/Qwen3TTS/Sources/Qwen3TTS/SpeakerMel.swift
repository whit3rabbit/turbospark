import MLX
import Qwen3TTSCore
import Foundation

// The speaker encoder uses log magnitude mels, never Whisper normalization.
// Pinned Python: mlx-audio e1b19b9, qwen3_tts.py:mel_spectrogram.
func speakerMel(_ audio: MLXArray) -> MLXArray {
    let padding = 384
    let left = audio[MLXArray((1...padding).reversed().map(Int32.init))]
    let right = audio[MLXArray(((audio.size - padding - 1)..<(audio.size - 1)).reversed().map(Int32.init))]
    let padded = concatenated([left, audio, right])
    let frames = asStrided(padded, [1 + (padded.size - 1024) / 256, 1024], strides: [256, 1])
    let window = MLXArray((0..<1024).map { Float(0.5 * (1 - cos(2 * Double.pi * Double($0) / 1023))) })
    let spectrum = MLXFFT.rfft(frames * window, axis: 1)
    let magnitude = sqrt(abs(spectrum).square() + Float(1e-9))
    let filters = melFilters(sampleRate: 24_000, nFft: 1024, nMels: 128, norm: "slaney", melScale: .slaney)
    return log(maximum(matmul(magnitude, filters), Float(1e-5)))
}
