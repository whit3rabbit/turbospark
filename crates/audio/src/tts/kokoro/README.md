# Kokoro TTS (82M)

Kokoro is an 82 million parameter Text-to-Speech (TTS) model family based on StyleTTS 2 with an iSTFTNet decoder, delivering natural, high-fidelity speech synthesis at 24 kHz.

## Upstream References and URLs

- **Hugging Face Repository**: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- **MLX Conversion**: [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
- **Upstream Reference**: [`mlx_audio/tts/models/kokoro/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/kokoro) in mlx-audio v0.5.7
- **Architecture Paper**: *StyleTTS 2: Towards High-Performance Style-Driven Text-to-Speech with Diffusion* (Li et al., 2023)
- **Vocoder Paper**: *iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Using Inverse Short-Time Fourier Transform* (Kaneko et al., 2022)

## Model Overview

Kokoro uses phoneme sequences and reference style vectors (or default voice profiles) to produce 24 kHz PCM audio.

```text
Phonemes -> PLBert -> Duration Predictor -> Duration Alignment
               │               │
               ▼               ▼
          Prosody Predictor (F0, Noise) -> Text Encoder -> iSTFTNet Vocoder -> 24 kHz Audio
```

### Architecture Specifications

| Property | Value |
|---|---|
| Parameters | ~82 million |
| Input Format | Phoneme sequence IDs (`tokenizers` / dictionary mapping) |
| Output Format | Single-channel 24 kHz IEEE f32 PCM |
| Text Encoder | PLBert (ALBERT architecture with weight-shared layers) |
| Alignment & Duration | Bidirectional LSTM duration predictor with AdaLayerNorm |
| Prosody Modeling | Joint F0 pitch and noise predictors conditioned on style embeddings |
| Vocoder | iSTFTNet: Multi-receptive field fusion (MRF) + inverse STFT (FFT 20, hop 5) |
| Total Upsample Ratio | 300x (10 * 6 * 5 samples per prosody frame) |

## Performance and Benchmarks

- **Runtime Target**: Real-time factor (RTF) < 0.20 on Apple Silicon M-series (CPU f32 software path).
- **Audio Output**: 24 kHz, 16-bit equivalent dynamic range.
- **Memory Footprint**: ~180 MiB resident memory with f32 weights.
- **Verification Gate**:
  - Deterministic alignment, duration prediction, and intermediate feature parity checked against Python/MLX golden tensors.
  - Phonation test: "Hello World" phoneme sequence correctly synthesizes intelligible 24 kHz audio.

## Example Usage

Run the end-to-end example:

```sh
cargo run -p turbospark-audio --example kokoro_demo -- <model-dir> <voice.safetensors> "həˈloʊ wɜːld" output.wav
```
