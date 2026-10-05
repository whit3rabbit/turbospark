# Nemotron 3 Diarization

Nemotron 3 Diarization is an advanced multi-speaker diarization architecture developed by NVIDIA NeMo and ported from `mlx-audio`, designed for high-resolution streaming diarization of up to eight concurrent speakers at 10 ms granularity.

## Upstream References and URLs

- **Hugging Face Repository**: [nvidia/nemotron-diarization](https://huggingface.co/nvidia/nemotron-diarization)
- **MLX Conversion**: [mlx-community/nemotron-diarization-mlx](https://huggingface.co/mlx-community/nemotron-diarization-mlx)
- **Upstream Reference**: [`mlx_audio/vad/models/nemotron_diarization/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/vad/models/nemotron_diarization) in mlx-audio v0.5.7
- **NeMo Reference**: NVIDIA NeMo Speech Diarization Suite

## Model Overview

Nemotron 3 Diarization replaces convolutional subsampling with a linear feature-stacking approach and employs a 31-layer RoPE Transformer combined with a subpixel upsampling convolutional head.

```text
PCM Audio (16 kHz) -> 80-bin Mel Spectrogram (Pre-emphasis + Centered STFT)
                              │
                              ▼
Feature-Stacking Subsampling (8 x 10 ms mel frames = 80 ms per token)
                              │
                              ▼
31-Layer RoPE Transformer Backbone
                              │
                              ▼
Speaker Prediction Head (Subpixel Upsampling Conv + Sigmoid) -> `[frames, 8]`
```

### Architecture Specifications

| Property | Value |
|---|---|
| Sample Rate | 16,000 Hz |
| Max Speakers | 8 simultaneous active speakers |
| Resolution | 10 ms time resolution output |
| Subsampling | 8x temporal grouping via linear projection |
| Backbone | 31-layer Transformer with Rotary Position Embeddings (NeoX RoPE) |
| Output Head | Subpixel 1D convolution reconstructing original 10 ms resolution |
| Cache Engine | Sortformer AOSC (Attentive Online Speaker Cache) with FIFO memory |

## Performance and Benchmarks

- **Output Granularity**: 100 frames/sec (10 ms resolution per prediction).
- **Speaker Capacity**: Robust detection across 1 to 8 concurrent speakers with speech overlap support.
- **Verification Gates**:
  - Deterministic stage fixtures: Feature stacking order, Hann window padding, NeoX RoPE tables, subpixel head reshape, and AOSC compression match reference values.
  - Full-window and streaming feed methods verified against MLX golden outputs.

## Example Usage

Run the Nemotron diarization demo:

```sh
cargo run -p turbospark-audio --example nemotron_diarize -- <model-dir> <audio.wav>
```
