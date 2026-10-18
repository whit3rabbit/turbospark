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

## Coupled mode (Frozen E)

`NemotronDiarization::coupled` runs the diarizer inside NeMo's multitalker
ASR loop (Speech 218d0b4 `configure_diar_streaming`): it takes the ASR's own
log-mel chunk minus the ASR pre-encode cache instead of computing its own
features, uses one 14-frame chunk per ASR step with no left or right
context, `fifo_len` 264 and `spkcache_update_period` 222, and returns 80 ms
predictions (the mean of each 8-frame group of the 10 ms head). The session
holds its own geometry, so `feed` callers are unaffected, and a regression
digest of `generate_stream` was captured before the refactor.

`coupled_chunk_preds_match_nemo` feeds NeMo's features for 54 steps of an
AMI clip (through the first FIFO overflow and AOSC compression) and matches
NeMo's chunk predictions within 2.3e-6 with identical cache and FIFO lengths.
Use weights converted from the pinned `.nemo` at `f667ed73`
(`scripts/convert_multitalker_nemo.py convert-diar`). The
`mlx-community/Nemotron-3-Diarization` conversion comes from revision
`723e19c6`.

`CoupledDiarizer::with_executor` swaps `encoder_forward` for an
`EncoderExecutor`, which is how `crates/runtime/src/nemotron_diar_metal.rs`
runs the encoder stack on Metal in f32. That path is tolerance-gated, not
bit-exact: `crates/runtime/tests/nemotron_diar_metal.rs` holds the encoder
to 1e-4 of RMS and the coupled chunk predictions to 1e-4 of NeMo with no
0.5 decision flips. The CPU path stays the default and the reference.

## Example Usage

Run the Nemotron diarization demo:

```sh
cargo run -p turbospark-audio --example nemotron_diarize -- <model-dir> <audio.wav>
```
