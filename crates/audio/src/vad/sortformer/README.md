# Sortformer Speaker Diarization

Sortformer is a neural speaker diarization model capable of tracking multiple concurrent speakers (up to four speakers simultaneously) directly from audio features without requiring explicit clustering or embedding separation stages.

## Upstream References and URLs

- **Hugging Face Repository**: [nvidia/diar_streaming_sortformer_4spk](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk)
- **MLX Conversion**: [mlx-community/Sortformer-4spk-v1](https://huggingface.co/mlx-community/Sortformer-4spk-v1)
- **Upstream Reference**: [`mlx_audio/vad/models/sortformer/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/vad/models/sortformer) in mlx-audio v0.5.7
- **NeMo Reference**: NVIDIA NeMo `SortformerEncLabelModel`
- **Research Paper**: *Sortformer: Multi-Speaker Diarization via Self-Attention and Chunk-Wise Processing*

## Model Overview

Sortformer takes 16 kHz audio and produces frame-level speaker activity probabilities for up to 4 distinct speakers.

```text
PCM Audio (16 kHz) -> 80-bin Slaney Log-Mel Filterbank
                              │
                              ▼
FastConformer Encoder (Depthwise striding subsampling + Conformer relative-attention blocks)
                              │
                              ▼
BART-style Post-LN Transformer Encoder (Learned positional encodings)
                              │
                              ▼
Sortformer Output Projection Modules -> Per-speaker Sigmoid Probabilities `[frames, 4]`
```

### Architecture Specifications

| Property | Value |
|---|---|
| Input Sample Rate | 16,000 Hz |
| Max Speakers | 4 simultaneous active speakers |
| Acoustic Frontend | 80-channel mel filterbank, 25 ms window, 10 ms hop |
| Acoustic Subsampling | 4x or 8x temporal downsampling via depthwise strided 2D/1D convolutions |
| Core Encoder | FastConformer blocks with relative positional multi-head self-attention |
| Temporal Modeling | Multi-layer Transformer encoder with learned positional encodings |
| Streaming Support | Offline batch inference and chunk-based streaming with AOSC/FIFO state cache |

## Performance and Benchmarks

- **Runtime Target**: Real-time factor (RTF) ~ 0.05 on Apple M4 Max CPU.
- **Accuracy**: Diarization Error Rate (DER) competitive with multi-stage clustering pipelines, with native overlap handling.
- **Verification Gates**:
  - Deterministic stage fixtures: FastConformer subsampling, relative self-attention, and Bart encoder outputs match golden fixtures within `1e-5`.
  - AOSC speaker-cache compression matches reference fixture byte-for-byte.

## Example Usage

Run the offline diarization or streaming examples:

```sh
# Offline speaker diarization
cargo run -p turbospark-audio --example sortformer_diarize -- <model-dir> <audio.wav>

# Low-latency streaming diarization
cargo run -p turbospark-audio --example sortformer_stream -- <model-dir> <audio.wav>
```
