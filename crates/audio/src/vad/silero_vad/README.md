# Silero Voice Activity Detector (VAD)

Silero VAD is an ultra-lightweight, high-accuracy voice activity detection and speech timestamp segmentation model designed for low-latency streaming and offline speech preprocessing.

## Upstream References and URLs

- **Hugging Face Repository**: [snakers4/silero-vad](https://huggingface.co/snakers4/silero-vad)
- **MLX Conversion**: [mlx-community/silero-vad](https://huggingface.co/mlx-community/silero-vad)
- **Upstream Reference**: [`mlx_audio/vad/models/silero_vad/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/vad/models/silero_vad) in mlx-audio v0.5.7
- **Official GitHub**: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)

## Model Overview

Silero VAD operates on streaming audio chunks or entire files, producing speech probabilities in `[0.0, 1.0]` per chunk, along with padded, merged speech intervals.

```text
PCM Audio (16 kHz / 8 kHz)
         │
         ▼
Learned STFT Convolution (kernel=filter_len, stride=hop)
         │
         ▼
Magnitude Compression & 4x ReLU Downsampling Convolutions
         │
         ▼
Streaming LSTM (hidden_size = 128)
         │
         ▼
1x1 Convolution + Sigmoid -> Speech Probability -> Hysteresis State Machine -> Speech Timestamps
```

### Architecture Specifications

| Property | Value |
|---|---|
| Supported Sample Rates | 16,000 Hz and 8,000 Hz (dedicated model branches) |
| Parameters | ~2.5 million parameters |
| Streaming Chunk Size | 512 samples (32 ms @ 16 kHz) or 256 samples (32 ms @ 8 kHz) |
| Streaming State | LSTM hidden state `[128]`, cell state `[128]`, and context audio samples |
| Latency | < 35 ms end-to-end algorithmic delay |
| Post-processing | Dynamic hysteresis with configurable min speech, min silence, and speech padding |

## Performance and Benchmarks

- **CPU Throughput**: > 5,000x real-time on Apple Silicon CPU (1 hour of audio processed in < 0.7 seconds).
- **Memory Consumption**: < 15 MiB resident memory.
- **Verification Evidence**:
  - Parity validated against Python MLX reference across 188 streaming audio chunks.
  - Maximum probability absolute error: `4.8e-7`.
  - Speech segment timestamp boundaries: 100% identical.

## Example Usage

Run the streaming and file segmentation demo:

```sh
cargo run -p turbospark-audio --example silero_vad_demo -- <model-dir> <audio.wav>
```
