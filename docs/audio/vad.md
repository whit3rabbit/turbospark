# Voice Activity Detection (VAD) & Diarization Models

Voice Activity Detection (VAD) and Diarization models detect speech presence and separate multiple speakers in conversation ("who spoke when").

## Implemented Model Families

### 1. Silero VAD

- **In-Crate Directory**: [`crates/audio/src/vad/silero_vad/`](../../crates/audio/src/vad/silero_vad/)
- **Documentation**: [`crates/audio/src/vad/silero_vad/README.md`](../../crates/audio/src/vad/silero_vad/README.md)
- **Upstream URLs**:
  - Model Repository: [snakers4/silero-vad](https://github.com/snakers4/silero-vad)
  - MLX Conversion: [mlx-community/silero-vad](https://huggingface.co/mlx-community/silero-vad)
- **Architecture**: Dual 16 kHz / 8 kHz learned-STFT branches, 4x downsampling ReLU convolutions, 128-dim streaming LSTM, and hysteresis timestamp segmentation.
- **Benchmarks**:
  - Processing speed: > 5,000x real-time on CPU.
  - Latency: ~32 ms streaming chunks.
  - Parity: Max probability error `4.8e-7` vs Python MLX reference.

### 2. Sortformer Diarization

- **In-Crate Directory**: [`crates/audio/src/vad/sortformer/`](../../crates/audio/src/vad/sortformer/)
- **Documentation**: [`crates/audio/src/vad/sortformer/README.md`](../../crates/audio/src/vad/sortformer/README.md)
- **Upstream URLs**:
  - Model Repository: [nvidia/diar_streaming_sortformer_4spk](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk)
  - NeMo Reference: `SortformerEncLabelModel`
- **Architecture**: FastConformer encoder with relative positional self-attention + BART post-LN Transformer with learned positional table.
- **Benchmarks**:
  - Simultaneous Speakers: Up to 4 active speakers with overlap support.
  - RTF: ~0.05 on Apple M4 Max CPU.

### 3. Nemotron 3 Diarization

- **In-Crate Directory**: [`crates/audio/src/vad/nemotron_diarization/`](../../crates/audio/src/vad/nemotron_diarization/)
- **Documentation**: [`crates/audio/src/vad/nemotron_diarization/README.md`](../../crates/audio/src/vad/nemotron_diarization/README.md)
- **Upstream URLs**:
  - Model Repository: [nvidia/nemotron-diarization](https://huggingface.co/nvidia/nemotron-diarization)
  - MLX Conversion: [mlx-community/nemotron-diarization-mlx](https://huggingface.co/mlx-community/nemotron-diarization-mlx)
- **Architecture**: Linear feature-stacking subsampling (8x 10 ms), 31-layer NeoX RoPE Transformer backbone, subpixel 1D upsampling head.
- **Benchmarks**:
  - Resolution: 10 ms time resolution output (100 fps).
  - Speaker Count: Up to 8 concurrent speakers.
