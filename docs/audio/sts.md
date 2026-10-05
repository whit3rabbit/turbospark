# Speech-to-Speech (STS) and Speech Enhancement Models

Speech-to-Speech (STS) and Enhancement models remove acoustic noise, enhance speech clarity, or perform voice conversion.

## Implemented Model Families

### DeepFilterNet (DFN 1 / 2 / 3)

- **In-Crate Directory**: [`crates/audio/src/sts/deepfilternet/`](../../crates/audio/src/sts/deepfilternet/)
- **Documentation**: [`crates/audio/src/sts/deepfilternet/README.md`](../../crates/audio/src/sts/deepfilternet/README.md)
- **Upstream URLs**:
  - Official Repository: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
  - MLX Conversion: [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx)
  - Research Papers: Schroeter et al. (ICASSP 2022, IWAENC 2022, 2023)
- **Architecture**:
  - Sample Rate: Full-band 48 kHz audio.
  - Spectral Analysis: 960 FFT (20 ms window) and 480 hop (10 ms) with Vorbis windowing.
  - Multi-scale Processing: ERB-scale filterbank masking for higher frequencies, deep complex linear filtering for lower frequencies (`nb_df`).
  - Backbone: Multi-layer Convolutional + Gated Recurrent Unit (GRU) encoder.
- **Benchmarks**:
  - Latency: 10 ms algorithmic frame delay with lookahead compensation.
  - Verification: Waveform correlation 0.964 on noisy benchmark audio.
