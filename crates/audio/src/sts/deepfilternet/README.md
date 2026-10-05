# DeepFilterNet (Speech Enhancement / Noise Suppression)

DeepFilterNet is a low-complexity neural speech enhancement and background noise suppression framework operating on full-band 48 kHz audio, combining Equivalent Rectangular Bandwidth (ERB) spectrogram masking with deep complex filtering.

## Upstream References and URLs

- **Official Repository**: [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **MLX Conversion**: [mlx-community/DeepFilterNet-mlx](https://huggingface.co/mlx-community/DeepFilterNet-mlx)
- **Upstream Reference**: [`mlx_audio/sts/models/deepfilternet/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/sts/models/deepfilternet) in mlx-audio v0.5.7
- **Research Papers**:
  - DeepFilterNet: *A Low-Complexity Speech Enhancement Framework for Full-Band Audio based on Deep Filtering* (Schroeter et al., ICASSP 2022)
  - DeepFilterNet2: *Towards Real-Time Speech Enhancement on Mobile and Edge Devices* (Schroeter et al., IWAENC 2022)
  - DeepFilterNet3: *High-Fidelity Real-Time Speech Enhancement on Resource-Constrained Hardware* (2023)

## Model Overview

DeepFilterNet processes audio in the STFT domain, dividing frequencies into low-frequency bins (subject to complex linear predictive deep filtering) and high-frequency bins (subject to real ERB gain masking).

```text
Full-Band Audio (48 kHz) -> STFT (960 FFT, 480 hop, Vorbis Window)
                                    │
                                    ▼
       ERB Filterbank + Deep Filter (DF) Feature Extraction
                                    │
                                    ▼
                 Convolutional + GRU Encoder Backbone
                                    │
                                    ├──> ERB Mask Decoder (High Bins)
                                    └──> Complex Filter Decoder (Low Bins)
                                    │
                                    ▼
         Combined Filtered Spectrum -> Inverse STFT (iSTFT) -> Enhanced 48 kHz Audio
```

### Architecture Specifications

| Property | Value |
|---|---|
| Sample Rate | 48,000 Hz full-band audio |
| FFT Geometry | 960 window (20 ms), 480 hop (10 ms) with square-root Hann (Vorbis) window |
| Supported Versions | DeepFilterNet v1, DeepFilterNet v2, and DeepFilterNet v3 |
| Spectral Features | Equivalent Rectangular Bandwidth (ERB) scale + Exponential Moving Average (EMA) normalization |
| Encoder Backbone | Convolutional feature extraction layers + Gated Recurrent Units (GRU) |
| Filtering Strategy | Deep filtering of low frequency bins (`nb_df`), ERB mask gains for higher frequencies |

## Performance and Benchmarks

- **Audio Quality**: High PESQ and POLQA improvements, low speech distortion, significant suppression of stationary and non-stationary noises.
- **Latency**: 10 ms algorithmic frame delay, with lookahead compensation.
- **Verification Gates**:
  - DeepFilterNet v3 local witness: waveform correlation `0.964`, max absolute error `0.021`.
  - Intermediate ERB and deep filter coefficient predictions validated against reference checkpoints.

## Example Usage

Run audio enhancement on a noisy WAV file:

```sh
cargo run -p turbospark-audio --example dfn_enhance -- <model-dir> <noisy.wav> <enhanced.wav>
```
