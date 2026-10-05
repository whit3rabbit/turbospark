# Text-to-Speech (TTS) Models

Text-to-Speech (TTS) models synthesize high-fidelity natural speech from input text or phonemes.

## Implemented Model Families

### Kokoro (82M)

- **In-Crate Directory**: [`crates/audio/src/tts/kokoro/`](../../crates/audio/src/tts/kokoro/)
- **Documentation**: [`crates/audio/src/tts/kokoro/README.md`](../../crates/audio/src/tts/kokoro/README.md)
- **Upstream URLs**:
  - Model Hub: [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
  - MLX Conversion: [mlx-community/Kokoro-82M-bf16](https://huggingface.co/mlx-community/Kokoro-82M-bf16)
  - Code Reference: [`mlx_audio/tts/models/kokoro/`](https://github.com/Blaizzy/mlx-audio/tree/main/mlx_audio/tts/models/kokoro)
  - Architecture: StyleTTS 2 (Li et al., 2023) + iSTFTNet Vocoder (Kaneko et al., 2022)
- **Architecture Overview**:
  - **Text Encoder**: PLBert (ALBERT-based phoneme sequence encoder with weight sharing).
  - **Prosody & Duration**: Bidirectional LSTM duration predictor with AdaLayerNorm style injection.
  - **Vocoder**: iSTFTNet combining multi-receptive field fusion and inverse STFT at 24 kHz.
- **Benchmarks**:
  - Sample Rate: 24,000 Hz.
  - Speed: RTF < 0.20 on Apple Silicon M-series CPUs.
  - Memory: ~180 MiB resident footprint.
