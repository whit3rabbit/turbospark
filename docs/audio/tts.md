# Text-to-Speech (TTS) Models

Text-to-Speech (TTS) models synthesize high-fidelity natural speech from input text or phonemes.

## Implemented Model Families

### Qwen3-TTS (native Swift MLX)

[Qwen3-TTS in Audio](../QWEN3_TTS.md) covers Base voice cloning,
CustomVoice speech, VoiceDesign, supported checkpoints, and verification.
This provider lives in `swift/TurboSparkApp/Vendor/Qwen3TTS`; it is not a Rust
inference module in `crates/audio/src/tts`.

### MegaTTS3 (zero-shot voice cloning)

- **In-Crate Directory**: [`crates/audio/src/tts/megatts3/`](../../crates/audio/src/tts/megatts3/)
- **Documentation**: [`crates/audio/src/tts/megatts3/README.md`](../../crates/audio/src/tts/megatts3/README.md)
- **Upstream URLs**:
  - Model Hub: [ByteDance/MegaTTS3](https://huggingface.co/ByteDance/MegaTTS3)
  - Code Reference: [bytedance/MegaTTS3](https://github.com/bytedance/MegaTTS3) (there is no mlx-audio port)
  - Paper: [MegaTTS 3](https://arxiv.org/abs/2502.18924)
- **Status**: the whole English pipeline is ported and gated against the reference on real weights (prompt aligner, WavVAE encoder and decoder, G2P, duration LM, latent DiT with classifier-free guidance, loudness matching). Voice cloning needs the community WavVAE encoder, which ByteDance withheld; its origin and license are unverified. Weights are converted from the upstream PyTorch checkpoints by `crates/audio/tools/convert_megatts3.py`.
- **Entry point**: `turbospark-clone speak` ([CLI](../CLI.md)) from a local converted directory. It is not in the catalog, the runtime, the FFI, the server, or the Swift app yet.
- **Limits**: English only (the English text normalizer is a documented subset), 24 kHz output, and the CPU path is slow.

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
