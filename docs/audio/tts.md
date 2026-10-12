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
- **Entry points**: the Voice Clone page in the app, `POST /v1/audio/speech` with `reference_audio` and `consent: true` ([HTTP API](../AUDIO_API.md)), the native audio session (family `megatts3`, reference audio as the job's PCM), and `turbospark-clone speak` ([CLI](../CLI.md)). Every cloning request needs consent. The catalog row `megatts3-f32` installs from the app once its install terms are accepted; every entry point also takes a local converted directory.
- **Limits**: English only (the English text normalizer is a documented subset), 24 kHz output. On macOS the diffusion transformer and the WavVAE decoder run on Metal by default, about five times faster than the portable CPU path. Steps default to 16 (Balanced), with 8 (Fast) and 32 (Best) as the other presets, and the [step study](../../crates/audio/src/tts/megatts3/README.md#steps-and-quality) found no intelligibility or speaker-similarity loss from 8 steps up.

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

## Additional Implemented TTS Family Ports

The full inventory of ported and scaffolded families is maintained in [`crates/audio/MODELS.md`](../../crates/audio/MODELS.md). Family ports in `crates/audio/src/tts/` include:

| Family | Backend / Codec | Status & Documentation |
|---|---|---|
| `bailingmm` | Audio VAE | [Family port](../../crates/audio/src/tts/bailingmm/README.md) |
| `bark` | EnCodec 24kHz | [Family port](../../crates/audio/src/tts/bark/README.md) (semantic + coarse + fine transformers) |
| `breeze_tts` | Qwen3-TTS speech tokenizer | [Family port](../../crates/audio/src/tts/breeze_tts/README.md) |
| `confucius4` | BigVGAN | [Family port](../../crates/audio/src/tts/confucius4/README.md) |
| `dia` | Descript DAC | [Family port](../../crates/audio/src/tts/dia/README.md) |
| `echo_tts` | Fish S1 DAC | [Family port](../../crates/audio/src/tts/echo_tts/README.md) |
| `fish_qwen3_omni` | Dual-track transformer | [Family port](../../crates/audio/src/tts/fish_qwen3_omni/README.md) |
| `higgs_audio` | HiggsAudio Codec | [Family port](../../crates/audio/src/tts/higgs_audio/README.md) |
| `higgs_audio_v3` | HiggsAudio Codec | [Family port](../../crates/audio/src/tts/higgs_audio_v3/README.md) |
| `indextts` | Neural vocoder | [Family port](../../crates/audio/src/tts/indextts/README.md) |
| `kitten_tts` | Acoustic decoder | [Family port](../../crates/audio/src/tts/kitten_tts/README.md) |
| `kugelaudio` | VibeVoice decoder / DiT | [Family port](../../crates/audio/src/tts/kugelaudio/README.md) |
| `llama` | SNAC | [Family port](../../crates/audio/src/tts/llama/README.md) |
| `longcat_audiodit` | AudioDiT | [Family port](../../crates/audio/src/tts/longcat_audiodit/README.md) |
| `melotts` | Bert frontend | [Family port](../../crates/audio/src/tts/melotts/README.md) |
| `moss_tts` | Qwen3 backbone | [Family port](../../crates/audio/src/tts/moss_tts/README.md) |
| `moss_tts_nano` | Transformer | [Family port](../../crates/audio/src/tts/moss_tts_nano/README.md) |
| `omnivoice` | HiggsAudio Codec | [Family port](../../crates/audio/src/tts/omnivoice/README.md) |
| `outetts` | Audio tokenizer | [Family port](../../crates/audio/src/tts/outetts/README.md) |
| `pocket_tts` | Lightweight vocoder | [Family port](../../crates/audio/src/tts/pocket_tts/README.md) |
| `qwen3` | SNAC | [Family port](../../crates/audio/src/tts/qwen3/README.md) |
| `rumik_oss` | Acoustic decoder | [Family port](../../crates/audio/src/tts/rumik_oss/README.md) |
| `sesame` | Transformer | [Family port](../../crates/audio/src/tts/sesame/README.md) |
| `soprano` | Flow-matching | [Family port](../../crates/audio/src/tts/soprano/README.md) |
| `spark` | BiCodec | [Family port](../../crates/audio/src/tts/spark/README.md) |
| `zonos2` | DAC / Diffusion | [Family port](../../crates/audio/src/tts/zonos2/README.md) |
