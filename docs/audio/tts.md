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
- **Entry points**: the Voice Cloning page in the app, `POST /v1/audio/speech` with `reference_audio` and `consent: true` ([HTTP API](../AUDIO_API.md)), the native audio session (family `megatts3`, reference audio as the job's PCM), and `turbospark-clone speak` ([CLI](../CLI.md)). Every cloning request needs consent. The catalog row `megatts3-f32` installs from the app once its install terms are accepted; every entry point also takes a local converted directory.
- **Limits**: English only (the English text normalizer is a documented subset), 24 kHz output. On macOS the diffusion transformer and the WavVAE decoder run on Metal by default, about five times faster than the portable CPU path. Steps default to 16 (Balanced), with 8 (Fast) and 32 (Best) as the other presets, and the [step study](../../crates/audio/src/tts/megatts3/README.md#steps-and-quality) found no intelligibility or speaker-similarity loss from 8 steps up.

### Fish Audio Drama 3 (multi-speaker TTS and voice cloning)

- **In-Crate Directory**: [`crates/audio/src/tts/fish_drama3/`](../../crates/audio/src/tts/fish_drama3/) for the prompt, sampler, tower and generation loop, [`crates/runtime/src/fish_drama3.rs`](../../crates/runtime/src/fish_drama3.rs) for the Metal runner
- **Documentation**: [`docs/FISH_DRAMA3.md`](../FISH_DRAMA3.md) (user guide) and [`docs/FISH_DRAMA3_PHASE0.md`](../FISH_DRAMA3_PHASE0.md) (facts, measurements, refusals)
- **Upstream URLs**:
  - Model Hub: [mlx-community/fish-audio-drama-3-4bit](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit), pinned at `cc639e27aba7edb73429d3e3806f471dcd79d316`
  - Code Reference: [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) `fish_drama3`
- **Status**: runs end to end on Metal. A Qwen 3.6 35B-A3B trunk streams its routed experts from disk, the audio tower and the Fish S1 DAC codec run on the GPU, and the codec decodes on its own thread while generation continues. The memory oracle and leak gate pass on an M4 Max. Teacher-forced parity with mlx-audio is measured separately in the Phase 0 doc.
- **Entry points**: `turbospark-speak` ([CLI](../CLI.md)), `POST /v1/audio/speech` ([HTTP API](../AUDIO_API.md)), and the native audio session (family `fish_drama3`, reference audio as the job's PCM). The catalog profile `fish-drama3-4bit` installs by converting the upstream checkpoint on the Mac, and `turbospark-model pull-drama3` does the same into a directory you pick. A reference makes the request a clone and needs consent and the reference's transcript.
- **Limits**: Metal only, 44.1 kHz mono output, no speed control, no instruction text (cues go inline, such as `[whispers]`). Only the 4-bit conversion is supported. It is under the Fish Audio Community License, so a product that uses it shows "Built with Fish Audio".

### OmniVoice (multilingual zero-shot TTS)

- **In-Crate Directory**: [`crates/audio/src/tts/omnivoice/`](../../crates/audio/src/tts/omnivoice/)
- **Documentation**: [`crates/audio/src/tts/omnivoice/README.md`](../../crates/audio/src/tts/omnivoice/README.md)
- **Upstream URLs**:
  - Model Hub: [k2-fsa/OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) and the pinned MLX conversions [mlx-community/OmniVoice-8bit](https://huggingface.co/mlx-community/OmniVoice-8bit) / [OmniVoice-bfloat16](https://huggingface.co/mlx-community/OmniVoice-bfloat16)
  - Code Reference: [`mlx_audio/tts/models/omnivoice/`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/omnivoice)
- **Architecture Overview**: a bidirectional Qwen3-0.6B backbone (causal mask disabled) performing iterative masked-diffusion decoding of 8 HiggsAudioV2 codebooks (1025 vocab with the mask token, 24 kHz at 25 frames per second), decoded through the shared Higgs codec port; voice cloning prepends the reference's code rows and flips the style prefix to `<|denoise|>`.
- **Status**: the port is fixture-verified and the released checkpoints are gated: cross-engine encode parity is exact against the Python reference on bfloat16 (70 frames x 8 codebooks), the 8-bit rowwise checkpoint generates deterministic, oracle-transcribed speech (3.2 GiB footprint). The engine runs with native Metal acceleration on macOS and transparent CPU fallback.
- **Entry points**: the native audio session (family `omnivoice`, reference audio as the job's PCM, cloning requires `consent: true`) and `POST /v1/audio/speech` with optional `reference_audio`. The catalog rows `omnivoice-8bit` (default) and `omnivoice-bf16` install from the app once the Boson tokenizer install terms are accepted. The Voice Cloning page in the app natively includes OmniVoice via Metal acceleration (`clonesNatively`).
- **Limits**: 24 kHz output; steps default to 32 (1 through 100 accepted) and `speed` maps onto the estimated duration (a documented deviation); the file-path reference preprocessing (RMS normalization, silence removal) is not ported, so in-memory PCM encodes directly; long text runs as one utterance (the request validator caps the length); the bundled Higgs audio tokenizer carries Boson AI's community license (attribution and a commercial cap), acknowledged at install.

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

### ArkTTS (Audio8 TTS preview)

- **In-Crate Directory**: [`crates/audio/src/tts/arktts/`](../../crates/audio/src/tts/arktts/)
- **Documentation**: [`crates/audio/src/tts/arktts/README.md`](../../crates/audio/src/tts/arktts/README.md)
- **Upstream URLs**:
  - Model Hub: Audio8 TTS Preview 0.1b and 0.6b checkpoints
  - Code Reference: [`mlx_audio/tts/models/arktts/`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models/arktts) at mlx-audio 0.5.8 (`e1b19b9054bf163f5d812221a54fcc346f1890e9`)
- **Architecture Overview**: DualAR transformer plus a bundled 44.1 kHz decode-only codec. A slow AR stack predicts one semantic token per audio frame, and a fast AR stack predicts that frame's residual codec codebooks. Two configurations: pure-attention slow stack (0.6b) and hybrid Mamba-2 + attention slow stack via [`crates/audio/src/lm/falcon_h1.rs`](../../crates/audio/src/lm/falcon_h1.rs) (0.1b).
- **Status**: Fixture-verified against the pinned mlx-audio Python reference on CPU: op-level traces for pure-attention and Falcon slow stacks, fast AR loop, sampling math, and the 44.1 kHz decode-only codec.
- **Entry points**: Native audio session (family `arktts`), caller-managed prompt construction and Mistral tokenization.
- **Limits**: 44.1 kHz output; decode-only codec (encoding is refused; callers pass precomputed reference codes); no Metal acceleration.

## Additional Implemented TTS Family Ports

The full inventory of ported and scaffolded families is maintained in [`crates/audio/MODELS.md`](../../crates/audio/MODELS.md). Family ports in `crates/audio/src/tts/` include:

| Family | Backend / Codec | Status & Documentation |
|---|---|---|
| `arktts` | DualAR / Audio8 Codec | [Family port](../../crates/audio/src/tts/arktts/README.md) |
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
