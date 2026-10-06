# turbospark-audio

Unified portable audio crate: DSP primitives (waveform, WAV I/O, resampling, FFT, STFT, mel frontends), neural tensor ops, and audio models organized by role (`music/`, `stt/`, `tts/`, `vad/`, `sts/`, `codec/`, `lid/`).

## Read first

- [Detailed module guide](../../.claude/docs/modules/audio.md)
- [Audio layer scope and provenance](../../docs/AUDIO.md)
- [Audio family inventory and model catalog](MODELS.md)
- [Root audio documentation](../../docs/audio/README.md)
- [Testing rules](../../docs/TESTING.md)

## Audio layer ownership

| Layer | Owns |
|---|---|
| `crates/audio` | Portable PCM, containers, DSP frontends, neural tensor ops, and model implementations organized by role (`music`, `stt`, `tts`, `vad`, `sts`, `codec`, `lid`) |
| [Runtime](../runtime/AGENTS.md) | Sessions, state, device execution, Metal offload, and family integration |
| [Catalog](../catalog/AGENTS.md) | Pinned profile metadata, probes, and local installation validation |
| [FFI](../ffi/AGENTS.md) and [Swift](../../swift/AGENTS.md) | Qualified C ABI capabilities and native wrappers |

## Organization by Role

All models and model families reside under their respective role directories:
- `src/music/<family>/`: Music generation (e.g. `minimax_music3`)
- `src/stt/<family>/`: Speech-to-text (e.g. `canary`, `fireredasr2`, `fun_asr_nano`, `glmasr`, `granite_speech`, `granite_speech5_ctc`, `higgs_audio_3`, `mega_asr`, `mms`, `moonshine`, `moss_transcribe_diarize`, `nemotron_asr`, `parakeet`, `phonon`, `qwen3_asr`, `qwen3_forced_aligner`, `sensevoice`, `whisper`)
- `src/tts/<family>/`: Text-to-speech (e.g. `kokoro`)
- `src/vad/<family>/`: Voice activity detection and diarization (e.g. `silero_vad`, `sortformer`, `nemotron_diarization`)
- `src/sts/<family>/`: Speech-to-speech and enhancement (e.g. `deepfilternet`)
- `src/codec/<family>/`: Neural audio codecs and vocoders (e.g. `bigvgan`, `dacvae`, `descript`, `ecapa_tdnn`, `encodec`, `fish_s1_dac`, `higgs_audio`, `mimi`, `mimo_audio_tokenizer`, `moss_audio_tokenizer`, `nemotron_voicechat`, `s3`, `snac`, `vocos`)
- `src/lid/<family>/`: Spoken language identification (e.g. `ecapa_tdnn`, `wav2vec2_lid`)

Every model family directory contains its implementation code, configuration, weight loader, and a comprehensive markdown document (`README.md`) detailing:
1. Overview & model description
2. URLs (HuggingFace checkpoints, upstream papers, reference code)
3. Benchmarks, quality metrics, latency, and verified stages
4. Model architecture specifications (parameters, sample rate, filter dimensions, quantization support)

## Model implementation and porting rules

- The normative reference is mlx-audio 0.5.7 at
  [`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio).
  Compare against this snapshot, not a floating branch. Fix disagreements
  or document the reason for a divergence in the family README.
- Verify the local clone at `../mlx-audio` matches the reference commit.
  Each model header names its reference file and commit.
- Keep this crate buildable off macOS. Plain f32 arithmetic, no Metal, no
  macOS APIs. Metal offload is a later runtime optimization.
- All tensors are row-major f32 slices. Linear weights load in the HF
  `[out, in]` layout; helpers convert the exceptions at load time, not in
  the kernels.
- MLX-quantized checkpoints are the norm. `quant.rs` dequantizes groupwise
  affine tensors (2/3/4/6/8-bit). A model module must refuse a quant scheme
  it has not verified rather than silently mis-decode.
- Every model module ships op-level parity tests against golden tensors
  produced by the pinned Python reference, plus an end-to-end run for
  checkpoints small enough to install locally.
- Record regeneration commands, digests, and evidence locations per family.
  Keep regeneration separate from tests. Tests must not download assets.
- Record implementation, fixture parity, checkpoint parity, task quality,
  runtime, catalog, FFI, and Swift gates separately. Preserve historical
  reports and their limitations in the task list and public inventory.
  Compilation and parity alone do not establish quality or device performance.
- Text frontends (G2P, phonemizers) are part of the model module that needs
  them. They follow the reference implementation's behavior, including its
  fallbacks, and pin their behavior with tests.

## Numeric contracts and signal processing rules

- Keep audio CPU implementations independent of Metal, MLX, and macOS APIs. Verify portable target compilation (`cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio`).
- Waveforms are interleaved f32 (`samples[frame * channels + channel]`). Planar forms exist only inside `conversion`.
- Preserve FFT unscaled forward and 1/(n/2) inverse conventions.
- Slaney area normalization uses `2 / (upper_hz - lower_hz)`, never a mel-coordinate width.
- Whisper drops the final centered STFT frame before computing global log-mel peak and clamp.
- NeMo mel frontends maintain separate contracts: Parakeet uses constant signal padding and feature normalization; Nemotron uses reflect padding with no normalization.
- Numeric contracts (eps values, clamps, normalization order) are load-bearing. Change them only with new evidence from the reference.
- Unsupported geometry or corrupt checkpoints return descriptive `AudioError` or `SpeechError` variants.

## Verification

```sh
cargo test -p turbospark-audio
cargo check --target x86_64-unknown-linux-gnu -p turbospark-audio
cargo fmt -p turbospark-audio --check
cargo clippy -p turbospark-audio --tests
```
