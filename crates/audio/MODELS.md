# Audio family and model inventory

This is the public planning map for `turbospark-audio`. It separates upstream
architecture families, checkpoint profiles, and the evidence needed to expose
a capability. A source directory, table row, or scaffold does not establish
Rust support.

Audio models are unified under `crates/audio` and organized by role (`music/`,
`stt/`, `tts/`, `vad/`, `sts/`, `codec/`, `lid/`).

## Source baseline

The reference is [Blaizzy/mlx-audio at
`e1b19b9054bf163f5d812221a54fcc346f1890e9`](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9),
whose [source version](https://github.com/Blaizzy/mlx-audio/blob/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/version.py)
is `0.5.7`. The sibling reference checkout was clean at that commit when this
inventory was checked. Future ports record that full source commit, source
files, attribution, and any deliberate divergence in their family README.

Upstream organizes `mlx_audio/<task>/models/<family>`. Its
[loader](https://github.com/Blaizzy/mlx-audio/blob/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/utils.py)
accepts an optional model revision, and its README examples often use floating
Hugging Face repository names. Organized families and immutable model pins
are separate properties. This plan requires both.

There are exactly 101 immediate source directories: 41 TTS, 28 STT, 9 STS,
5 VAD, 2 LID, 1 music, and 15 codec directories. Two TTS directories are
compatibility aliases. This is a complete source inventory, rather than a
copy of the shorter upstream supported-models table.

## Current implementation and evidence

Except for the Nemotron, Qwen3-ASR, Qwen3-ForcedAligner, and SenseVoice rows, the evidence
below is reported in the existing local task tracker and port comments, not
rerun for this update. Some earlier fixture claims still need checked-in
provenance and regeneration instructions before they count as reproducible
public gates.

| Family / profile | Current implementation | Reported evidence | Limits and remaining gates |
|---|---|---|---|
| `vad/silero_vad` | [Family port](src/vad/silero_vad/mod.rs) | Real-checkpoint comparison over 188 chunks: max probability difference `4.8e-7`, identical segment timestamps | One local WAV witness; quality, full branch/profile qualification, runtime, catalog, FFI and Swift gates remain separate |
| `stt/whisper` | [Speech model](src/stt/whisper/mod.rs) & [Runtime](../runtime/src/whisper/mod.rs) | Historical CPU end-to-end and Metal convolution frontend evidence | Recorded Whisper/MLX transcript and encoder witnesses predate the corrected mel frontend and require rerun; full encoder remains CPU; [STT documentation](../../docs/SPEECH_TO_TEXT.md) owns the limits; no current quality or throughput qualification |
| `stt/whisper` distilled profile | Same config-driven Whisper runtime | `distil-large-v3` matched the Python greedy transcript on one Kokoro-generated speech clip | Requires a fresh corrected-frontend checkpoint witness before current qualification; non-speech tone produced different single-token output; other distilled profiles and quality remain unqualified |
| `stt/moonshine` | [Family port](src/stt/moonshine/mod.rs) and [experimental Metal runner](../runtime/src/moonshine.rs) | One Tiny checkpoint clip matches CPU and Metal transcript; Metal encoder matches CPU within f32 noise | Metal remains opt-in after a higher diagnostic peak footprint than CPU; broader quality, paired performance, catalog, FFI, and Swift gates remain open |
| `stt/canary` | Upstream MLX reference only | Pinned q8 conversion loaded with mlx-audio 0.5.7 and returned the expected transcript on one generated 16 kHz clip | Rust port, stage fixtures, broader quality and product integration remain open |
| `stt/granite_speech` | Upstream MLX reference only | Pinned three-shard 1B checkpoint loaded after adding optional `jinja2` to the local reference environment; returned the expected phrase on one generated 16 kHz clip | Rust port, stage fixtures, translation modes, broader quality and product integration remain open |
| `stt/parakeet` | [Family port](src/stt/parakeet/mod.rs) | Pinned MLX and Rust v2, v3, and Redux checkpoints returned the same smoke transcript; v2 stage fixtures from the float32 reference (mel 8.2e-5, encoder 7.5e-6, all 21 decode steps and 19 emitted tokens identical) and matching transcripts on two generated clips | runtime/catalog, FFI, Swift, performance and memory; v3/Redux second-clip re-runs and broader quality remain open |
| `stt/nemotron_asr` | [Family port](src/stt/nemotron_asr/README.md) | Pinned mlx-audio 0.5.7 and Rust runs returned the exact same transcript on one generated 16 kHz clip; first-block attention and final encoder stage were compared | Offline inference only; cache-aware streaming, checked-in stage fixtures, broader quality, runtime/catalog, FFI, Swift, performance and memory remain open |
| `stt/granite_speech5_ctc` | [Port in progress](src/stt/granite_speech5_ctc/README.md) | Earlier pinned MLX and Rust checkpoints matched; frontend fixture and synthetic batched midpoint/final-logit parity pass | Batched path needs checkpoint rerun; broader quality, runtime/catalog, FFI, Swift, performance and memory gates remain open |
| `stt/qwen3_asr` | [Port in progress](src/stt/qwen3_asr/mod.rs) | Earlier pinned Rust checkpoint matched the pinned MLX transcript; frontend and selected audio-tower outputs match their fixtures; a synthetic incremental-cache test matches full-prefix decoding | The prior debug run used full-prefix recomputation. The new cache needs pinned-checkpoint rerun, broader quality, runtime/catalog, FFI, Swift, performance and memory gates |
| `stt/qwen3_forced_aligner` | [Port in progress](src/stt/qwen3_forced_aligner/README.md) | Pinned Rust checkpoint matched all nine MLX word spans within 1 ms on the checked-in WAV | English and Chinese segmentation only; Japanese/Korean segmentation, broader quality, workspace, runtime/catalog, FFI, Swift, performance and memory gates remain open |
| `stt/sensevoice` | [Family port](src/stt/sensevoice/README.md) | Pinned F32 checkpoint: MLX and Rust match transcript, tags, and CTC token IDs on one clip; Kaldi FBANK/LFR/CMVN and selected encoder stages match within `0.01` | One English clip; 936 MB F32 profile; broader quality, memory/performance, runtime/catalog, FFI and Swift gates remain open |
| `stt/glmasr` | [Family port](src/stt/glmasr/README.md) | Pinned 4-bit checkpoint: selected MLX/Rust frontend, first/last encoder, and adapter values pass their tolerances; transcript matches on one English clip | Single clip up to 30 seconds; small f32 drift across deep encoder stages; broader quality, memory/performance, runtime/catalog, FFI and Swift gates remain open |
| `tts/kokoro` | [Family port](src/tts/kokoro/mod.rs) | Deterministic-stage comparisons; phoneme-driven output transcribed as "Hello World" | Stochastic source prevents exact waveform parity; current example accepts phonemes and an external voice asset, with text frontend, language, profile, integration and quality gates still open |
| `sts/deepfilternet` | [Family port](src/sts/deepfilternet/mod.rs) | v3 local tone/noise witness: waveform correlation `0.964`, max difference `0.021`; intermediate-stage comparisons reported | v2 shares a layout switch but lacks an independent checkpoint witness; v1 is uncovered; no unit tests in this module or task-quality qualification implied |
| `music/minimax_music3` | [Family port](src/music/minimax_music3/mod.rs) | Tiny-config fixture parity against the pinned Python reference: MLX-compatible RNG bit-exact (keys, splits, uniforms, categorical tokens with recorded decision margins), AR codebooks token-exact over recorded short and long traces, prefill/depth/DiT/vocoder outputs within 1e-4 to 5e-4, two-chunk flow stitching on 4096 recorded waveform spots within 1e-4, official-tree conversion and affine 8-bit loading gates; nine mutation checks caught. Stress gates: request boundaries (steps 1/30, prompt at exactly MAX_PROMPT_TOKENS), 200/201/300/301-frame chunk schedules, the 9000-frame ceiling, the end-token-first-frame error path (patched-tree copy), and an early-end-token `generate()` fix. Regeneration determinism: two clean-tree generator runs reproduced all 44 content SHA-256 digests byte-identically, equal to the checked-in fixtures. Timing (Apple M4 Max, release, CPU f32, one test thread; the module [README](src/music/minimax_music3/README.md) owns the commands): tiny AR 9000 frames 64.3 s after the measured O(t^2) fixes (78-85 s before); reference on the same host, trees, and seeds: MLX CPU 28.2 s, MLX GPU 30.7 s; tiny flow per chunk 169.1 ms Rust CPU vs 38.5 ms MLX CPU and a 145-frame end-to-end at RTF 13-14; real-width AR (6 of 36 layers, vocab 8192, fp32 synthetic) 4.47 s/frame with the depth decoder expansions the dominant term, roughly 12 s/frame at full real dims by linear component composition (labeled projection, not a measurement); real-dims flow (real DiT and vocoder, latent length 689): 132.5 s per DiT forward, 530 s per 2-step chunk, vocoder 281 s for 8 s of audio, a 30-step chunk projects to about 2.3 hours (linear in steps) | Tiny random-init configuration for parity; synthetic enlarged trees for timing (seeded default-init weights, end token pushed out of the sampling mask; `tools/gen_minimax_music3_bench_tree.py`). All timing rows are CPU f32 software-path evidence, never Metal, quality, or real-checkpoint claims; the full MXFP8 checkpoint at `d00a12c3c7f80eb66379dd02dd0f30ed0ce2d96e` generated 25 frames and 44,032 finite non-silent stereo samples; controlled f32 MLX 0.32.3 comparison matches three emitted codebooks exactly, hiddens within 2e-4, and waveform within 2e-4; native BF16 activation equivalence and song quality remain open; official-tokenizer prompt generation, the seven pinned catalog rows, and tiny-fixture CLI output are implemented, full-size converted checkpoint generation now uses `runtime::Music3Runner` and resident packed Metal kernels on macOS; fixture GPU parity is separate from real-checkpoint parity, song quality, memory/performance, catalog network-rot, FFI and Swift gates |

Infrastructure is also reported complete in local task 1. Op and quantization
code exist in [ops.rs](src/ops.rs) and [quant.rs](src/quant.rs). A passing unit
test or successful dequantization does not qualify every model, bit width,
checkpoint layout, or product surface that uses them.

## Family and profile layout

Use a task-qualified family identity such as `tts/qwen3_tts` or
`codec/higgs_audio`. The same basename under another task is a different
component. Multitask models keep one implementation in their upstream task
namespace and advertise explicit capabilities rather than copying the model
into several task directories.

The layout aligned with `mlx_audio/<task>/models/<family>` is:

```text
crates/audio/
  AGENTS.md
  MODELS.md
  README.md
  src/
    lib.rs
    ops.rs
    quant.rs
    music/
      mod.rs
      minimax_music3/
        README.md
        mod.rs
    stt/
      mod.rs
      whisper/            # Whisper & Distil-Whisper family
        README.md
        mod.rs
      moonshine/          # Useful Sensors Moonshine
        README.md
        mod.rs
      parakeet/           # NeMo Parakeet TDT
        README.md
        mod.rs
      granite_speech5_ctc/ # IBM Granite Speech 5.0 TurboCTC
        README.md
        mod.rs
      mms/                # Meta MMS Wav2Vec2
        README.md
        mod.rs
      qwen3_asr/          # Qwen3 ASR
        README.md
        mod.rs
      qwen3_forced_aligner/ # Qwen3 Forced Aligner
        README.md
        mod.rs
      nemotron_asr/       # NeMo Nemotron 3.5 ASR
        README.md
        mod.rs
    tts/
      mod.rs
      kokoro/             # Kokoro TTS (82M)
        README.md
        mod.rs
    vad/
      mod.rs
      silero_vad/         # Silero VAD
        README.md
        mod.rs
      sortformer/         # Sortformer speaker diarization
        README.md
        mod.rs
      nemotron_diarization/ # Nemotron 3 Diarization
        README.md
        mod.rs
    sts/
      mod.rs
      deepfilternet/      # DeepFilterNet speech enhancement
        README.md
        mod.rs
    codec/
      README.md
      mod.rs              # Codecs & learned tokenizers planning
    lid/
      README.md
      mod.rs              # Language identification planning
  testdata/               # committed fixtures and golden tensors
  examples/               # verified runnable examples
  tests/                  # committed DSP and model regressions
  testdata/                 # small offline fixtures plus provenance
```

Do not create empty inference modules or one Rust module per model size or
quantization. Keep Base/CustomVoice/VoiceDesign, size, precision, version and
checkpoint revision in profile records under the family README until a
qualified catalog schema owns them. Existing flat aliases preserve callers.

Shared DSP stays in [turbospark-audio](../audio/AGENTS.md); shared learned
codecs, vocoders and speaker components belong under `models/codec/`.
Family-specific codecs remain local unless their contracts are actually
shared. Generic Llama/Qwen backbones do not identify an audio capability by
themselves, as the upstream
[registry](https://github.com/Blaizzy/mlx-audio/blob/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/registry.py)
also recognizes.

## Pinned profiles

A profile identifies the task-qualified family, upstream model/version,
behavior variant, parameter size, artifact format, precision and explicit
capabilities. Preserve existing public aliases when catalog integration
arrives. Each profile records:

- Reference repository commit, source files, port differences, attribution
  and licenses.
- Every artifact repository and immutable revision, including separate
  tokenizers, codecs, vocoders, speaker encoders, voice assets, adapters,
  phonemizer/G2P assets and language resources.
- Exact file list, byte sizes and SHA-256 values for weights, shard indices,
  configs, preprocessing, tokenization and generation settings. Verify source
  shards in place; do not stage large checkpoints in temporary directories.
- Required sample rates, channels, PCM convention, token/codebook contracts,
  quantization layout and supported request options, including streaming.
- Fixture/checkpoint provenance, numeric mode, tolerances, expected outputs,
  and the evidence for quality, performance, memory and product readiness.

Unknown revisions, sizes and digests stay explicitly unqualified. Do not
invent hashes or turn a floating repository name into a curated catalog row.
Tests consume small local fixtures and never download checkpoint assets.

The current [speech catalog](../catalog/src/speech.rs) has nine pinned Whisper
metadata rows in [speech_models.json](../catalog/src/speech_models.json), plus
local probe and receipt validation. These are catalog/probe metadata, not a
completed speech pull path. `TokenizerSource` currently carries only `repo`
and `file`, with no independent tokenizer revision. A model revision and an
install-time receipt do not establish immutable provenance for every upstream
artifact. Qualifying a future pull path must close that gap and pin the full
artifact graph before claiming complete profile pinning.

## Qualification and Swift

Keep these gates independent for each profile:

| Gate | Required evidence |
|---|---|
| Implementation | Real loader, config validation, operators and task path; unsupported options fail explicitly |
| Fixture parity | Small independent reference fixtures with source commit, digests, shapes, numeric mode, regeneration and focused regressions |
| Checkpoint parity | Explicit installed checkpoint/profile, matched frontend/tokenizer/codec assets, request and Python reference output |
| Quality | Labeled task evaluation or task-appropriate listening/enhancement/alignment protocol; one transcript or waveform correlation is insufficient |
| Performance / memory | Matched end-to-end workload on eligible hardware, including complete retained streaming state |
| Runtime / catalog | Verified dispatch, admission, cancellation, lifecycle and artifact probe/receipt, followed separately by a qualified pull route |
| C ABI / Swift | Aligned capability query, typed task I/O, ownership, cancellation and streaming tests, plus existing FFI/Swift build gates |
| Packaging | Tested distributed library/app and supported platforms; portable compilation does not establish Metal or iOS readiness |

Rust owns model inference. The existing C ABI is the Swift integration path.
Future Swift wrappers consume profile/capability information and return typed
task results with bounded PCM/audio chunks. Cancellation follows native
semantics: current STT cancel discards buffered audio, and cannot interrupt
or run concurrently with synchronous finish. The
[audio binding plan](../../docs/SWIFT_BINDINGS.md#audio-binding-plan) owns
handle lifetimes, serialized calls and future wrapper tests. The Swift app
owns AVFoundation capture/playback and UI state. Codec dependencies do not
become selectable user models just because they appear in this inventory.
New family readiness must trace through model I/O, runtime, FFI, Swift,
catalog and relevant CLI/server surfaces before exposure.

[mlx-audio-swift](https://github.com/Blaizzy/mlx-audio-swift/tree/8d86630ade569728aaea3dc1a29fc44e2efa719b)
is a separate Swift/MLX inference implementation, with different family
coverage. It is useful as a source reference; this plan adds Swift bindings
over Rust rather than a second inference implementation. This documentation
scaffold adds no C ABI or Swift types.

## Inventory labels

`README` means the pinned upstream supported-models table lists the family.
`Source` means a source directory exists beyond that table. `Shared` identifies
a reusable dependency, and `Alias` identifies a compatibility re-export.
These labels describe upstream organization, not Rust readiness. `None` in the
Rust column means there is no current implementation at that path.

For each task below, the default canonical identity is `<task>/<directory>`.
Alias rows name their canonical family instead. Task numbers refer to the
optional local tracker; `None` means it had no matching implementation task.
Source-only families and dependencies still need their own approved port and
qualification work before implementation is claimed.

## TTS

Text-to-speech. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/tts/models).

CSM/MisoTTS belongs to `sesame`, despite the old local task 30 pointing
to `llama`. MOSS delay/local directories re-export `moss_tts`; preserve their
variant distinction without creating duplicate inference families.

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `arktts` | Source | None | Default task-qualified family | None |
| `bailingmm` | README | 42 | Ming Omni MoE | None |
| `bark` | Source | None | Default task-qualified family | None |
| `breeze_tts` | Source | None | Aliases `breeze`, `breeze-tts` | None |
| `chatterbox` | README | 32 | Default task-qualified family | None |
| `chatterbox_turbo` | Source | None | Default task-qualified family | None |
| `confucius4` | Source | None | Default task-qualified family | None |
| `dense` | README | 41 | Ming Omni Dense; generic config needs audio identity | None |
| `dia` | README | 31 | Default task-qualified family | None |
| `dramabox` | Source | None | Alias `dramabox-tts` | None |
| `echo_tts` | Source | None | Default task-qualified family | None |
| `fish_qwen3_omni` | Source | None | Default task-qualified family | None |
| `higgs_audio` | README | 34 | Default task-qualified family | None |
| `higgs_audio_v3` | README | 45 | Alias `higgs_multimodal_qwen3` | None |
| `indextts` | Source | None | Default task-qualified family | None |
| `irodori_tts` | Source | None | Default task-qualified family | None |
| `kitten_tts` | README | 25 | Alias `kitten` | None |
| `kokoro` | README | 24 | Default task-qualified family | [Flat port](src/tts/kokoro/mod.rs) |
| `kugelaudio` | README | 40 | Default task-qualified family | None |
| `llama` | Source | None | SNAC TTS wrapper over generic Llama backbone | None |
| `longcat_audiodit` | README | 39 | Aliases `audiodit`, `longcat` | None |
| `melotts` | README | 27 | Default task-qualified family | None |
| `moss_tts` | README | 44 | Delay-pattern and local-transformer variants | None |
| `moss_tts_delay` | Alias | 44 | Alias of `tts/moss_tts` | None |
| `moss_tts_local` | Alias | 44 | Alias of `tts/moss_tts` | None |
| `moss_tts_nano` | README | 33 | Default task-qualified family | None |
| `omnivoice` | README | 38 | Default task-qualified family | None |
| `outetts` | README | 29 | Default task-qualified family | None |
| `pocket_tts` | Source | None | Default task-qualified family | None |
| `qwen3` | Source | None | SNAC TTS wrapper over generic Qwen3 backbone | None |
| `qwen3_tts` | README | 35 | Default task-qualified family | None |
| `rumik_oss` | README | 36 | Default task-qualified family | None |
| `sesame` | README | 30 | CSM / MisoTTS; aliases `csm`, `marvis` | None |
| `soprano` | README | 26 | Default task-qualified family | None |
| `spark` | README | 28 | Default task-qualified family | None |
| `tada` | Source | None | Default task-qualified family | None |
| `vibevoice` | Source | None | Alias `vibevoice_streaming` | None |
| `voxcpm` | Source | None | Alias `voxcpm1.5` | None |
| `voxcpm2` | README | 37 | Default task-qualified family | None |
| `voxtral_tts` | README | 43 | Default task-qualified family | None |
| `zonos2` | Source | None | Default task-qualified family | None |

## STT

Speech-to-text and alignment. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/stt/models).

Whisper and Distil-Whisper share one source directory and runtime.
Forced alignment and audio understanding retain their own capabilities.

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `canary` | [Family port](src/stt/canary/README.md) | 11 | Pinned `Mediform/canary-1b-v2-mlx-q8` @ `0b6b32ee10f30c89e3ead7249bb636445e3019ee`; Rust checkpoint transcript byte-identical (15 tokens), mel 9.4e-6, encoder/prefill relative parity gates passed | [Family port](src/stt/canary/mod.rs) |
| `cohere_asr` | Source | 54 | Official `CohereLabs/cohere-transcribe-03-2026` @ `b1eacc2686a3d08ceaae5f24a88b1d519620bc09` is gated; public MLX mirror generated gibberish through mlx-audio, so no runnable profile is verified | None |
| `fireredasr2` | [Family port](src/stt/fireredasr2/README.md) | 55 | Pinned `mlx-community/FireRedASR2-AED-mlx` @ `f3212eacfa49b851130b97c63653c8e06ee09bdb`; full Rust decode matches the MLX transcript and selected encoder stages | One English clip; broader quality, memory/performance, runtime/catalog, FFI, and Swift gates remain open |
| `fun_asr_nano` | Family port | 56 | Pinned `mlx-community/Fun-ASR-Nano-2512` @ `a7bc96fceaafce39ed6748e0c0fa9a9508b67f86`; MLX and Rust matched the transcript and selected LFR, encoder, and adaptor values within `0.01` on one English clip | [Family port](src/stt/fun_asr_nano/README.md) |
| `glmasr` | Family port | 57 | Alias `glm`; pinned `mlx-community/GLM-ASR-Nano-2512-4bit` @ `35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef`; one-clip MLX/Rust transcript and stage parity passed | [Family port](src/stt/glmasr/README.md) |
| `granite_speech` | README | 13 | Default task-qualified family | None |
| `granite_speech5_ctc` | Port in progress | 14 | Pinned 470M TurboCTC profile; exact one-clip Python/Rust transcript match | [Family port](src/stt/granite_speech5_ctc/mod.rs) |
| `granite_speech_nar` | Source | 58 | Pinned MLX conversion `mlx-community/granite-speech-4.1-2b-nar-mlx` @ `6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c`; English MLX smoke passed, Rust port pending | None |
| `higgs_audio_3` | Source | 59 | Pinned `bosonai/higgs-audio-v3-stt` @ `2ffd1aa39f5a1266931e405cba12e404a9f994b2`; English MLX smoke passed, Rust port pending | None |
| `lasr_ctc` | Source | 60 | Google MedASR official pin is gated; public `drankush-ai/medasr-mlx-fp32` @ `3b967580b5176144bc633fac60420d1b122dfba8` runs only after bypassing an upstream double-transpose, and upstream has no working generate path | None |
| `mega_asr` | [Family port](src/stt/mega_asr/README.md) | 17 | Pinned `mlx-community/Mega-ASR-8bit` @ `b9c3c7020f94944205df7f7b5d5d1ce96678d74f`; router and full 539-module LoRA inventory fixture-verified; pinned checkpoint transcript byte-identical (13 tokens); reuses the qwen3_asr tower and decoder | [Family port](src/stt/mega_asr/mod.rs) |
| `mms` | Port in progress | 12 | Pinned English adapter; MLX requires explicit `model_type="mms"`; Rust matches one transcript, but sampled numeric drift through the encoder remains open | [Profile and pin](src/stt/mms/README.md) |
| `moonshine` | README | 8 | Default task-qualified family | [Port](src/stt/moonshine/mod.rs) |
| `moss_music` | README | 23 | Pinned 4-bit profile; explicit transcription prompt required | None |
| `moss_transcribe_diarize` | README | 22 | Pinned 4-bit MLX profile; smoke passed, Rust port pending | None |
| `nemo` | Shared | None | Alignment helpers reused by Parakeet/Nemotron ASR | None |
| `nemotron_asr` | Family port | 10 | Pinned offline RNN-T transcription; cache-aware streaming remains open | [Family port](src/stt/nemotron_asr/README.md) |
| `parakeet` | README | 9 | v2/v3/Redux profiles; alias `parakeet_tdt` | [TDT v2/v3](src/stt/parakeet/mod.rs) |
| `phonon` | Source | 61 | Pinned `FermionResearch/Phonon-1` @ `0428da04625c51b6f069a9829c7060e6b167b92a`; MLX English smoke passed, Rust port pending | None |
| `qwen2_audio` | README | 21 | Pinned 7B 4-bit profile; MLX smoke passed, Rust port pending | None |
| `qwen3_asr` | README | 15 | Pinned 0.6B 8-bit profile; exact one-clip Rust/MLX transcript parity, debug run took 491.94 seconds | [Port](src/stt/qwen3_asr/mod.rs) |
| `qwen3_forced_aligner` | README | 16 | Pinned 0.6B 8-bit alignment profile; MLX and Rust fixture parity passed | [Port](src/stt/qwen3_forced_aligner/README.md) |
| `sensevoice` | Family port | 62 | Pinned `mlx-community/SenseVoiceSmall` @ `8ddd966bd96243cff196422f81f0c5d955814792`; MLX/Rust transcript, tags, token IDs, frontend, and selected encoder stages match on one English clip | [Family port](src/stt/sensevoice/README.md) |
| `vibevoice_asr` | README | 20 | Pinned 1.5B profile; loader needs explicit family route | None |
| `voxtral` | README | 18 | Pinned 3B BF16 profile; MLX smoke passed | None |
| `voxtral_realtime` | README | 19 | Pinned 4B 4-bit profile; MLX smoke passed | None |
| `wav2vec` | Source | None | Default task-qualified family | None |
| `whisper` | README | 6, 7 | Whisper and distilled profiles share one family | [Port](src/stt/whisper/mod.rs) & [Runtime](../runtime/src/whisper/mod.rs) |

### Initial pinned MLX reference runs

Each successful checkpoint was loaded with mlx-audio 0.5.7 and an immutable Hub revision,
then processed the same 16 kHz macOS say clip. Nemotron, Canary, all three
Parakeet profiles, Qwen3-ASR, both Voxtral profiles, VibeVoice, Qwen2-Audio,
and Mega-ASR returned the expected spoken phrase. MOSS-Music returned it with
the explicit transcription prompt, and MOSS-Transcribe-Diarize added speaker
and timestamp markers. Granite5 and Granite Speech lowercased the phrase and
omitted terminal punctuation. MMS returned "the quick bround fox jumps over
the laisy dog" with its English adapter. Qwen3-ForcedAligner returned nine
word intervals. Downloaded model directories were removed after the runs;
the shared small WAV remains in `/tmp` for further checks.

| Profile | Hugging Face repository | Revision | Result |
|---|---|---|---|
| Nemotron 3.5 ASR | `mlx-community/nemotron-3.5-asr-streaming-0.6b` | `e550040c0478027ed679b2b6b0d055502c103663` | MLX 0.5.7 and Rust loaded the pin and returned "The quick brown fox jumps over the lazy dog." on the same clip |
| Canary-1B-v2 q8 | `Mediform/canary-1b-v2-mlx-q8` | `0b6b32ee10f30c89e3ead7249bb636445e3019ee` | MLX 0.5.7 loaded and transcribed the same clip |
| TDT v2 | `mlx-community/parakeet-tdt-0.6b-v2` | `8ae155301e23d820d82aa60d24817c900e69e487` | MLX load and transcription passed |
| TDT v3 | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | MLX load and transcription passed |
| Redux | `moondream/parakeet-redux` | `2bf128600aac4b16946f7ed8372e56117fe5e23b` | MLX load and transcription passed |
| Granite Speech 5.0 TurboCTC 470M | `ibm-granite/granite-speech-5.0-470m-turboctc` | `947f59af40db9791170a0628cf0f3f4812d720f1` | MLX 0.5.7 and Rust returned "the quick brown fox jumps over the lazy dog" |
| Granite Speech 1B | `ibm-granite/granite-4.0-1b-speech` | `bd87ab862416353633ea431fe49b1614003623c5` | MLX 0.5.7 returned the expected phrase in lowercase; local environment needed optional `jinja2` |
| MMS 1B English | `facebook/mms-1b-fl102` | `d483345545bea550895b1aa0c6ba40236b9f1e22` | MLX 0.5.7 required `model_type="mms"`; MLX and Rust returned "the quick bround fox jumps over the laisy dog"; sparse sampled activation drift remains documented |
| Phonon-1 | `FermionResearch/Phonon-1` | `0428da04625c51b6f069a9829c7060e6b167b92a` | MLX 0.5.7 returned "The quick brown fox jumps over the lazy dog."; temporary model cache removed |
| SenseVoice Small | `mlx-community/SenseVoiceSmall` | `8ddd966bd96243cff196422f81f0c5d955814792` | MLX 0.5.7 and Rust returned the exact transcript, tags, and CTC token IDs; four selected encoder stages and frontend tensors match within `0.01`; temporary model cache removed |
| Qwen3-ASR 0.6B 8-bit | `mlx-community/Qwen3-ASR-0.6B-8bit` | `89e96d92ba34aca20b3e29fb10cc284097d1219f` | Pinned MLX and Rust runs returned the exact same full transcript; Rust debug run took 491.94 seconds |
| Qwen3-ForcedAligner 0.6B 8-bit | `mlx-community/Qwen3-ForcedAligner-0.6B-8bit` | `0e1a68e91d815300c7c9754b2a7639378b23db15` | MLX 0.5.7 and Rust matched all nine word spans within 1 ms; Rust fixture is checked in |
| Mega-ASR 8-bit | `mlx-community/Mega-ASR-8bit` | `b9c3c7020f94944205df7f7b5d5d1ce96678d74f` | MLX 0.5.7 returned the expected phrase |
| Voxtral Mini 3B BF16 | `mlx-community/Voxtral-Mini-3B-2507-bf16` | `ab69f21c8e2c52b05335bf282898e141c8c42477` | MLX 0.5.7 returned the expected phrase; Transformers processor required local `jinja2`, `librosa`, `mistral-common`, and PyTorch |
| Voxtral Realtime 4B 4-bit | `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit` | `fdebf7b2af834a1db4b8a3c99ab7480b333adf9e` | MLX 0.5.7 returned the expected phrase |
| VibeVoice ASR Streaming 1.5B | `microsoft/VibeVoice-ASR-Streaming-1.5B` | `4262d23d8a539a6530cf64fbd0b1751ef9a30853` | MLX 0.5.7 returned the expected phrase with speaker prefix when routed as `vibevoice_asr` |
| Qwen2-Audio 7B 4-bit | `mlx-community/Qwen2-Audio-7B-Instruct-4bit` | `c65570002626f41b4dc08b7b54f42f99f3e82e7f` | MLX 0.5.7 returned the expected phrase |
| MOSS-Transcribe-Diarize 4-bit | `vanch007/mlx-MOSS-Transcribe-Diarize-4bit` | `d42a296ee807e933ddd7588e2041dbbc84aff85d` | MLX 0.5.7 returned the phrase with speaker and timestamp markers |
| MOSS-Music 8B Thinking 4-bit | `mlx-community/MOSS-Music-8B-Thinking-4bit` | `b14123a419b6254b92d9e55b8a5b6f7285e05c70` | Explicit transcription prompt returned the expected phrase and one 0.00-2.795 s segment |
| FireRedASR2-AED | `mlx-community/FireRedASR2-AED-mlx` | `f3212eacfa49b851130b97c63653c8e06ee09bdb` | mlx-audio 0.5.7 and the full Rust encoder/beam decoder returned "the quick brown fox jumps over the lazy dog"; selected frontend and first-block stages match within 0.03; temporary model cache removed |
| Fun-ASR-Nano-2512 | `mlx-community/Fun-ASR-Nano-2512` | `a7bc96fceaafce39ed6748e0c0fa9a9508b67f86` | MLX 0.5.7 returned "The quick brown fox jumps over the lazy dog."; temporary model cache removed |
| GLM-ASR-Nano-2512 4-bit | `mlx-community/GLM-ASR-Nano-2512-4bit` | `35553fa5bebfcc3ece3ce7d47b98827cb0ac9eef` | MLX 0.5.7 and Rust returned "The quick brown fox jumps over the lazy dog."; Transformers warned about model_type, inference succeeded; temporary cache removed |
| Granite Speech 4.1 2B NAR MLX | `mlx-community/granite-speech-4.1-2b-nar-mlx` | `6acb7892068dd30227f20aba6eb7c4b0ae5c7e7c` | MLX 0.5.7 returned "the quick brown fox jumps over the lazy dog...."; peak memory 5.10 GB; temporary model cache removed. The official IBM checkpoint loaded but failed inference because its PyTorch Conv1d weight layout is not the MLX layout |
| Higgs Audio v3 STT | `bosonai/higgs-audio-v3-stt` | `2ffd1aa39f5a1266931e405cba12e404a9f994b2` | MLX 0.5.7 returned "the quick brown fox jumps over the lazy dog"; temporary model cache removed |
| Cohere Transcribe 03-2026 official | `CohereLabs/cohere-transcribe-03-2026` | `b1eacc2686a3d08ceaae5f24a88b1d519620bc09` | Hub returned 403 because this environment lacks gated access; partial cache removed |
| Cohere Transcribe 03-2026 MLX mirror | `mlx-community/cohere-transcribe-03-2026-mlx-8bit` | `a0acb7f93cd32d82c4fbf801b6d8fb39d20c509f` | The `mlx-int8/` subfolder loaded through mlx-audio 0.5.7 but emitted incoherent text on the reference clip; rejected as a runnable profile and temporary cache removed |
| MedASR official | `google/medasr` | `ae1e4845b4b07479735d93e1e591e566435b7104` | Hub returned 403 because this environment lacks gated access |
| MedASR MLX FP32 candidate | `drankush-ai/medasr-mlx-fp32` | `3b967580b5176144bc633fac60420d1b122dfba8` | Stock load() double-transposed MLX Conv1d weights and the upstream generate/decode helpers are broken. Reloading the raw MLX weights and using LASRProcessor produced "quick brown fx jumps over the laazy dog." from model forward logits; partial smoke only, temporary cache removed |
| MedASR MLX FP16 candidate | `ainergiz/medasr-mlx-fp16` | `b72d5fd4605f7beb84cc6c895b29be0923a890ac` | Upstream weight sanitization failed on an unexpected Conv1d tensor shape; temporary model cache removed |

These single-clip runs do not qualify broader recognition quality or a
catalog profile. Stage-level encoder comparisons, broader quality, runtime,
FFI, and Swift gates remain open.

The Rust implementation loads each pinned v2/v3 checkpoint and the pinned
Redux checkpoint, matching the MLX Python reference on the same generated
16 kHz smoke WAV. Redux ternary weights expand to temporary F32 tensors during
load; its temporary safetensors file and all downloaded checkpoints were
removed afterward. This is one transcription witness per profile. Stage-level
fixtures, broader quality, runtime/catalog, FFI, Swift, performance, and
memory qualification remain open.

## STS

Speech-to-speech and audio processing. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/sts/models).

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `deepfilternet` | README | 46 | Enhancement; v1/v2/v3 are separate profile gates | [Flat port](src/sts/deepfilternet/mod.rs) |
| `dialogue_sidon` | README | 49 | Separation / restoration; alias `dialoguesidon` | None |
| `lfm_audio` | README | 50 | STS / TTS / STT; aliases `lfm2_audio`, `lfm2.5` | None |
| `mel_roformer` | Source | None | Default task-qualified family | None |
| `mimo_audio` | README | 51 | Base/Instruct; STS / TTS / STT / understanding | None |
| `moshi` | Source | None | Alias `moshiko` | None |
| `mossformer2_se` | README | 47 | Enhancement; alias `mossformer2` | None |
| `nemotron_voicechat` | README | 52 | Full-duplex STS / transcription / function calling | None |
| `sam_audio` | README | 48 | Text-guided source separation; alias `samaudio` | None |

## VAD

Voice activity, diarization and turn detection. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/vad/models).

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `fsmn` | Source | None | Default task-qualified family | None |
| `nemotron_diarization` | README | 5 | Streaming speaker diarization | None |
| `silero_vad` | README | 2 | Detection; aliases `silero`, `silero-vad` | [Flat port](src/vad/silero_vad/mod.rs) |
| `smart_turn` | Source | None | Turn detection, separate from speech activity | None |
| `sortformer` | README | 3, 4 | v1 and streaming v2.1 diarization profiles | None |

## LID

Language identification. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/lid/models).

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `ecapa_tdnn` | Source | None | Language ID; alias `ecapa-tdnn` | None |
| `wav2vec2` | Source | None | Language ID; distinct from `stt/wav2vec` | None |

## MUSIC

Music generation. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/music/models).

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `minimax_music3` | [Family port](src/music/minimax_music3/mod.rs) | 53 | Song generation; precision profiles qualify separately | Yes |

## CODEC

Codecs and shared learned components. [Pinned source directory](https://github.com/Blaizzy/mlx-audio/tree/e1b19b9054bf163f5d812221a54fcc346f1890e9/mlx_audio/codec/models).

| Source directory | Class | Local task(s) | Identity / profile notes | Current Rust path |
|---|---|---|---|---|
| `bigvgan` | Family port | None | Tiny fixture end to end (codes/wave parity, kaiser Activation1d, checkpointed filters); snake refused, see [family README](src/codec/bigvgan/README.md) | Yes |
| `dacvae` | Shared | None | Default task-qualified family | None |
| `descript` | Family port | None | Tiny fixture end to end (all-level codes exact with margins, z_q/audio/from_codes parity, compress round trip); delay-0 quirk documented, see [family README](src/codec/descript/README.md) | Yes |
| `ecapa_tdnn` | Family port | None | Tiny fixtures (plain + global-context pooling) embedding parity, see [family README](src/codec/ecapa_tdnn/README.md) | Yes |
| `encodec` | Family port | None | Tiny fixture end to end (codes exact at two bandwidths with Euclidean margins, audio parity, isolated LSTM); mono + weight_norm only, see [family README](src/codec/encodec/README.md) | Yes |
| `fish_s1_dac` | Shared | None | Default task-qualified family | None |
| `higgs_audio` | Shared | None | Audio tokenizer; distinct from TTS generation family | None |
| `mimi` | Shared | None | Default task-qualified family | None |
| `mimo_audio_tokenizer` | Shared | None | Default task-qualified family | None |
| `moss_audio_tokenizer` | Shared | None | Default task-qualified family | None |
| `nemotron_voicechat` | Shared | None | Codec dependency; distinct from STS pipeline | None |
| `s3` | Shared | None | Tokenizer / codec dependency | None |
| `snac` | Family port | None | snac_24khz verified end to end (fixture codes exact, real checkpoint encode exact + zero-noise decode parity); LocalMHA refused, see [family README](src/codec/snac/README.md) | Yes |
| `stepaudio2` | Shared | None | Token-to-waveform dependency | None |
| `vocos` | Family port | None | Tiny fixture end to end (log-mel front end + ConvNeXt + iSTFT head parity, numpy-convention irfft); mel variant, see [family README](src/codec/vocos/README.md) | Yes |
