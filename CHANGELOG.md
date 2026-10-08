# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Every publishable crate in the workspace shares one version number
(`workspace.package.version` in the root `Cargo.toml`), so a single entry
here covers a release even when only some crates changed in it. See
[`docs/RELEASE.md`](docs/RELEASE.md) for how a release is cut, including
when this file gets updated relative to the version bump and the tag.

## [Unreleased]

### Security
- `swift/TurboSparkApp`: file tools resolve symlinks on the deepest existing
  ancestor before the containment check, so `write_file`, `apply_patch`,
  `multi_edit`, `notebook_edit` and the REPL can no longer create files
  outside the project through a symlinked directory. `search_code` skips
  symlinks and non-regular files, so a link to `~/.ssh` is not read.
- `swift/TurboSparkApp`: `web_fetch` validates every redirect hop and the
  connected peer address, and the host check normalizes IPv4-mapped IPv6,
  trailing-dot and alternate loopback spellings. DNS rebinding is narrowed,
  not closed: the check runs after the connection, so a rebound host can still
  receive a request, but its response is discarded.
- `swift/TurboSparkApp`: project custom tools (`.turbospark/tools`,
  `.agents/tools`) are blocked until the user trusts them per project, cannot
  take a built-in tool's name, and cannot declare a category below terminal
  (command and script tools) or web (http tools). An approval sheet shows the
  full command, environment and headers. Reject is the default.
- `swift/TurboSparkApp`: one `ToolArgumentResolver` now backs the executor,
  approval card, diff formatter and risk classifier, and calls that supply
  conflicting aliases are refused. Write aliases such as `write_to_file`
  categorize as file writes, so batch children reach the sensitive-path gate.
  `call_mcp_tool` permission and risk use the resolved target tool.
- `swift/TurboSparkApp`: scheduled cron and wakeup prompts are delivered to the
  model as plain text and no longer run as `!` shell commands that skipped the
  permission engine.
- `swift/TurboSparkApp`: Agent mode auto-allows read-only asks only. File
  writes, interpreters, custom tools and scheduling go through the classifier
  or an approval card.
- `swift/TurboSparkApp`: every app git call on a project repository runs with
  `core.fsmonitor`, hooks, pagers, external diff and textconv disabled
  (`HardenedGit`), so a planted `.git/config` cannot execute code on a status
  refresh or `@diff`.
- `swift/TurboSparkApp`: approval sheets for repo-declared and imported MCP
  servers show the full command, working directory, environment and header
  names, and flag loader variables (`NODE_OPTIONS`, `DYLD_*`, `LD_*`, `PATH`).
  Return no longer approves a repo-declared server. The multi-edit and
  `http_request` approval cards show their payload with secrets hidden.
- `swift/TurboSparkApp`: browser dialogs show the source origin and switch to
  their tab, a navigate approval no longer transfers to a padded or different
  URL, the encrypted export requires recent authentication on a protected
  profile, the plaintext export no longer includes MCP credentials, and
  sensitive plugin options stay out of the plaintext hook options file.

- `swift/TurboSparkApp`: enabling profile protection now rotates the master
  key instead of wrapping the old plaintext one. The new key is wrapped and
  durable before any data changes, the database and every asset and recovery
  copy are re-encrypted, and the old key is dropped last. An interrupted
  rotation resumes on the next unlock. The manifest is v2 only while a
  rotation is pending, so older builds refuse it. Profiles protected by an
  older build are not rotated retroactively. Disable and re-enable
  protection to rotate them.
- `swift/TurboSparkApp`: key derivation and Keychain prompts no longer run
  under the vault lock, and locking waits for or cancels in-flight asset
  imports instead of wiping the key under them. Exact backup export snapshots
  rows and asset ids together and authenticates every asset, and restore no
  longer writes a plaintext-key file to scratch.
- `crates/repack`, `crates/catalog`, `swift/TurboSparkApp`: the Hugging Face
  token is sent only over https to `huggingface.co`, `hf.co` and their
  subdomains. A configured mirror or a redirect target never receives it, and
  token validation reports a mirror instead of sending the token to it.
- `swift/TurboSparkApp`: a locked or denied Keychain is no longer read as "no
  key", so it cannot delete a stored server or TypeSafe key. `stop_agent`
  treats another chat's agent as unknown.
- `crates/ffi`: the string, session, image and server teardown entry points
  run inside `abi::guard_value`, so a panic cannot unwind across `extern "C"`.
  A test fails on any new unguarded entry point.
- `crates/audio`, `swift/TurboSparkApp`: bounded audio normalization and enforced
  resource limits, validating decoded sample rates (8,000..=192,000 Hz) and channel
  counts (1..=8), capping probe frame counts, chunking resampling buffers, and
  adding RAII cleanup guards to prevent memory exhaustion and DoS on malformed audio.

### Added
- `crates/audio`: added speech-to-speech (STS) model family implementations and test
  fixtures for DialogueSidon, Mel-RoFormer, MossFormer2-SE, and SAM-Audio.
- `crates/audio`, `crates/runtime`, `crates/ffi`, `swift/TurboSparkApp`: native audio
  workspace architecture with audio capture, waveform library management, media IO,
  controls, visualizers, recording setup, playback, and app model audio integration.
- `crates/audio`, `crates/runtime`, `crates/cli`, `crates/gpu`: MiniMax Music 0.3 Metal
  performance optimizations, including affine wide GEMV packing, exact-bf16 16-bit
  scale storage, eight 4-bit codes per word decoding in tiled linear kernels, tiled
  MMA convolution kernels, cancellation and progress callback APIs, and benchmark drivers.
- `swift/TurboSparkApp`: approval sheet for project custom tools
  (`ProjectToolApprovalSheet`) backed by `CustomToolTrustStore`.
- `swift/TurboSparkApp`: MCP stdio servers keep one persistent session per
  server and configuration (`McpStdioSession`). Concurrent calls multiplex over
  it by JSON-RPC id, a crash fails in-flight calls with the stderr tail and the
  next call restarts it, and idle sessions close after five minutes.
  Discovery still spawns a fresh process.
- `crates/repack`: GGUF installs record the repo, the 40-hex commit and the file
  in `.resume-provenance.json` and resume from leftover expert layers only when
  the record matches.
- `swift/TurboSparkApp`: `make swift-test-app` and `make swift-test-qwenimage`
  run the TurboSparkApp and QwenImage suites, which no make target ran before.
  See `docs/TESTING.md`. The QwenImage suite needs the Metal toolchain.
- `swift/TurboSparkApp`: new regression suites for symlink containment, web
  rebinding, persistence migration, wire decoding, tool vocabulary parity and
  the agent loop fixes below.
- `crates/audio`: consolidated and shared core neural network ops and layers
  (`nn.rs`: `Conv1d`, `ConvTranspose1d`, `Linear`, `LayerNorm`, `RMSNorm`,
  `Embedding`), shared vocoder and codec blocks (`conv.rs`, `dac.rs`,
  `wnconv.rs`), and vectorized DSP primitives with window caching for FFT,
  STFT, iSTFT, mel filterbanks, and resampling; optimized hot decoder paths
  for FireRedASR 2, Granite Speech LLM, Qwen3-ASR, SenseVoice SANM, and Wav2Vec.
- `swift/TurboSparkApp`: added audio model awareness to the Server pane,
  including `ServerModelRow` audio badges, live model count aggregates across
  text, image, and audio models, and dynamic audio model ejection and
  attachment via `AppModel+ServerAttachment`.
- `crates/server`, `crates/ffi`, `swift/TurboSparkApp`: prompt-to-PNG image
  endpoint (`POST /v1/images/generations`) backed by `ImageBridge`, event
  polling, and serial image generation jobs.
- `swift/TurboSparkApp`: added API workspace destination with mode selector
  supporting Text serving, Image serving, and TypeSafe OpenKind playground
  with model controls, question builders, and latency benchmarks.
- `crates/gpu`: added batched attention Metal kernels for Llama prefill
  (`batch_attention_partial.metal`, `batch_attention_combine.metal`), dynamic
  tiling, row isolation, and scratch layout allocation.
- `crates/gpu`: added Metal pipeline caching, precompiled shader libraries,
  and opt-in kernel warmup (`TURBOSPARK_METAL_KERNEL_WARMUP=1`).
- `swift/TurboSparkApp`: native WebKit browser automation pane with tab engine,
  element locator resolution, DOM snapshot walker, native dialog policy, and
  bookmark import.
- `swift/TurboSparkApp`: sandboxed JavaScript REPL tool with isolated
  JavaScriptCore worker, deny-by-default capability security, persistent worker
  evaluation context, and top-level await execution.
- `crates/workflows`: workflow execution engine, step validation, and execution
  facade.
- `swift/TurboSparkApp`: chat runtime quality improvements, including validated
  streamed tool-call dispatch, mid-turn presentation labels, chat quality
  settings, stream recovery, and provenance metadata tracking.
- `crates/server`: added server idle policy, idle connection sweeper, and
  dynamic multi-model registry detachment.
- `swift/TurboSpark`: added native Swift speech-to-text bindings (`Speech.swift`)
  wrapping the C ABI: `TurboSparkSTTModel`, `TurboSparkSTTStream`, `STTOptions`,
  `STTAppendProgress`, `STTSegment`, and `STTTranscription`.
- `swift/TurboSpark`: added audio model management to `TurboSparkServer`,
  including `ServerOptions` audio model maps (`sttModels`, `ttsModels`,
  `musicModels`), `ServerInfo.audioModels`, and dynamic `attachAudioModel` and
  `detachAudioModel` methods.
- `crates/ffi`: exported `ts_server_attach_audio_model` and
  `ts_server_detach_audio_model` C ABI entry points in `turbospark.h`, wired
  `RealAudioProvider` into `ServerState` for startup and dynamic audio model
  attachment and detachment, and populated `audio_models` in `ServerInfo`.
- `crates/server`: added REST and streaming audio API endpoints
  (`POST /v1/audio/transcriptions`, `POST /v1/audio/speech`,
  `POST /v1/audio/music`, `GET /v1/audio/models`, `POST /v1/audio/models/attach`,
  `POST /v1/audio/models/detach`, and `GET /v1/audio/transcriptions/stream`),
  backed by multi-threaded `RealAudioProvider` workers and OpenAPI documentation.
- `crates/audio`: added speech-to-text model families: Granite Speech,
  Higgs Audio 3, LASR CTC, Moss Transcribe & Diarize, Phonon, and Wav2Vec.
- `crates/audio`: added neural audio codecs and tokenizers: DAC-VAE, Fish S1 DAC,
  Higgs Audio Codec, Mimi, Mimo Audio Tokenizer, Moss Audio Tokenizer,
  Nemotron VoiceChat Codec, S3 Codec, and StepAudio2 Codec.
- `crates/audio`: added Kokoro multi-voice speech synthesis support (`synth.rs`).
- `swift/TurboSparkApp`: encrypted profile memory system featuring
  `ProfileDatabase`, `ProfileMemoryStore`, and Keychain-backed `ProfileVault`.
- `swift/TurboSparkApp`: added `MemoryLedger` for append-only audit tracking
  of memory operations and `MemorySettingsPaneView` for user control and search.
- `swift/TurboSparkApp`: integrated `MemoryTool` with chat generation loop and
  scheduler (`AppModel+MemoryScheduler.swift`).
- `crates/audio`: added MiniMax Music 0.3 flow matching DiT, transformer depth
  decoder, and vocoder supporting MXFP4, MXFP8, and NVFP4 formats.
- `crates/gpu`: added Metal compute kernels and dispatches for MiniMax Music 0.3
  (`music3.metal`, `music3_attention_fallback.metal`).
- `crates/runtime`: added `Music3Runner` with Metal acceleration, multi-chunk
  flow matching, and stereo waveform generation.
- `crates/cli`: added `turbospark-music` CLI binary for native music generation
  and `turbospark-model list-music` catalog command.
- `crates/catalog`, `crates/repack`: added resumable and cancellable HTTP
  downloads with retry backoff and progress tracking.
- `crates/catalog`: added audio and music catalogs (`audio_catalog.rs`, `music.rs`,
  `audio_assets.json`, `music_models.json`).
- `crates/audio`: added neural audio codecs: SNAC, EnCodec, Descript Audio
  Codec (DAC), ECAPA-TDNN, BigVGAN, and Vocos.
- `crates/audio`: added speech-to-text model families: Canary, Qwen3-ASR with Metal
  acceleration, SenseVoice, FireRedASR-2, FunASR Nano, GLM-ASR, and Mega-ASR,
  as well as Sortformer VAD.
- `crates/model-io`: added `SpeechFamily::Qwen3Asr`.
- `crates/ffi`: added in-memory cached embedding encoder (`api/embedding.rs`) and
  multi-family STT runner dispatch supporting Qwen3-ASR alongside Whisper.
- `crates/audio`: new unified portable audio crate (`turbospark-audio`)
  implementing DSP primitives (waveform buffers, RIFF WAV I/O, linear and
  sinc-Hann resampling, real FFT/STFT/iSTFT, mel filterbanks, OpenAI Whisper
  and NeMo log-mel frontends), neural tensor ops, MLX groupwise affine
  dequantization, and model families organized by role (`music/`, `stt/`,
  `tts/`, `vad/`, `sts/`, `codec/`, `lid/`).
- `crates/audio`: Qwen3-ASR descriptor-only checkpoint validation
  (`stt::qwen3_asr::checkpoint::validate_checkpoint`) and a config-derived
  CPU resident estimate (`Qwen3Config::estimated_resident_bytes`) backing
  the qwen3_asr install probe and receipt.
- `crates/compute`: added `whisper.rs` containing FP32 CPU reference kernels
  for Whisper 1D convolution, Pre-LN encoder blocks, and incremental decoder
  with KV cache.
- `crates/gpu`: added Metal compute kernels and dispatches for Whisper 1D
  convolution (`whisper_conv.metal`, `whisper_conv.rs`), Pre-LN transformer
  encoder blocks (`whisper_encoder.metal`, `whisper_encoder.rs`), and Moonshine
  frontend (`moonshine_frontend.metal`, `moonshine_frontend.rs`), validated
  by parity test suites.
- `crates/model-io`: added `SpeechFamily::Whisper`, `WhisperConfig` parsing
  and validation, and typed `SpeechInstallReceipt` schema for speech model
  directories.
- `crates/model-io`: added `SpeechFamily::Qwen3Asr` (wire string
  `qwen3_asr`) as the second speech family.
- `crates/runtime`: added `WhisperRunner` and Moonshine execution paths
  supporting audio preprocessing, 30-second windowing, cross-attention caching,
  language identification, and timestamped segment emission.
- `crates/runtime`: added `Qwen3AsrRunner`, the portable CPU execution path
  for the qwen3_asr speech family, emitting one clip-level segment.
- `crates/catalog`: added speech model catalog (`speech_models.json`),
  `SpeechCatalogEntry`, and speech install verification.
- `crates/catalog`: the speech probe now dispatches on the staged
  `config.json` model type, validates qwen3_asr installs (tokenizer assets,
  descriptor-only checkpoint shapes, family runtime file set), and the
  bundled table gains the pinned `qwen3-asr-06b-8bit` row plus the
  `mlx-8bit-bf16` format.
- `crates/ffi`: `ts_stt_open` now dispatches on the installed speech
  family, opening Qwen3-ASR installs through the same streaming surface.
- `crates/cli`: added `turbospark-model list-speech` subcommand to display
  bundled speech models and local installation status.
- `crates/ffi`: added C ABI speech-to-text surface (`ts_stt_open`,
  `ts_stt_close`, `ts_stt_stream_open`, `ts_stt_stream_append`,
  `ts_stt_stream_finish`, `ts_stt_stream_cancel`, `ts_stt_stream_close`)
  and `TsSttModel`/`TsSttStream` handles in `include/turbospark.h`.
- `docs/AUDIO.md`, `docs/SPEECH_TO_TEXT.md`, `docs/WHISPER.md`: comprehensive
  architecture guides, model catalog, and verification protocols for audio and
  speech-to-text inference.
- `swift/TurboSparkApp/Vendor/QwenImage`: original Swift + MLX port of the
  Qwen-Image-2.1 text-to-image pipeline (Qwen3-VL text encoder, single-stream
  block-causal DiT with prefix KV cache, RMS-norm VAE decoder, flow-match
  Euler scheduler with dynamic exponential shifting), with component-level
  parity tests against the diffusers formulas and a pinned-provenance
  `UPSTREAM.md`.
- `swift/TurboSparkApp`: Qwen-Image-2.1 generation in the app through
  `QwenImageGenerationSession`, family dispatch by model ID, the
  "Qwen-Image 2.1" model family row, and a 32 GiB memory tier for the
  `qwen-image-2.1-mlx-4bit` download recommendation.
- `swift/TurboSparkApp`: `QwenImageMLXBenchmark` executable target mirroring
  the `ZImageMLXBenchmark` timing and peak-memory contract.
- `crates/catalog`, `crates/image`, `crates/ffi`: added the
  `qwen-image-2.1-mlx-4bit` catalog row pinning
  `mlx-community/Qwen-Image-2.1-MLX-4bit` at revision
  `4db4e8c0c0e7a1debf0320415bec8388e888494c`, a per-family image install
  envelope (40 steps, one forward per step, guidance 1.0, 64-channel
  latents), `processor/` layout intake, and `mlx-community/` source
  retention.
- `docs/QWEN_IMAGE_21_MLX.md`: implementation and evidence record for the
  new family, including the head-to-head benchmark against the Z-Image
  Turbo MLX rows.
- `crates/image`, `crates/ffi`, `crates/cli`: added support and preflight
  validation for bounded image resolutions (sides from 512 to 1024 pixels in
  multiples of 16, up to 1024x1024 total pixels) with dynamic latent shaping
  and VAE decoding in `MetalImageBackend`, while keeping the CPU reference
  backend pinned to 1024x1024 as a diagnostic-only path.
- `crates/cli`: added multi-stage progress bar in `turbospark image generate`
  displaying real-time progress across prompt encoding, denoising steps, VAE
  decoding, and PNG writing.
- `swift/TurboSparkApp`: added `ImageResolutionPreset` with square (512, 768,
  1024), 4:3, and 16:9 portrait and landscape options, decoupled resolution
  selection from the installed model, added an image size picker to
  `ImageGenerationSettings`, and updated progress tracking and tests.

- `crates/catalog`, `crates/image`, `crates/ffi`, `swift/TurboSparkApp`: added
  scaffolding and intake validation for `baa-ai/Krea-2-Turbo-RAM-9GB-MLX`
  pinned at revision `58d4ebf1d13c3b086bde5ee45c038b299347dae9`, including
  the catalog entry, manifest envelope (8 steps, guidance 1.0, 1024 prompt
  bound), required file list, and Swift `Krea2RAMQuantization` validation for
  `ram_bits.json` 4/5/6/8-bit module assignments and safetensors shard layout.
- `ROADMAP.md`: added priority item tracking Krea 2 Turbo Native MLX bring-up
  status, safety requirements, and remaining pipeline integration tasks.

### Changed
- `swift/TurboSparkApp`: streamlined `ImagesSectionView` with centered progress
  overlay tracking granular generation stages (`downloading_model`,
  `loading_tokenizer`, `loading_text_encoder`, `loading_transformer`,
  `loading_vae`, `text_encoder`, `transformer`, `vae_decoder`, `png_encode`)
  with localized labels in `Localizable.xcstrings`.
- `swift/TurboSparkApp`: focused desktop image generation on tested Z-Image
  Turbo MLX models and removed unused QwenImage package dependencies and
  session wrappers.
- `swift/TurboSparkApp/Vendor/ZImage`: added validation in `ZImageQuantizer` to
  enforce supported quantization modes (`affine`, `mxfp4`), bit widths (2-8),
  and valid group sizes while rejecting conflicting settings across shards;
  improved error reporting in `ModuleWeightsApplier` and weights mapping.
- `crates/server`: added validation guard ensuring `turbospark-server --model`
  refuses speech model directories, directing users to the STT C ABI.
- `swift/TurboSparkApp/Vendor/ZImage`: updated `ZImagePipeline` denoising loop
  to fire progress notifications upon step completion.
- `docs/CLI.md`, `docs/IMAGE_GENERATION.md`, `docs/ZIMAGE_TURBO.md`: documented
  the bounded image resolution envelope and clarified that pinned quality and
  hardware resource evidence remains specific to the 1024x1024 baseline.
- `swift/TurboSparkApp`: SQLite child tables (`messages`, `alternates`,
  `attachments`, `artifacts`) are keyed per chat and `todos` by
  `(chat_id, ordinal)`, with an automatic migration on open. Older app builds
  cannot read a database written by this one.
- `swift/TurboSparkApp`: scheduled prompts are injected as model-origin text
  instead of through the composer. They wait for an idle chat and a loaded
  model, and no longer overwrite a draft.
- `swift/TurboSparkApp`: `todo_write` refuses a missing or unparseable list
  instead of clearing it. Send `"[]"` to clear.
- `swift/TurboSparkApp`: the workflow script checker rejects undeclared
  identifiers, duplicate bindings and reserved binding names, so a saved
  workflow that used one now fails the check.
- `swift/TurboSparkApp`: `Config set`, `ListMcpResources`, `ReadMcpResource`,
  `ProposeGoal` and `SendFeedback` return an explicit "not implemented"
  error instead of reporting success.
- `swift/TurboSparkApp`: the app bundle includes and signs the `turbospark` CLI,
  and `make dmg` fails if it is missing.
- `swift/TurboSparkApp`: fetching a marketplace no longer saves a source. An
  explicit Add source button does, and it refuses a name that already points at
  a different source. The worktree switch is relabeled "View in Git pane",
  because it never moved the agent's working root.
- `crates/ffi`: `ts_store_relocate` returns `TS_ERR_INVALID_ARGUMENT` while an
  install is registered. The check is not atomic against an install that
  starts afterwards, so the app must also stop queueing installs during a move.
- `.github/workflows/release.yml`: the app cask links the bundled `turbospark`
  binary. This was checked by rendering the casks with dummy values, not by a
  release run.
- `crates/catalog`: installers announce the byte transfer with a
  `[phase] transfer` progress line, which the app maps to the running state.
  Later repack lines still read as packing because there is no end marker yet.

### Fixed
- `swift/TurboSparkApp`: use native `AudioObjectID` array rather than `NSNumber` in
  `CATapDescription` initializers to fix compilation on macOS 15+ SDK.
- `swift/TurboSparkApp`: parenthesized optional existential type in
  `AppModel.swift` as `(any ImageGenerationSession)?` to satisfy Swift 6 and
  Xcode 16 syntax requirements on macos-15 packaging runners.
- `swift/TurboSparkApp`: upgrading from a pre-vault build no longer deletes
  project memory and tool observations. The legacy migration walked only to
  the first directory and then removed the whole root.
- `swift/TurboSparkApp`: a todo id reused across chats, or a branched or
  duplicated chat, no longer makes every later chat save fail with a UNIQUE
  constraint error.
- `swift/TurboSparkApp`: exact backup export skips orphan asset files and
  writes atomically, restore creates a missing `profiles/` directory, and
  asset headers take their length from the bytes actually encrypted.
  Undecodable chat and project rows are no longer deleted by the next save.
- `swift/TurboSpark`: `ChatMessage` decodes the null `path` or `base64` field
  Rust emits for image parts, so a history with an image no longer reports
  that the conversation does not fit.
- `swift/TurboSparkApp`: image generation holds its permit until the producer
  task finishes, so Stop no longer lets two generations share one pipeline.
- `swift/TurboSparkApp`: MCP tool arguments keep their JSON types, and MCP
  children get a PATH containing the resolved command's directory and the
  Homebrew locations, so `npx` servers start when the app opens from Finder.
- `swift/TurboSparkApp`: non-fork `/skill` commands run, a guardrail retry
  that is declined no longer leaves the chat stuck generating, `/rewind`
  resets the compaction boundary, duplicate todo ids no longer crash, and
  changing the approval mode keeps MCP rules and the browser allowlist.
- `swift/TurboSparkApp`: `read_file` in `time_machine` mode no longer hangs on
  more than 64 KB of git output, and `apply_patch` refuses a new-file header
  over an existing file.
- `swift/TurboSparkApp`: a failed background-shell launch no longer closes
  pipes its reader threads still use, and bang commands, the classifier
  timeout and Stop no longer cancel each other's work.
- `swift/TurboSparkApp`: `glob` honors its pattern, `search_code` honors regex
  and filters, `apply_patch` is hunk-aware and keeps CRLF files CRLF, and
  `notebook_edit` targets the right cell.
- `swift/TurboSparkApp`: cron `nextFire` no longer skips a minute, one-shot
  wakeups keep their fire time, and impossible schedules are rejected.
- `swift/TurboSparkApp`: the chat sidebar layout is built once per body, worktree
  diffs parse once per distinct text, and the status bar no longer refreshes
  telemetry on every streamed token. No speedup was measured.
- `swift/TurboSparkApp`: tool, state, browser, model hub, server and
  presentation fixes from a full review of `swift/`. Of the review's 28
  critical and high issues all are fixed. Most of the 244 medium and 319 low
  findings were triaged and about a third to three quarters fixed. The rest
  are deferred and recorded in the review.

## [0.2.0] - 2026-09-26

### Added
- `crates/gpu`: added GPU top-k block selection kernel (`qsa_topk.metal`,
  `qsa_topk.rs`) and state management (`qsa_indexer_state.rs`) for Qwen4-Exp
  QSA attention, eliminating host-side commit-and-wait synchronization in
  sequential decode and chunked prefill.
- `crates/catalog`, `crates/cli`, `crates/ffi`, `crates/image`: added intake
  and builder support for mflux-converted Z-Image repositories
  (`deepsweet/Z-Image-Turbo-6B-MLX-Q4` and `deepsweet/Z-Image-6B-MLX-Q8`),
  synthesizing missing family/quantization configs and supporting
  `model.safetensors.index.json` component naming.
- `swift/TurboSparkApp`: added bundled image style preset catalog
  (`image-styles.json`), `AppImageStyle`, style picker menu
  (`ImageStylePicker`), and favorites persistence store
  (`ImageStyleFavoriteStore`) with multi-language localization.
- `swift/TurboSparkApp`: in-process Apple MLX Swift integration for Z-Image
  Turbo generation. Vendored `Z-Image.swift` under
  `swift/TurboSparkApp/Vendor/ZImage` to provide standalone, hermetic builds
  without remote Git package dependency overhead.
- `swift/TurboSparkApp`: `ZImageMLXBenchmark` executable target under
  `Benchmarks/ZImageMLXBenchmark` for headless timing, model loading, and
  pipeline throughput evaluation.
- `docs/ZIMAGE_TURBO_MLX_SPEED.md`: exploratory speed and schedule investigation
  for 8-bit Z-Image-Turbo MLX generation on Apple M4 Max.
- `docs/DESKTOP_FEATURE_PARITY.md`: comprehensive 530+ feature comparison
  matrix and architecture review evaluating `swift/TurboSparkApp` against
  ZCode v3.14.3 and Unsloth Desktop/Studio, documenting parity, implementation
  routes, and pinned hardware evidence.

### Changed
- `swift/TurboSparkApp`: added `--prompt` and `--seed` options to `ZImageMLXBenchmark`
  for custom prompt and repeatable seed evaluations.
- `.gitignore`: ignored `docs/experiments` and `docs/verification` experiment
  output directories.
- `crates/runtime`: integrated QSA GPU block selection into Qwen4-Exp
  attention, prefill, and sequential decode paths with per-layer buffers,
  buffer barriers, sticky NaN detection, and count-from-buffer dispatches in
  indexed FP16 and TurboQuant Metal kernels.
- `swift/TurboSparkApp`: enhanced ZImage weights mapping and canonicalization
  for mflux-converted models, including conditional NHWC transpose for VAE
  convolutions, module key canonicalization, and packed affine tensor
  restoration.
- `swift/TurboSparkApp`: integrated image style selection into
  `ImageComposerView` and `ImageGenerationSettings`, appending selected style
  prompts non-destructively to user drafts.
- `swift/TurboSparkApp`: migrated app image generation dispatch from the
  legacy native image session to `MLXImageGenerationSession` using the
  vendored ZImage MLX pipeline, with multi-stage progress reporting (prompt
  encoding, diffusion steps, VAE decoding, PNG export), seed preservation,
  cancellation handling, and warm-session caching.
- `docs/IMAGE_GENERATION.md` and `docs/ZIMAGE_TURBO.md`: updated execution
  topology documentation to reflect Swift app dispatch via MLX while keeping
  `crates/image` as the CLI and native Metal reference/benchmark engine.
- `docs/BENCHMARKING.md` and `docs/BENCHMARKS.md`: recorded standalone Swift
  MLX pipeline benchmarks on Apple M4 Max (57.61 s median across 3 runs for
  1024x1024 9-step generation, achieving roughly 63x speedup over the
  3,608 s native packed experiment).
- `docs/QWEN4_EXP.md`: documented QSA GPU selection kernel implementation and
  parity gates.
- `ROADMAP.md`: reconciled completed Priority 1 items and updated active
  disk artifacts.
- `AGENTS.md`: documented Agentic SDLC and Kiro-style Spec-Driven Development
  workflow.

## [0.1.0] - 2026-09-20

### Security
- `swift/TurboSparkApp`: subagent runs go through
  `AppToolPermissionEngine.evaluate` plus
  `TerminalCommandClassifier.isAutoApprovable`. `SubagentRunner` previously
  reached `AppToolRegistry.execute` after checking only a tool NAME list,
  which the built-in `general-purpose` agent leaves empty -- so a subagent
  reached from the `agent` tool or from `/explore` ran `/bin/zsh -c`
  unprompted under a project whose terminal permission was `.ask` or
  `.deny`, on one approval of the agent call itself. An isolated run DENIES
  where the main loop would ask, having no UI to prompt with.
- `swift/TurboSparkApp`: a project-scope agent taking a BUILT-IN's name is
  held to that built-in's tool ceiling (its `disallowedTools` unioned in,
  its `tools` allowlist intersected). A cloned repository shipping
  `.claude/agents/explore.md` with no `disallowedTools` turned `/explore`
  into a write-and-shell agent. Overriding the prompt, description and turn
  budget still works; only widening is refused.
- `swift/TurboSparkApp`: `ProjectRuleDetector` refuses a rules file that
  resolves outside the project. `AGENTS.md -> ~/.aws/credentials` in a
  cloned repository put that file's first 64 KB into every turn's system
  prompt. Symlinks inside the project still resolve.
- `swift/TurboSparkApp`: an approved tool call runs under the project its
  permission decision was computed against, not whichever project is
  selected when the user clicks Approve.

### Fixed
- `crates/ffi`: `ts_session_open` validates `expertCacheSlots` against
  `ALLOWED_CACHE_SLOTS` and returns an error naming the option. It built
  `ExpertCacheSlots::Fixed` from any non-negative integer, and an
  out-of-set value panics inside the expert cache -- fatal in process for a
  GUI host that links the engine. The header had documented `8/16/24/32`
  all along. `swift/TurboSparkApp`'s picker offered four values the engine
  refuses and omitted the legal 24; it now offers exactly the engine's set.
- `swift/TurboSparkApp`: `generating` stays true while a tool runs, so Send
  cannot start a second turn beside it and Stop is enabled for the window a
  shell command runs in; the agent loop's continuation is routed by chat id
  rather than by the current selection; and Stop reaches work spawned from
  the approval card.
- `swift/TurboSparkApp`: the JSON stores quarantine an unreadable file
  instead of letting the next write overwrite it, and report a failed write
  instead of swallowing it. The quit flush moved from a view modifier to
  `applicationWillTerminate` and now stops the server and persists while a
  turn is running.

### Added
- `swift/TurboSparkApp`: user profiles. Each user gets a folder under
  `~/Library/Application Support/TurboSpark/profiles/<id>/` holding its own
  settings, chats, projects, global MCP servers, model favorites, appearance,
  hooks, custom tools, skills, agents, plugins and marketplace installs, all
  routed by redirecting the one `AppStorageRoot` seam plus a new
  `UserProfileStore.userScopeSubdirectory` for the content outside it. The
  Default user is the machine's existing setup rather than a folder: its
  stores stay at the root and its user-scope content stays in the shared
  `~/.turbospark` tree with its cross-agent roots, so nothing migrates.
  Non-default profiles are fully self-contained and read no shared or
  cross-agent tree. No passwords; switching is a save-and-relaunch
  (`AppModel.switchToProfile`), because every store hangs off a cached
  `static let` root. Deletion moves the folder to the Trash. Launch
  overrides: `-TurboSparkProfile <id>` and `TURBOSPARK_PROFILE`.
  `swift/docs/SWIFT_PROFILES.md`.
- `crates/runtime`: chunked prefill for the `qwen4_exp` family
  (`families/qwen4/prefill.rs`), the seventh `ChunkedPrefillRunner` and the
  same "step 1" shape as the other six -- the existing per-token kernels
  looped inside a micro-batch, batching command buffers rather than GEMVs,
  with no new kernel. QSA and GDN needed no change at all. Five buffers were
  widened to `MAX_PREFILL_BATCH` rows and four encoder signatures gained row
  offsets; the fifth (PLE's `ngram_emb`) is the one that mattered, because
  its upload is a HOST `write_buffer_bytes` that does not respect
  command-buffer commit order, so a single-row buffer silently fed every
  token but the last of a micro-batch the wrong n-gram embedding. Verified
  byte-identical to the sequential path on seven synthetic cases and on the
  real REAP-288 install.
- `crates/runtime`: `families/qwen4/prefill.rs` refuses
  `TURBOSPARK_ROUTED_BATCH` and `TURBOSPARK_BATCHED_GEMV` by name, the pair
  every other chunked driver already carried. Neither seam is wired for this
  family, and serving the per-token loop under either arm's label would
  report the unbatched engine as the batched one.
- `crates/bench`: `--prefill-chunk off|auto|N` on `--model` mode, which is
  the only way this binary reaches `run_raw_completion_chunked`. It
  **defaults off**, because every frozen row in that crate was measured on
  the sequential prefill path; the resolved path is echoed in the header on
  both arms, along with `TURBOSPARK_ROUTED_BATCH` and
  `TURBOSPARK_BATCHED_GEMV` (which need no flag -- they are read inside the
  runtime's chunk drivers and go live the moment the driver is engaged, and
  are marked INERT on the sequential arm). An install whose family has no
  chunked driver is refused by name rather than falling back.
  `TURBOSPARK_PREFILL_CHUNK` is likewise refused rather than inherited or
  ignored, the one place this binary deliberately diverges from
  `crates/cli`. Unblocks the prefill energy capture, which could not be
  measured through `scripts/power.sh` at all before this.
- `scripts/power.sh`: a `seq|chunked` arm pair on the prefill axis,
  exclusive of every other arm.
- `crates/runtime`: `TurnSplitter` and `TurnEvent`, one home for the
  structured streaming pipeline that decides whether a turn goes through
  `StructuredAssistantDecoder` and splits its events into content,
  reasoning, and parsed tool calls. The wiring had three drifting copies
  (`crates/cli/src/generate/format.rs::ChannelSplit`, a byte-identical
  `ChannelSplit` in `crates/ffi` documented as ported from it, and the
  inline `needs_decoder` block in `crates/server/src/handler/exec.rs`);
  all three consumers now wrap their own decode loop with the shared
  splitter, behavior-preserving.
- `crates/ffi`: two new `ts_generate` event kinds, `TS_EVENT_TOOL` (a
  parsed call as `{"id","name","arguments"}` JSON) and `TS_EVENT_FINISH`
  (fired once, last, on every successful turn, with the stop reason and
  token counts), plus a `toolCalls` array in `result_json`. Hosts ignore
  unknown kinds, so this is not a version break. No `GenerateOptions`
  field offers tools yet, so `TOOL` cannot fire today; the pipeline is
  wired so growing the binding to offer tools is an options change.
- `swift/TurboSpark`: `GenerationEvent` gains `.toolCall` and `.stopped`
  cases mapped from the new kinds; `TurboSparkApp` consumers updated.
- `docs/STREAMING.md`: the streaming architecture -- the push-callback
  primitive and why the engine has no iterator/async facade, the channel
  adapter and its no-backpressure policy, cancellation semantics, the
  empty-delta and `finish` traps, the wire-format and FFI-event tables.

- `qwen4_exp` (Qwen3.8-Flash-Next) runs its QSA (query-sparse attention)
  indexer above `indexer_budget` instead of refusing context past 2,048
  tokens: every token projects and caches the indexer key and pools newly
  completed 4-token blocks, and above 512 complete blocks the pooled blocks
  are scored, the top 512 plus the ragged tail selected on the host, and a
  new indexed decode-attention kernel (`attention_decode_indexed_partial`,
  the dense kernel walking a position list) attends over them. At or below
  the budget the dispatch stream is byte-identical to before, which the
  frozen quality-gate row reproducing proves. `TURBOSPARK_QSA_FORCE_DENSE=1`
  keeps dense attention above budget as a diagnostic arm. See
  `docs/QWEN4_EXP.md`'s QSA sections.
- Server-side `presence_penalty`, `frequency_penalty`, and `min_p` on
  `POST /v1/chat/completions`, honored end to end by `turbospark-selection`.
  The two penalties follow llama.cpp's convention (the GENERATED suffix of
  history only, never the prompt) rather than OpenAI's whole-context one.
  `POST /v1/completions` accepts `min_p` but not the two penalties, since its
  legacy wire shape has no such fields. See `DEVIATIONS.md`'s sampling-knob
  entry and `crates/selection/CLAUDE.md`.
- `crates/ffi`: `ts_server_start` / `ts_server_stop` / `ts_server_info_json`,
  an in-process HTTP server sharing an already-open `ts_session_open`
  session's engine rather than opening a second one -- serves the same
  OpenAI/Anthropic-compatible routes `turbospark-server` does, minus vision
  and tool-call guardrails. `Session` split into a thin `Arc` handle over a
  `SessionCore` to make the sharing possible; see `crates/ffi/CLAUDE.md`
  Gotcha 13.
- `swift/TurboSpark`: `TurboSparkSession.startServer(options:)` and
  `TurboSparkServer`, the Swift wrapper over the above. The macOS app's
  Engine settings tab gained an "In-Process Server" section (a toggle, the
  bound port once running, and an optional API key); loading a different
  model or unloading stops a running server rather than leaving it pinned to
  a model the UI no longer shows as loaded.
- Prefix KV reuse (cached-prompt continuation): a turn continues from the
  previous turn's KV cache wherever the two prompts agree on their leading
  token ids, instead of resetting and re-prefilling the whole transcript.
  Both prefill loops consult it. Off unless a caller opts in per session
  (`RealForwardRunner::set_prefix_reuse`); `turbospark-check --chat` was the
  first caller and reports `[prefix-reuse] N/M` per turn on stderr (silenced
  by `--quiet` or `TURBOSPARK_PREFIX_REUSE=quiet`). `crates/ffi`'s `open()` now
  opts in unconditionally too, since a `swift/TurboSparkApp` session is
  multi-turn by construction, and `turbospark-server` gained a real
  `--prefix-reuse on|off` flag (default on) paired with a swap-based
  `--session-slots N` pool that fixes the cross-conversation KV-stomping
  hazard a single-runner server has.
  `--prompt` and `--messages-file` are unaffected and their output is
  byte-identical. Measured on a real Gemma 4 install: prefill 1.777s to
  0.153s on a transcript-shaped prompt, with the generated tokens identical
  to the re-prefilled reference. `RawDecodeResult` gains
  `reused_prefix_tokens`.
- `TURBOSPARK_PILOT_PROBE`: diagnostic that records a one-layer-ahead router
  prediction beside the actual expert selection in an `TURBOSPARK_ROUTER_HIST`
  capture (Gemma 4 only), analysed by `scripts/pilot_ceiling.py`. Used to
  measure router-lookahead expert prefetch to a negative result, recorded in
  `docs/EXPERT_ROUTING.md`; `=self` validates the instrument itself.
- macOS app packaging: `scripts/make-app-bundle.sh` assembles
  `swift/TurboSparkApp`'s bare SwiftPM executable into a real
  `TurboSpark.app` (Info.plist, `com.whit3rabbit.turbospark` bundle
  identifier, SwiftPM resource bundles, ad-hoc code signature) with the
  three CLI binaries inside it, and `scripts/make-dmg.sh` wraps that into
  `TurboSpark-<ver>-arm64.dmg`, mounting the image and asserting its
  contents rather than trusting `hdiutil`'s exit code. `make app-bundle`
  and `make dmg` are the local entry points. Note the bundle identifier
  moves the app's `@AppStorage` settings to a new preferences domain, so
  appearance/text-size/language do not carry over from a `swift run`
  build; the JSON stores under `~/Library/Application Support/TurboSpark/`
  are unaffected.
- The DMG is now a release asset beside the CLI zip, built in the same
  `build-macos` job so the CLI binaries in both are the same build.
- A second Homebrew cask, `turbospark-cli`: the command-line tools with no
  desktop app. `brew install --cask whit3rabbit/tap/turbospark` now
  installs `TurboSpark.app` to `/Applications` plus the three commands
  (linked from inside the bundle), and uninstalls both together; the two
  casks declare `conflicts_with` on each other. `--zap` additionally
  trashes the app's settings and chat archive, and deliberately leaves
  `~/.turbospark` model installs alone.
- `package-macos` job in `.github/workflows/ci.yml`: builds the app bundle
  and the DMG on every push to `main`, so a tag push is not the first
  thing to exercise the packaging path.
- Directional steering (`--steering <path.gguf>` on `turbospark-check` and
  `turbospark-server`): runtime abliteration, ActAdd (`add`), feature clamping
  (`clamp`), and norm-preserving projection (`renorm`) via Metal shader
  `steer_direction_fp16`, with support across 7 of 8 families (`qwenGdnDense`,
  `qwenGdnMoe`, `llama`, `qwen3moe`, `gemma4`, `gptoss`, `museGlimmer`),
  speculative verify (`produce_batched`), and Gemma 4 chunked prefill
  (`x_off`). `DeepSeek-V4-Flash` stays refused at open by name -- its
  compressed-attention kernels are unported, so there is no decode flow at
  all to hook. Includes activation capture (`TURBOSPARK_RESID_CAPTURE`) and
  extraction (`scripts/extract_direction.py`).
- `gpt-oss` and `museGlimmer` steering measured on real installs
  (2026-08-25): a self-extracted 4-pair direction and a byte-identical null
  control on each, plus a clean memory-oracle replay. Both share one
  honestly-reported shortfall -- the steering probe's single-position
  divergence check does not clear its (`qwen3_5`-borrowed) floor at the
  prompts tried, despite real per-layer coefficients and CLI-visible
  divergence over a full generation. `gpt-oss`'s write-up traces the
  shortfall to an exact cause rather than a guess: the position that check
  reads decodes to Harmony's near-fixed `<|channel|>` token, so it measures
  the chat template's own determinism, not the direction's absence. Full
  ablation on `gpt-oss` also fails differently than `qwen38-27b`'s collapse
  at the same operating point -- an unresolved reasoning loop past ~900
  tokens rather than an immediate stop. Full write-up:
  `docs/OBLITERATION.md`'s "museGlimmer, measured on a real install" and
  "gpt-oss, measured on a real install" sections.
- `ChatDialect::Llama3` in `turbospark-tokenizer`: detection on
  `<|start_header_id|>` / `<|eot_id|>` and fallback template renderer.
- `turbospark-catalog`: the curated model table (17 rows, each naming a
  repository and revision that were streamed and run on real hardware), the
  header-only Hugging Face probe, the install driver, and the `~/.turbospark`
  store. Nothing in it decodes, so it builds on every platform.
- `turbospark-model`, a second binary on `turbospark-cli`: `list`, `info`,
  `probe`, `pull`, `path`, `rm`. `probe` reads KB off a repository and reports
  whether a checkpoint would run before any of it is downloaded; `pull` fetches
  and verifies the tokenizer sidecars before a byte of weight data moves.
- `--model` accepts a catalog alias as well as a path, on both
  `turbospark-check` and `turbospark-server`. An existing directory always
  wins, so nothing that previously worked changes; the server prints which
  directory an alias resolved to at startup, since it is the one that runs
  unattended.
- Documentation: `docs/OBLITERATION.md` (runtime steering design, refutations,
  measurements, and interop), `docs/MODELS.md` (the catalog, the probe, adding a
  row), and `docs/RELEASE.md` (how a release is cut).
- `scripts/mlx_qmm_reference.py`: MLX's own `mx.quantized_matmul` `c(M)` at this
  port's seven INT4 GEMM shapes, on the same machine and the same yardstick as
  `crates/gpu`'s bench. It replaces an extrapolation with a measurement, and the
  saturation point moved: mlx's curve is FLAT from M=32 to M=512, not M=36 or
  M=64, and its `ms` column is identical at M=16 and M=32, which is a `BM=32`
  tile paying for empty rows. Numbers in `docs/BENCHMARKS.md`.
- `crates/gpu/tests/gdn_prefill_share_bench.rs`: prices `gdn_delta_step_prefill`
  against the INT4 matrices one prefill micro-batch walks, weighted by the real
  layer counts. **4.66%**, an upper bound, which closes the gated-DeltaNet
  threadgroup-staging idea. It exists because `TURBOSPARK_DISPATCH_PROFILE=1`
  waits on every command buffer at commit and did not finish a 150-token
  prefill in 12 minutes; this answers the same question in 0.45 s.
- `FC_MMA_STAGE_X` (function constant 110) on `dequant_int4_gemm_mma`, with
  `encode_dequant_int4_gemm_mma_resident_staged`: stages `x` through
  threadgroup memory instead of `simdgroup_load`ing it transposed from device.
  Bit-identical to the un-staged arm and **a measured 3.3x to 5.9x LOSS at
  every width**, kept behind a default-off constant so the negative is
  reproducible rather than a note. Nothing dispatches it.
- `FC_MMA_SKIP_DEQUANT` (function constant 111) and
  `encode_dequant_int4_gemm_mma_resident_skip_dequant`: a DIAGNOSTIC that
  fills the matrix kernel's weight tile with a constant, so its time is the
  matrix path with the unpack deleted. Output is meaningless and a test
  asserts it. It settles where that kernel's cost lives: the dequant is 36%
  at M=2 and 11% at M=64, and **with the dequant entirely free the kernel
  still reads 0.46 to 0.50 past M=16 against MLX's 0.145** -- so the loader
  is not the lever, and neither is `kMmaTile`.
- MXFP4 phase-2 down-reduce (`gpt-oss`) is specialized by `top_k`:
  `moe_phase2_down_reduce_k8_mxfp4` now masks compute for unused routed
  slots instead of always reducing all 8, saving half the down-GEMV on
  `gpt-oss`'s top-4-of-32 routing. Bit-identical to the unspecialized
  kernel; measured ~1.13x end-to-end decode throughput on a real
  `gpt-oss-20b` install.
- `qwen4_exp` (Qwen3.8-Flash-Next) intake: the family and config surface,
  an n-gram table on-disk format and streaming writer
  (`model_io::ngram_table`, `crates/repack`), and classification of that
  table out of the resident expert index. No decode flow yet; see
  `docs/QWEN4_PHASE0.md` and `ROADMAP.md` section 2.

### Changed
- The release workflow requires a `CHANGELOG.md` entry for the tag, publishes
  crates one at a time in dependency order so a re-run of a partially published
  tag resumes correctly, and ships `turbospark-model` in the macOS archive.
- `docs/NEW_MODEL.md` gained a phase for making a new family reachable from the
  CLI: the probe and install driver's family match sites, and the catalog row.

