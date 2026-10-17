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

### Added
- ArkTTS multilingual zero-shot TTS preview family port (`crates/audio/src/tts/arktts/`).
  DualAR transformer architecture plus bundled 44.1 kHz decode-only speech codec:
  pure-attention slow stack on 0.6b and hybrid Mamba-2 + attention Falcon H1
  backbone (`crates/audio/src/lm/falcon_h1.rs`) on 0.1b, fast AR residual codec
  loop, reference-exact legacy top-k/top-p and RAS repetition rescue sampling,
  and offline fixture parity against mlx-audio 0.5.8 (`e1b19b90`).
- Routed-expert cache slot tiers 192 and 256 added to `ALLOWED_CACHE_SLOTS`
  in `turbospark-core` (`runtime_config.rs`) and `turbospark-model-io`
  (`expert_cache_policy.rs`). The `auto` slot tiering policy resolves to 192
  when free memory is at least 64 GiB and 256 when free memory is at least
  128 GiB (one quarter of free memory covering slot memory). Validated on the
  512-expert `qwen4exp` install streamed from external SSD storage.
- `docs/EXPERT_ROUTING.md`: Phase 0 and Phase 1 measurements for external SSD
  streaming of the 512-expert `qwen4exp` model over 10 Gb/s USB, documenting
  page-cache double-counting vs slot cache hits, `F_NOCACHE` baseline device
  rates (~627 MB/s), and GPU command buffer completion wait bounds.
- `swift/TurboSparkApp`: In-memory Qwen3-TTS session residency in
  `AudioWorkspaceController` (`residentQwenSession(for:)`), caching the loaded
  MLX model graph across repeated speech jobs to eliminate re-initialization
  overhead while honoring explicit Load and Unload actions.
- OmniVoice multilingual zero-shot TTS with voice cloning, end to end with Native Metal acceleration.
  `crates/audio/src/tts/omnivoice/` completes its fixture-verified port with
  the intake layers the released checkpoints need: the rowwise int8 scheme
  (`w = q * scale` over signed int8 groups with f16 scales) parsed from
  `mlx_manifest.json`, the released k2-fsa layout (`llm.` prefix, fused audio
  embedding/head tables split per codebook), the Higgs tokenizer's
  Transformers-format loader (PyTorch conv layouts, `codebook.embed`, snake
  `[1, C, 1]`), the Wav2Vec2 semantic tower behind the encode path's
  `output_hidden_states` mean, the torchaudio-compatible sinc resampler, and
  the rule-based duration estimator with Unicode category tables generated
  from the pinned CPython. Metal acceleration in Rust (`crates/audio/src/lm/execution.rs`
  and `crates/runtime/src/omnivoice.rs`) accelerates the bidirectional Qwen3
  backbone projections, bidirectional multi-head attention, MLP blocks, and
  8 prediction heads via Metal GPU device execution (`Music3Device`) with
  transparent CPU fallback. Catalog rows `omnivoice-8bit` (default) and
  `omnivoice-bf16` (`mlx-community/OmniVoice-8bit` at `b222e4d8`,
  `OmniVoice-bfloat16` at `6119f707`) install through the app with the Boson
  Higgs tokenizer's community-license terms and are promoted to `backend: "metal"`,
  `can_run: true`. Runtime engine (family `omnivoice`, Metal by default with
  portable CPU fallback, consent-gated cloning, per-step progress and cancellation),
  HTTP `/v1/audio/speech` with optional `reference_audio`, HTTP model management
  (`GET /v1/audio/models`, `POST /v1/audio/models`, `GET /v1/models`), and the
  macOS app's Voice Cloning page (`clonesNatively`) are fully wired and verified.
  Gates: cross-engine encode parity exact against the Python reference on
  bfloat16 (70 frames x 8 codebooks); 8-bit profile deterministic and verified;
  catalog, runtime, FFI, server, and Swift suites green.
- Chunked prefill for the qwen MoE flow (`qwenGdnMoe`: Qwen 3.6 35B-A3B,
  Ornith 35B, MiniCPM-V 4.7), `crates/runtime/src/families/qwen/moe_prefill.rs`.
  The attention and router half of up to 16 tokens runs before one wait per
  layer, then the routed experts token by token. Output is byte-identical to
  sequential prefill, checked on synthetic fixtures for both routed layouts
  and on the real MiniCPM install. A 2,940-token prompt on MiniCPM prefills
  1.21-1.31x faster (107-120 s against 141-145 s on M4 Max). On by default in
  `turbospark-check` and `turbospark-server`. `TURBOSPARK_QWEN_MOE_CHUNKED=0`
  turns it off.
- MLX `mode: "mxfp4"` safetensors intake, first carrier
  `mlx-community/MiniCPM-V-4.7-35B-A3B-mxfp4` (catalog alias
  `minicpm-v47-35b-mxfp4`, text-only, `verified`). Routed experts are
  re-blocked losslessly into GGUF MXFP4 and run on the MXFP4 MoE pair, whose
  phase-1 activation is now chosen by family (plain SiLU here, `swiglu_oai`
  for `gpt-oss`). Other mxfp4 matrices are dequantized exactly to BF16. Adds
  a BF16 embedding lookup kernel, per-tensor group sizes for an mxfp4
  checkpoint's affine gates, a raw-BF16 manifest slot, and
  `has_foreign_vision_tower` for VLM wrappers whose tower is not this port's.
  Frozen gates: perplexity 7.5693, 1,491 MiB peak at 4,096 context and 16
  slots, 24-26 tok/s on M4 Max.
- `crates/embeddings` (`turbospark-embeddings`): portable CPU embedding
  models ported from the mlxrs `embeddings` module and checked against
  `transformers` 4.57.1 and `sentence-transformers` 5.1.2. EmbeddingGemma
  (`google/embeddinggemma-300m` plus its mlx-community F16 and affine
  4/8-bit conversions), SigLIP2 NaFlex image and text, and CLAP
  HTSAT-unfused audio and text. Snowflake arctic-embed text encoders:
  xs, s, m, m-v1.5, l, and l-v2.0 (BERT and XLM-RoBERTa) with the
  mlx-community l-v2.0 bf16 and 8/6/4-bit conversions, m-v2.0 (GTE), and
  m-long (Nomic BERT). Every checkpoint is pinned to a commit in
  `crates/embeddings/MODELS.md`. Also shared sentence-transformers
  pooling, Matryoshka truncation, cosine and ColVision MaxSim scoring.
- `crates/coreml` (`turbospark-coreml`): Apple Core ML runner via
  `objc2-core-ml` (compile cache, stateful and multifunction models, macOS 15
  APIs gated at runtime), with pipelines in `crates/runtime` for three
  FluidInference exports (macOS 15+). See `docs/COREML.md`.
  - Parakeet Redux speech-to-text (`parakeet-redux-coreml`): native audio
    sessions, `--stt-model`, FFI STT sessions, and the
    `parakeet-redux-coreml` audio catalog profile. Token-identical to the CPU
    Redux decoder on the gate clips.
  - EmbeddingGemma 2 text, image and audio embeddings
    (`embeddinggemma-2-coreml`) with task prompts and Matryoshka sizes:
    `/v1/embeddings`, `ts_embedding_encode_json`, the new
    `ts_embedding_encode_request_json`, and Swift
    `TurboSparkEmbedding.encode(request:)`.
  - Unlimited OCR (the Core ML parts of `unlimited-ocr-mlx-coreml`):
    `turbospark ocr`, the `ts_ocr_*` C API, and Swift `TurboSparkOcr`.
  - Pinned EmbeddingGemma 2 install: `embedding_models.json` lists every
    file by size and SHA-256, `ts_embedding_catalog_json` and
    `ts_embedding_install` download and verify it into
    `models/embeddings/<alias>`, and the alias resolves anywhere an
    embedding model path is accepted, but only to an install of the
    current pin. `ts_embedding_install` takes the revision the host showed
    and refuses any other; a download replaces an older pin's install. The
    profile carries `status: verified` with the gates behind it.
  - Pinned Unlimited OCR install (`unlimited-ocr-coreml`, the nine Core ML
    files of `4568571e`, 4.09 GB): `ocr_models.json`, `ts_ocr_catalog_json`,
    `ts_ocr_install`, Swift `OcrCatalog`, and `ts_ocr_open` by alias. No app
    UI yet. Network gates: `pinned_coreml_ocr_install_network` (installs and
    reads a page) and `pinned_dir_catalog_network` (checks every embedding
    and OCR pin against the Hub).
  - Benchmark probes and drivers for the Core ML pipelines and for
    EmbeddingGemma 300m on MLX vs Core ML vs the CPU port
    (`crates/runtime/scripts/benchmark_coreml.py`,
    `benchmark_embeddinggemma3.py`); readings in `docs/COREML.md`.
  - `turbospark_coreml::compute_plan` and the `coreml_plan` example: the
    device Core ML plans for each op (`MLComputePlan`, macOS 14.4+), to
    check whether an export really reaches the Neural Engine.
- `crates/audio`: VoxCPM (MiniCPM backbone, LongRoPE factor tables, CFG
  zero-star flow DiT, causal-conv AudioVAE) and VoxCPM2 (fusion-concat
  conditioning, two-token mu DiT, reference/continuation prompts, and
  SR-conditioned 48 kHz AudioVAE) TTS family ports with catalog assets.
- `swift/TurboSparkApp`: an Embeddings tab in the API workspace. It picks
  the server's embedding model (saved, and attached again after every start),
  attaches and detaches it live without blocking the UI on a first Core ML
  load, lists served embedding models apart from chat models
  (`ServerInfo.embeddingModels`), and sends a test `POST /v1/embeddings`. The
  same picker replaces the Memory embedding field: pinned profiles show
  Verified only when their gates ran on that pin, other paths are checked
  with the new `ts_embedding_probe_json` and shown as Not verified, or
  refused with the reason when the family does not run here.
- `swift/TurboSparkApp`: bundled open-source fonts for code and UI rendering
  with license notices and Settings catalog entries: Iosevka, Geist, Geist Mono,
  Cascadia Code, Hack, and Source Code Pro.
- `swift/TurboSparkApp`: conversation recall on syntext's context engine
  (`SyntextContext`). Each chat's user messages and prose answers are
  journaled to a per-profile store (`context-recall/`, mode 0700, excluded
  from backups). Once turns leave the prompt through compaction or the window
  fit, the ones relevant to the newest message come back as cited
  `<evidence>` blocks on that message, within at most 1,024 tokens. On by
  default (Settings > General > Context Compaction). Off for Ghost chats and
  passphrase-protected profiles, because the store is plaintext. Deleting or
  clearing a chat forgets its turns. See `swift/docs/SYNTEXT_CONTEXT.md`,
  which also sets it against memory.

### Changed
- `swift/TurboSparkApp`: syntext moves to 2.7.1 (`exact: "2.7.1"`) and adds
  the `SyntextContext` product. The previous `exact: "2.5.0"` pin linked the
  2.4.0 xcframework, because syntext's `vX.Y.Z` tags through 2.7.0 pin the
  previous release's binary, so `grep_search`'s `literal_search` and context
  lines silently did nothing. 2.7.1 is the first syntext tag that pins its
  own binary, and its blake3 objects no longer target macOS 14.5 against the
  app's 14.0 floor.
- `swift/TurboSparkApp`: the project code index moved to
  `~/Library/Caches/TurboSpark/syntext/<digest>.syntext`. The old base64 name
  failed for project paths over about 185 bytes. Existing indexes rebuild
  once, and the old folder is removed. A first search no longer waits for a
  build (it uses the unindexed scan meanwhile), builds run one at a time
  across projects, and the selected project's index opens at launch.
- Embedding models are checked by family before they load, in the standalone
  C ABI, the server attach and `turbospark-server --embedding-model`:
  `EncoderRunner` accepts only `model_type` `bert` or `xlm-roberta` with CLS
  pooling (a `1_Pooling/config.json` asking for mean or other pooling is
  refused), so a GTE, Nomic BERT, EmbeddingGemma 300m or mean-pooled BERT
  directory no longer runs and returns wrong vectors. A missing `model_type`
  is refused rather than taken as BERT. The server pane's Advanced settings
  no longer has an embedding field; the API Embeddings tab replaces it.
- OpenKind (the TypeSafe daemon and Swift package) moves from 0.1.0 to
  0.3.0 (`v0.3.0`, `2dba87572c028aeaf34b27220a3bbd3d8d8dbc76`) in
  `make-app-bundle.sh`, `ci.yml`, `release.yml` and `docs/RELEASE.md`.
  `make-app-bundle.sh` now points mlx-sys's bindgen at the Xcode SDK, which
  the daemon build needs under Xcode 27.
- `swift/TurboSparkApp`: the Server pane's API key follows the same rule as
  the TypeSafe key. It is no longer saved to the Keychain, or read back at
  launch and on every settings save, unless **Remember this key in the
  Keychain** is on. A new install keeps a typed key for the session only. A
  key an earlier version already stored is kept: the first launch finds the
  item without reading the secret and turns the option on, so nobody's server
  quietly starts without authentication. Turning it off deletes the item.
- `swift/TurboSparkApp`: the icon rail is always on screen, and beside it
  the selected section's panel: the chat list in Chat, every Audio page in
  Audio. The labeled section rows that sat above the chat list are gone,
  which returns their height to chat history. Other sections show the rail
  alone. The profile menu moved to the foot of the rail and the temporary
  chat button beside the Chat/Projects tabs. View > Toggle Sidebar (was
  Toggle Chat Sidebar) hides the panel, never the rail, and is disabled
  where there is no panel. The window minimum with a panel shown is 953pt
  (was 901).
- The qwen MoE chunked prefill (`qwenGdnMoe`) batches its resident weight
  reads by default on INT4-affine installs (Qwen 3.6, Ornith 35B): the GDN
  and attention projections and the shared expert run as M-row GEMMs on
  Steel at a 64-row micro-batch, and the routed experts keep the per-token
  loop. A 2,940-token prompt on Qwen 3.6 prefills 1.42-1.56x faster than the
  per-token chunked arm (53.7-58.5 s against 83.4-83.9 s on M4 Max, two pairs
  on a shared host). The output is no longer byte-identical to sequential
  prefill: top-8 routing near-ties flip under the batched arm's reduction
  order (mean final-position KL 3.9e-3 nats against the per-token arm). The
  text stays coherent. `TURBOSPARK_BATCHED_GEMV=0` restores the per-token arm,
  which still matches sequential prefill exactly. MiniCPM-V 4.7 (BF16
  resident matrices) keeps the per-token arm. Ornith 35B, checked on a real
  install, passes the same probe at 6.7e-3 to 8.2e-3 nats with its frozen
  quality and memory gates reproduced.
- `swift/TurboSparkApp`: Voice Cloning filters candidate voice cloning models to
  those with native Metal or MLX acceleration (`clonesNatively`). Portable CPU-only
  models such as OmniVoice are not presented as voice clone engines until a
  Metal or MLX offload is implemented.
- `swift/TurboSparkApp`: `AudioModelPickerView` now offers explicit Load and
  Unload actions and a Loaded badge matching the model action bars across other
  workspaces, with loading spinners during model verification. Resident audio
  models are wired into `AppModel`'s `canUnloadAnyModel` and top-bar unload menu.

### Fixed
- `crates/catalog`: `InstalledModel::effective_modality` now derives the `Audio`
  modality from install path, `audio-receipt.json`, or speech metadata if a model
  was recorded with default `text` modality in `installed.json`. `installed_report`
  also discovers verified `.gturbo` audio models in `models/audio/` that were
  missing from the manifest. This fixes an issue where already-installed audio
  models offered a spurious download button that failed with "audio install already exists".
- `swift/TurboSparkApp`: `AudioModelPickerView` suppresses download and retry
  buttons when an audio model is already installed.
- `swift/TurboSparkApp`: `ToolRiskClassifier.isSensitivePath` now also flags
  `.pem` and `.key` file extensions and `.npmrc` / `.netrc` credential files,
  preventing sensitive file exposure across the agent permission engine, attachment
  picker, `@` mention completion, and project tree view.
- `swift/TurboSparkApp`: polished prompt composer editor layout, model download
  state progress, and model management cards.
- `crates/audio` nemo alignment: `merge_longest_contiguous` and
  `merge_longest_common_subsequence` dropped every token of the second chunk
  past the overlap window. They now keep the rest of the chunk, as upstream
  mlx-audio does.
- `swift/TurboSparkApp` `grep_search`: once syntext refused a commit as
  overlay-full, the error was swallowed and every file created afterwards
  stayed unsearchable until a manual Reindex. It now rebuilds in the
  background. `multiedit`, `notebook_edit`, and patch deletes now update the
  index, and a search sees a write notified just before it.
- `swift/TurboSparkApp` `grep_search`: a `path_filter` no longer drops the
  `path` scope, an escaping or absolute `path` is an error instead of a
  whole-repository search, and results are capped at 500 matches and 10
  context lines.

### Security
- `swift/TurboSparkApp` `grep_search`, indexed and unindexed, no longer
  returns hidden files, `.git/` contents (`.git/config` can hold credentials
  in remote URLs), `node_modules`, `target`, `.build`, or credential files
  (`.env*`, `*.pem`, `*.key`, `credentials`, `.npmrc`, `.netrc`). Search is
  auto-allowed, so on an opted-in project these were readable without a
  prompt. A sensitive `path` argument is now classified high risk.

## [0.3.2] - 2026-10-13

This release stops the app asking for Keychain access by default. Upgrade with
`brew upgrade --cask whit3rabbit/tap/turbospark`.

### Changed
- `swift/TurboSparkApp`: the TypeSafe API key is no longer written to the
  Keychain by default. It used to be generated and stored on first launch,
  which asked for Keychain access again after every update (an ad-hoc signed
  app has no stable identity for the old item to trust) and left the TypeSafe
  pane unable to start when no default keychain was available. The key is now
  generated fresh per launch and kept in memory, and the Keychain is not
  touched. **Remember this key in the Keychain** in the TypeSafe tab turns
  storage on. Turning it off deletes the item. A Keychain failure now shows the
  system's own reason and code. An item an earlier build created is left alone
  and unused: remove it with
  `security delete-generic-password -s TurboSpark.typesafe`.

## [0.3.1] - 2026-10-13

This release replaces v0.3.0, whose app crashed on launch on every Mac but the
build machine. Upgrade with `brew upgrade --cask whit3rabbit/tap/turbospark`.

### Fixed
- `swift/TurboSparkApp`: the app crashed on launch on every Mac that was not
  the build machine ("could not load resource bundle"), in v0.2.0 and v0.3.0.
  The CI toolchain (Xcode 16.4) generates a SwiftPM resource accessor that
  only looks at the `.app` root and a build-machine path, while the bundle
  must sit in `Contents/Resources` (`codesign` rejects anything else at the
  root, symlinks included). The resources now live in their own
  `TurboSparkAppResources` target whose hand-written `AppResources.bundle`
  looks in `Bundle.main.resourceURL` first. `Bundle.module` in the app
  forwards to it, so no call site changed. The resource bundle is now named
  `TurboSparkApp_TurboSparkAppResources.bundle`.
- The app icon showed a gray box on macOS 26. A light drop-shadow rim from the
  source art survived in the icon, and macOS 26 answers an icon like that by
  shrinking it into a system plate. `scripts/generate_app_icons.py` now cuts
  the artwork to a clean 824px squircle.
- README: the Homebrew install section now says how to get past the first-launch
  block, and no longer recommends `brew install --no-quarantine`, which
  Homebrew has removed.

### Changed
- CI: the release build moves from `macos-15` / Xcode 16.4 to `macos-26` /
  Xcode 26.6, selected by path and asserted. `package-macos` builds with both
  Xcode 26.6 and 16.4.
- CI: new `scripts/verify-app-bundle.sh` launches the built app for ten seconds
  and checks every Mach-O's deployment target against the plist's
  `LSMinimumSystemVersion`. It runs in both `ci.yml` and `release.yml` before
  the DMG is built, so a bundle that cannot start no longer ships.
- The generated Homebrew casks use `depends_on macos: :sonoma` (the former
  `">= :sonoma"` form is deprecated in Homebrew).

### Security
- `Cargo.lock`: `rustls` 0.23.43 to 0.23.45 and `rustls-webpki` 0.103.13 to
  0.103.15. The rustls bump fixes a TLS 1.3 flaw where handshake messages
  were accepted across encryption level boundaries (Dependabot alert 1). The
  only consumer is the `reqwest` client in `crates/catalog`, so the exposure
  is outbound model and catalog downloads.
- `scripts/z_image_reference_requirements.txt`: `urllib3` 2.7.0 to 2.8.0.
  Fixes HTTPS proxy TLS settings being ignored or overridden, unbounded
  buffering of a chunked-encoding size line, and an infinite loop on
  chunked deflate streams (Dependabot alerts 2, 3, and 4). The pin is used
  only by the Z-Image reference script, not by the shipped app.

## [0.3.0] - 2026-10-12

### Security
- `crates/ffi`: request-triggered model loading in the embedded HTTP server
  is now opt-in. By default a request that names a model nobody attached is
  refused without consulting the installed store, exactly as the server
  behaved before the installed-model loader existed; a host that wants the
  load-on-demand behavior sets `allowRequestModelLoad` in
  `ts_server_start` options. This closes a memory-exhaustion path where an
  unauthenticated loopback client (or a tailnet client holding the shared
  inference key) could open every installed model by name, with no
  inference required and no bound on cumulative residency.
- `crates/ffi`: a resident-model policy now bounds what the embedded server
  holds regardless of who attaches. `maxAttachedModels` (default 8) caps
  simultaneously attached models; `maxAttachedModelBytes` optionally caps
  the cumulative committed-weights-plus-KV footprint. Request-opened models
  are marked on-demand and are released by the default idle sweep after 30
  idle minutes, so an opted-in server's residency is self-healing; models
  the host attached are never timed out. Byte ceilings are opt-in because
  streamed or mapped expert residency deliberately keeps most weight bytes
  unresident.
- `crates/server`: `POST /v1/completions` validates cheap request fields
  (`suffix`, `n`, sampling config, prompt shape) before resolving the
  model, so a request the endpoint was always going to refuse can no longer
  cause a heavyweight model open on a loading-enabled server before
  returning 400.
- `crates/ffi`, `crates/server`: documented that the bounded-resolution
  executor and single-flight load coalescing limit concurrent work, not
  cumulative residency; the residency policy is the bound.
- `swift/TurboSparkApp`: a screen lock, sleep, or session deactivation can no
  longer be undone by a profile unlock that was still authenticating when it
  fired. Passphrase derivation and the Keychain prompt deliberately run
  outside the vault store lock, but nothing invalidated an authentication
  that had started before `lockVault` returned: the stale completion
  reopened the encrypted database and the coordinator returned to the
  unlocked state with a fresh AppModel. The vault now carries a lock
  generation that `lockVault` bumps before clearing the session and every
  unlock revalidates before installing its session, wiping the derived key
  buffers when the install is rejected as stale; the coordinator also
  invalidates unlock attempts on lock, so a superseded completion leaves the
  vault cleanly locked instead of unlocking or erroring over the lock.

### Changed
- `crates/runtime`: Qwen3.8-27B (and the other `qwen35` dense installs)
  decode with the GQA-shared attention partial by default. Long-context
  decode measured 1.24x at 5.8K tokens and 1.76x at 18K with greedy output
  unchanged in those runs. The reduction order moved three frozen quality
  rows at near-ties, and they are re-frozen with the old values recorded:
  `qwen38` 4.9432 -> 4.9444, `ternary27b` 6.8350 -> 6.8351, `bonsai2`
  5.8134 -> 5.8141. `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` restores the old
  kernel and the old rows.
- `crates/runtime`, `crates/gpu`: MTP speculative decoding skips the
  full-vocab lm_head on each round's row-only draft step, and the
  gated-DeltaNet rollback snapshot is a GPU blit into pooled buffers instead
  of a 147 MiB host copy (3.7 ms to 0.06 ms per round on the host). Block-2
  MTP on Qwen3.8 measured 29.65 -> 32.16 tok/s with output byte-identical.
- `crates/runtime`: dense qwen chunked prefill takes the batched arm by
  default wherever the install qualifies (INT4, unfolded, text-only, no
  `--kv-bits`), with its INT4 GEMMs on the vendored Steel kernel at a
  64-row micro-batch. A 2,727-token Qwen3.8 prompt prefills in 21.0 s,
  against 46.6 s for the scalar batched arm and 138.6 s per token, with
  greedy output identical across all three. `TURBOSPARK_BATCHED_GEMV=0`
  keeps the per-token arm and `TURBOSPARK_PREFILL_STEEL=0` the scalar
  kernel.
- `crates/runtime`: an MTP install prefills through the chunked driver,
  priming the MTP head after each micro-batch instead of running a full
  forward pass per prompt token. 2,727 tokens: 24.6 s, down from 181.7 s,
  output identical. `TURBOSPARK_MTP_CHUNKED_PREFILL=0` restores the token
  walk. DFlash2 still prefills one token at a time.
- `crates/runtime`: on the batched prefill arm the MTP head is primed one
  micro-batch per command buffer through the trunk's M-row encoders instead
  of one prompt row per command buffer. A 2,727-token prompt prefills
  1.0-3.3 s faster in interleaved pairs, and the head now adds about 0.6 s
  over the headless prefill. Output and accepted tokens per round are
  unchanged. `TURBOSPARK_MTP_BATCHED_PRIME=0` restores per-row priming.
- `crates/gpu`, `crates/runtime`: dense qwen chunked prefill runs each
  micro-batch's attention through a multi-row GQA partial in one dispatch
  instead of the one-row kernel per row: 1.8-4.2x on the kernel at 64 rows,
  and an 18K-token Qwen3.8 prompt prefills in 156 s against 176 s, with
  output identical. Prefill only, held to the prefill KL bound (1.89e-5
  nats mean); decode and speculative verify keep the one-row kernel.
  `TURBOSPARK_PREFILL_ROWS_ATTN=0` restores it.
- `crates/catalog`: Qwen3.8-27B-MTP is promoted to verified status with
  measured Apple M4 Max performance (21.9-28.7 tok/s decode, 1.44x end to
  end), passing `qwen38_quality_gate`, `qwen38_memory_oracle`, and
  `mtp_generation_gate`.
- `crates/ffi`: with nothing attached, the embedded HTTP server now answers
  503 for a named model unless `allowRequestModelLoad` is set (previously
  the loader answered 404 with suggestions). Hosts attaching models
  explicitly see no change.
- `swift/TurboSparkApp`: browser automation authorizes subframe navigations
  like main frames. An iframe, an iframe-targeted form or a script-driven
  frame navigation to a loopback, LAN, metadata or otherwise unapproved
  origin is now denied at the WebKit navigation boundary instead of loading
  unchecked, and a frame's post-redirect response origin is revalidated, so
  an approved outer page can no longer reach private services through a
  subframe. `browser_screenshot` refuses to capture a page that embeds frame
  content from an origin that never received a grant, keeping those pixels
  out of model-visible tool results.
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
- `crates/gpu`: `attention_decode_gqa_partial`, a one-row decode attention
  partial that loads each K/V row once for every Q head of its group and
  runs positions across simdgroups with no per-position barrier. 3-13x the
  barrier kernel from 512 to 32K positions at Qwen3.8's shape. Opt-in per
  `AttentionScratch` (`with_decode_kernel`).
- `crates/gpu`: `encode_dequant_int4_gemm_steel`, Apple MLX's Steel quantized
  GEMM vendored from MLX v0.31.1 (MIT, `NOTICE`) and adapted to the resident
  INT4 layout by `scripts/vendor_mlx_steel_qmm.py`. 2.5-3.1x the scalar
  batched INT4 kernel at 32-128 rows. Prefill only, since it is not
  bit-exact against the decode kernels.
- `crates/bench`: `qwen38_prefill_arm_probe`, which holds the default
  prefill arm to a KL bound against the per-token arm on the real install
  (2.0e-5 nats mean measured).
- `crates/audio`: added MegaTTS3 zero-shot voice cloning pipeline (latent diffusion
  transformer over WavVAE latents, prompt aligner, duration LM, relative transformer,
  text G2P frontend subset, and `MegaTts3` facade) with full test fixtures.
- `crates/audio`: added TTS model families: KittenTTS (`kitten_tts`), KugelAudio
  (`kugelaudio`), MOSS-TTS (`moss_tts`), Spark-TTS (`spark`), and Zonos-v0.2 (`zonos2`).
- `crates/audio`: added neural audio codecs: WavVAE (`wavvae`) and Spark BiCodec
  (`spark_bicodec`), and refactored the Vocos backbone loader.
- `crates/cli`: added `turbospark-clone` binary for zero-shot voice cloning from a
  reference recording and English text prompt using MegaTTS3.
- `crates/runtime`, `crates/ffi`, `crates/server`, `swift/TurboSparkApp`: MegaTTS3 voice
  cloning end to end. The native audio session opens family `megatts3` and takes the
  reference voice as the job's PCM; `POST /v1/audio/speech` accepts a base64 WAV
  `reference_audio` with `consent`, `steps`, `seed`, `p_w` and `t_w`; the app's Voice
  Clone page follows the catalog's `voice_clone` capability. Every cloning request
  requires consent, and catalog profiles can list install terms that
  `ts_audio_install_with_options` must acknowledge before any download.
- `crates/audio`, `crates/runtime`, `crates/cli`, `swift/TurboSparkApp`: MegaTTS3
  runs on Metal by default on macOS (the diffusion transformer and the WavVAE
  decoder on the GPU), about five times faster than the CPU path, which now runs
  only as the portable opt-in's fallback or with `turbospark-clone --cpu`. Metal
  matched the CPU path across 8 speakers and two multi-chunk passages.
- `crates/audio`, `crates/cli`, `crates/server`, `swift/TurboSparkApp`: MegaTTS3
  quality presets. Steps default to 16 (Balanced), with 8 (Fast) and 32 (Best)
  as the alternatives: the app's Quality picker, `turbospark-clone --quality`,
  and the HTTP `steps` default. A step study over 8 speakers found no
  intelligibility or speaker-similarity loss from 8 steps up.
- `crates/runtime`, `crates/audio`: `megatts3_bench` and
  `scripts/benchmark_megatts3.py` time cloning requests and compare each peak
  footprint with what runtime admission checked.
- `crates/audio`: `convert_megatts3.py` writes a root `config.json`, `LICENSE`,
  `NOTICE.md` and a model card, so its output can be published as is.
- `crates/catalog`: catalog row `megatts3-f32`, the converter output pinned at
  `cowWhySo/MegaTTS3-TurboSpark` revision `a55e74f5`. Installing it requires
  accepting the `megatts3_community_encoder` term.
- `swift/TurboSparkApp`: added Humanize advanced controls drawer and panel
  (`HumanizeAdvancedPanel.swift`, `HumanizeChrome.swift`), custom word replacements,
  avoid-words highlighting, protected terms integrity verification (`HumanizeRules.swift`),
  and multi-segment draft splitting for long passages.
- `crates/server`, `crates/ffi`, `swift/TurboSpark`, `swift/TurboSparkApp`: Voice cloning
  support on `POST /v1/audio/speech` with optional fields for `reference_audio`
  (base64-encoded 24 kHz mono WAV/PCM), `reference_text`, `instruction`, and `language`.
- `crates/ffi`, `crates/server`: In-process speech bridge (`SpeechBridge`, `BridgedAudioProvider`)
  allowing HTTP speech requests to route into host Swift MLX models (such as Qwen3-TTS)
  via C ABI entry points (`ts_server_attach_audio_speech_model`, `ts_server_detach_audio_speech_model`,
  `ts_server_poll_audio_speech_events_json`, `ts_server_complete_audio_speech_request`).
- `swift/TurboSpark`: Swift bindings for voice cloning (`VoiceCloneRequest`) and server speech
  events (`ServerSpeechEvent`, `ServerSpeechRequest`) for bridging synthesis requests.
- `swift/TurboSparkApp`: Audio API workspace destination (`AudioAPIPaneView`) in the server pane
  providing attached model controls, curl examples with voice cloning parameters, an interactive
  test form with reference audio recording/playback, and automated background event dispatch
  to MLX Qwen3-TTS.
- `crates/catalog`, `crates/image`, `crates/ffi`, Swift app: Qwen-Image-2.1-Turbo
  (`qwen-image-2.1-turbo-mlx-mixed-4bit`, pinned `Sawfwair` export, 8 steps,
  Qwen Research License, non-commercial) is a catalog row and a pinned model in
  the app, generating through the vendored Swift MLX pipeline. The family is
  scoped to the Turbo repository, not its publisher. The pipeline loads a
  mixed 4-bit and 8-bit checkpoint by inferring each layer's width from its
  tensor shapes, uses the published fixed sigma ladder, and releases the text
  encoder, transformer and VAE in turn (`staged` residency). The app decodes
  in bfloat16. Measured on one contended host, peak footprint at 1024 x 1024
  was 11.2 GiB with a bfloat16 decode and 20.5 GiB with float32; see
  `docs/QWEN_IMAGE_21_TURBO.md` for the conditions and open items.
- `crates/ffi`, Swift app: image installs are found in the "Additional Model
  Folders" the text models use. New `ts_image_installed_in_json` (and
  `TurboSparkCatalog.imageInstalled(in:)`) scans a folder and its immediate
  subdirectories, or one install bundle. Removing a found model from the list
  excludes it and leaves its files, as for text models; only installs in the
  shared store are deleted.
- `crates/model-io`: chunk-observed hashing (`hash_file_observed`) with
  cooperative chunk-granularity cancellation for large file verification.
- `crates/catalog`: observed audio profile verification (`verify_observed`)
  with progress reporting callbacks and cooperative cancellation (`VERIFY_CANCELLED`).
- `crates/ffi`: standalone tokenizer C ABI (`ts_tokenizer_open`, `ts_tokenizer_close`,
  `ts_tokenizer_encode`, `ts_tokenizer_decode`, `ts_tokenizer_count_tokens`).
- `crates/ffi`: tool-aware prompt token counting (`ts_session_count_tokens_with_tools`)
  and window fitting (`ts_session_fit_window_with_tools_json`) taking tool schemas
  into prompt budget calculations.
- `crates/ffi`: model family capabilities query (`ts_family_capabilities_json`)
  and architecture KV-quantization capability probe (`ts_kv_quant_supported`).
- `crates/ffi`: cancellable audio session open (`ts_audio_session_open_cancellable`)
  with `TsAudioOpenToken` lifecycle controls and progress reporting callbacks.
- `crates/ffi`: legacy audio model deletion (`ts_audio_delete_legacy`) for unmanaged
  or corrupted receiptless installations.
- `swift/TurboSpark`: standalone `TurboSparkTokenizer` wrapper for direct tokenization
  outside model sessions.
- `swift/TurboSpark`: tool-aware token counting and window fitting methods on
  `TurboSparkSession`.
- `swift/TurboSpark`: runtime family capabilities queries (`Runtime.familyCapabilities`).
- `swift/TurboSpark`: `AudioOpenToken` and cancellable audio opening with live progress
  in `TurboSparkAudio`.
- `crates/runtime`: Whisper transcription progress reporting (`WhisperProgress`)
  with window-level and token-level streaming callbacks.
- `crates/audio`, `crates/runtime`, `crates/cli`: MiniMax Music 0.3 flow guidance reuse
  (`FlowGuidance`, `turbospark-music generate --flow-uncond-interval N
  [--flow-uncond-warmup N]`), benchmark drivers (`music3_bench`,
  `ab_music3_flow.py`), and a spectral distance tool. Opt-in; the default
  output is byte-identical. Interval 3 cut DiT time by 30 to 41 percent in an
  8 s, 30-step measurement on a busy host (see the module README for limits).
- `swift/TurboSparkApp`: native tool calling engine and comprehensive test suite
  in `NativeToolCallingTests`.
- `swift/TurboSparkApp`: audio model installation manager and unmanaged legacy cleanup
  in `AudioWorkspaceController+Installs`.
- `swift/TurboSparkApp`: `EngineImageTranscoder` helper for image decoding and scaling.
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
- `crates/audio`: added speech recognition support for Cactus Compute's Whistle
  checkpoint (`turbospark_audio::models::stt::whistle`), including CQ format
  weights loader, acoustic frontend, causal decoder, tokenizer, unit tests, and
  a CLI example (`crates/audio/examples/whistle_transcribe.rs`).
- `swift/TurboSparkApp`: context window and prefill time budgeting for native
  tool calling (`AppModel.nativeToolBudget`, `PrefillRateStore`), bounding
  definitions to 40% of context (`nativeToolWindowFraction`) and 60 seconds of
  observed prefill speed (`nativeToolPrefillBudgetSeconds`), with an opt-in
  toggle in General settings ("Use it even when the first turn is slow").
- `swift/TurboSparkApp`: audio workspace export panel (`AudioExportPanel`), live
  generation progress overlay (`AudioGenerationProgress`), and reusable prompt
  library view (`AudioPromptLibraryView`).
- `swift/TurboSparkApp`: unified sidebar and navigation rail presentation with
  dynamic theme scaling, accessibility font sizing, grouped navigation
  (`SidebarNavigationGroup`), and localized compact relative timestamps in
  `MessageTimestampFormatter`.
- `swift/TurboSparkApp`: added a Prompt generator to the Music tab that builds
  a MiniMax Music style description from genre, mood, tempo, groove,
  instruments, vocals, structure, and production choices (following the
  published MiniMax Music prompt guide), warns when the vocal choice and the
  lyrics field disagree, and includes a lyrics guide with section-tag and
  song-layout insertion. Nothing is written to the description until
  "Use as Description" is pressed.

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
- `.github/workflows/publish-crates.yml`: `turbospark-audio` is published with
  the other crates, in dependency order ahead of `turbospark-repack`. The job
  runs on `macos-15`, because `turbospark-server` does not compile against
  `turbospark-runtime`'s macOS-gated API on Linux. The `turbospark-audio` and
  `turbospark-gpu` packages exclude their test-only fixtures to stay under the
  crates.io size limit. `turbospark-catalog` reads the bundled Kokoro frontend
  files through `turbospark-audio` (`tts::kokoro::frontend::bundled`) instead
  of a relative path into that crate's source tree, which a published crate
  cannot reach.

### Fixed
- `crates/catalog`: an audio catalog install downloaded every file and then
  refused to publish it. The hub reports the repository id in lowercase, and
  publication compared it case-sensitively against the pinned id, which is
  mixed case in every audio row. Publication now compares it the way source
  resolution does.
- `swift/TurboSparkApp`: the Settings window opens large. Its min/ideal frame
  was ignored by the Settings scene, so it opened at the 940x660 floor on
  every display. It now takes a fixed size from the visible frame (76% x 86%,
  clamped to 940x660 through 1480x1000). Pane content shares one set of insets
  (`SettingsLayout`) instead of each pane picking 16, 20, or 24pt, and the
  sidebar rows and groups have more room. Form panes now use the theme's page
  background instead of the system gray box.
- `swift/TurboSparkApp`: fixed font propagation in `AudioAPIPaneView.swift`
  by using `.themedFont(.small)` instead of hardcoded `.font(.caption)`.
- Swift app: the text-model folder scan no longer lists an image install as a
  text model that cannot load.
- `Vendor/QwenImage`: a module parameter with no checkpoint tensor now fails
  the load. It used to log a warning and keep its random initialization, so
  generation ran and returned noise.
- `Vendor/QwenImage`: the VAE decode evaluates and releases each block
  instead of holding the whole lazy graph. At 512 x 512 the peak fell from
  18.2 to 8.3 GiB with a byte-identical image.
- `turbospark-model pull-image --repo` now shares file selection with the app
  installer. It had no `processor/` rule and could not pack any Qwen-Image
  row.
- `crates/catalog`: `pull-audio` for MiniMax Music 3 failed after a complete,
  hash-verified download with "unexpected repository pin" because the transfer
  returns the Hugging Face repo id lowercased and the check compared it
  byte-for-byte. The repo id now compares ASCII case-insensitively, as
  `hub.rs` already does; the revision pin is still exact.
- `.github/workflows`, `scripts`: install protobuf in CI and release workflows,
  and add an automated Homebrew protobuf check in `make-app-bundle.sh` for
  building `openkind-server`.
- `swift/TurboSparkApp`: resolved Swift compiler warnings and concurrency diagnostics
  across `MLXImageGenerationSession`, `CodemodeWorkerContext`, `REPLHostFacade`,
  `AppModel+Downloads`, and excluded `Tools/Codemode/README.md` in `Package.swift`.
- `crates/audio`, `crates/catalog`, `crates/model-io`: resolved Rust dead-code warnings
  and lint issues across audio STT and Mimi models, formatted `sha256.rs`, and
  updated loopback test assertions in `hub_cache.rs`.
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

