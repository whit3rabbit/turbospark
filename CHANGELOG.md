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

