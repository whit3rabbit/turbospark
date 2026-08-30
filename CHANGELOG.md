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
- Prefix KV reuse (cached-prompt continuation): a turn continues from the
  previous turn's KV cache wherever the two prompts agree on their leading
  token ids, instead of resetting and re-prefilling the whole transcript.
  Both prefill loops consult it. Off unless a caller opts in per session
  (`RealForwardRunner::set_prefix_reuse`); `turbospark-check --chat` is the
  only caller that does today, and reports `[prefix-reuse] N/M` per turn on
  stderr (silenced by `--quiet` or `MFERENCE_PREFIX_REUSE=quiet`).
  `--prompt` and `--messages-file` are unaffected and their output is
  byte-identical. Measured on a real Gemma 4 install: prefill 1.777s to
  0.153s on a transcript-shaped prompt, with the generated tokens identical
  to the re-prefilled reference. `RawDecodeResult` gains
  `reused_prefix_tokens`.
- `MFERENCE_PILOT_PROBE`: diagnostic that records a one-layer-ahead router
  prediction beside the actual expert selection in an `MFERENCE_ROUTER_HIST`
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
  all to hook. Includes activation capture (`MFERENCE_RESID_CAPTURE`) and
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
  threadgroup-staging idea. It exists because `MFERENCE_DISPATCH_PROFILE=1`
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

### Changed
- The release workflow requires a `CHANGELOG.md` entry for the tag, publishes
  crates one at a time in dependency order so a re-run of a partially published
  tag resumes correctly, and ships `turbospark-model` in the macOS archive.
- `docs/NEW_MODEL.md` gained a phase for making a new family reachable from the
  CLI: the probe and install driver's family match sites, and the catalog row.

