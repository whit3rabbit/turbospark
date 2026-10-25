# AGENTS.md

Short, always-loaded orientation for the TurboSpark Rust and Swift workspace.
Deep procedures and measurements live in the linked docs. Do not turn this
file into a changelog.

## First read

- Read the nearest crate or app instructions before changing code:
  `crates/<name>/AGENTS.md` or `swift/AGENTS.md`.
- ALWAYS read [DESIGN.md](DESIGN.md) before any macOS app UI change: views,
  controls, copy, color, type, layout, menus, shortcuts, or motion. It holds
  the accessibility, theming, localization, and layout rules that earlier UI
  defects came from.
- Then read [docs/design-system/](docs/design-system/README.md) for token
  values, the type scale, and per-control SwiftUI guides. DESIGN.md decides
  rules and the Swift theme code decides values. The claude.ai Design System
  artifact is a rendered copy of that folder, so read the folder.
- Read [API workspace](docs/API_WORKSPACE.md) when changing Text, Image, or
  TypeSafe serving. Text and Image share the TurboSpark listener. TypeSafe
  owns a separate loopback OpenKind daemon and Keychain key.
- Read [Audio workspace](docs/AUDIO_WORKSPACE.md) and
  [Swift audio and meeting notes architecture](swift/docs/SWIFT_AUDIO_MEETING_NOTES.md)
  when changing speech transcription, voice cloning, audio capture, speaker
  intelligence, or meeting notes.
- Read `docs/DEVELOPMENT.md` before setup or build failures.
- Read `docs/TESTING.md` before adding or changing tests.
- Read `docs/BENCHMARKING.md` and `docs/BENCHMARKS.md` before quoting or
  re-freezing a number. Protocols and frozen rows live there and in
  `.claude/docs/benchmarks.md`, so never add benchmark numbers to AGENTS.md files.
- Read `docs/NEW_MODEL.md` before adding a model family or checkpoint path.
- Read `docs/RELEASE.md` before any release work, tag creation, or changelog
  update. It covers the release flow, `CHANGELOG.md` requirements, and the
  decoupled on-demand crate publishing workflow.
- Read [Kiro workflow](.claude/docs/kiro-workflow.md) before spec or steering
  work (`.kiro/specs/`, `.kiro/steering/`, `.agents/skills/kiro-*`).
- Read the relevant page under `.claude/docs/` for deep verification,
  benchmark, harness, install, power, cross-engine, or diagnostic work.

The root references are:

- [verification reference](.claude/docs/verification.md)
- [engineering guardrails](.claude/docs/engineering-gotchas.md)
- [benchmark reference](.claude/docs/benchmarks.md)
- [agent harness and repository workflow](.claude/docs/agent-harness.md)
- [model gates](.claude/docs/model-gates.md)
- [checkpoint installs](.claude/docs/checkpoint-installs.md)
- [diagnostics](.claude/docs/diagnostics.md)
- [cross-engine checks](.claude/docs/cross-engine-kl.md)
- [power measurement](.claude/docs/power-measurement.md)

## Workspace contract

- macOS is the primary development platform. Rust uses edition 2021, MSRV
  1.82, and a committed `Cargo.lock`.
- `crates/gpu` and the real runtime path require macOS and a Metal-capable
  device. `crates/vision-io` is intentionally part of the portable
  cross-target check.
- Keep source, comments, and docs ASCII. Do not use emojis or em dashes.
- Preserve concurrent work. Inspect `git status` before editing, never reset,
  clean, or check out another session's files, and stage only intended paths,
  re-checking status right before staging.
- A shared file (`CHANGELOG.md`, `README.md`, `docs/`) can hold another
  session's hunks, or already be committed with yours. Read its `git diff`
  before staging. If an uncommitted dependency makes a partial commit unsafe,
  report it instead of staging unrelated work.
- `target/`, build products, staged generated Swift FFI files, and local
  machine notes are not release source.
- Use `st` for repository search. Run `st index` when its index is stale or
  missing. Use exact searches before broad searches.
- If `mf.sqlite3` exists in the repository, use the memoryfield workflow
  before exploring the codebase.

## Safety invariants

- Decode flow is selected by `ArchConfig.family`, never by tensor names.
  A wrong family can produce fluent, incorrect output.
- Producers return logits. Sampling owns normalization. Never softmax in both
  the producer and sampler.
- Repeated Metal encoding loops use `gpu::autorelease_pool`. Without an
  inner pool, autoreleased command objects accumulate for the life of the
  process.
- KV cache sizing and ring capacity are part of correctness. Derive the ring
  from the layer mask and put capacity and every kernel function constant
  (split-KV GQA: group, simdgroups, head_dim, unroll) in the pipeline cache
  key. A missing constant silently reuses the wrong pipeline.
- Routed expert slots are dispatched in router ranking order. Slot order is
  reduction order.
- Token IDs crossing crate boundaries remain signed 32-bit
  (`turbospark_core::TokenId`). Resolve fixture IDs from a loaded tokenizer.
  Never copy numeric IDs from fixture JSON.
- Family support is cross-layer. Trace the family enum and persisted string
  through model-io, runtime, FFI, Swift, catalog, CLI, server, docs, and tests.
- Unsafe code is intentional only in the documented ABI, mapping, and streaming
  boundaries. FFI entry points must stay inside the ABI guard and must not
  unwind across `extern "C"`.
- A platform claim is not a gate. When changing cfgs, dependency tables,
  memory mapping, kernels, or FFI, run the relevant cross-target and real
  hardware checks from the verification reference.
- A composite measurement inherits the limits of its components. Do not turn a
  compile, CPU run, projection, synthetic fixture, or contaminated host reading
  into a hardware-gate, Metal, quality, or memory claim.
- Do not widen quality or resource tolerances without new evidence. Preserve
  negative findings and refusal paths in the canonical docs.
- Keep Metal JIT warmup opt-in (`TURBOSPARK_METAL_KERNEL_WARMUP=1`). Read
  [MoE startup](docs/MOE_STARTUP.md) before changing the default.
- Steel is the dense qwen prefill default. Another family needs its own
  real-model quality, memory, and paired performance gates before it takes
  Steel or the GQA decode attention.
- The ANE+Steel prefill split for dense Qwen is opt-in (`ane-prefill` cargo
  feature, `TURBOSPARK_ANE_PREFILL`, or Swift runtime settings). Auto mode
  checks Apple Silicon M4+ / A18+ before activating and gracefully degrades to
  Steel if programs are missing or on older chips.
- Qwen batched prefill, speculative verify, and MTP priming encode their small
  per-row ops (RMSNorm, RoPE, gate/act, residual adds) in a single multi-row
  dispatch. `TURBOSPARK_QWEN_BATCHED_ROW_OPS=0` restores the per-row form.
- Read [MLX kernel experiments](docs/MLX_KERNELS.md) before reusing oMLX or
  MLX kernels, and the [speedup playbook](docs/SPEEDUP_PLAYBOOK.md) before
  kernel performance work.
- Qwen MoE batched prefill has its own drift bounds and opt-out. Read
  `crates/runtime/AGENTS.md` and `docs/BATCHED_PREFILL.md` before changing it.
- Voice cloning with MegaTTS3 requires explicit caller consent up front
  (`consent: true`) because community WavVAE encoder provenance is unverified.
  In the app, consent covers one exact reference region and is never carried
  over by reusing or recording a reference (`swift/AGENTS.md`).
- Release builds pin the runner and Xcode by name (`XCODE_VERSION` in
  `release.yml`, asserted after selection), never `latest`.
- `ci.yml`'s `package-macos` also builds with the oldest supported Xcode,
  because the toolchain changes SwiftPM's generated resource lookup. Read
  `swift/AGENTS.md` before touching resources or `Package.swift` targets.
- Credentials are not stored by default. Keychain use is opt-in per credential
  and never happens at launch for an unused feature.
- macOS privacy prompts (microphone, screen recording) appear only after the
  user chooses Allow, and a declined permission never blocks features that do
  not need it. Only `RecordingPermissions` asks. Both rules: `swift/AGENTS.md`.
- Every UI string uses bundled `Text` or `AppStrings.localized` to pass the
  ten localization parity gates. `AppJSONStore` must quarantine corrupt files
  rather than overwriting them, preventing silent data loss.
- Audio meeting notes summarization is strictly non-destructive: model
  generation mutates only `item.summary` and must never overwrite or mutate
  original transcript segments or text.

## Minimal verification

For ordinary Rust changes:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

For Swift or FFI changes:

```sh
make swift-lib
make swift-test
```

For app or release artifacts:

```sh
make swift-app
make app-bundle
scripts/verify-app-bundle.sh
make dmg
```

`verify-app-bundle.sh` runs the built app. Packaging success alone shipped two
releases that crashed on launch.

The commands above are the baseline only. Changes to decode, quantization,
sampling, KV, Metal, model intake, FFI, or real-install behavior require the
additional gates documented in [verification](.claude/docs/verification.md)
and [model gates](.claude/docs/model-gates.md).

Judge a chat checkpoint only through its template: `--messages-file` or
`--chat`. `--prompt` and the `turbospark run <model> "text"` shorthand encode
the text verbatim, so an instruction-tuned model continues the user's message
and reads as a broken forward pass. The `--messages-file` banner says `chat
template applied`, and `TURBOSPARK_DEBUG_PROMPT_IDS=1` prints the ids.

## Change boundaries

- Keep crate-local architecture and gotchas in the nearest crate file. Do not
  copy root rules into every module.
- Release and packaging procedures live in `docs/RELEASE.md` and the
  verification reference, design rationale in `docs/`, and unfinished work in
  the issue tracker. Do not append session summaries or dated handoff notes here.
- Local green is not release evidence. Your Xcode can differ from the pinned
  one, your `.build` can hold stale objects that hide a clean-build failure,
  and your tree can differ from your commit. Follow the pre-publish checklist
  in `docs/RELEASE.md` and let the CI matrix decide.
- Do not launch the installed or an older build to compare behavior while
  testing credential storage. It can recreate the very state under test
  (`swift/AGENTS.md` lists the affected versions).
- Every new test must exercise the behavior it claims to cover. Mutation-check
  new guards when the behavior is subtle, and assert that a mutation applied
  before treating a survivor as evidence.

## Project map

- `crates/model-io`: checkpoint formats, manifests, mappings, and resident
  buffers.
- `crates/repack`: model intake and `.gturbo` writing.
- `crates/runtime`: family dispatch, forward passes, state, and sessions.
- `crates/gpu`: Metal kernels and dispatch.
- `crates/compute`, `crates/core`, `crates/selection`,
  `crates/tokenizer`, `crates/streaming`, `crates/invocation`: portable
  primitives and protocol boundaries.
- `crates/window-fit`: deterministic conversation-window fitting.
- `crates/image`: native text-to-image generation pipeline.
- `crates/vision-io`: portable vision preprocessing.
- `crates/audio`: portable audio crate (DSP, neural ops, TTS, STT, music,
  VAD, codecs). See `docs/AUDIO.md`.
- `crates/embeddings`: portable embedding models with pinned checkpoints. See
  `crates/embeddings/MODELS.md`.
- `crates/coreml`: the Apple Core ML runner (macOS stub elsewhere). Model
  pipelines on it live in `crates/runtime/src/coreml/`. See
  [Core ML models](docs/COREML.md).
- `crates/ffi`: the C ABI consumed by Swift.
- `crates/cli`, `crates/catalog`, `crates/server`: user-facing install,
  command, and service surfaces.
- `crates/bench`: measurement harnesses. Read its local pointer and the
  benchmark docs before running or interpreting a gate.
- `crates/lab`: `tslab`, the evaluator for kernel and method experiments
  and the autoresearch loop (`crates/lab/program.md`).
- `swift/TurboSparkApp`: the macOS app, persistence, tools, localization,
  and Swift-facing model controls.

## Final handoff

Report the files changed, focused checks run, any unrelated checkout failures,
and whether a real hardware or release gate was actually run.
