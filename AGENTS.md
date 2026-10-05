# AGENTS.md

This file is the short, always-loaded orientation for the TurboSpark Rust and
Swift workspace. Keep detailed procedures, measurements, and historical
evidence in the linked project docs. Do not turn this file into a changelog.

## First read

- Read the nearest crate or app instructions before changing code:
  `crates/<name>/AGENTS.md` or `swift/AGENTS.md`.
- Read [API workspace](docs/API_WORKSPACE.md) when changing Text, Image, or
  TypeSafe serving. Text and Image share the TurboSpark listener; TypeSafe
  owns a separate loopback OpenKind daemon and Keychain key.
- Read `docs/DEVELOPMENT.md` before setup or build failures.
- Read `docs/TESTING.md` before adding or changing tests.
- Read `docs/BENCHMARKING.md` and `docs/BENCHMARKS.md` before quoting or
  re-freezing a number.
- Read `docs/NEW_MODEL.md` before adding a model family or checkpoint path.
- Read `docs/RELEASE.md` before any release work, tag creation, or changelog update.
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
- Preserve concurrent work. Inspect `git status` before editing, do not reset,
  clean, or check out another session's files, and stage only intended paths.
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
  from the layer mask and include capacity in pipeline-cache constants.
- Routed expert slots are dispatched in router ranking order. Slot order is
  reduction order.
- Token IDs crossing crate boundaries remain signed 32-bit
  (`turbospark_core::TokenId`). Resolve fixture IDs from a loaded tokenizer;
  never copy numeric IDs from fixture JSON.
- Family support is cross-layer. Trace the family enum and persisted string
  through model-io, runtime, FFI, Swift, catalog, CLI, server, docs, and tests.
- Unsafe code is intentional only in the documented ABI, mapping, and streaming
  boundaries. FFI entry points must stay inside the ABI guard and must not
  unwind across `extern "C"`.
- A platform claim is not a gate. When changing cfgs, dependency tables,
  memory mapping, kernels, or FFI, run the relevant cross-target and real
  hardware checks from the verification reference.
- A composite measurement inherits the limits of its components. Do not turn a
  projection, CPU run, synthetic fixture, or contaminated host reading into a
  real Metal, quality, or memory claim.
- Do not widen quality or resource tolerances without new evidence. Preserve
  negative findings and refusal paths in the canonical docs.
- Keep Metal JIT warmup opt-in (`TURBOSPARK_METAL_KERNEL_WARMUP=1`);
  the Qwen MoE experiment found no repeatable startup gain. Read
  [MoE startup](docs/MOE_STARTUP.md) before changing the default.
- Read [MLX kernel experiments](docs/MLX_KERNELS.md) before reusing oMLX
  kernels. Steel has synthetic prefill evidence; runtime defaults still
  require real-model quality, memory, and paired performance gates.

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
make dmg
```

Before cutting an app release or tag, always review `docs/RELEASE.md` for the
release flow, changelog update requirements (`CHANGELOG.md`), and the decoupled
on-demand crate publishing workflow.

The commands above are the baseline only. Changes to decode, quantization,
sampling, KV, Metal, model intake, FFI, or real-install behavior require the
additional gates documented in [verification](.claude/docs/verification.md)
and [model gates](.claude/docs/model-gates.md). Do not claim a hardware gate
from compilation or a CPU-only result.

## Change boundaries

- Keep crate-local architecture and gotchas in the nearest crate file. Do not
  copy root rules into every module.
- Keep benchmark protocols and frozen rows in `docs/BENCHMARKING.md`,
  `docs/BENCHMARKS.md`, and `.claude/docs/benchmarks.md`. Do not add
  benchmark numbers to AGENTS.md files.
- Keep release and packaging procedures in `docs/RELEASE.md` and the
  verification reference. Always review `docs/RELEASE.md` before releasing.
- Keep project planning in the existing planning document, design rationale in
  `docs/`, and unfinished work in the issue tracker. Do not append
  session summaries or dated handoff notes here.
- For a contested file, re-check status immediately before staging. If a
  dependency on another uncommitted file makes a clean partial commit unsafe,
  report the dependency instead of staging unrelated work.
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
- `crates/audio`: unified portable audio crate: DSP primitives (waveform,
  WAV, resampling, FFT, STFT, mel), neural tensor ops, and models organized
  by role (music, STT, TTS, VAD, STS, codec, LID); see `docs/AUDIO.md`.
- `crates/ffi`: the C ABI consumed by Swift.
- `crates/cli`, `crates/catalog`, `crates/server`: user-facing install,
  command, and service surfaces.
- `crates/bench`: measurement harnesses. Read its local pointer and the
  benchmark docs before running or interpreting a gate.
- `swift/TurboSparkApp`: the macOS app, persistence, tools, localization,
  and Swift-facing model controls.

## Final handoff

Report the files changed, focused checks run, any unrelated checkout failures,
and whether a real hardware or release gate was actually run. Keep claims
bounded by the evidence in the current checkout.


# Agentic SDLC and Spec-Driven Development

Kiro-style Spec-Driven Development on an agentic SDLC

## Project Memory
Project memory keeps persistent guidance (steering, specs notes, component docs) so Codex honors your standards each run. Treat it as the long-lived source of truth for patterns, conventions, and decisions.

- Use `.kiro/steering/` for project-wide policies: architecture principles, naming schemes, security constraints, tech stack decisions, api standards, etc.
- Use local `AGENTS.md` files for feature or library context (e.g. `src/lib/payments/AGENTS.md`): describe domain assumptions, API contracts, or testing conventions specific to that folder. Codex auto-loads these when working in the matching path.
- Specs notes stay with each spec (under `.kiro/specs/`) to guide specification-level workflows.

## Project Context

### Paths
- Steering: `.kiro/steering/`
- Specs: `.kiro/specs/`

### Steering vs Specification

**Steering** (`.kiro/steering/`) - Guide AI with project-wide rules and context
**Specs** (`.kiro/specs/`) - Formalize development process for individual features

### Active Specifications
- Check `.kiro/specs/` for active specifications
- Use `$kiro-spec-status [feature-name]` to check progress

## Development Guidelines
- Think in English, generate responses in English. All Markdown content written to project files (e.g., requirements.md, design.md, tasks.md, research.md, validation reports) MUST be written in the target language configured for this specification (see spec.json.language).

## Minimal Workflow
- Phase 0 (optional): `$kiro-steering`, `$kiro-steering-custom`
- Discovery: `$kiro-discovery "idea"` -- determines action path, writes brief.md + roadmap.md for multi-spec projects
- Phase 1 (Specification):
  - Single spec: `$kiro-spec-quick {feature} [--auto]` or step by step:
    - `$kiro-spec-init "description"`
    - `$kiro-spec-requirements {feature}`
    - `$kiro-validate-gap {feature}` (optional: for existing codebase)
    - `$kiro-spec-design {feature} [-y]`
    - `$kiro-validate-design {feature}` (optional: design review)
    - `$kiro-spec-tasks {feature} [-y]`
  - Multi-spec: `$kiro-spec-batch` -- creates all specs from roadmap.md in parallel by dependency wave
- Phase 2 (Implementation): `$kiro-impl {feature} [tasks] [--review required|inline|off]`
  - Without task numbers: autonomous mode (subagent per task + independent review + final validation)
  - With task numbers: manual mode (selected tasks in main context, still reviewer-gated before completion)
  - `--review off` skips task-local review; use it intentionally and keep `$kiro-validate-impl {feature}` as the final quality gate
  - `$kiro-validate-impl {feature}` (standalone re-validation)
- Progress check: `$kiro-spec-status {feature}` (use anytime)

## Skills Structure
Skills are located in `.agents/skills/kiro-*/SKILL.md`
- Each skill is a directory with a `SKILL.md` file
- Use `/skills` to inspect currently available skills
- Invoke a skill directly with `$kiro-<skill-name>`
- `kiro-review` -- task-local adversarial review protocol used by reviewer subagents
- `kiro-debug` -- root-cause-first debug protocol used by debugger subagents
- `kiro-verify-completion` -- fresh-evidence gate before success or completion claims
- Use skills explicitly requested by the user and skills relevant to the task's domain, including design, accessibility, and UX.
- Select skills from their descriptions or metadata first, then read only the selected skills and the references needed for the task.
- Follow explicit host and project rules and retain required workflow checks. Do not skip relevant skills just because the task is small.

## Development Rules
- Keep steering current and verify alignment with `$kiro-spec-status`
- Follow the user's instructions precisely, and within that scope act autonomously: gather the necessary context and complete the requested work end-to-end in this run, asking questions only when essential information is missing or the instructions are critically ambiguous.

## Steering Configuration
- For spec and implementation work, load the core steering files below from `.kiro/steering/`. Reuse current context rather than rereading unchanged files.
- Load additional steering only when required by project rules or relevant to the task.
- Default files: `product.md`, `tech.md`, `structure.md`
- Custom files are supported (managed via `$kiro-steering-custom`)
