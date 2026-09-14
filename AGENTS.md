# AGENTS.md

CLAUDE.md is a symlink to this file.

Conventions, gotchas, and commands for working in this Rust workspace,
a behavior-compatible port of the Mference Swift inference engine (see
`ROADMAP.md` for the forward roadmap and descope record, and
`DEVIATIONS.md` for what is scaffolded rather than fully wired). Keep all code, comments, and docs
ASCII: no emojis and no em dashes (project rule).

## Where the depth lives

Each page below is the HOME for its numbers; this file carries none of them.
Several record measured NEGATIVES, and the rule is the same for all of them:
read the page before proposing the thing it refutes.

| Page | Covers | Read it before |
|---|---|---|
| `docs/DEVELOPMENT.md` | prerequisites, the build order, the test and run loop, what CI does NOT cover | setting the tree up, or a build that fails before your change |
| `docs/TESTING.md` | what the suite proves, how tests are gated | adding or gating a test |
| `docs/BENCHMARKING.md` | the three `turbospark-bench` modes, peak-memory measurement, the memory oracle | quoting a throughput or footprint number |
| `docs/BENCHMARKS.md` | the frozen rows: quality, throughput, memory, cross-engine KL | re-freezing anything |
| `docs/POWER_BASELINE.md` | watts and joules-per-token; the one page measured on BATTERY | reading a power row |
| `docs/EXPERT_ROUTING.md` | domain-restricted expert sets AND router-lookahead prefetch, two measured negatives | proposing expert pruning, pinning, or prefetch |
| `docs/EXPERT_RESIDENCY.md` | reading routed experts in place out of an `mmap` instead of a pinned slot | quoting a footprint, or touching the slot cache |
| `docs/ACTIVATION_SPARSITY.md` | dense-FFN neuron caching, measured negative | proposing to load "only what the token uses" |
| `docs/SPECULATIVE_DECODING.md` | speculative decoding and DFlash on the MoE family | proposing a drafter |
| `docs/MTP_SPECULATIVE.md` | the same question for the DENSE family, where the answer differs | re-costing that decision |
| `docs/MTP.md` | the MTP head: architecture and measured facts, nothing projected | changing the head |
| `docs/BATCHED_PREFILL.md` | the batched prefill driver, its seams and its dead ends | touching that driver |
| `docs/NEW_MODEL.md` | end-to-end checklist for wiring a new model family | starting one |
| `docs/CLI.md` | every flag on all three binaries, incl. the steering walkthrough | adding or changing a flag |
| `docs/MODELS.md` | the catalog, the probe, `pull`, how to add a row | touching `models.json` |
| `docs/GTURBO.md` | the `.gturbo` install format this port reads and writes | changing the writer |
| `docs/MODEL_FAMILY.md` | GGUF `general.architecture` and HF `model_type` tables | adding an architecture row |
| `docs/DECODE_BUDGET.md` | where a decoded token's time goes; three decode dead ends | optimizing decode |
| `docs/LOAD_GUARD.md` | the memory guardrail tiers, the AutoFit floor, the pressure watcher | changing what a session may commit |
| `docs/DFLASH2.md` | the DFlash2 block drafter: architecture, state, verify | touching that drafter |
| `docs/OBLITERATION.md` | live directional steering, with measurement | changing steering |
| `docs/TRUBOQUANT.md` | TurboQuant KV-cache quantization: the assessment, and the `--kv-bits` feature built on it | touching KV quantization, or quoting the assessment it reversed |
| `docs/VISION.md` | the vision PIPELINE: injection, mRoPE dispatch, the four gates, the cross-engine rows | touching anything an image passes through |
| `docs/ZIMAGE_TURBO.md` | the Z-Image-Turbo image-model case study, IG0/IG1 lessons, resource protocol, and reusable image bring-up sequence | starting another image-generation model, or quoting the Z-Image phase order |
| `docs/VISION_PHASE0.md` | the vision CHECKPOINT: tensors, mRoPE semantics, activation magnitudes, the INT4 decision | reading a tower fact off the checkpoint |
| `docs/QWEN4_PHASE0.md` | `qwen4_exp` (Qwen3.8-Flash-Next) Phase 0 fact-finding: config, tensor layout, two independent references cross-checked | reading a `qwen4_exp` fact, or continuing that bring-up |
| `docs/QWEN4_EXP.md` | `qwen4_exp` bring-up beyond Phase 0: intake, decode wiring, memory policy, the router/shared-expert-gate dtype bug and fix, first real-hardware decode | touching `families/qwen4/` or the safetensors write path, or continuing that bring-up |
| `docs/SPARK_PHASE0.md` | `spark2_5` (Spark-X2.5-4B) Phase 0 fact-finding: config, per-class RoPE, the headwise gate, the fused QKV, GGUF tensor inventory, tokenizer frame | reading a `spark2_5` fact, or touching `families/spark/` |
| `docs/MINIMAX_M2_PHASE0.md` | MiniMax-M2 split-GGUF intake, FP32 sigmoid routing, whole-projection norms, pinned evidence, and release-gate status | continuing MiniMax-M2 bring-up or changing its intake and execution contract |
| `docs/SKILL_STATE.md` | the SKILL.state bounded-state agent runtime, a measured POSITIVE with its scale caveat | proposing agent context compaction, structured output, or constrained decoding |
| `docs/SWIFT_BINDINGS.md` | the C ABI and the Swift package: contract and limits | changing the FFI |
| `swift/docs/SWIFT_TOOLS.md` | Swift native tool implementation: execution, containment, adding new tools | implementing or changing tools in TurboSparkApp |
| `swift/docs/SWIFT_TOOL_CATALOG.md` | full reference catalog: all built-in tools, parameters, schemas, permission matrix, and patterns | inspecting or using specific tool APIs and parameters |
| `swift/docs/SYNTEXT.md` | Syntext indexed code search, project indexing, grep_search, live file buffer, and dual-staticlib symbol localization | touching Syntext integration, grep_search, or project code search |
| `swift/docs/SWIFT_AGENT_MODE.md` | the `.agentAuto` permission mode: a local-model classifier judges the ask-band, with hard gates, fallback counters, and hints | touching the classifier routing, `hardGated`, `AgentModeGate`, or quoting a permission verdict |
| `swift/docs/SWIFT_SKILLS.md` | Swift skills: architecture, scopes, file layout, and marketplace integration | changing skills, discovery, or marketplace |
| `swift/docs/SWIFT_PLUGINS.md` | Swift plugins ported from Claude Code: manifest, contributions, enable cascade, marketplace, the two deviations | touching anything under the plugin system, or quoting a plugin rule |
| `swift/docs/SWIFT_PROFILES.md` | user profiles: the registry, the Default-user contract, shared versus per-profile storage, save-and-relaunch switching | adding a store, or touching `AppStorageRoot`, `UserProfileStore`, or a `~/.turbospark` path in the app |
| `swift/docs/SWIFT_COMPACTION.md` | context compaction: the threshold, the summarizer, the boundary, the ghost rule, the fail-open policy | touching `AppChatCompaction`, the prompt-assembly boundary skip, or quoting an auto-compact threshold |
| `swift/docs/SWIFT_MEMORY.md` | auto-memory: the per-project memory directory, the `MEMORY.md` index and its budgets, the `memory` tool, the `#` quick-save, the `/memory` command | touching `MemoryStore`, `MemoryPromptBuilder`, the `memory` tool, or quoting an index budget |
| `swift/docs/SWIFT_CHAT_SEARCH.md` | the Cmd+K Search Chats dialog: what is searched, ghost exclusion BY THE FLAG, AND-of-substring matching over in-memory haystacks, no scoring | touching `ChatSearch`, adding a searched field, or adding a result surface |
| `swift/docs/SWIFT_GOALS.md` | the `/goal` loop: the stop-seam evaluator (met / not-met / impossible), deferral while background work runs, 30-min doubling idle check-ins capped at 3 between prompts, stall pause at 3 tool-free rounds, restore-keeps-condition-only | touching `AppModel+Goal`, `ChatGoal`, `GoalEvaluator`, the goal arm atop `dispatchStopAndContinueIfBlocked`, or quoting a check-in interval |
| `swift/docs/SWIFT_CONTEXT_RING.md` | the composer's context ring: the 70/90 tint tiers, the breakdown popover, the exact-total vs apportioned-rows split, the one-builder rule, the state#116 double count | touching the context ring, `ContextUsageSummary`, `buildEstimateParts`, `buildSystemPromptSections`, or quoting a context-fill number |
| `swift/docs/SWIFT_MESSAGE_EDITING.md` | message retry, edit and branch: the `alternates` field, version switching, response-variant seeding, the compaction clamp | touching `AppModel+MessageEditing`, the `alternates` field, or the message hover bar |
| `swift/docs/SWIFT_LOCALIZATION.md` | the string catalog pipeline: the xcstrings source, `compile-strings.sh`, the `bundle: .module` rule and its label-closure workarounds, the six parity gates, the greetings.json second system | adding or translating a UI string, adding a language, touching the catalog or `compile-strings.sh`, or quoting a language count |
| `swift/docs/SWIFT_TURN_PIPELINE.md` | the message queue (prompts submitted mid-turn park, then drain at the agent loop's step boundaries as steers or at the turn tail; user prompts before task notifications) and system reminders (ephemeral todo/plan `<system-reminder>` blocks at assembly time) | touching `AppModel+Queue`, `SystemReminders`, the `canQueue` gating, the turn-tail drains, or quoting the todo-staleness rule |
| `swift/docs/SWIFT_SETTINGS_AUDIT.md` | the settings audit: every persisted key traced to its consumers, every pane control checked, the font-propagation cause and its open task list | adding a setting or a pane control, or proposing the themed-font conversion |
| `swift/docs/SWIFT_QWEN_PARITY2.md` | the second web-shell parity pass: `/diff` `/log` `/prs`, bang `!cmd`, read-only split panes, sidebar time buckets + project accents, echarts preview, the interactive AskUserQuestion (static waiter, banner-not-row, dismiss-is-an-answer) and the QR descoping | touching any of those features, the `answerWaiter` static, `ShellMessageContent`, or `ProjectAccentColor` |
| `swift/docs/KEYBOARD_SHORTCUTS.md` | keyboard shortcuts, menu commands, and VoiceOver accessibility integration | changing shortcuts or accessibility in TurboSparkApp |
| `docs/STREAMING.md` | the token streaming pipeline: the push-callback primitive, `TurnSplitter`, the per-consumer adapters, the FFI event kinds | adding a generation consumer, touching the split/decode wiring, or proposing an engine-side iterator or async stream |
| `docs/PERMISSION_GATE.md` | the local command classifier, its corpora, and a measured negative | touching `.auto`, or quoting a hazard score |
| `docs/RELEASE.md` | release checklist, versioning, tags, rot guards | cutting a release |
| `docs/TOOL_CALLING.md` | the three tool-call implementations (native decoder, rescue tier, Swift), per-family coverage, the special-token wrinkle | touching tool-call parsing, or adding a dialect or rescue format |

Two standing rules come out of those pages rather than from any one crate.
**Cost an optimization by the terms it does NOT touch**: one kernel fix was
decisive on the dense family and worth about two points on the MoE one,
because the term it improves is half of MoE decode while a fifth cannot
amortize at all. And **a composite built on a broken or unoptimized component
measures the component, not the question** -- both speculative pages record a
reversal of exactly that shape.

Do your best to keep code files under 400 lines but it's a suggestion not a hard rule. If over 400, decide if refactoring makes sense.

## Stack

- Language: Rust, edition 2021, MSRV 1.82 (see `rust-toolchain.toml` and
  `[workspace.package] rust-version`).
- Toolchain pin: stable, with the `rustfmt` and `clippy` components.
- Build system: cargo, resolver "2".
- License: MIT.
- GPU: `crates/gpu` is macOS-only and needs a Metal-capable device plus
  Xcode's `metal` toolchain (`xcrun -sdk macosx metal`) to run its tests; on
  other platforms the crate compiles to nothing (see its Gotcha below).

## Verification policy

Every change should keep these green before handoff:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

Anything that touches the decode path, the output head, the KV cache, or a
Metal encode loop additionally needs the real-model gates from "Real-model
smoke" above, all three of them. **"Touches" includes MOVING it.** A pure
refactor of a family flow is a numerics change until a real model says
otherwise: `5279c88` split `real_forward_qwen.rs` into `families/qwen/`,
picked up a Gemma sandwich norm on the way, and shipped a Qwen whose
reference perplexity read 255,409 against a frozen 6.2536, with the whole
workspace suite green (`crates/runtime/CLAUDE.md` Gotcha 11). Run the gates
per FAMILY the change touches, not once for the workspace.

**NEW FAMILY SUPPORT IS A CROSS-LAYER CHANGE.** A family is not supported when
Rust parses it alone. When adding a `ModelFamily`, or making an existing family
reachable by a new checkpoint, trace both the enum variant and its canonical
persisted string through every consumer before calling the work complete:

- `crates/model-io`: `as_str`, `parse`, `ALL`, architecture/config mappings,
  and family tables.
- `crates/runtime` and `crates/ffi`: decode-flow dispatch, capability
  predicates and refusal reasons, session-info fields, and wire/ABI tests.
- `swift/`: pre-open family detection and feature badges, loaded-session
  `info` gating, Safety & Steering and Model Settings controls, chat/footer
  status, server model rows/settings, and family visual or alias matching. A
  family that is true in Rust but absent from a Swift allowlist is incomplete
  support.
- Catalog, install, CLI, server, documentation, and tests, including
  `docs/MODEL_FAMILY.md` and `docs/NEW_MODEL.md`. For directional steering or
  obliteration, also update the steering capability predicate, Swift status
  surfaces, and server-model reporting described in `docs/OBLITERATION.md`.

Use `st -n` for the exact enum variant and canonical string across
`crates/`, `swift/`, `docs/`, and tests, then inspect every hit. Do not rely on
compiler errors to find duplicated string allowlists. Add coverage for the
Rust supported/unsupported result, Swift pre-open detection, loaded-session
capability reporting, and server `supported`, `off`, and `active` states where
the feature has a UI surface. A new checkpoint that reuses an existing family
must instead verify the resolver, catalog row, and Swift descriptor without
inventing a second family string.

**A COMPILE ERROR IN A CRATE YOU DID NOT TOUCH IS PROBABLY NOT YOURS.** This
tree is routinely worked by more than one session at once, and the failure
arrives as a normal-looking build break minutes after your own suite went
green. Check mtimes before debugging it, and do not "fix" another session's
half-finished edit. The same applies to `git add`: re-run `git status`
immediately before staging, and to the GATES themselves: `cargo fmt --check`
and `cargo clippy --workspace --tests` are tree-wide, so read the PATHS they
name before believing a red one. `rustfmt --check --edition 2021 <your files>`
is the per-file form and attributes cleanly. Interactive `git add -p` is
unavailable here, so a CONTESTED file is staged by RECONSTRUCTION:
`git show HEAD:<path>` into a temp file, apply only your edits with an
assert-on-missing, `git hash-object -w`, `git update-index --cacheinfo`. Two
traps. Match patterns must come from HEAD and not the working copy, which
carries the other session's edits. And reconstruction leaves the WORKING COPY
BEHIND THE INDEX -- your committed text is not on disk, so the next session to
`git add` that path silently reverts it; re-apply the edit to the working copy
afterwards and confirm `git diff` shows only foreign hunks. And list paths
LITERALLY in any per-file gate: zsh does not word-split an unquoted variable,
so a loop over `$FILES` checks ONE nonexistent path and reports clean.
AND QUOTE ANY GLOB YOU PASS TO A TOOL: zsh expands `--include=*.swift` itself
and aborts the command with `no matches found` when nothing matches in the
CWD, so `grep -rn x --include=*.swift .` reports 0 hits. A 0 from a usage
survey reads as "nothing calls this", which is a finding rather than a typo.
Write `--include="*.swift"`.

1. greedy generation stays coherent (catches broken math),
2. SAMPLED generation stays coherent (catches distribution bugs that greedy
   cannot see -- Gotcha 16),
3. the memory oracle passes (catches allocation and retain bugs that
   correctness cannot see -- Gotchas 17 to 19).

A change that could move NUMERICS (a kernel, the head, quantization, the
sampler) also runs the quality gate, which ADDS to the three above rather
than replacing any of them: coherence is judged by eye and cannot see a
few percent of drift, which is exactly what a quantization change does
when it is subtly wrong rather than broken. A digest mismatch there is not
automatically a failure -- reduce order legitimately changes bytes -- but
it is never allowed to pass unexplained, and the perplexity number is the
tiebreak. That tiebreak has been calibrated rather than assumed: shifting
one quantization level in 0.195% of routed-expert bytes moves perplexity
+37.7% and in 0.0122% moves it +10.5%, while 0.0015% moves it +0.54% and
is missed, so the gate's detection floor sits between those last two
(`crates/bench/tests/quality_sensitivity.rs`, curve in
`docs/BENCHMARKS.md`).

Numerics parity with any upstream implementation is explicitly out of scope;
only the structural and configuration contracts are exercised by the tests,
except where a real CPU-vs-GPU parity test exists (`crates/gpu`'s
`rms_norm_parity.rs`).

The four commands above cover everything except the `#[ignore]`d tests,
which are opt-in and not part of the handoff gate. They are the checkpoint
downloads, the per-family memory oracles and quality gates, the quality
sensitivity proof, the cross-engine dumps, and the real-install behaviour
gates (prefix KV reuse is one). **That list is deliberately not exhaustive
and the COUNT lives in `docs/TESTING.md`, not here** -- it has rotted past
2x twice, because nothing goes red when a number in prose goes stale. Run an oracle when a change could move memory or decode
throughput, and a quality gate when it could move numerics. The
sensitivity test is not part of routine verification: run it when the
gate's own credibility is in question, for example after changing the
corpus, the scoring, or the quantization path itself. `docs/TESTING.md` documents the gating
conventions and the test-writing rules (never hardcode a fixture token
id, never assert generated text, prefer exact assertions over
thresholds); `docs/BENCHMARKING.md` documents the benchmark modes and
baselines.

Every new test is MUTATION-CHECKED before it is believed, and the loop is
seconds: `cp f /tmp/f.bak`, mutate with `perl -0pi -e 's/A/B/' f`, run that
one target, `cp /tmp/f.bak f`. Assert each mutation reddens ONLY its own
case. One that reddens everything is not a failure of the test -- it usually
means an INVARIANT is doing the work, which is its own finding and worth
recording rather than tuning away.

**ASSERT THE MUTATION APPLIED, or a survivor is meaningless.** `cargo fmt`
wraps and re-indents match arms and long calls, so a `perl -0pi -e` pattern
written from the source you drafted stops matching the source on disk --
silently, since perl reports nothing when a substitution finds no target.
Three mutations "survived" in one sitting that way and read as three weak
tests; two were fine and one was a real gap. Substitute through a helper that
fails when the old text is absent (`assert old in s` in a two-line python
heredoc), and prefer patterns short enough to survive reformatting.

**AND ASSERT IT APPLIED WHERE YOU MEANT.** Presence is not uniqueness: a
pattern written at one indent level is a SUBSTRING of the same line at a
deeper one, so `replace(old, new, 1)` silently takes the first match. Two
sibling call sites in `families/llama/mod.rs` (12-space and 16-space) gave
byte-identical mutation results twice, which reads as "one hook covers both"
rather than as a mis-aimed pattern. Assert `count(old) == 1`, or anchor on a
neighbouring line.

**A SURVIVOR WHOSE MUTATION DID APPLY IS A MISSING TEST, NOT A WEAK ONE.**
Deleting the pointer clause from `speculation_blocker`'s no-head arm left all
37 cases in `speculation_policy_tests.rs` green, because that module feeds
FIXTURE strings into `resolve_speculation` and never calls the blocker: it
pins the ROUTING and can see no text change at all. Ask what actually CALLS
the mutated function before writing the guard, and expect the answer to be a
different file with a differently-shaped fixture.

See `DEVIATIONS.md` for the full list of what this port scaffolds versus
fully implements, `ROADMAP.md` for the forward roadmap and descope
record, and
`docs/NEW_MODEL.md` for the end-to-end checklist for wiring a new model
family (what to map, what to specialize, what to measure, in order), and
`docs/MODELS.md` for the catalog, the probe, and how to install a model that
is not in it.

## Build, test, dev commands

```sh
# Build every crate in the workspace.
cargo build --workspace

# Run the whole test suite.
cargo test --workspace

# Run one crate only.
cargo test -p turbospark-core
cargo test -p turbospark-compute
cargo test -p turbospark-invocation
cargo test -p turbospark-selection
cargo test -p turbospark-window-fit
cargo test -p turbospark-tokenizer
cargo test -p turbospark-model-io
cargo test -p turbospark-streaming
cargo test -p turbospark-gpu       # macOS only; needs a real Metal device
cargo test -p turbospark-runtime
cargo test -p turbospark-cli
cargo test -p turbospark-repack
cargo test -p turbospark-catalog
cargo test -p turbospark-server
cargo test -p turbospark-bench
cargo test -p turbospark-ffi

# The Swift bindings (macOS). `swift-lib` builds crates/ffi as a staticlib
# and copies it plus the canonical header into the SwiftPM package, which
# cannot reach outside its own directory to find either; both `swift test`
# targets below FAIL with a missing-header error without it having run.
make swift-lib

# Proves the HAND-WRITTEN turbospark.h matches the Rust side. Nothing else
# can: crates/ffi's own tests reach the same function bodies through the
# `rlib`, so they pass even against a wrong declaration in the header. It
# has already caught one real drift (the catalog's on-disk rows are
# snake_case where this binding's own wire shapes are camelCase).
make swift-test

# The same, plus the end-to-end arm against a real install: open, stream,
# cancel mid-generation, read the phase counters. Minutes. Without MODEL the
# real cases SKIP with a note rather than failing.
make swift-test-real MODEL=~/models/gemma4.gturbo

# BLOCKED is a SECOND install and reaches what MODEL structurally cannot:
# speculation resolves from the ARTIFACT, so one variable gates one shape and
# the refusal path was covered by nothing until it existed. Point it at a MoE
# or sub-4-bit install; the tests CHECK that it is one, because a merely
# headless dense install passes every other line while re-testing a case
# already covered (`crates/ffi/CLAUDE.md` Gotcha 11).
make swift-test-real MODEL=~/models/qwen38-27b-mtp.gturbo \
                     BLOCKED=~/models/ornith35b.gturbo

# The macOS app (`swift/TurboSparkApp`). `swift-demo` is an ALIAS for
# `swift-app` and `TurboSparkDemo` is gone; this used to describe a
# deliberately minimal demo and no longer does. What ships is multi-chat with
# persistence, a project/agent system that EXECUTES tools including a shell,
# a model hub with catalog install and Hugging Face probing, document
# attachment, and a phase-counter inspector.
#
# Both `make` targets depend on `swift-lib`, which touches every `.swift` file
# in both packages (`swift/CLAUDE.md` Gotcha 3), so each one pays a full Swift
# rebuild. Iterating on SwiftUI alone, call SwiftPM directly and skip it:
# `cd swift/TurboSparkApp && swift run TurboSparkApp`.
make swift-app
make swift-demo

# The RELEASE artifacts, and the only way to get a real `.app` out of this
# tree: `swift build` emits a bare executable, so the Info.plist, the bundle
# identifier and the resource-bundle copy all live in the script rather than
# in an Xcode project (`swift/CLAUDE.md` Gotcha 12). `dmg` additionally MOUNTS
# what it built and asserts the contents, because `hdiutil create` exits 0
# over an incomplete staging directory. Both land in `dist/` (gitignored) and
# both are what `.github/workflows/release.yml` calls, so a local run and a
# release build the same bytes. Minutes each. See `docs/RELEASE.md`.
make app-bundle
make dmg

# Formatting check (must stay clean; enforced in verification).
cargo fmt --check

# Apply formatting.
cargo fmt

# Lint the workspace and its tests (must stay clean).
cargo clippy --workspace --tests

# Does this still build OFF macOS? Nothing above asks: every command here
# compiles the macOS arm of every cfg, so a crate can declare a macOS-only
# DEPENDENCY while calling it unconditionally and stay green forever. Seconds,
# no download, target already installed. `--workspace` does NOT work: onig_sys
# (via tokenizer) and other cc-rs build deps need an x86_64-linux-gnu-gcc that
# is not installed here, so runtime/repack/catalog/server/cli/bench cannot be
# checked on this machine at all. These nine can, and are green. Run it when
# touching a cfg, a dependency table, or anything unsafe. See Gotcha 8.
# `turbospark-vision-io` is the one crate that is here by DESIGN rather than
# by luck: the whole point of splitting vision preprocessing out is that the
# decode, resize and position-table arithmetic lives somewhere a non-macOS
# build can reach, so a cfg or a dependency that drops it from this list is a
# bug in the crate rather than an accepted limitation.
cargo check --target x86_64-unknown-linux-gnu \
  -p turbospark-core -p turbospark-compute -p turbospark-model-io \
  -p turbospark-streaming -p turbospark-selection -p turbospark-invocation \
  -p turbospark-window-fit -p turbospark-gpu -p turbospark-vision-io \
  -p turbospark-image

# Run the CLI (validates the invocation; on macOS also attempts real
# generation against --model in all three modes: --prompt (raw text),
# --messages-file (JSON conversation, chat template applied), and --chat
# (interactive REPL). See DEVIATIONS.md for scope.
cargo run -p turbospark-cli --bin turbospark-check -- --model /path/to/model --prompt "hi"

# `--help` lists every flag with its default; `--version` prints the
# workspace version. Both short-circuit at the token they are reached at, so
# neither needs `--model`, and `turbospark-server` answers both too.
cargo run -p turbospark-cli --bin turbospark-check -- --help
cargo run -p turbospark-cli --bin turbospark-check -- --version

# `--max-context` defaults to `auto`: the checkpoint's own trained context
# (`arch.trainedContext` in the install's manifest), capped by what memory
# holds, and 4,096 when the install declares none -- which is every install
# written before that field existed, so nothing on disk changed footprint.
# The startup line prints the resolved window, the model's own, the KV bytes
# and what `auto` would have chosen. Exceeding the trained context WARNS;
# exceeding what memory holds is REFUSED with the subtraction (Gotcha 55).
cargo run --release -p turbospark-cli --bin turbospark-check -- \
  --model gemma4 --messages-file /tmp/p.json --max-context auto

# Find, inspect and install models (docs/MODELS.md). `--model` above takes an
# ALIAS as well as a path, resolved against the store; an existing directory
# always wins, so nothing that used to work changes.
cargo run -p turbospark-cli --bin turbospark-model -- list
cargo run -p turbospark-cli --bin turbospark-model -- info gemma4

# What should THIS machine run? Ranks the curated table by whether it fits
# here and by how much is known about it. Offline by default: a row with a
# frozen `measured` block in models.json gets that peak, a row without one
# reports `unknown` rather than a guess. Two size columns, and they answer
# different questions -- ALLOCS is what `open()` allocates (slot cache + KV,
# what phys_footprint charges for) and ON DISK is the whole install, which
# STREAMS on an MoE and need not fit. See Gotcha 58.
cargo run -p turbospark-cli --bin turbospark-model -- recommend --context 8192

# `--probe` reads every row's header (~60 s, no weights), which is what turns
# the unknowns into arithmetic. `--discover` adds the most-downloaded GGUF
# repositories on Hugging Face, each gated through the SAME probe, so a
# discovered row is refused in the same words `probe` would use.
cargo run --release -p turbospark-cli --bin turbospark-model -- recommend --probe
cargo run --release -p turbospark-cli --bin turbospark-model -- recommend --discover 20

# The header-only probe: what this engine makes of an arbitrary HF repo,
# reading KB rather than GB. Reports the architecture verdict (with the
# registry's own `needs` clause for a recognized-but-unported one), the block
# types against the kernels that exist, the EXPERT-SLOT ARITHMETIC that
# decides whether a model fits at all (Gotcha 36), and which tokenizer
# sidecars the repo actually has. Exits 0 only if it would run.
cargo run -p turbospark-cli --bin turbospark-model -- probe Qwen/Qwen3-30B-A3B-GGUF \
  --file Qwen3-30B-A3B-Q4_K_M.gguf --sidecar-repo Qwen/Qwen3-30B-A3B

# Install. Streams the checkpoint a layer at a time (it is never written to
# disk whole) and CANNOT RESUME: a failure restarts the walk. Fetches and
# VERIFIES the tokenizer sidecars FIRST, so a wrong sidecar list costs seconds
# rather than a re-stream (Gotcha 47).
cargo run --release -p turbospark-cli --bin turbospark-model -- pull tinyllama
cargo run --release -p turbospark-cli --bin turbospark-model -- \
  pull --repo owner/name --alias mine --sidecar-repo owner/original

# The catalog's rot guard: does every row still describe a real artifact?
# Reads file lists, HEAD responses, and GGUF headers, including all split
# shards. Downloads no weight payloads.
# Run it after adding or editing a row and paste its published byte figure
# back -- `download_bytes` is the only fingerprint a `main`-pinned row has.
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture

# Run the server against a real install (macOS; one runner per process, so
# requests are served one at a time). It serves OpenAI
# `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`. Add
# `--bind tailnet` to bind this machine's Tailscale IPv4 address instead of
# loopback. It requires --api-key or TURBOSPARK_API_KEY and provides no TLS.
# `--model` takes a catalog ALIAS as well as a directory, through the same
# `catalog::resolve_model_arg` `turbospark-check` uses; the startup line
# prints what an alias resolved to.
cargo run --release -p turbospark-server --bin turbospark-server -- --model ~/models/gemma4.gturbo
cargo run --release -p turbospark-server --bin turbospark-server -- --model gemma4

# Speculative decoding on the server, with the CLI's grammar and meanings
# (`--speculative off|auto|N`, `--speculative-drafter auto|mtp|dflash`). Both
# are PROCESS-level, resolved once at open like the rate cap, because the
# drafter's state is allocated there. ONE thing differs from the CLI and it is
# inherent: acceptance is exact only at temperature 0, which is a property of
# the PROCESS there and of the REQUEST here -- so a sampled request falls back
# to the sequential loop silently, and since most clients send a non-zero
# temperature a server started this way speculates on a MINORITY of its
# traffic. The startup line says which drafter resolved and why.
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/qwen38-27b-dflash2.gturbo --speculative-drafter dflash

# The OLLAMA-compatible routes, for tooling that speaks only that. Framing is
# NDJSON (one JSON object per line, no [DONE]) rather than SSE, and `stream`
# defaults to TRUE here unlike every OpenAI-shaped route.
curl -s localhost:8080/api/tags
curl -sN localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}'

# Point an Anthropic-native client straight at it, no proxy in between.
claude --settings '{"env":{"ANTHROPIC_BASE_URL":"http://127.0.0.1:8080","ANTHROPIC_API_KEY":"unused","CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY":"true","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}}' \
  --model claude-turbospark-<canonical-model-id>

# TOOL-CALL GUARDRAILS, on by DEFAULT (`crates/server/CLAUDE.md` Gotcha 18).
# Rescues a call the decoder could not parse out of the raw text, checks a
# call's arguments against the schema the request itself sent, and re-asks
# ONCE with a nudge. Over `forge-guardrails` at `default-features = false`:
# 6 new packages, against the ~290 its proxy half would add and a duplicate
# `anyllm_translate` beside this crate's 0.16.
#
# THE ONE BEHAVIOUR CHANGE TO KNOW: a request carrying TOOLS is BUFFERED
# rather than streamed while this is on, because a verdict needs the whole
# turn -- a call worth rescuing is one the decoder did not parse, so it is
# indistinguishable from prose until the turn ends. Requests WITHOUT tools
# stream exactly as they always did, byte for byte. Turn it off to get the
# pre-guardrail path back.
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo --guardrails off

# Same server, portable scripted backend (canned responses; DEVIATIONS.md).
cargo run -p turbospark-server --bin turbospark-server -- <tokenizer-dir> [port]

# Run the throughput benchmark harness (scripted producer; see DEVIATIONS.md).
cargo run -p turbospark-bench --bin turbospark-bench -- <tokenizer-dir>

# Real-install benchmark (macOS): frozen community protocol against a real
# .gturbo install, reporting split prefill/decode tok/s and peak
# phys_footprint (the Swift-parity memory counter). Use --release.
# The CONTEXT WINDOW and the GENERATION BUDGET are resolved from the
# install's own family and printed in the header, the same pair the oracles
# take (`real_model::protocol_parameters`): 4,096/1,024 for gemma4, qwen36
# and qwen3moe, 8,192/1,024 for the dense `llama` half, 8,192/3,072 for
# gpt-oss, 8,192/2,048 for museGlimmer. Read a peak or a tok/s row WITH those two numbers -- a dense
# `llama` number taken before 2026-08-12 is at the old shared 4,096, where
# `long-synthesis` did not fit at all.
cargo run --release -p turbospark-bench --bin turbospark-bench -- --model ~/models/gemma4.gturbo

```

Everything above is what a session needs by default. The per-family gate
matrix, the checkpoint installs, the power harness, the cross-engine KL
arms and the header probes are one link away, in files that load when you
follow them rather than on every task:

- [.claude/docs/model-gates.md](.claude/docs/model-gates.md): per-family memory oracles and quality gates.
- [.claude/docs/checkpoint-installs.md](.claude/docs/checkpoint-installs.md): installing the real checkpoints.
- [.claude/docs/cross-engine-kl.md](.claude/docs/cross-engine-kl.md): this port against mlx-lm and llama.cpp.
- [.claude/docs/power-measurement.md](.claude/docs/power-measurement.md): watts and joules per token.
- [.claude/docs/diagnostics.md](.claude/docs/diagnostics.md): header probes, convention checks, decode-path probes.

### Real-model smoke (needs the pinned install)

Run BOTH of these on any change to the decode path, the output head, the
KV cache, or a Metal encode loop. Greedy alone is not a smoke test: it is
`argmax`, and `argmax` is invariant under every monotone transform of the
distribution, so it stays byte-identical to correct through bugs that
destroy sampling entirely (Gotcha 16).

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy. Catches broken math.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. SAMPLED, at the CLI defaults (T=0.2, top-k 64, top-p 0.95). Catches
#    distribution bugs greedy cannot see. Must stay coherent for the whole
#    run and reach EndOfTurn on a short question.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

A bare `--prompt` on an instruction-tuned model babbles: that is the chat
template missing, not a decode bug. `--messages-file` applies it for you.

`--reasoning off|low|medium|high|xhigh` (default `off`) asks the checkpoint's
own template to think first; on a dialect whose reasoning is separable the
ANSWER goes to stdout and the REASONING to stderr, so `>/dev/null` keeps the
reasoning and `2>/dev/null` keeps the answer. Read Gotcha 55 before reading a
level as a model property: `off` is not "the model's default", it is thinking
disabled, and the vendor's advertised default is what a caller gets by
sending nothing at all.

Add each new crate directory to the `members` list in the root `Cargo.toml`
as it lands, and keep the member list in sync with the directories under
`crates/`.

A `Makefile` wraps the common cases: `make build-debug`, `make
build-release`, `make test-debug`, `make test-release`, `make fmt`, `make
fmt-check`, `make clippy`, `make check` (fmt-check + clippy + test-debug),
`make clean`, `make install` (installs all binaries to `~/.local/bin` by default;
configurable via `PREFIX` or `BINDIR`), and `make uninstall`.

## Gotchas

1. Downstream crates depend on `turbospark-core` under an alias, for example
   `foundation = { package = "turbospark-core", path = "../core" }`, and refer to it as
   `foundation`. The same aliasing pattern is used for every intra-workspace
   dependency (`compute`, `selection`, `tokenizer`, `model_io`, `runtime`,
   `invocation`) so the alias, not the crate's real package name, is what
   integration tests and downstream `src/` code import by.
   **BUT THE ALIAS IS PER CRATE, AND DEV-DEPENDENCIES ARE THE EXCEPTION.**
   `crates/bench` aliases repack to `repack` while `crates/runtime`'s
   dev-dependency is unaliased, so its tests import `turbospark_repack` --
   22 call sites to bench's 3. Read the crate's own `Cargo.toml` rather than
   carrying a name across from the file you were just in.

2. Runtime knobs are validated against const allowed-value sets, and a default
   drifting out of its own set is a compile error rather than a runtime
   surprise. Moved to [crates/core/CLAUDE.md](crates/core/CLAUDE.md) Gotcha 1.

3. FP16 is the `half` crate, BF16 is a hand-rolled bit-shift pair. Do not hand-roll FP16. Moved to
   [crates/compute/CLAUDE.md](crates/compute/CLAUDE.md) Gotcha 1.

4. Token ids cross crate boundaries as signed 32-bit integers
   (`pub type TokenId = i32`). Keep that interchange width when wiring
   downstream crates.

5. `Cargo.lock` is committed on purpose. This workspace targets command-line
   and server binaries, so the lockfile stays in version control for
   reproducible builds. Do not delete or gitignore it.

6. The cargo build output lives in `/target` and is gitignored. It is large;
   never commit it.

7. `crates/cli` (`turbospark-check`) is the process entry point over `turbospark-invocation`'s pure parse. Moved to
   [crates/cli/CLAUDE.md](crates/cli/CLAUDE.md) Gotcha 11.

8. **A PLATFORM CLAIM NOBODY RUNS IS A COMMENT, NOT A GATE.** `crates/gpu`
   is meant to be `#[cfg(target_os = "macos")]` throughout, so a non-macOS
   build compiles it to nothing. That sentence was FALSE the first time
   anyone checked it (2026-08-15), twice over: `crates/streaming` declared
   `libc` under `cfg(target_os = "macos")` while calling
   posix_memalign/free/sysconf unconditionally, and `gpu`'s
   `mod dequant_int4_gemv;` had lost the gate its ~30 siblings carry.
   Neither was reachable from any command in the verification policy, which
   is why the cross-target `cargo check` above now sits beside them. Run it
   when touching a cfg, a dependency table, or anything unsafe.

   **THE DEPENDENCY TABLE IS WHERE PORTABILITY IS REALLY DECIDED**, not the
   `cfg`s in `src/`. `crates/runtime` declares `model_io`, `gpu`, `compute`
   and `streaming` under `[target.'cfg(target_os = "macos")'.dependencies]`,
   so it is macOS-only however portable its source reads, and any crate
   taking a dependency on it inherits that. Read the target's
   `[target.'cfg(...)']` block before adding a cross-crate edge: this is
   what forced the two sizing policies down into `model-io` rather than
   letting `catalog` reach up. `turbospark-vision-io` is in the cross-target
   list by DESIGN rather than by luck, so a cfg or dependency that drops it
   from that list is a bug in the crate rather than an accepted limitation.

   The kernel inventory lives in
   [crates/gpu/CLAUDE.md](crates/gpu/CLAUDE.md); `DEVIATIONS.md` has the
   wired versus unwired list.

9. `crates/model-io`, `crates/streaming` and `crates/ffi` are the three
   crates that intentionally carry unsafe code and platform `cfg`s (mmap in
   `model-io::resident_buffer`, the macOS `F_RDADVISE` `fcntl` in
   `streaming::rdadvice`, and the raw destination pointers plus borrowed
   `RawFd` that `streaming::read_pool`'s parked worker threads use --
   sound only because `run_batch` blocks until every claim is dropped).
   **`crates/ffi` is unsafe by definition rather than by exception**: it IS
   the C ABI, so every entry point takes raw pointers, and the rule that
   keeps it honest is that each `extern "C"` body is a call to `abi::guard`
   and nothing else. Unwinding across the FFI boundary is undefined
   behaviour and this workspace cannot opt out of unwinding (the root
   `Cargo.toml` records why `panic = "abort"` must stay off: two `Drop`
   impls are load-bearing on the unwind path), so a panic that escapes
   `catch_unwind` there is a real hazard rather than a theoretical one.
   `compute`, `core`, `repack`, `runtime`, and `tokenizer`
   have `#![forbid(unsafe_code)]`. `gpu`, `invocation`, `selection`,
   `server`, `window-fit`, `cli`, and `bench` currently have no such
   attribute and no workspace-level lint enforces it, so unsafe code is not
   actually compiler-blocked there today, even though none uses any.

10. `crates/runtime`'s `LogitProducer` trait has `RealForwardRunner` (macOS/GPU
    only) as its real GPU-forward-pass-backed implementation, while
    `ScriptedLogitProducer` (a fixed replayed logit sequence) is what unit
    tests and `crates/server`'s `ScriptedChatModel` drive the raw-completion
    loop with on any platform. `crates/server`'s `RealChatModel` drives it
    with `RealForwardRunner` on macOS (`turbospark-server --model`).
    Chunked prefill is wired regardless (`ChunkedPrefillRunner`,
    `run_raw_completion_chunked`) -- `ScriptedLogitProducer` implements it
    by consuming one scripted step per chunk; see `DEVIATIONS.md`.

11. Tokenizer fixture gotcha: the vendored test fixtures under
    `crates/*/tests/fixtures/{ChatMLTokenizer,DeepseekTokenizer}` embed
    high placeholder token ids (e.g. `248044`) in their `tokenizer.json`
    `added_tokens` list, but the `tokenizers` crate's loader renumbers
    added tokens sequentially after the base vocab (which has only 258
    entries in these toy fixtures) -- so the *actual* ids only exist at load
    time. Never hardcode a token id from reading the fixture JSON; resolve
    it from a loaded `MfTokenizer` (`token_to_id`, `end_of_turn_id`, etc.)
    instead.

12. **The DECODE FLOW is chosen by `ArchConfig.family`, never by tensor
    naming, and picking a neighbour's flow yields fluent WRONG output rather
    than an error.** Gemma 4 and Qwen 3.6 both carry
    `language_model.model.embed_tokens.weight`, so a naming probe cannot
    tell them apart. `RealForwardRunner` is macOS/GPU only, supports dense
    and MoE FFN with resident or streamed experts, and refuses compressed
    (mask 3/4) layers at `open()`.

    The per-family flows, their state types, the env seams
    (`TURBOSPARK_SHARED_CB`, `TURBOSPARK_ROUTED_PIPELINE`,
    `TURBOSPARK_DISPATCH_PROFILE`), the expert-cache slot policy and the
    synthetic install builders are documented where they live:
    [crates/runtime/CLAUDE.md](crates/runtime/CLAUDE.md) Gotchas 3, 7 and
    13, and [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotchas 1 and
    3 for what synthetic fixtures can and cannot prove. Expert
    prefetch and speculation are a measured dead end; see `DEVIATIONS.md`
    before re-deriving them.

13. `st` (the ripgrep-alike used here) skips gitignored files, so it finds
    nothing in a gitignored one. **`ROADMAP.md` IS NOT ONE OF THEM, whatever
    this entry used to say.** It is tracked and has been for as long as the
    log goes back (last touched by `612be53`), it is not in `.gitignore`, and
    it IS carried into a worktree -- checked directly on 2026-09-05, because
    the claim was load-bearing for where to edit the file and was simply
    false. `CLAUDE.local.md` is the genuinely gitignored one, and everything
    below about worktrees applies to it alone. `st` also needs a per-tree
    index, so a fresh worktree answers every query with "no index found"
    until `st index` has run once.
    **AND NEITHER IS UNCOMMITTED WORK.** A worktree opened for a task whose
    subject sits unstaged in the main checkout starts EMPTY at HEAD, with
    none of the code the task describes -- which reads like the task being
    wrong rather than the tree being empty. Run `git status` in the MAIN
    checkout before accepting a worktree, and work there when the subject
    is uncommitted.
    **A SPAWNED BACKGROUND TASK GETS ONE OF THESE.** A chip started while your
    fix is uncommitted reinvents it against HEAD: on 2026-08-22 one reached the
    same design independently and asserted strings the same session had just
    reworded. Save the work before discarding such a tree
    (`git diff > /tmp/<name>.patch`); `git worktree remove --force` is the only
    way to remove a dirty one and it keeps nothing.
    **STAGING A CONTESTED FILE CAN COMMIT HALF A FEATURE, AND THE TELL IS A
    BUILD ERROR NAMING A SYMBOL YOU NEVER TOUCHED.** On 2026-08-30
    `AppModel+Persistence.swift` carried a chat-persist debounce. Its property
    lives in the uncommitted `AppModel.swift`. Committing the first alone
    produced `cannot find 'chatPersistDebounceTask' in scope`. Reconstruction
    (above) is the fix. Adding the sibling usually is not, because it drags in
    whatever else that file is mid-flight on. `AppModel.swift` would have
    brought the entire unfinished server surface with it.
    **AND VERIFY SUCH A COMMIT BY ERROR-COUNT DELTA AGAINST `main`, NOT AGAINST
    ZERO.** A tree carrying a hundred uncommitted changes is green only as a
    whole. `main` may not build at all: it read 72 errors that day, none of them
    anyone's current work. Build a detached worktree at your commit and at
    `main`, then compare counts. Equal means you added nothing. Reading the
    absolute count as damage you caused wastes a cycle, and chasing it into
    another session's refactor wastes several.

14. Adding one flag to `crates/invocation` touches FIVE places, and a missing parser arm is a runtime panic rather than a compile error. Moved to
    [crates/invocation/CLAUDE.md](crates/invocation/CLAUDE.md) Gotcha 1.

15. `families/gemma4/mod.rs`'s per-token function interleaves
    `let real = self.real.as_ref()` bindings with `&mut self` calls. A new
    `&mut self` call between such a binding and its last use is E0502;
    re-bind `real` after the call (the file already does this repeatedly).
    The same shape recurs in every other family's `mod.rs`.

16. **`LogitProducer::produce` writes LOGITS, never probabilities.**
    `selection::select` softmaxes whatever it is handed. A producer that
    also normalizes makes it `softmax(softmax(z))`, which over V=262144
    collapses to near-uniform: top-p/top-k still rank correctly (softmax is
    monotone) but the temperature reweight is destroyed, so sampling
    degenerates into a coin flip among the surviving top-k. GREEDY LOOKS
    FINE THROUGHOUT -- `argmax` is monotone too -- so a greedy-only smoke
    test proves nothing here. This bit the Gemma 4 head once: it dispatched
    the fused `logit_softcap_softmax`, mirroring Swift's kernel, but Swift
    samples on the GPU from probs while this port samples on the host from
    logits. The head now dispatches the cap alone
    (`utility.metal`'s port-local `logit_softcap_fp16`) and returns
    `softcap * tanh(z / softcap)`, which is also what HF's
    `*ForCausalLM.forward` returns. Guarded by the softcap-bound assertion
    in `crates/runtime/tests/real_forward_gemma4.rs` and the
    does-not-normalize assertion in `crates/gpu/tests/utility_and_pass.rs`.
    Rule for any new model: decide where the normalization lives ONCE, put
    it in the sampler, and never in a producer.

17. **Wrap every repeated Metal encode in `gpu::autorelease_pool`.**
    `MTLCommandQueue.commandBuffer` and
    `MTLCommandBuffer.computeCommandEncoder` return AUTORELEASED objects.
    The `metal` crate's `to_owned()` adds our retain and drops it, but the
    pool's retain survives until the pool drains -- and a plain Rust binary
    has exactly one pool, around `main`. Without an inner pool every
    command buffer the process ever created stays alive to exit: measured
    at ~6 KiB per command buffer, 31 per token, ~180 KiB per decoded token,
    linear and unbounded. It reads as "memory grows with prompt length"
    because longer prompts mean more `produce` calls.
    `RealForwardRunner::produce` opens one pool per token. Any new decode
    loop, prefill path, or benchmark that encodes in a loop needs the same.
    Caught by `memory_oracle.rs`'s steady-state guard, not by
    `gpu_buffer_allocations()` -- these are not our allocations.

18. **`KvCacheManager::new`'s `fp16_ring_enabled` is not cosmetic, and its
    comment can lie.** Passing `false` gives every sliding-window layer a
    full `max_context` buffer. On real Gemma 4 (25 SWA layers of 30, 1024
    window, 4096 context) that is 922 MiB of KV instead of 320 MiB, and it
    is invisible in output correctness -- a linear layout is simply a ring
    big enough to never wrap, so nothing fails, the process is just fat.
    The runner enables it, sizes SWA layers at
    `min(max_context, sliding_window + 128)`, and passes
    `ring_capacity(layer)` into `encode_attention_decode`, which
    specializes `FC_ATTN_RING_CAP` into the pipeline. Two traps in that
    last step: 0 must keep the identity addressing full-attention layers
    need, and the capacity MUST go into the pipeline-cache constants key
    or ring dispatches silently reuse the linear pipeline. When adding a
    model, derive the ring flag from the layer mask, not from an
    assumption about the family, and verify with a prompt plus generation
    longer than the ring -- coherent text past position `sliding_window +
    128` is the proof.

19. `phys_footprint` counts the resident weight mapping, and the slot term dominates it. Moved to
    [crates/bench/CLAUDE.md](crates/bench/CLAUDE.md) Gotcha 1.

20. **The FIRST timed run after a build is a cold GPU, and it is not a
    baseline.** `TURBOSPARK_PHASES=1`'s `gpu busy` buckets come from
    `GPUStartTime`/`GPUEndTime`, so they look like pure device time and
    invite being trusted as-is. They are not clock-invariant: the first
    run on an idle GPU executes at low DVFS clocks. Measured on the real
    26B install, identical settings, same prompt: cb1 read 7.98 ms/token
    cold and 5.20 ms/token on every warm run after it. That is a 53%
    error, far larger than any single change this port has landed, and it
    silently inflates whatever you measure first -- which, in an A/B, is
    usually the baseline. Always discard at least one warmup run, then
    interleave the variants pair by pair (the existing rule for tok/s in
    CLAUDE.local.md; it applies to the GPU-busy buckets too, for a
    different reason). A corollary for reading old notes: a phase number
    with no warmup discipline recorded against it may be a thermal
    artifact, so re-measure before building on it.

21. **Every `TURBOSPARK_PHASES=1` number is divided by ALL forward passes,
    prefill included.** `print_phases` uses `p.calls`, and prefill runs
    one `produce` call per prompt token, so a run with a 2252-token
    prompt and `--max-new 150` puts 94% of its divisor at short context.
    A phase number is therefore an average over the run's whole context
    range, never a number at the final context, and labelling one "~2300
    context" is wrong by roughly 2x. To get a number AT a context,
    difference two runs: `total(N) = per_token * calls`, and
    `(total(N2) - total(N1)) / (calls2 - calls1)` is the marginal cost
    over that range. This is not academic -- it is why the 2026-08-05
    session measured split-KV as a no-op and reverted a change that is
    actually worth 25% of decode throughput at 800 context (see
    DEVIATIONS.md's split-KV entry). A corollary for A/B work: use a
    SHORT prompt and a long generation, so the divisor is decode.

22. Record the power source next to any absolute number, and never A/B across sessions. The axis that moves is THERMAL HEADROOM, not energy. Moved to
    [crates/bench/CLAUDE.md](crates/bench/CLAUDE.md) Gotcha 3.

23. **Every profiling surface in this repo measures the inside of
    `produce`. The decode loop is bigger than that.** `TURBOSPARK_PHASES=1`,
    its GPU-busy attribution, and `TURBOSPARK_DISPATCH_PROFILE=1` all live in
    `RealForwardRunner`, so the sampler, the streaming detokenizer, and the
    stop matcher -- everything `run_raw_completion` does AFTER `produce`
    returns -- appear in none of them. This is not hypothetical: it hid a
    ~18.9 ms/token full sort in `selection::select` (V=262144) for the
    whole life of the port, which was the entire measured 1.5x decode gap
    against Swift (`docs/BENCHMARKS.md`). The check that catches it is
    arithmetic and takes one subtraction: **the phase report's own total
    must come out near the footer's `decode=` seconds.** It read 26.1 s
    against 41.3 s and nobody had compared them. Do that comparison before
    concluding a phase table accounts for a run. Two corollaries: the
    greedy smoke's `--temperature 0.0001` is NOT the argmax fast path
    (`is_deterministic` is `temperature == 0.0` exactly), so it exercises
    the sampler like any other run; and when adding a phase bucket, prefer
    widening the timed region over adding another bucket inside it.

24. **A manifest that OMITS a family-extension field is validated against
    GEMMA's value for it, whatever family it claims.**
    `arch_validation.rs` resolves every optional `arch` field with
    `.unwrap_or(gemma_defaults.<field>)`, so a Qwen install that leaves out
    `attnOutputGate` / `ffnSandwichNorms` / `ropeNeoxSubdim` / the five
    `linear*` fields can never load: each one compares against Gemma's.
    `gturbo_writer.rs::build_manifest_json` therefore writes all of them
    UNCONDITIONALLY (Gemma installs are unaffected -- those are exactly
    Gemma's fallbacks). Two corollaries. First, `manifest_peek.rs` has to
    resolve the family FIRST and start from `known_architecture(family)`,
    not from a Gemma baseline. Second, the float fields are compared with
    `!=` on `f64` and serde_json's default parser is only accurate to ~1
    ULP (exactness is behind its `float_roundtrip` feature), so any
    `attentionScale` that is not a binary fraction fails to round-trip:
    the real families' 1.0 / 0.0625 / 2^-4.5 are fine, an invented
    `32^-0.5` is not (it cost a red test in the Qwen session).

25. `gdn_qk_norm` and `gdn_gated_norm` are correct at EXACTLY 128 threads per threadgroup. Moved to
    [crates/gpu/CLAUDE.md](crates/gpu/CLAUDE.md) Gotcha 5.

26. Qwen's `linear_attn.A_log` and `dt_bias` carry NO `.weight` suffix, and its routed marker differs from Gemma's. Moved to
    [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotcha 15.

27. A layer's routed slots dispatch in the ROUTER'S RANKING, because slot order is summation order. Moved to
    [crates/runtime/CLAUDE.md](crates/runtime/CLAUDE.md) Gotcha 8.

28. **Thermal pressure silently rewrites BOTH throughput and energy, and
    nothing in the standing gate looks at it.** Every existing harness
    here records the power source (Gotcha 22) and none records
    `Current pressure level`. Measured 2026-08-07 on battery, same binary
    and prompt, Gemma `medium-review`: a Nominal run decoded 39.34 tok/s
    at 18.58 W and 0.4568 J/token, while a Heavy-pressure run of the SAME
    case decoded 31.47 tok/s at 10.21 W and 0.3169 J/token. Note the
    direction, because it is a trap: throttling made the run SLOWER and
    simultaneously more energy-efficient per token (voltage-frequency
    scaling is superlinear), so a throttled arm does not look broken in a
    power table, it looks GOOD. In a tok/s table it just looks like a bad
    sample. THIS IS A FUNCTION OF THE INSTALL'S WATTAGE, NOT OF THE POWER
    SOURCE -- the 2026-08-07 session read it as a battery phenomenon
    because every install it measured drew 14-18 W, and at that draw the
    contrast is sharp: the protocol's `long-synthesis` case left Nominal
    on every battery run of both installs (it prefills ~3,000 tokens for
    63-72 s before decoding anything), while the SAME binary running the
    SAME protocol on AC held Nominal on 50 of 50 sampled arms. Then
    gpt-oss-20b (2026-08-12), then the highest-wattage install here at ~36 W
    combined / ~32 W GPU, dropped one of two AC decode windows to Heavy,
    and the throttled arm read 3.7% BETTER J/token than the clean one --
    the same trap, now reachable on AC. **`muse_glimmer` (2026-08-16) is the
    limit case and takes the wattage record**: ~38 W combined on a quiet
    machine, and it saturated on EVERY measured arm of three captures, so its
    governed J/token is all the harness can produce and it wanders 25% run to
    run. Its one stable reading is the WARMUP window, which the summary
    excludes by design -- so on a hot enough install the only publishable
    number is the one the protocol throws away (`docs/POWER_BASELINE.md`).
    **THAT LAST CLAUSE IS NO LONGER TRUE, AND THE FIX IS EXTERNAL TO THIS
    REPO.** `scripts/power.sh COOLING=max` pins the fans through
    ThermalForge (MIT, a root LaunchDaemon, not a dependency of this
    workspace) for the duration of a capture. Measured 2026-08-18 on the
    same install and case: 12 of 12 measured rows Nominal where every
    performance decode had gone Heavy, the performance arm's J/token spread
    25% -> 2.0%, its tok/s reproducing to 0.08%. So a saturating install is
    a MISSING EXPERIMENTAL CONDITION rather than an unmeasurable one, and
    the thing to reach for is cooling, not a longer idle (30 minutes of it
    moved the warmup 1.4%).
    Two caveats that keep this from being a free upgrade. A pinned-fan row
    is an UPPER-HEADROOM operating point that no user occupies, so it is
    published beside the uncooled rows and never instead of them; the
    `cooling` column in `rows.tsv` and the `system.txt` provenance line
    exist so no row can silently be one. And cooling cannot be interleaved
    the way `ARMS` can -- it is a property of the whole capture -- so the
    cooled-vs-uncooled delta is cross-capture and carries Gotcha 22's
    caveat, while the arms measured INSIDE one cooled capture do not.
    Worth recording that pinning fans made J/token BETTER (1.6176 against
    the unconstrained 2.0316) where the prediction was that it would be
    worse: "a cooler chip boosts higher" needs headroom to boost into, and
    at an already-unconstrained point what cooling removes is leakage.
    Two further things that capture settled. The governed point is not a
    fixed discount: six governed decodes ran 18% to 35% below the
    unconstrained one, so the DIRECTION reproduces and the magnitude does
    not. And a `--power-profile efficiency` arm on the same install held
    Nominal on 6 rows of 6 where performance went Heavy on 3 of 3, which
    inverts the natural guess -- capping the rate keeps a machine OUT of
    thermal governance, so the cap is sometimes the only way to get a
    repeatable power number at all. (That ~36 W is itself ~3 W of
    background load; the install's own clean draw is ~33 W. See Gotcha
    43, which is the other half of this one: pressure is not the only
    thing that silently rewrites a power row, and the Nominal check
    cannot see the other.)
    `scripts/power.sh` WARNS on any run whose pressure leaves Nominal but
    its summary still averages that run in -- exclusion is by hand, from
    the per-run rows in `rows.tsv` (read the awk, not the warning:
    warning and excluding are different things); `scripts/parity.sh`
    and the oracles do not even warn, so a surprising throughput row from
    a long session on either power source is worth checking against the
    pressure level before it is believed. **`pmset -g therm` is NOT that
    check and never was**: it reports thermal WARNING LEVELS, and on this
    machine it prints three `Note: No ... has been recorded` lines and
    nothing containing the word "pressure", so a
    `pmset -g therm | grep -i pressure` matches nothing and reads as a
    clean run rather than as an absent instrument (it did exactly that
    through every run of the 2026-08-17 qwen38 capture). The two real
    sources are `powermetrics -s thermal`'s `Current pressure level:`
    line, which is what `scripts/power.sh` already parses and which needs
    sudo, and `runtime::power::thermal_level`'s four-level
    `NSProcessInfo` enum, which does not. The
    corollary for A/B work is stronger than "prefer AC": an effect
    smaller than a few percent CANNOT be measured on battery at all. The
    read-pool QoS seam read as a clear loss on battery (one pair at +8.8%
    energy) and as a null result on AC, and the AC reading is the correct
    one (`docs/POWER_BASELINE.md`).

29. **A GGUF IS NOT A SAFETENSORS CHECKPOINT WITH DIFFERENT NAMES, AND
    EVERY DIFFERENCE THAT BITES IS SILENT.** Each produces a plausible,
    non-crashing wrong answer if assumed. The data region is ALIGNED rather
    than adjacent; dims are stored fastest-varying FIRST, so a logical
    `[out, in]` matrix sits on disk as `[in, out]` and getting it wrong
    transposes every shape while leaving every byte correct; Gemma fuses
    gate and up into one routed tensor where MLX keeps them apart, with gate
    the FIRST half (measured, not assumed); and `general.architecture` is
    the CONVERTER's name rather than the family's.

    **Whether a GGUF install loads is decided PER BLOCK TYPE, not per
    format**, and a type needs up to three kernels (resident GEMV, embedding
    lookup, routed pair) before it runs. The two gates are independent on
    purpose: `model_io::validate_quant` reads the manifest's per-slot
    `ggmlType`, and `RealForwardRunner::open` reads the resident index's
    dtype TAGS, which is the backstop for a hand-edited manifest. They are
    TWINS AND NOT COPIES: MXFP4 is executable in the first sense and not the
    second, so it passes the manifest gate and is stopped by the backstop.

    The per-type kernel matrix, the transcode decisions and the
    V-head convention live in
    [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotchas 4, 6, 7, 9
    and 17, and in [crates/model-io/CLAUDE.md](crates/model-io/CLAUDE.md).

30. **A routed expert row can be ALL ZEROS in a real checkpoint, and
    `pearson` returns 0.0 on a constant input by design.** Measured
    2026-08-08 while cross-checking the Q4_K reference: 11 of the first 16
    rows of layer 0 expert 0's gate are identically zero in the real
    `Qwen3.6-35B-A3B-Q4_K_M.gguf`, and in the MLX-derived `.gturbo` install
    of the same model. So a correlation check that averages over a fixed row
    range silently divides a good result by the number of dead rows: the
    first run of `gguf_q4_k_network.rs` read +0.2479 against a +0.95 bar,
    which is 0.9916 (one real row) divided by four. THE SHAPE OF THE
    DIAGNOSTIC IS THE REUSABLE PART, because the number looks exactly like a
    broken unpacker: correlate PER ROW rather than over the pooled range,
    and where two independent readers agree a row is constant, that is the
    data rather than a bug in either (a 1152-byte Q4_K run and a 1024-byte
    INT4 run plus scale planes cannot land on the same zero set by
    accident). The test now selects non-constant rows and asserts the two
    sides agree on which those are. `pearson`'s own doc says it returns 0.0
    on a constant input so a caller's threshold behaves; that is correct and
    is what made the failure look like disagreement instead of NaN.

31. **A candidate transform must be tested INVARIANT TO ORDERING, or a
    correct one reads as a decisive rejection.** Measured 2026-08-08 while
    settling Qwen's GGUF conventions: `-exp(A_log)` is exactly what
    llama.cpp stores, and the first two element-wise checks of it scored a
    worst relative error of 2081 and 2.44, which look like proof it is
    wrong. The tensor was ALSO permuted, and a permutation defeats an
    element-wise comparison whatever the transform. Sorting both sides first
    reduced the same data to 0.003088 at correlation +1.00000 (0.003 being
    BF16 rounding). So when comparing a candidate against a reference:
    compare sorted, decide the transform, THEN recover the permutation.
    The permutation is recovered by printing an index map, but note its one
    failure mode -- matching by VALUE finds the first equal element, so on a
    tensor with repeats (Qwen's `conv1d.weight` has 32,768 values and many
    duplicates) the map reports an identity prefix and then noise. That is
    the matcher, not the data. Compare per row or per channel there.

32. `PassEncoder` ends encoding on drop, and that is load-bearing rather than tidy. Moved to
    [crates/gpu/CLAUDE.md](crates/gpu/CLAUDE.md) Gotcha 6.

33. A source-convention difference belongs to an AXIS, not to the tensors you could compare. Moved to
    [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotcha 7.

34. **A cross-engine comparison needs the BACKEND matched, not just the
    bytes, and getting it wrong reads as a defect in your own engine.**
    ROADMAP Phase G's last gate clause was closed by running llama.cpp on
    the exact GGUF this port installed (`scripts/kld_llamacpp.py`). The
    first reading, against llama.cpp on CPU, was 0.05838 mean nats at 95.1%
    top-1 -- 40x the shape floor measured beside it, which is what a real
    kernel bug looks like. It was not one. ggml's own CPU and Metal paths
    disagree with EACH OTHER by 0.05510 nats and 4.9% of the argmaxes on
    this model, and re-running the reference on Metal (which a 26.9 GB model
    does fit under, contrary to the wired-limit guess that put it on CPU)
    collapsed the number to 0.00845 at 98.2%. The whole apparent gap was in
    the reference. So a cross-engine KL needs TWO floors, not one: the shape
    floor `kld.py` established (batched vs cached, 0.00134 here) and a
    backend floor, which was 41x larger. The generalisation past ggml: any
    reference engine with more than one arithmetic backend has this axis,
    and it is invisible unless measured, because both arms are "the same
    engine on the same file".
    Two things it settled on the way, worth not re-deriving. llama.cpp DOES
    apply Gemma's `final_logit_softcapping` (max |logit| 29.9993, so the
    heads are the same function and comparable), and its CPU and Metal
    perplexities differ by 1.8% on identical bytes, which is a real
    calibration for `quality_common`'s 2% `PERPLEXITY_REL_TOLERANCE`.

35. **A measurement tool must not inherit an implicit power default, and the CLI/bench asymmetry that follows is deliberate.** ROADMAP Phase P2 added `--power-profile` and `--max-tokens-per-sec` to `turbospark-check`, `turbospark-server` and `turbospark-bench`. The first two call `runtime::resolve_profile`, which asks the OS about Low Power Mode and selects `efficiency` when it is on and no profile was named -- correct for a user-facing binary. `turbospark-bench` does NOT: it defaults to `performance` and changes only when told. The failure it avoids is quiet and total. Phase P2's own gate is an interleaved A/B of `performance` against `efficiency` under `scripts/power.sh`; run it on an LPM-enabled machine with the implicit default and the `performance` arm is silently capped at reading speed too, so both arms pace identically, the J/token difference collapses, and nothing in `rows.tsv` records that the arm label stopped meaning anything. The same reasoning is why both memory oracles and both quality gates pass `RateControl::default()` explicitly -- an oracle asserts a decode tok/s FLOOR, so any inherited cap turns a green run red for a reason that is not a regression. The general form: when a knob has an environment-sensing default, the harness that measures the knob is exactly the caller that must not sense.
36. **"Is it MoE?" is the wrong question for this engine. "How FINELY does it
    split its experts?" is the right one, and it is one multiplication off the
    header.** The expert slot cache is `slots x layers x expert_stride`, so
    what decides whether a checkpoint can stream is the size of ONE expert,
    not the size of the model. Measured 2026-08-09 while bringing up the
    `llama` family (ROADMAP Phase M2):

    | | Gemma 4 26B-A4B | Mixtral 8x7B | Qwen3-30B-A3B |
    |---|---|---|---|
    | experts per layer | 128 (top-8) | 8 (top-2) | 128 (top-8) |
    | one expert blob | ~3.2 MiB | **108.9 MiB** | 2.5 MiB |
    | slot cache at 16 slots | 1.5 GiB | **54.5 GiB** | 1.90 GiB |

    Mixtral is the SMALLER model by parameter count and cannot stream on this
    engine in any useful configuration: at the default 16 slots it wants
    54.5 GiB of pinned host memory, and at `slots == num_experts` it pins the
    entire 27.2 GiB expert table, which is not streaming. `open_expert_streamers`
    now caps the slot count at the expert count and reports the working set
    when the streamer cannot get its memory, because "cannot allocate" without
    the number sends the reader looking for a leak instead of at the
    arithmetic.
    THE PART THAT GENERALISES IS WHEN IT WAS KNOWABLE: `expert_count` and
    `feed_forward_length` are both in the GGUF header that the Phase 0 probe
    already read, so `8 x 14336 wide -> ~109 MiB` was available before the
    26 GB download and was not computed. Header probes had answered every
    other question in that phase. Do this multiplication in Phase 0, next to
    the layer graph.
    THE THIRD COLUMN IS WHAT THE FINDING BOUGHT. `qwen3moe` was chosen by
    running that multiplication FIRST, and it is the same layer graph and the
    same decode flow as Mixtral -- so the flow Mixtral's bring-up paid for is
    what a streamable checkpoint now runs on. Note also what the table says
    about "small model, small working set": Qwen3-30B-A3B and Mixtral 8x7B are
    within a factor of two on parameters and a factor of 29 apart on the
    number that decides whether either one runs here.
    **RUN THE MULTIPLICATION THE OTHER WAY TOO, BECAUSE THE SLOT ALLOWLIST AND
    NOT FREE RAM IS WHAT CAPS RESIDENCY.** `ALLOWED_CACHE_SLOTS` tops out at
    32, so the largest cache any install can ask for is
    `32 x sum(expert_stride)` however much memory the machine has. On a
    fine-grained model that is a small FRACTION of the expert table:
    `qwen4_exp` is 48 layers at 2.7648 MB, so one slot costs 126.6 MiB, 32
    slots is 3.95 GiB, and that holds 11% of a 288-expert table or 6.25% of a
    512-expert one. A model can pass this gotcha's fit test and still be
    residency-starved by the allowlist, which is a const change plus whatever
    validates against it rather than a property of the checkpoint. Read
    Gotcha 64 with it: `top_k` sets a FLOOR of `2 * top_k` for the pipelined
    prefill path, so a top-10 model has only 24 and 32 legal today.

37. **A per-DIALECT constant standing in for a per-MODEL property is correct
    until the second model arrives, and it fails at the first decoded
    token.** `MfTokenizer::vocab_size` is resolved by chat DIALECT, and its
    own comment said what it really was: "the model's padded
    embedding/lm_head row count, not the tokenizer's actual vocab". ChatML's
    row read 248,320, which is Qwen 3.6's padded head. Qwen3-30B-A3B is also
    ChatML and pads to 151,936, so bringing it up produced
    `vocab mismatch: model has 151936, caller expected 248320` -- with a
    correct install, a correct manifest and a correct decode flow. SIX call
    sites had taken the width from the tokenizer (`crates/cli`'s two,
    `crates/server`'s real model, `crates/bench`'s real model, plus
    `quality_common`, `logit_dump` and the nondeterminism probe), and every
    one of them had a `RealForwardRunner` in hand. They now call
    `RealForwardRunner::vocab_size()`, which reads `arch.vocab_size`.
    Two things worth carrying. The existing families were UNMOVED by the fix
    (Gemma's perplexity and both digests reproduced to the last hex
    character), which is what says the old value was right by coincidence
    rather than by design -- for one model per dialect the two numbers are
    equal. And the general form: when a table is keyed by X and holds a
    property of Y, it is a latent bug that stays invisible for exactly as
    long as the X-to-Y mapping happens to be injective. The tokenizer's own
    `vocab_size` is still correct for the SCRIPTED paths, which have no model
    to ask.

38. **A MEASUREMENT tool's model-specific constant is a wrong ANSWER waiting
    for its second caller, and it fails LOUDLY in the one direction that
    looks like your engine's fault.** Sibling of 37 one layer out: that one
    was a per-dialect constant standing in for a per-model property, this is
    a per-MODEL constant standing in for a universal, and both stay invisible
    while exactly one model exercises them. `scripts/kld_llamacpp.py` carried
    `SOFTCAP_BOUND = 30.0` and aborted any run whose reference logits
    exceeded it, on the correct reasoning that if llama.cpp skips a softcap
    this port applies, the two heads are not the same function and no
    divergence below means anything. 30 is Gemma's
    `final_logit_softcapping`, and Phase G, Phase S and every other caller
    were Gemma. `qwen3moe` declares `finalLogitSoftcap: 0.0` and both engines
    read max |logit| ~51.9, so the FIRST non-Gemma run of a script written to
    be model-agnostic died with a message blaming the heads. The bound now
    comes from the install's own `manifest.json`, and where a family declares
    none there is no transform to mismatch, so both engines' maxima are
    REPORTED (51.93 against 51.97) rather than checked against an invented
    tolerance. Three things worth carrying. Read the property, never recall
    it -- the fix is `softcap_of(install)` and it costs one file read.
    Prefer a reported observable to a fabricated threshold when the
    assertion genuinely does not apply. And the check had a second, quieter
    bug the move fixed for free: it hung off the FRESH-RUN branch, so it was
    skipped precisely when an arm was reused from cache, which is most
    re-runs; it now computes from the returned array and runs either way.
    A guard that only fires on a cold path is close to no guard.
    Look for siblings before assuming this one is done. That clause used to
    end here naming `scripts/kld.py` as Gemma-pinned throughout (`REPO`, and
    a docstring asserting softcap 30) and calling it BY DESIGN, since its
    reference checkpoint genuinely is a Gemma repo. **THE AUDIT IT RECORDED
    AS OWED WAS PAID 2026-08-21** and the "by design" reading did not
    survive it: the pin is a keyed `CHECKPOINTS` table now, the name is
    REQUIRED rather than defaulted, and the softcap comes from
    `softcap_of(install)` -- the same three moves this gotcha's own fix made
    one file over. A defensible constant and a correct one are different
    things, and the tell that it was the first is that nothing about `REPO`
    had to change for it to become wrong, only the arrival of a caller.
    **THE SECOND CALLER TURNED OUT NOT TO BELONG IN THAT FILE AT ALL**,
    which is the part worth carrying: `docs/BENCHMARKS.md` had recorded for
    months that Qwen 3.6 has no cross-engine number because "`kld.py`'s
    reference is pinned to the Gemma repo", i.e. that the pin was the whole
    obstacle. It was not. That checkpoint is an MoE and `kld.py` has no
    reference guard, so the row went to `kld_mlx_affine.py`, whose
    `assert_reference_matches` exists for exactly that shape. A stale
    sentence naming one blocker is worth re-deriving before it is believed;
    this one had been true when written and was wrong in two ways by the
    time anyone acted on it.

39. **An OPTIONAL metadata key that falls back to a BASELINE is a bug; it has
    to fall back to what the format says its absence MEANS.** Third instance
    of 37's shape and the cheapest one to state: `arch_from_gguf` set
    `head_dim` only when `attention.key_length` was present, so a GGUF
    omitting it kept `known_architecture(family)`'s value. llama.cpp's own
    default is `embedding_length / head_count`, and that -- not the
    baseline -- is what an absent key means. The two agree for Mixtral 8x7B,
    Mistral 7B and Llama 3.1, all 32 heads of 128, which is why three real
    files passed over it; TinyLlama 1.1B is 32 of 64 and fails at the FIRST
    `q_proj` with "Q6_K packed size 3440640 does not match 4096x2048",
    nowhere near the config that produced it. Reach for the FORMAT's default
    when a key is optional, and if the format has none, refuse. Note the
    baseline fallback is correct for the field class it was written for --
    family-EXTENSION fields, where Gotcha 24 requires it -- so this is about
    SHAPE fields borrowing that habit.
    Two siblings found in the same phase, both the same species. GGUF's
    `rope_freqs.weight` was mapped `Ignored` with the reason "derived from
    `rope_theta` at runtime", which is true of every file that OMITS it and
    false of every file that SHIPS it (Llama 3.1's is learned per-dimension
    scaling); it is refused by name now, because dropping it yields a model
    wrong only past the training length, which no smoke test reaches. And
    `manifest.quant`'s five fixed slots answered `absent` for every component
    a given architecture lacks -- three of them on a dense model -- which
    `validate_quant` then refuses by design. THE GENERAL FORM ACROSS ALL
    THREE: a default is a claim about what silence means, and the three ways
    to get it wrong are to inherit a neighbour's answer, to state a value the
    file contradicts, and to state a value no consumer accepts.

40. **`phys_footprint` does NOT count a dense install's resident weights, and
    Gotcha 19 is stated of a STREAMED MoE install.** Measured 2026-08-10 on
    the real Mistral 7B Q4_K_M, resident region 4,371,570,688 bytes: peak
    `phys_footprint` 683.9 MiB from `turbospark-bench`'s mach sampler and
    684.2 MiB from `/usr/bin/time -l`, agreeing to under a MiB, with a
    maximum RSS of 46.8 MiB. Both counters are far UNDER the 4.07 GiB the
    weights occupy, and KV at 4096 context (32 layers x 8 KV heads x 128 x 2
    x 2 bytes = 537 MiB) is most of what they do count. So the ROADMAP M4
    Phase 0 inference -- "nothing streams in a dense install, therefore the
    resident floor IS the working set" -- is refuted, and it was derived from
    Gotcha 19 rather than measured. What Gotcha 19 says about the MoE
    installs it was written for still stands and is unretested here; what
    does not transfer is the general claim that a `newBufferWithBytesNoCopy`
    mapping is always counted. Re-derive it per install shape rather than
    quoting it, and note the practical consequence is the pleasant one: a
    dense 7B runs in well under a gigabyte of counted footprint.
    THE COROLLARY FOR AN ORACLE ROW: with the weights out of the picture and
    no expert slot cache, KV is what is left, so the row is mostly asserting
    the CONTEXT WINDOW. `mistral_memory_oracle.rs` runs at 8,192 and reads
    1,201 MiB, of which 1,024 is KV; at the 4,096 the other three families
    use it reads 684. Neither number is comparable to a sibling's without
    the window, which is why `run_oracle_at_context` prints it.

41. CHAT FRAMING IS A PROPERTY OF THE CHECKPOINT, and the dialect is not evidence about it. Moved to
    [crates/tokenizer/CLAUDE.md](crates/tokenizer/CLAUDE.md) Gotcha 1.

42. A routed sub-tensor is not always a matrix, and not everything in an expert blob is a quantization scheme. Moved to
    [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotcha 16.

43. **`powermetrics` MEASURES THE MACHINE, NOT YOUR PROCESS, AND THE
    THERMAL CHECK CANNOT SEE THE DIFFERENCE.** Gotcha 28 is about pressure
    silently rewriting a power row; this is its sibling, and the two need
    separate instruments. Measured 2026-08-13 on the gpt-oss install, AC,
    every arm Nominal: `medium-review` decode read **1.7072 J/token on p1
    and 1.0799 on p2 for byte-identical work** (2,597 tokens both times, a
    37% spread), and `scripts/power.sh` averaged them into 1.3936 -- a
    number describing neither run. Nothing was throttling. `cpu_W` fell
    monotonically through the capture (4.76 / 4.07 / 3.34 / 3.03 early
    against 1.50 / 1.62 / 1.66 / 1.81 late) with `gpu_W` tracking it,
    because Combined Power is SYSTEM-wide and this machine's desktop UI was
    busy early and idle late. A re-run on a quiet machine reproduced to
    0.4% and 1.7%.
    THREE THINGS TO CARRY. **The tells are `cpu_W` against the install's
    own norm and DISPERSION between arms doing identical work**, and
    `scripts/power.sh` prints neither in its summary -- read `rows.tsv` per
    arm before believing any row, exactly as Gotcha 28 requires for
    pressure. **A published row can carry it silently**: the one-case
    gpt-oss row of 2026-08-12 read 36.67 W against a clean 32.92, and the
    whole 3.75 W difference is `cpu_W` (4.32 against 1.48) while `gpu_W`
    agrees to 2.9% and tok/s to 2% -- so it was never an engine reading.
    And **the contaminating load can be the thing watching the
    measurement**: the top consumers here were the desktop app rendering
    this session plus WindowServer, so a capture driven from an interactive
    session has to be left alone while it runs, not watched.
    The cheap discipline: run each case at least twice, compare the ARMS
    rather than the mean, and treat any within-case spread over a few
    percent as contamination until a quiet re-run says otherwise.

    **BOTH OF THOSE TELLS ARE BLIND TO A STEADY LOAD, AND THE THIRD ONE IS
    NOW IN THE HARNESS** (2026-08-21). Dispersion caught the gpt-oss capture
    because that load DRIFTED -- the desktop UI was busy early and idle late,
    so two arms doing identical work disagreed by 37%. A CONSTANT background
    load contaminates every arm equally: dispersion reads clean and only the
    `cpu_W`-against-norm tell fires, which needs a norm, i.e. a previous
    clean capture of the same install. **The first capture of a new install
    has no norm at all**, which is exactly when a power row is most likely to
    be published.
    Measured on the first ornith35b capture: `gpu_W` reproduced to 0.18% and
    tok/s to 0.29% across arms, every one of six rows Nominal, and `cpu_W`
    read 10.67 W -- against a Finder stuck at 99% of a core, `iconservicesagent`
    at 25%, and 269% of CPU summed across the machine. Nothing in the summary
    said so, and the row was one paste from `docs/POWER_BASELINE.md`.
    **The statistic that needs no norm is the MINIMUM CPU power anywhere in
    the log**, gaps between generations included: a capture spans model opens
    and settling pauses, so a quiet machine touches near-idle at some point
    and a contaminated one never does. Calibrated on three real captures
    here -- 94 mW and 392 mW on the two clean DFlash2 ones against 3,361 mW
    on the contaminated one, with no sample under 3 W in 106 seconds.
    `scripts/power.sh` prints it as `contamination floor` and warns over
    2,000 mW, and its summary now carries a `J/tok+/-` spread column plus a
    per-group warning over 10%, so both tells are in the output rather than
    recoverable by hand from `rows.tsv`. The sentence above saying the script
    "prints neither in its summary" was true when written and is the thing
    that got fixed. **`gpu_W` and tok/s survive this kind of contamination**
    (GPU work is insulated, and CPU contention depresses throughput rather
    than inflating it), so a throughput row from a contaminated capture is
    conservative while its `watts` and `J/tok` are not.

44. **A SPECIAL TOKEN'S DELTA IS THE EMPTY STRING, so a consumer that reads
    MARKUP must key on the token ID and must not skip an empty delta.**
    `StreamingDetokenizer` renders special tokens to nothing, so every frame
    token a dialect uses -- Harmony's `<|channel|>` / `<|message|>` /
    `<|end|>` / `<|start|>`, Gemma's channel pair, ChatML's `<think>` --
    reaches `RawDecodeProgress::Token` as `(id, "")`. Keying a decoder on
    TEXT therefore cannot work at all, which is obvious and is why every arm
    of `StructuredAssistantDecoder` keys on ids. THE SECOND CONSEQUENCE IS
    THE ONE THAT BITES: an `if text.is_empty() { return; }` guard placed
    BEFORE the decoder -- an ordinary, harmless-looking line in any printing
    loop, and one that predates the decoder in every loop that has one --
    swallows every state transition, so the machine never leaves its initial
    state and the whole turn reads as one run of content with the markup's
    words in it. Measured 2026-08-13 on the real gpt-oss install: the first
    end-to-end run after wiring the CLI's Harmony splitter printed
    `analysisThe user asks...assistantfinalThe sky appears blue...`, which
    looks exactly like a decoder that was never constructed. No error
    anywhere, and the whole suite green -- a fixture test feeds `(id, "")`
    correctly by construction, so it cannot see this, and the decoder's own
    seven tests all passed while the feature did nothing.
    The emptiness check belongs on the decoder's OUTPUT, never on its input.
    `crates/server/src/handler/exec.rs` gets this right and
    `crates/cli/src/generate.rs` did not.

45. A writer may only record a tag some reader honours; the walk NARROWS every unquantized tensor to BF16. Moved to
    [crates/repack/CLAUDE.md](crates/repack/CLAUDE.md) Gotcha 9.

46. **XET IS A TRANSFER LAYER, NOT A FORMAT, AND EVERY REAL INSTALL HERE WAS
    ALREADY STREAMED THROUGH IT.** Researched 2026-08-14. Hugging Face replaced
    Git LFS with Xet (content-defined chunking, ~64 KiB chunks, dedup in a CAS),
    which reads like a fifth checkpoint format to sit beside safetensors and
    GGUF and is nothing of the kind: the client reconstructs the file
    byte-identically, so `gguf_header`, `safetensors_header`, the `.gturbo`
    writer, every kernel and MLX are all untouched by it. There was no
    compatibility work to do, and the reason is worth stating so nobody
    re-derives it: THE PORT HAD BEEN READING XET-BACKED BYTES FOR MONTHS. A GET
    on `resolve/main` for `ggml-org/gpt-oss-20b-GGUF` and
    `prism-ml/Bonsai-27B-mlx-1bit` returns an `x-xet-hash` header and a 302 into
    `us.aws.cdn.hf.co/xet-bridge-us/...`, and that bridge answers an arbitrary
    `Range` with a correct `206` and `Content-Range`, which is exactly what
    `HttpRangeSource::read_chunk` already asserts. The MLX side needed nothing
    either: the `hf` CLI here runs `hf_xet` already, and `mlx`/`mlx-lm` only ever
    see local files.
    **WHAT THE QUESTION ACTUALLY TURNED UP IS A THROUGHPUT CEILING, and it has
    been in every repack timing in this repo.** The bridge is the LFS-compatible
    path, which is SINGLE-STREAM by construction: one connection to one
    CloudFront edge, and `xet-core` issue #821 documents 65-75% of those edges
    capped at exactly 8.7 MB/s while the rest run 60-70. That is not a
    hypothesis about this repo's numbers, it is a match to them -- gpt-oss
    12.1 GB in 25 min is 8.1 MB/s and Bonsai-27B 5.13 GB in 9.7 min is 8.8.
    Measured here at the 64 MiB chunk size the walk dispatches, over a 512 MiB
    span: serial 60.4 s (8.9 MB/s) against 8-way 17.6 s (30.5 MB/s), a 3.4x.
    The serial arm landing on 8.9 against the documented 8.7 is what says the
    cap is the thing being measured. `read_range` now issues its chunks
    concurrently.
    **BUT KNOW WHICH WALKS THAT TOUCHES, because it is not "repacks are 3.4x
    faster" and the obvious verification cannot see it.** The concurrency
    engages only when ONE `read_range` exceeds the 64 MiB chunk cap, and
    `gguf_checkpoint::read_tensor` issues one call per TENSOR. On an MoE
    checkpoint that is most of the bytes, because a routed tensor is the whole
    expert table for a layer (Gemma's `ffn_gate_up_exps` is ~410 MiB). On a
    DENSE one it is almost nothing: TinyLlama re-streamed in 3:28 against a
    recorded ~3 min, i.e. unchanged, because not one of its 201 tensors is over
    the cap. That run was still the right CORRECTNESS gate (`model_weights.bin`
    came out SHA-256-identical to the install already on disk) and the wrong
    THROUGHPUT one, which is worth remembering the next time a cheap fixture is
    picked to verify a change: the small model was chosen because it was cheap,
    and cheapness is exactly what made it unable to exercise the property.
    Extending the win to dense checkpoints means reading several tensors
    concurrently, which is a change to the walk and not to `ranged_download.rs`.
    **THE TRAP IN COLLECTING IT IS THAT THE OPTIMIZATION IS THE CLIENT SETTING,
    NOT THE CONCURRENCY.** The bridge speaks HTTP/2 and reqwest will happily
    multiplex N concurrent range GETs onto ONE connection, which is one edge,
    which is the same cap -- so the parallel version does the same work at the
    same speed, with no error and nothing in any log to say the knob did
    nothing. `HttpRangeSource::new` sets `http1_only()` for that reason alone.
    Any future change to that client, or any measurement of that constant, has
    to confirm the connections are actually distinct before believing a number.
    **NATIVE `hf-xet` WAS COSTED AND DECLINED**, so this is a closed question
    rather than an unexplored one. It is Apache-2.0 and maintained, but it pulls
    tokio and a large tree into a crate that is `#![forbid(unsafe_code)]` and
    needs a `xet-read-token` auth flow, and what it buys is adaptive concurrency
    (had far more cheaply above) plus chunk dedup, which is worth NOTHING to
    this workload: every checkpoint here is streamed exactly once, never written
    to disk, and two quantizations of one model share no chunks. Re-open it only
    if the walk ever needs a file over the bridge's size ceiling, which no
    checkpoint here approaches.

47. **A NEW CHECKPOINT IS NOT A NEW FAMILY, AND THE CHEAPEST WAY TO FIND OUT
    IS TO PARSE ITS CONFIG BEFORE DOWNLOADING ANYTHING.** `Qwen/Qwen3.8-27B`
    was published 2026-08-14 and needed ZERO changes to `crates/model-io`,
    `crates/gpu` or `crates/runtime`: its `text_config` agrees with
    `prism-ml/Bonsai-27B-mlx-1bit`'s on 33 of 35 keys, so it parses to the
    existing `qwen_gdn_dense_27b()` baseline exactly and runs the existing
    `families/qwen/` dense flow. The two keys that differ (`eos_token_id`,
    and the `quantization` object) reach no `ArchConfig` field at all --
    one is the tokenizer's business and the other
    `parse_gemma4_quantization`'s. Both checkpoints also carry 2,180 tensors
    and 333 `vision_tower.` ones, which is independent evidence: the config
    comparison and the tensor inventory share no input.
    THE WORKFLOW IS THE REUSABLE PART, because the alternative was assuming
    a seventh family and budgeting a bring-up. `config.json` is a few KB
    over HTTP; diffing it against every shipped baseline costs seconds and
    answers "is this new?" before any of the expensive questions are asked.
    `every_published_checkpoint_parses_to_one_baseline`
    (`crates/repack/tests/qwen35_config.rs`) is that diff turned into an
    offline assertion, so a future point release that DOES move a shape key
    reddens a millisecond test rather than failing a 16 GB stream at some
    tensor offset.
    TWO CONSEQUENCES WORTH KEEPING. The baseline is named for the
    ARCHITECTURE (`qwen_gdn_dense_27b`, renamed off `bonsai_27b`) because a
    checkpoint name on a shared baseline misleads every later reader. And
    the family is this repo's first CONTROLLED quantization comparison: same
    architecture, same tokenizer, same flow, at 1-bit group 128, 2-bit group
    128 and INT4 group 64. **IT IS A TRIPLE SINCE 2026-08-15**, and the third
    point is what settles the reading: 18.3 tok/s at one bit (3.9 GB of
    weights), 14.2 at two (7.6 GB) and 19.0 at four (15.1 GB). Decode does
    not track the weight bytes AT ALL -- not even monotonically -- so the
    flow is COMPUTE-bound at every width, which the 1-bit entry suspected
    and a two-point line could still have been read as bandwidth. The 2-bit
    GEMV is simply doing more per byte than either neighbour (four elements
    a byte, and no `+/-1` shortcut).
    Note it is not a clean quantization ablation in the OTHER direction:
    Bonsai and its ternary sibling are prism-ml's own QAT checkpoints while
    Qwen3.8 is Qwen's release quantized by mlx-community, so a TRAINING
    separates the perplexities as well as a width. The two prism-ml files
    are the closest thing to a clean pair (same publisher, same base) and
    read 6.8350 at two bits against no frozen row at one, since Bonsai never
    got a quality gate.

48. **A SUMMARY STATISTIC THAT IS INVARIANT UNDER THE MUTATION YOU ARE
    TESTING FOR PROVES NOTHING, AND SUB-4-BIT PACKING PRODUCES ONE EVERY
    TIME.** Second instance, on a second width, which is what makes it a rule
    rather than an anecdote. At ONE bit a wrong bit order permutes elements
    within a byte and leaves every magnitude, every group scale and the total
    POPCOUNT untouched (the 1-bit entry's step 2 records it). At TWO bits the
    same thing happens to the LEVEL HISTOGRAM: permuting the four 2-bit
    fields inside a packed word permutes their multiset without changing it,
    so `prism-ml/Ternary-Bonsai-27B-mlx-2bit`'s striking property -- level 3
    never occurs in 245,760 elements, which is what makes it ternary -- is
    equally true of every wrong field order. It reads exactly like evidence
    about the packing and is evidence about the DATA.
    The rule that follows has two halves and the second is the one usually
    skipped. Reach for an ORACLE (an independent implementation decoding the
    same bytes) rather than a self-consistency check, which is
    `scripts/mlx_2bit_oracle.py` here and `scripts/ggml_tables.c` for the IQ
    codebooks. And then ASSERT THAT THE FIXTURE DISCRIMINATES before relying
    on it: `reversed_field_order_within_the_byte_is_not_the_convention`
    checks that the frozen bytes are not palindromic in their fields at all,
    because a tidier fixture would pass the whole test while proving nothing,
    and `the_level_histogram_cannot_see_field_order` states the invariance
    itself as a test so nobody re-derives it as a finding.
    **THE SAME TRAP AT THE DISPATCH LEVEL COST A REAL GAP, found by mutation
    and not by review.** Swapping `encode_embed_lookup_int2` for its 1-bit
    sibling left every end-to-end case in `real_forward_qwen35.rs` green: the
    wrong kernel still reads the table, still produces finite logits and
    still moves them when the table is perturbed -- it strides by `D / 8`
    where a 2-bit table strides by `D / 4`, so it reads the WRONG ROW, which
    no assertion about "does the table reach the logits" can see. The case
    that closes it patches a BYTE RANGE the wrong stride does not read for
    that token, and asserts the two ranges are disjoint first. Generalise:
    when two code paths differ only in a STRIDE, a test that perturbs the
    whole tensor cannot tell them apart.

49. **A STREAM DECODER WHOSE TERMINATOR IS A STOP TOKEN CAN NEVER SEE ITS OWN
    TERMINATOR, AND THE FAILURE IS SILENCE.** `run_raw_completion` breaks out
    of its loop on a stop token BEFORE the progress callback, deliberately: a
    stop token is framing, and no consumer wants it in the reply. Harmony ends
    a tool call with `<|call|>` and `<|call|>` is in `gpt-oss`'s stop set, so
    `StructuredAssistantDecoder` is handed the whole call and then never told
    it ended. Every other dialect's tool span closes on an ORDINARY token that
    either arrived or did not, which is why `finish()` treats an open span as
    an error for those three and as the normal case for this one.
    THE PART THAT GENERALISES IS THE FAILURE MODE, not the fix. A consumer
    driving only `consume` gets no error, no markup on the wire and no
    truncated output -- just an assistant turn with nothing in it, which reads
    as a model that declined to answer. `crates/server`'s `stream_blocking` had
    never called `finish()` at all (nothing had needed it), and `crates/cli`
    still does not. So when adding a span to a streaming decoder, ask what
    CLOSES it before asking what parses it, and if the answer is a token the
    loop swallows, the exit path is part of the feature rather than a tidy-up.
    A SECOND, SMALLER INSTANCE RODE ALONG in the same item and is the same
    shape one layer out: `StopReason::ToolCalls` fired on
    `tokenizer.tool_response_id`, which is Gemma's marker and
    `NO_SUCH_TOKEN_ID` on every other dialect, so a Harmony call fell through
    the ladder to `StopReason::Eos` and reached a client as
    `finish_reason: "stop"` -- WITH a correct `tool_use` block beside it, which
    is what makes it hard to notice. The field is `tool_call_stop_id` now,
    because the question the ladder asks ("does this stop token mean the model
    is invoking something") is not the question a markup id answers, and the
    two dialects that have one do not spell it with the same kind of token.

50. A NORMALIZATION CONVENTION IS A PROPERTY OF THE TENSOR, NOT OF THE FAMILY, and one model can use two. Moved to
    [crates/gpu/CLAUDE.md](crates/gpu/CLAUDE.md) Gotcha 10.

51. Every test in a perturbation-style fixture file can be self-relative, and then the file catches almost nothing. Moved to
    [crates/runtime/CLAUDE.md](crates/runtime/CLAUDE.md) Gotcha 23.

52. **A DIALECT PROBE KEYED ON "SPECIFIC-LOOKING" TOKENS IS A COINCIDENCE
    WAITING FOR ITS SECOND CHECKPOINT, and the probe that broke had a comment
    saying so.** `detect_dialect` resolved Harmony on `<|start|>` plus
    `<|message|>`, reasoning in a comment that two markers are "a far more
    specific pair than a single `<|im_end|>`" and that Gotcha 41's lesson is
    that a probe stops being injective when a second checkpoint arrives.
    `mlx-community/Muse-Glimmer-30B-4bit` carries both and is not Harmony: no
    `<|channel|>`, no `<|return|>`, no `<|call|>`, no `<|end|>`, no
    `<|startoftext|>`, and an `<atem:function_calls>` tool DSL instead of a
    channel recipient.
    **THE ACTUAL BUG WAS THAT THE PROBE AND THE RESOLVER DISAGREED ABOUT WHAT
    THE DIALECT IS.** `resolve_harmony` requires six tokens; the probe tested
    two. Any checkpoint in the gap resolves to a dialect that then fails to
    load -- which is the good failure mode (it did fail loudly, on
    `<|startoftext|>`) and is still the wrong answer, because the model is
    refused rather than run. The fix is to test a token the resolver requires
    and the impostor lacks (`<|channel|>`, the format's defining feature), not
    to add an arbitrary third marker. **The general rule: a detection probe
    must not be able to pass where its own resolver will fail.** Grep for
    resolvers whose `required_id` set is larger than their probe's.

53. **minijinja REJECTS A CONDITIONAL EXPRESSION AS A KEYWORD ARGUMENT, and
    real chat templates use one.** `f(k=a if c else d)` is valid Jinja2 and
    minijinja 2.22.0 (the latest 2.x) answers
    `syntax error: unexpected identifier, expected ","`. Muse Glimmer's
    template has `namespace(name=tcid if tcid else '')`, so the WHOLE template
    failed to parse and `--messages-file` could render no prompt at all.
    `jinja_chat_template.rs::parenthesize_conditional_kwargs` rewrites
    `k=EXPR` to `k=(EXPR)` before `add_template`, which is exactly Jinja2's
    own precedence for a keyword-argument value and therefore changes no
    semantics by construction.
    **IT IS A SHIM AND IS MEANT TO BE DELETED.** No stable minijinja has the
    fix (3.0.0-alpha.0 is untested here and deliberately not taken, and no
    issue has been filed upstream from this repo); when one does, delete the
    function and its call site and re-run
    `tests/jinja_chat_template.rs` -- `a_conditional_keyword_argument_parses`
    is the test that says whether the engine handles it directly, and
    `the_shim_is_a_no_op_on_templates_that_do_not_need_it` must pass either
    way.
    Two hazards it handles STRUCTURALLY rather than by pattern-matching,
    because a template is mostly prose: the scan only enters `{{ }}` and
    `{% %}` blocks (never text, never `{# #}` comments), and a `=` counts only
    when it is not part of `==`, `!=`, `<=` or `>=`. String literals are
    tracked so a `,` or `)` inside `'...'` cannot end an argument early.

54. A cache that already deduplicates makes a "union" saving vanish. Do not reach for the expert-union plan. Moved to
    [crates/runtime/CLAUDE.md](crates/runtime/CLAUDE.md) Gotcha 14.

55. **THE CHECKPOINT'S TRAINED CONTEXT IS INSTALL METADATA, NOT AN
    `ArchConfig` FIELD, AND THAT IS A DECISION ABOUT WHAT VALIDATION IS FOR.**
    `--max-context` defaults to `auto` since 2026-08-17, resolving to
    `min(trained context, largest window fitting a quarter of the memory
    pool)`. The trained context comes from `max_position_embeddings`
    (safetensors, read from `text_config` before the root -- the multimodal
    wrappers nest the text model and put a VISION config beside it) or
    `<arch>.context_length` (GGUF), and lands in `manifest.json` as
    `arch.trainedContext`, written by `catalog::install` after the walk.

    **It is deliberately not in `ArchConfig`**, even though it looks like one
    of that struct's shape fields, and the reason is that `arch_validation`
    compares one field by field against a per-FAMILY baseline while a trained
    context is per-CHECKPOINT: a YaRN-extended release declares a longer one
    than the base it was built from. Putting it there would make every such
    pair a baseline mismatch, would need a per-checkpoint claim inside a
    per-architecture table, and would redden the whole-struct
    `assert_eq!(derived, baseline)` comparisons in
    `gguf_checkpoint_network.rs`. It is also read by no kernel. The cost of
    keeping it out was measured before choosing: the field would have touched
    26 full `ArchConfig` literals, and threading it through the walk instead
    would have touched ~60 call sites across 28 files, so the third option --
    annotate the manifest after the walk, in the ONE place both intake formats
    meet -- is what landed.

    **THREE DEFAULTS HERE ARE CLAIMS ABOUT WHAT SILENCE MEANS** (Gotcha 39's
    rule, three times in one feature). An install declaring NO trained context
    resolves `auto` to `DEFAULT_MAX_CONTEXT` and never to what memory allows:
    that is every install written before the field existed, and sizing from
    free RAM alone would take a 13 GB install from its documented 4,096 to
    ~250,000 the first time anyone re-ran the same command. A declared value of
    ZERO is what a missing key looks like after a cast and reads as unknown,
    never as a window of zero. And `physical_memory()` answering 0 means the
    PROBE is unavailable (it does, off macOS) rather than that the machine has
    no memory -- reading it as an empty budget would refuse every explicit
    window and resolve every `auto` to a context of zero.

    **The two failure modes are different kinds of thing on purpose.** Past
    the trained context WARNS (RoPE extrapolates rather than failing, some
    checkpoints carry YaRN scaling meant to exceed it, and the check cannot
    apply at all to an install that declares none, so refusing would be
    enforced on some installs and not others). Past what memory holds is
    REFUSED with the whole subtraction shown, because `KvCacheManager::new`
    allocates every layer up front and its failure is a Metal allocation error
    with no number in it naming the flag. Measured on the real ternary 27B:
    `--max-context 1000000 needs 61.0 GiB of KV cache; 25.0 GiB available
    (36.0 GiB physical - 7.0 GiB weights and expert cache - 4.0 GiB reserve).
    Largest context that fits: 408576`.

    Two arithmetic traps in estimating the KV, both of which make an estimate
    wrong by a factor rather than a margin. A sliding-window layer is a RING
    capped at `sliding_window + 128`, so past that cap it stops growing and a
    model's per-token cost is its FULL layers alone -- Gemma 4 is 5 of 30 and
    costs 20 KiB/token where a dense 7B costs 128, and a per-token model that
    misses this is 6x high on Gemma. And `committed_bytes` is NOT the weight
    file's size: Gemma's `model_weights.bin` is 1.26 GiB while its expert
    table is 12 GB, of which the slot cache pins ~3.0 GiB at 32 slots, so
    counting the mapped file alone lets an explicit window claim memory the
    slot cache is about to take.

    The estimate is cross-checked against the one KV figure in this repo
    measured on two independent counters: 32 layers at 8,192 comes out to
    exactly the 1,024 MiB of the 1,201 MiB `mistral_memory_oracle` peak that
    Gotcha 40 attributes to KV.

56. **A REASONING LEVEL IS A PROMPT PROPERTY, IT LIVES IN THE CHECKPOINT'S
    TEMPLATE, AND THE DEFAULT YOU INHERIT DEPENDS ON WHAT YOU DO NOT SEND.**
    `Qwen/Qwen3.8-27B`'s model card says it reasons at `xhigh` by default, and
    that is true of transformers and mlx-lm and was never true here. Its
    template reads

    ```jinja
    {%- if enable_thinking is undefined or enable_thinking is true %}
        {%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}
    ```

    so `xhigh` is what a caller gets by leaving BOTH keys undefined. This port
    passed `enable_thinking: false` on every render, which closes that gate:
    it took neither the default nor a level, and its generation prompt ended
    in a pre-closed `<think>\n\n</think>`. Not a bug -- it is why the shared
    1,024-token budget suffices for that family -- but "the model card says
    xhigh" was evidence about upstream's call site, not about this one.
    `--reasoning off|low|medium|high|xhigh` (and `reasoning_effort` on both
    server endpoints) is the knob; `Off` is the default and renders the exact
    bytes every earlier release did, which is what leaves every frozen digest
    where it is.

    FOUR THINGS THAT FOLLOW, each a decision rather than an observation.

    **A LEVEL IMPLIES `enable_thinking: true`.** Every template that has both
    reads the effort key INSIDE the thinking gate, so setting one without the
    other is a flag that renders nothing, reports nothing and looks like it
    worked.

    **BOTH SPELLINGS ARE SET.** Qwen 3.8 and Harmony say `reasoning_effort`,
    `muse_glimmer` says `reasoning_strength`. A template reads the one it
    knows and ignores the other, so setting both costs nothing and avoids a
    per-family table that rots on the next checkpoint.

    **THE ACCEPTED SET IS THE CHECKPOINT'S, NOT THIS PORT'S.** The union is
    accepted at the CLI and the template validates: Qwen 3.8 takes
    `xhigh`/`medium`/`low` and RAISES on `high`, which the other two accept.
    A per-family allowlist here would be a second, staler copy of a set the
    checkpoint already states by name in its own error.

    **A TEMPLATE THAT CANNOT EXPRESS A LEVEL IS THREE CASES, NOT TWO**
    (`MfTokenizer::reasoning_support`): `Level` (an effort key), `ToggleOnly`
    (`enable_thinking` alone -- Qwen3.5-era and Gemma 4, where thinking still
    turns on and the LEVEL is dropped, so the CLI warns), and `None` (no
    template at all, where a level is REFUSED rather than dropped). Silence
    was the failure mode to avoid; each of the three says something different.

    THE TRAP THAT REACHED A REAL MODEL, and it is Gotcha 44's shape one layer
    up: turning thinking on is only half the feature, because the reasoning
    then has to be SEPARATED from the answer. `StructuredAssistantDecoder`
    emitted Harmony's `analysis` channel as `Reasoning` and DISCARDED ChatML's
    `<think>` body and Gemma's labelled thought channel -- correct while no
    knob could turn those on, and wrong the moment one could. Both now emit
    `Reasoning`, and the shared `runtime::TurnSplitter` (which replaced the
    three per-consumer copies of that decision on 2026-09-05) builds a
    decoder for those dialects when a level was asked for. Skipping it is not
    cosmetic: measured on the real Gemma 4 install, the first `--reasoning
    low` run printed a bare `thought` (the channel LABEL, as prose), then the
    model's scratch work, then its answer, all as one run of content, because
    the frame tokens render to the empty string.

    **AND EMITTING THE CHANNEL WAS STILL ONLY HALF OF THAT HALF, which took a
    second real-model run months later to find.** A ChatML generation prompt
    OPENS the `<think>` frame itself -- Qwen's template ends
    `<|im_start|>assistant\n<think>\n` when thinking is on -- so the model's
    first generated token is already scratchpad and `think_start_id` never
    arrives. The arm was taught to EMIT reasoning and was never ENTERED:
    `StructuredAssistantDecoder` started in the visible channel unconditionally
    and stayed there, since the `</think>` that arrives later flips Visible to
    Visible. Measured on the real `qwen38-27b` install at `--reasoning low`:
    1,413 bytes of answer and 0 of reasoning, against 676 and 737 after
    `new` was given the prompt ids to scan. Same tokens either way -- this is
    a routing bug, not a generation one, which is why no digest could see it
    and why `crates/bench` (which builds no decoder) was unmoved.
    Two things to carry. **A state machine fed a stream that begins MID-FRAME
    needs to be told where it starts**, and the honest source is the rendered
    prompt rather than the flag that produced it: keying on
    `reasoning != Off` would open the frame on a checkpoint whose template
    thinks without prefilling the tag, and hand the caller an EMPTY reply --
    a worse failure than the one being fixed. And a fix like this one has
    exactly two arms worth checking end to end, the one that should move and
    the one that must not: the Gemma `--reasoning off` smoke is byte-identical
    across a single-variable A/B, which is what says the pass-through path
    never acquired a decoder.

    **SWEEPING THE OTHER DIALECTS FOR THE SAME SHAPE FOUND ONE MORE BUG, AND
    IT WAS A DIFFERENT MECHANISM.** Rendering every installed checkpoint's
    template at every level it accepts and reading where the prompt leaves the
    model (6 dialects, 10 installs, seconds, no GPU) says ChatML was the ONLY
    instance of the initial-state bug: Gemma opens no channel at a level and
    pre-closes an empty one at `off`, Harmony ends outside any frame, DeepSeek
    pre-closes with `</think>`. What the sweep turned up instead is
    `muse_glimmer`, which had no decoder arm at all and printed its `to=self`
    scratchpad as the reply -- on EVERY turn, not just when a level was asked
    for, because its template puts a reasoning directive in the system message
    unconditionally and defaults the strength to `high`. So the class is
    broader than the fix: **ask where the prompt leaves the model AND whether
    anything is built to read it**, because the second question has its own
    wrong answer. Both now share one rule -- the decoder's initial state comes
    from the rendered prompt -- and `crates/tokenizer` Gotchas 7 and 8 record
    the two frames it is applied to.

57. **A DEGENERATE MEASUREMENT MUST FAIL, NOT PRINT -- AND "RANK OF THE TRUTH"
    IS WHAT SEPARATES BROKEN FROM WEAK.** `mtp_accept_length_probe.rs` read 0
    accepted of 7,168 proposals and printed a tidy `loses` table anyone could
    quote as a verdict on MTP. A weak drafter still lands common tokens, so
    exactly zero is a bug; the probe now ASSERTS a functional drafter (first
    proposal accepted above 2%) and fails rather than reporting. That is the
    same failure the top of `docs/MTP_SPECULATIVE.md` records one level up --
    a composite built on a broken arm measures the arm, not the question.
    THE INSTRUMENT THAT CLASSIFIED IT IN ONE NUMBER: rank the EXPECTED output
    in the component's own distribution and read it against the random
    baseline (`vocab/2`). Three diagnoses rather than two -- near the top is
    WEAK, near `vocab/2` is UNRELATED, and near LAST is ANTI-ALIGNED, which no
    pairing or position fix rescues (the MTP head reads median 248,308 of
    248,320, i.e. ~11th from the top under negation). Pair it with a
    correlation against a working reference at SEVERAL offsets before hunting
    weights: a head that is merely mis-paired peaks positively at some offset,
    and this one was negative at every one (-0.28 / -0.25 / -0.23), which is
    what ruled out the whole class in a single run.
    **AND A NULL A/B ON TOP OF A DOMINANT DEFECT IS EVIDENCE ABOUT NOTHING.**
    While that head's norms were wrong, swapping `fc`'s concat order end to
    end changed nothing and shifting every RoPE position moved the
    correlation -0.2820 to -0.2818. Both read as "not the cause"; both were
    really "the output is garbage either way". When an experiment's two arms
    are equally broken its null result is uninformative, and it does not
    announce itself -- so establish that the component works AT ALL before
    A/Bing its conventions.

58. **A NUMBER MEASURED IN ONE FILE AND ASSERTED IN ANOTHER IS A COUNT THAT
    ROTS, AND THE TIE HAS TO RUN OFFLINE.** `turbospark-model recommend`
    quotes what the memory oracles measured, so those numbers now live in
    `models.json` as `measured` blocks -- OBSERVATIONS -- while the oracles
    keep their ceilings and floors, which are ASSERTIONS with a per-row margin
    and a paragraph justifying it. Two files, one run, and nothing structural
    keeping them in step; that is exactly the shape commit `186d295`'s audit
    went looking for. `oracle_common::assert_agrees_with_catalog` is the tie,
    and the load-bearing part is that it is NOT `#[ignore]`d and needs no
    install: a contradiction fails on the edit rather than the next time
    somebody happens to have a 13 GB install on disk. It checks only rows
    whose `source` says "this port" -- requiring a catalog row for the Swift
    rows on chips nothing here has ever run would mean inventing measurements.

    **THE TRAP UNDERNEATH IT IS THAT A FROZEN PEAK IS A PEAK AT ONE CONTEXT
    AND ONE SLOT COUNT.** Both of its terms move with those: KV is a pure
    function of the window (Gotcha 40) and the slot cache is
    `slots x layers x expert_stride` (Gotcha 36). Gemma 4 reads 2,175 MiB at
    4,096/16 and 3,654 at 4,096/32, and `Auto` resolves 32 on this machine
    while the protocol pins 16 -- so an estimate at `Auto` compared against a
    frozen row reads as a 55% overestimate and is really two configurations
    being compared. Any function that computes a footprint therefore takes the
    slot policy as a PARAMETER, and any caller comparing against a measurement
    pins what the measurement pinned.

    **And a shape nobody has read resolves 16 slots BY IGNORANCE.** With no
    `ArchConfig` there is no expert stride, `Auto` divides by nothing and
    returns `DEFAULT_CACHE_SLOTS`, which is the same 16 the protocol pins --
    so the two look like agreement and a measured row looks applicable when
    nothing has established that it is. Reporting a measurement AT ITS OWN
    STATED CONFIGURATION is honest; reporting it as this machine's answer is
    not, and the difference is invisible without the check.

59. **NaN SCORES A PERFECT RESULT ON A RANK OR TOP-K INSTRUMENT, because
    every comparison against NaN is false.** Gotchas 30 and 57 record
    instruments that return a NEUTRAL value on degenerate input (`pearson`
    is 0.0 on a constant, a rank sits at `vocab/2` when unrelated). This is
    the turn of the screw past both: two instruments read their BEST
    POSSIBLE value on an all-NaN row, and nobody investigates a perfect
    score.
    Measured 2026-08-19 on the DFlash2 drafter, whose draft pass overflowed
    FP16 and returned 248,320 NaNs per row for the life of the feature.
    - `dflash_select`'s top-16 scan admits on `v <= val[k - 1]`, which NaN
      fails, so a NaN row was admitted at every candidate; then `score >
      best_score` was false at each, so the walk kept `cand[0]`, and `cand`
      was initialized to zeros. The drafter proposed token id 0 eight times
      a round for 256 rounds, with no error anywhere, and the accept-length
      table printed 0.16x -- which reads as a WEAK DRAFTER and sent a day of
      work at the loop, the context write, `fc`, the RoPE and the selector.
    - `dflash2_bisect_probe`'s `rank_of` counts `logits[i] > logits[token]`,
      which NaN also fails, so it ranked the true token FIRST on every one
      of 48 teacher-forced steps. Its "median rank 0, top-1 48/48" was
      written down as proof that the backbone, the aux capture, the norms,
      the conv and the selector were all correct. It was the instrument
      reading its ceiling on garbage.
    THE CHEAP DEFENCE is a finiteness assertion at the point a measurement
    is TAKEN, not at the point it is used: three characters of `is_finite`
    ahead of any argmax, rank or top-k over model output. `dflash_select`
    now refuses a non-finite row by name, and the drafter's frozen-digest
    test asserts finiteness BEFORE comparing the digest, because NaN hashes
    as stably as any other bit pattern and a digest alone would have frozen
    the broken drafter.

60. **A RESIDUAL STREAM'S DYNAMIC RANGE IS A PORTING AXIS WHEN THE REFERENCE
    IS BF16 AND THIS PORT IS FP16.** Every family here holds activations in
    FP16, whose largest finite value is 65,504; BF16 reaches 3e38. That
    difference is invisible on the five families brought up before DFlash2,
    whose residuals sit in the hundreds, and it is fatal on a drafter whose
    residual peaks at 113,920 -- a conv with coefficients running to ~10
    multiplying sublayer outputs in the thousands, which is the
    checkpoint's own design and not a bug to find. The overflow presents as
    NaN, i.e. as Gotcha 59, i.e. as anything but an overflow.
    CHECK THE MAGNITUDE, not just the block layout, when porting a new
    component: dequantize a few rows and look at the scale the arithmetic
    will run at. This is the fixture-dynamic-range trap (Phase G's `inf`
    logits, Phase S's squared MoE chain) arriving in the STORAGE WIDTH of a
    real model rather than in a test's inputs.
    THE FIX NEED NOT BE A WIDER BUFFER. Where the stream is read only by
    RMS norms and by its own residual add, a power-of-two scale is exact and
    costs one kernel argument: `rms_norm` is scale-invariant, so dividing
    the stream and every addend by `S` cancels at the next norm, and a power
    of two shifts the exponent while leaving the mantissa alone. BUT THE EPS
    MUST BE DIVIDED BY `S * S` -- `(x/S) / sqrt(mean(x^2)/S^2 + eps)` equals
    `x / sqrt(mean(x^2) + eps*S^2)`, so an unscaled eps behaves as if it
    were `S^2` larger, which on an embedding row (mean square ~4e-4) is a
    factor of ten rather than a rounding difference. Getting that wrong is
    the worst outcome available: finite, plausible, and wrong -- it put the
    true token at rank 13,202 where the corrected pass puts it at 0.

61. **THE SECOND HALF OF A SHARED ARCHITECTURE IS NOT COVERED BY THE FIRST
    HALF'S TESTS, AND THREE SEPARATE `== ModelFamily::QwenGdnMoe` CONDITIONS
    PROVED IT IN ONE SESSION.** `qwen35` (dense) and `qwen35moe` share a
    baseline's every behavioural field, one decode flow, one name table and
    one GGUF converter -- so a condition naming only the MoE half reads as
    correct and is a latent bug for exactly as long as no dense checkpoint
    exists. Bringing up `ornith-ai/Ornith-1.5-9B` found all three:
    `arch_from_gguf` assigned `linear_attention` for the MoE half alone (the
    dense file kept `qwen_gdn_dense_27b()`'s `num_v_heads: 48` against a real
    32, deriving `qkv_dim` 10240 for an 8192-row tensor); `v_head_axis`
    returned `None` for it (no de-interleave, so Gotcha 33's fluent word
    salad); and `gguf_config`'s mask arm refused it outright.
    **THE FIRST IS THE ONE TO INTERNALISE, because the MoE half CANNOT SEE
    IT**: `qwen_gdn_moe_35b_a3b()` declares 32 and the real file says 32, so
    the missing assignment read the right answer from the baseline either
    way. Gotcha 37's shape, with the injective mapping being one family to
    one checkpoint. GREP FOR THE FAMILY NAME before adding a checkpoint of a
    shared architecture's other half, and prefer `matches!(family, A | B)`
    over `== A` wherever the two share the code below it.
    A COROLLARY FOR DIAGNOSTICS: a probe that RANKS tensors must sort NaN
    rather than `.expect("finite")`. A non-finite score is the most
    informative outcome available and a panicking comparator destroys it --
    Gotcha 59's rule (NaN reads as a perfect score) arriving one layer out,
    where NaN instead reads as no score at all.

62. **READ A PROPERTY OFF THE ARTIFACT, NOT OFF THE NOTE ABOUT THE ARTIFACT.**
    A published norm range of "0.0052 to 2.5433" was carried from a handoff
    into a measured section describing a DIFFERENT file: two control vectors
    from one publisher, one for a model this port runs and one for a model it
    cannot, differing only by path. The real range was 0.0017 to 0.4268 and
    the quoted figure was off by 6x, in a paragraph whose whole argument is
    that the number explains the result. It cost ONE command to check and was
    checkable from the moment it was written. The tell is a number that
    arrived as PROSE rather than as OUTPUT: if a figure was not printed by the
    run being described, re-derive it before publishing. Same family as
    Gotchas 30, 38, 57 and 59 -- a value that reads plausibly, is wrong, and
    is cheap to falsify.

63. A frozen digest that stops reproducing is not necessarily a regression: bisect to AND INCLUDING the commit that wrote it. Moved to
    [crates/bench/CLAUDE.md](crates/bench/CLAUDE.md) Gotcha 24.

64. **`ALLOWED_CACHE_SLOTS` SAYS A SLOT COUNT IS LEGAL; IT SAYS NOTHING ABOUT
    WHETHER CHUNKED PREFILL CAN PLACE IT.** `--expert-cache-slots 8` on the
    real Gemma 4 install (top_k 8) panics on the FIRST multi-token prompt
    with `expert cache cannot place requested misses`
    (`crates/streaming/src/expert_cache.rs:196`), reached through
    `prefill_chunk_real_gemma4` -> `prefill_micro_batch_gemma4_inner` ->
    `plan_experts_cached`, never through plain single-token decode. Measured
    2026-08-28, deterministic on 3 of 3 attempts against a 36-token prompt.
    THE MECHANISM, verified with `RUST_BACKTRACE=full` rather than assumed:
    `routed_pipeline_banks(slots, top_k)` (`moe_prefill_pipeline.rs`) picks
    `ROUTED_BANKS`-deep pipelining only when `slots >= 2 * top_k`, and falls
    back to `banks = 1` otherwise -- which reads like a guard that makes the
    narrower case safe. IT DOES NOT. In every per-token routed prefill loop
    that calls it (`families/{gemma4,gptoss,llama}/prefill.rs`, all three
    sharing the helper), `previous_slots` is captured from the prior
    token's `used` set and passed as `protect` to the NEXT token's plan
    UNCONDITIONALLY, regardless of `banks`. The `banks == 1` branch does
    call `retire_routed` before planning, so the previous command buffer is
    provably no longer in flight by the time `protect` is consulted --
    `protect`'s stated purpose ("slots a command buffer still in flight is
    reading") has already been satisfied by the wait, and the reservation
    is stale conservatism, not a safety requirement. At `slots == top_k`
    exactly (the only `ALLOWED_CACHE_SLOTS` value that fails
    `slots >= 2 * top_k` for an 8-expert-per-token model), a stale
    reservation of one whole token's worth of slots leaves ZERO room for
    the next token's misses, and two adjacent tokens routing to disjoint
    experts out of 128 is the common case, not an edge one -- so this is a
    near-certain failure on any real multi-token prompt, not a rare race.
    Two things worth carrying. First, this is invisible to a decode-only
    smoke: `slot.protect` is genuinely empty on the sequential decode path
    (`families/gemma4/moe.rs`'s own comment is accurate there), so a
    workflow that never chunks its prefill -- or a prompt short enough to
    be one token -- cannot see it, which is exactly Gotcha 51's shape one
    layer out (a test whose inputs cannot reach the mutation proves
    nothing). Second, this is why `docs/BENCHMARKING.md`'s per-family
    oracles and `crates/bench`'s protocol runs default to 16 slots and
    never to 8: at 16, `16 >= 2 * 8` holds, the pipelined branch engages,
    and the failure mode does not exist at that width. Do not read a
    passing oracle at 16 slots as evidence that 8 works.

    **THIS BIT A SECOND FAMILY ON 2026-09-05, AT THE DEFAULT SLOT COUNT, AND
    THAT IS THE PART TO CARRY.** Everything above is written as though 8
    slots were the exotic configuration that reaches the bug. It is not the
    slot count that matters, it is `slots < 2 * top_k`, and the gemma4 story
    reads as being about `8` only because that family routes top-8. The real
    `qwen4-reap288` install routes **top-10**, so the bench's own pinned
    `PROTOCOL_EXPERT_CACHE_SLOTS` of 16 fails `16 >= 20`, degrades to
    `banks == 1`, reserves the previous token's 10 slots and leaves 6 places
    for a token that can miss on 10. It panics on the first multi-token
    prompt with the identical message, on the DEFAULT configuration of a
    supported family, reached by an ordinary `turbospark-bench --model` run.

    **The fix is the one this entry already argued for and nobody had
    applied**: at `banks == 1` the branch calls `retire_routed` BEFORE
    planning, so nothing is in flight when `protect` is read and the correct
    reservation is the EMPTY set. `families/qwen4/prefill.rs` passes
    `HashSet::new()` there since 2026-09-05, pinned by
    `the_one_bank_fallback_reproduces_the_sequential_logits`, whose mutation
    (reverting to the unconditional `previous_slots`) reproduces the real
    install's exact panic on the synthetic fixture.

    **`families/{gemma4,gptoss,llama}/prefill.rs` STILL PASS IT
    UNCONDITIONALLY.** They are unreachable at their own default slot counts
    (top-8 of 16 leaves exactly 8, which is enough), so this is latent there
    rather than live, and it stays open deliberately: changing them means
    re-running three families' byte-identity gates for a path none of them
    takes by default.

    **AND NOTE WHAT LET IT SHIP.** `real_forward_qwen4_chunked.rs` carried a
    case named `a_cache_too_small_to_pipeline_still_reproduces_the_sequential_logits`
    which passes `2 * TOP_K` -- a value that SATISFIES `>=` and therefore
    pipelines. The fallback had a test named after it and no test covering
    it, which is Gotcha 51's shape (a fixture whose inputs cannot reach the
    thing it claims to check) hiding behind a correct-sounding name. When a
    threshold is `>=`, a test at exactly the threshold is on the WRONG side
    of it.

65. **WHEN A REFERENCE KERNEL DIFFERS ON SEVERAL AXES AT ONCE, CHANGING ONE
    IS NOT A CONTROLLED EXPERIMENT -- IT IS A THIRD, WORSE KERNEL.**
    Measured 2026-08-29. MLX's `qmm_t_impl` beats this port's
    `dequant_int4_gemm_mma` by 3.0x and differs from it in three ways at
    once: 128 threads in four SIMD groups against one, `BM` 16 to 128
    against `kMmaTile = 8`, and BOTH operands staged into threadgroup
    memory against only the weights. Staging `x` was isolated and built
    first, on the reasoning that a transposed `simdgroup_load` from device
    with row stride N is a strided gather in the innermost loop and
    therefore the obvious cause. **It lost 3.3x to 5.9x, on every shape and
    every width, with the penalty GROWING in B.** Apple's tile load handles
    that access fine; hand-staging the same bytes through 32 LANES is a
    serial copy of thousands of halfs per n-block. MLX stages `x` and wins
    because it has 128 threads to do the staging and a `BM` of 32 to 128 to
    amortize it over -- so staging is a CONSEQUENCE of the wider
    threadgroup, not a separate lever, and the axes are not orthogonal.
    The rule: before isolating one difference against a faster reference,
    ask whether that difference is load-bearing ON ITS OWN or only in
    combination. When the answer is "only in combination", a one-variable
    A/B measures a kernel nobody would ship and its null (or negative)
    result says nothing about the question. This kernel has now punished
    the same instinct three times -- its header already records that
    widening the staged K block eightfold "changed nothing" and that
    running past M=16 to 32 and 64 "changed nothing".

    **AND THE KERNEL'S OWN EXPLANATION OF ITSELF SURVIVED BECAUSE NOBODY DID
    THE ARITHMETIC.** Its header attributed the plateau to dequant work being
    "independent of B", and the natural fix that follows is a bigger weight
    tile. One threadgroup there dequantizes `8 * N` elements to produce
    `8 * B` outputs, so dequant per output is `N / B` -- a function of the
    TOKEN count, not of the weight rows per threadgroup. MLX's is `K / BM`
    with `BM` also the token tile: **the same number at the same width.**
    The kernel already runs at B=64, already sits where MLX sits at BM=64,
    and is still 3.5x behind. So the amortization story is refuted by two
    lines of counting, the `kMmaTile` lever it implies is refuted with it,
    and the real mechanism remains unidentified. Do the counting before
    accepting a performance explanation that arrives as prose -- Gotcha 62's
    rule, applied to a mechanism rather than to a measured value.

    **THE COROLLARY IS THAT A DEAD-END VERDICT INHERITS THE SCOPE OF THE
    SHAPE IT WAS MEASURED AT**, which is Gotcha 62's shape at the level of
    a kernel rather than a number. "`simdgroup_matrix` is a measured dead
    end" was true of `kMmaTile = 8` with one SIMD group and was written as
    though it were true of matrix hardware. The tell is that nothing about
    the verdict had to change for it to become misleading, only the arrival
    of a reference measured at a different tile.

    **THE PRESCRIPTION WAS FOLLOWED AND THE COMBINED CHANGE STILL LOST, WHICH
    CLOSES THIS ENTRY'S OWN LOOP.** 2026-09-05: four SIMD groups and
    `FC_MMA_STAGE_X` moved TOGETHER, as this gotcha said they had to. The
    re-tile is worse than the narrow tile at every width up to 32 and reads
    2.10x the exact kernel at M=16 (`ROADMAP.md` Do Not Revisit 16). Two
    things worth carrying past the verdict. **"Measure them together" is
    necessary and not sufficient**: the combined arm can still lose, and when
    it does it retires the whole cluster rather than one lever -- staging
    lost INSIDE the wide shape too, which refutes entry 13's reversal
    condition directly rather than leaving it open. And **the mechanism the
    combination was supposed to fix turned out not to be the cost at all**:
    with the dequant deleted on both tiles the floors are identical, so the
    threadgroup width the header had narrowed to was never the term. A
    correct experimental design can sit on top of a wrong diagnosis, and only
    the measurement separates them.

66. **AN INSTRUMENT THAT SERIALIZES WHAT IT MEASURES CANNOT PRICE A HOT
    PATH, AND THE COST DOES NOT FALL WITH THE INPUT.**
    `TURBOSPARK_DISPATCH_PROFILE=1` is the only surface in this repo that
    attributes GPU time BY KERNEL NAME, so it is the obvious answer to
    "what share does kernel X hold". It "waits on every command buffer at
    commit" (`crates/gpu/src/dispatch_profile.rs`), which destroys exactly
    the pipelining that makes prefill fast. Measured on the real
    `qwen38-27b` install: 2,940 tokens ran 17.5 min without reaching the
    report, 582 tokens 17.5 min, ~150 tokens over 12 min. **Shortening the
    prompt is not the fix** -- that was the natural next move and it bought
    nothing, because the overhead is per command buffer rather than per
    token. A session that starts down this road loses an hour.
    What worked instead was a targeted bench: time the suspect kernel
    against its neighbours at the REAL shapes and weight by the real layer
    counts, no model and no install
    (`crates/gpu/tests/gdn_prefill_share_bench.rs`, **0.45 seconds**, and
    its answer of 4.66% agreed with an independent traffic calculation to
    0.2 points). The general form: the profiler is for finding the kernel
    you did NOT suspect; once you have a suspect, price it directly. And
    prefer two cheap independent methods over one expensive one -- their
    agreement is what makes a share believable, and neither alone was.

67. **SECURITY DOCUMENTATION MUST DESCRIBE THE ACTUAL TRUST BOUNDARY.**
    Loopback limits network exposure, but it does not isolate a listener from
    other local processes, and Tailnet ACLs do not replace application
    authentication. When documenting a server or ABI, state the available API
    key configuration and the limits of each bind mode together.

68. **RUN THE RUST BUILD BEFORE STAGING THE SWIFT BINDING.** The Swift package
    can hide a Rust baseline failure because its staged header and static
    library are copied only after a release build. In this tree, a comparison
    between a `String` field and `&str` in the repack path kept the workspace
    build red while Swift-only checks looked unrelated. Keep the type and
    format gates green before interpreting Swift test results.

69. **`make swift-lib` CAN PASS WHILE `make swift-test` STILL CANNOT LINK.** The
    staging script removes `_rust_eh_personality` from the Rust static library
    to avoid collisions when the app links multiple Rust archives. The
    standalone `swift/TurboSpark` package then has no remaining definition for
    that symbol and its test link fails, even though the Rust build and staging
    succeeded. Treat this as a baseline Swift packaging failure, not as a
    regression in an unrelated Swift source change, and record it before
    merging such a PR.

## Per-Crate Documentation

When working on code inside a specific crate, refer to that crate's `CLAUDE.md` file for crate-specific architecture, key modules, dev commands, and localized gotchas. The Swift tree is not a crate and has one too:

- [`crates/bench/CLAUDE.md`](crates/bench/CLAUDE.md): Throughput benchmark harness, mach memory sampler, frozen protocol, memory oracle test rules.
- [`crates/catalog/CLAUDE.md`](crates/catalog/CLAUDE.md): the curated model table, the header-only Hugging Face probe, the install driver, and the `~/.turbospark` store.
- [`crates/cli/CLAUDE.md`](crates/cli/CLAUDE.md): CLI binaries (`turbospark-check`, `turbospark-model`), process entry point, real model smoke tests, interactive chat REPL.
- [`crates/compute/CLAUDE.md`](crates/compute/CLAUDE.md): CPU reference kernels (RmsNorm, RoPE, Attention, Quant), numerical ground truth for GPU tests.
- [`crates/core/CLAUDE.md`](crates/core/CLAUDE.md): Shared primitives (`TokenId`, `LogitValue`), runtime configuration, allowed sets, chunk sizing.
- [`crates/ffi/CLAUDE.md`](crates/ffi/CLAUDE.md): the C ABI for native GUI hosts, its ownership and threading contract, and the Swift package over it.
- [`crates/gpu/CLAUDE.md`](crates/gpu/CLAUDE.md): macOS Metal context, pipeline caches, MSL shaders, KV cache, zero-copy weights, profiling flags.
- [`crates/image/CLAUDE.md`](crates/image/CLAUDE.md): the Z-Image-Turbo pipeline, packed component installs, the CPU reference and Metal image backends, and the opt-in parity gates.
- [`crates/invocation/CLAUDE.md`](crates/invocation/CLAUDE.md): Pure CLI argument parser, `InvocationRequest`, 5-place rule for adding new flags.
- [`crates/model-io/CLAUDE.md`](crates/model-io/CLAUDE.md): Manifest validation, architecture baselines, packed expert layout, mmap resident weight index.
- [`crates/repack/CLAUDE.md`](crates/repack/CLAUDE.md): Safetensors header parsing, ranged HTTP downloads, `.gturbo` writer, synthetic model builders.
- [`crates/runtime/CLAUDE.md`](crates/runtime/CLAUDE.md): Raw completion generation loop, `LogitProducer` contract, `RealForwardRunner` decode engine.
- [`crates/selection/CLAUDE.md`](crates/selection/CLAUDE.md): Candidate token selection, temperature/top-k/top-p shaping, repetition penalty, logits contract.
- [`crates/server/CLAUDE.md`](crates/server/CLAUDE.md): the `turbospark-server` HTTP server (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, `/v1/models`), Axum handlers, SSE streaming, `anyllm_translate` wire types.
- [`crates/streaming/CLAUDE.md`](crates/streaming/CLAUDE.md): Routed expert `pread` streamer, LFU/LRU slot cache policy, chunked reads on a persistent `read_pool`, macOS `F_RDADVISE` hints.
- [`crates/tokenizer/CLAUDE.md`](crates/tokenizer/CLAUDE.md): Tokenizer wrapper (`MfTokenizer`), chat dialects, Jinja template rendering, stop matcher, fixture token IDs.
- [`crates/vision-io/CLAUDE.md`](crates/vision-io/CLAUDE.md): portable vision preprocessing -- PIL-bicubic smart resize, patchify's transposed inner order, the three position tables, and the mlx-vlm oracle fixtures.
- [`crates/window-fit/CLAUDE.md`](crates/window-fit/CLAUDE.md): Pure conversation window fitting (`fit_conversation_window`), turn dropping logic.
- [`swift/CLAUDE.md`](swift/CLAUDE.md): the two SwiftPM packages over `crates/ffi` -- the binding's serial-queue-not-actor cancel design, the staging step every Swift build depends on, and the macOS app's state layer, tool execution and on-disk stores. Read it WITH `crates/ffi/CLAUDE.md`: the two halves of the boundary are documented on opposite sides of it and neither is complete alone.

## Layout

Workspace directory structure and crate layout:

```
.
+-- Cargo.lock         # lockfile committed for reproducible workspace builds
+-- Cargo.toml         # workspace manifest declaring members and workspace metadata
+-- AGENTS.md          # developer guide and gotchas (CLAUDE.md is a symlink to this)
+-- CHANGELOG.md       # project changelog and release notes
+-- CLAUDE.local.md    # local developer notes (gitignored)
+-- DEVIATIONS.md      # scaffolded vs fully wired feature inventory
+-- LICENSE            # MIT license
+-- NOTICE             # third-party material: what was vendored, from where
+-- Makefile           # build, test, fmt, clippy wrapper targets
+-- README.md          # repository overview and quickstart
+-- ROADMAP.md         # forward roadmap + descope record (gitignored)
+-- rust-toolchain.toml # toolchain pin (stable Rust 1.82+)
+-- crates
|   +-- bench          # turbospark-bench binary & harness (throughput benchmark)
|   +-- catalog        # the model catalog, the HF probe & the install driver
|   +-- cli            # turbospark-check & turbospark-model binaries (process entry points)
|   +-- compute        # CPU reference kernels & compute strategy marker
|   +-- core           # shared primitives (TokenId, LogitValue), allowed runtime-knob sets, chunking
|   +-- gpu            # Metal pipeline cache & GPU kernel dispatches (macOS only)
|   +-- invocation     # CLI argument parsing, request assembly & exit status routing
|   +-- model-io       # manifest validation, packed-expert layout, resident index & mmap
|   +-- repack         # safetensors + GGUF header parsing, ranged downloads, int4/8 repack, gturbo writer
|   +-- runtime        # raw-completion prefill+decode loop & RealForwardRunner (macOS)
|   +-- selection      # token sampling (temperature, top-k, top-p, repetition penalty, choose)
|   +-- server         # OpenAI-compatible Chat Completions HTTP server (axum)
|   +-- ffi            # C ABI over the engine for a native GUI host (staticlib + turbospark.h)
|   +-- streaming      # pread-based expert streamer, LFU/LRU slot cache & read pool
|   +-- tokenizer      # tokenizer wrapper, chat templates (text/Jinja), stop matcher, DSL parser
|   +-- vision-io      # portable vision preprocessing: decode, PIL-bicubic smart resize, patchify, position tables
|   \-- window-fit     # deterministic conversation-window fitting & turn dropping
+-- swift
|   +-- TurboSpark     # SwiftPM package wrapping crates/ffi (session class on a
|   |                  # serial queue -- NOT an actor, see swift/CLAUDE.md
|   |                  # Gotcha 1 -- plus AsyncStream and the catalog)
|   \-- TurboSparkApp  # SwiftUI chat app; verifies the binding end to end
+-- scripts
|   +-- extract_direction.py # per-layer steering direction from two capture sets
|   +-- ffn_sparsity.py# dense FFN activation sparsity probe
|   +-- ggml_mxfp4_oracle.c # ggml MXFP4 oracle generator
|   +-- ggml_q5_k_oracle.c  # ggml Q5_K oracle generator
|   +-- ggml_tables.c  # ggml IQ codebook table generator
|   +-- kld.py         # cross-engine KL vs mlx-lm (reads tests/logit_dump.rs's output)
|   +-- kld_llamacpp.py# the same, vs llama.cpp on the same GGUF bytes (Gotcha 34)
|   +-- kld_mlx_affine.py # the same, vs MLX at ONE or TWO bits (the first needs the PrismML mlx fork)
|   +-- llamacpp_logits.c # its harness: ids in, full-vocab logits out, via libllama
|   +-- mlx_1bit_oracle.py # MLX 1-bit affine reference oracle generator
|   +-- mlx_2bit_oracle.py # MLX 2-bit affine reference oracle generator
|   +-- mlx_prefill.py  # cross-engine PREFILL throughput vs mlx-lm, same machine
|   +-- mlx_qmm_reference.py # the same one level down: MLX's own c(M) at this port's GEMM shapes
|   +-- mtp_bisect.py  # MTP drafter norm & agreement bisection script
|   +-- parity.sh      # head-to-head protocol run against the Swift MferenceCLI
|   +-- qwen3vl_vision_oracle.py # probes mlx-vlm for crates/vision-io's five golden fixtures
|   +-- phasediff.sh   # bucket-level decode phase diff against the Swift engine
|   +-- power.sh       # watts & joules-per-token over the protocol (needs sudo)
|   +-- router_hist.py # expert routing activation histogram analyzer
|   +-- router_window.py # expert cache temporal windowing analyzer
|   +-- skill_state_probe.py # SKILL.state valid-patch rate vs append-only (docs/SKILL_STATE.md)
|   \-- swift-lib.sh   # staticlib + turbospark.h build helper for SwiftPM
\-- docs
    \-- (see the doc index at the top of this file; a count here rots)
```

## References and measurement scripts

`scripts/` holds the measurement surfaces that cannot be a `cargo test`:
two need the Swift engine built next door, `kld.py` needs a 14.6 GB
reference checkpoint plus a Python environment, and `kld_llamacpp.py` needs
a 26.9 GB GGUF plus a llama.cpp install (brew's; it compiles
`llamacpp_logits.c` against that header on first run and caches the binary
in `/tmp`). `kld.py` runs mlx-lm under `uv run --with mlx-lm`, an ephemeral
env, so no Python dependency is installed globally or enters this
workspace; `kld_llamacpp.py` needs only numpy and reuses `kld.py`'s
divergence and perplexity functions rather than restating them.

**FINDING A REFERENCE: CHECK mlx-vlm AS WELL AS mlx-lm, AND CHECK WHETHER THE
COMPONENT SHIPS ALONE.** Every script above uses mlx-lm, which makes it the
obvious place to look and is not always the right one: mlx-lm 0.31.3 has no
`qwen3_5_mtp`, while mlx-vlm 0.6.14 implements it at
`speculative/drafters/qwen3_5_mtp/`. A sub-component may also be published as
its own checkpoint (`mlx-community/Qwen3.8-27B-MTP-4bit`, 239 MB), far cheaper
to load than its parent and named by the config's `model_type`. READING a
reference settles convention questions that measuring them cannot -- 2026-08-14
for mrope, 2026-08-18 for the MTP head's five design choices. Note
`safetensors.numpy` CANNOT decode BF16 (`TypeError: data type 'bfloat16' not
understood`); parse the container directly, which is ~15 lines and keeps the
decoder independent anyway (Gotcha 48).
And READ THE PRIOR ART YOUR OWN DOCS NAME. A "take no source" note is a
LICENSING decision about copying and never an instruction not to look: the MTP
head's norm convention sat one grep away in the project whose headline result
`docs/MTP_SPECULATIVE.md`'s first sentence quotes, and was rediscovered by
two hours of bisection instead.
**AND A TRUE FACT ABOUT A REFERENCE IS NOT A READING OF IT.** The control
vector numbering was DERIVED from two correct facts about llama.cpp and came
out one block off, under an honest `UNVERIFIED` that made it look
measured-open rather than reasoned-and-wrong; the refutation was internal from
the first commit (`crates/repack/CLAUDE.md` Gotcha 11). Check a derived
convention against the line that IMPLEMENTS it -- brew ships llama.cpp's
headers to `/opt/homebrew/include` and its sources are one
`raw.githubusercontent.com` fetch away, so this class of question costs no
download at all.

## Code discovery, search, and explore agent harness

This repository uses Syntext as its indexed code search engine (see `swift/docs/SYNTEXT.md` and `swift/docs/SWIFT_TOOLS.md`).

- **Default search tool**: Use `grep_search` for code discovery, symbol lookup, and pattern matching. It uses the project's Syntext index for sub-millisecond regex and literal search with line numbers and context lines in ripgrep format.
- **Do not shell out for searching**: Never shell out to `grep`, `find`, or `ripgrep` via Bash or terminal execution when `grep_search` or `Glob` is available.
- **File locating and reading**: Use `Glob` (`list_directory`) for file path patterns and `FileRead` (`read_file`) for inspecting specific files or line ranges.
- **Explore subagent contract**:
  - The `explore` agent (`AgentManager+BuiltIns.swift` and `.turbospark/agents/explore.md`) is strictly read-only.
  - Allowed tools: `FileRead`, `Glob`, `Grep`, `grep_search`, `Bash` (strictly read-only commands: `ls`, `git status`, `git log`, `git diff`), `WebFetch`, `WebSearch`.
  - Disallowed tools: all write and edit tools (`write_file`, `edit_file`, `apply_patch`, `notebook_edit`), subagent creation (`agent`, `subagent`, `task`), planning mode tools (`enter_plan_mode`, `exit_plan_mode`), and artifact/worktree tools (`todowrite`, `enter_worktree`, `exit_worktree`).
  - Project instructions are omitted (`omitsProjectInstructions` / `omit_claude_md: true`) to preserve context window and reduce prefill latency for fast search fan-out.
  - All findings must report absolute paths and avoid emojis.
- **Plan subagent contract**:
  - The `plan` agent (`AgentManager+BuiltIns.swift` and `.turbospark/agents/plan.md`) is strictly read-only for designing architectural and implementation plans.
  - Allowed tools: `FileRead`, `Glob`, `Grep`, `grep_search`, `Bash` (strictly read-only commands: `ls`, `git status`, `git log`, `git diff`), `WebFetch`, `WebSearch`.
  - Disallowed tools: all write and edit tools (`write_file`, `edit_file`, `apply_patch`, `notebook_edit`), subagent creation (`agent`, `subagent`, `task`), planning mode tools (`enter_plan_mode`, `exit_plan_mode`), and artifact/worktree tools (`todowrite`, `enter_worktree`, `exit_worktree`).
  - Structured process: Understand Requirements, Explore Thoroughly (using Syntext `grep_search` and `Glob`), Design Solution, Detail the Plan.
  - Required output ending: Concludes with "### Critical Files for Implementation" listing 3-5 critical files. All findings avoid emojis.
- **General-purpose subagent contract**:
  - The `general-purpose` agent (`AgentManager+BuiltIns.swift`, `.turbospark/agents/general-purpose.md`, and `.claude/agents/general-purpose.md`) is the fallback execution and research subagent.
  - Unlike `explore` and `plan`, it is unconstrained (has no tool ceiling and no disallowed tools), enabling multi-step task execution, file editing, and command running.
  - Retains project instructions (`omitsProjectInstructions == false`) for full codebase and architectural context.
  - Uses Syntext `grep_search` for fast indexed code and pattern search across large codebases.
  - Completes tasks fully without gold-plating or leaving half-done, returning a concise report with essentials.

<!-- BEGIN AGENT-CONFIG:mf -->
Before exploring this codebase, run `mf search "<question>" --field notes`.
Before finishing, write what you learned as a page with `mf write <draft> --field notes`, or stage it with `mf raw add --field notes`.
<!-- END AGENT-CONFIG:mf -->
