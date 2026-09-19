# Build and verification reference

This page contains the detailed build, test, real-model, release, and benchmark-adjacent verification reference moved out of the always-loaded root instructions. Keep the root file focused on invariants and the shortest safe command matrix.

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
workspace suite green (`crates/runtime/AGENTS.md` Gotcha 11). Run the gates
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
# already covered (`crates/ffi/AGENTS.md` Gotcha 11).
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
# in both packages (`swift/AGENTS.md` Gotcha 3), so each one pays a full Swift
# rebuild. Iterating on SwiftUI alone, call SwiftPM directly and skip it:
# `cd swift/TurboSparkApp && swift run TurboSparkApp`.
make swift-app
make swift-demo

# The RELEASE artifacts, and the only way to get a real `.app` out of this
# tree: `swift build` emits a bare executable, so the Info.plist, the bundle
# identifier and the resource-bundle copy all live in the script rather than
# in an Xcode project (`swift/AGENTS.md` Gotcha 12). `dmg` additionally MOUNTS
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

# TOOL-CALL GUARDRAILS, on by DEFAULT (`crates/server/AGENTS.md` Gotcha 18).
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

- [.claude/docs/model-gates.md](model-gates.md): per-family memory oracles and quality gates.
- [.claude/docs/checkpoint-installs.md](checkpoint-installs.md): installing the real checkpoints.
- [.claude/docs/cross-engine-kl.md](cross-engine-kl.md): this port against mlx-lm and llama.cpp.
- [.claude/docs/power-measurement.md](power-measurement.md): watts and joules per token.
- [.claude/docs/diagnostics.md](diagnostics.md): header probes, convention checks, decode-path probes.

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
