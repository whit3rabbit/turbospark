# AGENTS.md

CLAUDE.md is a symlink to this file.

Conventions, gotchas, and commands for working in this Rust workspace,
a behavior-compatible port of the Mference Swift inference engine (see
`ROADMAP.md` for the forward roadmap and descope record, and
`DEVIATIONS.md` for what is scaffolded rather than fully wired). Keep all code, comments, and docs
ASCII: no emojis and no em dashes (project rule).

`docs/TESTING.md` covers what the suite proves and how tests are gated
(macOS, `#[ignore]`d, env-var). `docs/POWER_BASELINE.md` covers watts and
joules-per-token and is the only page here measured on battery.
`docs/BENCHMARKING.md` covers the three
`turbospark-bench` modes, how peak memory is measured, and the memory
oracle that asserts this port against per-chip baseline rows (mostly the
published Swift numbers; see `docs/BENCHMARKING.md` for which rows are
Swift parity claims and which are this port measuring itself).
`docs/EXPERT_ROUTING.md` records the measured-negative answer to
domain-restricted expert sets (coding routes to ~67 of 128 experts per
layer, not a prunable region) -- read it before proposing expert pruning
or pinning. `docs/SPECULATIVE_DECODING.md` does the same for speculative
decoding and DFlash: worth ~1.1x here at a SMALL block size, which inverts
the datacenter result, because 19% of decode compute has no weights to
amortize -- read it before proposing a drafter, and note that the lever it
identifies is the batched-MoE kernel rather than the drafter.
`docs/ACTIVATION_SPARSITY.md` is the dense-side sibling of the expert-routing
negative: Muse Glimmer's FFN spreads 95% of its activation mass over 91% of
its neurons, so PowerInfer-style neuron caching and streaming is a measured
dead end here -- read it before proposing to shrink a dense model's working
set by "loading only what the token uses".
`docs/MTP_SPECULATIVE.md` answers the speculative question for the DENSE
family, where `docs/SPECULATIVE_DECODING.md` answers it for the MoE one, and
the two now disagree: on `qwen3_5` a checkpoint's own multi-token-prediction
head is a nearly free drafter (~1.5% of a pass) against a 6.4%
un-amortizable floor and no expert-union term, so the ceiling is ~2.07x and
a good drafter is worth 1.35-1.58x. **`docs/MTP.md` is the FACTS half of that
pair** -- the head's architecture and every measured number, with nothing
projected in it; read it before changing the head, and read
`MTP_SPECULATIVE.md` before re-costing the decision. **MEASURED END TO END
2026-08-18 AND THE
SHAPE HELD**: block 2 pays **1.66x**, block 4 1.47x, block 8 1.17x, block 15
loses. Single-step acceptance is 0.94, and the projected 1.35-1.58x band is
cleared everywhere below block 15; the small-block conclusion survives because
verify cost scales close to linearly in M while acceptance decays.
**THE BATCHED VERIFY THOSE RATIOS PROJECT WAS THEN BUILT AND CLOCKED, AND IT
PAYS 1.44x AT BLOCK 2 RATHER THAN 1.66x.** The 13% is a term no composite on
either page has: a rejected batched round cannot stop early, so on a family
with a recurrent half it restores a whole gated-DeltaNet snapshot and replays
the accepted prefix, and the odds of paying that rise with the block (10% at
2, 84% at 8, 98% at 15). So batching BEATS a sequential verify at block 2 and
LOSES to it at 8 and 15, which inverts the projection's shape and gives the
small-block answer a third independent leg. Add a rollback term before
trusting any block-size table on a recurrent architecture. Getting
there needed a real fix, twice over: ALL SEVEN of the head's norms are
CENTERED (`x * (1 + w)`) where the trunk's are plain, and reading them plainly
put the true token at median rank 248,308 of 248,320. That is Gotcha 50 on a
second family; the fix dispatches `rmsnorm_bf16w_centered`, which already
existed, plus `rmsnorm_bf16w_perhead_centered`, which did not, and took top-1
agreement from 0/24 to 24/24. **The per-head pair was deferred for a session
on the strength of 23/24 top-1 and was worth 21 to 75 percent of the
speedup** -- a saturated scalar cannot bound a curve, and the earlier reading
of "the chain saturates at ~2.05" was an artifact of that deviation rather
than a property of the head. **It also supersedes that older page's
`c(M)` table in both directions**: `dequant_int4_gemm_simd` was missing the
function-constant specialization `46617c6` gave the GEMV, and fixing that
plus bounding its unroll roughly HALVED `c(M)` on every shape. **THE MoE
VERDICT WAS RE-DERIVED 2026-08-18 AND DID NOT MOVE** (block 4 pays
1.14-1.19x, block 8 is 0.95-1.00x, block 16 loses, against a recorded
1.14 / 0.97 / 0.87), which makes the PAIR of pages the thing to read: one
kernel fix, measured the same week, is decisive on the dense family and
worth about two points on the MoE one, because the GEMV it improves is
52.7% of MoE decode compute while 19% cannot amortize at all and a further
26% is a kernel nobody has written. Cost an optimization by the terms it
does NOT touch. Read both before proposing
MTP, DFlash, EAGLE or Medusa -- and note the reversal recorded at the top of
the MTP page, where a composite built on an unoptimized kernel measured the
kernel rather than the question. **`simdgroup_matrix` is CLOSED**, measured
the same day: built as `dequant_int4_gemm_mma` it loses at every batch width
and plateaus at ~0.5 past M=16, because a packed INT4 run cannot be
`simdgroup_load`ed and the dequant-into-threadgroup work that forces is
independent of B while the MACs matrix hardware accelerates scale with B.
The exact kernel is both faster and bit-identical, so the losslessness trade
that lever was priced in does not have to be made.

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

# The demo chat app. Model picker, streaming transcript with collapsible
# reasoning, a Stop button, and a status footer. Deliberately minimal: it
# exists to verify the binding, not to be a product.
make swift-demo

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
# checked on this machine at all. These eight can, and are green. Run it when
# touching a cfg, a dependency table, or anything unsafe. See Gotcha 8.
cargo check --target x86_64-unknown-linux-gnu \
  -p turbospark-core -p turbospark-compute -p turbospark-model-io \
  -p turbospark-streaming -p turbospark-selection -p turbospark-invocation \
  -p turbospark-window-fit -p turbospark-gpu

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

# The catalog's rot guard: does every row still describe a real artifact? A
# file list and a HEAD per row, ~26 s for the whole table, downloads NOTHING.
# Run it after adding or editing a row and paste its published byte figure
# back -- `download_bytes` is the only fingerprint a `main`-pinned row has.
cargo test -p turbospark-catalog --test catalog_network --release -- --ignored --nocapture

# Run the server against a real install (macOS; one runner per process, so
# requests are served one at a time). It serves OpenAI
# `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`. Add
# `--bind tailnet` to bind this machine's Tailscale IPv4 address instead of
# loopback (no auth, no TLS: the Tailnet ACL is the only access control).
# `--model` takes a catalog ALIAS as well as a directory, through the same
# `catalog::resolve_model_arg` `turbospark-check` uses; the startup line
# prints what an alias resolved to.
cargo run --release -p turbospark-server --bin turbospark-server -- --model ~/models/gemma4.gturbo
cargo run --release -p turbospark-server --bin turbospark-server -- --model gemma4

# Point an Anthropic-native client straight at it, no proxy in between.
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true claude

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

# The memory oracle: asserts endOfTurn on every protocol case, peak
# footprint under the ceiling, no growth on a replayed warm case, and
# (where a row exists) a decode tok/s floor. Each row records whether it
# came from Swift's docs/BENCHMARKS.md or from this port measuring itself
# -- printed every run. Skips with a note if the env var is unset. Takes
# ~10 minutes. See docs/BENCHMARKING.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture

# The SEVENTH family's two gates (dense, MLX INT4). Its protocol runs at
# 8,192 context AND a 2,048 budget, because the model REASONS to a `to=self`
# message before its `to=user` answer -- the three cases need 1,132 / 1,498 /
# 1,552 sampled tokens, so the SHORT one already exceeds the shared 1,024.
# The quality gate needs an ASSISTANT PREFIX (` to=user<|message|>`): the
# generation prompt ends at `<|start|>assistant` and the model's next
# emission is a RECIPIENT, so a reference answer spliced in raw lands in no
# message at all -- gpt-oss's 148,421.76 failure, one family over.
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-bench --test museglimmer_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-bench --test museglimmer_quality_gate --release -- --ignored --nocapture

# Install it (19.4 GB in, ~15 GB out, ~21 min, never written to disk whole).
TURBOSPARK_MUSEGLIMMER_INSTALL_DIR=~/models/museglimmer-30b.gturbo \
  cargo test -p turbospark-repack --test museglimmer_checkpoint_network --release -- --ignored --nocapture

# Same oracle for Qwen 3.6. A SEPARATE target, not a second #[test]: the
# footprint assertion is a whole-session peak and the two families have
# different ceilings (~2,200 vs ~1,600 MiB), so they need one process each.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The dense family's oracle (ROADMAP M4). RUNS AT 8,192 CONTEXT where the
# other three run at 4,096, and that is not a knob: the protocol freezes the
# PROSE, and its token count belongs to the checkpoint's tokenizer -- the same
# text is 3,444 tokens under Mistral's 32k vocab against qwen3moe's 2,842, and
# 3444 + 1024 does not fit 4,096, so `long-synthesis` does not run at all.
# Raising the shared PROTOCOL_MAX_CONTEXT would resize KV for every family and
# move every frozen peak, so the window is a per-target parameter and is
# printed on every run beside the ceiling. Its 1,300 MiB ceiling is NOT
# comparable to the MoE rows: 1,024 of the measured 1,201 MiB is KV, and the
# 4.07 GiB of weights count for nothing (Gotcha 40). ~8 min.
TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
  cargo test -p turbospark-bench --test mistral_memory_oracle --release -- --ignored --nocapture

# Same oracle for Qwen3-30B-A3B (`qwen3moe`). Its ceiling is 2,900 MiB, ABOVE
# the other two families' 1,600-2,300: the slot cache is
# `slots x layers x expert_stride` and this model is 48 layers deep at a
# ~2.9 MiB expert, i.e. 2,094 MiB of slot capacity at 16 slots plus a 916 MiB
# pinned resident core. It streams (the whole expert table is 16.36 GiB); it
# just does not land inside the band the README quotes.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_memory_oracle --release -- --ignored --nocapture

# The SIXTH family's two gates (ROADMAP M5). BOTH differ from their four
# siblings in a way that has to be read with the numbers. The oracle runs at
# 8,192 context AND a 3,072-token budget, because Harmony puts the model's
# reasoning in an `analysis` channel BEFORE its answer -- the three cases need
# 818 / 2,153 / 1,108 tokens to reach `<|return|>`, so at the shared 1,024 two
# of three stop on maxTokens and the validity gate refuses them. Its ceiling
# is 5,700 MiB, far above the README's 1.6-2.2 GiB band, and that is
# `slots x layers x expert_stride` = 4,854 MiB of slot cache, not a
# regression (Gotcha 36). The quality gate PINS A DATE, because Harmony's
# template writes `Current date: ` into its system preamble and a digest over
# a clock-reading prompt expires at midnight.
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-bench --test gptoss_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-bench --test gptoss_quality_gate --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q): teacher-forced perplexity of a fixed
# reference answer in the ASSISTANT slot (an instruction-tuned checkpoint
# is never trained to predict prompt tokens, so scoring those measures
# nothing), plus frozen greedy and sampled output digests, plus a
# constrained-working-set repeat at 8 expert-cache slots (digest frozen,
# throughput floored). Split per family for the same one-model-per-process
# reason as the oracle. About 80 seconds each. Numbers and caveats:
# docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test qwen36_quality_gate --release -- --ignored --nocapture
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-bench --test qwen3moe_quality_gate --release -- --ignored --nocapture

# Cross-engine check (ROADMAP Phase Q, last item): does this port agree
# with mlx-lm on the SAME quantized bytes? Two steps. The first dumps this
# port's full-vocab logits for the quality corpus plus the exact token ids
# it walked (~275 MiB, ~30 s). The second replays those IDS -- never the
# prose, or a tokenizer difference would read as a numerics gap -- through
# mlx-lm and prints the KL. mlx-lm runs in a uv ephemeral env, so nothing
# is installed globally and nothing is added to this workspace. Needs
# `hf download mlx-community/gemma-4-26b-a4b-it-4bit --revision
# 0d77464eeb233a2da68ebf9d7dc4edaac7db956d` first (14.6 GB). Read the
# floor, not just the number: docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/turbospark \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/turbospark

# Same dump with NO warmup walk, which is the condition quality_gate takes
# its perplexity under. Reproduces the frozen row exactly; that is the
# cross-check that the dump measures what the gate measures.
TURBOSPARK_LOGIT_DUMP_COLD=1 TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/cold \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture

# The same question one layer down, for the GGUF path: does this port agree
# with llama.cpp on the SAME GGUF bytes? Closes Phase G's last gate clause
# and is the same-precision reference Phase S needs. Dump from a
# GGUF-derived install, then replay the ids through llama.cpp (brew's
# `llama.cpp`; a small harness against its own header, since no shipped
# binary teacher-forces an id list). Needs the 26.9 GB GGUF locally --
# llama.cpp cannot stream it the way the repack walk does. Metal by
# default, and that is not cosmetic: see Gotcha 34 before running it on CPU.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/gguf-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download ggml-org/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-Q8_0.gguf \
  --local-dir ~/models/gguf-ref
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-Q8_0.gguf /tmp/kld/gguf-warm /tmp/kld/turbospark

# Proof that the gate above can SEE quantization damage, rather than just
# asserting it could. Clones the install (APFS clonefile, so the original
# is untouched and only written pages cost disk), shifts one quantization
# level in a strided subset of the routed experts, and re-measures. About
# 30 seconds. Curve and floor: docs/BENCHMARKS.md.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test quality_sensitivity --release -- --ignored --nocapture

# Power baseline over the frozen protocol (ROADMAP Phase P1): watts and
# joules-per-token, split prefill/decode. NEEDS SUDO (powermetrics is
# root-only) and so cannot be run non-interactively. ~12 min per install.
# Windows the capture with the `[power-window ...]` markers turbospark-bench
# emits, so the model open and the discarded warmup stay out of the total.
# Numbers and caveats: docs/BENCHMARKS.md.
LABEL=battery OUT=/tmp/power-gemma MODEL=~/models/gemma4.gturbo scripts/power.sh 2

# Same harness driving an interleaved A/B. Arms alternate WITHIN each
# pair, not as two consecutive batches, because consecutive batches carry
# thermal drift. `ARMS` names the axis and understands THREE kinds of
# token: `default`/`utility` set MFERENCE_READ_QOS (Phase P1, measured and
# rejected -- it loses on both joules and tok/s, kept as a documented dead
# end), `performance`/`balanced`/`efficiency` pass --power-profile
# (Phase P2), and a BARE NUMBER passes --max-tokens-per-sec. The QoS axis
# may not be mixed with the other two and the script refuses it (the arm is
# one column of rows.tsv, so a run varying two things cannot say which).
# An unknown arm is refused UP FRONT, before sudo and before the fans are
# pinned: `ARMS=bogus scripts/power.sh` is a one-second test that needs
# nothing built. `QOS` is still read as the old spelling of `ARMS`.
LABEL=battery MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  ARMS=default,utility scripts/power.sh 3

# The Phase P2 gate: does the efficiency profile actually buy joules per
# token, and does performance stay where docs/POWER_BASELINE.md left it?
# Budget for it: an efficiency arm decodes at ~10 tok/s, so its window is
# several times longer than a performance arm's.
LABEL=ac MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  ARMS=performance,efficiency scripts/power.sh 3

# The rate-cap SWEEP, which is what the numeric arms are for. That gate
# above measured the shipped `efficiency` cap of 10 tok/s at 14.9% of the
# energy for 51.5% of the throughput -- a poor trade, and unplaceable with
# only two points on the curve. `default` is the uncapped reference arm.
# Run it under COOLING=max: a governed arm's J/token is a thermal control
# loop's output rather than the cap's (Gotcha 28). gemma4 rather than a
# bigger install because the constants in `crates/runtime/src/power.rs` are
# GLOBAL and this one decodes ~44 tok/s, so the arms span a 4.4x range
# instead of museGlimmer's 1.9x. ~20 min; note each run pays the cap twice,
# since the discarded warmup is paced with the same RateControl.
LABEL=ac MODEL=~/models/gemma4.gturbo CASES=short-explanation COOLING=max \
  ARMS=default,30,20,15,10 OUT=/tmp/power-cap-sweep scripts/power.sh 3

# GGUF intake (ROADMAP Phase G). Reads only the HEADER of the real
# published GGUFs -- a few MB off a 20-27 GB file, ~4 s each -- and checks it
# three ways: the parser agrees with what llama.cpp's converter writes, every
# tensor name maps, and the ArchConfig derived from GGUF metadata equals the
# one the corresponding .gturbo install declares. Set the install vars to get
# the last two cross-checks; without them it still parses and reports.
# The three `scopes_phase_s_*` cases in the same file are the ROADMAP Phase S
# ingest survey: same cost, and their output is a ggml TYPE HISTOGRAM in
# bytes. Read that histogram's UNSIZED rows before its percentages -- a type
# with no `ggml_type_block` row is one this port cannot ingest, which on a
# mixed file is usually the routed experts, and printing it as 0 bytes once
# made an imatrix file look like it was 76% Q8_0.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# ROADMAP Phase M1: the admission gate for `arch_registry.rs`'s planned
# architecture rows. Re-reads the header of the real published file each row
# was taken from and asserts the `general.architecture` string still matches,
# so the table cannot rot into folklore. A header per row, seconds each, no
# install and no checkpoint. Run it after adding a row.
cargo test -p turbospark-repack --test arch_registry_network --release -- --ignored --nocapture

# Settles which half of Gemma's fused ffn_gate_up_exps is the gate (Stage 2's
# first use of the Q8_0 reference). Correlates a dequantized layer 0 expert 0
# against the same expert in the MLX install. Also a real-data check on the
# Q8_0 dequant itself: a sign, scale or block-layout error cannot correlate
# at +0.9957 with an independently-produced INT4 install. Few KB, ~5 s.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# The evidence behind the transcode decision (Gotcha 29): GGUF's F32 norms
# are upcast BF16 and narrow back bit-exactly, and INT8-transcoding its F32
# router does not move the routing decision. No install needed, few KB, ~6 s.
cargo test -p turbospark-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The same real-data check for Q4_K, against the real Qwen 3.6 Q4_K_M: a
# dequantized layer 0 expert 0 gate row correlates +0.9930 with the same row
# in the MLX install, while the unrelated `up` matrix reads +0.12. It is the
# one check that can catch a decoder and its fixture quantizer being wrong
# TOGETHER, which nothing in crates/compute can. Few KB, ~5 s. Read Gotcha 30
# before believing a low number out of it.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# ROADMAP Phase G Stage 2 item 8: install the REAL published Gemma 4 Q8_0
# GGUF. Streams the 26.9 GB file from HF a layer at a time (it is never
# written to disk) and produces a ~25 GB install. ~24 min. Must NOT point at
# the MLX-derived ~/models/gemma4.gturbo; the test asserts it does not.
TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_install_network --release -- --ignored --nocapture

# ROADMAP Phase M2: install the real published Mixtral 8x7B Q4_K_M. Streams
# the 26 GB file from HF a layer at a time (never written to disk) into a
# ~29 GB install. The THIRD family, and the first whose architecture string
# (`llama`) covers two different models: a dense Llama 3.1 reports the same
# string and is refused at open, because only `expert_count` says which half
# a file is. Also the first real file to put Q6_K in a ROUTED expert (16 of
# 32 layers, against Q4_K on the other 16) and Q5_K on `attn_output`.
TURBOSPARK_MIXTRAL_INSTALL_DIR=~/models/mixtral-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture

# The SIXTH family (ROADMAP M5), and the one whose expert block type has NO
# resident GEMV: MXFP4 lives in `ffn_*_exps` and nowhere else, with attention,
# `token_embd` and `output` all Q8_0. Streams the 12.1 GB file a layer at a
# time (never written to disk) into a ~12 GB install, ~25 min. It is also the
# first file with a BIAS beside every projection, a sink vector per block, and
# a RANK-2 per-expert bias inside the routed blob -- all three exercised
# against `SyntheticGptOssShape` first, which is what kept the two walk holes
# it exposed to milliseconds each instead of a re-stream apiece (Gotcha 42).
TURBOSPARK_GPTOSS_INSTALL_DIR=~/models/gptoss-20b.gturbo \
  cargo test -p turbospark-repack --test gguf_gptoss_install_network --release -- --ignored --nocapture

# The FOURTH family, and the one Mixtral's granularity finding asked for
# (Gotcha 36): install the real published Qwen3-30B-A3B Q4_K_M. Streams the
# 17.3 GB file a layer at a time (never written to disk) into a ~18 GB
# install. Same LAYER GRAPH as Mixtral and the same decode flow
# (`crates/runtime/src/families/llama/`), differing only in per-head q/k
# norms and an RMS epsilon of 1e-6 -- but 128 experts of 2.5 MiB against
# Mixtral's 8 of 108.9, so its slot cache is 1.90 GiB at 16 slots rather
# than 54.5 and the memory oracle and quality gate are worth running on it.
# NO new kernels: Q4_K and Q6_K are both already executable, asserted off
# the header before the download.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen3moe_install_network --release -- --ignored --nocapture

# ROADMAP M4: the FIFTH family install and the first DENSE one. Streams the
# real Mistral-7B-Instruct-v0.3 Q4_K_M (4.1 GB, never written to disk) into a
# ~4.1 GB install with ZERO packed-expert files -- a dense model has no routed
# experts, so nothing streams at decode time. ~5 min. Its `[INST]` chat
# dialect landed back in M2, so `--messages-file` renders correctly for it.
# Gotcha 36's slot-cache multiplication does not apply; nor does the
# resident-floor-is-the-working-set reading of Gotcha 19 (see Gotcha 40:
# measured peak is 684 MiB against 4.07 GiB of weights).
TURBOSPARK_MISTRAL_INSTALL_DIR=~/models/mistral7b-dense.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture repacks_the_real_mistral

# The same family at a twentieth the size: TinyLlama-1.1B-Chat Q6_K, 0.84 GB
# in and ~0.9 GB out, ~3 min. The cheapest real dense checkpoint on the shelf
# and the one that caught Gotcha 39 (its head_dim is 64, where every other
# real `llama` file agrees with the Mixtral baseline's 128). Without the env
# var it installs to a temp dir and deletes it, which is the walk-only case.
# Its chat framing is Zephyr, not `[INST]`, and `--messages-file` now renders
# that correctly: the checkpoint's own template wins over the dialect (Gotcha
# 41). This used to be M4's one open item and needed a raw `--prompt`.
TURBOSPARK_DENSE_LLAMA_INSTALL_DIR=~/models/tinyllama-dense.gturbo \
  cargo test -p turbospark-repack --test gguf_mixtral_install_network --release -- --ignored --nocapture repacks_a_real_dense_llama_gguf

# ROADMAP Phase G Stage 2 item 9: the same, for the K-quants. Streams the real
# published Qwen 3.6 Q4_K_M (~20 GB, never written to disk) into a ~20 GB
# install. This is the MIXED case: Q4_K experts and embedding, Q8_0 attention,
# one Q6_K tensor. Must NOT point at ~/models/qwen36.gturbo; asserted.
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_install_network --release -- --ignored --nocapture

# ROADMAP Phase G Stage 2 item 10: Qwen's V-head convention (Gotcha 33), on
# EVERY layer rather than the layer 0 the transforms were characterized on.
# Reports per tensor how well the bytes on disk and the de-interleaved
# candidate agree with the MLX install. Read-only; TURBOSPARK_QWEN_PATCH=1
# rewrites the install in place, which is how a coherence test costs seconds
# instead of a 23-minute repack. Idempotent: it patches only what improves.
# ~40 s, no network, needs both Qwen installs.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_convention_patch --release -- --ignored --nocapture

# The diagnostic that found the five QUANTIZED tensors on that axis, which a
# BF16 probe structurally cannot see. Recovers the permutation outright where
# a tensor has one row per head. ~1 s, needs both Qwen installs.
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN36_GGUF_INSTALL_DIR=~/models/qwen36-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_qwen_quant_probe --release -- --ignored --nocapture

# The GGUF install's resident BF16 core must be BIT-IDENTICAL to the MLX
# install's: norms, router.scale, per_expert_scale, layer_scalar. Settles
# the Gemma norm "+1" convention and the shape-matched name mappings.
# ~1 s, no network, needs both installs.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_GEMMA4_GGUF_INSTALL_DIR=~/models/gemma4-gguf.gturbo \
  cargo test -p turbospark-repack --test gguf_norm_convention_probe --release -- --ignored --nocapture

# ROADMAP Phase S: install the real sub-4-bit candidate. Streams the 12 GB
# `unsloth/gemma-4-26B-A4B-it-UD-Q3_K_M` from HF a layer at a time (never
# written to disk) into a ~12 GB install whose experts are 9.6 GiB against the
# MLX install's 12. ~18 min. MIXED ALONG TWO AXES, which is what it is for:
# IQ3_XXS gate/up over an IQ4_NL down in one expert, and a layer 29 that is
# IQ4_XS over Q8_0. Asserts the per-layer stride saved over 30% on disk.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-repack --test gguf_iq_install_network --release -- --ignored --nocapture

# Its quality gate. A SEPARATE target with its own chip row, not the Gemma
# gate with an env var moved: the existing rows are keyed on the chip and
# freeze the MLX INT4 goldens, so pointing an install var at a different
# artifact asserts the wrong digests. ~2 min.
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-bench --test iq3_quality_gate --release -- --ignored --nocapture

# The check that says the IQ kernels are faithful rather than merely
# plausible: this port against llama.cpp on the IDENTICAL bytes. Needs the
# 12 GB file locally (llama.cpp cannot stream it) and MUST run on Metal --
# Gotcha 34. Reads 0.00440 mean nats at 97.5% top-1, against a 0.03741
# backend floor.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/iq3-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/gemma-4-26B-A4B-it-UD-Q3_K_M.gguf /tmp/kld/iq3-warm

# The same check for the `qwen3moe` family (ROADMAP M3), and the ONLY
# instrument that can see the two things its fixture tests record themselves
# as blind to: the per-head q/k norms' ORDER relative to RoPE, and the RMS
# epsilon. Reads 0.00320 mean nats at 97.9% top-1 between a 0.00135 shape
# floor and a 0.00988 backend floor, so both are right. Metal, not CPU
# (Gotcha 34). Needs the 17.3 GB GGUF locally; the install was STREAMED from
# it, so it is not on disk by default. ~30 s per arm plus ~12 min to fetch.
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/qwen3moe-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
hf download Qwen/Qwen3-30B-A3B-GGUF Qwen3-30B-A3B-Q4_K_M.gguf --local-dir ~/models/gguf-ref
uv run --python 3.12 --with numpy scripts/kld_llamacpp.py \
  ~/models/gguf-ref/Qwen3-30B-A3B-Q4_K_M.gguf /tmp/kld/qwen3moe-warm

# ROADMAP M4 Phase 0: which dense `llama` checkpoint the dense half should
# target. Three real headers, no download, ~36 s. The two gates it reads are
# whether the file carries `rope_freqs.weight` (Llama 3.1's RoPE scaling,
# which ships as a TENSOR and has no kernel input here) and whether every
# block type in it already has a kernel. Also prints each candidate's
# RESIDENT FLOOR, which for a dense model is the whole weight file: nothing
# streams, so Gotcha 36's slot-cache multiplication does not apply and the
# 1.6-2.2 GiB band cannot.
cargo test -p turbospark-repack --test gguf_checkpoint_network --release -- \
  --ignored --nocapture scopes_the_dense_llama_candidates

# The cheapest gate in the repo and the one to run BEFORE the quality gates
# whenever prompt rendering moves. Its second case is the same guard for
# `--reasoning` (Gotcha 55): that `off` renders the frozen bytes on every
# install, and that a level either MOVES the render or is refused -- there is
# no third outcome that is not a silent no-op. It asserts that each family's own chat
# template differs from the per-dialect renderer by nothing but `trim`, and
# pins per family WHETHER that template trims -- which is the axis that
# decides whether the frozen digests move, and the one that already moved
# `qwen3moe`'s. It compares the real protocol prompt, trailing newline and
# all, because on a tidy one-line string `trim` is a no-op and the guard
# sees nothing. Seconds, no GPU, no model load beyond the tokenizer.
# See Gotcha 41.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
TURBOSPARK_QWEN3MOE_INSTALL_DIR=~/models/qwen3moe-gguf.gturbo \
TURBOSPARK_GEMMA4_IQ_INSTALL_DIR=~/models/gemma4-iq3.gturbo \
  cargo test -p turbospark-tokenizer --test installed_template -- --ignored --nocapture

# The determinism check nothing else makes: one runner, the same greedy
# generation six times, asserting ONE distinct output. This is what caught
# Gotcha 27's cache-state-dependent reduce order; run it at 8/16/32 slots
# after any change to the routed-expert path. ~15 s per slot count.
TURBOSPARK_PROBE_SLOTS=32 TURBOSPARK_PROBE_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test gguf_nondeterminism_probe --release -- --ignored --nocapture

# Phase D1's gate: does rollback return the engine to the state a fresh run
# would have reached? Asserts BIT-IDENTICAL logits against a reset-and-replay
# ground truth, because a stale recurrent state or a clobbered KV row produces
# fluent wrong text that no coherence smoke can see. Run it on BOTH families:
# Qwen exercises the recurrent half (30 linear layers of 40, unbounded
# rollback), Gemma the sliding-window ring (25 of 30, rollback capped at the
# ring's 128-token slack). ~4 s each.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test rollback_probe --release -- --ignored --nocapture

# Phase D2's last unknown: how many proposed tokens does the target accept?
# Needs NO batched kernels -- only the ratio matters, so the verify pass runs
# sequentially. The drafter is n-gram/prompt-lookup (zero weights), which is a
# LOWER bound on a trained one. Also the end-to-end losslessness check: every
# block size must produce a token stream byte-identical to speculation off.
# ~40 s.
TURBOSPARK_PROBE_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-bench --test accept_length_probe --release -- --ignored --nocapture

# The two measurement surfaces behind ROADMAP's speculative-decoding item
# (its Phase D0 gate: does a batched verify pay on this engine, and at what
# block size?). Neither needs a drafter or a new kernel.
#
# 1. Expert union. `MFERENCE_ROUTER_TRACE=1` adds the top-k ids IN PASS
#    ORDER to the histogram file, which the counts throw away; a batched
#    verify of M tokens reads their UNION, so its expert cost is the
#    distinct-expert count over a window of M consecutive passes. Pass the
#    run's PROMPT TOKEN COUNT as the second argument -- the trace covers
#    prefill too, and prefill routes differently. Read `breakeven_accept`:
#    a block pays, on this axis, only if it accepts more than that.
MFERENCE_ROUTER_HIST=/tmp/rq.json MFERENCE_ROUTER_TRACE=1 \
  ./target/release/turbospark-check --model ~/models/qwen36.gturbo \
  --messages-file /tmp/p.json --max-new 300 --temperature 0.0001 --top-k 1
python3 scripts/router_window.py /tmp/rq.json 22 2,4,8,16

# 2. Compute headroom. Is `dequant_int4_gemv_simd` already bandwidth-bound
#    at the shapes decode dispatches? If it is, batching cannot help. It is
#    at the big projections and is NOT at the routed-expert shape. Reports
#    ratios against the same kernel at a large row count, never against a
#    spec number; read its module doc before quoting an absolute GiB/s.
cargo test -p turbospark-gpu --test gemv_bandwidth_bench --release -- --ignored --nocapture

# The SECOND checkpoint of the `qwen3_5` family (2026-08-14), and the one
# that makes the pair a CONTROLLED comparison: `Qwen/Qwen3.8-27B` shares
# Bonsai-27B's architecture EXACTLY (33 of 35 text_config keys equal; the
# two that differ reach no ArchConfig field), so the only thing that varies
# between the two installs is the quantization -- 1-bit group 128 against
# INT4 group 64. Streams the 16.08 GB mlx-community artifact a tensor at a
# time (never written to disk) into a ~16 GB install. Nothing in `src/`
# changed for it; the offline `qwen35_config` target is what says so, and it
# is the gate to run BEFORE this one.
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-repack --test qwen38_checkpoint_network --release -- --ignored --nocapture

# Its two gates, the FIRST this family has ever had (Bonsai got neither, so
# the DENSE half of `families/qwen/` had no sentinel at all until now).
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-bench --test qwen38_memory_oracle --release -- --ignored --nocapture
TURBOSPARK_QWEN38_INSTALL_DIR=~/models/qwen38-27b.gturbo \
  cargo test -p turbospark-bench --test qwen38_quality_gate --release -- --ignored --nocapture

# The THIRD checkpoint of the `qwen3_5` family (ROADMAP's ternary entry) and
# the one that made it a WIDTH rather than a family: MLX affine at TWO bits,
# group 128, FP16 companions. Its `text_config` is Bonsai-27B's to the key --
# the two files differ in their `quantization` object ALONE -- so nothing in
# `crates/model-io` or `crates/runtime`'s flow moved for it. Streams the
# 8.49 GB artifact a tensor at a time (never written to disk) into a ~7.6 GB
# install, ~14 min. The offline `qwen35_config` target asserts the parse and
# is the gate to run BEFORE this one.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-repack --test ternary_checkpoint_network --release -- --ignored --nocapture

# Its two gates. NOTE the perplexity is NOT comparable to `qwen38_quality_gate`'s
# as a quantization result even though the two share an architecture: one is
# prism-ml's QAT checkpoint and the other Qwen's release, so a TRAINING
# separates them as well as a width.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-bench --test ternary_quality_gate --release -- --ignored --nocapture
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
  cargo test -p turbospark-bench --test ternary_memory_oracle --release -- --ignored --nocapture

# Its cross-engine KL, through the SAME driver the 1-bit one uses. The
# checkpoint name is REQUIRED and not defaulted: the two published
# checkpoints have the same module count and the same shapes, so pairing a
# dump with the wrong reference passes every check inside the script and
# reads as a kernel bug. Upstream mlx supports bits=2, so unlike the 1-bit
# arm this one needs no fork.
TURBOSPARK_TERNARY_INSTALL_DIR=~/models/ternary27b.gturbo \
TURBOSPARK_LOGIT_DUMP_DIR=/tmp/kld/ternary-warm \
  cargo test -p turbospark-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with 'mlx-lm==0.31.2' --with numpy \
  scripts/kld_mlx_affine.py /tmp/kld/ternary-warm ternary-2bit

# The other #[ignore]d tests: real checkpoint downloads (many GB).
cargo test -p turbospark-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p turbospark-repack --test hf_checkpoint_network --release -- --ignored --nocapture
TURBOSPARK_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p turbospark-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture
TURBOSPARK_QWEN35_INSTALL_DIR=~/models/bonsai27b.gturbo \
  cargo test -p turbospark-repack --test qwen35_checkpoint_network --release -- --ignored --nocapture
```

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

2. Runtime configuration numeric setters abort construction by panicking when a
   value is outside its documented allowed set. This is an intentional fatal
   precondition failure, not a recoverable error: there is no `Result`-returning
   variant and no clamping. Callers that must not abort on bad input should
   validate the value first, or contain the panic with
   `std::panic::catch_unwind`. The allowed sets are exposed as the const arrays
   in `crates/core/src/runtime_config.rs`; read from them rather than
   re-hardcoding the literals. Automatic chunk-size resolution (the
   three-state rule that turns an unknown-or-known input length into one
   concrete allowed chunk size) lives in `crates/core/src/chunk_sizing.rs`
   and reads the same allowed-set constants; it does not redeclare them.

3. The half-precision logit element is backed by the maintained `half` crate
   (version 2, MIT OR Apache-2.0) because the native `f16` type is unstable on
   the stable toolchain. Do not hand-roll IEEE-754 binary16 storage or
   arithmetic. `crates/compute`'s BF16 helpers (`bf16_to_f32`/`f32_to_bf16`)
   are a separate, hand-rolled bit-shift pair (BF16 is exactly the top 16
   bits of an FP32 word), not routed through `half`.

4. Token ids cross crate boundaries as signed 32-bit integers
   (`pub type TokenId = i32`). Keep that interchange width when wiring
   downstream crates.

5. `Cargo.lock` is committed on purpose. This workspace targets command-line
   and server binaries, so the lockfile stays in version control for
   reproducible builds. Do not delete or gitignore it.

6. The cargo build output lives in `/target` and is gitignored. It is large;
   never commit it.

7. **Process-entry-point decision (resolved):** `crates/cli` (binary name
   `turbospark-check`) is the process entry point that reads `argv`, calls
   `turbospark-invocation::parse`, and applies its pure exit-status/stream-routing
   decisions. This resolves what an earlier note here called the reserved,
   not-yet-created `turbospark-entrypoint` name; that name is not used. On macOS
   it also attempts real generation against `--model` via `RealForwardRunner`,
   in all three modes (`--prompt`, `--messages-file`, `--chat`) (see
   `DEVIATIONS.md`).

8. `crates/gpu` is the one crate with a hard platform gate: everything in
   `src/` is meant to be `#[cfg(target_os = "macos")]`, so `cargo build
   --workspace` / `cargo test --workspace` succeed on Linux with the crate
   compiling to (effectively) nothing. THAT SENTENCE WAS FALSE THE FIRST TIME
   ANYONE CHECKED IT (2026-08-15), and twice over: `crates/streaming` declared
   `libc` under `cfg(target_os = "macos")` while calling
   posix_memalign/free/sysconf unconditionally (4x E0433), and `lib.rs:49`'s
   `mod dequant_int4_gemv;` lost the gate its ~30 siblings carry (14 errors,
   E0432 on `metal`/`half` and E0433 on `crate::bytes`/`crate::context`).
   Neither is reachable from any command in the verification policy, which is
   why a cross-target `cargo check` now sits beside them. A platform claim
   nobody runs is a comment, not a gate.
   **AND THE DEPENDENCY TABLE IS WHERE PORTABILITY IS REALLY DECIDED**, not the
   `cfg`s in `src/`: `crates/runtime` declares `model_io`, `gpu`, `compute` and
   `streaming` under `[target.'cfg(target_os = "macos")'.dependencies]`, so it
   is macOS-only however portable its source reads, and any crate taking a
   dependency on it inherits that. Read the target's `[target.'cfg(...)']`
   block before adding a cross-crate edge -- this is what forced the two
   sizing policies down into `model-io` rather than letting `catalog` reach up.
   Dispatched, parity-tested Metal pipelines
   (`rmsnorm_no_scale`, `rms_norm_bf16w`, the port-local
   `rmsnorm_bf16w_centered` (the `x * (1 + w)` form `muse_glimmer`'s four
   per-layer norms take; a SEPARATE kernel rather than a function constant
   on its plain sibling, because that model uses BOTH conventions and a
   specialization axis missing the pipeline cache's key would silently
   collapse them -- Gotcha 50), its per-head sibling
   `rmsnorm_bf16w_perhead_centered` (the `qwen3_5` MTP head's
   `q_norm`/`k_norm`, where the TRUNK's tensors of the same names through the
   same call are plain), all three `_perhead` norm variants,
   `rope_proportional_neox` (which with `rotated_pairs = head_dim/2` IS
   default full-head NeoX -- no separate default-rope wrapper exists),
   the port-local `logit_softcap_fp16`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`
   (both with offset-bound resident variants), the port-local
   `dequant_q8_0_gemv_simd`, `dequant_q4_k_gemv_simd` and
   `dequant_q6_k_gemv_simd` (GGUF Q8_0, Q4_K and Q6_K, each also with a
   resident variant), `embed_lookup_q8_0` and `embed_lookup_q4_k`, the
   port-local `dequant_int1_gemv_simd` and
   `dequant_int1_gemv_symmetric_simd` (MLX `affine` at ONE bit; the second
   is a separate kernel and not a fast path inside the first, because
   factoring the group scale out reassociates the sum -- Gotcha 27's
   territory -- and because it binds no bias plane, so a caller that has
   not measured the symmetry cannot reach it) plus `embed_lookup_int1` and `embed_lookup_int2`
   (both checkpoints quantize their embedding TABLE at their own width too, so this
   type's footing is a GEMV and a lookup and NO routed pair -- read off
   the safetensors header, and the model is dense), and
   `moe_gguf.metal`'s three routed-expert decode pairs
   (`moe_phase1_gate_up_act_{q8_0,q4_k,mxfp4}` +
   `moe_phase2_down_reduce_k8_{q8_0,q4_k,mxfp4}`), all port-local because
   Swift has no GGUF intake. Q6_K has a GEMV and no siblings on purpose:
   the only real file using it puts it in `output.weight`. MXFP4 is the
   MIRROR of that (ROADMAP M5): both routed phases and NO resident GEMV,
   because `gpt-oss` puts it in `ffn_*_exps` and nowhere else. Its pair
   also carries gpt-oss's expert MATH, not just its block layout -- the
   clamped SwiGLU (`min(gate, limit)`, a swish with `alpha`, times
   `(clamp(up) + 1)`) and the per-expert biases -- selected by
   `Mxfp4Activation` as a UNIFORM rather than a function constant, since
   `moe_gguf.rs`'s `constants_key` is one byte wide and a second
   specialization axis that misses the key silently reuses the wrong
   pipeline (Gotcha 18's trap, one file over). Then
   `router_gemv_gemma4_r4`,
   two-pass split-KV `attention_decode` (multi-chunk, split up to 16 ways by
   `chunks_for`, and since ROADMAP M5 optionally folding gpt-oss's
   ATTENTION SINKS into the combine pass -- one learned logit per q head
   added to the softmax DENOMINATOR and to nothing else, behind
   `FC_ATTN_HAS_SINKS`, whose byte is in `attention_constants_key`),
   `moe_decode` decode pair, all eight of `gdn.metal`'s
   gated-DeltaNet kernels, the port-local `rope_neox_freqs` (M5's YaRN: a
   PRECOMPUTED per-pair frequency table plus a magnitude scale, because
   the three older rope kernels derive every frequency from a scalar theta
   and YaRN is not expressible that way), and
   `utility` elementwise kernels including the port-local `scalar_mul_fp16`,
   the port-local `bias_add_bf16_fp16` (M5: a separate pass rather than a
   bias argument on seven GEMV kernels and four families' dispatch sites)
   and Qwen's three gating kernels)
   are compiled from vendored MSL source at
   runtime, matching how Mference itself builds pipelines. `MetalContext::pipeline`
   takes caller-supplied `FunctionConstantValues`, so a new kernel module owns
   its own specialization rather than sharing one hardcoded set. The
   library/function/pipeline caches key on the shader source's ADDRESS, not
   its text: a caller MUST pass the same `&'static str` (an `include_str!`
   constant) for a given file every call, or it recompiles instead of
   hitting the cache. Keying on the text meant rehashing tens of kilobytes
   of MSL per dispatch, ~900 dispatches per decoded token, which was most
   of this port's CPU encode cost.
   `KvCacheManager` (`kv_cache.rs`) is `RealForwardRunner`'s production KV cache
   (persistent per-layer buffers, K written in place by the GEMV).
   `GdnStateManager` (`gdn_state.rs`) is the Qwen 3.6 flow's recurrent
   state (delta-rule `S` plus conv tail, both advanced in place per token);
   `Dsv4StateManager` (`dsv4_state.rs`)
   allocates real per-layer Metal buffers but stays unwired (its compute
   kernels are unported); `PrefillChunkScratchLayout`/`PrefillChunkScratchBuffers`
   (`prefill_scratch.rs`) size and allocate the chunked-prefill scratch
   buffers, also undispatched (the tile kernel is descoped). The memory
   path is zero-copy end to end: `ResidentGpuWeights` wraps the resident
   mmap in ONE `newBufferWithBytesNoCopy` MTLBuffer, `PassEncoder` batches
   a whole token into one command buffer, and the vendored `moe.metal` decode
   kernels read streamed expert blobs in place through a `RoutedBlobs` argument
   buffer. `moe_phase2_down_reduce_k8` reduces ALL EIGHT slots
   unconditionally: unused slots need a zero routing weight, a valid blob
   pointer (bind() duplicates blobs[0]), AND a finite acts row -- the acts
   buffer must be sized for 8 slots and zero-filled, not sized top_k
   (garbage in a padded row times a zero weight is still NaN). See `DEVIATIONS.md` for the full wired/unwired list, and why `sample`
   (`logit.metal`) was deliberately left unported rather than shipped without a
   way to verify it.

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
   `compute`, `repack`, `runtime`, and `tokenizer`
   have `#![forbid(unsafe_code)]`. `core`, `gpu`, `invocation`, `selection`,
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

12. `crates/runtime::RealForwardRunner` (macOS/GPU only, gated the same way
    `crates/gpu` is) supports dense and MoE FFN (resident or streamed
    experts), full-attention (mask 1) and sliding-window (mask 0) layers
    everywhere, and linear (2) layers under the `qwen36` family; `open()`
    rejects compressed (3/4) layers, whose kernels are unported. It has
    THREE decode flows. The FAMILY picks first (`ArchConfig.family`, NOT
    tensor naming -- Gemma 4 and Qwen 3.6 both carry
    `language_model.model.embed_tokens.weight`, so a naming probe cannot
    tell them apart): `QwenGdnMoe` builds `RealQwenState` and runs
    `families/qwen/` (gated DeltaNet
    on mask-2 layers, gated full attention on mask-1, one post-attention
    norm feeding router + shared + routed, no sandwich norms, no softcap);
    `QwenGdnDense` builds the SAME state and runs the SAME flow, its DENSE half
    (ROADMAP's 1-bit entry) -- every behavioural field of `qwen_gdn_dense_27b()`
    equals `qwen_gdn_moe_35b_a3b()`'s and every shape field differs, so the FFN
    is the only thing that forks, on `num_experts == 0` and never on
    tensor naming. THREE published checkpoints run that flow, at three
    quantizations of ONE architecture: `prism-ml/Bonsai-27B-mlx-1bit` (1-bit),
    `prism-ml/Ternary-Bonsai-27B-mlx-2bit` (2-bit) and `Qwen/Qwen3.8-27B`
    (INT4) (see Gotcha 47);
    `DeepseekV4Flash` is refused. `GptOss` builds `RealGptOssState` and
    runs `families/gptoss/` (ROADMAP M5) -- plain GQA with a BIAS on all
    four projections, YaRN rope through a precomputed frequency table,
    ATTENTION SINKS, an alternating 128-token window on the EVEN layers,
    and MXFP4 routed experts carrying a clamped SwiGLU and per-expert
    biases. A SEVENTH FAMILY AND A SIXTH FLOW landed after it: `MuseGlimmer`
    builds `RealMuseState` and runs `families/museglimmer/` -- dense GQA
    with an alternating three-sliding/one-full window, CENTERED per-layer
    norms (`x * (1 + w)`) against a PLAIN final one, TWO RMS epsilons,
    NoPE on the full layers, a separate attention output gate, a second Q
    scale and a logit softcap behind an output multiplier (Gotcha 50).
    A FIFTH FLOW rather than a sixth family on an existing one,
    because all four of those are inside the layer; every one of them
    produces fluent WRONG output rather than an error if a neighbour's flow
    is used instead. Two of them are not where a reader would look: the
    ROUTER bias is added in `moe.rs`, on the host, between the router
    readback and the top-k (llama.cpp's `SOFTMAX_WEIGHT` selects on the
    biased RAW logits, so after the top-k would pick the wrong experts and
    still read fluently), and the clamped SwiGLU plus the per-expert biases
    are named nowhere in the flow at all -- they ride inside the MXFP4 pair
    and arrive through `RoutedBlobLayout`.
    Within `Gemma4`, naming picks: synthetic
    short names (`layer0.q_proj`) get
    the plain no-scale flow in `real_forward.rs`; verbatim
    real-checkpoint names (`language_model.model.layers.0...`, what
    `turbospark_repack::write_gemma4_install` writes) get the full Gemma 4
    learned-weight flow in `families/gemma4/` (BF16 norms, per-head
    q/k/v norms, INT8 router + effective scale, kernel-semantics top-k,
    INT8 shared-expert branch, sandwich tail, `layer_scalar`). Real
    Gemma 4 26B-A4B is PROVEN end to end: the pinned checkpoint repacks
    through the streamed pipeline and generates coherent chat-formatted
    answers via `turbospark-check` (see DEVIATIONS.md -- raw prompts babble,
    the IT model needs its `<|turn>` markup, which `--messages-file` and
    `--chat` now render for you). `RealForwardRunner::phase_counters`
    accumulates per-phase decode timings (GPU wait, router readback,
    expert `pread`, routed bind) plus expert-cache hit counts and
    per-command-buffer GPU busy attribution (`GPUStartTime`/`GPUEndTime`,
    a separate axis from the wall-clock buckets: cb1 = attention+router,
    routed cb, final head; shared/hit buffers stay unattributed); run
    `turbospark-check` with `MFERENCE_PHASES=1` to print the breakdown.
    One level below that, `MFERENCE_DISPATCH_PROFILE=1` ranks the
    individual dispatches INSIDE each command buffer
    (`crates/gpu/src/dispatch_profile.rs`), which is what a per-buffer
    number cannot tell you (Gotcha 20's whole failure mode). Apple GPUs
    sample counters only at encoder boundaries, so the mode encodes one
    compute encoder per dispatch and waits on every buffer to resolve its
    timestamps: read the module doc before quoting an absolute number
    from it. It is a debugging aid, never on by default, and never a
    throughput measurement.
    The shared-expert branch rides its own command buffer, committed
    before the router wait so it overlaps the host's expert `pread`
    (`PassEncoder::commit` -> `CommittedPass::wait`); `MFERENCE_SHARED_CB=0`
    reverts to encoding it after the pread, which is the A/B seam any
    throughput claim here should be measured against (interleave the two,
    the run-to-run spread is wider than the effect). There used to be a
    sibling seam, `MFERENCE_HIT_CB`, applying the same trick to the
    routed experts by ordering each layer's slots cache MISSES first then
    hits so the resident hits' phase-1 GEMV could ride its own command
    buffer. IT IS GONE, and the reason is Gotcha 27: that order fed phase
    2's reduce, which made generated bytes a function of CACHE STATE. Its
    cost is now MEASURED rather than estimated (2026-08-08, AC, interleaved
    pairs against a binary built at the pre-fix commit): -0.5% at 16 slots
    and -1.9% at 32, scaling with the hit rate as the mechanism predicts.
    An earlier cross-session oracle comparison suggested -7% and was
    session drift. `MFERENCE_ROUTED_PIPELINE=0` is the other seam: by default a
    layer's routed work commits as its own command buffer at the END of
    the layer (Swift's one-layer-pipelined routed CB) and is retired
    after the next layer's router wait, where it has provably completed;
    `=0` reverts to rolling it uncommitted into the next layer's first
    buffer. Worth ~+2.5% (the GPU starts routed work during the host's
    next-layer attention encode); it cannot hide the pread, which
    depends on the router output downstream of the routed tail. Expert
    prefetch/speculation is deliberately NOT wired up: Swift benched
    every shape to a dead end (7% cross-layer predictor hits, prefetch a
    measured no-op, RDADVISE unstable -- see DEVIATIONS.md's MoE entry
    for the pointers), so do not re-derive it.
    The Qwen path has NONE of those three seams (DEVIATIONS.md).
    Slot count comes from `open_with_options`
    (`--expert-cache-slots`, allowed 8/16/24/32, ~3.2 MB of
    pinned host memory per slot per layer on the 26B); it was hardcoded
    to 16 before, so measurements taken with the flag set are only
    meaningful from that change on. **THE DEFAULT IS `auto` SINCE
    2026-08-16, AND THE TWO WAYS IN ARE DELIBERATELY DIFFERENT.**
    `open_with_slot_policy` takes an `ExpertCacheSlots` and resolves
    `Auto` against this machine and this install at open
    (`crates/runtime/src/expert_cache_policy.rs`: the largest allowed
    count whose `slots x sum(expert_stride)` fits a quarter of
    `physical - resident - 4 GiB`); `open_with_options` still takes a
    plain count and is what every harness calls. The floor is
    `DEFAULT_CACHE_SLOTS`, so `Auto` can only ever CLIMB -- a machine
    with no headroom gets exactly the 16 it always got, and no user can
    be made slower by the feature existing. What licenses an
    environment-sensing default at all is that a slot-count change is a
    THROUGHPUT axis only: output is byte-identical across 8/16/24/32
    since Gotcha 27's fix, so no digest and no perplexity can move.
    What licenses the two entry points is Gotcha 35: every measuring
    caller passes `PROTOCOL_EXPERT_CACHE_SLOTS` through the count-taking
    form, so no frozen footprint row can acquire the sensing default.
    Qwen 3.6 runs on a
    SYNTHETIC install (`build_synthetic_qwen_gdn_moe_install`); no real
    checkpoint has been repacked. DeepSeek-V4-Flash remains blocked on
    DSV4. Build a test/demo install with
    `turbospark_repack::build_synthetic_gemma4_install` (dense) or its
    `_swa`/`_moe`/`_moe_streamed` variants,
    `build_synthetic_gemma4_real_install` (real naming, exercises the
    real checkpoint repack pipeline), or
    `build_synthetic_qwen_gdn_moe_install` (the Qwen sibling), instead of
    hand-writing an `ArchConfig`; their non-shape fields are pinned to match
    `gemma4_26b_a4b()`'s own values on purpose (see their module docs for
    why: `manifest.json`'s optional fields fall back to the Gemma 4
    baseline when omitted, so anything else needs those fields written
    explicitly). The weights these produce are deterministic but not
    trained, so generated tokens are structurally real but semantically
    meaningless -- and a short generation can decode to the EMPTY STRING.
    Tests may assert that generation ran (token counts, stop reason, the
    log lines) but never that specific text appeared.
    These fixtures' routers are also near-UNIFORM: the top-k routing
    weights barely differ, so permuting which slot holds which expert is
    numerically a no-op and no test on them can catch a weight-to-slot
    pairing bug. Fuse such pairings structurally instead of testing them.
    To get a real cache hit/miss mix, pass `open_with_options` a slot
    count BELOW `num_experts` (the allowed 8/16/24/32 set is the CLI
    flag's, not this function's).

13. `st` (the ripgrep-alike used here) skips gitignored files, so it finds
    NOTHING in `ROADMAP.md` -- that file is gitignored on purpose (see
    Gotcha 5's sibling note about local-only edits). Read or `grep` it
    directly. **And in a WORKTREE it does not exist at all**, because
    gitignored files are not carried into one: `ROADMAP.md` and
    `CLAUDE.local.md` live only in the main checkout, so edit them at that
    path rather than relative to the worktree you are working in. `st` also
    needs a per-tree index, so a fresh worktree answers every query with
    "no index found" until `st index` has run once.

14. Adding one flag to `crates/invocation` touches five places, two of them
    non-obvious: the `OPTIONS` table, BOTH parser dispatch `match`es (each
    ends in `unreachable!`, so a missing arm is a runtime panic, not a
    compile error), the `InvocationRequest` literal, and
    `tests/usage_and_status.rs`'s hardcoded option-count assertion.

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

19. **`phys_footprint` counts the resident weight mapping.** A read-only
    `mmap` on its own would not (clean file-backed pages are excluded), but
    `newBufferWithBytesNoCopy` makes Metal pin the range. So the honest
    accounting for an install is `resident weights + KV + expert slot
    capacity + process baseline`. Do not "explain" a footprint number by
    assuming the weights are free -- on the real install they are 1,291 of
    its MiB. `crates/bench/tests/memory_oracle.rs` asserts the session
    peak against the published Swift ceiling, and separately that a
    replayed warm case stops growing (Gotcha 17).

20. **The FIRST timed run after a build is a cold GPU, and it is not a
    baseline.** `MFERENCE_PHASES=1`'s `gpu busy` buckets come from
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

21. **Every `MFERENCE_PHASES=1` number is divided by ALL forward passes,
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

22. **Record the power source next to any absolute number, and never
    A/B across sessions.** `pmset -g ps` is the check; `pmset -g therm`
    does not flag battery operation, and `powermode` reads 0 either way.
    NOW MEASURED, 2026-08-07: one binary ran the whole protocol on both
    power sources (`docs/POWER_BASELINE.md`), and the answer has two
    halves. ENERGY is not the axis that moves -- watts and
    joules-per-token differ by a few percent with no consistent sign
    (Gemma `short-explanation` 3.7% worse on AC, Qwen `medium-review`
    5.9% better), which is ordinary cross-session drift. THERMAL HEADROOM
    is the axis that moves, and it is decisive: on AC 50 of 50 arms held
    Nominal pressure, while on battery `long-synthesis` left Nominal on
    every run of BOTH installs and two further runs were lost the same
    way, so the battery run has holes the AC run does not (Gotcha 28).
    Keep recording the source, but expect the difference to show up as
    throttling, not as a power-source correction to the watts. What is
    also established is that cross-session
    absolute numbers here have repeatedly failed to reproduce (see the
    2026-08-05 rows in CLAUDE.local.md) while ratios measured back to
    back within one session have held. Prefer the ratio. That is why
    `crates/gpu/tests/attention_chunk_bench.rs` reports speedups rather
    than absolute microseconds.

23. **Every profiling surface in this repo measures the inside of
    `produce`. The decode loop is bigger than that.** `MFERENCE_PHASES=1`,
    its GPU-busy attribution, and `MFERENCE_DISPATCH_PROFILE=1` all live in
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

25. **`gdn_qk_norm` and `gdn_gated_norm` are correct at EXACTLY 128
    threads per threadgroup.** Both reduce their SIMD partials with a
    hardcoded `for (i = 0; i < 4; ++i)` over `threadgroup float
    partial[32]`: fewer threads sums uninitialized partials, more silently
    drops the extra SIMD groups' work. Neither shape is a compile error or
    a crash, just a wrong norm. `crates/gpu/src/gdn.rs` pins the constant
    (`NORM_THREADS`); do not make it a function of the head dim. The delta
    kernels have a matching fixed shape for a different reason: threads
    `(32, 4)` because each lane owns `Dk/32` state elements in a
    `float s[8]` register tile, which is where `GdnShape::validate`'s
    `Dk % 32 == 0` and `Dk / 32 <= 8` come from.

26. **Qwen's `linear_attn.A_log` and `linear_attn.dt_bias` have NO
    `.weight` suffix**, unlike every other tensor in the checkpoint. They
    are plain BF16 `[num_v_heads]` parameters, not projections. Appending
    `.weight` by analogy gets a `MissingTensor` at open, which is the good
    case; the bad case is a repack-side name filter keyed on `.weight`
    quietly dropping them. Qwen's routed experts also live under
    `.mlp.switch_mlp.`, not Gemma's `.experts.switch_glu.` -- and THAT one
    fails silently in the other direction (an unrecognized routed marker
    makes every expert a resident tensor, which loads and generates fine,
    just with the whole expert table pinned). `routed_marker` in
    `gemma4_checkpoint/` owns the mapping;
    `crates/repack/tests/synthetic_qwen.rs` pins both families' markers.

27. **The routed-slot dispatch order must depend on the ROUTE ALONE, and
    for two months it depended on cache state instead.** Phase 2 reduces
    `blob[slot] * routing_w[slot]` over slots in index order; FP addition
    is not associative, so the slot order IS the summation order and
    whatever the slot order depends on, the output depends on.
    The Gemma flow (`families/gemma4/`) used to order a layer's slots cache
    MISSES first then hits (so the resident hits' phase-1 GEMV could ride
    its own command buffer). The hit/miss split is not a function of the prompt --
    the expert cache carries over between generations -- so THE SAME
    PROMPT COULD DECODE TO DIFFERENT TEXT. Measured 2026-08-08: 4 distinct
    outputs in 6 warm greedy runs of one prompt in one process on a Q8_0
    GGUF install at 16 slots, and 2 in 6 on the MLX install at 32. It hid
    because `quality_gate` only ever ran at 16 and 8, where the MLX
    install happens to converge.
    FIXED by dispatching in the router's own ranking, which costs the ~1%
    the overlap bought (a scattered hit set cannot form one contiguous
    dispatch, so the seam went with the permutation). Both families are
    now byte-identical across 8/16/32 slots and across cold vs warm, and
    `quality_common` asserts that identity rather than freezing a second
    digest per slot count.
    THE REUSABLE PART IS THE TEST THAT WAS MISSING, not the fix: nothing
    in the suite ran the same generation twice on one warm runner, so a
    whole class of "output depends on hidden state" was unobservable.
    `crates/bench/tests/gguf_nondeterminism_probe.rs` is that test now.
    When adding a decode-path optimization, ask what its ordering is a
    function of before asking what it costs.

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
    the per-run rows in `rows.tsv` (this sentence used to claim the
    script excludes it, which its own awk refutes); `scripts/parity.sh`
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

29. **A GGUF is not a safetensors checkpoint with different names, and the
    four differences that bite are all silent.** ROADMAP Phase G Stage 1
    ingests GGUF (`crates/repack/src/gguf_*.rs`); every item below was read
    off the real published files rather than off the format docs, and each
    one produces a plausible, non-crashing wrong answer if assumed.
    - **The data region is ALIGNED, not adjacent.** Tensor offsets are
      relative to the end of the tensor table rounded UP to
      `general.alignment` (default 32). Skip the rounding and every tensor
      in the file shifts by up to 31 bytes.
    - **Dims are stored fastest-varying first.** A logical `[out, in]`
      matrix is stored `[in, out]`, so the resident index's shape is the
      REVERSE of what the file says. Getting this wrong transposes every
      shape while leaving every byte correct, which no byte-level
      assertion catches.
    - **Gemma 4 fuses gate and up into one routed tensor**
      (`ffn_gate_up_exps`, `[in, 2 * ffn, experts]`) where MLX keeps them
      apart; Qwen 3.6 does not fuse. Gate is the FIRST half: measured
      2026-08-07 by correlating a dequantized layer 0 expert 0 against the
      MLX install (+0.9957 matched, -0.0084 crossed), not assumed. The
      standing test is `gguf_fused_gate_network.rs` and it costs a few KB
      of ranged reads. Getting this backwards is the format's nastiest
      trap: two unrelated matrices swap inside every routed expert and the
      model keeps generating fluent, wrong text.
    - **`general.architecture` is the converter's name, not the family's.**
      Qwen 3.6 GGUFs say `qwen35moe`. Deriving it from
      `ModelFamily::as_str()` recognizes no real Qwen GGUF.
    Two more that are merely surprising rather than dangerous: Gemma 4's
    GLOBAL layers carry no `attn_v` at all (true of the MLX install too, so
    a missing per-layer tensor is not an error), and GGUF ships F32 norms
    and an F32 router where an MLX install carries BF16 and INT8, which is
    why a GGUF install needs more than expert kernels to run -- and, on the
    kernel side, more than a GEMV: routed experts and the embedding table
    have their own kernels, so a block type needs three before it runs.
    **That last one is settled as a repack-time TRANSCODE, not an F32
    path** (ROADMAP Phase G Stage 2), and the tradeoff the roadmap wrote it
    up as turned out not to exist. Measured 2026-08-07 off the real file
    (`gguf_f32_transcode_network.rs`): llama.cpp UPCAST norms that were
    BF16 in the original checkpoint, so all 15,592 values across seven norm
    tensors narrow back with zero bit loss, and the transcode is exactly
    lossless. The router is lossy either way, but it is the SAME INT8
    affine this port's MLX path already applies to the same tensor, and
    the routing decision survives it: top-1 unchanged on 32 of 32 random
    activations, and every top-8 membership flip sat 0.148 quantization
    noise-widths from the cut, i.e. a tie the quantizer could not see
    rather than a reordering of a decided pair. LANDED, so nothing F32
    reaches an install any more: `gguf_checkpoint/transcode.rs::transcode_f32`
    narrows to BF16 by default and INT8s the router. The INT8 set is keyed
    by CANONICAL NAME per family, not by rank -- Qwen's
    `linear_attn.conv1d.weight` is rank-2 F32 that the runtime reads as
    BF16, while `mlp.gate.weight` is rank-2 and must be dtype 5 -- and the
    BF16 default is safe because a mis-targeted tensor fails loudly at
    `open()` rather than quietly.
    **WHETHER A GGUF INSTALL LOADS IS DECIDED PER BLOCK TYPE, NOT PER
    FORMAT** (Stage 2, 2026-08-08). Q8_0 and Q4_K run: each has a resident
    GEMV, an embedding lookup, and a routed-expert decode pair, all
    parity-tested. Q6_K runs on a resident GEMV plus an embedding lookup, the
    second added by ROADMAP Phase S when a second real file finally put
    `token_embd` there; it has no MoE pair, so an install with Q6_K experts
    passes the manifest gate and fails at the routed dispatch, by name.
    PHASE M2 ADDS Q5_K on the narrowest footing yet (a resident GEMV and
    nothing else: Mixtral 8x7B's Q4_K_M puts it on `attn_output` alone) and
    gives Q6_K a routed PHASE 2, because that same file puts Q6_K in the
    `ffn_down_exps` of 16 of its 32 layers -- the first mixture whose two
    types differ across LAYERS of one model rather than across the phases of
    one expert. PHASE S ADDS IQ3_XXS, IQ4_NL AND IQ4_XS ON THE SAME PARTIAL FOOTING, and
    the partition is its candidate's rather than a symmetry: IQ3_XXS and
    IQ4_XS have a routed phase 1 (gate/up) and IQ4_NL a routed phase 2
    (down), because that is where the real file puts each, and all three have
    a resident GEMV. Q4_0 has nothing and is refused.
    ROADMAP M5 ADDS MXFP4 ON A FOOTING THAT IS NARROW IN A NEW DIRECTION:
    BOTH routed phases and NO resident GEMV and no embedding lookup, where
    Q5_K and Q6_K are the reverse. That is `gpt-oss-20b-MXFP4.gguf`'s own
    shape rather than a choice -- it puts MXFP4 in
    `ffn_{gate,up,down}_exps` and keeps attention, `token_embd` and
    `output` at Q8_0 -- so the usual three-kernels-per-type rule cost two.
    The two gates are independent on purpose, and each reads a different
    thing. `model_io::validate_quant` reads the manifest's per-SLOT
    `ggmlType` against `model_io::EXECUTABLE_GGUF_TYPES`;
    `RealForwardRunner::open` reads the resident index's dtype TAGS against
    `EXECUTABLE_GGUF_DTYPES`, which is the backstop for a hand-edited
    manifest and believes the bytes rather than the claim. Both directions
    are asserted (`crates/runtime/tests/gguf_install_refused.rs`, which
    also decodes), so widening either means landing kernels rather than
    editing a list.
    **THE TWO LISTS ARE TWINS AND NOT COPIES, AND MXFP4 IS THE FIRST TYPE
    TO SHOW IT.** This gotcha used to say they "have to move together or
    one of those tests reddens", which held while every type was executable
    in both senses and is not a rule. They answer different questions:
    "does this type have the kernels its SLOT needs" against "can this
    TENSOR be dispatched". MXFP4 is executable in the first sense and not
    the second, so `"mxfp4"` joins `EXECUTABLE_GGUF_TYPES` and its tag 14
    does NOT join `EXECUTABLE_GGUF_DTYPES` -- an install with MXFP4
    attention passes the manifest gate and is stopped by the backstop,
    which is the layering working rather than a leak. Both halves are
    asserted on one install by
    `mxfp4_is_refused_as_a_resident_tensor_though_its_experts_run`.
    A related drift the same phase found: `GGUF_BLOCK_DTYPES`, the list the
    backstop tests membership of FIRST, had been one behind the writer
    since Q5_K landed in M2, so a resident Q5_K tensor was invisible to it.
    Harmless while Q5_K was executable, and exactly what a bare-literal
    list invites; it is spelled out of the named constants now.
    A MIXED file is the case a Gemma GGUF cannot exercise, and it is the
    normal one: `Q4_K_M` means Q4_K experts and embedding, Q8_0 attention and
    shared experts, one Q6_K tensor. So the block type has to be read PER
    TENSOR at each dispatch site, not decided once at open. THE ROUTED BLOBS
    USED TO BE THE ONE EXCEPTION and are not any more: Phase S's candidate
    reads IQ3_XXS gate/up against an IQ4_NL down in the SAME expert, and its
    layer 29 is IQ4_XS over Q8_0, so `RoutedBlobLayout` is per LAYER and per
    PHASE now, resolved from `packed_experts/layout.json`'s per-sub-tensor
    dtypes rather than from the manifest's one `ggmlType` (which stays as the
    dominant type, beside an optional `ggmlTypes` array the gate checks member
    by member). THE EXPERT STRIDE IS PER LAYER FOR THE SAME REASON, and that
    one is not tidiness: layer 29's blob is 1.6x the others', so the old
    model-wide maximum would pad a 10.33 GB expert table to 16.23 GB, turning
    the phase's -20% into a +35% regression with NO visible symptom, since a
    zero-padded blob decodes correctly. `SyntheticGgufShape::k_quant()` builds
    that mixture as a fixture; every K-quant row in it is 256 elements
    because a superblock cannot be partial.
    A FIFTH DIFFERENCE IS NOT ABOUT THE FORMAT AT ALL and is the one that
    decided whether Qwen runs: the same tensor can MEAN something else in
    llama.cpp's checkpoint than in mlx-community's, with the name mapping
    correct either way. See Gotcha 33.

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

32. **Every error raised inside an encode sequence used to abort the
    process, and the abort said nothing.** A `PassEncoder` dropped on a `?`
    path never sent `endEncoding`, so Metal killed the process from
    `-[_MTLCommandEncoder dealloc]` with "Command encoder released without
    endEncoding" while the real error was still travelling up the stack.
    Fixed by a `Drop` impl (`crates/gpu/src/context/pass.rs`), and the cost of
    having one is that `commit` takes its profile with `Option::take` and
    clones the command buffer rather than moving fields out. Two corollaries:
    any older note describing a Metal crash on this path may be describing an
    ordinary error message, and a new pass-like wrapper needs the same
    treatment or it reintroduces the blindfold.

33. **A source-convention difference belongs to an AXIS, not to the tensors
    you happened to be able to compare.** Qwen's GGUF orders V heads
    interleaved where the MLX checkpoint keeps them contiguous. Three
    tensors were characterized first (`A_log`, `dt_bias`, `conv1d.weight`)
    and all three transforms were correct, verified on every layer of the
    real install at worst relative error 0.000000. **The model still
    generated gibberish**, because five more tensors index the same axis
    (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `out_proj`) and
    the only thing that had made the first three visible was that they are
    the ones GGUF ships as F32, so a BF16-only probe could compare them
    directly. The selection criterion was the INSTRUMENT'S REACH, and it
    read as a finding. When a convention gap is found, enumerate by the
    dimension it indexes and check every tensor carrying that dimension
    before believing a list. The instrument for the rest was
    `gguf_qwen_quant_probe.rs`: dequantize one representative row per head
    and correlate same-index against the candidate (0.08-0.24 versus
    0.993-0.995 here), because two installs hold different quantizations and
    equality is unavailable. Where a tensor has one row per head the
    permutation is RECOVERABLE outright by argmax over a row-by-row
    correlation matrix, which is stronger than confirming a guess.
    **A SECOND INSTANCE LANDED IN ROADMAP PHASE M2, on a different axis and
    a different family, found the same way.** llama.cpp rotates ADJACENT pairs
    `(2i, 2i+1)` for the `llama` architecture where this port's
    `rope_proportional_neox` rotates half-split pairs `(i, head_dim/2 + i)`,
    and its HF converter absorbs the difference into the WEIGHTS: each head's
    rows are reshaped `(2, D/2)` and swapped to `(D/2, 2)`. Every name mapped,
    every shape checked out, the install opened and decoded, and the output
    was degenerate. `transcode.rs::unpermute_rotary_rows` undoes it for
    `attn_q` and `attn_k` only (V never goes through RoPE). The lesson to
    carry: when a new family's real install produces WORD SALAD rather than an
    error, suspect a convention on an axis some kernel indexes, and reach for
    the in-place patch loop before a repack -- it turned a 35-minute
    hypothesis test into a 10-second one here too.
    Two corollaries. Permuting a quantized tensor needs no dequantization:
    block layouts tile along the fastest-varying dim, so a row is a
    contiguous byte run and a head-wide column group is a whole number of
    blocks (checked, not assumed -- a Q4_K superblock is wider than a
    128-element head and must be refused rather than split). And verifying
    this class of fix does not need a repack: the tensors sit at fixed
    offsets, every transform preserves length, and `open()` runs no receipt
    or SHA-256 check, so patching the install in place turns a 23-minute
    loop into a seconds-long one
    (`crates/repack/tests/gguf_qwen_convention_patch.rs`). Do it as that
    kind of Rust TEST rather than an ad-hoc script: binary-patching a
    multi-GB install from a shell one-liner is refused as irreversible local
    destruction, and rightly.

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
    Look for siblings before assuming this one is done: `scripts/kld.py` is
    Gemma-pinned throughout (`REPO`, and a docstring asserting softcap 30),
    which is BY DESIGN there -- its reference checkpoint is a Gemma repo --
    but the same audit is owed to any measurement script that claims to take
    an arbitrary install.

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

41. **CHAT FRAMING IS A PROPERTY OF THE CHECKPOINT. The dialect resolved from
    the special-token table is not evidence about it, and the two come apart
    inside a single family.** Fourth instance of 37's shape, and the one that
    reaches the user's screen rather than a counter.
    `dialect.rs` picked `ChatDialect::Mistral` for any tokenizer carrying
    `<s>`/`</s>` and no Gemma/ChatML/DeepSeek marker, then rendered `[INST]`
    from it. TinyLlama-1.1B-Chat presents exactly that table -- `<unk>`,
    `<s>`, `</s>` and nothing else, because Zephyr's `<|user|>` /
    `<|assistant|>` are PLAIN TEXT and never enter the table -- so the probe
    cannot tell it from Mistral-7B-Instruct and never could. Fed `[INST]`, it
    echoes the markup back instead of answering: fluent output, not an answer,
    and no error anywhere. `apply_chat_template` now prefers the checkpoint's
    own Jinja template and keeps the dialect for what it IS evidence for --
    BOS/EOS and turn ids, the stop set, and the fallback render for a
    checkpoint that ships no template.
    THE TEMPLATE HIDES IN TWO PLACES, which is why this looked like an
    absence rather than a mis-read: HF moved it out of
    `tokenizer_config.json`'s `chat_template` key into a standalone
    `chat_template.jinja` partway through, and llama.cpp's GGUF converter
    still writes the old one. `load_from_dir` read only the file, so every
    GGUF-derived install looked template-less. Note the split does not follow
    family: gemma4, qwen36 and gemma4-iq3 ship the file, while qwen3moe and
    both dense `llama` installs ship the key. The key also has a NAMED-LIST
    form (`default` / `tool_use`), which a string-only reader reports as
    absent rather than rejecting.
    ONE FAMILY'S FROZEN ROW MOVED, and the axis is `trim`. The per-dialect
    renderers call `raw.trim()` on message content unconditionally; a
    template trims only where it says `| trim`. Gemma's and Qwen 3.6's do, so
    those three rows are untouched to the last hex character. Qwen3-30B-A3B's
    does not, and the protocol prompt is a file ending in `\n`, so ONE
    TRAILING NEWLINE now reaches the model: `qwen3moe_quality_gate`'s
    perplexity went 14.7576 -> 14.5988 (-1.08%, inside the 2% tolerance and
    in the improving direction) and both digests changed. Re-frozen, not
    suppressed -- the new values reproduce across two processes, and the
    verbatim content is what HF, vLLM and llama.cpp all send.
    THE INSTRUMENT MISTAKE IS THE PART TO REMEMBER. The guard test that was
    supposed to answer "will the digests move" said no, because its fixture
    was a tidy one-line string on which `trim` is a no-op. The real protocol
    case is a multi-line file with a trailing newline, and a comparison
    fixture chosen for tidiness cannot see the property being asserted.
    `crates/tokenizer/tests/installed_template.rs` now `include_str!`s the
    protocol prompt itself rather than retyping it, and pins per family both
    that the two renders differ by NOTHING but the trim and whether that
    family's template trims. Env-gated, seconds, no GPU: run it before the
    quality gates, not after -- it answers "will the digests move, and for
    which family" for the price of a string compare.

42. **A ROUTED SUB-TENSOR IS NOT ALWAYS A MATRIX, AND NOT EVERYTHING IN AN
    EXPERT BLOB IS A QUANTIZATION SCHEME.** ROADMAP M5's `gpt-oss` is the
    first checkpoint to put a per-expert BIAS in the routed blob, and both
    halves of the walk had quietly assumed otherwise since Phase G. Neither
    assumption was wrong when written; both were unstated defaults that one
    new model turned into refusals.
    - **`plan.rs` demanded RANK 3.** Every routed tensor before this was
      `[in, out, experts]`, a matrix per expert. A bias is `[out, experts]`,
      a VECTOR per expert, and the walk refused it by name. `routed_body_dims`
      now drops the trailing expert axis and takes whatever is left, so the
      two ranks are one case. The nastier half was the shape RECORD rather
      than the refusal: `out_total` was read as `dims[1]`, which on a rank-2
      bias is the EXPERT COUNT -- a plausible number that would have written
      a correctly-sized blob with a nonsense logical shape, and no byte
      assertion catches that.
    - **`gguf_manifest_quant` counted the F32 bias planes as routed BLOCK
      TYPES**, so `validate_quant` refused a perfectly runnable install for
      "F32 has no kernel". A bias is a COMPANION: `moe_gguf.metal` reads it
      as a plain `device const float*` off an offset the weight's kernel
      already resolved, exactly as the affine layout's `gate_scales` relates
      to its packed run. Those escaped notice only because an affine blob's
      sources are not GGUF tensors, so the histogram never saw one. Excluded
      by ROLE rather than by `dtype != F32`, so a future BF16 bias is
      classified by what the sub-tensor IS.
    THE REUSABLE PART IS WHEN IT WAS FOUND. Both cost milliseconds against
    `SyntheticGptOssShape` and would have cost a 25-minute re-stream apiece
    against the real 12.1 GB file -- `crates/repack/CLAUDE.md` Gotcha 8's
    rule (build the structurally-real fixture BEFORE the download) paying off
    for the second time, on a phase that was warned to be M2-sized. Note the
    fixture had to differ from the real file to be useful: gpt-oss's `hidden`
    and expert width are both 2880, so the published checkpoint cannot tell a
    gate/up bias from a down bias, and a fixture that copied its proportions
    could not either.

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

45. **A WRITER MAY ONLY RECORD A TAG SOME READER HONOURS. The safetensors
    walk had three raw dtype tags and the runtime reads exactly one of
    them.** `raw_dtype_tag` mapped `BF16`/`F16`/`F32` onto tags 1/2/3 and
    nothing in `crates/runtime` has ever read 2 or 3: `norm_view`,
    `read_bf16_host` and every kernel binding a `device const bfloat*`
    identify an unquantized tensor by BYTE SIZE and decode it as BF16. So an
    F16 norm written verbatim is not refused, it is MISREAD -- same width,
    every length check passes, and the values come out wrong by up to 2^112.
    Found 2026-08-14 by reading the real `prism-ml/Bonsai-27B-mlx-1bit`
    header, where EVERY unquantized tensor is F16; the four checkpoints
    before it are BF16 throughout, which is why a tag with no reader survived
    since the first repack walk.
    THE FIX IS THE SHAPE THE GGUF SIDE ALREADY HAD, and the difference between
    the two is the part worth keeping. `transcode_f32` narrows F32 to BF16
    because llama.cpp had UPCAST from BF16, so that narrowing was measured to
    be exactly lossless (Gotcha 29). `narrow_raw_to_bf16` does the same thing
    to F16 and it is NOT lossless -- F16 carries 10 mantissa bits against
    BF16's 7 -- so the two decisions look identical and rest on opposite
    measurements. Measured off the real file before writing any code: the
    five RMS-norm families lose 19.5% of their values at a worst relative
    error of 0.003891 (2^-8, BF16's own quantum), and the gated-DeltaNet
    tensors lose NOTHING because that QAT checkpoint stores them on a grid
    coarse enough to be exact in both (its layer 0 `A_log` has one distinct
    value across 48 elements). The real repack reproduced it exactly: 161
    tensors narrowed lossily, 546,190 values, and the 192 GDN tensors clean.
    Accepting the loss is a JUDGEMENT, recorded as one: the alternative is an
    FP16-weight variant of `rms_norm_bf16w` and its `_perhead` sibling plus a
    dtype threaded through every `norm_view` call site in four family flows,
    and 0.4% on a norm scale is not the limiting error in a model whose
    weight matrices are ONE BIT. If the cross-engine KL for this family lands
    above its backend floor, this is the first place to look.
    TWO STRUCTURAL COROLLARIES, both landed. The walk narrows rather than
    tags, so no install can carry 2 or 3 (`raw_dtype_tag` is DELETED rather
    than left beside its replacement -- a function whose only use would be to
    record a claim no reader honours). And `RealForwardRunner::open` now
    refuses any resident dtype with no reader BY NAME, which is the same
    backstop relationship the GGUF dtype gate has to
    `model_io::validate_quant`: the writer states, the reader verifies, and
    neither is trusted to be the only check.

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

50. **A NORMALIZATION CONVENTION IS A PROPERTY OF THE TENSOR, NOT OF THE
    FAMILY, AND ONE MODEL CAN USE TWO.** `muse_glimmer`'s four per-layer
    norms are `CenteredRMSNorm` -- the stored weight is an OFFSET FROM UNITY,
    so the effective scale is `1 + w` -- while its FINAL norm
    (`model.norm.weight`) is a plain `nn.RMSNorm` at `w`. Both live in one
    forward pass, so "which norm does this family use" has no answer; only
    "which norm does this TENSOR use" does. `rmsnorm_bf16w_centered` and
    `rmsnorm_bf16w` are therefore separate kernels selected per dispatch
    site, and `families/museglimmer/mod.rs` calls each by name.
    THREE THINGS THAT FOLLOW, each a decision rather than an observation.
    **The `+1` is a SEPARATE KERNEL and not a function constant.** A
    specialization axis whose byte missed `MetalContext::pipeline`'s
    `constants_key` silently reuses whichever pipeline compiled first
    (`crates/gpu` Gotcha 1), and here that would make the two norms of one
    model the same function -- fluent, wrong, and invisible. A distinct
    kernel name is a distinct pipeline by construction.
    **The `1 +` is applied on the FP32 accumulator and never baked into the
    stored weight at repack time.** Baking is the obvious cheaper option and
    it is lossy where it matters: a centred weight sits near ZERO, BF16's
    absolute resolution near 1.0 is 2^-8 = 0.0039, so a `w` of 0.01 comes
    back with ~39% of its own magnitude destroyed -- on the scale of a
    normalization. Contrast Gotcha 45's F16-to-BF16 narrowing, which IS
    accepted: that one was measured at 0.0039 worst case on values whose
    magnitude is ~1, not ~0.01.
    **Gemma's `+1` was a real candidate and was REFUTED by measurement**
    (`gguf_norm_convention_probe`: its GGUF and MLX installs' resident cores
    are bit-identical, and llama.cpp's converter does add 1 to Gemma norms),
    so `rms_norm` stays the plain form and this is an addition rather than a
    switch. Do not "unify" them.
    THE TESTING LESSON IS SEPARATE AND GENERALISES FURTHER. A parity fixture
    whose norm weights sit near zero CANNOT TELL THE TWO CONVENTIONS APART,
    because `x * w` and `x * (1 + w)` converge as `w -> 0`.
    `the_two_norm_conventions_are_different_functions` asserts that the
    fixture discriminates before the parity case is believed -- the same
    discipline Gotcha 48 states for sub-4-bit packing, on a third axis.
    **A SECOND INSTANCE LANDED 2026-08-18 AND IT IS THE SHARPER FORM OF THE
    RULE.** `muse_glimmer`'s two conventions sit on tensors with DIFFERENT
    names (per-layer against final), so a reader can at least tell them apart
    by sight. The `qwen3_5` family's do not: its trunk and its
    multi-token-prediction head both carry `self_attn.q_norm.weight` and
    `self_attn.k_norm.weight`, at the same shape, resolved by the same
    `norm_view` call inside the same `encode_full_attention_block` -- and the
    head's are centered while the trunk's are plain, because mlx-vlm's
    converter bakes the `+1` into the published trunk weights and leaves the
    head's alone. So the convention is not even a property of the NAME; it is
    a property of the tensor at a call site, and `QkNormConvention::{Plain,
    Centered}` is a parameter on that shared function rather than anything
    derivable from the prefix. `rmsnorm_bf16w_perhead_centered` is the kernel.
    Two further things this instance settled, both against the grain of the
    paragraphs above. **The near-zero argument against baking does NOT apply
    to every centered tensor**: these two weights read 0.780 and 0.797, so
    baking would cost a factor of two in resolution rather than 39% of the
    magnitude, and MTPLX does bake them. A separate kernel was still taken,
    on exactness and on the smaller API, but the objection has to be re-argued
    per tensor rather than quoted. And **"the deviation is measurably small"
    needs the measurement that would be SENSITIVE to it**: this pair was
    deferred a session on the strength of 23/24 top-1 agreement, a scalar
    already saturated at the top of the curve, and it was worth 21 to 75
    percent of the speculative speedup once per-position acceptance -- the
    metric that could see it -- was read instead (`docs/MTP.md`).

51. **EVERY TEST IN A PERTURBATION-STYLE FIXTURE FILE CAN BE
    SELF-RELATIVE, AND THEN THE FILE CATCHES ALMOST NOTHING.** The
    reachability pattern `docs/NEW_MODEL.md` recommends -- perturb a tensor,
    require the logits to move -- rebuilds its own baseline inside the same
    binary. So a mutation that changes the MATH for every arm equally leaves
    every case green: both the baseline and the perturbed run carry it.
    Measured on `crates/runtime/tests/real_forward_muse.rs`, 2026-08-15: six
    mutations, and only ONE reddened (dropping the attention output gate,
    which makes a tensor unreachable and so breaks a REACHABILITY invariant
    rather than an arithmetic one). Dropping a Q scale, rotating a NoPE
    layer, swapping a norm convention, dropping an output multiplier and
    collapsing two epsilons ALL SURVIVED a file of twelve tests.
    The fix is one number: a FROZEN DIGEST over a deterministic synthetic
    install's logits, which is the only assertion in such a file that
    compares against something computed BEFORE the mutation. It took the
    same six mutations to five reddening. It is a CHANGE DETECTOR and not a
    correctness claim -- untrained weights cannot say the arithmetic is
    right, only that it is what it was -- so re-freezing it needs a stated
    reason, and a digest updated reflexively protects nothing.
    ONE MUTATION SURVIVES EVEN THE DIGEST and is worth knowing as a limit:
    collapsing an RMS epsilon of 1e-5 into 1e-8. At FP16 with `mean_sq` near
    1 those differ by ~5e-6 relative, an order of magnitude under FP16's
    resolution. Epsilon VALUES are pinnable offline against the checkpoint's
    own `config.json`; which epsilon reaches which norm site is a question
    only a real-model quality gate can answer.
    **CONFIRMED ON A SECOND FILE 2026-08-18, and this time the file had NO
    digest at all.** `real_forward_qwen35.rs` (the dense `qwen3_5` trunk) was
    seven reachability cases and nothing else, so flipping that flow's q/k
    norms to the CENTERED convention -- a wrong model that still decodes --
    left it green, left `real_forward_qwen.rs`'s eight green, and was caught
    only by `qwen38_quality_gate` on a real 14 GB install two minutes away.
    `the_dense_trunk_logits_have_a_frozen_digest` closes it in 0.2 s.
    **AND ITS POSITION MATTERS, WHICH IS A TRAP THE DIGEST PATTERN DOES NOT
    WARN ABOUT.** The obvious implementation reuses the file's existing
    `first_logits` helper, which produces at position 0 -- where a softmax over
    one key is exactly 1.0, so attention returns V alone and NO q/k transform
    is observable. Measured: cut to a single position, the correct model and
    the flipped one digest IDENTICALLY (`312e17a0`); over eight positions they
    differ. That is the same blindness that made `qwen3moe`'s divergence test
    measure nothing until it was moved off position 0, arriving inside a change
    detector instead. A digest is only a change detector for changes its
    inputs can reach.

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

54. **A CACHE THAT ALREADY DEDUPLICATES MAKES A "UNION" SAVING VANISH, AND
    THE TOUCHES-VERSUS-MISSES SLIP READS AS A 3.3x WIN.** The first draft of
    `docs/BATCHED_PREFILL.md` costed batched prefill's largest term by
    comparing the DISTINCT experts a 16-token chunk routes to (~38 per
    layer) against the REQUESTS 16 sequential tokens issue (`16 x top_k` =
    128), and read a 3.3x cut in expert bytes off the ratio. The sequential
    arm never paid 128: the slot cache turns those requests into MISSES, and
    measured on the real install at 32 slots that is 1.507 per layer per
    token, i.e. 24.1 over the same window. **The union is LARGER than what
    the engine already loads**, at every M and both slot counts but one.
    `scripts/router_window.py`'s own docstring says this ("the sequential
    arm's true cost is misses, not touches") and the doc was written past it.
    THREE THINGS TO CARRY, because the shape recurs whenever a cache sits
    under an optimization.
    **The saving a union can offer is bounded by intra-window EVICTION
    re-reads, not by the request ratio**, and at a cache big enough to hold
    the working set there are none: 24.1 loads against 41.5 distinct means
    most of the window was already resident before the window began.
    **The one cell where it wins is the cell where it aborts.** At 16 slots
    and M=16 the union is 41.5 against 48.7 sequential, a real 15% -- and
    `ExpertCache::plan_if_possible` ASSERTS `experts.len() <= slot_count`,
    so asking for 41.5 experts against 16 slots kills the process rather
    than degrading. Legal only at M<=2 (16 slots) or M<=8 (32), where it
    saves nothing. The mechanism is useful exactly where it is unavailable.
    **And the borrowed table was a DECODE one.** `union(M)` had only ever
    been measured with prefill excluded (`router_window.py`'s `skip` is the
    prompt token count); prefill's own union runs higher, 41.5 against
    37.8-39.4. That is Gotcha 38's species on a third axis and the same
    mistake the doc's attention section already warned about two paragraphs
    up -- a share measured on decode does not transfer to prefill.
    The driver that landed instead batches the ATTENTION half of a layer and
    leaves the routed half per token (`crates/runtime` Gotcha 14), measures
    1.22x, and moves the hit rate from 81.2% to 81.4%, which is what "the
    union does nothing" looks like from the other side.

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
    `Reasoning`, and both callers (`ChannelSplit`, `needs_decoder`) build a
    decoder for those dialects when a level was asked for. Skipping it is not
    cosmetic: measured on the real Gemma 4 install, the first `--reasoning
    low` run printed a bare `thought` (the channel LABEL, as prose), then the
    model's scratch work, then its answer, all as one run of content, because
    the frame tokens render to the empty string.

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

## Per-Crate Documentation

When working on code inside a specific crate, refer to that crate's `CLAUDE.md` file for crate-specific architecture, key modules, dev commands, and localized gotchas:

- [`crates/bench/CLAUDE.md`](crates/bench/CLAUDE.md): Throughput benchmark harness, mach memory sampler, frozen protocol, memory oracle test rules.
- [`crates/catalog/CLAUDE.md`](crates/catalog/CLAUDE.md): the curated model table, the header-only Hugging Face probe, the install driver, and the `~/.turbospark` store.
- [`crates/cli/CLAUDE.md`](crates/cli/CLAUDE.md): CLI binaries (`turbospark-check`, `turbospark-model`), process entry point, real model smoke tests, interactive chat REPL.
- [`crates/compute/CLAUDE.md`](crates/compute/CLAUDE.md): CPU reference kernels (RmsNorm, RoPE, Attention, Quant), numerical ground truth for GPU tests.
- [`crates/core/CLAUDE.md`](crates/core/CLAUDE.md): Shared primitives (`TokenId`, `LogitValue`), runtime configuration, allowed sets, chunk sizing.
- [`crates/ffi/CLAUDE.md`](crates/ffi/CLAUDE.md): the C ABI for native GUI hosts, its ownership and threading contract, and the Swift package over it.
- [`crates/gpu/CLAUDE.md`](crates/gpu/CLAUDE.md): macOS Metal context, pipeline caches, MSL shaders, KV cache, zero-copy weights, profiling flags.
- [`crates/invocation/CLAUDE.md`](crates/invocation/CLAUDE.md): Pure CLI argument parser, `InvocationRequest`, 5-place rule for adding new flags.
- [`crates/model-io/CLAUDE.md`](crates/model-io/CLAUDE.md): Manifest validation, architecture baselines, packed expert layout, mmap resident weight index.
- [`crates/repack/CLAUDE.md`](crates/repack/CLAUDE.md): Safetensors header parsing, ranged HTTP downloads, `.gturbo` writer, synthetic model builders.
- [`crates/runtime/CLAUDE.md`](crates/runtime/CLAUDE.md): Raw completion generation loop, `LogitProducer` contract, `RealForwardRunner` decode engine.
- [`crates/selection/CLAUDE.md`](crates/selection/CLAUDE.md): Candidate token selection, temperature/top-k/top-p shaping, repetition penalty, logits contract.
- [`crates/server/CLAUDE.md`](crates/server/CLAUDE.md): the `turbospark-server` HTTP server (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, `/v1/models`), Axum handlers, SSE streaming, `anyllm_translate` wire types.
- [`crates/streaming/CLAUDE.md`](crates/streaming/CLAUDE.md): Routed expert `pread` streamer, LFU/LRU slot cache policy, chunked reads on a persistent `read_pool`, macOS `F_RDADVISE` hints.
- [`crates/tokenizer/CLAUDE.md`](crates/tokenizer/CLAUDE.md): Tokenizer wrapper (`MfTokenizer`), chat dialects, Jinja template rendering, stop matcher, fixture token IDs.
- [`crates/window-fit/CLAUDE.md`](crates/window-fit/CLAUDE.md): Pure conversation window fitting (`fit_conversation_window`), turn dropping logic.

## Layout

Workspace directory structure and crate layout:

```
.
+-- Cargo.lock         # lockfile committed for reproducible workspace builds
+-- Cargo.toml         # workspace manifest declaring members and workspace metadata
+-- AGENTS.md          # developer guide and gotchas (CLAUDE.md is a symlink to this)
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
|   +-- core           # shared primitives (TokenId, LogitValue), RuntimeConfig, chunking
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
|   \-- window-fit     # deterministic conversation-window fitting & turn dropping
+-- swift
|   +-- TurboSpark     # SwiftPM package wrapping crates/ffi (session actor, AsyncStream, catalog)
|   \-- TurboSparkDemo # minimal SwiftUI chat app; verifies the binding end to end
+-- scripts
|   +-- kld.py         # cross-engine KL vs mlx-lm (reads tests/logit_dump.rs's output)
|   +-- kld_llamacpp.py# the same, vs llama.cpp on the same GGUF bytes (Gotcha 34)
|   +-- kld_mlx_affine.py # the same, vs MLX at ONE or TWO bits (the first needs the PrismML mlx fork)
|   +-- llamacpp_logits.c # its harness: ids in, full-vocab logits out, via libllama
|   +-- parity.sh      # head-to-head protocol run against the Swift MferenceCLI
|   +-- phasediff.sh   # bucket-level decode phase diff against the Swift engine
|   \-- power.sh       # watts & joules-per-token over the protocol (needs sudo)
\-- docs
    +-- ACTIVATION_SPARSITY.md # dense-FFN neuron caching/streaming, measured negative
    +-- BATCHED_PREFILL.md # the one open gap against Swift: scope, cost, order of work
    +-- BENCHMARKS.md  # the FROZEN rows: quality, throughput, memory, cross-engine KL
    +-- BENCHMARKING.md# benchmark modes, mach memory sampling & memory oracle details
    +-- DECODE_BUDGET.md # where a decoded token's time goes; three decode dead ends
    +-- EXPERT_ROUTING.md # domain-restricted expert sets, measured negative
    +-- GTURBO.md      # the .gturbo install format this port reads and writes
    +-- MODELS.md      # the catalog, the probe, `pull`, and how to add a row
    +-- MODEL_FAMILY.md# GGUF `general.architecture` / HF `model_type` tables
    +-- MTP.md         # the MTP head: architecture and MEASURED findings only
    +-- MTP_SPECULATIVE.md # native MTP heads on the DENSE family; pays, after a c(M) fix
    +-- NEW_MODEL.md   # end-to-end checklist for wiring a new model family
    +-- SPECULATIVE_DECODING.md # DFlash / batched verify, measured marginal
    +-- SWIFT_BINDINGS.md # the C ABI and the Swift package: examples, contract, limits
    +-- POWER_BASELINE.md # watts, joules-per-token, hygiene audit (ROADMAP Phase P1)
    \-- TESTING.md     # test suite organization, platform gating & testing rules
```

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

- `crates/core`: shared primitives (`TokenId`, `LogitValue`, `LogitsView`), error types (`CoreError`), runtime configuration (`RuntimeConfig`, `RuntimeConfigBuilder`), allowed value sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`), automatic chunk-size resolution (`chunk_sizing.rs`), and prefill chunking primitives (`prefill.rs`). Details in [`crates/core/CLAUDE.md`](crates/core/CLAUDE.md).
- `crates/compute`: CPU reference kernels (RmsNorm, WHT, RoPE incl. Qwen's `rope_neox_subdim`, causal attention, int4/int8 affine quant + GEMV, the sub-4-bit MLX affine references (`quant_1bit.rs` and `quant_2bit.rs`, the second ROADMAP's ternary entry: same container, four elements per byte, a ternary grid MEASURED rather than assumed), the GGUF block-quant reference (`quant_gguf/`: Q8_0 and Q4_K dequant/quant/GEMV plus `pearson`; `quant_gguf_iq.rs` for the IQ codebooks; `quant_gguf_mxfp4.rs` for MXFP4, ROADMAP M5), embedding lookup, MoE FFN, the gated-DeltaNet chain (`gdn.rs`) and Qwen's gating kernels (`gating.rs`), logit softcap-softmax, RelError/tolerance table, sampling helpers) plus destination compute strategy marker type (`ComputeStrategy`). These are the numerical ground truth `crates/gpu`'s Metal kernels are validated against. Details in [`crates/compute/CLAUDE.md`](crates/compute/CLAUDE.md).
- `crates/invocation`: pure translation of command-line argument tokens into a validated invocation request (`InvocationRequest`), options definition (`OPTIONS`), diagnostics (`diagnostics.rs`), typed failures (`InvocationFailure`), usage rendering (`render_usage`), and pure outcome-to-exit-status and outcome-to-stream routing decisions. Performs no filesystem, environment, or process I/O. Details in [`crates/invocation/CLAUDE.md`](crates/invocation/CLAUDE.md).
- `crates/selection`: candidate selection (`select`, `select_from_logits`) from a per-candidate score vector under a validated shaping configuration (temperature, top-k, top-p, repetition penalty, seed), accumulated history, step position, determinism, and distribution guards. Numeric parity with any upstream implementation is out of scope; only the observable contract is exercised. Details in [`crates/selection/CLAUDE.md`](crates/selection/CLAUDE.md).
- `crates/window-fit`: pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation (`FitOutcome`, `DroppedTurn`), using a caller-supplied whole-conversation length measurement, until the measured length is under a caller-supplied bound or nothing eligible remains. An optional leading instruction turn and the newest turn are never removed. Performs no input or output and holds no state between calls. Details in [`crates/window-fit/CLAUDE.md`](crates/window-fit/CLAUDE.md).
- `crates/tokenizer`: wraps HF `tokenizers` crate (`MfTokenizer`); resolves Gemma 4 / ChatML (Qwen) / DeepSeek-V4 chat dialect from special tokens (`dialect/`); renders text-only chat templates plus DeepSeek's native tool chat (`chat_template/`); generic Jinja-templated tool chat for Gemma/ChatML (`minijinja` + `pycompat`, rendering `chat_template.jinja`); streaming detokenizer (`StreamingDetokenizer`) and stop matcher (`StopMatcher`) (stop set unions dialect stops with `generation_config.json` `eos_token_id` list); Gemma/Qwen/DeepSeek tool-call DSL parsers and streaming structured assistant-output decoder (`structured_decoder/`, which also splits `gpt-oss`'s Harmony channels into content and REASONING -- the one dialect whose frame is a header/body triple rather than a bracketing token pair, and, since `--reasoning` landed, no longer the only one whose thought channel is emitted rather than discarded (Gotcha 55); its TOOL CALLS come out of the same header parser plus `JsonValue::parse`, and out of `finish` rather than a token, since `<|call|>` terminates a call and is a stop -- Gotcha 49). Details in [`crates/tokenizer/CLAUDE.md`](crates/tokenizer/CLAUDE.md).
- `crates/model-io`: `manifest/` decode and field-by-field validation against a resolved `ArchConfig` (`arch_config/`, with canonical Gemma 4, Qwen 3.6, DeepSeek-V4-Flash and -- ROADMAP's 1-bit entry -- Bonsai-27B `qwen3_5` baselines in `arch_baselines/`), `packed_experts/layout.json` decode (`PackedExpertsLayout`), `model_weights.bin` resident tensor index reader (`ResidentIndex`), `mmap`'d resident-buffer view (`ResidentBuffer`), streaming SHA-256 verification (`sha256.rs`), and trusted install receipt (`InstallReceipt`). Allowed a narrow amount of `unsafe` (the `mmap` call). Details in [`crates/model-io/CLAUDE.md`](crates/model-io/CLAUDE.md).
- `crates/streaming`: routed-expert `pread` streamer (`PreadExpertStreamer`) with a fixed per-layer slot cache. The LFU/LRU eviction policy (`ExpertCache`) is pure logic, separated from file I/O so it can be tested against access traces without a model install. Cache misses are split into chunks and read on `read_pool`, a process-wide set of parked worker threads, so a layer that misses once still reads at full width (the `pread` is a page-cache memcpy, not disk I/O). `rdadvice` and `read_pool` are the other `unsafe`-carrying modules (macOS `F_RDADVISE`, a documented no-op elsewhere; raw destination pointers across worker threads). Details in [`crates/streaming/CLAUDE.md`](crates/streaming/CLAUDE.md).
- `crates/ffi`: the C ABI a native GUI drives the engine through (`staticlib` plus a hand-written `include/turbospark.h`), and the third crate carrying `unsafe`. An opaque session handle over a `Mutex<RealForwardRunner>`, a streaming event callback, JSON options and telemetry, and the catalog/probe/install surface. **The cancel flag lives OUTSIDE the session mutex**, which is the whole design: a GUI generates on a background thread and presses Stop on the main one, so a flag behind the lock would make Stop wait for the generation it is stopping. `swift/TurboSpark` wraps it and `swift/TurboSparkDemo` is a SwiftUI app proving the stack end to end. Details in [`crates/ffi/CLAUDE.md`](crates/ffi/CLAUDE.md).
- `crates/gpu`: Metal device/pipeline-cache context (`MetalContext`, `PassEncoder`, `CommittedPass`) and per-kernel dispatch. macOS-only; compiles to nothing elsewhere. Dispatched, parity-tested kernels (`rmsnorm_no_scale`, `rms_norm_bf16w`, both `_perhead` norm variants, `rope_proportional_neox`, `rope_neox_subdim`, `logit_softcap_softmax`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd` with resident variants, the port-local GGUF set (`dequant_q8_0_gemv_simd`, `dequant_q4_k_gemv_simd`, `dequant_q6_k_gemv_simd`, `embed_lookup_q8_0`, `embed_lookup_q4_k`, and `moe_gguf/` decode pairs -- ROADMAP Phase G), the port-local sub-4-bit set (`dequant_int1_gemv_simd` and `dequant_int2_gemv_simd`, each with a resident variant, the `+/-1` `dequant_int1_gemv_symmetric_simd`, and `embed_lookup_int1` / `embed_lookup_int2` -- ROADMAP's 1-bit and ternary entries; the 2-bit set is TWO kernels rather than three, because a ternary fast path would reassociate the sum exactly as the 1-bit one does and that one is already reachable from nothing; the general GEMV's resident form and the lookup are dispatched by the `qwen3_5` flow, the `+/-1` one is parity-tested and reachable from no decode path, deliberately), `router_gemv_gemma4_r4`, two-pass split-KV `attention_decode` (multi-chunk, split up to 16 ways by `chunks_for`), `moe_decode` decode pair, `gdn.metal`'s eight gated-DeltaNet kernels, and `utility` elementwise kernels incl. Qwen's three gating kernels) are compiled from MSL source at runtime, vendored from Swift except where marked port-local. `power_state.rs` wraps `NSProcessInfo`'s `thermalState` and `isLowPowerModeEnabled` for ROADMAP Phase P2 (here rather than in `runtime`, which forbids unsafe; nothing GPU about them beyond the `metal::objc` reach). `KvCacheManager` allocates and manages real per-layer Metal KV buffers used by `RealForwardRunner`. `ResidentGpuWeights` wraps resident mmap in zero-copy MTLBuffer. `GdnStateManager` is the Qwen flow's recurrent state; `Dsv4StateManager` allocates real per-layer Metal buffers (unwired kernels); `PrefillChunkScratchLayout`/`PrefillChunkScratchBuffers` size scratch buffers (undispatched tile kernel). The `sample` kernel and fused lm_head are not yet vendored or dispatched. Details in [`crates/gpu/CLAUDE.md`](crates/gpu/CLAUDE.md).
- `crates/runtime`: power policy for the decode loop (`power.rs`: `PowerProfile`, the `stepped_cap` thermal ladder, `RateControl`, and cfg-paired OS probes; `pacing.rs`: the pure-deadline `Pacer` -- ROADMAP Phase P2), and the raw-completion prefill+decode loop (`run_raw_completion`, `run_raw_completion_chunked`), wiring a `LogitProducer`, the tokenizer's streaming detokenizer and stop matcher, and `selection::select` into one token generation loop. `ScriptedLogitProducer` is what unit tests and `crates/server`'s `ScriptedChatModel` drive the loop with (see Gotcha 10). `RealForwardRunner` (macOS/GPU only, `src/real_forward.rs` plus one `src/families/<family>/` module per flow, each `mod.rs` + `attn.rs` + `moe.rs` + `state.rs`) is a real `LogitProducer`: a genuine transformer forward pass through real GPU kernels (including real GPU decode attention) and real quantized weights, supporting dense and MoE FFN layers. Dense bridges gated FFN on CPU via `turbospark_compute::run_ffn`; MoE runs real GPU router GEMV plus real GPU GEMVs for each selected expert, host-side top-k selection, and CPU-bridged gated activation. Supports synthetic short names, verbatim real Gemma 4 checkpoint names (learned-weight flow), the Qwen hybrid linear/full-attention flow (`src/families/qwen/`), which serves BOTH `qwen36` and -- since ROADMAP's 1-bit entry -- the DENSE, one-bit `qwen3_5`, forking at the FFN alone (`dense.rs`, no new kernel), one plain-GQA flow (`src/families/llama/`) serving BOTH the `llama` and `qwen3moe` families AND both halves of `llama` itself: Mixtral's routed experts and, since ROADMAP M4, the dense gated FFN of Mistral and Llama 2/3.x (`dense.rs`, no new kernel). `llama` and `qwen3moe` differ only in per-head q/k norms and an RMS epsilon. A FIFTH flow (`src/families/gptoss/`, ROADMAP M5) serves `gpt-oss`: the same plain GQA plus a bias on all four projections, YaRN rope off a precomputed frequency table, attention sinks, an alternating window on the EVEN layers, and MXFP4 routed experts carrying a clamped SwiGLU and per-expert biases. See Gotcha 12. Details in [`crates/runtime/CLAUDE.md`](crates/runtime/CLAUDE.md).
- `crates/catalog`: the model CATALOG (`models.json`, fourteen curated rows, each naming a repository and revision that were streamed and run on real hardware, with the gate targets that assert it, and for the eight that have been through an oracle a `measured` block carrying the peak and decode range that oracle was calibrated from), the RECOMMENDATION engine (`recommend/`: does this fit THIS machine, at what context, ranked by evidence -- reusing `model_io`'s two sizing policies rather than restating them, and gating discovered Hugging Face repositories through the probe; shape adapted from shoehorn, see `NOTICE`), the header-only Hugging Face PROBE (`probe/`: architecture through `repack`'s registry, block types against `model_io::EXECUTABLE_GGUF_TYPES` or the affine `(bits, group)` conjunction, expert-slot arithmetic, tokenizer sidecars -- KB and seconds, never a download), the INSTALL DRIVER (`install.rs`: the shape every install-writing `crates/repack/tests/*_network.rs` file repeats, written once and with the sidecars verified BEFORE any weight byte moves), and the `~/.turbospark` STORE (`store.rs`, `installed.json`, alias-to-path resolution in which an existing directory always wins). Builds on every platform; nothing here decodes. Details in [`crates/catalog/CLAUDE.md`](crates/catalog/CLAUDE.md) and `docs/MODELS.md`.
- `crates/cli`: the `turbospark-check` binary process entry point (see Gotcha 7). Parses `argv`, applies `invocation`'s exit-status and stream-routing decisions, prints the resolved request for a validated invocation, and (macOS, `src/generate.rs`) attempts real generation against `--model` via `RealForwardRunner` (see Gotcha 12) in all three modes: `--prompt` (raw text), `--messages-file` (rendered through chat template), and `--chat` (interactive REPL in `src/chat.rs`, trimming turns with `turbospark-window-fit`). `--model` accepts a catalog ALIAS as well as a path, resolved in `generate.rs` rather than in `invocation` (which is pure and keeps the value an opaque string). A SECOND binary, `turbospark-model` (`src/bin/model.rs` plus `src/bin/model_cmd/`), is the catalog and download surface: `list`, `info`, `probe`, `pull`, `path`, `rm`, with its own small subcommand parser because `invocation` is flat, pure and requires `--model`. Details in [`crates/cli/CLAUDE.md`](crates/cli/CLAUDE.md).
- `crates/repack`: safetensors header parsing (pure, tested against synthetic fixtures), `RangeSource` trait for ranged reads (`ranged_download/`, HTTP-backed for real installs, in-memory for tests) with two-step header-fetch plan, per-row int4/int8 quantization repack (reusing `turbospark_compute`'s quantizer), byte-exact `.gturbo` directory assembly (`write_gturbo_install`), real named resident-tensor index writer (`write_gturbo_install_with_resident_index`), synthetic install builders (`synthetic_model/`, `synthetic_real.rs`, `synthetic_qwen/` -- with dense taking the affine WIDTH as a parameter, so one dense fixture serves the 1-bit and 2-bit checkpoints), Hugging Face Llama checkpoint repacker (`hf_checkpoint.rs`), Gemma 4 mlx-community checkpoint repacker & streamed pipeline (`gemma4_checkpoint/`, family-parameterized so Qwen 3.6 goes through the same walk), Qwen 3.6 `config.json` parser (`qwen36_config.rs`, the one family-specific piece of that walk), install verifier (`install_verifier.rs`), manifest peeker (`manifest_peek.rs`), and the GGUF intake (`gguf_header/` parser, `gguf_names/` name mapping, `gguf_config/` metadata-to-`ArchConfig`, `gguf_checkpoint/` repack walk (expert bytes verbatim, resident F32 core transcoded to BF16/INT8, Qwen's V-head source convention undone at `v_head_axis`), `synthetic_gguf/` fixture writer -- ROADMAP Phase G; Q8_0, Q4_K and Q6_K installs are executable and Q4_0 is refused, see Gotchas 29 and 33). Details in [`crates/repack/CLAUDE.md`](crates/repack/CLAUDE.md).
- `crates/server`: HTTP server on loopback (`turbospark-server` binary, axum framework) serving OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`, both generation endpoints supporting full-response (non-streaming) and SSE-streaming responses. The wire types come from `anyllm_translate` (crates.io, default features: pure and IO-free), which also translates an Anthropic request into the OpenAI request the existing path understands and translates the result back, so Anthropic-native clients need no proxy. Tool calling is wired on both endpoints (request `tools` render through the checkpoint's `chat_template.jinja`, generated calls come back through `StructuredAssistantDecoder`); an incoming request's images and `thinking` CONFIG are dropped, some of it reported on an `x-anyllm-degradation` header, but a gpt-oss RESPONSE now carries its reasoning out as OpenAI `reasoning_content` and as an Anthropic `thinking` block, and its Harmony tool calls out as `tool_calls` / `tool_use` with the matching finish reason. Two backends behind the `ChatModel` trait: `RealChatModel` (macOS, `--model <install-dir>`, one mutex-serialized `RealForwardRunner` per process) and `ScriptedChatModel` (portable, canned completions, what the integration tests drive). Details in [`crates/server/CLAUDE.md`](crates/server/CLAUDE.md).
- `crates/bench`: the `turbospark-bench` binary plus benchmark library (`turbospark_bench`). The scripted default (three fixed prompts, fixed seed, discarded warmup) measures loop overhead via `ScriptedLogitProducer`. `--model <install-dir>` (macOS) is the real Swift-comparison mode: frozen community protocol (`protocol.rs`) driven through `RealForwardRunner`, reporting split prefill/decode tok/s and peak `phys_footprint` from the mach sampler (`memory.rs`). `tests/memory_oracle.rs` (`#[ignore]`d, gated on `TURBOSPARK_GEMMA4_INSTALL_DIR`) asserts peak footprint against per-chip baseline rows, plus a steady-state replay guard; each row carries a `source` recording whether it is a Swift parity number or this port's own measurement. The quality axis lives here too, all `#[ignore]`d: `tests/quality_gate.rs` and its Qwen sibling (per-install perplexity plus golden digests), `tests/quality_sensitivity.rs` (proof the perplexity responds to quantization damage), and `tests/logit_dump.rs` (full-vocab logits plus the exact token ids, feeding `scripts/kld.py`'s cross-engine KL against mlx-lm -- the one external reference in the whole quality section). Full details in [`crates/bench/CLAUDE.md`](crates/bench/CLAUDE.md) and `docs/BENCHMARKING.md`.
- `docs/`: repository documentation directory. `docs/BENCHMARKING.md` details benchmark harness modes, mach memory sampling, and the memory oracle baseline assertions; `docs/POWER_BASELINE.md` records watts and joules-per-token per install plus the power-hygiene audit (ROADMAP Phase P1), and is the one page here measured on BATTERY rather than AC; `docs/TESTING.md` documents test suite organization, macOS and environment-variable gating conventions, and test writing rules; `docs/MODELS.md` documents the model catalog, the header-only probe, `turbospark-model pull`, the `~/.turbospark` store, and the admission rule for adding a catalog row.

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

The four commands above cover everything except the `#[ignore]`d tests
(the checkpoint downloads, the two memory oracles, the two quality gates,
and the quality sensitivity proof), which are opt-in and not part of the
handoff gate. Run an oracle when a change could move memory or decode
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

See `DEVIATIONS.md` for the full list of what this port scaffolds versus
fully implements, `ROADMAP.md` for the forward roadmap and descope
record, and
`docs/NEW_MODEL.md` for the end-to-end checklist for wiring a new model
family (what to map, what to specialize, what to measure, in order), and
`docs/MODELS.md` for the catalog, the probe, and how to install a model that
is not in it.
