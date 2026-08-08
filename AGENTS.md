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
`mference-bench` modes, how peak memory is measured, and the memory
oracle that asserts this port against per-chip baseline rows (mostly the
published Swift numbers; see `docs/BENCHMARKING.md` for which rows are
Swift parity claims and which are this port measuring itself).

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
cargo test -p mrefrust-core
cargo test -p mrefrust-compute
cargo test -p mrefrust-invocation
cargo test -p mrefrust-selection
cargo test -p mrefrust-window-fit
cargo test -p mrefrust-tokenizer
cargo test -p mrefrust-model-io
cargo test -p mrefrust-streaming
cargo test -p mrefrust-gpu       # macOS only; needs a real Metal device
cargo test -p mrefrust-runtime
cargo test -p mrefrust-cli
cargo test -p mrefrust-repack
cargo test -p mrefrust-server
cargo test -p mrefrust-bench

# Formatting check (must stay clean; enforced in verification).
cargo fmt --check

# Apply formatting.
cargo fmt

# Lint the workspace and its tests (must stay clean).
cargo clippy --workspace --tests

# Run the CLI (validates the invocation; on macOS also attempts real
# generation against --model in all three modes: --prompt (raw text),
# --messages-file (JSON conversation, chat template applied), and --chat
# (interactive REPL). See DEVIATIONS.md for scope.
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model --prompt "hi"

# Run the server against a real install (macOS; one runner per process, so
# requests are served one at a time). It serves OpenAI
# `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`. Add
# `--bind tailnet` to bind this machine's Tailscale IPv4 address instead of
# loopback (no auth, no TLS: the Tailnet ACL is the only access control).
cargo run --release -p mrefrust-server --bin mference-server -- --model ~/models/gemma4.gturbo

# Point an Anthropic-native client straight at it, no proxy in between.
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true claude

# Same server, portable scripted backend (canned responses; DEVIATIONS.md).
cargo run -p mrefrust-server --bin mference-server -- <tokenizer-dir> [port]

# Run the throughput benchmark harness (scripted producer; see DEVIATIONS.md).
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir>

# Real-install benchmark (macOS): frozen community protocol against a real
# .gturbo install, reporting split prefill/decode tok/s and peak
# phys_footprint (the Swift-parity memory counter). Use --release.
cargo run --release -p mrefrust-bench --bin mference-bench -- --model ~/models/gemma4.gturbo

# The memory oracle: asserts endOfTurn on every protocol case, peak
# footprint under the ceiling, no growth on a replayed warm case, and
# (where a row exists) a decode tok/s floor. Each row records whether it
# came from Swift's docs/BENCHMARKS.md or from this port measuring itself
# -- printed every run. Skips with a note if the env var is unset. Takes
# ~10 minutes. See docs/BENCHMARKING.md.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test memory_oracle --release -- --ignored --nocapture

# Same oracle for Qwen 3.6. A SEPARATE target, not a second #[test]: the
# footprint assertion is a whole-session peak and the two families have
# different ceilings (~2,200 vs ~1,600 MiB), so they need one process each.
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_memory_oracle --release -- --ignored --nocapture

# The quality gate (ROADMAP Phase Q): teacher-forced perplexity of a fixed
# reference answer in the ASSISTANT slot (an instruction-tuned checkpoint
# is never trained to predict prompt tokens, so scoring those measures
# nothing), plus frozen greedy and sampled output digests, plus a
# constrained-working-set repeat at 8 expert-cache slots (digest frozen,
# throughput floored). Split per family for the same one-model-per-process
# reason as the oracle. About 80 seconds each. Numbers and caveats:
# docs/BENCHMARKS.md.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_gate --release -- --ignored --nocapture
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-bench --test qwen36_quality_gate --release -- --ignored --nocapture

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
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_LOGIT_DUMP_DIR=/tmp/kld/mrefrust \
  cargo test -p mrefrust-bench --test logit_dump --release -- --ignored --nocapture
uv run --python 3.12 --with mlx-lm --with numpy scripts/kld.py /tmp/kld/mrefrust

# Same dump with NO warmup walk, which is the condition quality_gate takes
# its perplexity under. Reproduces the frozen row exactly; that is the
# cross-check that the dump measures what the gate measures.
MREFRUST_LOGIT_DUMP_COLD=1 MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_LOGIT_DUMP_DIR=/tmp/kld/cold \
  cargo test -p mrefrust-bench --test logit_dump --release -- --ignored --nocapture

# Proof that the gate above can SEE quantization damage, rather than just
# asserting it could. Clones the install (APFS clonefile, so the original
# is untouched and only written pages cost disk), shifts one quantization
# level in a strided subset of the routed experts, and re-measures. About
# 30 seconds. Curve and floor: docs/BENCHMARKS.md.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-bench --test quality_sensitivity --release -- --ignored --nocapture

# Power baseline over the frozen protocol (ROADMAP Phase P1): watts and
# joules-per-token, split prefill/decode. NEEDS SUDO (powermetrics is
# root-only) and so cannot be run non-interactively. ~12 min per install.
# Windows the capture with the `[power-window ...]` markers mference-bench
# emits, so the model open and the discarded warmup stay out of the total.
# Numbers and caveats: docs/BENCHMARKS.md.
LABEL=battery OUT=/tmp/power-gemma MODEL=~/models/gemma4.gturbo scripts/power.sh 2

# Same harness driving an interleaved A/B of the read-pool QoS seam. Arms
# alternate WITHIN each pair, not as two consecutive batches. Measured and
# rejected once already (it loses on both joules and tok/s); the seam is
# kept as a documented dead end.
LABEL=battery MODEL=~/models/gemma4.gturbo CASES=short-explanation \
  QOS=default,utility scripts/power.sh 3

# GGUF intake (ROADMAP Phase G). Reads only the HEADER of the real
# published GGUFs -- a few MB off a 20-27 GB file, ~4 s each -- and checks it
# three ways: the parser agrees with what llama.cpp's converter writes, every
# tensor name maps, and the ArchConfig derived from GGUF metadata equals the
# one the corresponding .gturbo install declares. Set the install vars to get
# the last two cross-checks; without them it still parses and reports.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test gguf_checkpoint_network --release -- --ignored --nocapture

# Settles which half of Gemma's fused ffn_gate_up_exps is the gate (Stage 2's
# first use of the Q8_0 reference). Correlates a dequantized layer 0 expert 0
# against the same expert in the MLX install. Also a real-data check on the
# Q8_0 dequant itself: a sign, scale or block-layout error cannot correlate
# at +0.9957 with an independently-produced INT4 install. Few KB, ~5 s.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-repack --test gguf_fused_gate_network --release -- --ignored --nocapture

# The evidence behind the transcode decision (Gotcha 29): GGUF's F32 norms
# are upcast BF16 and narrow back bit-exactly, and INT8-transcoding its F32
# router does not move the routing decision. No install needed, few KB, ~6 s.
cargo test -p mrefrust-repack --test gguf_f32_transcode_network --release -- --ignored --nocapture

# The same real-data check for Q4_K, against the real Qwen 3.6 Q4_K_M: a
# dequantized layer 0 expert 0 gate row correlates +0.9930 with the same row
# in the MLX install, while the unrelated `up` matrix reads +0.12. It is the
# one check that can catch a decoder and its fixture quantizer being wrong
# TOGETHER, which nothing in crates/compute can. Few KB, ~5 s. Read Gotcha 30
# before believing a low number out of it.
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test gguf_q4_k_network --release -- --ignored --nocapture

# The other #[ignore]d tests: real checkpoint downloads (many GB).
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture
MREFRUST_QWEN36_INSTALL_DIR=~/models/qwen36.gturbo \
  cargo test -p mrefrust-repack --test qwen36_checkpoint_network --release -- --ignored --nocapture
```

### Real-model smoke (needs the pinned install)

Run BOTH of these on any change to the decode path, the output head, the
KV cache, or a Metal encode loop. Greedy alone is not a smoke test: it is
`argmax`, and `argmax` is invariant under every monotone transform of the
distribution, so it stays byte-identical to correct through bugs that
destroy sampling entirely (Gotcha 16).

```sh
cargo build --release -p mrefrust-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy. Catches broken math.
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. SAMPLED, at the CLI defaults (T=0.2, top-k 64, top-p 0.95). Catches
#    distribution bugs greedy cannot see. Must stay coherent for the whole
#    run and reach EndOfTurn on a short question.
./target/release/mference-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

A bare `--prompt` on an instruction-tuned model babbles: that is the chat
template missing, not a decode bug. `--messages-file` applies it for you.

Add each new crate directory to the `members` list in the root `Cargo.toml`
as it lands, and keep the member list in sync with the directories under
`crates/`.

A `Makefile` wraps the common cases: `make build-debug`, `make
build-release`, `make test-debug`, `make test-release`, `make fmt`, `make
fmt-check`, `make clippy`, `make check` (fmt-check + clippy + test-debug),
`make clean`.

## Gotchas

1. Downstream crates depend on `mrefrust-core` under an alias, for example
   `foundation = { package = "mrefrust-core", path = "../core" }`, and refer to it as
   `foundation`. The same aliasing pattern is used for every intra-workspace
   dependency (`compute`, `selection`, `tokenizer`, `model_io`, `runtime`,
   `invocation`) so the alias, not the crate's real package name, is what
   integration tests and downstream `src/` code import by.

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
   `mference-check`) is the process entry point that reads `argv`, calls
   `mrefrust-invocation::parse`, and applies its pure exit-status/stream-routing
   decisions. This resolves what an earlier note here called the reserved,
   not-yet-created `mrefrust-entrypoint` name; that name is not used. On macOS
   it also attempts real generation against `--model` via `RealForwardRunner`,
   in all three modes (`--prompt`, `--messages-file`, `--chat`) (see
   `DEVIATIONS.md`).

8. `crates/gpu` is the one crate with a hard platform gate: everything in
   `src/` is `#[cfg(target_os = "macos")]`, so `cargo build --workspace` /
   `cargo test --workspace` succeed on Linux with the crate compiling to
   (effectively) nothing. Dispatched, parity-tested Metal pipelines
   (`rmsnorm_no_scale`, `rms_norm_bf16w`, both `_perhead` norm variants,
   `rope_proportional_neox` (which with `rotated_pairs = head_dim/2` IS
   default full-head NeoX -- no separate default-rope wrapper exists),
   the port-local `logit_softcap_fp16`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`
   (both with offset-bound resident variants), the port-local
   `dequant_q8_0_gemv_simd` and `dequant_q4_k_gemv_simd` (GGUF Q8_0 and
   Q4_K, each also with a resident variant), `embed_lookup_q8_0`, and
   `moe_gguf.metal`'s Q8_0 routed-expert decode pair
   (`moe_phase1_gate_up_act_q8_0` + `moe_phase2_down_reduce_k8_q8_0`),
   all port-local because Swift has no GGUF intake, `router_gemv_gemma4_r4`,
   two-pass split-KV `attention_decode` (multi-chunk, split up to 16 ways by
   `chunks_for`), `moe_decode` decode pair, all eight of `gdn.metal`'s
   gated-DeltaNet kernels, and
   `utility` elementwise kernels including the port-local `scalar_mul_fp16`
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

9. `crates/model-io` and `crates/streaming` are the two crates that
   intentionally carry unsafe code and platform `cfg`s (mmap in
   `model-io::resident_buffer`, the macOS `F_RDADVISE` `fcntl` in
   `streaming::rdadvice`, and the raw destination pointers plus borrowed
   `RawFd` that `streaming::read_pool`'s parked worker threads use --
   sound only because `run_batch` blocks until every claim is dropped). `compute`, `repack`, `runtime`, and `tokenizer`
   have `#![forbid(unsafe_code)]`. `core`, `gpu`, `invocation`, `selection`,
   `server`, `window-fit`, `cli`, and `bench` currently have no such
   attribute and no workspace-level lint enforces it, so unsafe code is not
   actually compiler-blocked there today, even though none uses any.

10. `crates/runtime`'s `LogitProducer` trait has `RealForwardRunner` (macOS/GPU
    only) as its real GPU-forward-pass-backed implementation, while
    `ScriptedLogitProducer` (a fixed replayed logit sequence) is what unit
    tests and `crates/server`'s `ScriptedChatModel` drive the raw-completion
    loop with on any platform. `crates/server`'s `RealChatModel` drives it
    with `RealForwardRunner` on macOS (`mference-server --model`).
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
    tell them apart): `Qwen36` builds `RealQwenState` and runs
    `real_forward_qwen.rs` + `real_forward_qwen_attn.rs` (gated DeltaNet
    on mask-2 layers, gated full attention on mask-1, one post-attention
    norm feeding router + shared + routed, no sandwich norms, no softcap);
    `DeepseekV4Flash` is refused. Within `Gemma4`, naming picks: synthetic
    short names (`layer0.q_proj`) get
    the plain no-scale flow in `real_forward.rs`; verbatim
    real-checkpoint names (`language_model.model.layers.0...`, what
    `mrefrust_repack::write_gemma4_install` writes) get the full Gemma 4
    learned-weight flow in `real_forward_gemma4.rs` (BF16 norms, per-head
    q/k/v norms, INT8 router + effective scale, kernel-semantics top-k,
    INT8 shared-expert branch, sandwich tail, `layer_scalar`). Real
    Gemma 4 26B-A4B is PROVEN end to end: the pinned checkpoint repacks
    through the streamed pipeline and generates coherent chat-formatted
    answers via `mference-check` (see DEVIATIONS.md -- raw prompts babble,
    the IT model needs its `<|turn>` markup, which `--messages-file` and
    `--chat` now render for you). `RealForwardRunner::phase_counters`
    accumulates per-phase decode timings (GPU wait, router readback,
    expert `pread`, routed bind) plus expert-cache hit counts and
    per-command-buffer GPU busy attribution (`GPUStartTime`/`GPUEndTime`,
    a separate axis from the wall-clock buckets: cb1 = attention+router,
    routed cb, final head; shared/hit buffers stay unattributed); run
    `mference-check` with `MFERENCE_PHASES=1` to print the breakdown.
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
    the run-to-run spread is wider than the effect). `MFERENCE_HIT_CB=0`
    is the sibling seam for the same trick applied to the routed experts:
    each layer's slots are ordered cache MISSES first, then hits, so the
    already-resident hits' phase-1 GEMV can be dispatched on its own
    command buffer (with its own `RoutedBlobsBuffer` -- rebinding the
    shared one would race that dispatch) before the `pread`, leaving the
    misses to the main pass with an `acts` offset of zero. Weight and
    blob for a slot are carried in one list so the permutation cannot
    drift apart. It buys less than it looks: it already hides ALL the
    phase-1 work the hits can offer, but that is only ~1.15 ms/token of
    GPU wait against ~0.76 ms/token of extra host bind+commit, ~+1%
    net. `MFERENCE_ROUTED_PIPELINE=0` is the third seam: by default a
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
    (`--expert-cache-slots`, allowed 8/16/24/32, default 16, ~3.2 MB of
    pinned host memory per slot per layer on the 26B); it was hardcoded
    to 16 before, so measurements taken with the flag set are only
    meaningful from that change on. Qwen 3.6 runs on a
    SYNTHETIC install (`build_synthetic_qwen36_real_install`); no real
    checkpoint has been repacked. DeepSeek-V4-Flash remains blocked on
    DSV4. Build a test/demo install with
    `mrefrust_repack::build_synthetic_gemma4_install` (dense) or its
    `_swa`/`_moe`/`_moe_streamed` variants,
    `build_synthetic_gemma4_real_install` (real naming, exercises the
    real checkpoint repack pipeline), or
    `build_synthetic_qwen36_real_install` (the Qwen sibling), instead of
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
    directly.

14. Adding one flag to `crates/invocation` touches five places, two of them
    non-obvious: the `OPTIONS` table, BOTH parser dispatch `match`es (each
    ends in `unreachable!`, so a missing arm is a runtime panic, not a
    compile error), the `InvocationRequest` literal, and
    `tests/usage_and_status.rs`'s hardcoded option-count assertion.

15. `real_forward_gemma4.rs`'s per-token function interleaves
    `let real = self.real.as_ref()` bindings with `&mut self` calls. A new
    `&mut self` call between such a binding and its last use is E0502;
    re-bind `real` after the call (the file already does this repeatedly).

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
    `gemma4_checkpoint.rs` owns the mapping;
    `crates/repack/tests/synthetic_qwen.rs` pins both families' markers.

27. **Whether generated bytes survive a change of `--expert-cache-slots`
    is a property of the FLOW, not of the engine.**
    `real_forward_gemma4.rs` orders a layer's routed slots misses first
    then hits, so the already-resident hits' phase-1 GEMV can ride its own
    command buffer. That order is what `moe_phase2_down_reduce_k8` reduces
    in, and FP addition is not associative, so changing the hit/miss split
    changes Gemma's output bytes: measured, 8 slots and 16 slots produce
    different (individually deterministic) text. `real_forward_qwen.rs`
    does no such reordering and IS byte-identical across slot counts.
    Three consequences. Any A/B that must hold output constant compares
    within ONE slot count. A golden digest needs a WARM cache, since a
    cold one has a different miss set for the same reason (this is why
    `quality_common`'s run order is frozen). And `MFERENCE_HIT_CB=0` is
    not the lever it looks like: it toggles the separate command buffer,
    not the ordering, and measured directly it changes no digest at all.
    `crates/bench/tests/quality_gate.rs` freezes a second digest for the
    constrained arm rather than asserting the two equal, which on the
    Gemma flow would be asserting FP associativity.

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
    sample. THIS IS A BATTERY PHENOMENON on this machine, and the
    contrast is sharp: the protocol's `long-synthesis` case left Nominal
    on every run of both installs on battery (it prefills ~3,000 tokens
    for 63-72 s before decoding anything), while the SAME binary running
    the SAME protocol on AC held Nominal on 50 of 50 sampled arms.
    `scripts/power.sh` flags any run whose pressure leaves Nominal and
    excludes it; `scripts/parity.sh` and the oracles do NOT, so a
    surprising throughput row from a long battery session is worth
    checking against `pmset -g therm` before it is believed. The
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
    reaches an install any more: `gguf_checkpoint.rs::transcode_f32`
    narrows to BF16 by default and INT8s the router. The INT8 set is keyed
    by CANONICAL NAME per family, not by rank -- Qwen's
    `linear_attn.conv1d.weight` is rank-2 F32 that the runtime reads as
    BF16, while `mlp.gate.weight` is rank-2 and must be dtype 5 -- and the
    BF16 default is safe because a mis-targeted tensor fails loudly at
    `open()` rather than quietly.
    **WHETHER A GGUF INSTALL LOADS IS DECIDED PER BLOCK TYPE, NOT PER
    FORMAT** (Stage 2, 2026-08-08). Q8_0 runs: it has a resident GEMV, an
    embedding lookup, and a routed-expert decode pair, all parity-tested.
    Q4_K, Q6_K and Q4_0 do not, and are refused. The two gates are
    independent on purpose, and each reads a different thing.
    `model_io::validate_quant` reads the manifest's `ggmlType` against
    `model_io::EXECUTABLE_GGUF_TYPES`; `RealForwardRunner::open` reads the
    resident index's dtype TAGS against its own copy of that set, which is
    the backstop for a hand-edited manifest and believes the bytes rather
    than the claim. Both directions are asserted
    (`crates/runtime/tests/gguf_install_refused.rs`, which also decodes),
    so widening the set means landing kernels, not editing a list: the two
    lists have to move together or one of those tests reddens.
    A Q4_K install therefore still installs and still refuses -- it has a
    CPU reference and a resident GEMV, but no MoE or embedding kernel, and
    routed experts are where a GGUF's bytes actually are.

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

## Per-Crate Documentation

When working on code inside a specific crate, refer to that crate's `CLAUDE.md` file for crate-specific architecture, key modules, dev commands, and localized gotchas:

- [`crates/bench/CLAUDE.md`](crates/bench/CLAUDE.md): Throughput benchmark harness, mach memory sampler, frozen protocol, memory oracle test rules.
- [`crates/cli/CLAUDE.md`](crates/cli/CLAUDE.md): CLI binary (`mference-check`), process entry point, real model smoke tests, interactive chat REPL.
- [`crates/compute/CLAUDE.md`](crates/compute/CLAUDE.md): CPU reference kernels (RmsNorm, RoPE, Attention, Quant), numerical ground truth for GPU tests.
- [`crates/core/CLAUDE.md`](crates/core/CLAUDE.md): Shared primitives (`TokenId`, `LogitValue`), runtime configuration, allowed sets, chunk sizing.
- [`crates/gpu/CLAUDE.md`](crates/gpu/CLAUDE.md): macOS Metal context, pipeline caches, MSL shaders, KV cache, zero-copy weights, profiling flags.
- [`crates/invocation/CLAUDE.md`](crates/invocation/CLAUDE.md): Pure CLI argument parser, `InvocationRequest`, 5-place rule for adding new flags.
- [`crates/model-io/CLAUDE.md`](crates/model-io/CLAUDE.md): Manifest validation, architecture baselines, packed expert layout, mmap resident weight index.
- [`crates/repack/CLAUDE.md`](crates/repack/CLAUDE.md): Safetensors header parsing, ranged HTTP downloads, `.gturbo` writer, synthetic model builders.
- [`crates/runtime/CLAUDE.md`](crates/runtime/CLAUDE.md): Raw completion generation loop, `LogitProducer` contract, `RealForwardRunner` decode engine.
- [`crates/selection/CLAUDE.md`](crates/selection/CLAUDE.md): Candidate token selection, temperature/top-k/top-p shaping, repetition penalty, logits contract.
- [`crates/server/CLAUDE.md`](crates/server/CLAUDE.md): the `mference-server` HTTP server (OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, `/v1/models`), Axum handlers, SSE streaming, `anyllm_translate` wire types.
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
+-- Makefile           # build, test, fmt, clippy wrapper targets
+-- README.md          # repository overview and quickstart
+-- ROADMAP.md         # forward roadmap + descope record (gitignored)
+-- rust-toolchain.toml # toolchain pin (stable Rust 1.82+)
+-- crates
|   +-- bench          # mference-bench binary & harness (throughput benchmark)
|   +-- cli            # mference-check binary (process entry point & CLI runner)
|   +-- compute        # CPU reference kernels & compute strategy marker
|   +-- core           # shared primitives (TokenId, LogitValue), RuntimeConfig, chunking
|   +-- gpu            # Metal pipeline cache & GPU kernel dispatches (macOS only)
|   +-- invocation     # CLI argument parsing, request assembly & exit status routing
|   +-- model-io       # manifest validation, packed-expert layout, resident index & mmap
|   +-- repack         # safetensors + GGUF header parsing, ranged downloads, int4/8 repack, gturbo writer
|   +-- runtime        # raw-completion prefill+decode loop & RealForwardRunner (macOS)
|   +-- selection      # token sampling (temperature, top-k, top-p, repetition penalty, choose)
|   +-- server         # OpenAI-compatible Chat Completions HTTP server (axum)
|   +-- streaming      # pread-based expert streamer, LFU/LRU slot cache & read pool
|   +-- tokenizer      # tokenizer wrapper, chat templates (text/Jinja), stop matcher, DSL parser
|   \-- window-fit     # deterministic conversation-window fitting & turn dropping
+-- scripts
|   +-- kld.py         # cross-engine KL vs mlx-lm (reads tests/logit_dump.rs's output)
|   +-- parity.sh      # head-to-head protocol run against the Swift MferenceCLI
|   +-- phasediff.sh   # bucket-level decode phase diff against the Swift engine
|   \-- power.sh       # watts & joules-per-token over the protocol (needs sudo)
\-- docs
    +-- BENCHMARKING.md# benchmark modes, mach memory sampling & memory oracle details
    +-- POWER_BASELINE.md # watts, joules-per-token, hygiene audit (ROADMAP Phase P1)
    \-- TESTING.md     # test suite organization, platform gating & testing rules
```

`scripts/` holds the measurement surfaces that cannot be a `cargo test`:
two need the Swift engine built next door, and `kld.py` needs a 14.6 GB
reference checkpoint plus a Python environment. `kld.py` runs mlx-lm under
`uv run --with mlx-lm`, an ephemeral env, so no Python dependency is
installed globally or enters this workspace.

- `crates/core`: shared primitives (`TokenId`, `LogitValue`, `LogitsView`), error types (`CoreError`), runtime configuration (`RuntimeConfig`, `RuntimeConfigBuilder`), allowed value sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`), automatic chunk-size resolution (`chunk_sizing.rs`), and prefill chunking primitives (`prefill.rs`). Details in [`crates/core/CLAUDE.md`](crates/core/CLAUDE.md).
- `crates/compute`: CPU reference kernels (RmsNorm, WHT, RoPE incl. Qwen's `rope_neox_subdim`, causal attention, int4/int8 affine quant + GEMV, the GGUF block-quant reference (`quant_gguf.rs`: Q8_0 and Q4_K dequant/quant/GEMV plus `pearson`), embedding lookup, MoE FFN, the gated-DeltaNet chain (`gdn.rs`) and Qwen's gating kernels (`gating.rs`), logit softcap-softmax, RelError/tolerance table, sampling helpers) plus destination compute strategy marker type (`ComputeStrategy`). These are the numerical ground truth `crates/gpu`'s Metal kernels are validated against. Details in [`crates/compute/CLAUDE.md`](crates/compute/CLAUDE.md).
- `crates/invocation`: pure translation of command-line argument tokens into a validated invocation request (`InvocationRequest`), options definition (`OPTIONS`), diagnostics (`diagnostics.rs`), typed failures (`InvocationFailure`), usage rendering (`render_usage`), and pure outcome-to-exit-status and outcome-to-stream routing decisions. Performs no filesystem, environment, or process I/O. Details in [`crates/invocation/CLAUDE.md`](crates/invocation/CLAUDE.md).
- `crates/selection`: candidate selection (`select`, `select_from_logits`) from a per-candidate score vector under a validated shaping configuration (temperature, top-k, top-p, repetition penalty, seed), accumulated history, step position, determinism, and distribution guards. Numeric parity with any upstream implementation is out of scope; only the observable contract is exercised. Details in [`crates/selection/CLAUDE.md`](crates/selection/CLAUDE.md).
- `crates/window-fit`: pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation (`FitOutcome`, `DroppedTurn`), using a caller-supplied whole-conversation length measurement, until the measured length is under a caller-supplied bound or nothing eligible remains. An optional leading instruction turn and the newest turn are never removed. Performs no input or output and holds no state between calls. Details in [`crates/window-fit/CLAUDE.md`](crates/window-fit/CLAUDE.md).
- `crates/tokenizer`: wraps HF `tokenizers` crate (`MfTokenizer`); resolves Gemma 4 / ChatML (Qwen) / DeepSeek-V4 chat dialect from special tokens; renders text-only chat templates plus DeepSeek's native tool chat; generic Jinja-templated tool chat for Gemma/ChatML (`minijinja` + `pycompat`, rendering `chat_template.jinja`); streaming detokenizer (`StreamingDetokenizer`) and stop matcher (`StopMatcher`) (stop set unions dialect stops with `generation_config.json` `eos_token_id` list); Gemma/Qwen/DeepSeek tool-call DSL parsers and streaming structured assistant-output decoder (`StructuredDecoder`). Details in [`crates/tokenizer/CLAUDE.md`](crates/tokenizer/CLAUDE.md).
- `crates/model-io`: `manifest.json` decode and field-by-field validation against a resolved `ArchConfig` (with canonical Gemma 4, Qwen 3.6, and DeepSeek-V4-Flash baselines), `packed_experts/layout.json` decode (`PackedExpertsLayout`), `model_weights.bin` resident tensor index reader (`ResidentIndex`), `mmap`'d resident-buffer view (`ResidentBuffer`), streaming SHA-256 verification (`sha256.rs`), and trusted install receipt (`InstallReceipt`). Allowed a narrow amount of `unsafe` (the `mmap` call). Details in [`crates/model-io/CLAUDE.md`](crates/model-io/CLAUDE.md).
- `crates/streaming`: routed-expert `pread` streamer (`PreadExpertStreamer`) with a fixed per-layer slot cache. The LFU/LRU eviction policy (`ExpertCache`) is pure logic, separated from file I/O so it can be tested against access traces without a model install. Cache misses are split into chunks and read on `read_pool`, a process-wide set of parked worker threads, so a layer that misses once still reads at full width (the `pread` is a page-cache memcpy, not disk I/O). `rdadvice` and `read_pool` are the other `unsafe`-carrying modules (macOS `F_RDADVISE`, a documented no-op elsewhere; raw destination pointers across worker threads). Details in [`crates/streaming/CLAUDE.md`](crates/streaming/CLAUDE.md).
- `crates/gpu`: Metal device/pipeline-cache context (`MetalContext`, `PassEncoder`, `CommittedPass`) and per-kernel dispatch. macOS-only; compiles to nothing elsewhere. Dispatched, parity-tested kernels (`rmsnorm_no_scale`, `rms_norm_bf16w`, both `_perhead` norm variants, `rope_proportional_neox`, `rope_neox_subdim`, `logit_softcap_softmax`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd` with resident variants, the port-local GGUF set (`dequant_q8_0_gemv_simd`, `dequant_q4_k_gemv_simd`, `embed_lookup_q8_0`, and `moe_gguf.metal`'s Q8_0 decode pair -- ROADMAP Phase G), `router_gemv_gemma4_r4`, two-pass split-KV `attention_decode` (multi-chunk, split up to 16 ways by `chunks_for`), `moe_decode` decode pair, `gdn.metal`'s eight gated-DeltaNet kernels, and `utility` elementwise kernels incl. Qwen's three gating kernels) are compiled from MSL source at runtime, vendored from Swift except where marked port-local. `KvCacheManager` allocates and manages real per-layer Metal KV buffers used by `RealForwardRunner`. `ResidentGpuWeights` wraps resident mmap in zero-copy MTLBuffer. `GdnStateManager` is the Qwen flow's recurrent state; `Dsv4StateManager` allocates real per-layer Metal buffers (unwired kernels); `PrefillChunkScratchLayout`/`PrefillChunkScratchBuffers` size scratch buffers (undispatched tile kernel). The `sample` kernel and fused lm_head are not yet vendored or dispatched. Details in [`crates/gpu/CLAUDE.md`](crates/gpu/CLAUDE.md).
- `crates/runtime`: raw-completion prefill+decode loop (`run_raw_completion`, `run_raw_completion_chunked`), wiring a `LogitProducer`, the tokenizer's streaming detokenizer and stop matcher, and `selection::select` into one token generation loop. `ScriptedLogitProducer` is what unit tests and `crates/server`'s `ScriptedChatModel` drive the loop with (see Gotcha 10). `RealForwardRunner` (macOS/GPU only, `src/real_forward.rs`, `src/real_forward_gemma4.rs`, and `src/real_forward_qwen{,_attn}.rs`) is a real `LogitProducer`: a genuine transformer forward pass through real GPU kernels (including real GPU decode attention) and real quantized weights, supporting dense and MoE FFN layers. Dense bridges gated FFN on CPU via `mrefrust_compute::run_ffn`; MoE runs real GPU router GEMV plus real GPU GEMVs for each selected expert, host-side top-k selection, and CPU-bridged gated activation. Supports synthetic short names, verbatim real Gemma 4 checkpoint names (learned-weight flow), and the Qwen 3.6 hybrid linear/full-attention flow. See Gotcha 12. Details in [`crates/runtime/CLAUDE.md`](crates/runtime/CLAUDE.md).
- `crates/cli`: the `mference-check` binary process entry point (see Gotcha 7). Parses `argv`, applies `invocation`'s exit-status and stream-routing decisions, prints the resolved request for a validated invocation, and (macOS, `src/generate.rs`) attempts real generation against `--model` via `RealForwardRunner` (see Gotcha 12) in all three modes: `--prompt` (raw text), `--messages-file` (rendered through chat template), and `--chat` (interactive REPL in `src/chat.rs`, trimming turns with `mrefrust-window-fit`). Details in [`crates/cli/CLAUDE.md`](crates/cli/CLAUDE.md).
- `crates/repack`: safetensors header parsing (pure, tested against synthetic fixtures), `RangeSource` trait for ranged reads (HTTP-backed for real installs, in-memory for tests) with two-step header-fetch plan, per-row int4/int8 quantization repack (reusing `mrefrust_compute`'s quantizer), byte-exact `.gturbo` directory assembly (`write_gturbo_install`), real named resident-tensor index writer (`write_gturbo_install_with_resident_index`), synthetic install builders (`synthetic_model.rs`, `synthetic_real.rs`, `synthetic_qwen.rs`), Hugging Face Llama checkpoint repacker (`hf_checkpoint.rs`), Gemma 4 mlx-community checkpoint repacker & streamed pipeline (`gemma4_checkpoint.rs`, family-parameterized so Qwen 3.6 goes through the same walk), Qwen 3.6 `config.json` parser (`qwen36_config.rs`, the one family-specific piece of that walk), install verifier (`install_verifier.rs`), manifest peeker (`manifest_peek.rs`), and the GGUF intake (`gguf_header.rs` parser, `gguf_names.rs` name mapping, `gguf_config.rs` metadata-to-`ArchConfig`, `gguf_checkpoint.rs` repack walk (expert bytes verbatim, resident F32 core transcoded to BF16/INT8), `synthetic_gguf.rs` fixture writer -- ROADMAP Phase G; a Q8_0 install is executable, other block types install and are refused, see Gotcha 29). Details in [`crates/repack/CLAUDE.md`](crates/repack/CLAUDE.md).
- `crates/server`: HTTP server on loopback (`mference-server` binary, axum framework) serving OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and `/v1/models`, both generation endpoints supporting full-response (non-streaming) and SSE-streaming responses. The wire types come from `anyllm_translate` (crates.io, default features: pure and IO-free), which also translates an Anthropic request into the OpenAI request the existing path understands and translates the result back, so Anthropic-native clients need no proxy. Tool calling is wired on both endpoints (request `tools` render through the checkpoint's `chat_template.jinja`, generated calls come back through `StructuredAssistantDecoder`); images and `thinking` are dropped, some of it reported on an `x-anyllm-degradation` header. Two backends behind the `ChatModel` trait: `RealChatModel` (macOS, `--model <install-dir>`, one mutex-serialized `RealForwardRunner` per process) and `ScriptedChatModel` (portable, canned completions, what the integration tests drive). Details in [`crates/server/CLAUDE.md`](crates/server/CLAUDE.md).
- `crates/bench`: the `mference-bench` binary plus benchmark library (`mrefrust_bench`). The scripted default (three fixed prompts, fixed seed, discarded warmup) measures loop overhead via `ScriptedLogitProducer`. `--model <install-dir>` (macOS) is the real Swift-comparison mode: frozen community protocol (`protocol.rs`) driven through `RealForwardRunner`, reporting split prefill/decode tok/s and peak `phys_footprint` from the mach sampler (`memory.rs`). `tests/memory_oracle.rs` (`#[ignore]`d, gated on `MREFRUST_GEMMA4_INSTALL_DIR`) asserts peak footprint against per-chip baseline rows, plus a steady-state replay guard; each row carries a `source` recording whether it is a Swift parity number or this port's own measurement. The quality axis lives here too, all `#[ignore]`d: `tests/quality_gate.rs` and its Qwen sibling (per-install perplexity plus golden digests), `tests/quality_sensitivity.rs` (proof the perplexity responds to quantization damage), and `tests/logit_dump.rs` (full-vocab logits plus the exact token ids, feeding `scripts/kld.py`'s cross-engine KL against mlx-lm -- the one external reference in the whole quality section). Full details in [`crates/bench/CLAUDE.md`](crates/bench/CLAUDE.md) and `docs/BENCHMARKING.md`.
- `docs/`: repository documentation directory. `docs/BENCHMARKING.md` details benchmark harness modes, mach memory sampling, and the memory oracle baseline assertions; `docs/POWER_BASELINE.md` records watts and joules-per-token per install plus the power-hygiene audit (ROADMAP Phase P1), and is the one page here measured on BATTERY rather than AC; `docs/TESTING.md` documents test suite organization, macOS and environment-variable gating conventions, and test writing rules.

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
smoke" above, all three of them:

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

See `DEVIATIONS.md` for the full list of what this port scaffolds versus
fully implements, `ROADMAP.md` for the forward roadmap and descope
record, and
`docs/NEW_MODEL.md` for the end-to-end checklist for wiring a new model
family (what to map, what to specialize, what to measure, in order).
