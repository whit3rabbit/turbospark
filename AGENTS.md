# AGENTS.md

CLAUDE.md is a symlink to this file.

Conventions, gotchas, and commands for working in this Rust workspace,
a behavior-compatible port of the Mference Swift inference engine (see
`ROADMAP.md` for phase-by-phase scope and `DEVIATIONS.md` for what is
scaffolded rather than fully wired). Keep all code, comments, and docs
ASCII: no emojis and no em dashes (project rule).

`docs/TESTING.md` covers what the suite proves and how tests are gated
(macOS, `#[ignore]`d, env-var). `docs/BENCHMARKING.md` covers the three
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

# Run the OpenAI-compatible server (scripted responses; see DEVIATIONS.md).
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

# The other two #[ignore]d tests: real checkpoint downloads (many GB).
cargo test -p mrefrust-repack --test gemma4_checkpoint_network --release -- --ignored --nocapture
cargo test -p mrefrust-repack --test hf_checkpoint_network --release -- --ignored --nocapture
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
   (both with offset-bound resident variants), `router_gemv_gemma4_r4`,
   two-pass split-KV `attention_decode`, `moe_decode` decode pair, and
   `utility` elementwise kernels including the port-local `scalar_mul_fp16`)
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
   `GdnStateManager` (`gdn_state.rs`) and `Dsv4StateManager` (`dsv4_state.rs`)
   allocate real per-layer Metal buffers but stay unwired (their compute
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
   `streaming::rdadvice`). `compute`, `repack`, `runtime`, and `tokenizer`
   have `#![forbid(unsafe_code)]`. `core`, `gpu`, `invocation`, `selection`,
   `server`, `window-fit`, `cli`, and `bench` currently have no such
   attribute and no workspace-level lint enforces it, so unsafe code is not
   actually compiler-blocked there today, even though none uses any.

10. `crates/runtime`'s `LogitProducer` trait has `RealForwardRunner` (macOS/GPU
    only) as its real GPU-forward-pass-backed implementation, while
    `ScriptedLogitProducer` (a fixed replayed logit sequence) is what unit
    tests and `crates/server`'s `ScriptedChatModel` drive the raw-completion
    loop with (since no trained `.gturbo` weights exist to validate against).
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
    experts) and both full-attention (mask 1) and sliding-window (mask 0)
    layers; `open()` rejects only linear (2) and compressed (3/4) layers,
    whose kernels are unported. It has TWO decode flows, selected by the
    resident index's naming: synthetic short names (`layer0.q_proj`) get
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
    expert `pread`, routed bind) plus expert-cache hit counts; run
    `mference-check` with `MFERENCE_PHASES=1` to print the breakdown.
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
    net.
    Slot count comes from `open_with_options`
    (`--expert-cache-slots`, allowed 8/16/24/32, default 16, ~3.2 MB of
    pinned host memory per slot per layer on the 26B); it was hardcoded
    to 16 before, so measurements taken with the flag set are only
    meaningful from that change on. Qwen 3.6 and
    DeepSeek-V4-Flash remain blocked on GDN/DSV4. Build a test/demo
    install with
    `mrefrust_repack::build_synthetic_gemma4_install` (dense) or its
    `_swa`/`_moe`/`_moe_streamed` variants, or
    `build_synthetic_gemma4_real_install` (real naming, exercises the
    real checkpoint repack pipeline), instead of hand-writing an
    `ArchConfig`; their non-shape fields are pinned to match
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
+-- ROADMAP.md         # phase-by-phase scope tracking (gitignored)
+-- rust-toolchain.toml # toolchain pin (stable Rust 1.82+)
+-- crates
|   +-- bench          # mference-bench binary & harness (throughput benchmark)
|   +-- cli            # mference-check binary (process entry point & CLI runner)
|   +-- compute        # CPU reference kernels & compute strategy marker
|   +-- core           # shared primitives (TokenId, LogitValue), RuntimeConfig, chunking
|   +-- gpu            # Metal pipeline cache & GPU kernel dispatches (macOS only)
|   +-- invocation     # CLI argument parsing, request assembly & exit status routing
|   +-- model-io       # manifest validation, packed-expert layout, resident index & mmap
|   +-- repack         # safetensors header parsing, ranged downloads, int4/8 repack, gturbo writer
|   +-- runtime        # raw-completion prefill+decode loop & RealForwardRunner (macOS)
|   +-- selection      # token sampling (temperature, top-k, top-p, repetition penalty, choose)
|   +-- server         # OpenAI-compatible Chat Completions HTTP server (axum)
|   +-- streaming      # pread-based expert streamer & LFU/LRU slot cache
|   +-- tokenizer      # tokenizer wrapper, chat templates (text/Jinja), stop matcher, DSL parser
|   \-- window-fit     # deterministic conversation-window fitting & turn dropping
\-- docs
    +-- BENCHMARKING.md# benchmark modes, mach memory sampling & memory oracle details
    \-- TESTING.md     # test suite organization, platform gating & testing rules
```

- `crates/core`: shared primitives (`TokenId`, `LogitValue`, `LogitsView`), error types (`CoreError`), runtime configuration (`RuntimeConfig`, `RuntimeConfigBuilder`), allowed value sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`), automatic chunk-size resolution (`chunk_sizing.rs`), and prefill chunking primitives (`prefill.rs`).
- `crates/compute`: CPU reference kernels (RmsNorm, WHT, RoPE, causal attention, int4/int8 affine quant + GEMV, embedding lookup, MoE FFN, logit softcap-softmax, RelError/tolerance table, sampling helpers) plus destination compute strategy marker type (`ComputeStrategy`). These are the numerical ground truth `crates/gpu`'s Metal kernels are validated against.
- `crates/invocation`: pure translation of command-line argument tokens into a validated invocation request (`InvocationRequest`), options definition (`OPTIONS`), diagnostics (`diagnostics.rs`), typed failures (`InvocationFailure`), usage rendering (`render_usage`), and pure outcome-to-exit-status and outcome-to-stream routing decisions. Performs no filesystem, environment, or process I/O.
- `crates/selection`: candidate selection (`select`, `select_from_logits`) from a per-candidate score vector under a validated shaping configuration (temperature, top-k, top-p, repetition penalty, seed), accumulated history, step position, determinism, and distribution guards. Numeric parity with any upstream implementation is out of scope; only the observable contract is exercised.
- `crates/window-fit`: pure, deterministic conversation-window fitting (`fit_conversation_window`). Drops the oldest eligible turns from a conversation (`FitOutcome`, `DroppedTurn`), using a caller-supplied whole-conversation length measurement, until the measured length is under a caller-supplied bound or nothing eligible remains. An optional leading instruction turn and the newest turn are never removed. Performs no input or output and holds no state between calls.
- `crates/tokenizer`: wraps HF `tokenizers` crate (`MfTokenizer`); resolves Gemma 4 / ChatML (Qwen) / DeepSeek-V4 chat dialect from special tokens; renders text-only chat templates plus DeepSeek's native tool chat; generic Jinja-templated tool chat for Gemma/ChatML (`minijinja` + `pycompat`, rendering `chat_template.jinja`); streaming detokenizer (`StreamingDetokenizer`) and stop matcher (`StopMatcher`) (stop set unions dialect stops with `generation_config.json` `eos_token_id` list); Gemma/Qwen/DeepSeek tool-call DSL parsers and streaming structured assistant-output decoder (`StructuredDecoder`).
- `crates/model-io`: `manifest.json` decode and field-by-field validation against a resolved `ArchConfig` (with canonical Gemma 4, Qwen 3.6, and DeepSeek-V4-Flash baselines), `packed_experts/layout.json` decode (`PackedExpertsLayout`), `model_weights.bin` resident tensor index reader (`ResidentIndex`), `mmap`'d resident-buffer view (`ResidentBuffer`), streaming SHA-256 verification (`sha256.rs`), and trusted install receipt (`InstallReceipt`). Allowed a narrow amount of `unsafe` (the `mmap` call).
- `crates/streaming`: routed-expert `pread` streamer (`PreadExpertStreamer`) with a fixed per-layer slot cache. The LFU/LRU eviction policy (`ExpertCache`) is pure logic, separated from file I/O so it can be tested against access traces without a model install. `rdadvice` is the other `unsafe`-carrying module (macOS `F_RDADVISE`; a documented no-op elsewhere).
- `crates/gpu`: Metal device/pipeline-cache context (`MetalContext`, `PassEncoder`, `CommittedPass`) and per-kernel dispatch. macOS-only; compiles to nothing elsewhere. Dispatched, parity-tested kernels (`rmsnorm_no_scale`, `rms_norm_bf16w`, both `_perhead` norm variants, `rope_proportional_neox`, `logit_softcap_softmax`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd` with resident variants, `router_gemv_gemma4_r4`, two-pass split-KV `attention_decode`, `moe_decode` decode pair, and `utility` elementwise kernels) are compiled from vendored MSL source at runtime. `KvCacheManager` allocates and manages real per-layer Metal KV buffers used by `RealForwardRunner`. `ResidentGpuWeights` wraps resident mmap in zero-copy MTLBuffer. `GdnStateManager` and `Dsv4StateManager` allocate real per-layer Metal buffers (unwired kernels); `PrefillChunkScratchLayout`/`PrefillChunkScratchBuffers` size scratch buffers (undispatched tile kernel). The `sample` kernel and fused lm_head are not yet vendored or dispatched.
- `crates/runtime`: raw-completion prefill+decode loop (`run_raw_completion`, `run_raw_completion_chunked`), wiring a `LogitProducer`, the tokenizer's streaming detokenizer and stop matcher, and `selection::select` into one token generation loop. `ScriptedLogitProducer` is what unit tests and `crates/server`'s `ScriptedChatModel` drive the loop with (see Gotcha 10). `RealForwardRunner` (macOS/GPU only, `src/real_forward.rs` and `src/real_forward_gemma4.rs`) is a real `LogitProducer`: a genuine transformer forward pass through real GPU kernels (including real GPU decode attention) and real quantized weights, supporting dense and MoE FFN layers. Dense bridges gated FFN on CPU via `mrefrust_compute::run_ffn`; MoE runs real GPU router GEMV plus real GPU GEMVs for each selected expert, host-side top-k selection, and CPU-bridged gated activation. Supports synthetic short names and verbatim real Gemma 4 checkpoint names (learned-weight flow). See Gotcha 12.
- `crates/cli`: the `mference-check` binary process entry point (see Gotcha 7). Parses `argv`, applies `invocation`'s exit-status and stream-routing decisions, prints the resolved request for a validated invocation, and (macOS, `src/generate.rs`) attempts real generation against `--model` via `RealForwardRunner` (see Gotcha 12) in all three modes: `--prompt` (raw text), `--messages-file` (rendered through chat template), and `--chat` (interactive REPL in `src/chat.rs`, trimming turns with `mrefrust-window-fit`).
- `crates/repack`: safetensors header parsing (pure, tested against synthetic fixtures), `RangeSource` trait for ranged reads (HTTP-backed for real installs, in-memory for tests) with two-step header-fetch plan, per-row int4/int8 quantization repack (reusing `mrefrust_compute`'s quantizer), byte-exact `.gturbo` directory assembly (`write_gturbo_install`), real named resident-tensor index writer (`write_gturbo_install_with_resident_index`), synthetic install builders (`synthetic_model.rs`, `synthetic_real.rs`), Hugging Face Llama checkpoint repacker (`hf_checkpoint.rs`), Gemma 4 mlx-community checkpoint repacker & streamed pipeline (`gemma4_checkpoint.rs`), install verifier (`install_verifier.rs`), and manifest peeker (`manifest_peek.rs`).
- `crates/server`: OpenAI-compatible `/v1/chat/completions` HTTP server on loopback (`mference-server` binary, axum framework), supporting both full-response (non-streaming) and SSE-streaming responses, powered by `ScriptedChatModel` wired to `mrefrust-runtime`. `ScriptedChatModel` is the only backend (see Gotcha 10); real weights are future work.
- `crates/bench`: the `mference-bench` binary plus benchmark library (`mrefrust_bench`). The scripted default (three fixed prompts, fixed seed, discarded warmup) measures loop overhead via `ScriptedLogitProducer`. `--model <install-dir>` (macOS) is the real Swift-comparison mode: frozen community protocol (`protocol.rs`) driven through `RealForwardRunner`, reporting split prefill/decode tok/s and peak `phys_footprint` from the mach sampler (`memory.rs`). `tests/memory_oracle.rs` (`#[ignore]`d, gated on `MREFRUST_GEMMA4_INSTALL_DIR`) asserts peak footprint against per-chip baseline rows, plus a steady-state replay guard; each row carries a `source` recording whether it is a Swift parity number or this port's own measurement. Full details in `docs/BENCHMARKING.md`.
- `docs/`: repository documentation directory. `docs/BENCHMARKING.md` details benchmark harness modes, mach memory sampling, and the memory oracle baseline assertions; `docs/TESTING.md` documents test suite organization, macOS and environment-variable gating conventions, and test writing rules.

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

Numerics parity with any upstream implementation is explicitly out of scope;
only the structural and configuration contracts are exercised by the tests,
except where a real CPU-vs-GPU parity test exists (`crates/gpu`'s
`rms_norm_parity.rs`).

The four commands above cover everything except the three `#[ignore]`d
tests (two checkpoint downloads and the memory oracle), which are opt-in
and not part of the handoff gate. Run the oracle when a change could move
memory or decode throughput. `docs/TESTING.md` documents the gating
conventions and the test-writing rules (never hardcode a fixture token
id, never assert generated text, prefer exact assertions over
thresholds); `docs/BENCHMARKING.md` documents the benchmark modes and
baselines.

See `DEVIATIONS.md` for the full list of what this port scaffolds versus
fully implements, `ROADMAP.md` for phase-by-phase scope, and
`docs/NEW_MODEL.md` for the end-to-end checklist for wiring a new model
family (what to map, what to specialize, what to measure, in order).
