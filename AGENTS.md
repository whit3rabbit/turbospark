# AGENTS.md

CLAUDE.md is a symlink to this file.

Conventions, gotchas, and commands for working in this Rust workspace,
a behavior-compatible port of the Mference Swift inference engine (see
`ROADMAP.md` for phase-by-phase scope and `DEVIATIONS.md` for what is
scaffolded rather than fully wired). Keep all code, comments, and docs
ASCII: no emojis and no em dashes (project rule).

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

# Run the CLI (validates the invocation; on macOS with --prompt mode also
# attempts real generation against --model (see DEVIATIONS.md for scope)).
cargo run -p mrefrust-cli --bin mference-check -- --model /path/to/model --prompt "hi"

# Run the OpenAI-compatible server (scripted responses; see DEVIATIONS.md).
cargo run -p mrefrust-server --bin mference-server -- <tokenizer-dir> [port]

# Run the throughput benchmark harness (scripted producer; see DEVIATIONS.md).
cargo run -p mrefrust-bench --bin mference-bench -- <tokenizer-dir>
```

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
   with `--prompt` mode, it also attempts real generation against `--model`
   via `RealForwardRunner` (see `DEVIATIONS.md`).

8. `crates/gpu` is the one crate with a hard platform gate: everything in
   `src/` is `#[cfg(target_os = "macos")]`, so `cargo build --workspace` /
   `cargo test --workspace` succeed on Linux with the crate compiling to
   (effectively) nothing. Dispatched, parity-tested Metal pipelines
   (`rmsnorm_no_scale`, `rms_norm_bf16w`, both `_perhead` norm variants,
   `rope_proportional_neox` (which with `rotated_pairs = head_dim/2` IS
   default full-head NeoX -- no separate default-rope wrapper exists),
   `logit_softcap_softmax`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`
   (both with offset-bound resident variants), `router_gemv_gemma4_r4`,
   two-pass split-KV `attention_decode`, `moe_decode` decode pair, and
   `utility` elementwise kernels including the port-local `scalar_mul_fp16`)
   are compiled from vendored MSL source at
   runtime, matching how Mference itself builds pipelines. `MetalContext::pipeline`
   takes caller-supplied `FunctionConstantValues`, so a new kernel module owns
   its own specialization rather than sharing one hardcoded set.
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
    answers via `mference-check` (see DEVIATIONS.md — raw prompts babble,
    the IT model needs its `<|turn>` markup). Qwen 3.6 and
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
    meaningless.

## Layout

Update layout as needed:

```
crates
├── bench              # mference-bench: fixed-prompt throughput harness
├── cli                # mference-check binary: the process entry point
├── compute            # CPU reference kernels + destination compute strategy
├── core               # shared primitives, errors, runtime config
├── gpu                # Metal pipeline cache + kernel dispatch (macOS only)
├── invocation         # CLI argument parsing, request assembly, diagnostics
├── model-io           # manifest/arch validation, packed-expert layout,
│                      # resident index, SHA-256 verify, install receipt
├── repack             # safetensors header parsing, ranged-download
│                      # planning, int4/int8 repack, full-install verify
├── runtime            # the raw-completion prefill+decode loop
├── selection          # token sampling: shaping, truncation, penalty, choose
├── server             # OpenAI-compatible Chat Completions server (axum)
├── streaming          # pread-based expert streamer + LFU/LRU cache policy
├── tokenizer          # tokenizer wrapper, chat templates, tool-call parsing
└── window-fit         # conversation-window fitting (turn dropping)
```

- `crates/core`: shared primitives (token id, logit value, logits view) and the
  public runtime configuration with its allowed value sets and builder.
- `crates/compute`: CPU reference kernels (RmsNorm, WHT, RoPE, causal
  attention, int4/int8 affine quant + GEMV, embedding lookup, MoE FFN,
  logit softcap-softmax, RelError/tolerance table) plus the
  destination-selected compute strategy marker type. These are the
  numerical ground truth `crates/gpu`'s kernels are validated against.
- `crates/invocation`: pure translation of command-line argument tokens into
  a validated invocation request, a help short-circuit, or one of six typed
  failures, plus usage-text rendering and the pure outcome-to-exit-status and
  outcome-to-stream routing decisions. Performs no filesystem, environment,
  or process I/O.
- `crates/selection`: candidate selection from a per-candidate score vector
  under a validated shaping configuration (temperature, top-k, top-p,
  repetition penalty, seed), an accumulated history, and a step position.
  Numeric parity with any upstream implementation is out of scope; only the
  observable contract is exercised.
- `crates/window-fit`: pure, deterministic conversation-window fitting.
  Drops the oldest eligible turns from a conversation, using a
  caller-supplied whole-conversation length measurement, until the
  measured length is under a caller-supplied bound or nothing eligible
  remains. An optional leading instruction turn and the newest turn are
  never removed. Performs no input or output and holds no state between
  calls.
- `crates/tokenizer`: wraps the HF `tokenizers` crate; resolves the Gemma
  4 / ChatML (Qwen) / DeepSeek-V4 chat dialect from a loaded tokenizer's
  special tokens; renders the text-only chat templates plus DeepSeek's
  hand-rolled native tool chat; the generic Jinja-templated tool chat for
  Gemma/ChatML (`minijinja` + `minijinja-contrib`'s `pycompat`, rendering
  the checkpoint's own `chat_template.jinja`, tested against the real
  vendored Qwen ChatML template); streaming detokenizer and stop matcher
  (the stop set unions the dialect's own stops with the checkpoint's
  `generation_config.json` `eos_token_id` list -- that file, not
  `tokenizer_config.json`, is the authority for multi-stop checkpoints);
  Gemma/Qwen/DeepSeek tool-call DSL parsers and a streaming structured
  assistant-output decoder.
- `crates/model-io`: `manifest.json` decode and field-by-field validation
  against a resolved `ArchConfig` (with the three canonical Gemma
  4/Qwen3.6/DeepSeek-V4-Flash baselines), `packed_experts/layout.json`
  decode, the `model_weights.bin` resident tensor index reader, an `mmap`'d
  resident-buffer view, streaming SHA-256 verification, and the trusted
  install receipt. Allowed a narrow amount of `unsafe` (the `mmap` call).
- `crates/streaming`: routed-expert `pread` streamer with a fixed per-layer
  slot cache. The LFU/LRU eviction policy (`ExpertCache`) is pure logic,
  separated from the actual file I/O so it can be tested against scripted
  access traces without a real model install. `rdadvice` is the other
  `unsafe`-carrying module (macOS `F_RDADVISE`; a documented no-op
  elsewhere).
- `crates/gpu`: Metal device/pipeline-cache context and per-kernel dispatch.
  macOS-only; compiles to nothing elsewhere. Multiple kernels
  (`rmsnorm_no_scale`, `rms_norm_bf16w`, `rope_proportional_neox`,
  `logit_softcap_softmax`, `dequant_int4_gemv_simd`, `dequant_int8_gemv_simd`,
  two-pass split-KV `attention_decode`, `moe_decode` decode pair, and
  `utility` elementwise kernels) are wired end to end and parity-tested against
  the matching `mrefrust_compute` reference on real hardware. `KvCacheManager`
  allocates and manages real per-layer Metal KV buffers used by
  `RealForwardRunner`. `GdnStateManager` and `Dsv4StateManager` allocate real
  per-layer Metal buffers but stay unwired (their compute kernels are unported);
  `PrefillChunkScratchLayout`/`PrefillChunkScratchBuffers` size and allocate the
  chunked-prefill scratch buffers, also undispatched (the tile kernel is
  descoped). The `sample` kernel (no CPU reference exists to verify a port
  against) and fused lm_head are not yet vendored or dispatched.
- `crates/runtime`: the raw-completion prefill+decode loop
  (`run_raw_completion`), wiring a `LogitProducer`, the tokenizer's
  streaming detokenizer and stop matcher, and `selection::select` into one
  token generation loop. `ScriptedLogitProducer` is what every test outside
  `real_forward.rs` drives the loop with (see Gotcha 10). `RealForwardRunner`
  (macOS/GPU only, `src/real_forward.rs`) is a real `LogitProducer`: a
  genuine transformer forward pass through real GPU kernels (including
  real GPU decode attention) and real quantized weights, supporting both
  dense and MoE FFN layers. Dense bridges the gated FFN on the CPU via
  `mrefrust_compute::run_ffn` (no GPU FFN kernel exists yet); MoE runs a
  real GPU router GEMV plus real GPU GEMVs for each selected expert, with
  host-side top-k selection and the same CPU-bridged gated activation.
  See Gotcha 12.
- `crates/cli`: the `mference-check` binary: the resolved process entry
  point (see Gotcha 7). Parses `argv`, applies `invocation`'s exit-status
  and stream-routing decisions, prints the resolved request for a
  validated invocation, and (macOS, `--prompt` mode only,
  `src/generate.rs`) attempts real generation against `--model` via
  `RealForwardRunner` (see Gotcha 12).
- `crates/repack`: safetensors header parsing (pure, tested against
  synthetic fixtures, no network needed), a `RangeSource` trait for ranged
  reads (HTTP-backed for real installs, in-memory for tests) with the
  two-step header-fetch plan, per-row int4/int8 quantization repack
  (reusing `mrefrust_compute`'s quantizer), byte-exact `.gturbo` directory
  assembly (`write_gturbo_install`, round-trip tested through every
  `mrefrust_model_io` loader), a real named resident-tensor index writer
  (`write_gturbo_install_with_resident_index`, unlike `write_gturbo_install`
  which only ever writes an empty index), and full-SHA256 install
  verification. `synthetic_model.rs`'s `build_synthetic_gemma4_install`
  (dense) and `build_synthetic_gemma4_moe_install` (routed-expert FFN) use
  the resident writer to build full, real, small "tiny Gemma 4" `.gturbo`
  installs with deterministic (not trained) INT4-affine weights: what
  `crates/runtime`'s `RealForwardRunner` runs against, since no trained
  checkpoint exists in this environment (see Gotcha 12).
  `hf_checkpoint.rs`'s `orchestrate_llama_checkpoint` walks a real
  *downloaded* HF checkpoint's Llama-family-named tensors through the
  quantizer and writer end to end: proven against a real ~269MB
  Hugging Face Hub download in a network-gated, `#[ignore]`d test
  (`tests/hf_checkpoint_network.rs`; run explicitly, not part of the
  default suite). Not proven to also run through `RealForwardRunner`
  (separate V projection, scaled RMSNorm: that runner doesn't support
  either yet); see `DEVIATIONS.md`. `gemma4_checkpoint.rs` is the real
  Gemma 4 mapping: `config.json`/quantization parsing, mlx-community
  tensor-name classification and Swift slot ordering, pre-quantized
  INT4/INT8 pass-through (no re-quantization), per-expert blob slicing
  with one 16 KiB-rounded stride, `write_gemma4_install`, and the
  multi-shard + streaming pair (`Gemma4Shards` merges N shard headers
  into one name registry; `write_gemma4_install_streamed` computes the
  expert stride from headers alone and writes one layer at a time, so
  peak memory is one layer's blobs). `synthetic_real.rs`'s
  `build_synthetic_gemma4_real_install` pushes a deterministic
  real-naming safetensors blob through that exact pipeline (what the
  runner's learned-weight flow and the CLI test open).
  `tests/gemma4_checkpoint_network.rs` is the network-gated (`#[ignore]`d)
  proof against the real pinned
  `mlx-community/gemma-4-26b-a4b-it-4bit` checkpoint (~14.6 GB; same
  commit + index SHA-256 pins as Swift's `SupportedModelSource.gemma4`).
- `crates/server`: OpenAI-compatible `/v1/chat/completions` on loopback
  (axum), both the full-response and SSE-streaming shapes, wired to
  `mrefrust-runtime`. `ScriptedChatModel` is the only backend (see
  Gotcha 10); real weights are future work.
- `crates/bench`: the `mference-bench` binary: three fixed prompts, a
  fixed seed, a discarded warmup run per prompt, driven through the real
  `run_raw_completion` loop and timed. Measures this port's loop overhead
  via a `ScriptedLogitProducer`, not real inference throughput (no real
  weights exist to measure); see `DEVIATIONS.md`.

## Verification policy

Every change should keep these green before handoff:

```sh
cargo build --workspace
cargo test --workspace
cargo fmt --check
cargo clippy --workspace --tests
```

Numerics parity with any upstream implementation is explicitly out of scope;
only the structural and configuration contracts are exercised by the tests,
except where a real CPU-vs-GPU parity test exists (`crates/gpu`'s
`rms_norm_parity.rs`).

See `DEVIATIONS.md` for the full list of what this port scaffolds versus
fully implements, and `ROADMAP.md` for phase-by-phase scope.
