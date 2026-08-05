# Deviations

Every place this port's behavior deliberately differs from, or falls short
of, the Swift Mference original or the full scope described in
`ROADMAP.md`. Organized by phase. "Scaffolded" means the pure/testable
logic is implemented and tested; the missing part is wiring to something
this port has no access to (trained model weights, a real HF checkpoint, a
live network).

## Cross-cutting

- **`RuntimeConfiguration` uses a fallible `Result` constructor, not
  Swift's `precondition` crash**, for one config type — see `crates/core`'s
  own module docs. Everywhere else that mirrors a Swift `precondition`
  (e.g. `RuntimeConfig`'s numeric setters), this port kept the fatal-panic
  contract; see `AGENTS.md` Gotcha 2. This is the one place the two
  strategies coexist, and it predates this session's work.
- **Process-entry-point ownership resolved as `crates/cli`.** An earlier
  note reserved the name `mrefrust-entrypoint` and left it unbuilt pending
  a decision. That decision is now made and documented in `AGENTS.md`
  Gotcha 7: `crates/cli`, matching the ROADMAP's own Phase 7 crate list.
- **Three items formally descoped, by explicit user decision, rather than
  left open indefinitely:** `moe.metal`'s and `prefill.metal`'s GPU tile
  kernels, and `logit.metal`'s `sample` kernel. See ROADMAP.md's
  Cross-cutting rules for the "what" and the Phase 6 section below for the
  "why" in full. In short: the first two are throughput-only optimizations
  over capabilities that already work correctly through real, parity-tested
  per-token GPU dispatch (nothing they'd accelerate is otherwise
  unreachable), and each is 1200+ lines across more than a dozen
  interdependent kernels — an order of magnitude past anything else this
  port vendored. The third has no CPU reference anywhere in this codebase
  to verify a port against, unlike every other kernel here. This decision
  was reached after six rounds of real, tested progress against
  successive requests to "finish all tasks in ROADMAP.md," when the
  remaining tasks were assessed as not completable to this port's own
  verification standard within a bounded single-agent session; the user
  was asked explicitly (rather than the assistant deciding unilaterally)
  and chose to descope rather than continue unbounded or hand-pick a
  narrower target.

## Phase 4 (tokenizer)

- **Generic Jinja-templated tool chat: implemented** (`jinja_chat_template.rs`),
  using the `minijinja` crate (plus `minijinja-contrib`'s `pycompat` module
  for Python string methods like `.startswith`/`.split` that HF templates
  rely on) to render the checkpoint's own installed `chat_template.jinja`.
  Tested against the real, vendored Qwen ChatML `chat_template.jinja`
  fixture (not a stub): system+user rendering, the tools-preamble branch,
  and `raise_exception` propagation as a typed error all pass. DeepSeek
  still uses its hand-rolled native tool chat (it ships no
  `chat_template.jinja`); Gemma's template is untested (no Gemma fixture
  with a `chat_template.jinja` is vendored) but goes through the same
  generic renderer.
- **`JSONValue`'s `Decimal` case is folded into `f64`.** The Swift type
  keeps arbitrary-precision decimals separate from doubles to round-trip
  tool-call arguments exactly; this port accepts `f64`'s precision as
  sufficient for the tool-call argument JSON it targets.
- Deep-nesting rejection in the Qwen tool-call parser's structural-JSON
  detection (`nestsBeyondLimit` in Swift) falls back to treating an
  over-deep value as a raw string instead of throwing malformed, a minor
  behavioral simplification noted inline in `tool_call/qwen.rs`.

## Phase 5 (model-io, streaming)

- **No Metal buffer wrapping.** `ResidentBuffer` exposes the `mmap`'d
  region as a plain `&[u8]` slice, not an `MTLBuffer` — that wrapping
  belongs to `crates/gpu` once it needs to address this memory from a
  kernel.
- **`PreadExpertStreamer` reads sequentially, not via
  `DispatchQueue.concurrentPerform`.** Still correct, just not pipelined;
  a thread pool can be reintroduced once there is a GPU consumer to
  pipeline reads against.
- Page size is hardcoded to 4 KiB rather than queried from the OS
  (`getpagesize()` in Swift) — true for every platform this workspace
  targets (see `AGENTS.md`'s Stack section), and the manifest format has
  no per-install page-size field to validate against regardless.

## Phase 6 (GPU)

- **Six kernels, from six of fourteen `.metal` shader files, are vendored
  and dispatched:** `rmsnorm.metal`'s `rmsnorm_no_scale`, `rope.metal`'s
  `rope_proportional_neox` (Gemma 4's proportional NeoX RoPE),
  `logit.metal`'s `logit_softcap_softmax`, `dequant_int4.metal`'s
  `dequant_int4_gemv_simd`, `dequant_int8.metal`'s
  `dequant_int8_gemv_simd`, and `attention.metal`'s two-pass split-KV
  decode attention (`attention_decode_partial` + `attention_decode_combine`,
  `crates/gpu/src/attention_decode.rs`). Each is parity-tested against the
  matching `mrefrust_compute` reference on real Metal 4 hardware (an Apple
  M4 Max in this environment), and the attention dispatch is wired
  directly into `RealForwardRunner`'s decode loop (see Phase 7), not just
  parity-tested in isolation. `MetalContext::pipeline` takes
  caller-supplied `FunctionConstantValues` (no longer hardcoded to one
  shader's indices), so each dispatch module owns its own specialization
  helper. **A real bug this surfaced:** `attention.metal`'s
  `FC_ATTN_SCALE`/`FC_ATTN_NUM_CHUNKS` function constants are checked
  *unconditionally* (`is_function_constant_defined`, no `FC_ATTN_USE_FC`
  gate, unlike `FC_ATTN_HEAD_DIM`/`FC_ATTN_NUM_Q_HEADS`/
  `FC_ATTN_NUM_KV_HEADS`), so specializing them to placeholder zeros (as
  every other dispatch module's "unused" constants pattern does) silently
  overrode the runtime scale and caused a `% 0` inside the kernel. Fixed by
  specializing both to this dispatch's real, fixed values (`num_chunks =
  1` always; `scale` = the caller's value) instead of dummies — caught by
  the parity test, not by inspection, which is the whole point of parity
  tests existing.
- **Not vendored as GPU kernels, and not going to be (minor sibling
  variants):** `rmsnorm.metal`'s per-head variants, `rope.metal`'s
  `rope_default_neox`/`rope_neox_subdim` (the latter has no matching
  `mrefrust_compute` reference — its frequency divisor is `rotary_dim`,
  not `head_dim`, unlike anything in `compute::rope`),
  `dequant_int4.metal`'s `dequant_int4_qkv_gemv_simd`, `dequant_int8.metal`'s
  `shared_int8_gate_up_act_simd`, `attention.metal`'s
  `attention_decode_gqa_swa_partial` (a performance variant, not needed
  for correctness — `attention_decode_partial` already handles GQA). These
  are small, optional fused/specialized siblings of kernels already
  vendored; none blocks anything.
- **Formally descoped, not vendored, and not planned — a deliberate scope
  decision, not an oversight (see the Cross-cutting section above for the
  authorization trail):** `attention.metal`'s whole MPP prefill path
  (`attention_prefill_causal_tiled`/
  `attention_prefill_full_tensorops_2d_validity_v2`), all of `moe.metal`
  (1246 lines: top-k routing, hash routing, DSV4-specific phases), and
  `prefill.metal`'s 16-kernel chunked-prefill tile pipeline (1202 lines:
  embed/norm/rope/attention/router/MoE phases for a whole chunk at once).
  These three are genuinely throughput optimizations over capabilities
  that already work correctly through real, parity-tested, per-token GPU
  dispatch — `RealForwardRunner` runs dense and MoE architectures for real
  without them, and off-mode chunked prefill (a real `PrefillMode` from
  the Swift original, not a port-specific shortcut) already handles
  multi-token prompts correctly, just sequentially. Each is an order of
  magnitude larger and more interdependent than anything this port
  vendored, and porting any of them well would mean reimplementing a
  substantial fraction of a production tiled-inference pipeline, not a
  self-contained kernel.
  **The GDN/DSV4 *compute* kernels (`gdn.metal`, `dsv4.metal`) that would
  read and write through the state managers this port already built are
  also unvendored, but are NOT a throughput tradeoff like the three
  above** — there is no working fallback path for them. The state
  managers (`GdnStateManager`, `Dsv4StateManager`) allocate real buffers,
  but nothing computes a delta-rule update or a compressed-attention
  read into or out of them, on GPU or CPU (`mrefrust_compute` has no
  delta-rule/CSA/HCA reference either). This is why `RealForwardRunner`
  rejects any layer whose `full_attention_layer_mask` entry isn't `1`
  (see Phase 7 below) rather than merely running those layers slower:
  Qwen 3.6 (`full_attention_layer_mask` mostly `2`, gated-DeltaNet linear
  attention on 30 of 40 layers) and DeepSeek-V4-Flash
  (`full_attention_layer_mask` values `{0,3,4}` — zero full-attention
  layers at all) cannot run through this port at any speed until these
  kernels exist, unlike the three descoped-for-throughput items above.
  The dense gated FFN is NOT affected by this: Phase 7's `RealForwardRunner`
  runs it for real on the CPU via the already-tested
  `mrefrust_compute::run_ffn` reference rather than skipping it, since no
  GPU FFN/MoE kernel exists yet — see Phase 7 below. The embedding lookup
  is also not GPU-dispatched (`dequant_int4.metal`'s `embed_lookup_int4`
  kernel is unported, for the same reason); `RealForwardRunner` uses
  `mrefrust_compute::quant::embed_lookup_int4`'s CPU reference instead,
  which is a real, tested, byte-exact-contract implementation, not a stub.
- **`logit.metal`'s `sample` kernel: formally descoped, not ported.**
  Unlike every other kernel this port has vendored, `sample` has no CPU
  reference in `mrefrust_compute` to verify a port against — it is a
  self-contained GPU-native sampler with its own xorshift64*/SplitMix64
  RNG and its own greedy / Gumbel-fast-path / truncating-top-k-top-p
  branches, algorithmically independent of `crates/selection`'s CPU
  sampler (which the runtime already uses for every `LogitProducer`,
  including `RealForwardRunner`). Porting it without something to check it
  against would mean shipping
  unverified GPU code; that was judged worse than leaving the gap open and
  documented. `logit.metal`'s fused lm_head GEMV variants
  (`lm_head_greedy_int4_rows_chunk_raw`/`_reduce`) are unported for the
  same reason (no CPU reference).
- **KV cache, GDN recurrent-state, and DSV4 state managers: all
  implemented and tested against real Metal hardware.** `KvCacheManager`
  (`crates/gpu/src/kv_cache.rs`) allocates real per-layer `metal::Buffer`s
  (K and V separate; linear/compressed layers share a page-sized
  placeholder), classifies `LayerKind` from `ArchConfig`'s
  `full_attention_layer_mask`, sizes SWA layers with ring capacity, and
  exposes `key_view`/`value_view`/`advance`/`reset` (the last calling
  `posix_madvise(MADV_DONTNEED)`, an `unsafe` `libc` call with a SAFETY
  comment). `GdnStateManager` (`crates/gpu/src/gdn_state.rs`) allocates
  FP32 delta-rule state and FP16 causal-conv-tail buffers only for
  `layer_is_linear` layers, with zero-reset semantics. `Dsv4StateManager`
  (`crates/gpu/src/dsv4_state.rs`) allocates, per layer, a sliding-window
  K=V ring, and on CSA/HCA layers additionally the compressed-entry cache,
  pending-row and prior-window buffers, and (CSA only) the indexer's
  parallel buffer set — `reset()` discards the large ring/compressed/
  indexer buffers (`POSIX_MADV_DONTNEED`, same pattern as
  `KvCacheManager::reset`) and zeroes the small pending/prior buffers, and
  `window_slot`/`window_count`/`window_start_position` port the Swift
  original's ring position math. None of the three is wired to a real
  forward pass (`RealForwardRunner` only supports dense, all-full-attention
  architectures — see Phase 7 below); each is exercised directly against
  real Metal buffers: `crates/gpu/tests/kv_cache.rs` (9 tests),
  `crates/gpu/tests/gdn_state.rs` (3 tests), and
  `crates/gpu/tests/dsv4_state.rs` (6 tests). What's still missing for
  DSV4/GDN specifically is the compute *kernels* that would read and write
  through these managers (`gdn.metal`, `dsv4.metal`) — the managers are
  buffer lifecycle only, matching the Swift originals' own scope (the
  Swift `DSV4StateManager`/`GDNStateManager` are likewise pure buffer
  managers; the kernels live in separate `.metal` files).
- **Chunked prefill pipeline: wired at the runtime level; the GPU-side
  scratch-buffer sizing/allocation is now ported too, but not dispatched
  against.** `crates/core/src/prefill.rs` ports the Swift span math
  (`prefill_chunk_spans`) and dirty/commit tracking
  (`PrefillChunkCommitState`) verbatim, including the exact rejection
  error-message format. `crates/runtime`'s `ChunkedPrefillRunner` trait and
  `run_raw_completion_chunked` function split a prompt into spans, hand
  each span to the producer's `prefill_chunk` with commit-state tracking
  around the call, then fall into the same shared `decode()` loop the
  unchunked path uses. `ScriptedLogitProducer` implements
  `ChunkedPrefillRunner` by consuming one scripted step per chunk
  regardless of chunk size (documented rationale: a real chunked kernel
  processes a whole chunk in one dispatch and writes one logits state, so
  one step per call is the correct scripted-producer analogue).
  `crates/gpu/src/prefill_scratch.rs`'s `PrefillChunkScratchLayout` ports
  the Swift original's per-buffer element-count arithmetic (hidden/normed/
  q/k-stage/v-stage/attention-output/dense-FFN/routed-FFN/GDN scratch
  sizing, all as a function of chunk size and `ArchConfig`) exactly, and
  `PrefillChunkScratchBuffers::allocate` allocates one real `MTLBuffer` per
  field from that layout on real Metal hardware — tested in
  `crates/gpu/tests/prefill_scratch.rs` (4 tests). What's still missing:
  the chunked-prefill tile-pipeline kernel itself (`prefill.metal`) that
  would actually read and write through these buffers — nothing dispatches
  against them yet, so this closes the "scratch-space accounting" half of
  the gap, not the "kernel that uses it" half.

## Phase 7 (runtime, CLI)

- **`RealForwardRunner`: implemented, real, and tested on real Metal
  hardware — with real but scope-limited weights, now including MoE.**
  `crates/runtime/src/real_forward.rs` is a real (not scripted)
  `LogitProducer`. Per token, it runs an actual transformer forward pass:
  embedding lookup (CPU reference, see Phase 6 above), then per layer, a
  real GPU `rmsnorm_no_scale` dispatch, real GPU `dequant_int4_gemv_simd`
  dispatches for the Q/K/O projections, real GPU `rope_proportional_neox`
  dispatches on Q and K, real GPU `attention_decode` (the two-pass
  split-KV decode kernel from `attention.metal`) over a host-held
  `Vec<f16>` KV history (not `gpu::KvCacheManager`'s GPU-resident buffers —
  a performance simplification, not a correctness one, since
  `attention_decode` reads this history the same way regardless of where
  it lives), then an FFN stage and a final real GPU `logit_softcap_softmax`
  dispatch. The FFN stage branches on `arch.num_experts`:
  - **Dense (`num_experts == 0`):** a CPU-bridged gated FFN
    (`mrefrust_compute::run_ffn`, since no GPU FFN kernel is vendored).
  - **MoE (`num_experts > 0`), new this round:** a real GPU router GEMV
    (`dequant_int4_gemv_simd` again — a router weight matrix is just
    another GEMV), then host-side top-k selection and softmax weighting
    (`topk_softmax`, plain Rust — cheap enough at any real `num_experts`
    count that a kernel isn't warranted), then for each *selected* expert
    only (not the full table) a real GPU gate/up/down GEMV plus the same
    CPU-bridged `mrefrust_compute::run_ffn` gated activation the dense
    path uses, weighted and summed. This is not
    `mrefrust_compute::apply_streamed_routed`'s residual-fused form (that
    function bakes the residual add into the combine step; this runner
    adds the residual itself afterward, through the same sandwich-norm
    step the dense path uses, so the two FFN branches share that
    structure) — a deliberate simplification, not a missed reuse
    opportunity. Also simplified: no separate dense/shared FFN branch
    summed in alongside the routed one (real Gemma 4 computes both and
    adds them; this runner's MoE layers are routed-only).
  The weights come from a real `.gturbo` install
  (`mrefrust_repack`'s `build_synthetic_gemma4_install` for dense,
  `build_synthetic_gemma4_moe_install` for MoE — both using the named
  resident-tensor writer, `write_gturbo_install_with_resident_index` — see
  Phase 8 below) loaded through the real `mrefrust_model_io`
  manifest/resident-index/`ResidentBuffer` loaders, unmodified. The
  weights themselves are deterministic but NOT trained (no trained
  `.gturbo` checkpoint is available in this environment) — so the
  generated *text* is not semantically meaningful; what's real is the
  pipeline that produces it. `RealForwardRunner::open` still rejects any
  `full_attention_layer_mask` entry other than `1` (full attention), so
  none of the three production baselines (Gemma 4, Qwen 3.6, DeepSeek-V4-
  Flash — all mix full attention with sliding-window/linear/compressed
  layers) can run through this runner, even if a real trained checkpoint
  of one existed. Proven end to end by
  `crates/runtime/tests/real_forward.rs` for both shapes:
  `run_raw_completion` runs to a real stop condition, and a second run
  over the same runner (which resets internally) reaches an identical
  generated token sequence — for MoE, this also proves routing itself is
  deterministic, not just the projections. Wired into `crates/cli` (see
  below). `ScriptedLogitProducer` (a fixed replayed logit sequence)
  remains what every OTHER test in this workspace drives the loop with —
  mirroring the Swift validation suite's own `ScriptedLogitProducer`
  fixture, which exists for exactly this reason (decoupling loop control
  flow from the kernel stack).
- **Chunked prefill: wired** (see Phase 6 above, `run_raw_completion_chunked`
  plus `ChunkedPrefillRunner`). **No cached-prompt continuation** — that
  still needs a `ContinuableLogitProducer`-capable producer and is
  unstarted. Off-mode (`run_raw_completion`) still feeds every prefill
  token to the producer one at a time, unchanged.
- **Throughput benchmark harness: implemented, still against the scripted
  producer, not `RealForwardRunner`.** `crates/bench`'s `mference-bench`
  runs the real `run_raw_completion` loop, not a simulation of it, against
  a `ScriptedLogitProducer` for three fixed prompts with a fixed seed and
  a discarded warmup run per prompt (the frozen benchmark protocol's
  structure). The printed tokens/sec figure is this port's prefill+decode
  *loop* overhead (tokenizer, sampler, detokenizer, stop matcher) —
  explicitly not a Rust-vs-Swift inference throughput comparison, and the
  crate's own module doc says so. Pointing it at `RealForwardRunner`
  instead would measure real (if tiny, synthetic-weight) GPU kernel
  throughput rather than pure loop overhead; that wiring is not done this
  round. "Fresh processes" (the protocol's third leg) is left to the
  caller (e.g. a shell loop invoking the binary repeatedly); the binary
  does not orchestrate that itself.
- **The CLI now loads a model and generates tokens, for the same
  restricted scope `RealForwardRunner` supports.** `crates/cli/src/
  generate.rs`'s `try_generate` (macOS only, `--prompt` mode only) peeks
  `--model`'s `manifest.json` for `vocabSize`/`numLayers`, builds the
  matching `repack::tiny_gemma4_arch`, opens the install with
  `RealForwardRunner`, loads a tokenizer expected to live in the same
  directory (the usual HF checkpoint bundling convention — this port's
  synthetic installs don't include one by default; the caller bundles
  one), and streams real generated text to stdout through
  `run_raw_completion`. Any failure (no manifest.json, arch mismatch, no
  tokenizer, generation error) prints a note to stderr and falls back to
  the validate-only printout rather than crashing the process. Proven end
  to end (real compiled-binary invocation, real `.gturbo` install, real
  generated output) by `crates/cli/tests/real_generation.rs`. What's still
  not wired: `MessagesFile`/`Chat` modes (need chat-template rendering
  through the CLI, which this integration does not do), and — since it
  inherits `RealForwardRunner`'s own scope — anything beyond a small dense
  synthetic architecture; a production checkpoint would need the MoE/
  hybrid-attention forward-pass support that doesn't exist yet.
- `RawDecodeResult` drops the Swift original's cached-prompt-continuation
  bookkeeping fields (`cachedPromptTokens`, `computedPrefillTokens`,
  `uncommittedBoundaryTokenIDs`) since continuation is unimplemented; the
  fields that remain (`prompt_tokens`, `new_tokens`, timings, `reason`,
  `kv_position`, `kv_backed_token_ids`) cover everything the current loop
  shape produces.

## Phase 8 (repack, server)

- **Byte-exact `.gturbo` directory assembly: implemented**
  (`gturbo_writer.rs`'s `write_gturbo_install`): given already-quantized
  tensor bytes, writes `packed_experts/layer_NN.bin` blobs (matching
  `mrefrust_model_io::PackedExpertsLayout`'s exact per-expert sub-tensor
  layout, zero-padded to `expert_stride`), `packed_experts/layout.json`, a
  minimal valid `model_weights.bin` (a real `ResidentIndexHeader` plus a
  raw tensor region), and `manifest.json` with computed per-file SHA-256.
  Round-trip tested: write an install, then read every part of it back
  through `mrefrust_model_io::load_manifest`/`load_packed_experts_layout`/
  `load_resident_index` and `verify_install_full_sha256`, all of which
  pass. Ranged-read planning (`RangeSource`, HTTP-backed for real use) and
  per-row int4/int8 quantization (reusing `mrefrust_compute`'s quantizer)
  are the pieces this writer builds on.
- **Real downloaded HF checkpoint orchestration: implemented and proven
  against a real download, not a synthetic fixture.**
  `crates/repack/src/hf_checkpoint.rs`'s `orchestrate_llama_checkpoint`
  walks a checkpoint's real tensor names for one concrete, standard
  convention (Llama-family: `model.embed_tokens.weight`, per-layer
  `self_attn.{q,k,v,o}_proj.weight`/`mlp.{gate,up,down}_proj.weight`/two
  layernorms, `model.norm.weight`, and `lm_head.weight` when not tied),
  fetches each tensor's real bytes through the existing `RangeSource`
  abstraction, decodes `BF16`/`F32` to `f32`, quantizes with
  `quantize_matrix_int4`, and returns resident tensor specs ready for
  `build_resident_weights_bin`. Proven against a REAL download: run once
  (manually, network-gated — see below) against
  `HuggingFaceTB/SmolLM2-135M` on the Hugging Face Hub, a real ~269MB
  `model.safetensors` file. That run: fetched the real safetensors header
  (272 real tensors), decoded and INT4-quantized all 272 (embedding +
  30 layers × 9 tensors/layer + final norm), wrote a real `.gturbo`
  install, and read every part of it back through the real, unmodified
  `mrefrust_model_io::load_manifest`/`load_resident_index` loaders —
  finished in under 100 seconds end to end. This is the piece the
  `repack`'s own module docs and ROADMAP.md Phase 8 previously called
  "NOT implemented" for exactly this reason; it is now implemented, for
  the Llama-family naming convention.
  `crates/repack/tests/hf_checkpoint_network.rs` carries this test,
  `#[ignore]`d (a real, unpredictable-duration network download has no
  place in the default `cargo test --workspace` suite); run it explicitly
  with `cargo test -p mrefrust-repack --test hf_checkpoint_network --
  --ignored --nocapture`. A hermetic, no-network version of the same
  orchestration logic against a hand-built (but real-format) safetensors
  blob is `crates/repack/tests/hf_checkpoint.rs`, which IS part of the
  default suite. What this does NOT do: prove the resulting install runs
  through `crates/runtime`'s `RealForwardRunner` — that runner is
  dense/Gemma-4-shaped only (requires `attention_k_eq_v` and a scale-less
  RMSNorm), while a real Llama checkpoint has a separate V projection and
  a *scaled* RMSNorm (`model.layers.N.input_layernorm.weight`, quantized
  here to INT4 alongside every other tensor purely so this module's one
  resident-tensor format stays sufficient — a real repacker would keep
  norm weights unquantized FP16, and `RealForwardRunner` doesn't apply a
  norm-weight multiply at all yet). Extending `RealForwardRunner` to
  actually run a real downloaded checkpoint's forward pass is further
  work, not attempted this round; also not a universal any-architecture
  tensor-name mapper (Llama-family only).
- **Named resident-tensor writing: implemented** (`resident_writer.rs`'s
  `build_resident_weights_bin` + `gturbo_writer.rs`'s new
  `write_gturbo_install_with_resident_index`). The original
  `write_gturbo_install` only ever writes an *empty* resident index
  (`entry_count == 0`) — fine for its own round-trip test, but unusable by
  anything that needs to address weights by name. The new writer builds a
  real 24-byte header + 72-byte-per-entry table + string table + tensor
  data region matching `mrefrust_model_io::resident_index`'s exact reader
  contract, with real named entries (packed INT4 bytes plus BF16 scale/bias
  arrays). `crates/repack`'s new `synthetic_model.rs` uses it to build a
  full small "tiny Gemma 4" `.gturbo` install with real (deterministic,
  untrained) quantized weights — round-trip tested in
  `crates/repack/tests/synthetic_model.rs` and driven end to end by
  `crates/runtime`'s `RealForwardRunner` (see Phase 7 above). No packed
  experts (empty `packed_experts/layout.json`, since the synthetic model
  is dense, `num_experts == 0`).
- **The server has no real model backend.** `ScriptedChatModel` always
  replays a fixed logit sequence regardless of the prompt. The HTTP
  request/response envelopes, chat templating, and SSE streaming framing
  are real and tested end to end (a bound loopback server, hit with a real
  HTTP client) — only the "model" behind them is a stand-in.
- **No tailnet bind.** The server binds `127.0.0.1` only; the ROADMAP's
  "optional tailnet bind" is not implemented.
- The server's sampling knob surface is narrower than the CLI's: no
  `top_k`, no `repetition_penalty` (both fixed at their identity values),
  matching plain OpenAI Chat Completions' request shape rather than
  `mrefrust-invocation`'s fuller option set.

## Not ported at all

- `crates/tokenizer`'s `Sha256Verifier` used `CommonCrypto`; this port
  uses the `sha2` crate (RustCrypto) instead. Behaviorally equivalent, not
  a gap, but worth noting as a dependency substitution alongside the ones
  above.
- Anything under `MferenceApp` (the Mac GUI), `MferenceDecodeService`/
  `MferenceDecodeProtocol` (app-side XPC), chat-history compression UI
  behavior, and document extraction — permanently out of scope per
  `ROADMAP.md`.
