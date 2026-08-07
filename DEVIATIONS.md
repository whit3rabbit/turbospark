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
  kernels, and `logit.metal`'s `sample` kernel. (`moe.metal`'s descope was
  later PARTIALLY REVERSED, also by explicit user decision: its decode
  pair is now vendored, dispatched, and production-wired — see Phase 6/7
  below; the routers, parallel top-k selectors, and DeepSeek INT2/hash
  family remain descoped.) See ROADMAP.md's
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

- **Metal buffer wrapping: RESOLVED.** `ResidentBuffer` still exposes the
  `mmap`'d region as a plain `&[u8]` slice (plus `mapped_bytes()`/
  `slice_shift()` accessors), and `crates/gpu`'s `ResidentGpuWeights`
  (`resident_metal.rs`) now wraps the whole mapping in ONE shared-storage
  `MTLBuffer` via `newBufferWithBytesNoCopy`, exactly as the Swift
  original's `ResidentBuffer.swift` does — zero-copy, proven by pointer
  identity on real hardware (`crates/gpu/tests/resident_metal.rs`). The
  mapping also gets `POSIX_MADV_RANDOM`, matching Swift.
- **`PreadExpertStreamer`: RESOLVED to the Swift shape.** Slots are now
  `posix_memalign`(2 MiB)-aligned, page-rounded allocations made once at
  open (`AlignedSlot`), exposing their base pointer so `crates/gpu` can
  wrap each slot zero-copy (`wrap_page_aligned_no_copy`); cache-plan
  misses are read in parallel, one thread per miss into its own disjoint
  slot (`std::thread::scope`, the `DispatchQueue.concurrentPerform`
  equivalent). The streamer is now wired into `crates/runtime`'s
  `RealForwardRunner` MoE path (its first production consumer): a
  streamed-expert install is proven to generate the exact token sequence
  the resident-expert install with identical weights generates
  (`crates/runtime/tests/real_forward.rs`,
  `streamed_expert_install_matches_resident_expert_install`). On streamed
  installs the GPU now reads expert weights IN PLACE from the slots'
  zero-copy Metal buffers via the `moe.metal` decode pair (see Phase 7);
  only resident-expert MoE (a synthetic-only shape) keeps the CPU
  `run_ffn` bridge.
- Page size: RESOLVED — now queried via `sysconf(_SC_PAGESIZE)` (16 KiB
  on Apple Silicon), no longer hardcoded to 4 KiB, and the repack
  resident-index writer 16 KiB-aligns the resident region (matching the
  Swift repacker's `Layout.pageBytes`) so the mapping starts page-aligned
  with a zero slice shift, as `newBufferWithBytesNoCopy` requires.

## Phase 6 (GPU)

- **The dispatched kernel set (grown well past the original six):**
  `rmsnorm.metal`'s `rmsnorm_no_scale`, `rms_norm_bf16w`, and both
  `_perhead` norm variants; `rope.metal`'s `rope_proportional_neox`
  (which with `rotated_pairs = head_dim/2` IS default full-head NeoX);
  `utility.metal`'s port-local `logit_softcap_fp16`, elementwise
  activation/residual kernels, and port-local `scalar_mul_fp16`;
  `embed_lookup_int4`; `dequant_int4.metal`'s `dequant_int4_gemv_simd`
  and `dequant_int8.metal`'s `dequant_int8_gemv_simd` (both with
  offset-bound resident variants); `router_gemv_gemma4_r4`;
  `attention.metal`'s two-pass split-KV decode attention
  (`attention_decode_partial` + `attention_decode_combine`,
  `crates/gpu/src/attention_decode.rs`, incl. SWA `kv_start` and the
  `FC_ATTN_RING_CAP` KV ring); and `moe.metal`'s decode pair
  (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`). Each
  is parity-tested against the matching `mrefrust_compute` reference on
  real Metal 4 hardware (an Apple M4 Max in this environment) and wired
  into `RealForwardRunner`'s decode loop (see Phase 7), not just
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
  variants):** `rope.metal`'s
  `rope_default_neox`/`rope_neox_subdim` (the former is subsumed by
  `rope_proportional_neox` at `rotated_pairs = head_dim/2`; the latter
  has no matching `mrefrust_compute` reference — its frequency divisor is
  `rotary_dim`, not `head_dim`, unlike anything in `compute::rope`),
  `dequant_int4.metal`'s `dequant_int4_qkv_gemv_simd`, `dequant_int8.metal`'s
  `shared_int8_gate_up_act_simd` (the real-checkpoint shared-expert
  branch runs the same math as three separate INT8 GEMV dispatches),
  `attention.metal`'s `attention_decode_gqa_swa_partial` (a performance
  variant, not needed for correctness — `attention_decode_partial`
  already handles GQA). These are small, optional fused/specialized
  siblings of kernels already vendored; none blocks anything.
  `attention_decode_gqa_swa_partial` remains unwired. Multi-chunk
  split-KV (`num_chunks > 1`) is now **wired and on by default** — see
  the split-KV entry below for the numbers and for why an earlier session
  concluded the opposite.
- **Formally descoped, not vendored, and not planned — a deliberate scope
  decision, not an oversight (see the Cross-cutting section above for the
  authorization trail):** `attention.metal`'s whole MPP prefill path
  (`attention_prefill_causal_tiled`/
  `attention_prefill_full_tensorops_2d_validity_v2`), `moe.metal`'s
  non-decode remainder (top-k routing selectors, hash routing,
  DSV4-specific phases — the decode pair itself IS vendored and
  production-wired, the partial reversal noted in Cross-cutting), and
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
  The dense gated FFN is NOT affected by this: it now runs fully on the
  GPU (gate/up/down GEMVs plus `utility.metal`'s
  `gelu_mul_fp16`/`silu_mul_fp16` activation — vendored and parity-tested
  in `crates/gpu/tests/utility_and_pass.rs`), as does streamed MoE (the
  `moe.metal` decode pair; see Phase 7 below); only resident-expert MoE
  (synthetic-only) keeps the CPU `run_ffn` bridge. The embedding lookup
  is GPU-dispatched too (`embed_lookup_int4`, parity-tested in
  `crates/gpu/tests/scaled_norm_and_embed.rs`, bound as offsets into the
  resident buffer). Sliding-window decode attention works through the
  full `attention_decode_partial` kernel's `kv_start` argument
  (parity-tested against the CPU `window` reference in
  `crates/gpu/tests/attention_swa.rs`), and the KV ring addressing
  (`FC_ATTN_RING_CAP`) IS dispatched: SWA layers allocate
  `min(max_context, sliding_window + 128)` KV rows (1152 for real Gemma 4
  at 4K, the Swift budget) and switch to the ring-specialized pipeline
  once `seq_len` exceeds the ring, the Swift activation rule
  (ring-layout parity in `attention_swa.rs`, wrap-vs-linear token
  equivalence in `crates/runtime/tests/real_forward.rs`, byte accounting
  in `crates/gpu/tests/kv_cache.rs`). Only the
  `attention_decode_gqa_swa_partial` performance variant remains
  undispatched. `rmsnorm_bf16w`
  (learned norm weights) and its per-head siblings are dispatched,
  parity-tested, and fed by the real-checkpoint tensor mapping (see the
  real Gemma 4 pipeline entry below).
- **Split-KV (`num_chunks > 1`) IS wired, and it is the single largest
  decode win this port has landed.** `attention_decode_partial`
  dispatches `num_q_heads * num_chunks` threadgroups. Gemma 4 26B has 16
  Q heads, so at one chunk a decode attention occupied 16 threadgroups on
  a 40-core M4 Max, each walking its whole KV range serially with two
  threadgroup-wide reductions per position. `chunks_for` in
  `crates/gpu/src/attention_decode.rs` now splits the range up to 16 ways
  (at least 16 positions per chunk, so short ranges stay at one chunk and
  therefore bit-identical to the unsplit path).

  Measured on the real 26B install, greedy, 32 expert slots, warm, short
  prompt (so the phase divisor is decode, per Gotcha 21):

  | forward passes | cb1 before | cb1 after | tok/s before | tok/s after |
  | --- | --- | --- | --- | --- |
  | 220 | 8.11 ms/token | 5.66 | 25.3 | 28.6 |
  | 620 | 13.09 | 5.71 | 22.3 | 28.6 |
  | ~810 | 15.63 | 5.98 | 20.9 | 27.8 |

  The memory oracle's three protocol cases, against the baseline rows
  recorded at merge a772b67, all still stopping `endOfTurn`:

  | case | before | after |
  | --- | --- | --- |
  | short-explanation | 20.27-20.40 tok/s | 22.73 |
  | medium-review | 15.77-15.97 | 21.13 |
  | long-synthesis | 11.60-11.71 | 20.71 |

  Peak footprint is unchanged (2,126 MiB against a 2,300 MiB ceiling,
  inside the run-to-run band), and the replayed warm case still grows
  +0.03 MiB, so this buys throughput without buying memory.

  cb1 goes from growing linearly in context to essentially flat. The
  isolated kernel bench (`crates/gpu/tests/attention_chunk_bench.rs`,
  `#[ignore]`d) measures 6x at 256 KV positions rising to 12-16x at 4096,
  across both the SWA and full-attention shapes, with 16 chunks at or
  near optimal everywhere and 32 already regressing.

  **An earlier session (2026-08-05) wired this, measured "no change", and
  reverted it.** That conclusion was wrong, and the reason is worth
  keeping: `MFERENCE_PHASES=1` divides every counter by ALL forward
  passes, prefill included. Its "~2300 context" row was a 2252-token
  prompt with `--max-new 150`, so 96% of the divisor was prefill calls
  running at short context, which flattened exactly the signal the A/B
  was looking for. Read any phase number as an average over the whole
  run's context range, not as a number at the final context.

  Because chunking reassociates the online-softmax partial sums, decode
  output is NOT bit-identical to the unsplit path at contexts past
  `16 * 16` positions. The parity tests
  (`split_kv_linear_layout_matches_cpu_window_reference` and its ring
  sibling in `crates/gpu/tests/attention_swa.rs`) hold it to the CPU
  reference instead.
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
- **The output head stops at softcapped LOGITS, not probabilities -- a
  consequence of the descoped `sample` kernel.** Swift's head hands
  `Sampler` raw FP16 logits and the sampler runs `logit_softcap_softmax`
  itself, because its GPU `sample` kernel consumes normalized probs. This
  port samples on the host through `selection::select`, whose documented
  input is a *score* vector and which runs its own softmax. So
  `RealForwardRunner` dispatches the cap alone (`utility.metal`'s
  port-local `logit_softcap_fp16`) and returns `softcap * tanh(z /
  softcap)` -- exactly what HF's `Gemma*ForCausalLM.forward` returns, and
  what `LogitProducer::produce` documents. Net math matches Swift; only
  the split between producer and sampler moves. Dispatching the fused
  `logit_softcap_softmax` here instead softmaxes twice: over V=262144 the
  second pass flattens a peaked distribution to near-uniform over the
  surviving top-k, which reads as fluent text derailing into word salad
  after a few dozen tokens. The bound is guarded in
  `crates/runtime/tests/real_forward_gemma4.rs` and the cap-without-
  normalization in `crates/gpu/tests/utility_and_pass.rs`.
- **Command buffers are pooled per token, not per process.**
  `MTLCommandQueue.commandBuffer` and
  `MTLCommandBuffer.computeCommandEncoder` return autoreleased objects,
  and a plain Rust binary has one autorelease pool, around `main`. Swift
  drains one per run-loop turn and so never had to think about it; this
  port wraps `RealForwardRunner::produce` in `gpu::autorelease_pool`.
  Without it every command buffer stayed alive to process exit: ~6 KiB
  each, 31 per token, ~180 KiB per decoded token, which reads as
  footprint growing with prompt length. Guarded by the steady-state
  replay test in `crates/bench/tests/memory_oracle.rs`.
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
  original's ring position math. `KvCacheManager` IS production-wired: it
  is `RealForwardRunner`'s persistent KV cache (K written in place by the
  GEMV, SWA ring addressing dispatched). The GDN and DSV4 managers stay
  unwired (their compute kernels are unported — see below); each of the
  three is also exercised directly against
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

- **The output head is skipped on non-final prompt tokens; Swift runs it
  on every one.** Swift's `RawCompletion.swift` off-mode prefill loop
  (`case .off:`) calls `producer.produce(token:position:into:)` per
  prompt token and, like this port's loop did, reads only the last
  result: `LogitProducer` has no way to say "these logits are going in
  the bin". This port adds one, a defaulted `produce_prefill` on the
  trait (`crates/runtime/src/producer.rs`) that `run_raw_completion`
  calls for every prompt token but the last. The default delegates
  straight to `produce`, so `ScriptedLogitProducer`, `ChunkedPrefillRunner`,
  and `crates/server`'s `Box<dyn LogitProducer + Send>` are all unchanged
  and unaffected. `RealForwardRunner` overrides it to set a `skip_head`
  flag that both of its head blocks (`real_forward.rs`'s short-name flow
  and `real_forward_gemma4.rs`'s learned-weight flow) honour, dropping
  the final norm, the full-vocab GEMV, the softcap, and the vocab-sized
  host readback. Everything else still runs: the open command buffer is
  committed AND waited on (the next token overwrites this one's scratch),
  and `kv.advance()` is unconditional.

  Generated output is unaffected by construction, since the last prompt
  token and every decode token still run the head; verified as
  md5-identical greedy output on the real 26B install before and after.
  Worth 6-7% of prefill wall clock on a 2252-token prompt (see ROADMAP.md
  item 4 for the measured pairs). Swift's chunked prefill avoids the same
  waste structurally instead, by running one head per chunk rather than
  per token; that path's GPU tile kernels are descoped here (see
  Cross-cutting rules in ROADMAP.md), which is why this port needed the
  off-mode fix.

- **`RealForwardRunner`: implemented, real, and tested on real Metal
  hardware — with real but scope-limited weights, now including MoE.**
  `crates/runtime/src/real_forward.rs` is a real (not scripted)
  `LogitProducer`. Per token, it runs an actual transformer forward pass:
  a real GPU `embed_lookup_int4` dispatch (bound as offsets into the
  resident buffer), then per layer, a
  real GPU `rmsnorm_no_scale` dispatch, real GPU `dequant_int4_gemv_simd`
  dispatches for the Q/K/O projections, real GPU `rope_proportional_neox`
  dispatches on Q and K, real GPU `attention_decode` (the two-pass
  split-KV decode kernel from `attention.metal`) over
  `gpu::KvCacheManager`'s persistent GPU-resident per-layer K/V buffers
  (the K projection is written directly into its cache slot by the GEMV
  and RoPE'd there in place; for `attention_k_eq_v` architectures the K
  buffer is bound as V too, so the V buffers stay untouched), then an FFN
  stage and a final real GPU `logit_softcap_fp16`
  dispatch. The memory path now matches the Swift original: the whole
  resident region is ONE zero-copy `MTLBuffer` over the mmap
  (`gpu::ResidentGpuWeights`), every projection binds weights/scales/
  biases as offsets into it, all activation scratch is preallocated at
  open, and a dense token is encoded as ONE command buffer (serial
  compute encoder, `gpu::PassEncoder`) with a single wait — residual
  adds and the gated-FFN activation run on the GPU in FP16
  (`utility.metal`), as Swift does, rather than in host f32 (a deliberate
  Swift-parity numeric change; the golden-token guard in
  `crates/runtime/tests/golden_tokens.rs` was re-verified across it).
  The FFN stage branches on `arch.num_experts`:
  - **Dense (`num_experts == 0`):** fully GPU: gate/up GEMVs,
    `gelu_mul_fp16`/`silu_mul_fp16`, down GEMV, all in the same command
    buffer as the rest of the token.
  - **MoE (`num_experts > 0`) on streamed installs (packed expert
    files):** the Swift CB1/CB2 decode shape. The router GEMV rides the
    token's first command buffer; its logits are the one host readback
    per MoE layer (host `topk_softmax` selects and weights — NOT the
    `router_topk_select_k8` kernel, whose softmax-over-top-k semantics
    differ from this port's softmax-over-all-then-renormalize; a
    documented deviation until the kernel selector is adopted
    wholesale); the selected experts are `pread` in parallel into the
    streamer's aligned slots (parallel per CHUNK of an expert blob, on
    the persistent `streaming::read_pool`, rather than Swift's one task
    per miss -- see the split-KV-adjacent measurement below); then a
    second command buffer runs the
    vendored `moe.metal` decode kernels
    (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`,
    parity-tested in `crates/gpu/tests/moe_decode.rs`) reading the
    expert blobs IN PLACE from the slots' zero-copy Metal buffers via a
    `RoutedBlobs` argument buffer — no expert byte reaches the host.
    The router readback is a full command-buffer wait rather than the
    Swift `MTLSharedEvent` passive wait: metal-rs 0.33 binds
    `signaledValue`/`setSignaledValue`/`notify` but not
    `waitUntilSignaledValue:timeoutMS:`, and spinning on `signaledValue`
    would steal the SoC power budget the GPU needs (the Swift original
    says so explicitly). The overlap that wait exists to buy is instead
    bought with a second command buffer: the shared-expert branch reads
    only `dense_x`, so it is committed on its own, queued behind the
    router's buffer, before the host waits — commit order on one queue
    is execution order, so it runs on the GPU through the router
    readback and the blocking `pread`. `MFERENCE_SHARED_CB=0` disables
    it (the A/B seam Swift keeps as `MFERENCE_ROUTER_EVENT=0`; same
    kernels, same order, identical output). Measured on the real 26B
    checkpoint (M4 Max, 32 slots, 5 interleaved pairs, overlap winning
    every pair): +4.0% decode throughput, 33.4 -> 34.8 tok/s. A separate
    fix to `MetalContext`'s pipeline cache (it keyed on the shader
    source's TEXT, so every one of the ~900 dispatches a token encodes
    rehashed tens of kilobytes of MSL; it now keys on the source's
    address) took the CPU encode bucket from 5.24 to 0.94 ms/token and
    decode from 34.8 to 42.6 tok/s.
    Swift's phase-1-hit CB is now ported too, on the same second-command-
    buffer principle: the layer's slot order is cache MISSES first then
    hits, so the hits (already in slot memory when the plan is built) get
    their phase-1 GEMV dispatched on its own command buffer before the
    `pread`, and the misses run in the main pass at `acts` offset zero.
    It takes a SECOND `RoutedBlobsBuffer`, since the host rebinds the
    main one for the full slot list while that dispatch may still be
    reading it; slot memory itself is safe because `ExpertCache::plan`
    reserves hit slots before choosing eviction victims, so the parallel
    miss reads never write a slot the dispatch reads.
    `MFERENCE_HIT_CB=0` is the A/B seam. Measured on the real 26B
    checkpoint (M4 Max, 32 slots, 5 interleaved pairs): +0.42 tok/s mean,
    winning 4 of 5 pairs, ~+1%; generated text md5-identical in all ten
    runs. The phase counters explain the small size and show the trick is
    at its ceiling rather than misfiring: GPU wait falls 17.2 -> 16.1
    ms/token (it hides ALL the phase-1 work the hits have to offer) and
    the new `hit_cb` bucket costs 0.76 ms/token of host bind-plus-commit,
    so about two thirds of the win is eaten by the extra command buffer
    per layer. The remaining exposed `pread` cannot be hidden this way:
    everything left depends on the bytes being read.
    Swift's one-layer-pipelined routed CB is ported too: a layer's routed
    phase-1/phase-2/sandwich tail commits as its OWN command buffer at the
    end of the layer (instead of rolling uncommitted into the next layer's
    first buffer), so the GPU starts it during the host's next-layer
    attention encode. It is retired right after the next layer's router
    wait, where it has provably completed (committed earlier on the same
    queue), so every later host buffer write (`routing_w` upload, the
    argument-buffer rebinds, the slot preads) stays race-free without any
    double buffering; the retire is an explicit timed wait
    (`pipeline_wait_nanos`, printed as `routed cb retire`) so correctness
    never rests on completion-order reasoning. Depth is pinned at one, as
    in Swift. Note what it does NOT buy: the pread cannot overlap the
    previous layer's routed work in either codebase (the pread needs the
    router output, which needs the attention that reads the routed tail's
    residual), so the win is the host encode window, not the I/O.
    `MFERENCE_ROUTED_PIPELINE=0` is the A/B seam. Measured on the real 26B
    checkpoint (M4 Max, 32 slots, 5 interleaved pairs, pipeline winning
    every pair): +0.63 tok/s mean (+2.5%), GPU wait 17.20 -> 16.07
    ms/token against 0.24 ms/token of retire cost, generated text
    md5-identical across all ten runs, across the full
    {ROUTED_PIPELINE, SHARED_CB, HIT_CB} seam grid, and across pipeline
    states at 16 slots.
    Expert prefetch/speculation stays deliberately unwired: the Swift
    original benched every shape to a dead end (cross-layer predictor
    Jaccard 0.039 with 7% copied-prediction hits, rejected before
    implementation; the previous-token predictor can never issue a read
    because per-layer private caches keep last token's experts resident;
    `MFERENCE_SPEC_PREFETCH=prefetch` measured as a no-op; RDADVISE "no
    stable production policy", off by default -- see Mference
    `docs/experiments/summaries/03-expert-cache-prediction-and-layout.md`
    and `04-rdadvise.md`). `crates/streaming`'s speculative APIs exist for
    parity and stay uncalled by the runtime on purpose.

    What IS possible, and landed on 2026-08-06, is making the exposed
    read SHORTER rather than hiding it. With the install's expert files
    in page cache the `pread` is a memcpy, not disk I/O (125 MiB per
    token in 5.26 ms, far past any SSD), so it is a bandwidth problem.
    Swift's one-task-per-miss shape collapses to a SINGLE-THREADED copy
    on the common warm-cache layer that misses exactly once (1.3 misses
    per layer at 32 slots), which measured 23.8 GiB/s against the 44.8
    GiB/s the same code reached when 8-slot runs forced ~5 concurrent
    misses. Splitting each miss into chunks took the bucket to 4.16
    ms/token, and moving those chunks onto a persistent pool
    (`streaming::read_pool`, needed because chunking multiplies the
    threads a layer wants and there are 30 layers per token) to 3.86 --
    27% off, measured in interleaved pairs as -1.93 ms/token. End to end
    that is ~8% off prefill (48-50s to 44-46s on a 2252-token prompt)
    and +6.5% decode (paired deltas +1.52/+2.55/+1.21 tok/s). Output
    stays md5-identical: the same bytes arrive, by a different route.

    `MFERENCE_PHASES=1` prints where the time goes and is what that
    should be judged against. A representative post-pipeline split
    (~200-token context, 32 slots, 83.9% hit rate, ~26 tok/s): GPU wait
    ~59%, expert `pread` ~33% (still largely exposed), CPU dispatch
    encoding ~4% (so `fused.metal`, which only cuts dispatch count, has
    little left to win here), hit-expert phase 1 ~2%, routed bind ~1%,
    routed cb retire ~1%, router readback+top-k ~0.5%. (The buckets move
    with the hit rate and cache state between runs, so re-measure per
    prompt rather than reusing a past split.) The same printout carries
    the per-command-buffer GPU BUSY attribution
    (`GPUStartTime`/`GPUEndTime`, a separate axis from the wall-clock
    buckets): on that run, cb1 (attention+router) 8.1 ms/token, routed
    FFN cb 2.8, final head 1.25, against 18.1 ms/token of wall-clock GPU
    wait — so ~5 ms/token is scheduling gap across the ~190 command
    buffers a token commits, and the attention decode path is both the
    largest GPU consumer and all of the context-length growth (cb1 busy
    grows ~2.3 ms/token per ~100 tokens of context; routed and final
    stay flat). Resident-expert MoE
    installs (a synthetic-only shape) still use the CPU
    `run_ffn` bridge (`moe_ffn_host`). The old bridge description: not
    `mrefrust_compute::apply_streamed_routed`'s residual-fused form (that
    function bakes the residual add into the combine step; this runner
    adds the residual itself afterward, through the same sandwich-norm
    step the dense path uses, so the two FFN branches share that
    structure) — a deliberate simplification, not a missed reuse
    opportunity. Also simplified: the SYNTHETIC short-name MoE flow has
    no separate dense/shared FFN branch summed in alongside the routed
    one; the real-checkpoint flow (see the real Gemma 4 pipeline entry
    below) does compute both and add them.
  The weights come from a real `.gturbo` install
  (`mrefrust_repack`'s `build_synthetic_gemma4_install` for dense,
  `build_synthetic_gemma4_moe_install` for MoE — both using the named
  resident-tensor writer, `write_gturbo_install_with_resident_index` — see
  Phase 8 below) loaded through the real `mrefrust_model_io`
  manifest/resident-index/`ResidentBuffer` loaders, unmodified. The
  weights themselves are deterministic but NOT trained (no trained
  `.gturbo` checkpoint is available in this environment) — so the
  generated *text* is not semantically meaningful; what's real is the
  pipeline that produces it. `RealForwardRunner::open` accepts full
  attention (mask 1) and sliding-window (mask 0) layers and rejects
  linear (2) and compressed (3/4) layers, whose kernels are unported —
  so Gemma 4's mask shape passes while Qwen 3.6 and DeepSeek-V4-Flash
  remain blocked on GDN/DSV4. Proven end to end by
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
- **Throughput benchmark harness: implemented, in three modes.** The
  scripted default: `crates/bench`'s `mference-bench` runs the real
  `run_raw_completion` loop, not a simulation of it, against a
  `ScriptedLogitProducer` for three fixed prompts with a fixed seed and
  a discarded warmup run per prompt (the frozen benchmark protocol's
  structure). The printed tokens/sec figure is this port's prefill+decode
  *loop* overhead (tokenizer, sampler, detokenizer, stop matcher) —
  explicitly not a Rust-vs-Swift inference throughput comparison, and the
  crate's own module doc says so. `--real` drives the same prompts
  through `RealForwardRunner` over a tiny synthetic install (real GPU
  dispatch path, still not a comparison number). `--model <install-dir>`
  (macOS) IS the Swift-comparison mode: the frozen community protocol
  (prompts vendored byte-exact from the Swift repo's
  `docs/benchmark-prompts/real-generation-v1`, seeds 20260721-23, temp
  0.2, top-k 64, top-p 0.95, max-new 1024, 4K context, chat-templated
  like the CLI) against a real repacked install, reporting split
  prefill/decode tok/s plus peak `phys_footprint` from a mach
  `task_info(TASK_VM_INFO)` sampler that matches the Swift
  `AppMemorySampler` counter and its every-8th-token cadence, and the
  Swift-spelling `[stop=...]` footer on stderr for the protocol's grep.
  On top of that, `crates/bench/tests/memory_oracle.rs` (`#[ignore]`d,
  needs `MREFRUST_GEMMA4_INSTALL_DIR`) is the memory oracle: it asserts
  the session peak footprint at or under the published Swift ceiling
  plus ~5 percent headroom (the Swift docs' own repeat-run variance),
  requires every measured case to stop `endOfTurn`, and on chips with a
  published Swift row (M5 Pro, M2) also asserts decode tok/s at or above
  the Swift floor; other chips get the memory assert plus reported-only
  throughput. "Fresh processes" (the protocol's third leg) is left to
  the caller (e.g. a shell loop invoking the binary repeatedly); the
  binary does not orchestrate that itself.
- **The CLI now loads a model and generates tokens, for the same
  restricted scope `RealForwardRunner` supports.** `crates/cli/src/
  generate.rs`'s `try_generate` (macOS only, all three modes) peeks
  `--model`'s `manifest.json` for `vocabSize`/`numLayers`, builds the
  matching `repack::tiny_gemma4_arch`, opens the install with
  `RealForwardRunner`, loads a tokenizer expected to live in the same
  directory (the usual HF checkpoint bundling convention — this port's
  synthetic installs don't include one by default; the caller bundles
  one), and streams real generated text to stdout through
  `run_raw_completion`. Any failure (no manifest.json, arch mismatch, no
  tokenizer, generation error) prints a note to stderr and falls back to
  the validate-only printout rather than crashing the process. All three
  invocation modes generate: `--prompt` encodes its text verbatim (no
  templating, matching the Swift original), `--messages-file` decodes a
  JSON `[{"role", "content"}]` conversation and renders it through the
  tokenizer's own dialect chat template (`add_bos` false, since the Gemma
  template emits the `<bos>` mark itself), and `--chat`
  (`crates/cli/src/chat.rs`) is the interactive REPL ported from
  `MferenceCLI/Run.swift`'s `runChat`: `/clear`, `/history`, `/quit`,
  `/exit`, `--system` seeding the opening turn, per-turn window fitting
  through `mrefrust-window-fit` (the Swift `trimChatHistory` contract), and
  the assistant reply appended to the history. Both chat modes print the
  Swift original's `[stop=... prefill=... tok/s=...]` footer to stderr,
  silenced by `--quiet`. Proven end to end (real compiled-binary
  invocation, real `.gturbo` install, real generated output) for every mode
  by `crates/cli/tests/real_generation.rs`. What's still not wired: KV
  reuse across chat turns (each turn re-prefills from a reset cache, as in
  Swift, since `ContinuableLogitProducer` is unported), chunked prefill
  (`--prefill-chunk` stays parsed-and-printed-only), the tool-calling/Jinja
  template path, and — since it inherits `RealForwardRunner`'s own scope —
  the layer kinds that runner rejects.
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
- **Real Gemma 4 checkpoint pipeline: PROVEN against the real
  production checkpoint (2026-08-05).** The pinned
  `mlx-community/gemma-4-26b-a4b-it-4bit` (~14.6 GB, same commit +
  index-SHA-256 pins as Swift's `SupportedModelSource.gemma4`) was
  downloaded, streamed through `write_gemma4_install_streamed` (~16
  minutes end to end), validated by every `mrefrust_model_io` loader,
  and generates REAL COHERENT TEXT through `mference-check`: a
  chat-formatted `What is the capital of France?` answers
  `The capital of France is **Paris**.` and stops on EndOfTurn. Two
  verification notes from that run: (a) numerics were cross-checked
  against an independent NumPy replica built from the mlx-lm
  `gemma4_text.py` reference — per-layer residual-stream stats match
  the GPU runner to FP16 precision at multiple positions, and the MLX
  `mx.dequantize` oracle confirms the pass-through byte layout exactly;
  (b) the checkpoint is instruction-tuned with Gemma 4's turn markup
  (`<|turn>user ... <turn|>` and a `<|channel>` structured-output
  vocabulary), so RAW text prompts produce out-of-distribution babble
  while chat-formatted prompts produce real answers — `--prompt` mode
  does no templating, so pass the markup yourself or wait for the chat
  modes.
  `crates/repack/src/gemma4_checkpoint.rs` parses a Gemma 4 `config.json`
  (`text_config`, `layer_types` -> mask, dual `rope_parameters`) and its
  MLX `quantization` object (per-tensor bits overrides; group size other
  than 64 is rejected — the GPU kernels assume 64), classifies the
  mlx-community naming exactly as the Swift `RepackPlanner` does, orders
  residents with the Gemma slot ranking, passes pre-quantized u32
  weights + BF16 companions through byte-for-byte (the MLX affine
  packing viewed as LE bytes IS this port's packed layout — no
  re-quantization), and slices `.experts.switch_glu.` bundles into
  per-expert blobs with one model-wide 16 KiB-rounded stride. Tensor
  names stay VERBATIM from the checkpoint. On the runner side,
  `crates/runtime/src/real_forward_gemma4.rs` (selected by `open()` when
  the index carries `language_model.` names) implements the full Swift
  decode flow: learned BF16 norms everywhere (NO extra `(1+w)` fold —
  the Swift kernel applies `w` directly to the checkpoint's own bytes,
  so this port does too), per-head q/k norms + per-head no-scale v norm,
  separate per-kind attention dims (SWA `head_dim`/`num_kv_heads` vs
  full `full_head_dim`/`num_full_kv_heads`), full layers writing V
  through the K projection into its own slot (the K=V quirk projects V
  separately and norms it without RoPE), full-rotation NeoX for SWA
  layers vs proportional for full layers, the INT8 router GEMV
  (`router_gemv_gemma4_r4`, parity-tested) with per-layer effective
  scale buffers (`router.scale * 1/sqrt(D)` pre-folded at open), the
  `router_topk_select_k8` kernel's semantics computed on the host
  (softmax over the top-k only, times `per_expert_scale`; the logits
  readback already exists for the expert `pread`, so a GPU select would
  buy nothing), the INT8 shared-expert branch (three parity-tested INT8
  GEMV dispatches + the activation multiply, NOT the fused
  `shared_int8_gate_up_act_simd` kernel — same math, one more dispatch),
  the FFN sandwich tail (`h += rmsnorm(h1 + h2, post_ffn)`), and the
  per-layer `layer_scalar` multiply (a port-local `scalar_mul_fp16`
  kernel in `utility.metal` — Swift folds this into its unvendored
  `fused_layer_tail`). Proven end to end WITHOUT a network by
  `build_synthetic_gemma4_real_install` (`crates/repack`), which pushes
  a deterministic in-memory safetensors blob with the real naming, INT8
  router/MLP, and BF16 norms through the REAL
  `write_gemma4_install` pipeline; `crates/runtime/tests/
  real_forward_gemma4.rs` decodes it deterministically with a flat GPU
  allocation count, and `crates/cli/tests/real_generation.rs` drives
  `mference-check --prompt` over it. The `Gemma4Quant` bits-override map
  must come from the checkpoint's own config (`parse_gemma4_quantization`);
  8-bit routed experts are rejected (the MoE decode kernels are
  int4-only).
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
