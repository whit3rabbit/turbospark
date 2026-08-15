# Deviations

Every place this port's behavior deliberately differs from, or falls short
of, the Swift Mference original or the full scope of the original port
roadmap (now condensed into `ROADMAP.md`'s Port record section; that file
carries the forward roadmap). Organized by the port roadmap's phases. "Scaffolded" means the pure/testable
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
- **This port has a quality harness; the Swift original has none.** Not a
  deviation from a behavior, an addition on an axis Swift publishes
  nothing for: no perplexity, no KL divergence, no golden output. So no
  row in `docs/BENCHMARKS.md`'s Quality section is or can be a SWIFT
  parity claim, and most of them (perplexity, digests, constrained-cache
  arm, sensitivity curve) are this port measured against its own past.
  The exception is the cross-engine KLD, which does have an external
  reference, just not Swift: `crates/bench/tests/logit_dump.rs` plus
  `scripts/kld.py` run mlx-lm over the same corpus, the same token ids,
  and the same quantized checkpoint the install was repacked from. It
  reports 0.0264 mean nats at 95.6% top-1 agreement against an
  intra-engine floor of 0.0352 (mlx-lm across its own two forward
  shapes), so this port agrees with mlx-lm more closely than mlx-lm
  agrees with itself. Deliberately not wired into `cargo test`: it needs
  a 14.6 GB reference checkpoint and a Python environment, and mlx-lm
  runs in a `uv` ephemeral env so it never enters this workspace's
  dependency graph. Three consequences worth recording next to the
  parity tables.
  Upstream's memory-pressure acceptance proof, byte-identical output at
  unchanged throughput under a constrained working set, HOLDS ON BOTH
  FAMILIES as of 2026-08-08, and the gate asserts it rather than freezing
  a digest per slot count. It did not hold on Gemma before then: that
  flow's misses-first routed-slot ordering (a port-local overlap
  optimization, now removed) fed phase 2's reduce, and FP addition is not
  associative, so halving the expert cache changed bytes. The deeper
  problem that ordering caused, and the reason it went rather than being
  documented around, is AGENTS.md Gotcha 27. And the gate's sensitivity is
  measured rather than assumed: `quality_sensitivity.rs` shifts one
  quantization level in a strided subset of routed experts and puts the
  detection floor between 0.0015% and 0.0122% of expert bytes.
- **Process-entry-point ownership resolved as `crates/cli`.** An earlier
  note reserved the name `turbospark-entrypoint` and left it unbuilt pending
  a decision. That decision is now made and documented in `AGENTS.md`
  Gotcha 7: `crates/cli`.
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
- **MEASURED against Swift, 2026-08-07: decode is at parity, within 1
  percent, on the same machine and the same install.** Full numbers,
  provenance, and caveats in `docs/BENCHMARKS.md`; reproduce with
  `scripts/parity.sh`. Apple M4 Max 36 GB, AC power, frozen
  `real-generation-v1` protocol, 16 expert-cache slots (both engines'
  default), Swift `1bb585c` against this port at `ef4e953` plus the
  partial-ranking sampler change, both opening `~/models/gemma4.gturbo`
  (written by THIS port's repack, which the Swift CLI accepts unmodified
  under its default `.fullSha256` policy). Swift 41.1 / 38.6 / 34.3 tok/s
  against this port's 40.7 / 38.4 / 34.6 on short / medium / long, ratios
  0.99 / 1.00 / 1.01.
  An EARLIER run of the same script measured 0.64 to 0.67 and recorded the
  cause as unattributed. The cause was `selection::select` full-sorting
  the whole candidate domain to rank it: ~18.9 ms per token at Gemma 4's
  V=262144, against a ~25 ms forward pass. Both truncation steps only ever
  keep a PREFIX of the ranked order, so with `top_k` on, everything past
  rank 64 was sorted and discarded. `truncation::rank_top_k` replaces the
  sort with `select_nth_unstable_by` plus a sort of the surviving 64:
  2.05 ms, output byte-identical, decode 25.5 -> 39.6 tok/s on a fixed
  prompt. A follow-up (same day) moved the hot path onto thread-local
  scratch buffers and ranks unnormalized `exp(s - max)` values with `u32`
  indices instead of materializing the full probability vector (division
  by the positive normalizer is monotone, so the order is identical;
  normalized probabilities are computed on the fly only for the ranked
  prefix and the survivors, bit-identical divisions). Ranking 2.41 ->
  1.98 ms/call in isolation, +0.24 tok/s mean over three interleaved
  sampled pairs (+0.6%), output md5-identical on both the greedy and
  sampled smokes. The dominant remaining sampler cost is the full-vocab
  f64 exp pass for the top-p normalizer, kept deliberately: an f32 exp
  would change the sampled stream. The reason it survived so long is that it is NOT INSIDE
  `produce`, so no phase bucket, GPU-busy attribution, or dispatch ranking
  in this repo could see it (see `crates/selection/CLAUDE.md` and
  AGENTS.md Gotcha 23), and the greedy smoke passes `--temperature
  0.0001`, which is not exactly zero and so paid the same sort. The
  `MTLSharedEvent` overlap named here as a suspect was never the cause; it
  is already bought another way (below) and measured at +4.0%.
  Prefill is now the only measured gap and is a scope difference, not a
  regression: Swift chunks at 128 and costs about 5.1 s fixed plus 7.5 ms
  per prompt token; this port has no fixed cost and 21.4 ms per token
  (independently reproducing its own 2026-08-06 attribution), so this port
  is faster to first token under roughly 350 prompt tokens and slower above
  it. MEMORY, the other half of the design's premise, is AT OR BETTER THAN
  parity in the same session: peak `phys_footprint` 2,108-2,182 MiB here
  against Swift's 2,217-2,235 on the same install, so this port holds the
  same ~2 GB working set on a 26B model with a 14 GB install and does it
  in 2 to 5 percent less.
- **Upstream experiment inventory cross-reference.** The Swift original is
  public at <https://github.com/drumih/turbo-fieldfare>; its
  `docs/experiments/EXPERIMENT_INVENTORY.md` catalogs the 103 experiments
  behind the design this port inherits. Checked against that inventory on
  2026-08-07, this port's own measurements corroborate every upstream
  finding it touches. Upstream absolute numbers are from 8 GB M2-class
  hardware and do not transfer to the M4 Max rows measured here; the
  directions and ratios do.

  | Upstream experiment | Upstream result | This port | Status |
  |---|---|---|---|
  | Bounded `pread` beats `mmap` for cold experts (2.79 vs 9.88 ms) | production | `PreadExpertStreamer` is the production path | ported |
  | LFU expert cache (io 72.6 -> 64.8 ms) | production | LFU/LRU policy in `crates/streaming` | ported |
  | Split attention (4.1x end to end at 4K) | production | `chunks_for` split-KV: 11.7-15.8x kernel-isolated at 4096, +25% decode at 800 context (Phase 6 below) | ported, confirmed |
  | FP16 SWA ring buffer (saved 575-591 MiB) | production | static KV accounting: 922.7 - 319.8 MB = 575.0 MiB saved (`docs/BENCHMARKING.md`) | ported, exact match |
  | Expert prefetch / RDADVISE policies | rejected | deliberately unwired, citing upstream's own dead end (Phase 7 below) | scope-consistent |
  | OUT-01 one-pass sampling + Top-64 (0.377 -> 5.86-5.89 tok/s) | production | same bug class found independently 2026-08-07: host full sort 18.9 ms -> `rank_top_k` 2.05 ms, decode 25.5 -> 39.6 tok/s. Host-side; the GPU `sample` kernel stays descoped | parallel finding |
  | PF-02 chunk-128 prefill (121 tok: 15.80 -> 9.34 s) | production | descoped with the tile kernels; the one measured gap (21.4 vs 7.5 ms per prompt token) | descoped |
  | PF-12 staged affine MPP, PF-17 Apple10 TensorOps | production | descoped with the same tile pipeline | descoped |
  | 24/32 expert-cache slots | conditional (memory cost) | 32 slots: +15% decode for +1.5 GB (`docs/BENCHMARKS.md`) | matches |
  | Quantized KV K4/V4 (delta-NLL +0.015197) | rejected (quality) | never attempted; KV stays FP16 | consistent |
  | DEC-03 persistent multi-threadgroup MoE (cb2 239 -> 60 ms) | production | inherited: the vendored `moe.metal` decode pair IS that kernel family | ported |

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
- **PLAIN TEXT chat also renders through the checkpoint's own template
  now**, not just tool chat: chat framing is a property of the checkpoint
  and the special-token dialect is not evidence about it (AGENTS.md Gotcha
  41). `chat_template.rs`'s per-dialect renderers are the fallback for a
  checkpoint that ships none. The template is read from either HF
  convention -- a standalone `chat_template.jinja` or
  `tokenizer_config.json`'s older `chat_template` key -- because the real
  installs here split across both and the split does not follow family.
  The two renders differ by nothing but `trim`: the dialect renderers strip
  surrounding whitespace from content unconditionally, a template only where
  it says `| trim`. Three of the four gated families' templates do, so their
  frozen rows are untouched; Qwen3-30B-A3B's does not, so its user turn
  regained the protocol prompt's trailing newline and its row was re-frozen
  (perplexity 14.7576 -> 14.5988, both digests). All of it is pinned per
  family in `crates/tokenizer/tests/installed_template.rs`.
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

- **`gdn.metal` is compiled from a CONCATENATED source string, and its
  input-projection specialization is unused.** The fused four-way input
  projection calls `dequant_int4_gemv_simd_body`, a `static inline` in
  `dequant_int4.metal`; the Swift build concatenates every shader module
  into one library, so the call resolves there. This port compiles one
  library per file, so `crates/gpu/src/gdn.rs`'s `SOURCE` is
  `concat!(include_str!("shaders/dequant_int4.metal"), "\n",
  include_str!("shaders/gdn.metal"))` — one `&'static str` with one stable
  address, which is what the address-keyed pipeline cache needs (AGENTS.md
  Gotcha 8), at the cost of compiling the INT4 kernels a second time in
  this library. `gdn_parity.rs`'s fused-vs-four-separate-GEMVs test asserts
  the result is BIT-identical, which is the check that makes the trick
  safe. Separately, the kernel's function constants 90-94 (constant-folded
  row counts for the decode shape) are declared but never specialized:
  every dispatch takes its shape at runtime. Swift measured ~102 GB/s
  unspecialized against ~141 GB/s specialized for a plain INT4 GEMV, so
  this is a real throughput item, deliberately left for the session that
  measures the real checkpoint.
- **The dispatched kernel set (grown well past the original six):**
  `rmsnorm.metal`'s `rmsnorm_no_scale`, `rms_norm_bf16w`, and both
  `_perhead` norm variants; `rope.metal`'s `rope_proportional_neox`
  (which with `rotated_pairs = head_dim/2` IS default full-head NeoX);
  `utility.metal`'s port-local `logit_softcap_fp16`, elementwise
  activation/residual kernels, and port-local `scalar_mul_fp16`;
  `embed_lookup_int4`; `dequant_int4.metal`'s `dequant_int4_gemv_simd`
  and `dequant_int8.metal`'s `dequant_int8_gemv_simd` (both with
  offset-bound resident variants); `router_gemv_gemma4_r4`;
  the port-local GGUF set (`dequant_q8_0.metal`'s `dequant_q8_0_gemv_simd`
  and `embed_lookup_q8_0`, `dequant_q4_k.metal`'s
  `dequant_q4_k_gemv_simd`, and `moe_gguf.metal`'s Q8_0 routed-expert
  decode pair -- port-local because the Swift engine has no GGUF intake,
  so there is no upstream kernel to mirror);
  `attention.metal`'s two-pass split-KV decode attention
  (`attention_decode_partial` + `attention_decode_combine`,
  `crates/gpu/src/attention_decode.rs`, incl. SWA `kv_start` and the
  `FC_ATTN_RING_CAP` KV ring); and `moe.metal`'s decode pair
  (`moe_phase1_gate_up_act_u16load` + `moe_phase2_down_reduce_k8`). Each
  is parity-tested against the matching `turbospark_compute` reference on
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
  has no matching `turbospark_compute` reference — its frequency divisor is
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
  **`gdn.metal` is now vendored, dispatched, and parity-tested; only
  `dsv4.metal` is still missing, and it is NOT a throughput tradeoff like
  the three above** — there is no working fallback path for it. All eight
  GDN kernels (fused four-way input projection, causal depthwise conv in
  decode/prefill/tail-update form, per-head q/k norm, the gated delta
  recurrence in decode and prefill form, and the gated output norm) are
  vendored verbatim into `crates/gpu/src/shaders/gdn.metal`, dispatched
  from `crates/gpu/src/gdn.rs`, and checked against the FP32 reference
  `turbospark_compute::GdnReference` in `crates/gpu/tests/gdn_parity.rs`.
  `Dsv4StateManager` still allocates real buffers with nothing computing
  a compressed-attention read into or out of them, on GPU or CPU
  (`turbospark_compute` has no CSA/HCA reference either), so
  `RealForwardRunner` still rejects mask 3/4 outright: DeepSeek-V4-Flash
  (`full_attention_layer_mask` values `{0,3,4}` — zero full-attention
  layers at all) cannot run through this port at any speed until those
  kernels exist, unlike the three descoped-for-throughput items above.
  Qwen 3.6 (`full_attention_layer_mask` mostly `2`) is no longer in that
  category: its decode flow is wired (see Phase 7 below).
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
  reference in `turbospark_compute` to verify a port against — it is a
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
  GEMV, SWA ring addressing dispatched). `GdnStateManager` is now
  production-wired too — the Qwen 3.6 decode flow owns one per open and
  advances both its delta-rule state and its conv tail in place every
  token (`RealQwenState`, `real_forward_qwen.rs`). Only `Dsv4StateManager`
  stays unwired (its compute kernels are unported — see below); each of
  the three is also exercised directly against
  real Metal buffers: `crates/gpu/tests/kv_cache.rs` (9 tests),
  `crates/gpu/tests/gdn_state.rs` (3 tests), and
  `crates/gpu/tests/dsv4_state.rs` (6 tests). What's still missing for
  DSV4 specifically is the compute *kernels* that would read and write
  through its manager (`dsv4.metal`) — the managers are
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

- **Decode rate control is PORT-LOCAL and has no Swift counterpart
  (ROADMAP Phase P2, 2026-08-09).** The Swift engine has no rate limiter,
  no thermal adaptation and no power profiles, so nothing here mirrors an
  upstream design and there is no parity question to answer. It is off by
  default in every binary: with `RateControl::default()` (both fields
  `None`) `raw_completion::decode` executes the identical statement
  sequence it did before the feature landed, so every published throughput
  and memory number remains a number about the same code path.
  Two limits worth stating rather than discovering. The cap paces DECODE
  only -- prefill runs one `produce` call per prompt token through a
  different loop and is untouched, so a long prompt still draws full power
  for its whole prefill. And the thermal ladder's constants (serious ->
  10 tok/s, critical -> 5) are heuristics chosen ahead of the measurement,
  not results: Phase P2's joules-per-token gate has not been run yet
  (it needs sudo), so nothing here is yet evidence that the efficiency
  profile buys energy rather than merely spending longer.

- **Qwen 3.6: PROVEN on the real 35B-A3B checkpoint (2026-08-07).**
  `mlx-community/Qwen3.6-35B-A3B-4bit` repacks through
  `write_qwen36_install_streamed` in 19 minutes into an 18 GB install and
  generates coherent chat-formatted answers via `turbospark-check`: greedy
  and sampled both stay coherent for 400 tokens, and a short question
  stops on `EndOfTurn` with a correct answer, so the ChatML stop set
  resolves. The whole frozen bench protocol reaches `endOfTurn` on all
  three cases. Everything below that predates this and describes the
  synthetic milestone; the two gaps that remain are throughput tuning and
  a memory-oracle row, not correctness.

  Peak `phys_footprint` is **1,587-1,610 MiB**, roughly 500 MiB UNDER
  Gemma 4 26B-A4B on the same machine despite the larger install. That is
  the hybrid paying off: 30 of 40 layers are linear and carry no KV at
  all, only ~2 MiB of fixed GDN state each, so context growth touches 10
  layers instead of 30. `crates/bench/tests/qwen36_memory_oracle.rs`
  holds the row (ceiling 1,700, floor 25.0) and the reasoning behind
  both; it is a separate target from `memory_oracle.rs` because the
  footprint assertion is against a whole-session peak and two families
  cannot share one.

  Decode is **32.6-38.0 tok/s** on the frozen protocol (M4 Max, AC, 16
  slots, warmup discarded, two independent readings agreeing to 0.13
  tok/s on the slowest case). Prefill is ~37 tok/s.

  That is up from 20.0-23.1 measured on the same install hours earlier,
  and the difference is entirely `selection`'s full `rank_indices` sort
  over V=248320 (AGENTS.md Gotcha 23). The tell was arithmetic:
  `MFERENCE_PHASES=1` accounted for only ~29 ms of a ~45 ms token. After
  the fix the same report accounts for 23.7 ms of a 23.3 ms token, i.e.
  all of it. Buckets, warm, 32 slots, before and after:

  | bucket | before | after |
  |---|---|---|
  | gpu wait (layer cb1) | 18.8-20.8 | 14.0-14.5 |
  | expert io (pread) | 5.7-5.8 | 5.0-6.2 |
  | encode + logit readback | 1.1 | 1.0 |
  | final wait (end of token) | 1.0-1.1 | 1.0 |
  | routed bind+upload | 0.8 | 0.66 |
  | router readback+topk | 0.34 | 0.25 |
  | routed cb retire | 0.00 | 0.00 |
  | **unaccounted (host sampler)** | **~15.5** | **~0** |

  `gpu wait` fell without any GPU-side change because the host used to
  spend 15 ms sorting while the GPU idled; removing that exposes less
  waiting, not less work. The two zeros are BY CONSTRUCTION, not a
  measurement failure: the Qwen flow has none of the three overlap seams,
  so there is no hit-CB and no pipelined routed CB to charge time to.
  Expert cache hit rate is 73.9% at 32 slots against 256 experts.
- **Qwen 3.6: wired end to end against a SYNTHETIC install only.** The
  decode flow (`crates/runtime/src/real_forward_qwen.rs` plus
  `real_forward_qwen_attn.rs`) runs both Qwen layer kinds -- gated
  DeltaNet (mask 2) through the eight `gdn.metal` kernels, and gated full
  attention (mask 1) through `split_q_gate_fp16` + per-head q/k norms +
  `rope_neox_subdim` + `sigmoid_gate_mul_fp16` -- with a sigmoid-gated
  shared expert and streamed INT4 routed experts on every layer.
  `RealForwardRunner::open` selects it from `ArchConfig.family`, not from
  tensor naming (both real families carry
  `language_model.model.embed_tokens.weight`). Proven by
  `crates/runtime/tests/real_forward_qwen.rs` (8 tests) against
  `turbospark_repack::build_synthetic_qwen36_real_install`, and end to end
  through `turbospark-check`. What is NOT done: no real ~20 GB Qwen
  checkpoint has been downloaded or repacked, so there is no throughput
  number and no memory-oracle row. Weights in the fixture are
  deterministic but untrained, so no test asserts on generated text
  (AGENTS.md Gotcha 12).
- **The synthetic Qwen fixture quantizes the shared expert INT8; the real
  checkpoint quantizes it INT4.** `mlx-community/Qwen3.6-35B-A3B-4bit`
  puts only `mlp.gate` (the router) and `mlp.shared_expert_gate` at 8
  bits; `mlp.shared_expert.{gate,up,down}_proj` take the global 4-bit
  default, so `manifest_quant`'s `sharedExpert` slot reads 4 there and 8
  on the fixture. Nothing needs changing -- `encode_gemv_any` picks the
  INT8 or INT4 resident GEMV from the resident index's dtype tag, not
  from the family -- but it does mean the fixture never exercises the
  INT4 shared-expert dispatch that the real install will take on every
  layer. `validate_quant` accepts both widths for that slot.
- **`parse_qwen36_config` exists and is proven against the production
  field values, but nothing has been repacked with it.**
  `crates/repack/src/qwen36_config.rs` parses
  `mlx-community/Qwen3.6-35B-A3B-4bit`'s `config.json` into an
  `ArchConfig` equal to `model_io::qwen36_35b_a3b()` field for field
  (`crates/repack/tests/qwen36_config.rs`, offline, real values inline).
  `write_qwen36_install_streamed` is the guarded streamed writer for it.
  The multi-shard walk itself is still UNPROVEN on this family: the only
  test that covers it is `tests/qwen36_checkpoint_network.rs`, which is
  `#[ignore]`d behind a ~20.4 GB download and has not been run. The
  synthetic fixture is one in-memory shard, so a companion tensor living
  in a different shard than its weight is not exercised by the default
  suite.
- **The Qwen path's router top-k runs on the host with UNIT scales.** It
  reuses `router_topk_gemma4` -- top-k by score, softmax over the selected
  scores only -- passing an all-ones `per_expert_scale`, because Qwen has
  neither `router.scale` nor `router.per_expert_scale` and
  `arch.router_scaled` is false. The INT8 router GEMV kernel takes an
  effective-scale vector regardless, so the flow binds a BF16 `[hidden]`
  buffer of ones built once at open. Same kernel, same semantics, no
  Qwen-specific router kernel.
- **The Qwen path has NEITHER of the two command-buffer overlap seams the
  Gemma path carries.** `MFERENCE_SHARED_CB` and
  `MFERENCE_ROUTED_PIPELINE` are throughput-only (the latter measured at
  ~+2.5% on Gemma), and each one is a correctness-sensitive reordering
  that needs its own identical-output A/B to land. A third,
  `MFERENCE_HIT_CB`, was removed on 2026-08-08: its reordering was not
  output-neutral after all, which is AGENTS.md Gotcha 27. The Qwen flow is the plain shape: one
  command buffer per layer up to the router, host readback plus expert
  `pread`, one buffer for the MoE tail. `PhaseCounters` still fills in, so
  `MFERENCE_PHASES=1` works; `pipeline_wait_nanos` stays zero by
  construction.
- **GDN chunked prefill is parity-tested but unwired.** `gdn.metal`'s
  `gdn_conv_mix_prefill`, `gdn_conv_tail_update`, and
  `gdn_delta_step_prefill` are dispatched and checked against a
  seven-row-chunk-equals-seven-decode-steps test
  (`crates/gpu/tests/gdn_parity.rs`), including the `T < K-1`
  ordered-shift tail path. Nothing in `crates/runtime` calls them: this
  port's prefill is token-at-a-time everywhere (see the chunked-prefill
  entry in Phase 6), and wiring GDN's chunked form alone would not change
  that. The kernels are there so the next session can.
- **The output head is skipped on non-final prompt tokens; Swift runs it
  on every one.** Swift's `RawCompletion.swift` off-mode prefill loop
  (`case .off:`) calls `producer.produce(token:position:into:)` per
  prompt token and, like this port's loop did, reads only the last
  result: `LogitProducer` has no way to say "these logits are going in
  the bin". This port adds one, a defaulted `produce_prefill` on the
  trait (`crates/runtime/src/producer.rs`) that `run_raw_completion`
  calls for every prompt token but the last. The default delegates
  straight to `produce`, so `ScriptedLogitProducer`, `ChunkedPrefillRunner`,
  and `crates/server`'s `ChatModel::with_producer` are all unchanged
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
  Worth 6-7% of prefill wall clock on a 2252-token prompt (paired deltas
  -6.0 / -6.7 / -10.4%, interleaved against a stashed pre-change binary,
  the third pair's before-arm a high outlier). Swift's chunked prefill avoids the same
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
    decode from 34.8 to 42.6 tok/s. Those two absolute numbers are 32-slot
    figures on an unrecorded prompt and sampling setting, so they are the
    A/B deltas only; the 42.6 is NOT comparable to the 16-slot protocol
    numbers in `docs/BENCHMARKS.md`, which explains the slot and sampler
    axes that separate them.
    Swift's phase-1-hit CB WAS ported here and has been REMOVED
    (2026-08-08). It ordered a layer's slots cache MISSES first then hits
    so the resident hits' phase-1 GEMV could ride its own command buffer
    before the `pread`. Measured on the real 26B checkpoint (M4 Max, 32
    slots, 5 interleaved pairs) it was worth +0.42 tok/s, ~+1%, with
    generated text md5-identical in all ten runs -- which is exactly why it
    survived: the A/B that qualified it held the expert cache constant, and
    the reordering's real dependency was on cache STATE. Two warm greedy
    runs of one prompt in one process could differ. Slots are now
    dispatched in the router's own ranking. See AGENTS.md Gotcha 27; the
    standing check is `crates/bench/tests/gguf_nondeterminism_probe.rs`.
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
    md5-identical across all ten runs, across the seam grid as it stood
    then ({ROUTED_PIPELINE, SHARED_CB, HIT_CB}), and across pipeline
    states at 16 slots.
    Expert prefetch/speculation stays deliberately unwired: the Swift
    original benched every shape to a dead end (cross-layer predictor
    Jaccard 0.039 with 7% copied-prediction hits, rejected before
    implementation; the previous-token predictor can never issue a read
    because per-layer private caches keep last token's experts resident;
    `MFERENCE_SPEC_PREFETCH=prefetch` measured as a no-op; RDADVISE "no
    stable production policy", off by default -- see upstream's
    `docs/experiments/summaries/03-expert-cache-prediction-and-layout.md`
    and `04-rdadvise.md` at
    <https://github.com/drumih/turbo-fieldfare/tree/main/docs/experiments>).
    `crates/streaming`'s speculative APIs exist for
    parity and stay uncalled by the runtime on purpose.
    The OFFLINE sibling of that idea, a domain-restricted expert set
    (profile which experts a coding corpus routes to, then prune or
    pre-warm that set), was measured to its own dead end on 2026-08-08:
    covering 95% of a layer's routed mass takes a mean 66.8 of 128
    experts on a coding corpus (66.5 general, Jaccard 0.455 between hot
    sets), so routing is domain-tilted, not domain-concentrated, and both
    the pruned install and the pinned warm set lose to the LFU cache.
    Method, numbers and the standing decision: `docs/EXPERT_ROUTING.md`
    (ROADMAP dead end 11). The instrument stays wired as a diagnostic
    (`MFERENCE_ROUTER_HIST` on `RealForwardRunner`, analyzed by
    `scripts/router_hist.py`).

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
    little left to win here), routed bind ~1%,
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
    `turbospark_compute::apply_streamed_routed`'s residual-fused form (that
    function bakes the residual add into the combine step; this runner
    adds the residual itself afterward, through the same sandwich-norm
    step the dense path uses, so the two FFN branches share that
    structure) — a deliberate simplification, not a missed reuse
    opportunity. Also simplified: the SYNTHETIC short-name MoE flow has
    no separate dense/shared FFN branch summed in alongside the routed
    one; the real-checkpoint flow (see the real Gemma 4 pipeline entry
    below) does compute both and add them.
  The weights come from a real `.gturbo` install
  (`turbospark_repack`'s `build_synthetic_gemma4_install` for dense,
  `build_synthetic_gemma4_moe_install` for MoE — both using the named
  resident-tensor writer, `write_gturbo_install_with_resident_index` — see
  Phase 8 below) loaded through the real `turbospark_model_io`
  manifest/resident-index/`ResidentBuffer` loaders, unmodified. The
  weights themselves are deterministic but NOT trained (no trained
  `.gturbo` checkpoint is available in this environment) — so the
  generated *text* is not semantically meaningful; what's real is the
  pipeline that produces it. `RealForwardRunner::open` accepts full
  attention (mask 1) and sliding-window (mask 0) layers everywhere,
  linear (2) layers under the `qwen36` family only, and rejects
  compressed (3/4) layers outright, whose kernels are unported — so
  Gemma 4 and Qwen 3.6 both pass while DeepSeek-V4-Flash remains blocked
  on DSV4.
  **The `llama` family (ROADMAP Phase M2) was wired for the MoE half of
  its architecture string only, and ROADMAP M4 CLOSED THAT SPLIT.**
  `general.architecture = "llama"` is both Mixtral and dense
  Llama 2/3.x/Mistral, and nothing in the string says which a file is --
  only `expert_count` does. Both halves now run: Mixtral-style checkpoints
  through the routed path, and dense ones through `families/llama/dense.rs`
  (`mlp.gate_proj` / `mlp.up_proj` / `silu_mul` / `mlp.down_proj`, all
  through `encode_gemv_any`, no new kernel), with real published Mistral
  7B and TinyLlama installs and coherent smokes. A dense install does NOT
  keep the memory ceiling and never could -- nothing streams -- so its
  parity row says so. What is still refused by name is a checkpoint
  shipping `rope_freqs.weight`, Llama 3.1's LEARNED per-dimension RoPE
  scaling: dropping it yields a model wrong only past the training length,
  which no smoke test reaches. ROADMAP M5's `rope_neox_freqs` is the kernel
  shape that would take it, and nothing wires it up.
  **`gpt-oss` (ROADMAP M5) is the current scaffolded-not-wired family, and
  the split is between KERNELS and FLOW rather than between halves of an
  architecture.** Landed and parity-tested: `ModelFamily::GptOss` with its
  baseline and name table, the MXFP4 routed pair (the first block type here
  with BOTH routed phases and no resident GEMV), per-projection biases
  (`bias_add_bf16_fp16`), attention sinks in the split-KV combine pass,
  YaRN rope scaling (`yarn_frequencies` + `rope_neox_freqs`), and the
  clamped SwiGLU with per-expert biases inside the MXFP4 phase-1 kernel.
  NOT landed: `crates/runtime/src/families/gptoss/`, which assembles them,
  and the Harmony stop set. `RealForwardRunner::open` therefore refuses the
  family BY NAME and lists those four, rather than falling through to a
  neighbouring flow -- each of them produces fluent WRONG output rather
  than an error, which is the failure mode `crates/runtime/CLAUDE.md`
  Gotcha 11 exists for. No real install has been walked; the fixture that
  exists (`SyntheticGgufShape::mxfp4()`) is Gemma-shaped with gpt-oss's
  block types, so it proves the routed pair dispatches inside a forward
  pass and nothing about the gpt-oss layer.
  A separate caveat that is about the CHECKPOINT rather than the port:
  Mixtral is a COARSE MoE (8 experts of 108.9 MiB against Gemma's 128 of
  ~3.2 MiB), and the expert slot cache is `slots x layers x expert_stride`,
  so it cannot stream at any useful slot count -- 54.5 GiB at 16 slots,
  and the whole 27.2 GiB expert table at 8. It runs and it is correct; it
  is not what this engine is for. See AGENTS.md Gotcha 36.
  **`qwen3moe` (Qwen3-30B-A3B) is that same flow's fine-grained checkpoint,
  and it is a FOURTH family rather than a fourth flow.** The layer graph is
  identical to Mixtral's, so `crates/runtime/src/families/llama/` serves
  both and the two differences are keyed on `ArchConfig.family` inside it:
  Qwen3 norms q and k per head before RoPE, and its RMS epsilon is 1e-6
  against 1e-5. It needed NO new kernels (Q4_K and Q6_K were already
  executable) and no new chat dialect (ChatML). 128 experts of 2.5 MiB
  puts its slot cache at 1.90 GiB at 16 slots, so unlike Mixtral it is a
  checkpoint the memory oracle and the quality gate can meaningfully
  measure. What the fixture tests CANNOT see is stated where they live
  (`crates/runtime/tests/real_forward_qwen3moe.rs`): the norms' ORDER
  relative to RoPE, and the epsilon's own value, are not separable on
  untrained weights and belong to the cross-engine gate.
  **`gptOss` (gpt-oss-20b) is a SIXTH family and a FIFTH FLOW** (ROADMAP
  M5), and it is the first bring-up here where the answer to "same graph?"
  was no. `crates/runtime/src/families/gptoss/` runs plain GQA plus four
  things that are each inside the layer: a bias on all four projections
  (a separate `bias_add_bf16_fp16` pass, applied BEFORE RoPE), YaRN rope
  off a precomputed per-pair frequency table with a magnitude scale of
  1.3465736, attention SINKS (one learned logit per query head, added to
  the softmax denominator alone), and an alternating 128-token window on
  the EVEN layers. Its MXFP4 routed experts additionally carry a clamped
  SwiGLU and per-expert biases, both of which ride inside the routed pair
  and are named nowhere in the flow. The real 12.1 GB checkpoint streams,
  opens, and generates coherent chat-formatted answers; both gates are
  frozen (perplexity 12.0801, peak 5,421 MiB). What the fixture tests
  CANNOT see is stated where they live
  (`crates/runtime/tests/real_forward_gptoss.rs`): passing a literal 1.0
  for the YaRN magnitude scale leaves every one of them green, because
  that scale is a function of the rope factor alone and no config
  difference isolates it from the frequency table. Its value and argument
  order are pinned by a unit test; the end-to-end property belonged to
  the cross-engine gate, which HAS now been run and passed (2026-08-12):
  0.00978 mean nats at 97.5% top-1 against llama.cpp on the identical
  bytes, under ggml's own 0.01181 Metal/CPU backend floor, which is what
  says the mscale value and the sink placement are right
  (`docs/BENCHMARKS.md`).
  HARMONY'S CHANNELS ARE NOW DECODED (this used to be one of two stated
  limitations here). `StructuredDecoder` reads the
  `<|channel|>HEADER<|message|>BODY<|end|>` frame as its own small state
  machine (the start/end token pair every other dialect keys on cannot
  express it, since `<|channel|>` has no closing counterpart), and reports
  the `final` channel as content and everything else as REASONING. It is
  the one arm here that emits its thought channel rather than discarding
  it, because `gpt-oss` puts most of its generated tokens there. The
  server surfaces the split as OpenAI `reasoning_content` and, through
  `anyllm_translate`'s existing mapping, as an Anthropic `thinking` block
  ahead of the text block; `turbospark-check` prints reasoning to STDERR
  and the answer to stdout, and only the answer enters `--chat` history,
  which is what Harmony's own convention asks for. Harmony TOOL CALLS
  remain undecoded and are a separate item: a call is framed as a
  recipient inside the channel header rather than as the bracketing token
  pair the decoder's tool contract describes, so its `commentary` body
  comes back as reasoning rather than as a parsed call. What has not
  changed: there is no fallback chat renderer for the dialect at all;
  Harmony's real
  template is 17 KB of system preamble, reasoning-effort knob and
  TypeScript tool namespace, so `apply_dialect_chat_template` refuses by
  name instead of inventing a partial frame (AGENTS.md Gotcha 41).
  Proven end to end by
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
  scripted default: `crates/bench`'s `turbospark-bench` runs the real
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
  needs `TURBOSPARK_GEMMA4_INSTALL_DIR`) is the memory oracle: it asserts
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
  checkpoint's own chat template, falling back to the dialect renderer only
  when it ships none (`add_bos` false either way, since a real template
  emits its own `<bos>`/`<s>` mark), and `--chat`
  (`crates/cli/src/chat.rs`) is the interactive REPL ported from
  `MferenceCLI/Run.swift`'s `runChat`: `/clear`, `/history`, `/quit`,
  `/exit`, `--system` seeding the opening turn, per-turn window fitting
  through `turbospark-window-fit` (the Swift `trimChatHistory` contract), and
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

- **GGUF ingestion: BOTH REAL PUBLISHED FILES RUN (ROADMAP Phase G, Stage 1
  and Stage 2 complete; Q4_0 is the one block type still refused).**
  A GGUF file now walks all the way to a `.gturbo` install
  (`gguf_header.rs` parses the v3 header, `gguf_names.rs` maps tensor names
  onto the canonical HF-style ones the rest of the pipeline speaks,
  `gguf_config.rs` rebuilds an `ArchConfig` from the metadata, and
  `gguf_checkpoint/` walks and writes). Quantized bytes are carried
  through VERBATIM -- there is no quantization step on them, because GGUF
  blocks arrive already quantized, which is the lossless-repack rule taken
  literally. The one exception is the resident F32 core, which is
  transcoded rather than carried; see below.
  **WHAT RUNS IS DECIDED PER BLOCK TYPE, NOT PER FORMAT (2026-08-08).**
  Q8_0, Q4_K and Q6_K installs open and decode on real Metal hardware; a
  Q4_0 install is written and then refused, by name. GGUF blocks are
  interleaved (the scale lives inside the block) where this port's affine
  kernels read three separate planes at group 64, so a block type needs its
  own kernels rather than a flag. Q8_0 and Q4_K each have all three, every
  one with a `turbospark_compute` reference and a parity test: a resident
  GEMV (`dequant_q8_0_gemv_simd`, `dequant_q4_k_gemv_simd`), an embedding
  lookup (`embed_lookup_q8_0`, `embed_lookup_q4_k`), and one of
  `moe_gguf.metal`'s two decode pairs for the streamed routed experts.
  Q6_K has a resident GEMV and no siblings ON PURPOSE, which is all any
  real file asks of it: Qwen's Q4_K_M carries exactly one Q6_K tensor and
  it is `output.weight`. An install that put Q6_K in an expert would pass
  the manifest gate and fail at the routed dispatch, by name. Q4_0 has
  nothing.
  **A MIXED install is the normal case, not an edge one.** `Q4_K_M` means
  Q4_K experts and embedding, Q8_0 attention and shared experts, one Q6_K
  tensor, so the block type is read PER TENSOR at each dispatch site rather
  than decided once at open (`RoutedBlobLayout`, `encode_embed_any`). The
  Qwen decode flow called the affine MoE and embedding kernels directly and
  so could only ever have run an MLX install; that was invisible while no
  Qwen GGUF could open. `SyntheticGgufShape::k_quant()` builds the mixture
  as a fixture and `crates/runtime/tests/gguf_install_refused.rs` decodes
  it.
  **The scoping lesson, since the roadmap got it wrong:** a resident GEMV
  is the small half of the job. Routed experts and the embedding table go
  through their own kernels (`moe_phase1_gate_up_act_u16load` /
  `moe_phase2_down_reduce_k8` and `embed_lookup_int4`), all three affine,
  so lifting the refusals cost three more kernels rather than a validation
  edit.
  The two refusals remain independent and asserted, but they now gate on a
  set rather than on the format: `model_io::validate_quant` compares the
  manifest's `ggmlType` against `model_io::EXECUTABLE_GGUF_TYPES`, and
  `RealForwardRunner::open` compares the resident index's dtype tags
  (6 = Q8_0, 7 = Q4_K, 8 = Q6_K, 9 = Q4_0) against the runtime's own copy
  of that set, which believes the bytes rather than a claim about them.
  Both directions are asserted in
  `crates/runtime/tests/gguf_install_refused.rs` -- the backstop after
  deliberately forging the manifest past the first -- and the same file
  decodes a Q8_0 install rather than only opening one.
  **A THIRD SET EXISTS AND IS DELIBERATELY WIDER THAN BOTH: what the
  header PARSER can size.** `gguf_header.rs::ggml_type_block` also lists
  Q2_K, Q3_K, Q5_0, Q5_1, Q5_K, IQ3_XXS, IQ4_NL and IQ4_XS, none of which
  has a kernel and none of which is executable. They are there so a
  candidate checkpoint can be HEADER-PROBED for scoping (ROADMAP Phase S)
  without downloading it, and adding a row buys nothing but parsing. The
  consequence to know is that the walk will now happily WRITE an install of
  such a type and the refusal lands later, at load, by name -- which is not
  new behaviour but the shape Q4_0 has always had. Widening executability
  still means landing kernels and moving the two sets above together.
  **Verified against the real published files, not just fixtures.**
  `crates/repack/tests/gguf_checkpoint_network.rs` reads the header of
  `ggml-org`'s Gemma 4 26B-A4B Q8_0 and Qwen 3.6 35B-A3B Q4_K_M (a few MB
  off a 20-27 GB file, about 4 seconds each) and checks three things: the
  parser agrees with what llama.cpp's converter writes, every tensor name
  in both files maps, and the `ArchConfig` derived from GGUF metadata is
  EQUAL to the one this port's own MLX-derived install declares. That last
  one is the strongest check available, because the two sides share no
  code and no input.
  The same file's three `scopes_phase_s_*` cases survey candidate sub-4-bit
  checkpoints at the same cost, and print a ggml type histogram in BYTES.
  Read that histogram's UNSIZED rows before its percentages: a type with no
  `ggml_type_block` row is one this port cannot ingest, which on a mixed
  file is usually the routed experts, so rendering it as zero bytes sorts
  the most important row to the bottom of the share column. It did exactly
  that once, making an imatrix file look 76% Q8_0 when its experts are
  IQ3_XXS. Cross-check the sized total against the published file size
  before believing any row.
  Beyond kernels, two things a GGUF install needed before it could run,
  both discovered by the above rather than assumed, are now DONE rather
  than outstanding: its norms are F32 where the runtime wants BF16, and its
  router is F32 where the runtime wants INT8. Both are transcoded at repack
  time (`gguf_checkpoint/transcode.rs::transcode_f32`), which was chosen over an F32
  path on a measurement rather than on the tradeoff the roadmap
  anticipated: the norms are upcast BF16 and narrow back with zero bit
  loss, and INT8-transcoding the router leaves top-1 routing unchanged with
  every top-8 flip inside quantization noise
  (`crates/repack/tests/gguf_f32_transcode_network.rs`). An F32 path would
  have cost two more kernels, each needing its own CPU reference and parity
  test, to buy nothing. A converter that did not upcast is not rejected --
  BF16 is where the values have to go regardless -- but every value that
  loses bits is counted and reported. A third is now
  settled rather than outstanding: Gemma's routed
  gate/up arrive FUSED in one tensor, and gate is the FIRST half, measured
  against the real file rather than assumed (`FUSED_GATE_FIRST`,
  `crates/repack/tests/gguf_fused_gate_network.rs`). A
  fourth is settled the same way: Q4_K's reference is held against the real
  `Qwen3.6-35B-A3B-Q4_K_M` by correlation
  (`crates/repack/tests/gguf_q4_k_network.rs`), because a decoder and the
  fixture quantizer feeding it come from one mental model and can agree
  while both are wrong.
  **A fifth is not a format question at all, and it is the one that decided
  whether Qwen runs: SOURCE CONVENTIONS.** llama.cpp interleaves Qwen's V
  heads where the mlx-community checkpoint keeps them contiguous (GGUF head
  `h` is MLX head `2h` for the first half and `2(h - heads/2) + 1` for the
  second), and it stores `-exp(A_log)` in the `ssm_a` slot where the
  install carries `A_log`. Both are undone at repack time by
  `v_head_axis` + `apply_source_convention{,_bytes}` in
  `gguf_checkpoint/transcode.rs`, never at runtime and never in a kernel, on the same
  rule that settled the F32 transcode. The quantized tensors are permuted
  AS BYTES, so the lossless-repack rule is untouched: a logical row is a
  contiguous block run and a head-wide column group is a whole number of
  blocks, checked rather than assumed. See AGENTS.md Gotcha 33 and
  `crates/repack/CLAUDE.md` Gotcha 7 for the measurements and the trap (the
  convention belongs to an AXIS, so it reaches eight tensors and not the
  three a BF16 probe could compare).
  **THE PHASE G GATE IS MET FOR BOTH FAMILIES (2026-08-08).** The real
  published `ggml-org/gemma-4-26B-A4B-it-GGUF` Q8_0 and
  `ggml-org/Qwen3.6-35B-A3B-GGUF` Q4_K_M checkpoints each install and
  decode coherent text, greedy and sampled
  (`crates/repack/tests/gguf_install_network.rs`,
  `gguf_qwen_install_network.rs`). Neither needed a local copy of the 20-27
  GB file: the walk streams over HTTP a layer at a time, so only the ~25 GB
  and ~19 GB installs are written, in about 21-24 minutes each. Gemma's
  resident BF16 core is BIT-IDENTICAL to the MLX install's, which is the
  strongest statement available that the name mapping and the F32 transcode
  are right (`gguf_norm_convention_probe.rs`); Qwen's norms are too, while
  its gated-DeltaNet tensors are the convention gap above
  (`gguf_qwen_core_probe.rs`, `gguf_qwen_quant_probe.rs`,
  `gguf_qwen_convention_patch.rs`).
  **The last clause, "within its quant's expected degradation", was closed
  2026-08-08 by measuring it** (`scripts/kld_llamacpp.py`,
  `scripts/llamacpp_logits.c`; numbers and method in `docs/BENCHMARKS.md`).
  It read as a defect first: Gemma's GGUF install scores 39.8808 perplexity
  against the MLX install's 37.4176, i.e. the higher-precision side 6.6%
  WORSE. llama.cpp b10310 on the same GGUF bytes reads 39.8541, 0.067% from
  this port, at 0.00845 mean nats and 98.2% top-1 agreement -- against
  0.57748 nats and 78.0% for a real weight difference (this port's INT4
  install against the same reference). So Q8_0 genuinely loses on this
  corpus and the GGUF path is faithful. The trap that cost the most is
  recorded in `docs/BENCHMARKS.md` and AGENTS.md Gotcha 34: the same
  measurement against llama.cpp on CPU reads 0.05838 and looks like a real
  gap, because ggml's own CPU and Metal paths disagree by 0.05510 on this
  model. Match the backend, not just the bytes. Not
  scaffolded, not attempted: a local-file `RangeSource` (the walk streams
  over HTTP like the safetensors one), Q4_K or Q6_K routed experts beyond
  what the real files use, and any
  Q5_K/Q3_K/Q2_K/i-quant block sizes (`ggml_type_block` answers only for
  types whose size was read off the spec, and names the rest in its error
  rather than guessing). Also not done: a memory-oracle or quality-gate
  ROW for either GGUF install, since they differ in resident bytes from the
  MLX ones and the chip rows are keyed on the chip rather than the install,
  so pointing an install var at a GGUF artifact asserts the MLX goldens
  against it -- a diagnostic use, not a supported one.
- **Byte-exact `.gturbo` directory assembly: implemented**
  (`gturbo_writer.rs`'s `write_gturbo_install`): given already-quantized
  tensor bytes, writes `packed_experts/layer_NN.bin` blobs (matching
  `turbospark_model_io::PackedExpertsLayout`'s exact per-expert sub-tensor
  layout, zero-padded to `expert_stride`), `packed_experts/layout.json`, a
  minimal valid `model_weights.bin` (a real `ResidentIndexHeader` plus a
  raw tensor region), and `manifest.json` with computed per-file SHA-256.
  Round-trip tested: write an install, then read every part of it back
  through `turbospark_model_io::load_manifest`/`load_packed_experts_layout`/
  `load_resident_index` and `verify_install_full_sha256`, all of which
  pass. Ranged-read planning (`RangeSource`, HTTP-backed for real use) and
  per-row int4/int8 quantization (reusing `turbospark_compute`'s quantizer)
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
  `turbospark_model_io::load_manifest`/`load_resident_index` loaders —
  finished in under 100 seconds end to end. This is the piece the
  `repack`'s own module docs and the old port roadmap's Phase 8 called
  "NOT implemented" for exactly this reason; it is now implemented, for
  the Llama-family naming convention.
  `crates/repack/tests/hf_checkpoint_network.rs` carries this test,
  `#[ignore]`d (a real, unpredictable-duration network download has no
  place in the default `cargo test --workspace` suite); run it explicitly
  with `cargo test -p turbospark-repack --test hf_checkpoint_network --
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
  data region matching `turbospark_model_io::resident_index`'s exact reader
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
  minutes end to end), validated by every `turbospark_model_io` loader,
  and generates REAL COHERENT TEXT through `turbospark-check`: a
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
  `crates/repack/src/gemma4_checkpoint/config.rs` parses a Gemma 4 `config.json`
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
  `turbospark-check --prompt` over it. The `Gemma4Quant` bits-override map
  must come from the checkpoint's own config (`parse_gemma4_quantization`);
  8-bit routed experts are rejected (the MoE decode kernels are
  int4-only).
- **The server's real backend is macOS-only and serves one request at a
  time.** `RealChatModel` (`crates/server/src/real_model.rs`, gated the
  same way `crates/gpu` is) drives a real `RealForwardRunner` against a
  `.gturbo` install: `turbospark-server --model <install-dir>`. A runner
  costs a multi-gigabyte mapping plus a Metal pipeline compile to open and
  takes `&mut self`, so there is one per process behind a `Mutex` and
  concurrent requests queue on it (each waiter pinning a tokio blocking
  thread). A client that disconnects mid-stream does not abort generation.
  `ScriptedChatModel` remains the portable backend the integration tests
  drive; on non-macOS it is the only one.
- **Tailnet bind: implemented, `--model` mode only.** `--bind
  loopback|tailnet` (default loopback) ports Swift's `ServerBindMode`:
  `tailnet` runs `tailscale ip -4` (spawned directly, no shell) and binds
  the single reported address, requiring it to be a dotted-quad inside
  100.64.0.0/10. Zero, several, out-of-range, IPv6, or malformed output all
  fail rather than falling back, so there is no path from `tailnet` to a
  wildcard or LAN bind. Resolution runs BEFORE the model is opened, so a
  missing Tailscale does not first cost a multi-gigabyte map and a pipeline
  compile. Deviation from Swift: the flag is absent from the portable
  `<tokenizer-dir>` scripted mode (which Swift does not have), which always
  binds loopback. Like Swift's, it is not authentication: the server has no
  auth and no TLS, so access is governed entirely by the Tailnet ACL.
- The server's sampling knob surface is narrower than the CLI's: no
  request-settable `top_k` (it defaults to 64, the CLI's default, because
  `ShapingConfig` rejects a `top_p` below 1.0 when `top_k` is 0, which
  would 400 every plain OpenAI request carrying only `top_p`) and no
  `repetition_penalty` (fixed at its identity value). That matches plain
  OpenAI Chat Completions' request shape rather than
  `turbospark-invocation`'s fuller option set.
- **Anthropic `POST /v1/messages`: implemented, text and tool calling, and
  an addition rather than a port.** Swift's server has no such endpoint. It exists here
  because `turbospark-server` took a dependency on `anyllm_translate`
  (crates.io 0.16, default features: pure, IO-free, no axum, no reqwest),
  which also supplies the OpenAI wire types `/v1/chat/completions` now uses
  in place of hand-rolled structs. An Anthropic request is translated into
  the OpenAI request the existing path already understands, run through the
  shared generation core, and translated back. Net effect: Anthropic-native
  clients (Claude Code, the Anthropic SDKs) need no proxy in front.

  **Tool calling works, both endpoints, streaming and not.** A request's
  `tools` are rendered into the prompt through the checkpoint's own
  `chat_template.jinja` (`encode_generic_tool_chat`), the generated tokens
  go through `StructuredAssistantDecoder`, and a parsed call comes back as
  OpenAI `tool_calls` / Anthropic `tool_use`. Verified end to end against
  the real Gemma 4 install: a call, a `tool_result` round trip, and the
  streamed `content_block_start` / `input_json_delta` / `content_block_stop`
  sequence. Three limits worth stating:

  - `tool_choice` is accepted and IGNORED. Nothing forces or forbids a call.
  - A streamed call arrives as ONE chunk carrying id, name, and the whole
    argument string, not as the argument fragments a remote OpenAI backend
    emits. The decoder only yields a call once its closing marker arrives,
    so there are no fragments to stream; `StreamingTranslator` handles the
    single-chunk shape and the Anthropic event structure is unaffected.
  - Tool-call ids are a per-response counter (`toolu_0`, `toolu_1`), unique
    within an assistant turn but not across a conversation. Both templates
    match a `tool` turn against the tool calls of the message immediately
    before it, so that is enough.

  If the dialect parser rejects what the model wrote (prose that merely
  looks like a call), the request does NOT fail: the decoder is abandoned
  and the rest of the run is emitted as plain text.

  What is still dropped: image and document content blocks, and `thinking`.
  A replayed `thinking` block is dropped from the prompt rather than
  rendered as assistant prose (`ChatMessage::effective_text` would fall
  back to `reasoning_content`; `handler::visible_text` does not). Of these,
  only `thinking` and document blocks appear on the
  `x-anyllm-degradation` response header -- `compute_request_warnings` has
  no notion of a dropped image, so that one is silent. A message with
  neither text nor tool calls is dropped rather than rendered as an empty
  turn; one with tool calls and no text is kept.

  The crate's own `middleware` feature is deliberately NOT enabled: it
  forwards over `reqwest` to a `backend_url` (this server's backend is
  in-process, so that would be a loopback hop to itself) and it depends on
  axum 0.8 against this crate's 0.7.
- **`GET /v1/models`: implemented, one entry.** There is one backend per
  process, so the list has exactly one model and requests are never routed
  on the `model` field -- whatever name a request carries is echoed back.
  The advertised id is the install directory's name (`gemma4.gturbo`) for
  `RealChatModel` and `scripted` for `ScriptedChatModel`; `manifest.json`
  has no model-name field to read instead. Swift's server has no such
  endpoint either.
- **Neither new endpoint adds authentication.** Same posture as
  `/v1/chat/completions`: loopback by default, and under `--bind tailnet`
  the Tailnet ACL remains the only access control.

## Not ported at all

- `crates/tokenizer`'s `Sha256Verifier` used `CommonCrypto`; this port
  uses the `sha2` crate (RustCrypto) instead. Behaviorally equivalent, not
  a gap, but worth noting as a dependency substitution alongside the ones
  above.
- Anything under `MferenceApp` (the Mac GUI), `MferenceDecodeService`/
  `MferenceDecodeProtocol` (app-side XPC), chat-history compression UI
  behavior, and document extraction — permanently out of scope per
  `ROADMAP.md`.
