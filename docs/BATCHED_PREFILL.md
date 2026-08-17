# Batched prefill: what it would take, and what it would buy

Scope, not a plan of record. Written 2026-08-16. Every number is cited to
the page that owns it except two, prefill's own dispatch ranking and
prefill's own expert-union and miss rates, both of which were measured for
this document because no page had them and borrowing decode's gave the
wrong answer twice.

> **CORRECTED 2026-08-16, same day, before any of it was built.** The first
> draft's largest single term was wrong: it charged the sequential arm for
> expert TOUCHES where the slot cache charges MISSES, and it took its
> `union(M)` from a decode measurement that excludes prefill. Measured on
> the real install, the union is LARGER than what the sequential path
> already loads in seven of eight (slots, M) cells, and the one cell where
> it wins is a cell where `ExpertCache::plan` aborts the process. The
> `pread` bucket does not batch at all. The corrected numbers are below;
> the composite went from ~2.1x to ~1.5x and step 1 alone from "most of
> ~1.75x" to ~1.1x. Both mistakes were the same one this document's
> attention section already warns about: a share measured on decode does
> not transfer to prefill.

Read this before proposing work on prefill throughput. It says which
kernels are actually needed (two, not ten), why the descoped tile pipeline
is **not** a prerequisite, and why the arithmetic that made speculative
decoding marginal does not carry over to this.

## The gap

Prefill runs one full forward pass per prompt token. Measured on this
machine, real Gemma 4 install:

| | this port | Swift |
| --- | ---: | ---: |
| per prompt token | 21.4 ms | 5.1 s fixed + 7.5 ms |
| 3,015-token prompt | 64.6 s | 27.5 s |

The 21.4 is the 2026-08-07 parity figure and is the one to quote. Re-run
2026-08-16 on the frozen `long-synthesis` prompt it reads 23.0 ms/token at
32 slots and 28.9 at 16, on a machine that was not quiet (Gotcha 43), which
is why the sections below use SHARES from those runs and not their totals.

It is also where long-prompt energy goes: 86% of the `long-synthesis`
case's joules are prefill on `qwen3moe` (`docs/POWER_BASELINE.md`).

This is the **only** measured gap against Swift. Decode is at parity
(`docs/BENCHMARKS.md`), and `docs/DECODE_BUDGET.md` retired the three
decode-side items that used to sit above it.

It is not a scaling problem: split-KV made prefill nearly flat in context
(cb1 5.39 ms at 21 tokens against 6.28 averaged over a 2,252-token prompt),
so the 64.6 s is 3,015 x a fixed cost, not a curve.

**It IS substantially attention, and the number every other page here would
give you is the wrong one.** The dispatch ranking in
`docs/SPECULATIVE_DECODING.md` puts attention at 2.3% of GPU busy; that is
a DECODE ranking taken at short context. Measured for prefill below, it is
22.3% and the single largest dispatch. Do not carry a decode share into a
prefill argument -- this document's first draft did, and reached the
opposite conclusion about what to build.

## What already exists

More than a reader expects, and the missing piece is narrower than
"chunked prefill" suggests.

| piece | where | state |
| --- | --- | --- |
| chunk span arithmetic | `crates/core/src/prefill.rs`, `chunk_sizing.rs` | done, tested |
| the driver loop | `runtime::run_raw_completion_chunked` | done, tested |
| the producer trait | `runtime::ChunkedPrefillRunner` | defined |
| scratch sizing + allocation | `crates/gpu/src/prefill_scratch.rs` | done, allocates real buffers, undispatched |
| batched INT4 GEMM | `crates/gpu/src/dequant_int4_batch.rs` | done, parity-tested, M <= 16 |
| batched MoE pair | -- | **missing** |
| `RealForwardRunner: ChunkedPrefillRunner` | `families/gemma4/mod.rs` | done 2026-08-16, Gemma 4 only |

`ScriptedLogitProducer` was the trait's only implementor until step 1
landed, which is what made the loop testable and also why nothing had
noticed the gap.

> `crates/core/src/prefill.rs`'s module doc still says the GPU-side scratch
> sizing "is not ported". It was, into `crates/gpu/src/prefill_scratch.rs`.
> Stale comment, worth fixing when someone is next in that file.

## The two kernels, not ten

The handoff names "the batched MoE pair" and it is easy to read that as ten
kernels across five block types in `moe_gguf.metal`. Against the install
that matters it is **two**.

`~/models/gemma4.gturbo` declares `routedExpert.scheme: "affine"`, so it
dispatches `moe.metal`'s `moe_phase1_gate_up_act_u16load` and
`moe_phase2_down_reduce_k8` -- not the GGUF pairs. It is the pinned install:
the standing smoke, the memory oracle, the sensitivity proof and the
cross-engine KL all run on it. So the first cut needs one batched form of
each of those two, and the GGUF and MXFP4 pairs follow later, per block
type, exactly as they were added in the first place.

**Swift already chose a shape for this and it is vendored but unported**:
`dsv4_prefill_moe_phase1_pairs_int2`, `dsv4_prefill_moe_down_pairs_int2`
and `dsv4_prefill_moe_reduce_pairs_k6` (`moe.metal:1138-1227`). Read their
signatures before designing a new one. The idea worth stealing is
`DSV4PrefillRoute`: a flat `(token, rank, local_slot)` route list with a
`route_start`/`route_count` window, so the kernel iterates ROUTES rather
than tokens and a chunk's ragged routing becomes one dense dispatch. That
is the whole reason a chunk of M tokens does not need M x top_k separate
launches.

**Read `dequant_int4_batch.rs`'s header before writing either.** Two
optimizations were tried on the batched GEMV and both lost on every shape:
threadgroup staging of `x` (expert shape 0.36 -> 0.55 at M=16) and register
blocking over rows (0.44 -> 0.79 at M=8). One finding twice -- the register
file cannot hold an M-wide activation tile and threadgroup barriers cost
more than they save. `MAX_BATCH_ROWS = 16` is a register-file limit, not a
clamp.

## What it would buy

Measured for this document, 2026-08-16, real Gemma 4 install, the frozen
`long-synthesis` prompt (3,015 tokens), `--max-new 8` so 99.7% of the
divisor is prefill, `MFERENCE_PHASES=1`. Two arms, because the expert
bucket is a strong function of the slot count and `auto` resolves to 32 on
this machine:

| bucket | 32 slots | share | 16 slots | share | batches? |
| --- | ---: | ---: | ---: | ---: | --- |
| gpu wait (layer cb1) | 13.20 | 57.5% | 13.61 | 47.1% | only its scheduling gap |
| expert io (`pread`) | 7.41 | 32.3% | 12.39 | 42.8% | **no -- see below** |
| encode + logit readback | 1.35 | 5.9% | 1.68 | 5.8% | commit overhead only |
| routed bind + upload | 0.58 | 2.5% | 0.82 | 2.8% | not at step 1 |
| routed cb retire | 0.25 | 1.1% | 0.26 | 0.9% | not at step 1 |
| router readback + top-k | 0.14 | 0.6% | 0.14 | 0.5% | yes, divides by M |
| final wait | 0.02 | 0.1% | 0.03 | 0.1% | yes |
| **total ms per prompt token** | **22.95** | | **28.93** | | |
| GPU busy (cb1 / routed / final) | 6.55 / 3.77 / 0.00 | | 6.76 / 3.98 / 0.00 | | |

The device-time row is the one to read against `gpu wait`: 10.32 ms/token
of real GPU time at 32 slots against 13.47 ms of host waiting, so the
**scheduling gap is 3.15 ms/token (13.7%)** and everything else in that
bucket is work the GPU actually did.

### The expert `pread` does not batch, and the first draft said it did

The reasoning that failed is worth keeping because it is seductive: a chunk
of M tokens reads the UNION of their routes rather than the sum, the decode
union table says 16 consecutive tokens touch ~38 of 128 experts per layer,
and 38 against `16 x 8 = 128` looks like a 3.3x cut.

**128 is requests, not reads.** The slot cache already deduplicates them,
and `scripts/router_window.py`'s own docstring says so in as many words
("the sequential arm's true cost is misses, not touches"). Measured, same
runs as the table above (misses per layer per token, and prefill's OWN
union from `MFERENCE_ROUTER_TRACE=1` analysed at `skip=0` so prefill is
INCLUDED -- the script's default excludes it):

| | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| prefill `union(M)`, distinct experts per layer | 13.2 | 20.4 | 29.9 | 41.5 |
| sequential loads over the window, 32 slots (1.507/layer/token) | 3.0 | 6.0 | 12.1 | 24.1 |
| sequential loads over the window, 16 slots (3.044/layer/token) | 6.1 | 12.2 | 24.4 | 48.7 |

The sequential arm is already BELOW the union in seven of eight cells. The
union can only ever recover intra-chunk eviction re-reads, and at 32 slots
there are none to recover: 24.1 loads against 41.5 distinct means most of
the window's experts were resident before the window began.

The one cell where the union wins is 16 slots at M=16, 41.5 against 48.7,
a 15% saving -- and `ExpertCache::plan_if_possible` **asserts**
`experts.len() <= slot_count`, so requesting 41.5 experts against 16 slots
aborts the process. The mechanism is legal only where it is useless:

| slot count | largest M with `union(M) <= slots` | union saving there |
| ---: | ---: | ---: |
| 16 | 2 | none (13.2 against 6.1) |
| 32 | 8 | none (29.9 against 12.1) |

Prefill's union also runs ABOVE decode's (41.5 against 37.8-39.4 at M=16),
which is the second half of the same lesson: the decode table was borrowed
into a prefill argument, exactly as the 2.3% attention share was.

**The real lever on this bucket is the slot count, and it is already
shipped.** Going 16 to 32 slots takes prefill from 87.4 s to 69.3 s on this
prompt, a 1.26x, larger than anything step 1 projects; `auto` resolves to
32 on this machine (`crates/runtime` Gotcha 13). A cache big enough to hold
`union(16) = 41.5` would take it further and is not reachable:
`ALLOWED_CACHE_SLOTS` stops at 32 and every oracle ceiling is frozen.

### Why the speculative-decoding arithmetic does not apply

`docs/SPECULATIVE_DECODING.md` concluded "about 1.1x at best" from a
composite built out of the same terms, so it is worth saying plainly why
that verdict does not transfer.

**Speculative decoding divides by an accept length and prefill does not.**
A verify pass of M drafted tokens keeps only the accepted prefix, so its
cost is amortized over ~4 of 8 tokens and the whole question is whether a
drafter predicts well enough. A prefill chunk of M tokens has no drafter,
no rejection and no probability: all M tokens are known, every one of them
is kept, and the divisor is exactly M. The two share `c(M)` and `union(M)`
and nothing else.

The second difference is the workload. That page's compute split is
DECODE's, where attention is 2.3% and there is nothing to amortize across
tokens that the slot cache is not already amortizing. Prefill's own split
is below, and attention alone is ten times larger in it.

So: read that page for `c(M)`, `union(M)` and the register-file dead ends,
which are properties of the kernels and transfer. Do not read its verdict.

### Prefill's own GPU split, measured

`MFERENCE_PHASES=1 MFERENCE_DISPATCH_PROFILE=1`, real Gemma 4 install, the
frozen `long-synthesis` prompt (3,014 tokens), `--max-new 8`, 32 slots.
3,022 forward passes, so 99.7% of the divisor is prefill. Absolute times
are inflated by the profiling mode (one encoder per dispatch, every buffer
waited on); **only the shares transfer**. Three command buffers, summing to
28.466 ms/token:

| work | ms/token | share of GPU | batches how? |
| --- | ---: | ---: | --- |
| `attention_decode_partial` + `_combine` | 6.351 | **22.3%** | M queries share one KV read |
| `dequant_int4_gemv_simd` (cb1 120x, shared 90x) | 8.453 | 29.7% | weight amortization, `c(M)` |
| `moe_phase1` + `moe_phase2` | 6.683 | 23.5% | weight amortization, and the pair to write |
| norms (six kinds) | 4.673 | 16.4% | **no** |
| `router_gemv_gemma4_r4` | 0.952 | 3.3% | weight amortization |
| residual add, rope, gelu/scalar mul, embed | 1.354 | 4.8% | **no** |

Two readings worth keeping. **Attention is the largest single dispatch in
prefill**, at ten times its decode share, because a prompt token at
position 1,500 attends to 1,500 keys where a decode ranking taken on a
21-token prompt attends to almost none. And **21.2% of prefill GPU work has
no weights to amortize** (norms plus the elementwise tail), which is the
prefill analogue of the 19% floor `docs/SPECULATIVE_DECODING.md` measured
for decode -- two independent measurements of different workloads landing
two points apart, which is the reason to believe either.

### Composing it

`c(M)` is measured for the GEMV shapes (`docs/SPECULATIVE_DECODING.md`):
0.36 on the expert shape at M=16, 0.67 to 0.78 on the big projections; call
it 0.6 weighted. At 32 slots, as a fraction of the 22.95 ms per prompt
token measured above:

| term | now | after step 1 | fully batched |
| --- | ---: | ---: | ---: |
| expert `pread` | 32.3% | 32.3% | 32.3% |
| GPU device time, cb1 (attention, GEMV, norms, router) | 28.5% | 28.5% | ~15.5% |
| GPU device time, routed pair | 16.4% | 16.4% | ~8.5% |
| host scheduling gap | 13.7% | ~5% | ~5% |
| encode + logit readback | 5.9% | ~5% | ~5% |
| bind + retire + router readback | 4.2% | ~3.5% | ~0.5% |
| **total** | **100%** | **~91%** | **~67%** |

**Predicted: step 1 alone about 1.1x, the whole program about 1.5x**,
landing near 15 ms per prompt token against Swift's 7.5. Parity would need
3.06x and is not reachable while the `pread` bucket is a third of prefill
and does not batch.

**Step 1 then measured 1.22x, and the column above is wrong in an
instructive direction. See "Step 1, measured" below.**

Three terms are soft, and the direction of each is worth knowing. `c(M)`
for the routed pair is extrapolated from the INT4 GEMV proxy (the real
phase-1/phase-2 kernels have no batched form to measure). Attention's share
of cb1 is folded in at the doc's earlier 22.3% dispatch ranking rather than
measured again per arm. And the step-1 column assumes the scheduling gap is
mostly phase A's, which the driver's own measurement will settle -- that is
why step 1 exists before any kernel.

### Step 1, measured

Landed and measured 2026-08-16, same day as the correction above. Real
Gemma 4 install, `long-synthesis` (3,015 prompt tokens), `--max-new 8`, 32
slots, `MFERENCE_PREFILL_CHUNK=128`, three interleaved pairs after a
discarded warmup on a machine that was NOT quiet (Gotcha 43):

| pair | sequential | chunked | |
| ---: | ---: | ---: | ---: |
| 1 | 55.84 s | 44.74 s | 1.248x |
| 2 | 57.48 s | 49.55 s | 1.160x |
| 3 | 56.12 s | 44.41 s | 1.264x |

Pair 2 is slower on BOTH arms and is the contention outlier; pairs 1 and 3
agree to 1.3%. Call it **1.22x mean, 1.25x on the two clean pairs.**

Where it came from, pair 3, ms per prompt token:

| bucket | sequential | chunked |
| --- | ---: | ---: |
| gpu wait (layer cb1) | 13.09 | **6.50** |
| routed cb retire | 0.25 | **2.75** |
| expert io (`pread`) | 3.10 | 3.68 |
| encode + logit readback | 1.34 | 1.07 |
| routed bind + upload | 0.63 | 0.55 |
| router readback + top-k | 0.14 | 0.14 |
| final wait | 0.03 | 0.01 |
| **total** | **18.57** | **14.70** |
| GPU busy, cb1 / routed | 6.57 / 3.77 | 6.25 / 3.86 |

**The prediction was 1.1x and the reality is 1.22x, and the model was wrong
about what a per-token wait costs.** The composite above treats GPU device
time as an irreducible floor and only the measured 13.7% "scheduling gap"
as recoverable. But blocking on `cb1` thirty times per token does not merely
cost the gap: it drains the pipeline and refills it, and the GPU is idle
across the seam. `cb1` device time is flat (6.57 against 6.25 ms/token) while
the WAIT on it halves, which is the shape of a fill/drain cost rather than a
queueing one. Reach for that explanation before a kernel one the next time a
sync-removal beats its own arithmetic.

**Read the `routed cb retire` row before quoting the `gpu wait` one.** Some
of the 6.59 ms/token that left `gpu wait` did not leave the run: 2.50 of it
reappears as retire, because the sequential path retires a layer's routed
buffer after the NEXT layer's `cb1` wait (where it has long finished) and
the chunk driver retires it inside the token loop (where it has not). The
honest saving is the total row, 3.87 ms/token.

**Third independent confirmation that the union does nothing**: the expert
cache hit rate is 81.2% sequential against 81.4% chunked, on identical
routes. Batching tokens did not deduplicate one expert read.

Extrapolating the rest from here rather than from the original composite:
10.11 of the remaining 14.70 ms/token is GPU device time, ~21% of which is
norms and elementwise with nothing to amortize. Taking the rest at
`c(M) ~ 0.6` puts the fully batched figure near 12 ms/token, i.e. **~1.5x
against the sequential baseline** -- unchanged from the corrected composite,
by coincidence rather than by construction, since step 1 over-delivered and
the kernels have correspondingly less left to take.

**A note on why this is still worth building at ~1.5x.** Prefill is the
only measured gap against Swift, it is 86% of a long prompt's joules
(`docs/POWER_BASELINE.md`), and 1.5x on the 69.3 s this prompt spends in
prefill is 23 s of wall clock. What it is NOT is a path to parity, and the
first draft's 2.1x implied one.

## The attention fork

The question is whether a chunk needs the descoped tile kernels
(`attention_prefill_causal_tiled`, `prefill.metal`'s 16-kernel pipeline --
`DEVIATIONS.md`) or whether it can run the existing split-KV
`attention_decode` per token inside the chunk.

**It can run per token, and it should, first** -- but for a weaker reason
than a 2.3% share would have given. Attention is 22.3% of prefill GPU work
and batching it is worth ~7 points of the ~33 the whole change is worth.
That is real and it is not a prerequisite: a chunk whose attention is still
per-token gets ~1.4x, and steps 2 to 4 below are independent of it.

So this is one phase followed by an optional one, rather than a fork:

- **Phase A**, per-token attention inside a batched chunk. ~1.4x. No new
  attention kernel, no touching the descoped pipeline.
- **Phase B**, batched attention. Takes ~1.4x to ~1.5x. And it does NOT
  need `prefill.metal`'s 16-kernel tile pipeline -- what it needs is one
  kernel where M queries share a KV read, which is a natural widening of
  `attention_decode_partial`'s existing split-KV structure (it already
  reads KV in chunks; the change is to hold M query rows per chunk instead
  of one). Scope it against that kernel, not against the descoped pipeline,
  whose 1,202 lines cover embed/norm/rope/router/MoE as well and are
  descoped for reasons that still hold.

## Order of work

0. ~~Measure prefill's own dispatch ranking.~~ **Done, 2026-08-16**; it is
   the table above, and it moved the design (attention 22.3%, not 2.3%).

1. ~~**`RealForwardRunner: ChunkedPrefillRunner`, looping the EXISTING
   per-token kernels inside each layer.**~~ **Done, 2026-08-16, measured at
   1.22x** (see "Step 1, measured"). Gemma 4 only; every other family is
   refused by name rather than falling back to the sequential loop, because
   a caller that asked for chunked prefill and quietly got the old path
   would measure the old engine and report it as the new one.

   Reachable through `MFERENCE_PREFILL_CHUNK=<tokens>` on
   `turbospark-check`, an A/B seam beside `MFERENCE_SHARED_CB` and
   `MFERENCE_ROUTED_PIPELINE`. **`--prefill-chunk` already exists in
   `crates/invocation` and is deliberately still wired to nothing**: it
   defaults to `Fixed(128)`, so wiring it would turn chunked prefill on by
   default, and this phase has one family and one measured install behind
   it. That flag is what the seam becomes, not a second mechanism.

   What it does, and the rest of this entry is the design rather than a
   plan: per chunk it advances M KV rows, the position by M, and
   commit-and-waits exactly as the per-token path does
   (`crates/runtime/CLAUDE.md` Gotcha 2 states the analogous contract for
   `produce_prefill`). On a family with recurrent state the GDN chain is
   sequential by definition and steps M times inside the chunk.

   **Split the layer at the point the flow already commits, and batch only
   the first half.** Phase A -- norms, q/k/v/o GEMVs, per-head norms, RoPE,
   attention, the residual adds, the pre-FFN norms and the router GEMV --
   is encoded for all M tokens into ONE command buffer per layer, so the
   per-layer blocking wait is paid once per chunk instead of once per
   token. Phase B -- router readback, host top-k, plan, `pread`, bind,
   phase 1/2, sandwich norms, `layer_scalar` -- stays per token, because
   `RoutedBlobsBuffer` is a single host-written argument buffer and
   `routing_w` / `moe_acts` are single-row, so M tokens cannot share a
   command buffer there without M copies of all three AND all M tokens'
   experts resident at once.

   **The expert `pread` union is NOT the mechanism.** An earlier draft of
   this list had one `plan_experts_cached` over the chunk's union replacing
   M per-token plans, worth "25.2% of prefill cut by 3.3x". Measured, it is
   worth nothing at 32 slots and aborts the process at M=16 (see "The
   expert `pread` does not batch" above). What step 1 collects is the
   scheduling gap and the per-token commit overhead, ~9% of prefill.

   Only M-row buffers are needed: `x`, `router_logits_f32`, `dense_x` and
   `routed_x`. Every other scratch stays single-row -- a serial compute
   encoder runs dispatches in order, so reuse inside a chunk is correct.
   That deliberately forgoes GPU parallelism across tokens in phase A;
   the parallelism arrives with the M-row kernels in steps 2 to 4.

   Gated by byte-identity against the sequential path, not by coherence:
   `crates/runtime/tests/real_forward_gemma4_chunked.rs` on a synthetic
   real-named install (six cases, mutation-checked), and on the real
   install the greedy and sampled smokes reproduce their frozen digests
   (`b2f16611...`, `0c383ac0...`) at chunk spans 32, 128 and 512.

2. **Batched `moe_phase1_gate_up_act_u16load`**, against
   `dsv4_prefill_moe_phase1_pairs_int2`'s route-list shape. Parity must be
   EXACT against M separate calls, mutation-checked, per this repo's habit.
3. **Batched `moe_phase2_down_reduce_k8`.** The reduce order is a
   correctness constraint, not a style choice (AGENTS.md Gotcha 27): a
   batched phase 2 must reduce each token's slots in the router's ranking,
   independently per token, or output becomes a function of the chunk
   boundary.
4. **Batched attention** (Phase B above), if the measured 1.4x is not
   enough. One kernel, widening `attention_decode_partial` to hold M query
   rows per KV chunk. Not the descoped tile pipeline.

   Note steps 2 and 3 carry a constraint step 1 does not: a batched routed
   pair needs all M tokens' experts resident at once, so M is capped at the
   largest value with `union(M) <= slot_count`. Measured above, that is
   M=8 at 32 slots and M=2 at 16. `c(M)` is worse at small M (worse than
   1.0 at M=2), so 32 slots is a precondition for step 3 paying at all.
5. **Widen to the GGUF pairs** per block type, if and when a GGUF MoE
   install is the one being optimized.

## Gates this owes

Prefill changes the logits the first decoded token is sampled from, so it
is a numerics change however carefully it is written.

- All three real-model gates plus `quality_gate`, per family touched.
- **Byte-identity is the bar, not coherence.** A chunked prefill must
  produce the same tokens as the sequential one; `accept_length_probe.rs`
  established the shape of that check for speculative decoding (compare
  against a NON-chunked reference run, never two chunked runs against each
  other, which passes when both are wrong the same way).
- Chunk size must not change the output. If it does, the reduce order in
  step 3 is wrong.
- The memory oracle: `prefill_scratch.rs` allocates real buffers sized by
  chunk, and every oracle ceiling is frozen.

## Which families, and in what order

`gemma4` first: it is the only MoE install on disk, it is the pinned one,
and it is where the 21.4 ms was measured. `ternary27b` and
`museglimmer-30b` are DENSE -- the MoE pair does not apply to them at all,
their prefill lever is the plain GEMV alone, and steps 2 and 3 above buy
them nothing. Every other family needs a 5 to 25 minute re-stream before it
can be gated (`docs/MODELS.md`).
