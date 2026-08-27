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

If you are about to propose work on prefill throughput, read this first.
It says which kernels are actually needed (two, not ten), why the descoped
tile pipeline is **not** a prerequisite, and why the arithmetic that made
speculative decoding marginal does not carry over to this.

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

**It is substantially attention, and the number every other page here would
give you is the wrong one.** The dispatch ranking in
`docs/SPECULATIVE_DECODING.md` puts attention at 2.3% of GPU busy; that is
a decode ranking taken at short context. Measured for prefill below, it is
22.3% and the single largest dispatch. Do not carry a decode share into a
prefill argument: this document's first draft did, and reached the
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
| batched MoE pair | `crates/gpu/src/moe_prefill_batch.rs` | **done 2026-08-18 (steps 2+3), bit-exact against M decode passes; see "Steps 2 and 3, measured"** |
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
`route_start`/`route_count` window, so the kernel iterates routes rather
than tokens and a chunk's ragged routing becomes one dense dispatch. That
is the whole reason a chunk of M tokens does not need M x top_k separate
launches.

**What steps 2 and 3 actually shipped (2026-08-18) takes the route list
and drops the tiling.** `crates/gpu/src/shaders/moe_prefill_batch.metal`
(port-local, concatenated after `moe.metal` so it calls the same INT4 row
helpers the decode pair calls) binds a 32-POINTER argument buffer --
`RoutedBlobsWide`, one entry per cache slot -- where Swift's tiles bound
eight, because this port's slot cache already holds a sub-batch's whole
union resident (`union(M) <= slot_count` is the sub-batch bound). That
buys two simplifications over the DSV4 trio: ONE dense dispatch per
kernel instead of per-tile windows, and a FUSED phase 2 --
`moe_prefill_phase2_fused_int4` is decode's
`moe_phase2_down_reduce_k8` with a token axis (one threadgroup per
`(token, d)`, SIMD group r owning rank r), so the rank-ordered reduce
that Gotcha 27 makes a correctness constraint holds by construction and
the DSV4 down/reduce split (which existed only to fit eight pointers)
has no reason to exist here. Parity is BIT-EXACT against M sequential
decode-pair calls, mutation-checked
(`crates/gpu/tests/moe_prefill_batch_parity.rs`); the hand mutation
that reversed the reduce order flipped exactly one bit in the
real-shape case and nothing in the small ones, so the real-shape case
is the order-sensitivity sentinel.

Two things the landing taught that no table above predicted:

- **A sub-batch must COMMIT ITS OWN command buffer, and the next
  sub-batch must WAIT for it before planning.** The first cut used one
  command buffer per layer; a later sub-batch's `pread` then evicted
  experts from slots an earlier sub-batch's dispatches still named, and
  because `pread`s are host-side work the queue's commit order says
  nothing about them. The 8-slot runtime byte-identity test caught it
  as fluent wrong logits -- union-of-micro-batch > slots is exactly
  what forces a second sub-batch. The LAST sub-batch stays in flight
  for the driver to retire after the next layer's router wait, so the
  cross-layer pipelining step 1 measured survives.
- **Routes stay in PAIR order, not slot-sorted.** The fused phase 2
  looks its routes up BY PAIR (`routes[token * top_k + rank]`), so a
  blob-locality sort would break it -- and at ~3.2 MiB per blob against
  caches far smaller, the sort would buy nothing anyway.

**Read `dequant_int4_batch.rs`'s header before writing either.** Two
optimizations were tried on the batched GEMV and both lost on every shape:
threadgroup staging of `x` (expert shape 0.36 -> 0.55 at M=16) and register
blocking over rows (0.44 -> 0.79 at M=8). One finding twice: the register
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
runs as the table above (misses per layer per token, and prefill's own
union from `MFERENCE_ROUTER_TRACE=1` analysed at `skip=0` so prefill is
included; the script's default excludes it):

| | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| prefill `union(M)`, distinct experts per layer | 13.2 | 20.4 | 29.9 | 41.5 |
| sequential loads over the window, 32 slots (1.507/layer/token) | 3.0 | 6.0 | 12.1 | 24.1 |
| sequential loads over the window, 16 slots (3.044/layer/token) | 6.1 | 12.2 | 24.4 | 48.7 |

The sequential arm is already below the union in seven of eight cells. The
union can only ever recover intra-chunk eviction re-reads, and at 32 slots
there are none to recover: 24.1 loads against 41.5 distinct means most of
the window's experts were resident before the window began.

The one cell where the union wins is 16 slots at M=16, 41.5 against 48.7,
a 15% saving. And `ExpertCache::plan_if_possible` **asserts**
`experts.len() <= slot_count`, so requesting 41.5 experts against 16 slots
aborts the process. The mechanism is legal only where it is useless:

| slot count | largest M with `union(M) <= slots` | union saving there |
| ---: | ---: | ---: |
| 16 | 2 | none (13.2 against 6.1) |
| 32 | 8 | none (29.9 against 12.1) |

Prefill's union also runs above decode's (41.5 against 37.8-39.4 at M=16),
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
decode's, where attention is 2.3% and there is nothing to amortize across
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

**Re-weighted 2026-08-18** against a fresh `c(M)` measurement, after
`46617c6`'s function-constant specialization reached `dequant_int4_batch.rs`
and roughly halved it (`docs/SPECULATIVE_DECODING.md`, "c(M), re-measured").
This paragraph used to read "call it 0.6 weighted", which was both stale and
never how the column below was actually computed -- the old `fully batched`
figures imply ~0.52 to 0.54, not 0.6.

Batched prefill is capped at `MAX_BATCH_ROWS = 16`, so M=16 is the column
that matters. Weighting the fresh numbers by prefill's own dispatch ranking
rather than by a single flat factor:

| | share of prefill GPU | `c(16)` |
| --- | ---: | ---: |
| attention | 22.3% | 0.447 (held at the GEMV rate; see below) |
| resident GEMV | 29.7% | 0.447 |
| routed pair | 23.5% | 0.287 (**proxy**, see below) |
| router GEMV | 3.3% | 0.447 |
| norms + elementwise | 21.2% | **1.0 -- nothing to amortize** |

That is `c(16) = 0.399` over the 78.8% that batches, and a whole-GPU
multiplier of **0.526**. At 32 slots, as a fraction of the 22.95 ms per
prompt token measured above:

| term | now | after step 1 | fully batched |
| --- | ---: | ---: | ---: |
| expert `pread` | 32.3% | 32.3% | 32.3% |
| GPU device time, cb1 (attention, GEMV, norms, router) | 28.5% | 28.5% | ~17.1% |
| GPU device time, routed pair | 16.4% | 16.4% | ~4.7% |
| host scheduling gap | 13.7% | ~5% | ~5% |
| encode + logit readback | 5.9% | ~5% | ~5% |
| bind + retire + router readback | 4.2% | ~3.5% | ~0.5% |
| **total** | **100%** | **~91%** | **~65%** |

**Predicted: step 1 alone about 1.1x, the whole program about 1.55x on this
base**, landing near 14.8 ms per prompt token against Swift's 7.5. The
routed-pair row is where the fresh measurement lands (16.4% to 4.7%, where
0.6 flat gave 9.8%); cb1 barely moves and goes the *wrong* way, because
holding its norms at 1.0 rather than batching them with everything else is
the correction the old column skipped.

**Parity is still not reachable**, and no `c(M)` can make it so: it needs
3.06x while the `pread` bucket is a third of prefill and does not batch at
all. A better kernel divides the 45% that is GPU device time and nothing
else.

**Step 1 then measured 1.22x, and the column above is wrong in an
instructive direction. See "Step 1, measured" below**, whose extrapolation
starts from a measured run rather than from this composite and lands
higher, at ~1.97x. **The two bracket rather than agree, and the spread is
not `c(M)`**: this table's base run reads 7.41 ms/token of expert `pread`
where step 1's pair 3 reads 3.10, on the same install at the same slot
count, on a machine that was not quiet (Gotcha 43). Read 1.55x to 1.97x as
the honest band and note which bucket it turns on.

Two terms are soft, both in the optimistic direction, and the re-weighting
did not fix either; it only made them easier to see.

**The routed pair's 0.287 was a proxy, and the measurement came in
worse.** It was a resident INT4 GEMV measured at a routed expert's
shape, taken because `moe_phase1_gate_up_act_u16load` and
`moe_phase2_down_reduce_k8` had no batched form to measure. Steps 2 and
3 (landed 2026-08-18) measured the real pair
(`crates/gpu/tests/moe_prefill_batch_bench.rs`, arms interleaved at the
round level after back-to-back runs showed clock wander moving c(2) from
0.44 to 1.07): **c(2) = 0.77, c(4) = 0.68, c(8) = 0.66** at the real
D=2816 F=704 shape with ragged routes at the measured union sizes. The
proxy was optimistic by 2.3x, and the reason is structural: a GEMV
amortizes one matrix across rows, while the routed pair's batched form
reads the SAME per-use expert bytes the sequential form reads (the union
only dedupes traffic a ~32 MiB L2 can hold, and 3.2 MiB blobs at a
union of 30 do not fit), so the win is occupancy and dispatch count,
not weight amortization. Re-weighting with 0.66: the composite's
whole-GPU multiplier moves from 0.526 to **0.614**, the whole-program
projection from ~1.55x to **~1.41x**, and the step-1-based
extrapolation from ~1.97x to **~1.71x** (14.70 ms/token of step-1 base:
cb1 3.75 + routed 2.55 + non-GPU 4.59 = 10.89). The honest band for the
whole program is **1.4x to 1.7x**, and the end-to-end rows below are
the arbiters.

**Attention is held at the GEMV rate**, which is conservative rather than
optimistic: M queries genuinely share one KV read, so a batched attention
should beat a batched GEMV. It is not measured because no such kernel
exists (step 4), and its 22.3% share is the doc's earlier dispatch ranking
rather than a per-arm re-measurement.

The step-1 column's assumption that the scheduling gap is mostly phase A's
is no longer soft -- the driver measured it, and it over-delivered.

### Step 1, measured

Landed and measured 2026-08-16, same day as the correction above. Real
Gemma 4 install, `long-synthesis` (3,015 prompt tokens), `--max-new 8`, 32
slots, `MFERENCE_PREFILL_CHUNK=128`, three interleaved pairs after a
discarded warmup on a machine that was not quiet (Gotcha 43):

| pair | sequential | chunked | |
| ---: | ---: | ---: | ---: |
| 1 | 55.84 s | 44.74 s | 1.248x |
| 2 | 57.48 s | 49.55 s | 1.160x |
| 3 | 56.12 s | 44.41 s | 1.264x |

Pair 2 is slower on both arms and is the contention outlier; pairs 1 and 3
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

Extrapolating the rest from here rather than from the original composite,
and **re-weighted 2026-08-18** with the fresh `c(16)`: 10.11 of the
remaining 14.70 ms/token is GPU device time, split 6.25 cb1 and 3.86 routed.
Within cb1, 27.7% is norms and elementwise with nothing to amortize and the
rest goes at 0.447, giving 3.75; the routed pair goes at its 0.287 proxy,
giving 1.11. With the 4.59 ms/token that is not GPU device time at all, the
fully batched figure is **~9.5 ms/token, i.e. ~1.97x against the sequential
baseline** -- against ~12 ms and ~1.5x at the old `c(M) ~ 0.6`.

**That is the largest thing the kernel fix bought anywhere in this
document**, and it is worth contrasting with where it bought nothing:
`docs/SPECULATIVE_DECODING.md`'s MoE verdict moved two points on the same
re-measurement, because a verify pass divides its cost by an accept length
and a prefill chunk keeps all M of its tokens. Same kernel, same week, and
the divisor is the whole difference.

Read it against the composite's 1.55x rather than instead of it. The two
differ almost entirely in the expert `pread` bucket (3.10 against 7.41
ms/token on two runs of the same install at the same slot count), which is
the bucket that does not batch and therefore sets the ceiling for both.

**A note on why this is worth building at 1.55x to 1.97x.** Prefill is the
only measured gap against Swift, it is 86% of a long prompt's joules
(`docs/POWER_BASELINE.md`), and even the low end of that band is 24 s of
wall clock off the 69.3 s this prompt spends in prefill. What it is NOT is
a path to parity, and the first draft's 2.1x implied one -- note the
re-weighted high end now brushes that number while still not reaching
parity, which is the reason to state the `pread` ceiling every time rather
than the multiplier alone.

### Steps 2 and 3, measured

Landed and measured 2026-08-18, real Gemma 4 install, the frozen
`long-synthesis` prompt (3,015 tokens), `--max-new 8`, 32 slots (auto),
AC, three interleaved A/B/C rounds after a discarded warmup of each arm.
Prefill seconds from the `[stop=...]` footer:

| round | sequential | chunked (step 1) | + batched routed (2+3) |
| ---: | ---: | ---: | ---: |
| 1 | 58.90 | 46.69 | 38.98 |
| 2 | 61.10 | 46.38 | 38.64 |
| 3 | 59.16 | 45.45 | 38.46 |
| mean | 59.72 | 46.17 | **38.69** |
| ms/token | 19.8 | 15.3 | **12.8** |

**Steps 2+3 buy 1.19x on top of step 1's chunking (46.17 -> 38.69), and
1.54x over the sequential path** -- mid-band of the 1.41x-1.71x the
measured `c(8) = 0.66` predicts above, and against Swift's 27.5 s for
this prompt the gap narrows from 2.17x to 1.41x. Step 1 alone measured
1.29x today against its 1.22x on 2026-08-16, which is ordinary
cross-session machine state and one more reason the INTERLEAVED 1.19x
is the number to quote for steps 2+3 rather than a difference of
cross-session totals. Within-arm spread is under 1% on the two chunked
arms.

Kernel-level, the pair's own `c(M)` (interleaved arms,
`crates/gpu/tests/moe_prefill_batch_bench.rs`, real D=2816 F=704 shape,
ragged routes at the measured union sizes): **c(2) = 0.77, c(4) = 0.68,
c(8) = 0.66**. M=16 is unreachable by construction -- the engine caps a
routed sub-batch at `union <= slot_count <= 32`, and the measured
union(16) is 41.5. The end-to-end 1.19x beats what 0.66 on a 16.4%
device-share term alone would suggest because the batched half also
collapses the HOST work per layer: one router readback, one plan, one
`pread` burst, one bind and one routed command buffer per sub-batch
instead of per token, and the shared-expert branches all commit before
the pread instead of interleaving with it.

Gates paid: byte-identity against the sequential path on the synthetic
real-named install (ten cases in `real_forward_gemma4_chunked.rs`,
including the 8-slot union-shrink case that caught the sub-batch
eviction bug), greedy and sampled smokes byte-identical on the real
install at chunk spans 32/128/512, the memory oracle green at a 2,187
MiB peak against the 2,300 ceiling (the batched scratch is ~0.4 MiB),
and the quality gate's frozen digests unchanged. The power capture the
Definition of Done asks for remains owed with the others in
`ROADMAP.md`'s table: `scripts/power.sh` needs sudo and cannot run
non-interactively.

## The attention fork

The question is whether a chunk needs the descoped tile kernels
(`attention_prefill_causal_tiled`, `prefill.metal`'s 16-kernel pipeline --
`DEVIATIONS.md`) or whether it can run the existing split-KV
`attention_decode` per token inside the chunk.

**It can run per token, and it should, first** -- but for a weaker reason
than a 2.3% share would have given. Attention is 22.3% of prefill GPU work
and batching it is worth ~5 points of the ~35 the whole change is worth on
the composite's base (it was ~7 of ~33 before the 2026-08-18 re-weighting;
a cheaper `c(M)` shrinks every batchable term's share of the saving, not
just this one). That is real and it is not a prerequisite: a chunk whose
attention is still per-token gets ~1.44x, and steps 2 to 4 below are
independent of it.

So this is one phase followed by an optional one, rather than a fork:

- **Phase A**, per-token attention inside a batched chunk. ~1.4x. No new
  attention kernel, no touching the descoped pipeline.
- **Phase B**, batched attention. Takes ~1.4x to ~1.5x. And it does not
  need `prefill.metal`'s 16-kernel tile pipeline: what it needs is one
  kernel where M queries share a KV read, which is a natural widening of
  `attention_decode_partial`'s existing split-KV structure (it already
  reads KV in chunks; the change is to hold M query rows per chunk instead
  of one). Scope it against that kernel, not against the descoped pipeline,
  whose 1,202 lines cover embed/norm/rope/router/MoE as well and are
  descoped for reasons that still hold.

## Order of work

0. ~~Measure prefill's own dispatch ranking.~~ **Done, 2026-08-16**; it is
   the table above, and it moved the design (attention 22.3%, not 2.3%).

1. ~~**`RealForwardRunner: ChunkedPrefillRunner`, looping the existing
   per-token kernels inside each layer.**~~ **Done, 2026-08-16, measured at
   1.22x** (see "Step 1, measured"). Gemma 4 first, and a SECOND family
   landed 2026-08-26: the dense half of `llama` (Mistral, Llama 2/3.x,
   `families/llama/prefill.rs`), structurally simpler since a dense layer
   needs no router readback at all, so the whole micro-batch runs every
   layer in ONE command buffer rather than one per layer. Every other
   family is still refused by name rather than falling back to the
   sequential loop, because a caller that explicitly asked for the chunked
   driver on an install it can't serve and quietly got the old path would
   measure the old engine and report it as the new one.

   Reachable through `MFERENCE_PREFILL_CHUNK=<tokens>` on
   `turbospark-check`, an A/B seam beside `MFERENCE_SHARED_CB` and
   `MFERENCE_ROUTED_PIPELINE`; that env var's contract is unchanged (still
   hard-fails on an unsupported family). **`--prefill-chunk` IS wired now
   (2026-08-26, `crates/cli/CLAUDE.md` Gotcha 7, `crates/runtime/CLAUDE.md`'s
   `MFERENCE_PREFILL_CHUNK` bullet), and the server dispatches automatically
   too, with no per-request flag** (`crates/server/CLAUDE.md` Gotcha 19).
   Neither routes through the chunked driver on an install it can't serve
   -- both check `RealForwardRunner::supports_chunked_prefill()`, the same
   predicate the driver's own refusal uses, and fall back to the sequential
   path with no error when it says no. That is what makes wiring the flag's
   `Fixed(128)` default safe even though only two families are served:
   the default was never something a caller who didn't type the flag
   explicitly asked for.

   What it does, and the rest of this entry is the design rather than a
   plan: per chunk it advances M KV rows, the position by M, and
   commit-and-waits exactly as the per-token path does
   (`crates/runtime/CLAUDE.md` Gotcha 2 states the analogous contract for
   `produce_prefill`). On a family with recurrent state the GDN chain is
   sequential by definition and steps M times inside the chunk.

   **Split the layer at the point the flow already commits, and batch only
   the first half.** Phase A (norms, q/k/v/o GEMVs, per-head norms, RoPE,
   attention, the residual adds, the pre-FFN norms and the router GEMV)
   is encoded for all M tokens into one command buffer per layer, so the
   per-layer blocking wait is paid once per chunk instead of once per
   token. Phase B -- router readback, host top-k, plan, `pread`, bind,
   phase 1/2, sandwich norms, `layer_scalar` -- stays per token, because
   `RoutedBlobsBuffer` is a single host-written argument buffer and
   `routing_w` / `moe_acts` are single-row, so M tokens cannot share a
   command buffer there without M copies of all three AND all M tokens'
   experts resident at once.

   **The expert `pread` union is not the mechanism.** An earlier draft of
   this list had one `plan_experts_cached` over the chunk's union replacing
   M per-token plans, worth "25.2% of prefill cut by 3.3x". Measured, it is
   worth nothing at 32 slots and aborts the process at M=16 (see "The
   expert `pread` does not batch" above). What step 1 collects is the
   scheduling gap and the per-token commit overhead, ~9% of prefill.

   Only M-row buffers are needed: `x`, `router_logits_f32`, `dense_x` and
   `routed_x`. Every other scratch stays single-row: a serial compute
   encoder runs dispatches in order, so reuse inside a chunk is correct.
   That deliberately forgoes GPU parallelism across tokens in phase A;
   the parallelism arrives with the M-row kernels in steps 2 to 4.

   Gated by byte-identity against the sequential path, not by coherence:
   `crates/runtime/tests/real_forward_gemma4_chunked.rs` on a synthetic
   real-named install (six cases, mutation-checked), and on the real
   install the greedy and sampled smokes reproduce their frozen digests
   (`b2f16611...`, `0c383ac0...`) at chunk spans 32, 128 and 512.

2. ~~**Batched `moe_phase1_gate_up_act_u16load`**, against
   `dsv4_prefill_moe_phase1_pairs_int2`'s route-list shape. Parity must be
   EXACT against M separate calls, mutation-checked, per this repo's habit.~~
   **Done 2026-08-18** as `moe_prefill_phase1_routes_int4` (see "What
   steps 2 and 3 actually shipped" above).
3. ~~**Batched `moe_phase2_down_reduce_k8`.** The reduce order is a
   correctness constraint, not a style choice (AGENTS.md Gotcha 27): a
   batched phase 2 must reduce each token's slots in the router's ranking,
   independently per token, or output becomes a function of the chunk
   boundary.~~ **Done 2026-08-18** as the FUSED
   `moe_prefill_phase2_fused_int4` -- rank-ordered per token by
   construction, which is what makes the Gotcha 27 proof trivial.
4. **Batched attention** (Phase B above), if the measured 1.4x is not
   enough. One kernel, widening `attention_decode_partial` to hold M query
   rows per KV chunk. Not the descoped tile pipeline. Deliberately deferred
   past the 2026-08-26 default-on and dense-llama work: it is real new-kernel
   engineering (a new function-constant axis, per-row online-softmax state
   generalized to M rows, and real register-pressure risk per
   `dequant_int4_gemm_simd`'s spill history in `crates/gpu/CLAUDE.md`),
   not a small increment to attempt alongside a flag-wiring pass.

   Note steps 2 and 3 carry a constraint step 1 does not: a batched routed
   pair needs all M tokens' experts resident at once, so M is capped at the
   largest value with `union(M) <= slot_count`. Measured above, that is
   M=8 at 32 slots and M=2 at 16. `c(M)` is worse at small M (worse than
   1.0 at M=2), so 32 slots is a precondition for step 3 paying at all.
5. **Widen to the GGUF pairs** per block type, if and when a GGUF MoE
   install is the one being optimized.
6. ~~**Batch the RESIDENT GEMVs**~~ -- landed behind
   `MFERENCE_BATCHED_GEMV=1`, and numbered SIXTH rather than inserted
   before step 4 because these numbers are cited from `ROADMAP.md`,
   `crates/runtime/CLAUDE.md` and a comment in `moe_batch.rs`, and
   renumbering would rot all three. It is independent of steps 4 and 5 and
   could have been done at any point after step 1.

   It is the 29.7% row of the dispatch ranking above, the largest single
   term after the routed pair, and it needed **no new kernel**:
   `dequant_int4_gemm_simd` already existed for the speculative verify, and
   `encode_gemm_any` already refused every other dtype by name. What
   changed is which encoder the chunk driver calls.

   **The four attention projections always; the shared expert's three only
   when `MFERENCE_ROUTED_BATCH` is also on.** The per-token routed pass
   reads a single-row `h1` at offset 0 and that read is on the DECODE
   path's signature (`families/gemma4/moe.rs`), so pointing it at an M-row
   buffer would be a decode change rather than a prefill one; the batched
   routed half already writes `batch_h1` per token and needs no such
   change. The shared expert is also where the host saving is largest --
   the per-token form opens and commits its OWN command buffer per token,
   so a 16-token micro-batch went from 16 command buffers per layer to 1.

   Norms, per-head norms, RoPE, attention, the residual adds and the
   router GEMV all still loop per token, which is the same line
   `families/qwen/batched_layers.rs` draws and the one the composite
   assumes (they are the 21.2% with nothing to amortize; attention is step
   4).

   **Byte-identity is the gate and it is a MEASURED claim.**
   `the_gemm_and_the_gemv_agree_on_data_that_can_see_reassociation` runs
   both kernels on ragged BF16 companions and full-mantissa FP16
   activations and they agree BIT-FOR-BIT, against a positive control
   (`dequant_int4_gemm_mma`) that differs on ~39% of outputs on the same
   fixture. So the seam is a throughput axis only, and
   `real_forward_gemma4_chunked.rs` asserts it against the SEQUENTIAL path
   at five chunk spans.

   Two things the bring-up found that the plan did not predict:

   - **A batched K/V projection can straddle the sliding-window ring's
     wrap**, and `k_slot` validates ONE row, so it would run past the
     layer's buffer with nothing in the way. Split with `ring_spans`, which
     the DFlash2 drafter already had for its own ring and which moved to
     `real_forward_utils.rs` for the second caller. `PROMPT` in the chunked
     test file is 11 tokens against a 136-token ring and cannot reach it;
     the case that does uses a 160-token prompt, and reverting the split
     reddens that one case and nothing else in the file.
   - **The synthetic fixture writes its shared MLP at EIGHT bits where the
     real install declares four**, so the default fixture cannot gate the
     dispatches this seam moves -- it is refused by name instead, correctly.
     That INT8 is deliberate elsewhere (it is the repo's only coverage of an
     INT8 resident GEMV inside a whole Gemma pass), so it stays, and
     `build_synthetic_gemma4_real_install_at_shared_bits` is the four-bit
     sibling the byte-identity cases build. Both the refusal and the
     agreement are pinned.

   **NO THROUGHPUT NUMBER IS RECORDED HERE YET**, deliberately: it landed
   on a machine running another build, and Gotcha 43 says a prefill A/B
   taken then is contamination rather than a measurement. The seam ships
   OFF and the interleaved pairs are owed on a quiet machine, alongside the
   prefill energy capture.

## Gates this owes

Prefill changes the logits the first decoded token is sampled from, so it
is a numerics change however carefully it is written.

- All three real-model gates plus `quality_gate`, per family touched.
- **Byte-identity is the bar, not coherence.** A chunked prefill must
  produce the same tokens as the sequential one; `accept_length_probe.rs`
  established the shape of that check for speculative decoding (compare
  against a non-chunked reference run, never two chunked runs against each
  other, which passes when both are wrong the same way).
- Chunk size must not change the output. If it does, the reduce order in
  step 3 is wrong.
- The memory oracle: `prefill_scratch.rs` allocates real buffers sized by
  chunk, and every oracle ceiling is frozen.

## Which families, and in what order

`gemma4` first: it is the only MoE install on disk, it is the pinned one,
and it is where the 21.4 ms was measured. `ternary27b` and
`museglimmer-30b` are dense: the MoE pair does not apply to them at all,
their prefill lever is the plain GEMV alone, and steps 2 and 3 above buy
them nothing. Every other family needs a 5 to 25 minute re-stream before it
can be gated (`docs/MODELS.md`).

**The dense half of `llama` landed second (2026-08-26)**, ahead of the MoE
families, precisely because it has no MoE pair to widen: it is the same
Phase A batching step 1 already built, applied to a layer with no router
readback, so a whole micro-batch fits in ONE command buffer instead of one
per layer (`families/llama/prefill.rs`). Widening to the MoE half of
`families/llama/` (Mixtral, Qwen3MoE, one flow file) or to `gpt-oss` would
each mean replicating steps 1 through 3 (attention AND routed-expert
batching) plus that family's own hazards -- gpt-oss's alternating window,
attention sinks and router-bias-before-topk, the ring-wrap and
shared-expert ordering Gemma 4's own bring-up already found -- not a small
increment. **`muse_glimmer` was a THIRD case, not a fourth MoE one, and
landed third (2026-08-27)**: it has no router at all (this doc's own line
above), so it needed only the SAME simpler no-mid-layer-commit driver dense
`llama` got, adapted to its own attention shape (three-sliding/one-full
window, centered norms on four tensors and plain on the final one, NoPE on
the full layers, an attention output gate, a logit softcap) -- no MoE steps
involved, and no ring-wrap hazard either, since attention stays per-token
inside the driver and the sliding-window ring is addressed by `position`
exactly as the sequential path already does it. `mlp::encode_mlp_block`
gained the same `x_off: u64` parameter `encode_llama_layer_dense` did;
`attn::encode_attention_block` needed no change, for the identical reason
dense `llama`'s did not (it never touches `scratch.x`). Verified
byte-identical against sequential on the synthetic fixture (chunk-span
sweep `[1, 2, 3, 4, 7, 11]`, including a span that crosses the sliding
window) and on the real `~/models/museglimmer-30b.gturbo` install: greedy
and sampled stdout md5-identical against a pre-change binary, at 40 new
tokens each (`crates/runtime/CLAUDE.md` Gotcha 14).

**THE MoE HALF OF `families/llama/` AND `gpt-oss` LANDED FOURTH AND FIFTH
(2026-08-27), AND NEITHER NEEDED STEPS 2/3 EITHER.** Both replicate Step 1
alone: a per-layer command buffer for the attention-and-router half
(`cb1`), then a per-token routed loop pipelined with `RoutedSlot` across the
micro-batch, exactly as Gemma 4's ORIGINAL driver did before the batched
routed kernel (steps 2/3) existed. The reason this was enough, and did not
need new kernel work the way the "not a small increment" line above
predicted: Step 1's routed dispatch reuses `encode_moe_phase1_any` /
`encode_moe_phase2_any`, the SAME per-token calls the sequential decode path
already makes, which are already layout-agnostic (Affine, GGUF Q4_K/Q6_K,
MXFP4). Steps 2/3's batched routed KERNEL is the INT4-affine-only piece, and
it stayed unwired for both -- a GGUF Qwen3MoE install and an MXFP4 gpt-oss
install both still refuse `MFERENCE_ROUTED_BATCH=1`'s widening today, which
is the correctly-scoped remainder of "Step 5 (GGUF Routed Pair Widening)"
below.

`RoutedSlot`, the bank/protect pipelining pattern, and `retire_routed` moved
out of `families/gemma4/moe.rs` into a shared
`crates/runtime/src/moe_prefill_pipeline.rs` ahead of this pair landing,
since three families now need the identical struct rather than three copies
of it; Gemma 4's own 16-test chunked-prefill suite confirmed the extraction
moved no bytes.

Neither new driver needed a ring-wrap fix, for two different reasons. The
MoE half of `llama` has no sliding-window layers at all (`RealLlamaState::build`
refuses one), so there is no ring to straddle. `gpt-oss` DOES have a real
alternating window, but this driver's attention stays per-token and
unbatched (no `MFERENCE_BATCHED_GEMV`-style widening for either family), so
there is no batched K/V projection to straddle it either -- the ring is
addressed by `position` exactly as the sequential path already does.
`gpt-oss`'s one real behavioral difference from `llama`'s MoE half, the
router bias added on the host between the readback and the top-k, needed no
new code in the chunked driver: it rides inside the SAME
`encode_gpt_oss_layer_moe` call the per-token loop already makes, which
already does the add internally per call.

Both verified byte-identical against sequential on their synthetic fixtures
(chunk-span sweep, plus a cache-too-small-to-pipeline case forcing the
`banks == 1` retire-before-encode fallback) and on real installs: `gpt-oss`
against `~/.turbospark/models/gptoss-20b.gturbo` (greedy and sampled stdout
md5-identical against a pre-change binary, prefill dropping from 7.56s to
3.79s on a 75-token prompt now that chunking engages), the MoE half of
`llama` against a freshly-pulled `Qwen/Qwen3-30B-A3B-GGUF` install (see
`crates/runtime/CLAUDE.md` Gotcha 14 for the exact md5s).
