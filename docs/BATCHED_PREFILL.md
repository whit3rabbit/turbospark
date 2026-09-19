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
| chunk span arithmetic | `crates/core/src/prefill.rs`, `chunk_sizing.rs` | done, tested, wired for the Unknown arm; Known arm unconsumed |
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
divisor is prefill, `TURBOSPARK_PHASES=1`. Two arms, because the expert
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
union from `TURBOSPARK_ROUTER_TRACE=1` analysed at `skip=0` so prefill is
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

`TURBOSPARK_PHASES=1 TURBOSPARK_DISPATCH_PROFILE=1`, real Gemma 4 install, the
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
slots, `TURBOSPARK_PREFILL_CHUNK=128`, three interleaved pairs after a
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

   Reachable through `TURBOSPARK_PREFILL_CHUNK=<tokens>` on
   `turbospark-check`, an A/B seam beside `TURBOSPARK_SHARED_CB` and
   `TURBOSPARK_ROUTED_PIPELINE`; that env var's contract is unchanged (still
   hard-fails on an unsupported family). **`--prefill-chunk` IS wired now
   (2026-08-26, `crates/cli/AGENTS.md` Gotcha 7, `crates/runtime/AGENTS.md`'s
   `TURBOSPARK_PREFILL_CHUNK` bullet), and the server dispatches automatically
   too, with no per-request flag** (`crates/server/AGENTS.md` Gotcha 19).
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
   (`crates/runtime/AGENTS.md` Gotcha 2 states the analogous contract for
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
   `dequant_int4_gemm_simd`'s spill history in `crates/gpu/AGENTS.md`),
   not a small increment to attempt alongside a flag-wiring pass.

   Note steps 2 and 3 carry a constraint step 1 does not: a batched routed
   pair needs all M tokens' experts resident at once, so M is capped at the
   largest value with `union(M) <= slot_count`. Measured above, that is
   M=8 at 32 slots and M=2 at 16. `c(M)` is worse at small M (worse than
   1.0 at M=2), so 32 slots is a precondition for step 3 paying at all.
5. **Widen to the GGUF pairs** per block type, if and when a GGUF MoE
   install is the one being optimized.
6. ~~**Batch the RESIDENT GEMVs**~~ -- landed behind
   `TURBOSPARK_BATCHED_GEMV=1`, and numbered SIXTH rather than inserted
   before step 4 because these numbers are cited from `ROADMAP.md`,
   `crates/runtime/AGENTS.md` and a comment in `moe_batch.rs`, and
   renumbering would rot all three. It is independent of steps 4 and 5 and
   could have been done at any point after step 1.

   It is the 29.7% row of the dispatch ranking above, the largest single
   term after the routed pair, and it needed **no new kernel**:
   `dequant_int4_gemm_simd` already existed for the speculative verify, and
   `encode_gemm_any` already refused every other dtype by name. What
   changed is which encoder the chunk driver calls.

   **The four attention projections always; the shared expert's three only
   when `TURBOSPARK_ROUTED_BATCH` is also on.** The per-token routed pass
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

   **A FIRST ATTEMPT 2026-09-05 CONFIRMS THE DIRECTION AND NOT THE
   MAGNITUDE, AND IS ITS OWN INSTANCE OF GOTCHA 43.** Real Gemma 4 install,
   frozen `long-synthesis` prompt (3,015 tokens), `--max-new 8`,
   `--expert-cache-slots 24` (`TURBOSPARK_ROUTED_BATCH=1` refuses anything
   at or above 32, which `auto` now resolves to on this machine since
   `ALLOWED_CACHE_SLOTS` widened past the 8/16/24/32 set this measurement
   predates), one discarded warmup, three interleaved pairs of
   `TURBOSPARK_ROUTED_BATCH=1` alone against the same plus
   `TURBOSPARK_BATCHED_GEMV=1`:

   | pair | no `BATCHED_GEMV` | with `BATCHED_GEMV` | ratio |
   | ---: | ---: | ---: | ---: |
   | 1 | 42.44 s | 36.05 s | 1.18x |
   | 2 | 45.19 s | 44.19 s | 1.02x |
   | 3 | 54.90 s | 39.78 s | 1.38x |
   | mean | 47.51 s | 40.01 s | **1.19x** |

   The direction is unambiguous (the batched arm was never slower across
   three pairs) but the SPREAD -- 1.02x to 1.38x on supposedly identical
   work -- is the exact tell Gotcha 43 names: `ps -A -o %cpu,comm` taken
   right after the run showed a background MCP process at 150% CPU plus a
   browser and two IDE helpers, none of it quiesced first. That is
   contamination, not noise to average away, so **1.19x is a directional
   number and not a citable one**. The interleaved pairs are still owed on
   a machine confirmed idle before the run starts (check `ps -A -o
   %cpu,comm | sort -rn | head` reads near-zero, not just `pmset -g therm`),
   alongside the prefill energy capture -- which is separately blocked
   until `crates/bench`'s model mode calls the chunked path at all
   (`ROADMAP.md`'s Active Tasks table).

   **A SECOND ATTEMPT 2026-09-06 IS ALSO NOT CITABLE, AND FOR A DIFFERENT
   REASON THAT CONFIRMS THIS IS STRUCTURAL RATHER THAN A ONE-OFF.** Same
   command, same prompt, one discarded warmup, three interleaved pairs, read
   from the CLI's own `prefill=Ntok/Ss` footer rather than wall-clock (which
   also carries process-spawn and model-load time neither arm's kernel
   change touches):

   | pair | no `BATCHED_GEMV` | with `BATCHED_GEMV` | ratio |
   | ---: | ---: | ---: | ---: |
   | 1 | 57.12 s | 51.81 s | 1.10x |
   | 2 | 64.94 s | 58.88 s | 1.10x |
   | 3 | 64.64 s | 64.26 s | 1.01x |
   | mean | 62.23 s | 58.32 s | 1.07x |

   `ps -A -o %cpu,comm | sort -rn | head` read near-idle immediately before
   the run started (the pre-check this file's own prior paragraph asked
   for), so the contamination this time built up mid-run rather than being
   present at the pre-check: the POST-run snapshot showed `fseventsd` at
   141% and two `ld` processes at 88%/77%, four separate `rustc` processes
   between 32% and 47%, none of which were running by the time the snapshot
   was taken minutes later -- a concurrent `cargo build` on this shared tree
   (AGENTS.md Gotcha 13: "routinely worked by more than one session at
   once"), not an idle-desktop artifact this time. Pair 3's ratio collapsing
   to 1.01x lines up exactly with when that build's CPU/GPU contention would
   have peaked, and pairs 1-2 (run before it ramped up) agree tightly with
   each other at 1.10x. **A pre-run `ps` check is therefore not sufficient
   on its own** -- it catches contamination already present, not a build
   that starts three minutes into an eight-run script -- so a third attempt
   needs `pgrep -x rustc` (or a broader cargo-process check) polled
   throughout the run, not just once at the start, before 1.10x or any other
   number here can be called citable. Direction is now confirmed on two
   independent attempts (7 of 7 timed pairs plus both warmups, batched never
   slower); magnitude remains open.

   **A THIRD ATTEMPT 2026-09-09 IS CITABLE, AND THE CONTAMINATION SOURCE WAS
   NEITHER OF THE FIRST TWO.** Same command, same prompt, one discarded
   warmup, three interleaved pairs, `prefill=Ntok/Ss` footer, AND a
   contamination monitor polling `ps -A -o %cpu,comm | sort -rn | head`
   every 10s for the whole run (the fix both prior attempts called for).
   Pair 1 ran clean. Pairs 2 and 3 did not: `mediaanalysisd` (Apple's Photos
   / media-analysis daemon) ramped from 77% to over 200% starting mid-way
   through pair 2, dragging in `ANECompilerService` and `corespotlightd`
   alongside it, and stayed hot through the end of pair 3 -- a THIRD
   contamination source, distinct from both the browser/MCP load of the
   first attempt and the `cargo build` of the second, and one the monitor's
   original `rustc`/`cargo`/`ld` grep would not have caught (it was only
   found because the monitor also logged the top-3 CPU processes by name,
   not just a fixed grep list). Pair 2's ratio dropped to 0.943x (the
   batched arm read SLOWER, the first inversion of the direction across all
   three attempts) and pair 3's baseline arm read 57.99s against the other
   two pairs' ~42-43s -- both consistent with contamination landing
   unevenly across arms rather than with a real effect.

   Discarding pairs 2 and 3 and re-running them once `mediaanalysisd` quieted
   (confirmed via the same top-3 CPU log, this time idle) gives:

   | pair | no `BATCHED_GEMV` | with `BATCHED_GEMV` | ratio |
   | ---: | ---: | ---: | ---: |
   | 1 | 42.78 s | 35.86 s | 1.193x |
   | 2 (re-run) | 42.37 s | 36.08 s | 1.174x |
   | 3 (re-run) | 42.46 s | 35.84 s | 1.185x |
   | mean | 42.54 s | 35.93 s | **1.18x** |

   Spread across the three clean pairs is 1.174x-1.193x, under 2% -- the
   first citable magnitude this A/B has produced, and it lands inside the
   first attempt's own directional range (1.02x-1.38x, mean 1.19x). Prefill
   throughput: 70.9 tok/s baseline against 83.9 tok/s batched, on the frozen
   3,015-token `long-synthesis` prompt, `--expert-cache-slots 24`, AC power.

   **The generalizable lesson is not "watch for `mediaanalysisd` specifically",
   it is that a contamination monitor keyed to a fixed process-name list only
   catches contamination it already anticipated.** Two attempts named
   `rustc`/`cargo`/`ld` after the fact; this one would have missed
   `mediaanalysisd` under the same fixed-list design if it had not also
   logged the top-3 CPU consumers by name at every poll, unfiltered. A
   monitor for this kind of A/B should always log the top few processes by
   CPU share, not merely grep for the specific process that burned the
   previous attempt.

   **A SECOND FAMILY WIRED TO THIS SEAM 2026-08-29, THE DENSE HALF OF THE
   QWEN LINEAR-ATTENTION FLOW, AND IT COST NO DISPATCH CODE AT ALL** --
   less than Gemma 4's did, which needed new batched encoders written.
   `families/qwen/batched_layers.rs` had already built all three (the
   full-attention block, the GDN linear block, the dense FFN) for the
   MTP/DFlash2 verify pass, and they take the parameters the chunk driver
   has and use the `t * hidden * 2` row convention it already writes. So
   the arm is a branch in `families/qwen/prefill.rs`'s layer loop plus a
   lazily-allocated `BatchedScratch`. The constant that makes it fit
   exactly is that `MAX_PREFILL_BATCH` IS `gpu::MAX_BATCH_ROWS`, both 16,
   so a full micro-batch is one GEMM dispatch and never sub-batches.

   **ITS GATE IS NOT BYTE-IDENTITY, AND THAT IS THE ONE REAL DIFFERENCE
   FROM GEMMA 4'S.** The claim four paragraphs up -- that this seam is a
   throughput axis only, because the GEMM and the GEMV agree bit-for-bit --
   is about those two KERNELS and does not extend to this family's pass as
   a whole. `produce_batched` and `produce` differ here by 6.2e-8 to 1.5e-5
   nats with the argmax agreeing on every row, which commit `e8deb6c`
   measured against a dense batched-vs-cached shape floor of 7.4e-6 on MLX
   for this same architecture and concluded is a FLOOR every engine has,
   not a defect. The eliminations behind that verdict are worth not
   re-deriving: present at M=1, so batch width was never the variable;
   absent at one and two keys and present from three; not the INT4 GEMM, not
   the GDN multi-row kernels, not the split-KV combine, not stale scratch.
   What is left is the attention reduction.

   **MEASURED 1.86x TO 2.13x ON PREFILL**, which is the first throughput
   number this step has on any family (Gemma 4's is still owed). Real
   `qwen38-27b` install, 2,940-token `long-synthesis` prompt, interleaved
   pairs with the warmup discarded: 157.28s/73.91s and 137.41s/72.08s, i.e.
   18.69 -> 39.78 and 21.40 -> 40.79 tok/s. The machine was NOT quiet, and
   the dispersion says which arm that hurt -- 14.5% between the default
   arm's two readings against 2.5% between the batched arm's -- so per
   Gotcha 43 (contention depresses tok/s) the pairs are a LOWER bound. All
   four runs plus the warmup produced byte-identical text. Full row in
   `docs/BENCHMARKS.md`, which also records that this leaves PP at roughly a
   fifth of the oMLX community build's 210.3 tok/s rather than closing it;
   what is still unbatched here is attention, which is step 4.

   Two consequences. The DEFAULT arm is what must stay byte-identical, and
   it is what a caller who sets no env var gets. And the synthetic fixture
   cannot see the floor at all (`real_forward_qwen35_batched_onset.rs` reads
   0 differing logits of 128 at every span on the same builder at the same
   width), so `real_forward_qwen35_chunked.rs`'s batched byte-identity cases
   pin the WIRING -- rows, offsets, ordering, the GDN state's advance -- and
   are documented as not being evidence about the arithmetic.

   Three guards the qwen arm carries that Gemma 4's does not need. The
   width is refused UP FRONT rather than at the first dispatch, because the
   1-bit and 2-bit checkpoints of this same architecture reach no M-row
   GEMM and a caller should learn that before any KV row is written
   (`encode_gemm_any` stays the per-tensor backstop for a mixed-width
   install one probe cannot see). The KV-wrap check `produce_batched`
   carries is copied in: this family has no sliding window, but "linear"
   still wraps at `max_context`, and a batched k/v projection straddling
   that boundary scatters into row 0. And the M-row scratch is allocated
   ONLY when the seam is on, which is stricter than Gemma 4's
   `ensure_batched` (that one also serves `TURBOSPARK_ROUTED_BATCH`, so its
   driver allocates unconditionally) -- `BatchedScratch` is ~10 MiB on the
   real install, most of it a `batch * vocab` logits plane this driver never
   reads, and `qwen38_memory_oracle`'s frozen row has to keep describing the
   engine that shipped before the arm existed. That is asserted with the
   buffer-allocation counter rather than argued, because a footprint ceiling
   with 87 MiB of headroom could not see it.

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
tokens each (`crates/runtime/AGENTS.md` Gotcha 14).

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
MXFP4). Steps 2/3's batched routed KERNEL was the INT4-affine-only piece
when these two flows landed, and it stayed unwired for both AT THE TIME. The
MXFP4 gpt-oss arm has since been built ("Step 5's MXFP4 arm, built and
measured" below); a GGUF Qwen3MoE install still refuses
`TURBOSPARK_ROUTED_BATCH=1`'s widening, by name, and that Q4_K/Q6_K arm is the
correctly-scoped remainder of "Step 5 (GGUF Routed Pair Widening)" below.

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
unbatched (no `TURBOSPARK_BATCHED_GEMV`-style widening for either family), so
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
`crates/runtime/AGENTS.md` Gotcha 14 for the exact md5s).

**THE DENSE HALF OF THE QWEN LINEAR-ATTENTION FLOW LANDED SIXTH
(2026-08-29), `qwenGdnDense` (`qwen38-27b.gturbo`), AND IT IS NEITHER OF THE
TWO SHAPES ABOVE.** The obvious precedent looked like `families/qwen/batched.rs`
-- the M-row GEMM machinery this family already built for the MTP/DFlash2
verify pass -- and that turned out to be the wrong one: it implements steps
2-6 (GEMVs become GEMMs), sized for tiny drafter block depths and allocated
only when a drafter is open. `families/qwen/prefill.rs` is Step 1 again,
same shape as dense `llama`'s and `muse_glimmer`'s: loop the EXISTING
per-token kernels (`attn::encode_linear_block` for the gated-DeltaNet mask-2
layers, `attn::encode_full_attention_block` for the mask-1 ones,
`dense::encode_qwen_layer_dense` for the FFN) inside a micro-batch, batching
COMMAND BUFFERS rather than GEMVs, one buffer for the whole micro-batch
since a dense layer has no router readback. **No new kernel, and no new
buffer either** -- the first family so far where that is true even of the
per-token scratch, because every intermediate the trunk's sequential flow
already owns (`qwen.moe_x`, `qwen.h2`, the GDN scratch fields) is a
single-row GPU-only buffer safe to reuse per token under commit-order
execution, exactly the property `crates/gpu/AGENTS.md` Gotcha 8 already
established for the other two Step-1-shaped drivers.

The one open question this family has that no other Step-1 driver does is
the GATED-DELTANET RECURRENT STATE, and it resolves for free rather than
needing new machinery: `encode_linear_block`'s decode-shaped kernels advance
`qwen.gdn.state_buffer(layer)` in place with no position argument at all,
so calling it once per token, strictly in increasing order, within one
layer's inner loop before moving to the next layer, reproduces sequential
decode's math exactly -- `crates/runtime/AGENTS.md` Gotcha 4's constraint
satisfied by construction. It is also what makes cross-chunk continuity
free: the state buffer is the one sequential decode already reads and
writes, so a prompt spanning several `prefill_chunk` calls carries it
forward automatically, with no state to hand between calls.

**AN IMAGE PROMPT CHUNKS SINCE 2026-09-06**, and this paragraph used to
record its refusal as one of two. The driver mirrors both halves of
`produce.rs`'s vision handling now -- the tower-row blit and the mRoPE
angle -- so it is the family's second embedding call site, exactly what
`crates/runtime/AGENTS.md` Gotcha 27 predicted a chunked driver would
create. What replaced the refusal there is a by-name refusal of
`TURBOSPARK_BATCHED_GEMV` PLUS an image, about the ANGLE rather than the
embedding: `encode_full_attention_block_batched` rotates at the raw
position and takes no `RopePosition`, and it is shared with the verify
pass. Correctness is confirmed on the real install (generated bytes
identical to sequential, greedy and sampled, at chunk 32/128/256); the
THROUGHPUT figure is still owed.

**THE FIRST ATTEMPT AT THAT FIGURE WAS DISCARDED RATHER THAN PUBLISHED, and
it is worth recording why.** Four interleaved prefill-only runs on the real
`qwen38-27b-vision.gturbo` (1,299-token prompt, 1,280 of them image) read
sequential 73.65 s / 82.48 s against chunked 91.76 s / 76.96 s. The
DIRECTION does not reproduce -- chunked is 25% slower in one pair and 7%
faster in the other -- and the within-arm spread is 12% and 19% on
byte-identical work. That is AGENTS.md Gotcha 43's dispersion tell exactly,
and `ps -A -o %cpu,comm | sort -rn` confirmed the cause rather than leaving
it inferred: a concurrent session was compiling Swift, with `fseventsd` at
130%, `swift-frontend` at 94% and two `rustc` besides. A quiet-machine
re-run is what this row needs; the numbers above are published as a record
of a refused measurement, never as a result.

An open drafter
(`self.real_mtp.is_some() || self.real_dflash.is_some()`) is refused too:
the sequential dense branch fires the DFlash2 aux-capture hook on every
forward pass, which this driver does not encode, so silently prefilling
through it would leave the drafter reading a stale or empty capture on its
first draft. `supports_chunked_prefill()` folds the DRAFTER condition into its
qwen clause so a caller routes around the driver entirely rather than
reaching that refusal in the ordinary case; the driver still carries the
check itself as a backstop. It deliberately does NOT fold in the
batched-GEMV one -- that predicate answers about the INSTALL, where the
seam is a per-RUN choice, and folding it in would let one env var change
what an install is reported to support. The MoE half (`qwenGdnMoe`) is left for a
follow-up, matching how `llama`'s dense and MoE halves landed as two
separate steps.

Verified byte-identical against sequential on the synthetic fixture
(`tests/real_forward_qwen35_chunked.rs`: chunk-span sweep `[1, 2, 3, 4, 7,
11]`, crossing this fixture's GDN-then-full-attention layer mask at every
span; plus the `TURBOSPARK_BATCHED_GEMV`, vision, open-drafter and MoE-still-
refused cases every sibling driver carries) and on the real
`~/models/qwen38-27b.gturbo` install: greedy and sampled stdout
byte-identical against the sequential path, `qwen38_quality_gate` unmoved
(reference-answer perplexity 4.9432, frozen digests all reproduced). This is
what unblocks a real PP figure for the oMLX comparison below -- measuring
and recording that figure is tracked separately, since it is a benchmark run
rather than a code change.

### Step 5's two arms, measured before building either

Measured 2026-08-27 on the two real installs left by the step-1 landing
above, to decide whether Step 5 pays and which arm to build first.
`TURBOSPARK_PHASES=1`, the frozen `long-synthesis` prompt, `--max-new 8` so
99.7% of the divisor is prefill, chunking already on at the default 128, slot
count `auto`. **The answer inverts the order the work was scoped in: build
the MXFP4 arm first, and the GGUF one may not be worth building at all.**

| | Gemma 4 (affine) | qwen3moe (Q4_K/Q6_K) | gpt-oss (MXFP4) |
| --- | ---: | ---: | ---: |
| experts/layer, top_k | 128, 8 | 128, 8 | **32, 4** |
| one expert blob | 3.2 MiB | 2.92 MiB | **12.64 MiB** |
| layers | 30 | 48 | 24 |
| slots at `auto` | 32 | 32 | **24** |
| prefill ms/token | 14.70 | 46.4 | 31.6 |
| routed pair, GPU device ms/token | 3.86 | 10.11 | **15.15** |
| ... as a share of all GPU device time | 38.2% | 36.5% | **61.4%** |
| expert `pread` (**does not batch**) | 25.0% | **37.1%** | **8.2%** |
| expert cache hit rate | 81.4% | 75.7% | **96.7%** |
| routed host structure (retire+bind+readback) | 3.44 | 8.95 | 17.12 |
| largest M with `union(M) <= slots` | 8 | 8 | **16 (the kernel's own cap)** |

Gemma's column is the step-1 pair-3 table above; the other two were measured
for this section. The gpt-oss row reproduced across two runs (89.60 s and
91.44 s wall, routed device 15.15 and 14.70 ms/token) on a machine that was
NOT quiet (Gotcha 43: `synrepo` and `rust-analyzer` both near a full core),
which is why the columns above are shares and reachable-M rather than
absolutes -- GPU device time is the bucket contamination does not reach.

**`gpt-oss` is the arm worth building, on three independent counts.** Its
routed pair is 61.4% of prefill GPU device time against Gemma's 38.2%, so
the term steps 2/3 attack is the workload rather than a quarter of it. Its
`pread` bucket -- the term that does not batch and that sets the ceiling for
both other families -- is 8.2% rather than 25-37%, because 32 experts at
top-4 fit a 24-slot cache almost entirely (96.7% hit). And **it is the only
family here that reaches M=16**: with only 32 experts to draw from, its union
saturates near 20 and stays under 24 slots at every M, so the binding
constraint is `MAX_BATCH_ROWS` rather than `union(M) <= slot_count`. Both
128-expert families cap at M=8, where the measured `c(M)` is worse.

**`qwen3moe` is the weaker arm and its ceiling is not the kernel.** 37.1% of
its prefill is expert `pread`, measured three separate times across this
document to batch not at all, and its 75.7% hit rate at the maximum 32 slots
says that bucket is already at the lever's end (`ALLOWED_CACHE_SLOTS` stops
at 32). Its routed device share, 36.5%, is Gemma's; so is its reachable M.
It would also cost TWO block-type kernels rather than one -- its blobs are
Q4_K on gate/up and **Q6_K on down**, so phase 1 and phase 2 need different
row helpers -- where MXFP4 covers both phases with one. Cheaper work, on a
larger term, for the family that reaches the wider batch: the ordering is not
close.

**What is NOT measured here is `c(M)` for either block type**, and that gap
is the reason this section quotes shares rather than a projected multiplier.
The only measured `c(M)` in this document is INT4-affine at Gemma's shape
(0.77/0.68/0.66 at M=2/4/8), and this document has already been burned once
for borrowing a proxy across kernels: the routed pair's 0.287 stand-in was
optimistic by 2.3x when the real pair was finally measured. MXFP4's blob is
12.64 MiB against affine's 3.2, so its union dedupes even less of what an L2
can hold, and the structural reason Gemma's pair only reached 0.66 -- the win
is occupancy and dispatch count, not weight amortization -- applies here
harder, not less. Measure `c(M)` on the real MXFP4 shape
(`moe_prefill_batch_bench.rs`'s arms, at D=2880 F=2880 top_k=4) before
quoting any end-to-end number for this arm. (SINCE DONE, 2026-08-27: the
MXFP4 `c(M)` is measured at 0.76/0.74/0.73/0.74 for M=2/4/8/16 -- see "the
pair's own c(M)" in the built-and-measured section below. The Q4_K/Q6_K
half of this paragraph still stands unmeasured.)

The kernel itself is a smaller piece of work than "a genuinely new kernel
shape" suggests, because both halves already exist and neither is the hard
one: `moe_prefill_batch.metal` owns the route list, the 32-pointer
`RoutedBlobsWide` buffer and the rank-ordered fused phase 2 that makes
Gotcha 27 hold by construction, while `moe_gguf.metal` owns
`dequant_mxfp4_row_simd`, `moe_activate_mxfp4`'s clamped SwiGLU and the
per-expert bias offsets. What has to be designed rather than composed is the
`Mxfp4Activation` uniform reaching the batched pair without acquiring a
function-constant axis that misses `constants_key` (Gotcha 1, and
`moe_gguf/`'s own note on why that activation is a uniform in the decode
pair).

### Step 5's MXFP4 arm, built and measured

Landed 2026-08-27, the same day the section above scoped it.
`moe_prefill_batch_gguf.metal`'s `moe_prefill_phase1_routes_mxfp4` and
`moe_prefill_phase2_fused_mxfp4`, driven from
`families/gptoss/moe_batch.rs` under the same `TURBOSPARK_ROUTED_BATCH=1`
seam. Real `~/.turbospark/models/gptoss-20b.gturbo`, the frozen
`long-synthesis` prompt (2,839 tokens), `--max-new 8`, 24 slots (auto),
three interleaved pairs after a discarded warmup:

| pair | per-token routed | batched routed | |
| ---: | ---: | ---: | ---: |
| 1 | 82.94 s | 64.35 s | 1.289x |
| 2 | 79.31 s | 59.83 s | 1.326x |
| 3 | 78.37 s | 59.81 s | 1.310x |
| mean | 80.21 s | **61.33 s** | **1.31x** |
| ms/token | 28.3 | **21.6** | |

**1.31x against Gemma 4's 1.19x for the same step on the affine pair**, and
the ordering measurement above is why: this family's routed pair is 61.4% of
its prefill GPU device time where Gemma's is 38.2%.

Where it came from, `TURBOSPARK_PHASES=1` on the batched arm against the
per-token one, ms per prompt token:

| bucket | per-token | batched |
| --- | ---: | ---: |
| gpu wait (layer cb1) | 10.93 | 18.40 |
| routed cb retire | 16.32 | **0.82** |
| expert io (`pread`) | 2.59 | **1.26** |
| routed bind + upload | 0.70 | 0.05 |
| encode + logit readback | 0.94 | 0.26 |
| **total** | **31.6** | **20.7** |
| GPU busy, cb1 / routed | 9.54 / 15.15 | 8.86 / **10.16** |

**Read the total row, not the `gpu wait` one.** That bucket ROSE, and it is
not a regression: the batched routed buffer is committed at the end of a
layer and retired after the NEXT layer's router wait, so the waiting that
used to be attributed to `routed cb retire` has migrated into `cb1`'s wait.
This is the same reattribution step 1's own table records in the opposite
direction, and it is why neither bucket alone is quotable.

**The pair's own `c(M)` is MEASURED on the bench's interleaved arms since
2026-08-27** (`moe_prefill_batch_bench.rs`'s `mxfp4` module, D=2880 F=2880
top-4, unions 6/10/13/17 off the measured prefill table; four serial runs,
spread under 0.01 on every M above 2):

| M | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: |
| c(M) | 0.76 | 0.74 | 0.73 | 0.74 |

**The 0.67 this paragraph first quoted for c(16) -- inferred from the phase
table's device-time rows (10.16 / 15.15) -- was 9% optimistic**, which is
the cross-instrument gap the bench arm existed to close: the end-to-end
rows fold in whole-command-buffer effects the kernel-level arms exclude.
The conclusion the inferred figure was drafted to support still stands, and
now on same-instrument terms: measured the same day on the same machine,
the affine arm reads c(8) = 0.73 beside MXFP4's 0.73, so the two block
types land within a point of each other and neither is anywhere near a
weight-amortization curve -- the win is occupancy and dispatch count, as
the affine measurement concluded. (The affine arm's own recorded
0.77/0.68/0.66 did not reproduce on this busier day either, reading
0.80-0.89 / 0.75-0.85 / 0.73-0.74; ratios hold within a session and
absolutes do not, AGENTS.md Gotcha 22.) One shape note: the MXFP4 curve is
FLAT from M=4 where the affine one keeps improving to M=8 -- at this shape
the sequential arm's per-token device cost is already constant (~0.58 ms),
so the batched win saturates early and M=16 buys dispatch count, not a
better c(M). And one instrument note now recorded in the bench itself: the
file's two arms must not time-share the device (a static mutex serializes
them; the first concurrent `-- --ignored` run read the affine c(8) as
0.38 with no error anywhere).

**M=16 IS ACTUALLY REACHED, measured rather than inferred from the mean
union.** The expert-cache counters say the greedy shrink essentially never
fires: the batched arm makes 75,762 plan requests against the 73,478 that
full-width 16-token sub-batches predict (178 sub-batches per layer, 24
layers, union(16) = 17.2), a 3.1% gap. On both 128-expert families the same
arithmetic would cap the sub-batch at 8.

**AND THE EXPERT `pread` BUCKET HALVED (2.59 to 1.26 ms/token), WHICH IS THE
ONE PLACE THIS FAMILY CONTRADICTS THE UNION FINDING ABOVE.** Absolute cache
MISSES -- the count that is bytes rather than bookkeeping -- fell 9,044 to
6,478, a 28% cut, where Gemma 4's step 1 measured its hit rate moving 81.2%
to 81.4% and this document calls the union worthless three separate times.
Read the ABSOLUTE misses and not the rate: the batched arm's hit rate looks
WORSE (96.7% to 91.4%) purely because its request count fell 3.6x with the
misses in the numerator.

**The mechanism WAS separated experimentally on 2026-08-27, and the
reasoned attribution this paragraph first carried was WRONG.** The
hypothesis was pipelining pressure: the per-token path passes each token's
plan a `protect` set naming the previous token's in-flight slots
(`RoutedSlot::protect`), which reserves slots and forces evictions the
routing alone would not, while the batched path retires before it plans and
passes an EMPTY set over the whole sub-batch's union at once. The seam that
tests it is `TURBOSPARK_ROUTED_PIPELINE=0` on the per-token arm, wired into
this family's chunked driver for exactly this experiment (banks = 1 and an
empty protect set TOGETHER -- sound only as a pair, because
retire-before-plan is what leaves no in-flight slot for the empty set to
endanger; the seam did not exist before, and the two rows above that first
proposed this experiment assumed it did). The answer is no: same prompt,
same config, misses read 9,024 with the protect set live and 9,400 with it
off, stdout md5-identical both ways, so removing the pipelining constraint
recovers NOTHING of the 9,024 -> 6,478 drop.

**The union really is deduplicating on this family**, which makes `gpt-oss`
the measured, family-scoped exception to this document's thrice-measured
"the union saves nothing" -- and the exception fits the standing finding's
own bound rather than contradicting it. AGENTS.md Gotcha 54 bounds the
union's saving by INTRA-WINDOW EVICTION RE-READS. On the 128-expert
families the sub-batch is capped at M<=8 by `union <= slot_count`, and at
those widths the re-reads were already near zero (Gemma's hit rate moved
81.2% to 81.4%). Here the full 16-token window's union (17.2) fits the 24
slots, so one plan per sub-batch dedups the whole window -- and with only
24 slots against 32 experts, evictions between one token's plan and the
next are real, the per-token plans re-pay ~2,500 of them per run, and the
union recovers exactly those. Porting rule for a third family: expect the
union term to be worth something only where the window's union fits the
slot cache while the cache does NOT comfortably hold the routed working
set -- both halves are one multiplication off the header.

Gates paid: bit-exactness against M sequential MXFP4 decode-pair calls at
the real D=2880/F=2880 shape, mutation-checked four ways
(`crates/gpu/tests/moe_prefill_batch_gguf_parity.rs`); byte-identity against
the sequential path on the synthetic fixture at chunk spans 1/2/3/5/8/11 and
at a cache too small to hold a sub-batch's union, mutation-checked two ways
(`crates/runtime/tests/real_forward_gptoss_chunked.rs`); and on the real
install a THREE-way md5 identity rather than the usual two -- the
PRE-CHANGE binary, the post-change binary with the seam off, and the
post-change binary with it on all produce `7281650e...` greedy (over three
interleaved pairs) and `78aae4b3...` sampled. The pre-change arm is what
says the default path did not move, which an on-vs-off comparison within one
binary cannot: both of its arms carry whatever the change did.

The memory oracle is unaffected STRUCTURALLY rather than by measurement, and
the reason is the lazy allocation: `ensure_batched` is called only under
`routed_batch_prefill`, which is off unless `TURBOSPARK_ROUTED_BATCH` is set,
and no oracle sets it. `BatchedRoutedScratch` is ~1.1 MiB against that
family's 5,700 MiB ceiling even when it is allocated.

**The Q4_K/Q6_K arm is still unbuilt and the case for it is unchanged --
but wiring a SECOND family to this seam exposed that the unwired one was
SILENTLY IGNORING it.** `TURBOSPARK_ROUTED_BATCH=1` on the real `qwen3moe`
install ran to completion with no message and no batching, because
`families/llama/moe_prefill.rs` simply had no branch reading the flag: a
caller who set it would have measured the per-token engine and reported it
under the batched arm's label, which is the `encode_gemm_any` doctrine's
exact failure. It is a named refusal now, and
`the_batched_routed_seam_is_refused_by_name_on_this_family` pins it.

Two things worth carrying from how that was found. It was a claim in THIS
document ("refused by layout, as it did before") checked against the binary
rather than a test failing -- the sentence was written from the affine
driver's guard and was never true of the family that has no driver at all.
And the DENSE drivers ignoring the same flag is correct and deliberately not
asserted: `TURBOSPARK_ROUTED_BATCH` asks for the routed half as one dispatch
pair, and a dense family has no routed half for it to refer to. The rule is
"refuse where the request is meaningful and unserved", not "refuse
everywhere unwired".

## Step 6's kernel term: row blocking, measured and wired

Step 6 made prefill's resident GEMVs into M-row GEMMs and won 1.86-2.13x on
the real dense `qwen38-27b` install. `docs/BENCHMARKS.md` then decomposed
what was LEFT -- a 4.94x gap against mlx-lm on the same machine and
checkpoint -- into two nearly equal factors, 2.3x of micro-batch WIDTH and
2.2x of KERNEL quality, and named M=64 as the width target.

**The width half of that reading was wrong, and the artifact said so before
anything was built.** `dequant_int4_gemm_simd`'s shipped `c(M)` row is
`M=2 0.50, M=4 0.55, M=8 0.46, M=16 0.44` -- FLAT. A per-row cost that does
not fall with M is what a COMPUTE-bound kernel looks like, so widening M
amortizes weight bytes that were never the cost. M=36 is mlx-lm's crossover,
computed from its 1.32 ms compute floor, and that floor is ~41 TFLOP/s on
this model: FP16 matrix hardware. This kernel is scalar FP32 `fma` and its
own floor is several times higher, crossed at a much smaller M. So the two
terms are not independent, the order of work is the kernel first, and the
same reading applies to the `simdgroup_matrix` form -- which was ALREADY
measured at M=32 (c=0.52) and M=64 (0.58), both worse than the SIMD kernel
at M=16, by `gemv_bandwidth_bench.rs`'s
`c_of_m_matrix_against_exact_at_qwen38_shapes`.

### What was built

`FC_GEMM_R` (function constant 104): one SIMD group owns `row_block`
CONTIGUOUS output rows instead of one. Two terms move, both per (block, lane):

- the two `half4` activation loads drop from `R*B` to `B`, because R rows now
  share one read of `x` instead of sitting in R separate SIMD groups;
- `sum = e0 + ... + e7` drops from `R*B` to `B`. It depends on `bi` alone,
  and the one-row kernel recomputed it once per ROW -- 7 of ~17 inner ALU ops.

The dot product's 8 `fma` per `(r, bi)` is the actual arithmetic and does not
shrink.

**It moves no bits at any width.** Each output `(row, bi)` still walks the
same blocks in the same order into one FP32 accumulator, and the 32-lane
partition of K feeding `simd_sum` is untouched; only which SIMD GROUP owns
the row changes. That is asserted rather than argued --
`row_blocking_does_not_move_a_single_bit` runs the whole `(R, B)` grid on the
ragged fixture built to SEE a reassociation, with `dequant_int4_gemm_mma`
required to differ on the same data as the positive control.

Note this is the second attempt at row blocking and the first one lost. The
shader header's note 2 held `float e[8][kMaxBatchRows]` -- a whole M-wide
activation TILE, ~208 floats of register array declared at the cap whatever R
was -- and measured worse even at R=1. The failure was the declared live set,
not the idea; `acc[R][B] + qv[R][8]` carries nothing per-row that scales with
B.

### Measured, AC, 2026-08-29

Three runs over three real shapes (gate/up 17408x5120, down 5120x17408,
packed_q 12288x5120), every batch width 1 to 16. Mean `c` per width; spread
within a cell 0.00 to 0.03.

|  M   |  R=1  |  R=2  |  R=4  | chosen | gain over R=1 |
|---|---|---|---|---|---|
|  1   | 1.004 | 1.043 | 1.222 |   1    | --      |
|  2   | 0.539 | 0.504 | 0.650 |   2    | 1.07x   |
|  3   | 0.508 | 0.356 | 0.459 |   2    | 1.43x   |
|  4   | 0.623 | 0.313 | 0.374 |   2    | 1.99x   |
|  5   | 0.534 | 0.520 | 0.488 |   4    | 1.10x   |
|  6   | 0.508 | 0.489 | 0.436 |   4    | 1.17x   |
|  7   | 0.506 | 0.469 | 0.428 |   4    | 1.18x   |
|  8   | 0.504 | 0.456 | 0.414 |   4    | 1.22x   |
|  9   | 0.491 | 0.467 | 0.441 |   4    | 1.11x   |
| 10   | 0.488 | 0.459 | 0.421 |   4    | 1.16x   |
| 11   | 0.490 | 0.448 | 0.424 |   4    | 1.15x   |
| 12   | 0.494 | 0.439 | 0.392 |   4    | 1.26x   |
| 13   | 0.484 | 0.449 | 0.449 |   4    | 1.08x   |
| 14   | 0.484 | 0.446 | 0.418 |   4    | 1.16x   |
| 15   | 0.481 | 0.436 | 0.419 |   4    | 1.15x   |
| 16   | 0.493 | 0.427 | 0.370 |   4    | 1.33x   |

**At the M=16 the prefill driver already uses, R=4 is 1.30-1.33x.** That
kernel is 85.4% of prefill GPU device time, which predicts **1.24x of
end-to-end prefill**.

### End to end on the real install

Predicted 1.24x, measured **1.26x**. Real `~/models/qwen38-27b.gturbo`, the
frozen `long-synthesis` prompt (2,940 tokens), `TURBOSPARK_PREFILL_CHUNK=128
TURBOSPARK_BATCHED_GEMV=1` on BOTH arms, AC, three interleaved pairs after a
discarded warmup per arm:

| pair | R=1 | per-width table | |
| ---: | ---: | ---: | ---: |
| 1 | 88.93 s | 70.81 s | 1.256x |
| 2 | 88.87 s | 70.31 s | 1.264x |
| 3 | 85.61 s | 68.14 s | 1.256x |
| mean | 87.80 s | **69.75 s** | **1.26x** |
| prefill tok/s | 33.5 | **42.1** | |

The prediction and the measurement agreeing to 2% is the check that says the
decomposition was right, not merely that the change helped: 1.30x on a term
independently measured at 85.4% of GPU device time has to land at 1.24x, and
it did.

**Output is BYTE-IDENTICAL across the change**, which is the gate this owes
instead of a quality gate. Real install, three ways: greedy and sampled at
the CLI defaults on a short prompt (`c908ca69...`, `4857cbb5...`), and the
2,940-token prompt THROUGH the batched arm, where the kernel is actually
reached (`fe37059f...`). The short-prompt arms matter less than they look --
plain decode is M=1 GEMV and never enters this kernel at all, so a smoke
test without `TURBOSPARK_BATCHED_GEMV=1` cannot see a change to it. That is
this repo's recurring "the fixture cannot reach the mutation" trap, and it is
why the third arm exists.

### Three things in that table that a summary would lose

**"R=4 is best" IS FALSE AT THE NARROW END, and a global constant would have
shipped two regressions.** At M=1 a wider block is a straight loss (1.00 to
1.22) and at M=2 it is a 29% one (0.504 to 0.650). Those are exactly the
widths a speculative verify runs at -- `docs/DFLASH2.md`'s published block
sizes are 2 to 8 -- and `encode_dequant_int4_gemm_resident` serves the MTP
and DFlash2 verify through `encode_gemm_any` as well as prefill. So the
choice is a per-width TABLE (`gpu::best_row_block`: 1 at M=1, 2 at M=2..4, 4
above), pinned by `the_chosen_width_is_never_one_the_measurement_calls_a_loss`.

**Selecting a kernel shape by batch width is safe here and is forbidden one
file over.** `dequant_int4_mma.metal` says "never make this a fast path
selected by a heuristic -- that would make the generated bytes a function of
a shape". That is correct OF THAT KERNEL, which reduces K in hardware and is
not bit-exact against the GEMV. `FC_GEMM_R` is bit-exact at every width, so
the table cannot move a byte -- which is precisely why the axis was built as
a function constant on one kernel rather than as a second kernel.

**M=4/R=2 (0.313) is the best cell in the table and is NOT a licence to
narrow the micro-batch.** It is a further 1.19x on the GEMM against
M=16/R=4's 0.370, and `c` is genuinely comparable across M here (verified by
re-running with `groups` FIXED at 64, so every M moves identical weight
bytes: the table reproduces cell for cell). But prefill's per-micro-batch
costs -- command-buffer commits, host encode -- scale with the NUMBER of
micro-batches, and 16 to 4 quadruples them. This bench prices the GEMM and
cannot see that, so the M=4 route needs an end-to-end run before it is
believed. Why M=4 is special at all: `unroll_count(4)` makes `b_dim == 4`
exactly one full unroll, and the R=1 column's own bump there (0.623 against
~0.51 either side) is that interaction going badly, which R=2 removes.

### Two measurement notes worth not re-deriving

**The frozen `count(4)` table in `dequant_int4_batch.metal` is not comparable
across sessions.** It reads 0.50 / 0.55 / 0.46 / 0.44 and the IDENTICAL code
read 0.51 / 0.61 / 0.50 / 0.49 on the day above -- up to 11% apart, same
machine, same shapes. The R=1 column looked like a regression from
`FC_GEMM_R` because of it. What settled that is an interleaved A/B against
`aa094b4^`, three pairs, baseline and R=1 agreeing to 0.01 on every cell --
which also confirms that `acc[kMaxRowBlock][kMaxBatchRows]` collapses at R=1.
AGENTS.md Gotcha 22, in a place nobody expected it: compare arms measured
beside each other, never against a frozen row from another day.

**Pipeline reflection cannot price a row block, and the check that says so is
the point.** `maxTotalThreadsPerThreadgroup` looked like a static answer to
"does this width spill" -- a Metal device, no model, no clock. It reads 1024
on every `(R, B)` shape, and raising `kMaxRowBlock` to 64 to declare an
`acc[64][16]` that fits no register file on any GPU STILL reads 1024. An
instrument reporting its ceiling on an impossible configuration is at its
rail (AGENTS.md Gotcha 59's shape), Metal exposes no register count publicly,
and `pipeline_reflection_cannot_see_this_kernels_register_pressure` records
that as a measured negative so it is not re-attempted. What survives is one
narrow guard worth keeping: `static_threadgroup_memory_length` must stay 0,
which is the cheap check against note 1's threadgroup staging being re-added
unmeasured.

## Step 7 candidates read off oMLX, and two of the three are closed

Reviewed 2026-08-29 against `jundot/omlx`'s `omlx/custom_kernels/qwen35_prefill`
(Apache-2.0), which targets this exact model family and is the build behind
the 210.3 tok/s bar in `docs/BENCHMARKS.md`. Three of its four kernel groups
are refuted, two of them by measurements that already existed here and one by
oMLX's own shipping choice. Recorded so none is re-proposed.

**The GEMM is the only live item, and it is not their kernel.**
`qwen35_qmm.metal` calls MLX's `qmm_t_impl`; the contribution is a tile sweep
worth less than the noise band between this machine's mlx-lm reading and the
community 210.3. What the review DID produce is
`scripts/mlx_qmm_reference.py` and the measured reference curve in
`docs/BENCHMARKS.md`, which replaces this doc's extrapolated 2.2x / 2.3x pair
with a same-session 1.50x kernel term at M=16 and a 2.00x width term from
M=16 to M=32. It also fixes the saturation point: **M=32, not M=36 or M=64**,
read off a `ms` column that is identical at M=16 and M=32 and exactly linear
past it, which is a `BM=32` tile paying for empty rows.

**Step 4 (batched attention) stays closed on this family.** oMLX ships an
`fa256` prefill attention, and 35 of its 35 lines are `instantiate_kernel`
over MLX's steel `attention` at `bd=256`. It targets the term measured at
**2.6%** of this family's prefill, in the 16 of 64 layers that have attention
at all. Cost an optimization by the terms it does not touch.

**The chunked-WY gated DeltaNet is a negative on THEIR side, which is the
cheapest kind of finding to accept.** `gdn.py` ships ~270 lines implementing
the flash-linear-attention chunked WY representation as two Metal kernels,
and its own module docstring says the production path is
`gated_delta_blocked_seq`, "the exact sequential recurrence used by mlx-lm",
at "half the FLOPs of the WY-chunked path". So the reformulation was built,
measured and declined by the people who wrote it. Do not build it here.

**What their blocked-sequential kernel does carry is a traffic argument that
applies to `gdn_delta_step_prefill` verbatim, and it is UNMEASURED here.**
They name the stock shape as re-reading k/q from device "once per
(Dv/4)-slice threadgroup => 32x redundant traffic (~13 GB per 16k-token
layer)", and fix it with a `DB=32` Dv block plus threadgroup staging of
q/k/v in token blocks. This port dispatches that kernel at exactly `(Hv,
Dv/4)` threadgroups and reads q/k straight from device in its token loop.

**IT IS 4.66%, AND THE ITEM IS CLOSED.** Measured by
`crates/gpu/tests/gdn_prefill_share_bench.rs`, which times
`gdn_delta_step_prefill` at the real `Hk=16, Hv=48, Dk=128, Dv=128` geometry
against every INT4 matrix a 16-token micro-batch walks, weighted by the real
layer counts (48 gated-DeltaNet, 16 attention, 64 FFN), at the shipped
`best_row_block(16) = 4`:

| kernel | per call ms | x layers | total ms |
|---|---|---|---|
| `gdn_delta_step_prefill` | 0.1577 | 48 | **7.569** |
| gate/up 17408x5120 | 0.8212 | 64 | 52.558 |
| down 5120x17408 | 0.8003 | 64 | 51.217 |
| gdn_inproj 16480x5120 | 0.7801 | 48 | 37.445 |
| packed_q 12288x5120 | 0.5639 | 16 | 9.022 |
| o_proj 5120x6144 | 0.2851 | 16 | 4.562 |

**4.66% of (GEMM + GDN), and that is an UPPER BOUND** -- the denominator
omits norms, the conv pair, RoPE, attention and the gated norm, every one of
which is real prefill cost the bench does not encode, so the true share is
smaller. Under this review's own decision rule the threadgroup-staging
rewrite is closed. Note what it would be worth even if it worked perfectly:
oMLX claims an 8x traffic reduction on a term holding under 4.66%, so the
ceiling is about 4% of prefill, against a GEMM holding 85.4%.

**TWO INDEPENDENT METHODS AGREE, which is why this is stated as settled
rather than as one reading.** Before the bench existed the same question was
answered by arithmetic: at `Hv=48, Dk=128` the uncached q/k re-read is 512 B
per threadgroup per token across `48 x 32` threadgroups, ~37.7 MB per token
over 48 GDN layers, against ~844 MB per token of weight traffic at a
16-token micro-batch, i.e. **4.5%**. The bench says 4.66%. A traffic
calculation and a clock landing within 0.2 points of each other is the
strongest form this kind of claim takes here.

**THE INSTRUMENT NOTE IS THE PART WORTH CARRYING.** The obvious way to get
this number is `TURBOSPARK_DISPATCH_PROFILE=1`, which is the only surface that
names kernels (`PhaseCounters` has no GDN bucket). It "waits on every command
buffer at commit" -- its own module doc -- which serializes exactly the
pipelining that makes prefill fast. Three attempts on the real `qwen38-27b`
install were abandoned: 2,940 tokens ran 17.5 min without reaching the
report, 582 tokens 17.5 min, and ~150 tokens over 12 min. **The cost does not
fall with prompt length**, so shortening the prompt is not the fix, and a
session that starts down that road loses an hour. The bench above answers the
same question in **0.45 seconds** with no model and no install. When the
question is "what share does kernel X hold", prefer timing X against its
neighbours at the real shapes over profiling a whole forward pass; the
profiler is for finding the kernel you did not suspect, not for pricing the
one you did.

### Staging `x` in the matrix kernel: measured, and a 3.3x to 5.9x LOSS

The reference curve above says MLX's `qmm_t_impl` reaches `c` 0.145 where
this port's matrix kernel plateaus at ~0.52, and names three structural
differences: it runs 128 threads in four SIMD groups against this kernel's
one, its `BM` is 32 to 128 against `kMmaTile = 8`, and it stages BOTH
operands where this one stages only the weights and `simdgroup_load`s `x`
transposed straight from device.

Staging `x` was the cheapest to try and the one predicted most likely to be
the cause: a transposed device load with row stride N, issued once per
`(n0, kt)`, reads like a strided gather in the innermost loop. Built behind
`FC_MMA_STAGE_X` (function constant 110) so both shapes live in one binary
and interleave pair by pair in one process, and asserted bit-identical to
the un-staged arm (`dequant_int4_mma_parity.rs::staging_x_through_threadgroup_memory_moves_no_bits`,
mutation-checked with an off-by-one that reddens only that case).

**It loses on every shape and every width, and the penalty grows with B**
(gate/up 17408x5120; the other five shapes agree to a few tenths):

| M | mma | mma + stageX | staged/plain |
|---|---|---|---|
| 2 | 3.54x | 12.49x | 3.52x |
| 4 | 1.78x | 6.27x | 3.52x |
| 8 | 0.89x | 3.19x | 3.58x |
| 16 | 0.66x | 2.53x | 3.81x |
| 32 | 0.54x | 2.25x | 4.14x |
| 64 | 0.52x | 2.71x | 5.19x |

**THE FINDING IS THAT THE THREE DELTAS ARE NOT INDEPENDENT, and testing one
alone was the wrong experiment.** A transposed `simdgroup_load` from device
is not the naive gather it reads as; Apple's tile load handles it. Staging
the same bytes by hand through 32 LANES is a serial copy of
`col_tiles * kMmaTile * kMmaK` halfs per n-block, 10,240 loads per lane at
B=64 over the whole K walk. MLX stages `x` as well and wins because it has
128 threads to do the staging and a `BM` of 32 to 128 to amortize it over.
So staging is a CONSEQUENCE of the wider threadgroup rather than a separate
lever. The remaining two deltas have to move together or not at all.

This is the second time this kernel has punished a one-variable change: its
header already records that widening the staged K block from 8 to 64
"changed nothing" and that running past M=16 to 32 and 64 "changed nothing".
The pattern is consistent -- its shape is a package, and picking one piece
out of it measures the piece rather than the question. Anyone re-opening the
matrix line should change the threadgroup width and `kMmaTile` in the same
step, and expect to keep `FC_MMA_STAGE_X` on when they do, since a
four-SIMD-group kernel has the threads that make staging pay.

**What it does NOT change:** the exact kernel still wins on both axes, it is
still what every wired call site dispatches, and the bit-exactness objection
to the matrix kernel (`AGENTS.md` Gotcha 27) is untouched. `FC_MMA_STAGE_X`
defaults to OFF and no production path sets it.

### Splitting the matrix kernel by deletion: the dequant is not the cost

Two explanations for `dequant_int4_gemm_mma`'s plateau died in one session,
leaving the mechanism unidentified: its own header's "dequant work is
independent of B" is refuted by arithmetic (dequant per output is `N / B`
here and `K / BM` in MLX, the same number at the same token width), and
staging `x` lost 3.3x to 5.9x. Total dequant work is `N * K` in both
engines and MLX spreads it over FEWER threads, so parallelism is not it
either.

`FC_MMA_SKIP_DEQUANT` (function constant 111) settles it by DELETION: fill
the weight tile with a constant, leave every barrier, `simdgroup_load` and
`simdgroup_multiply_accumulate` exactly where they were, and what remains is
the matrix path's cost with the unpack removed. Output is meaningless under
it, which `dequant_int4_mma_parity.rs::the_skip_dequant_diagnostic_is_reachable_and_its_output_is_wrong`
asserts rather than leaves implicit -- an unreachable diagnostic constant
would time the unmodified kernel twice and report "the dequant is free",
which is a wrong answer shaped exactly like a finding.

| M | exact | mma | mma, no dequant | nodq/mma |
|---|---|---|---|---|
| 2 | 0.50x | 3.59x | 2.30x | 0.64 |
| 8 | 0.43x | 0.90x | 0.58x | 0.65 |
| 16 | 0.38x | 0.68x | 0.51x | 0.76 |
| 32 | -- | 0.56x | 0.48x | 0.86 |
| 64 | -- | 0.56x | 0.50x | 0.89 |

gate/up 17408x5120; all six shapes agree to 0.03.

**The dequant is 36% of this kernel at M=2 and 11% at M=64** -- its share
shrinks as B grows, the opposite of the original story.

**THE DECISIVE NUMBER IS THE MIDDLE COLUMN AND NOT THE RATIO.** With the
dequant entirely free the kernel still reads **0.46 to 0.50** past M=16
against MLX's **0.145**. A perfect loader leaves it 3.2x behind, so
`QuantizedBlockLoader` is not the thing to copy. It is also still slower
with a free dequant (0.51 at M=16) than the plain scalar
`dequant_int4_gemm_simd` is with a real one (0.38).

**What that leaves, and what it closes.** Three candidate levers entered
this session and two are now settled dead ends: the loader (measured here)
and `kMmaTile` (refuted by arithmetic -- it scales dequant and outputs
together). What remains is the matrix path itself: one SIMD group per
threadgroup, `simdgroup_barrier` twice per 64-element K block, and eight
`simdgroup_float8x8` accumulators owned by 32 lanes. A re-tile has to change
THAT, and anyone starting one now knows which three things not to spend a
day on.

---

## The seventh flow, `qwen4_exp`, and a host-write hazard the pattern had not seen (2026-09-05)

`families/qwen4/prefill.rs` is step 1 again and needs no new kernel: a
per-layer `cb1` covering all M tokens' PLE / attn_hc / GDN-or-QSA /
hc_inject / mlp_hc / router, then a per-token routed loop pipelined through
the same `RoutedSlot` / `routed_pipeline_banks` / `pending_routed` module
every other MoE driver uses.

**Two predicted blockers turned out not to exist, and both predictions were
recorded in ROADMAP before anyone tried.** The claim was that the QSA
indexer's per-token key write and its block pooling would have to be
re-expressed per micro-batch, and that QSA would need a position list per
layer rather than the single shared buffer decode relies on. Neither was
true. `encode_full_attention_block` already takes `pass: &mut PassEncoder`
and already owns its above-budget mid-layer commit, so calling it once per
token in increasing order reproduces the sequential path exactly; and the
shared `qsa_positions` buffer stays safe for the same reason the routed
half's pipelining does not disturb it, namely that a layer's whole
attention-and-router half commits and waits before that layer's routed loop
starts, so no two QSA layers' writes are ever in flight at once. GDN needed
nothing either, matching dense qwen's precedent.

**PLE is where the real bug was, and it generalizes.** Four buffers needed
the familiar M-row widening for the familiar reasons. A fifth, PLE's
`ngram_emb`, needed it for a reason the standing rule does not cover. The
rule this document and `crates/runtime/AGENTS.md` both state is *what decides
which buffers need a per-token row is who WRITES them, not who reads them*,
and it is about GPU dispatches: command buffers on one queue execute in
commit order, so a GPU-only intermediate is safe to reuse across a chunk's
tokens. `ngram_emb` is uploaded with `gpu::write_buffer_bytes`, a HOST write
that executes the instant the encoding function runs rather than a dispatch
queued for later. It does not respect commit order at all. A single-row
buffer therefore held only the LAST token's embedding for the entire pass's
execution, and every earlier token in the micro-batch computed PLE from the
wrong n-gram embedding, silently, with no error and plausible output.

Caught by `the_chunk_boundary_does_not_move_the_logits` at chunk span 2, the
first multi-token micro-batch it tried. **Any future family whose flow does
its own host-side dequant-then-upload step owes this same check before
assuming commit order protects it.**

Both batching seams are refused by name here (`TURBOSPARK_ROUTED_BATCH` and
`TURBOSPARK_BATCHED_GEMV`), the pair every other chunked driver carries. The
routed seam is MEANINGFUL on this family rather than vacuous, since it has a
routed half, which is why it is owed at all.

## Measuring chunked prefill: the instrument, and what it could not see before

Until 2026-09-05 `crates/bench/src/real_model.rs` reached only
`run_raw_completion` and `run_raw_completion_speculative`, so the chunked
driver was unreachable from `turbospark-bench` and therefore from
`scripts/power.sh`, which drives it. A power or throughput row taken through
either tool measured the SEQUENTIAL path regardless of what
`TURBOSPARK_PREFILL_CHUNK` was set to, with nothing in the artifact saying
so. That is why this document's own energy row stayed open.

`--prefill-chunk off|auto|N` is the instrument, and it **defaults off**
because every frozen row in `crates/bench` was measured sequentially.

**Only one of the three seams ever needed wiring**, which is worth not
re-deriving: `TURBOSPARK_PREFILL_CHUNK` is read by front ends, so the bench
genuinely could not see it, while `TURBOSPARK_ROUTED_BATCH` and
`TURBOSPARK_BATCHED_GEMV` are read INSIDE the runtime's chunk drivers and go
live the moment the driver is engaged. Those two needed a header echo, not a
flag. The header prints the resolved path on both arms and marks the two
seams INERT on the sequential one, so an operator who exports
`TURBOSPARK_ROUTED_BATCH=1` and forgets `--prefill-chunk` can see from the
artifact that they measured sequential prefill.

`scripts/power.sh` carries the matching `seq|chunked` arm pair. `seq` passes
nothing and is byte-for-byte the same invocation as `default`, which is why
that axis is exclusive of every other arm: pairing them varies nothing and
puts one condition in two rows.
