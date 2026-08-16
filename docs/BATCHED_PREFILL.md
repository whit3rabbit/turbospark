# Batched prefill: what it would take, and what it would buy

Scope, not a plan of record. Written 2026-08-16. Every number is cited to
the page that owns it except one, prefill's own dispatch ranking, which was
measured for this document because no page had it and borrowing decode's
gave the wrong answer.

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
| `RealForwardRunner: ChunkedPrefillRunner` | -- | **missing** |

`ScriptedLogitProducer` is the trait's only implementor
(`producer.rs:108`), which is what makes the loop testable today and also
why nothing has noticed the gap.

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

Using the prefill attribution already measured (CLAUDE.local.md,
2026-08-06, 2,252-token prompt, `--max-new 8` so 99.6% of the divisor is
prefill; buckets summed to 20.89 ms against 20.89 ms of wall clock, so
nothing was hiding):

| bucket | ms/token | share | batches? |
| --- | ---: | ---: | --- |
| gpu wait (commit+wait) | 13.26 | 63.5% | partly -- see below |
| expert io (`pread`) | 5.26 | 25.2% | **yes, and this is the big one** |
| encode + logit readback | 1.04 | 5.0% | yes, divides by M |
| hit-expert phase1 cb | 0.65 | 3.1% | yes |
| routed bind + upload | 0.33 | 1.6% | yes, divides by M |
| routed cb retire | 0.23 | 1.1% | yes, divides by M |
| router readback + top-k | 0.12 | 0.6% | yes, divides by M |

**The expert `pread` is the term to look at first, and it is where prefill
differs most from decode.** A chunk of M tokens reads the UNION of their
routes, not the sum. Measured (`docs/SPECULATIVE_DECODING.md`, Gemma, 128
experts, top-8): 8.0 distinct at M=1, 19.0-19.7 at M=4, 27.4-28.7 at M=8,
37.8-39.4 at M=16. So at M=16 a chunk reads ~38 distinct experts per layer
where 16 sequential tokens issue 128 expert reads -- **a 3.3x cut in expert
bytes**, on a bucket that is a quarter of prefill.

That number needs no kernel: it falls out of issuing one
`plan_experts_cached` per chunk instead of per token, and the slot cache
already exploits the same temporal locality one token at a time.

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
it 0.6 weighted. At M=16, as a fraction of the current per-prompt-token
cost:

| term | now | batched |
| --- | ---: | ---: |
| expert `pread`, cut 3.3x by the union | 25.2% | 7.6% |
| host overheads (encode, bind, retire, router readback), /16 | 8.3% | 0.5% |
| GPU: norms + elementwise, no amortization | 13.5% | 13.5% |
| GPU: GEMV + MoE + router, at c(16) ~ 0.6 | 35.9% | 21.5% |
| GPU: attention, KV read shared ~M ways | 14.2% | 4.2% |
| **total** | **100%** | **~47%** |

**About 2.1x, landing near 10 ms per prompt token against Swift's 7.5.**
Leaving attention per-token instead gives ~57%, i.e. ~1.75x -- so batching
attention is worth about a fifth of the win, and is not the whole game.

Treat this as an order-of-magnitude estimate. Two terms are soft: `c(M)`
for the routed pair is extrapolated from the INT4 GEMV proxy (the real
phase-1/phase-2 kernels have no batched form to measure), and attention's
0.3 factor assumes the long-context case is KV-bandwidth bound, which is
consistent with the split-KV result but was not measured directly. It is
enough to say the work is worth doing and not enough to promise a number --
and note that neither arm reaches Swift's 7.5 ms, which would need 2.85x.

## The attention fork

The question is whether a chunk needs the descoped tile kernels
(`attention_prefill_causal_tiled`, `prefill.metal`'s 16-kernel pipeline --
`DEVIATIONS.md`) or whether it can run the existing split-KV
`attention_decode` per token inside the chunk.

**It can run per token, and it should, first** -- but for a weaker reason
than a 2.3% share would have given. Attention is 22.3% of prefill GPU work
and batching it is worth ~10 points of the ~53 the whole change is worth.
That is real and it is not a prerequisite: a chunk whose attention is still
per-token gets ~1.75x, and steps 2 to 4 below are independent of it.

So this is one phase followed by an optional one, rather than a fork:

- **Phase A**, per-token attention inside a batched chunk. ~1.75x. No new
  attention kernel, no touching the descoped pipeline.
- **Phase B**, batched attention. Takes ~1.75x to ~2.1x. And it does NOT
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

1. **`RealForwardRunner: ChunkedPrefillRunner`, looping the EXISTING
   per-token kernels inside each layer.** This is the enabler and it is
   deliberately first: no new kernel, and it alone collects the single
   largest term. Per chunk it must advance M KV rows, the position by M,
   and commit-and-wait exactly as the per-token path does
   (`crates/runtime/CLAUDE.md` Gotcha 2 states the analogous contract for
   `produce_prefill`). On a family with recurrent state the GDN chain is
   sequential by definition and steps M times inside the chunk.

   **The expert `pread` union falls out of this step and needs nothing
   else.** Once a chunk runs layer by layer, layer L has all M tokens'
   router outputs at once, so one `plan_experts_cached` over their union
   replaces M per-token plans -- 25.2% of prefill cut by 3.3x at M=16, with
   the existing phase-1/phase-2 kernels then dispatched M times against
   slots that are already resident. It is NOT independently landable ahead
   of the driver, which an earlier draft of this list had it as: the union
   is only knowable once M tokens have reached layer L together.

   Measure here before writing a kernel. This step alone should be worth
   most of the ~1.75x.

2. **Batched `moe_phase1_gate_up_act_u16load`**, against
   `dsv4_prefill_moe_phase1_pairs_int2`'s route-list shape. Parity must be
   EXACT against M separate calls, mutation-checked, per this repo's habit.
3. **Batched `moe_phase2_down_reduce_k8`.** The reduce order is a
   correctness constraint, not a style choice (AGENTS.md Gotcha 27): a
   batched phase 2 must reduce each token's slots in the router's ranking,
   independently per token, or output becomes a function of the chunk
   boundary.
4. **Batched attention** (Phase B above), if the measured 1.75x is not
   enough. One kernel, widening `attention_decode_partial` to hold M query
   rows per KV chunk. Not the descoped tile pipeline.
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
