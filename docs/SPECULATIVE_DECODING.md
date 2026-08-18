# Speculative Decoding and DFlash (Measured Marginal)

The question this page answers: should this engine adopt speculative
decoding, specifically DFlash -- a small block-diffusion drafter proposes a
whole block of tokens in one parallel forward pass, the target verifies the
block in one batched pass, and the longest correct prefix is accepted?

The answer, measured 2026-08-10 on the real Qwen 3.6 35B-A3B install and
re-derived unchanged 2026-08-18 after a kernel fix that halved the largest
term, is about 1.1x at best, at a small block size, and only once a batched
MoE kernel exists that does not yet. Not the 3.6x DFlash reaches at
concurrency 1 on datacenter GPUs, and not the 1.5x Meta measured for it on
an M4 Max with a dense model. Recorded here so the arithmetic is not
re-derived; it also appears in ROADMAP under the speculative-decoding item.

Nothing here refutes DFlash. It is a statement about this engine's verify
cost, and every number that would have to change for the answer to change
is named at the bottom.

> **THE `c(M)` TABLE BELOW IS SUPERSEDED, 2026-08-17, AND THE VERDICT IT
> SUPPORTS IS NOT. Re-derived 2026-08-18: the reading is unchanged.**
>
> `dequant_int4_batch.rs` was missing the function-constant specialization
> `46617c6` gave the GEMV, and it needed a bounded unroll besides. Fixing
> both roughly halved `c(M)` (`docs/MTP_SPECULATIVE.md`). That is a large
> improvement to the largest single term here and it moves this page's
> answer by about two points, because **the term it improves is 52.7% of
> decode compute and the two terms that dominate the composite are
> untouched**: a 19% un-amortizable floor that no kernel moves, and the
> routed pair, 26% of decode compute with still no batched form at all.
> The fresh numbers are in "c(M), re-measured" and "Reading" below.
>
> **The one thing the re-derivation did overturn is a number on this page,
> not a conclusion.** The prose under "The compute split" claimed
> `c(8) ~ 0.67`. The break-even column in "Reading" implies 0.557, the
> fresh measurement reads 0.572, and those two agree to 3%. So the reading
> table was right and its stated `c(8)` was wrong -- an inconsistency that
> survived because nobody had recomputed one from the other. Corrected in
> place.
>
> **Item 3, "a dense family", HAS been measured and it pays.** The dense
> `qwen3_5` 27B has no expert-union term and a 6.4% floor rather than 19%,
> and after the kernel fix its ceiling is ~2.07x with ~1.35-1.58x at a
> drafter as good as MTPLX reports. That work is scoped in
> `docs/MTP_SPECULATIVE.md`, not here. **That page and this one now
> disagree, and the disagreement is the finding**: one kernel fix, two
> families, and it is decisive on the dense one and inert on this one.

**Do not carry this verdict to batched prefill**, which reuses the same
`c(M)` and `union(M)` terms and reaches a different answer. A verify pass
divides its cost by an accept length (most of why 1.1x), while a prefill
chunk keeps every one of its M tokens, so its divisor is M with no
probability in it. The compute split below is also decode's, where
attention is 2.3%; in prefill it is 22.3%. See `docs/BATCHED_PREFILL.md`.
The kernel facts here (the `c(M)` table, `union(M)`, and the two
register-file dead ends) transfer; the conclusion does not.

## Where the question came from

DFlash (arXiv 2602.06036) is becoming the default speculative-decoding
attachment for new open-weight models rather than an exotic option. Meta's
Muse-Glimmer-30B ships a drafter head in-box, and z-lab publishes drafters
for both families this port already runs:

| drafter | size | target |
| --- | ---: | --- |
| `z-lab/Qwen3.6-35B-A3B-DFlash` | 772 MB BF16 | `~/models/qwen36.gturbo` |
| `z-lab/gemma-4-26B-A4B-it-DFlash` | 859 MB BF16 | `~/models/gemma4.gturbo` (gated) |

Read off the Qwen drafter's safetensors header (7 KB of ranged reads, no
download): 6 layers, hidden 2048 matching the target's, `fc.weight`
[2048, 16384] projecting eight concatenated target hidden states down to
one, `block_size` 16, tapping target layers 1/6/11/16/22/27/32/37 of 40.
It carries no embedding table and no LM head; it reuses the target's, both
already resident here. That is a genuine simplification and it is why the
file is 0.4B parameters rather than 1.5B.

No Rust crate helps. `lablup/mlxcel`'s `drafter/dflash` is the closest
implementation and is `mlx-rs` tensor code, so adopting it means running the
TARGET in MLX and discarding this port; `Aryagm/dflash-mlx` is Python/MLX.
Both are useful references for the round loop and the rollback rules.
crates.io has only framework-bound toys.

## What has to be true

Speculation pays when one verify pass over M proposed tokens costs less, in
decode-steps, than the number of tokens it gets accepted. Writing the verify
cost in decode-steps:

```
verify(M) = compute_share x M x c(M)          # the batched forward
          + expert_share x union(M)           # routed-expert bytes
          + overhead_share                    # paid once per block, not per token
```

- `c(M)` is the per-token cost of a batched pass against M sequential ones.
  Ideal is `1/M`; `1.0` means batching bought nothing.
- `union(M)` is the distinct routed experts M consecutive tokens touch,
  over `top_k`. A batched verify must read their UNION.
- accept length is the accepted prefix plus the bonus token every verify
  yields for free.

Four of those five terms were unknown when this started. Each got its own
instrument, and the order below is the order they were measured, cheapest
first.

## Instruments

All are `#[ignore]`d and need a real install; commands are in AGENTS.md.

| surface | answers |
| --- | --- |
| `MFERENCE_ROUTER_TRACE=1` + `scripts/router_window.py` | `union(M)` |
| `crates/gpu/tests/gemv_bandwidth_bench.rs` | bandwidth headroom, then `c(M)` |
| `MFERENCE_DISPATCH_PROFILE=1` | the compute shares |
| `crates/bench/tests/accept_length_probe.rs` | accept length, and losslessness |
| `crates/bench/tests/rollback_probe.rs` | that a rejected block can be undone exactly |

## Measurements

### Expert union: not the problem it looked like

`MFERENCE_ROUTER_HIST` counts per-layer expert selections but discards
which pass each came from, and a batched verify reads the union over a
WINDOW of passes, so the trace mode adds the top-k ids in pass order.
Two prompts per family, ~300 generated tokens, prefill excluded.

| M | Gemma distinct of 128 | Qwen distinct of 256 | `union(M)` (Qwen) |
| ---: | ---: | ---: | ---: |
| 1 | 8.0 | 8.0 | 1.00 |
| 4 | 19.0-19.7 | 19.6-22.9 | 2.44-2.86 |
| 8 | 27.4-28.7 | 30.3-37.6 | 3.78-4.70 |
| 16 | 37.8-39.4 | 45.9-59.5 | 5.74-7.44 |

Sixteen consecutive tokens touch 46-60 of Qwen's 256 experts per layer, not
the 128 a union-free model would charge, because adjacent tokens route
alike. That is the same temporal locality the expert slot cache already
lives on, restated per block. The `M = 1` row reading exactly 8.0 is the
identity check on the analysis.

### c(M): the term that decided it

The first draft of this measurement asked whether a kernel was needed at
all, and eliminated the two cheaper options by measurement:

- B dispatches of the existing GEMV over one matrix cost 0.65B, not ~1:
  the weights do not stay cached across them;
- encoding those into a concurrent compute encoder rather than the engine's
  serial one recovers only 1.07-1.40x, and 8 x the resulting rate lands on
  the ~355 GiB/s the kernel saturates at, which is the proof the hardware
  genuinely moved the weight bytes eight times.

So `dequant_int4_gemm_simd` exists: one dispatch, each packed nibble read
once and multiplied into B accumulators. Parity against B separate GEMV
calls is exact, not tolerant, and mutation-checked three ways.

### c(M), re-measured

`c_of_m_for_the_batched_kernel`, 2026-08-18, same machine, AC, after
`46617c6`'s specialization reached this kernel. Three warm rounds with the
cold first run discarded (Gotcha 20), `--test-threads=1` because the two
`c_of_m` tests contend for the GPU otherwise. Ranges, not point values:
**this bench's per-cell spread is up to 0.13**, which is wide enough that
only the shape of the table should be read, never a single cell.

| shape | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| expert 512x2048 | 0.48-0.64 | 0.32-0.40 | 0.32-0.39 | 0.23-0.33 |
| o_proj 2048x2048 | 0.56-0.65 | 0.46-0.57 | 0.41-0.49 | 0.35-0.40 |
| q_proj 4096x2048 | 0.55-0.66 | 0.58-0.67 | 0.41-0.49 | 0.38-0.51 |
| stacked 8192x2048 | 0.53-0.64 | 0.58-0.66 | 0.41-0.56 | 0.45-0.54 |

For comparison, the pre-fix table these replace, same order: expert
0.77/0.55/0.44/0.36, o_proj 1.01/0.80/0.71/0.67, q_proj
1.01/0.82/0.78/0.75, stacked 1.03/0.87/0.79/0.78. **Every cell improved and
M=2 is no longer worse than not batching at all**, which it was on three of
four shapes before.

**Read the `expert 512x2048` row as what it is: a proxy.** It is a resident
INT4 GEMV at a routed expert's shape, not the routed pair, which has no
batched form to measure. The other three rows are the resident projections
and are what the 52.7% GEMV share below is made of.

**Two optimizations were tried on this kernel and both lost.** Staging `x`
in threadgroup memory removes the per-batch device reads and is slower on
every shape (expert 0.36 -> 0.55 at M=16): the per-block barriers serialize
the eight SIMD groups for more than the saved reads cost, and the cache
already serves them. Register blocking over rows, so one activation read
serves R rows, is the textbook fix and preserves bit-exactness because each
row keeps its own summation order -- and it is worse even at R=1 (expert
0.44 -> 0.79 at M=8), because holding activations across rows needs
`float e[8][16]` beside `acc[R][16]`, ~208 floats of register array, which
spills. Both are recorded in the kernel header.

They are one finding twice: the register file cannot hold an M-wide
activation tile, and threadgroup memory's barriers cost more than they save.

### The compute split, and the floor it sets

`MFERENCE_DISPATCH_PROFILE=1` on the real Qwen install, 810 dispatches per
token. Absolute times are inflated by the profiling mode; only shares
transfer.

| family | share of GPU busy | batches? |
| --- | ---: | --- |
| GEMV family (int4, int8, `gdn_in_proj`, router, embed) | 52.7% | yes, per the table above |
| MoE expert kernels (phase 1 + phase 2) | 26.0% | **no batched form exists** |
| elementwise and norms | 13.7% | **no** -- no weights to amortize |
| GDN recurrent step and conv | 5.3% | **no** -- sequential by definition |
| attention | 2.3% | yes, well: M queries share one KV read |

**Nineteen percent of decode compute is per-token work with no weights to
amortize.** That is a floor under `c(M)` no kernel can move, and it is the
structural reason this engine cannot reach the near-free verify a
datacenter GPU gets.

### The composite, written out

The 2026-08-18 re-derivation, spelled out because the version it replaces
was not reproducible from what this page states -- its prose said
`c(8) ~ 0.67` while its own break-even column implied 0.557, and nothing
here let a reader tell which was load-bearing.

`c(M)` is the share-weighted mean over the five rows of the split above.
The GEMV term is the mean of `o_proj`, `q_proj` and `stacked` from the
table above; norms, elementwise and GDN enter at 1.0 because they do not
batch; attention enters at `1/M`. The routed pair is **granted**, not
measured, and both grants are reported:

| M | GEMV mean | `c(M)`, pair granted 0.5 | `c(M)`, pair at its GEMV proxy |
| ---: | ---: | ---: | ---: |
| 2 | 0.604 | 0.650 | 0.664 |
| 4 | 0.599 | 0.641 | 0.608 |
| 8 | 0.473 | **0.572** | **0.535** |
| 16 | 0.447 | 0.557 | 0.501 |

The second grant is the optimistic one: it assumes a batched routed pair
would reach whatever the INT4 GEMV reaches at a routed expert's shape.
Nobody has built it, so neither column is a measurement of it.

**`c(8) = 0.572` against the 0.67 this page used to state**, and against
the 0.557 its break-even column already implied. The fresh measurement
agrees with the table to 3% and not with the prose, which is why the
verdict below barely moves: the reading was computed from something close
to the right number all along.

Note the floor doing its work at the bottom of the range. At M=16 the
un-amortizable 19% plus the 0.5 grant on the routed pair contribute 0.32 of
the 0.557 between them, so **a perfect GEMV could not take `c(16)` below
~0.33** and no amount of kernel work on this arm reaches the dense family's
numbers.

### Accept length

Two sources, and they agree on the shape.

**An n-gram drafter, measured here** (`accept_length_probe.rs`, real Qwen,
greedy, 300 tokens per arm, sequential verify because only the ratio
matters): it fires on 10-11% of rounds and reaches 2.46 / 2.76 / 2.76
accepted-plus-bonus at blocks 4 / 8 / 16. **Refuted**, and the shape says
why: its accept length saturates: block 16 accepts exactly what block 8
does, because the matched continuation runs out long before the block does.
Both failure modes are specific to copying earlier text, and neither
afflicts a trained drafter, which always fires and predicts rather than
copies. So this refutes free drafting and says nothing about DFlash.

**A trained drafter, published.** This did not need building: DFlash reports
accept length as `completion_tokens / spec_verify_ct`, the accepted prefix
plus the bonus token, which is exactly what the probe above measures, so the
two compare with no conversion. Mean 6.49 across tasks at block 16, peaking
7.87 on MATH-500 (arXiv 2602.06036); the per-position curve at temperature 0
is 100 / 81.6 / 64.0 / 50.7 / 41.3 / 34.4 / 29.3 / 25.2 % (arXiv 2607.07409),
whose prefix sums give the shorter blocks.

## Reading

`verify(M) = 0.75 x M x c(M) + 0.25 x union(M)`, the formula at the top of
this page with the decode step's own 75/25 compute/expert-io split, `c(M)`
from the composite above and `union(M)` at the midpoint of its two-prompt
range. Speedup is DFlash's published accept length over that.

| block | DFlash accept length | break-even, pair granted 0.5 | verdict | at the optimistic grant |
| ---: | ---: | ---: | --- | ---: |
| 4 | 2.96 | 2.59 | **pays, 1.14x** | 1.19x |
| 8 | 4.26 | 4.49 | loses, 0.95x | 1.00x |
| 16 | 6.49 (mean) | 8.33 | loses, 0.78x | 0.85x |
| 16 | 7.87 (best task) | 8.33 | loses, 0.94x | 1.03x |

**Recorded before the kernel fix: 1.14x / 0.97x / 0.87x / 1.05x.** So the
whole table moved by at most a few points and not one verdict changed sign
except block 16's best-task row, which changed sign AGAINST speculation.
That is the re-derivation's actual result and it is worth stating as
plainly as the improvement was: **halving `c(M)` on the GEMV did not rescue
this family.**

**On this engine the optimum is a small block and the win is about 1.1x**,
which inverts the datacenter result. There verify is nearly free, so a
bigger block is always better; here verify cost scales almost linearly in M
-- 19% of compute cannot amortize and the expert union grows with the block
-- so every extra proposal costs nearly a full decode step while its
acceptance probability is already down to 50% by position 4. Expect this
inversion on any memory-bound single-stream engine.

Verifying only a 4-token prefix is not an off-design use of a block-16
drafter: the drafter runs one forward at its trained block size either way,
and verifying fewer of its proposals leaves the first four positions of the
acceptance curve untouched.

**Losslessness is settled, and separately from the economics.** Every block
size produces a token stream byte-identical to the same generation with
speculation switched off, so the accept walk and the rollback are lossless
in practice and not only by construction. The reference arm is a
non-speculative run rather than the other block sizes: comparing speculative
arms against each other passes even when all of them are wrong the same way.

## Standing decision

**Do not build it for 1.14x. Unchanged by the 2026-08-18 re-derivation**,
which is the answer to the question the correction block at the top used to
be asking. Three caveats all point the same way and the margin is inside
the composite's error bar:

- the break-even column assumes a batched MoE that does not exist; without
  one, every row loses;
- the per-position curve is published for Qwen3-4B rather than this target,
  and the 35B drafter is a later retrain;
- `c(M)` for the unbuilt MoE kernel is estimated, not measured, and the
  re-derivation reports both grants precisely because that estimate, not
  the GEMV, is now the largest soft term in the answer.

**The generalisable part is why a real 2x on the biggest term bought
nothing.** `c(M)` improved from 0.67 to 0.572 at M=8 and the speedup moved
0.97x to 0.95x, because the improved term is 52.7% of compute while 19%
cannot amortize at all and a further 26% is a kernel nobody has written.
Amdahl, restated for anyone about to optimize the same arm again: this
family's ceiling is set by the two terms the GEMV work does not touch, and
`docs/MTP_SPECULATIVE.md` reaches the opposite conclusion on the dense
family because those terms are 6.4% and zero there. Check which term a
proposed optimization moves against this split before costing it.

What was built along the way is kept, because it is all independently
useful: `dequant_int4_gemm_simd` and its exact parity test, the four
measurement surfaces, and the rollback primitives
(`KvCacheManager::rewind_by`, `GdnStateManager::snapshot`/`restore`,
`RealForwardRunner::checkpoint`/`rollback`) with `rollback_probe.rs`, which
is a standing correctness test whatever happens to this phase.

## What would change the answer

In order of leverage:

1. **The routed pair, and no longer `c(M)` in general.** This entry used to
   read "`c(M)`, not the drafter", and half of it has now been collected and
   spent: the GEMV arm was improved ~2x and bought two points (see "The
   composite, written out"). What is left is the specific term that
   improvement could not reach: `moe_phase1_gate_up_act_u16load` and
   `moe_phase2_down_reduce_k8`, 26% of decode compute, still with no batched
   form at all. Both grants in the composite are guesses about it, and the
   spread between them (0.95x against 1.00x at block 8) is the whole
   remaining uncertainty on this axis. **Note it is not enough on its own**:
   even the optimistic grant leaves block 8 at 1.00x, because the 19% floor
   does not move either. `docs/BATCHED_PREFILL.md` steps 2 and 3 specify
   these two kernels for a different reason and would answer this for free.
2. ~~**`simdgroup_matrix`.**~~ **Measured 2026-08-17 and closed.** It was
   built and benched against the exact kernel in one session
   (`dequant_int4_gemm_mma`, `c_of_m_matrix_against_exact_at_qwen38_shapes`)
   and it loses at every width (6.6x slower at M=2, 1.33x at M=16) and
   plateaus at ~0.5 past M=16, worse than what the exact kernel already
   reaches. A packed INT4 run cannot be `simdgroup_load`ed, so every weight
   element must be dequantized into threadgroup memory first, and that work
   is independent of B while the MACs matrix hardware accelerates scale with
   B. The kernel is dequant-bound and the matrix unit accelerates the wrong
   term. The good news is that the bit-exactness trade this entry priced
   does not have to be made at all. Details in `docs/MTP_SPECULATIVE.md`.
3. **A dense family.** The 19% floor is dominated by MoE-specific work
   (routing, the expert kernels' share, GDN). Meta measured 1.5x for DFlash
   on an M4 Max with the dense Muse-Glimmer-30B, and a dense target here
   would have neither the expert-union term nor the recurrent state.
4. **This target's actual accept length.** The one number in the reading
   above that is borrowed rather than measured here. `accept_length_probe.rs`
   is already shaped to take a real drafter in place of the n-gram one.

## Sources

- DFlash: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- Per-position acceptance curve: [DeLS-Spec, arXiv 2607.07409](https://arxiv.org/pdf/2607.07409)
- Drafter checkpoints: [z-lab on Hugging Face](https://huggingface.co/z-lab)
- Rust reference (Apache-2.0, `mlx-rs`): [lablup/mlxcel](https://github.com/lablup/mlxcel/tree/45dea248926c5d0a8f09bdfb2ce1d21aed8d504a/src/lib/mlxcel-core/src/drafter/dflash)
- Apple Silicon reference (MIT, Python/MLX): [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx)
