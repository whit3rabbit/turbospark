# Native MTP speculative decoding (worth building, after a kernel fix that halved c(M))

The question this page answers: `youssofal/MTPLX` reports 2.24x from
speculative decoding driven by a checkpoint's own multi-token-prediction
head rather than a separate draft model, on Apple Silicon, on a model this
port runs. Should this engine do the same?

Answer, measured 2026-08-17 on this machine: **yes, at a small block size,
worth roughly 1.35x to 1.58x with a drafter as good as MTPLX reports.** The
ceiling over all possible drafters is ~2.07x.

> **MEASURED END TO END 2026-08-18 AND THE SHAPE HELD: block 2 pays 1.66x.**
> This target's own accept length is no longer borrowed. Real numbers, real
> install, greedy, lossless against a non-speculative reference:
>
> | block | accepted/round | committed/round | break-even | speedup |
> | ---: | ---: | ---: | ---: | ---: |
> | 2 | 1.84 | 2.84 | 1.71 | **1.66x** |
> | 4 | 3.20 | 4.20 | 2.86 | **1.47x** |
> | 8 | 4.29 | 5.29 | 4.53 | **1.17x** |
> | 15 | 4.29 | 5.29 | 7.96 | 0.66x |
>
> Per-position acceptance is 0.92 / 1.00 / 0.93 / 0.64 / 0.67 / 0.83 / 0.67 /
> 0.80, so single-step acceptance is 0.94, slightly ABOVE MTPLX's reported
> ~90%, and the projected 1.35-1.58x band is cleared at every block below 15.
> The SMALL-block conclusion survives: block 2 is still the optimum, because
> verify cost scales close to linearly in M while acceptance decays. The
> ~2.07x ceiling stands as a ceiling, and block 2 is now at 80% of it.
>
> **An earlier version of this box read 1.37x and reported that the chain
> "saturates at ~2.05 accepted", blaming the projection for assuming 6.13.**
> Both were artifacts of a norm deviation (step 3 below), not properties of
> the head, and both are withdrawn. The projection was closer to right than
> the measurement that appeared to correct it.

> **THIS PAGE REACHED THE OPPOSITE CONCLUSION FIRST, AND THE REVERSAL IS THE
> MOST USEFUL THING ON IT.** The first pass measured `c(M)` at 0.82-0.89 at
> M=8, computed a ceiling of 1.16x, and closed the question. That was a true
> measurement of the code as it stood and a false statement about the
> engine: `dequant_int4_gemm_simd` was missing the function-constant
> specialization its GEMV sibling got in `46617c6`, which I had noticed and
> written down in the same session as a cheap unrelated win. Applying it
> halved `c(M)`. **A composite built on an unoptimized kernel measures the
> kernel, not the question** -- and the tell was available before the
> measurement, in the form of a known deficiency in one of the two arms.
>
> **IT THEN HAPPENED A THIRD TIME, one level further out, and the tell was
> again a known deficiency in one arm.** The first end-to-end accept length
> was taken while the head's per-head `q_norm`/`k_norm` were still read
> plainly -- a deviation this page had itself recorded, and dismissed as
> "small" on the strength of 23/24 top-1. It was worth 21 to 75 percent of the
> speedup, and the numbers it produced were quoted here as two findings about
> the head that were really findings about the deviation. The pattern across
> all three: **a measurement taken with a known defect in one arm measures the
> defect, and the phrase that licenses it every time is an estimate of how
> much the defect costs, made without measuring it.**

## What MTP is worth taking

`mtplx/` is ~15 subdirectories of MLX tensor code, a Mac app, a dashboard, a
vLLM-Metal path and an OpenAI/Anthropic server this repo already has. None of
that ports. Three ideas do, and none of them is code:

1. **MTP-head-as-drafter.** An architecture read off the checkpoint header.
2. **Exact rejection sampling with residual correction.** Published math
   (Leviathan et al., arXiv 2211.17192; Chen et al., arXiv 2302.01318).
3. **Draft-depth auto-tuning.**

**Take no MTPLX source.** Its license is verified stock Apache-2.0, so
inclusion is legal but adds NOTICE obligations to a repo that is uniformly
MIT. The head architecture comes from the safetensors header and the
acceptance algorithm from the papers.

**This is a copying decision, not a reading one, and conflating them cost
about two hours.** `mtplx/mtp_patch.py` documents the delta-encoded MTP norms
that `docs/MTP.md` rediscovered by bisection -- `_heal_raw_delta_mtp_norms`,
whose docstring says such a sidecar "poisons every draft" -- and states three
more of this port's hard-won conventions as plain `MTPContract` fields
(`hidden_variant: "post_norm"`, `concat_order: "embedding_hidden"`,
`mtp_quant_group_size: 64`). A convention is a fact about the checkpoint and
is nobody's copyrightable expression. Read the prior art; copy none of it.

## The head, off real bytes

`crates/repack/tests/mtp_head_network.rs`, ~114 KB of ranged reads against
`Qwen/Qwen3.8-27B`, deterministic. Not `mlx-community/Qwen3.8-27B-4bit`,
whose conversion drops the head.

| | |
| --- | --- |
| tensors | 15, all in `model-00018-of-00018.safetensors` |
| size | 849,398,784 B, BF16 throughout |
| blocks | 1 (`mtp_num_hidden_layers`), **FULL attention**, not gated-DeltaNet |
| shape | identical to a trunk full-attention layer, field for field |
| embedding / head | neither; both shared with the trunk |
| structure beyond the block | `fc` [5120, 10240], two input norms, one output norm |

**No new Metal kernel.** The block's q/k/v/o, per-head q/k norms, RoPE, dense
SwiGLU MLP and final norm are the trunk's own shapes, so a draft step is
`families/qwen/attn.rs` and `dense.rs` called with `mtp.layers.0.*` names.
`fc` is a plain GEMV over a `[2H]` buffer; the concatenation is a buffer
write. Rollback is a KV cursor move, because the block is full attention and
carries no recurrent state.

**No third-party artifact.** MTPLX publishes this head as an 849 MB
`mtp.safetensors`; the byte total above identifies that file as a verbatim
copy of Qwen's tensors. So it can be read from the official checkpoint's last
shard, which removes both an Apache-2.0 weights dependency and the
"calibrated against a different trunk" caveat.

At INT4 group 64 the head is ~226 MB against the trunk's 15.13 GB resident
region, i.e. **1.5% of the weight bytes**, so a draft step costs ~0.015
decode-steps. That is the entire appeal of MTP over a separate drafter.

## The dense compute split

`TURBOSPARK_PHASES=1 TURBOSPARK_DISPATCH_PROFILE=1` on `~/models/ternary27b.gturbo`,
the same architecture and the same decode flow as `qwen38`, and on disk.
Warmup discarded. 1,138.3 dispatches per token, and every count reconciles
against the layer graph (496 GEMVs = 16 full x 4 + 48 linear x 5 + 64 x 3
MLP + 1 head; 128 norms = 64 x 2; 16 `attention_decode_partial` = 16 full
layers), which is what says nothing is truncated. Absolute times are inflated
by the profiling mode; **only the shares transfer.**

| work | share of GPU busy | batches? |
| --- | ---: | --- |
| `dequant_int2_gemv_simd` (496x + head) | **93.1%** | yes, per `c(M)` below |
| norms and elementwise (7 kinds) | 4.1% | **no** -- no weights to amortize |
| GDN recurrent step and conv (4 kinds, 48x) | 2.2% | **no** -- sequential by definition |
| attention (partial + combine, 16x) | 0.6% | yes, M queries share one KV read |

**The un-amortizable floor is 6.4%**, against the MoE family's 19%. Three
things follow, and the first is why this family was the right one to ask
about: there is no expert-union term, no MoE pair with no batched form, and
almost nothing that cannot amortize.

Two structural notes for whoever designs the verify pass. This family commits
**one command buffer per token** (`gpu wait (layer cb1)` reads 0.0% and
`final wait` 95.4%), so the fill/drain saving that gave batched prefill its
1.22x on Gemma 4 does not exist here -- a batched verify must earn everything
from the kernels. And attention is 0.6% at decode context, so widening it
buys nothing on this path.

> **`MAX_SAMPLES` in `crates/gpu/src/dispatch_profile.rs` was 256 and had to
> go to 4096 to take this table.** It was sized when Gemma 4's cb1 encoded
> ~25 dispatches. At 2048 the profiler captured 447 of 497 GEMVs and printed
> a ranking that looked entirely plausible; the `(over sample capacity)` row
> is the only thing that said otherwise. A truncated profile does not look
> truncated.

## c(M), and the kernel fix that halved it

`crates/gpu/tests/gemv_bandwidth_bench.rs::c_of_m_at_qwen38_shapes`, real
`dequant_int4_gemm_simd`, two rounds, `--test-threads=1` (the two `c_of_m`
tests contend for the GPU if allowed to run concurrently, and their first
interleaved run produced an unusable table).

| shape | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| gate/up 17408x5120 | 0.49-0.51 | 0.53-0.54 | 0.46-0.47 | 0.44 |
| down 5120x17408 | 0.51-0.53 | 0.59 | 0.47-0.48 | 0.45-0.46 |
| gdn_inproj 16480x5120 | 0.49-0.50 | 0.53-0.54 | 0.45-0.46 | 0.43 |
| packed_q 12288x5120 | 0.51-0.55 | 0.54-0.56 | 0.44-0.48 | 0.43-0.47 |
| o_proj 5120x6144 | 0.47-0.53 | 0.54-0.57 | 0.43-0.47 | 0.46-0.49 |

**Against 0.98-1.15 / 0.85-0.95 / 0.82-0.89 / 0.77-0.89 before**, on the same
machine in the same session. What changed is `dequant_int4_batch.rs`, in two
steps that have to be read together.

**Baking M, N and B as function constants.** The GEMV got this in `46617c6`
and the batched kernel did not, which quietly moved every `c(M)` reading the
wrong way -- `c(M)` is the ratio of the two arms, so an optimization landing
on the sequential arm alone makes batching look worse. B is the one that
matters: with it a runtime argument the inner loops cannot unroll and
`acc[]` occupies all sixteen registers whatever the caller asked for, so a
B=4 dispatch paid a B=16 register footprint on a kernel whose own header
records the register file as the binding constraint.

**Bounding the unroll at 4, which was swept rather than chosen.** Baking B
lets the compiler unroll fully, and full unrolling is the difference between
the best and worst numbers this kernel has produced. On gate/up:

| unroll | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| none (before any of this) | 1.04 | 0.90 | 0.85 | 0.83 |
| baked, unroll disabled | 0.95 | 0.81 | 0.77 | 0.73 |
| baked, `unroll_count(2)` | 0.46 | 0.64 | 0.59 | 0.57 |
| baked, **`unroll_count(4)`** | **0.50** | **0.55** | **0.46** | **0.44** |
| baked, `unroll_count(8)` | 0.51 | 0.55 | 0.64 | 0.89 |
| baked, full unroll | 0.51 | 0.57 | 0.65 | **1.14** |

At B=16 a full unroll holds sixteen copies of `e0..e7` live at once, ~128
floats beside `acc[16]`, and spills -- which is the register-blocking failure
already recorded in that kernel's header arriving through a different door.
A fixed count of 4 keeps ~32 activation floats live regardless of B, and is
what makes the row monotonic in M for the first time.

**AND A THIRD STEP LANDED 2026-08-29, WHICH MATTERS MOST AT EXACTLY THE
BLOCK SIZES THIS PAGE CARES ABOUT.** `FC_GEMM_R` gives one SIMD group
`row_block` contiguous output rows, dividing the per-block activation loads
by R and hoisting the activation sum out of the row loop. The width is now
chosen per batch (`gpu::best_row_block`), and at a verify's block sizes the
gain is larger than at prefill's M=16:

| M | before (R=1) | chosen R | after | gain |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.004 | 1 | 1.004 | -- |
| 2 | 0.539 | 2 | 0.504 | 1.07x |
| 3 | 0.508 | 2 | 0.356 | **1.43x** |
| 4 | 0.623 | 2 | 0.313 | **1.99x** |
| 8 | 0.504 | 4 | 0.414 | 1.22x |
| 16 | 0.493 | 4 | 0.370 | 1.33x |

**Read the M=1 and M=2 rows before touching the table.** A wider block is a
straight LOSS at M=1 (1.00 to 1.22) and a 29% one at M=2, so the obvious
global `row_block = 4` -- the reading off prefill's M=16 row -- would have
regressed the narrow end this page lives at. That is why the choice is a
per-width table rather than a constant.

The M=4 cell is the largest single gain anywhere in this kernel's history,
and it is a repair rather than a new win: the R=1 column has a BUMP at M=4
(0.623 against ~0.51 either side), which is `unroll_count(4)` meeting
`b_dim == 4` -- exactly one full unroll -- and going badly. R=2 removes it.
Note also that the frozen tables above were taken in another session and read
up to 11% optimistic against the machine that measured this one; compare arms
measured beside each other, never across sessions (AGENTS.md Gotcha 22). The
full 1..16 sweep is in `docs/BATCHED_PREFILL.md`, "Step 6's kernel term".

**Parity is still exact**, not tolerant: `dequant_int4_gemm_parity.rs` compares
the batched kernel bit-for-bit against B separate GEMV calls at EVERY row
block as well as every batch width, which is the
property that makes speculative output provably identical to non-speculative
output. Specialization does not reorder any sum.

The cache key carries all three baked values, and
`a_second_shape_in_one_process_does_not_reuse_the_first_shapes_pipeline`
guards it -- mutation-checked on each of M, N and B, all three reddening. That
is crate Gotcha 1's trap, and the bench that measured the GEMV's own
specialization fell into it on its first run.

### simdgroup_matrix: measured, and a dead end

`docs/SPECULATIVE_DECODING.md` names matrix hardware as "the one remaining
lever" on `c(M)` and prices it in bit-exactness. It was built
(`dequant_int4_gemm_mma`, `crates/gpu/src/shaders/dequant_int4_mma.metal`)
and measured beside the exact kernel in one session
(`c_of_m_matrix_against_exact_at_qwen38_shapes`):

| M | exact | matrix | matrix / exact |
| ---: | ---: | ---: | ---: |
| 2 | 0.52 | 3.39 | 6.56x slower |
| 4 | 0.58 | 1.69 | 2.92x |
| 8 | 0.48 | 0.85 | 1.76x |
| 16 | 0.47 | 0.62 | 1.33x |
| 32 | -- | 0.52 | past the SIMD kernel's cap |
| 64 | -- | 0.58 | past the SIMD kernel's cap |

**It loses at every width, and the plateau past M=16 is the finding**: it
stops improving at ~0.5, which is worse than what the exact kernel already
reaches at M=16, so this is not an implementation more batching would
rescue. Two rescues were tried and neither moved it -- widening the staged K
block from 8 to 64, which cuts barriers eightfold, and running past the SIMD
kernel's register cap to M=32 and 64.

The reason is structural. A packed INT4 run cannot be `simdgroup_load`ed, so
every weight element must first be dequantized into threadgroup memory. That
work is proportional to `rows x K` and is **independent of B**, while the MAC
work matrix hardware accelerates is proportional to `rows x K x B`. At the
widths this engine has, the kernel is dequant-bound, and the matrix unit is
accelerating the term that is not the cost. Fixing it would take weights
already in a loadable format, not a better tiling.

So the exact kernel wins on both axes, faster and bit-identical to a
sequential decode, and the trade `docs/SPECULATIVE_DECODING.md` worried
about does not have to be made. **Treat "use simdgroup_matrix" as closed.**

**THAT VERDICT IS CORRECT FOR SPECULATION AND WAS RE-SCOPED FOR PREFILL ON
2026-08-29.** For a verify pass it is closed twice over: the matrix kernel
is slower AND it forfeits bit-identity, so there is nothing to trade. For
PREFILL, where bit-identity is already not the bar, the reasoning above was
found to be scoped to one tile -- `kMmaTile = 8` with ONE SIMD group -- and
its stated mechanism ("dequant work independent of B") is refuted by
arithmetic and by deletion. MLX's `qmm_t_impl` is the same algorithm at
128 threads with `BM` 32-128 and reaches `c` 0.145 where this reads 0.52.
Three of the four candidate levers are now measured dead ends
(`ROADMAP.md` "Do Not Revisit" 13-14); the fourth, the matrix path itself,
is untried. None of that reopens the speculation question.
The kernel and its bench are kept precisely so it is not re-proposed; the
tolerance-based parity test beside it (`dequant_int4_mma_parity.rs`) also
documents what the reassociation would have cost, 0.09% of the output range.

### The MoE shapes moved too

Re-run in the same session with the same binary:

| shape | recorded (`docs/SPECULATIVE_DECODING.md`) | before this fix | after |
| --- | ---: | ---: | ---: |
| expert 512x2048, M=8 | 0.44 | 0.60-0.62 | **0.32-0.39** |
| expert 512x2048, M=16 | 0.36 | 0.54-0.55 | **0.32-0.35** |
| o_proj 2048x2048, M=8 | 0.71 | 0.75-0.76 | **0.41-0.48** |

So that page's table was stale in both directions at different times, and is
now beaten.

**Its verdict was re-derived on 2026-08-18 and did not move**: block 4 pays
1.14-1.19x, block 8 is 0.95-1.00x, block 16 loses, against a recorded
1.14 / 0.97 / 0.87. The prediction made here held exactly: the MoE
composite is dominated by a 19% un-amortizable floor and by the routed pair,
26% of decode compute with still no batched form, and a better GEMV fixes
neither. **The pair of pages is now the useful artifact rather than either
one**: the same kernel fix, measured the same week, is worth 1.35-1.58x on
the dense family here and about two points on the MoE one there, because
that family's corresponding terms are 6.4% and zero. Read both before
costing an optimization by the size of the term it improves.

## Composing it

Weighting `c(M)` by the split above -- GEMV 93.1%, attention 0.6% batching to
~1/M, and the 6.4% floor at 1.0:

| M | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: |
| composite `c(M)` | 0.54 | 0.58 | 0.49 | 0.48 |

A round with block M verifies M+1 positions in one pass (the confirmed token
plus M drafts), yields the accepted prefix plus a free bonus token, and costs
`(M+1) x c(M+1) + M x 0.015` decode-steps.

| block | verify cost | max tokens | ceiling | DFlash accept | at DFlash | MTP at 90% | at MTP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.71 | 3 | 1.75x | -- | -- | 2.71 | **1.58x** |
| 4 | 2.86 | 5 | 1.75x | 2.96 | 1.03x | 4.10 | **1.43x** |
| 8 | 4.53 | 9 | 1.99x | 4.26 | 0.94x | 6.13 | **1.35x** |
| 15 | 7.96 | 16 | 2.01x | ~6.3 | 0.79x | 8.15 | 1.02x |

The asymptotic ceiling is `1 / c(inf)` = **2.07x**, and unlike the first
pass's 1.19x that is a number worth chasing.

Two readings. **The optimum is a small block**, which still inverts the
datacenter result and for the same reason: verify cost here scales close to
linearly in M while acceptance probability decays, so extra proposals cost
nearly a full step each. And **the drafter's quality is now the binding
term**, where before the kernel was: at DFlash's published accept lengths it
is a wash, and at MTPLX's reported ~90% single-step acceptance it pays
1.35-1.58x. That is what makes measuring this target's real accept length
worth doing, which the first pass had ruled out.

Block 16 is not reachable: `MAX_BATCH_ROWS = 16` is a register-file limit, so
a 17-position verify does not fit. Block 15 is the largest legal one.

## What to build, in order

1. **Ingest the head.** `crates/repack`: an optional extra `RangeSource`
   carrying `mtp.*` from the official checkpoint's last shard, quantized to
   INT4 by the existing `repack.rs` quantizer (drafter quality is a
   throughput axis, never a correctness one) with norms through
   `narrow_raw_to_bf16`. `manifest.json` gains an optional `mtpHead` block
   where absent means "no head" (Gotcha 39's rule). **Fixture before
   download**, per `crates/repack/CLAUDE.md` Gotcha 8.
   **Done.** With one correction worth carrying: the ingest landed in the
   non-streamed writer alone, and every real install goes through the
   streamed one. The first stream that asked for a head wrote a
   byte-identical headless install (851
   resident tensors, a 15,132,916,736-byte region, no error and nothing in
   the progress log), because `write_gemma4_install_streamed` classified `mtp.*` correctly into
   `plan.mtp_bases` and then never read it. Every fixture took the other
   path. **A fixture has to exercise the writer the download will use**, not
   just the walk they share; `both_writers_carry_the_mtp_head` is that test
   and is the only one of fourteen that reddens without the fix. Also no
   `mtpHead` manifest block: the resident index already answers the question
   and cannot drift from the bytes.
2. **The draft step.** `crates/runtime/src/families/qwen/mtp.rs` plus an
   `MtpState` owning its own one-layer KV rather than widening
   `KvCacheManager`, whose sizing is driven by `ArchConfig` and whose every
   family's oracle peak is frozen. Off by default behind
   `TURBOSPARK_MTP_DRAFT=<depth>`.

   **Done.** `~/models/qwen38-27b-mtp.gturbo` is the install (14 GB;
   resident region 15,371,847,680 bytes, 228 MiB more than the headless
   one). With drafting off the flow is provably inert: `qwen38_quality_gate`
   reproduces perplexity 4.9432 and both frozen digests exactly, and
   `qwen38_memory_oracle` reads 659.5 MiB against the headless install's
   recorded 659.4, so the head's 228 MiB of weights are not counted, which
   is AGENTS.md Gotcha 40 holding a fourth time. Both smokes stay coherent.
   With drafting on the head opens and drafts; a headless install is refused
   at open by name.
3. **Accept length, sequential verify.** `accept_length_probe.rs`'s shape
   with the MTP head in place of the n-gram drafter. Keep both of its
   disciplines: verify one `produce` at a time (only the ratio matters), and
   gate against a non-speculative reference run.

   **Done 2026-08-18, and it took finding a real defect first.** The probe is
   `crates/bench/tests/mtp_accept_length_probe.rs`; the numbers are in the box
   at the top of this page. It asserts a functional drafter (first-proposal
   acceptance above 2%) rather than printing whatever it measures, because
   its first run read **0 accepted of 7,168 proposals** and a table saying
   "loses" would have closed this question with the wrong answer, the same
   failure the reversal box above records, one level up.

   **The bug was a norm convention.** The head's five whole-vector norms are
   centered (the checkpoint stores an offset from unity and the effective
   scale is `1 + w`) while the trunk's are plain. This port read them
   plainly, so every norm in the head scaled by ~0 instead of ~1. That is
   AGENTS.md Gotcha 50's "one model, two conventions" arriving on a second
   family, and the fix was to dispatch the `rmsnorm_bf16w_centered` kernel
   that already existed for `muse_glimmer`. Measured on the real install:

   | | before | after |
   | --- | ---: | ---: |
   | rank of true `t[i+2]` (of 248,320) | median 248,308 | median **0** |
   | top-1 agreement | 0/24 | **23/24** |
   | pearson vs the trunk | -0.28 | **+0.60** |

   **What found it was the reference, and the route to it is the reusable
   part.** The drafter ships standalone as
   `mlx-community/Qwen3.8-27B-MTP-4bit` (31 tensors, 239 MB, INT4 affine
   group 64 -- the same scheme this port writes), and the implementation is
   `mlx-vlm`, not `mlx-lm`, at
   `mlx_vlm/speculative/drafters/qwen3_5_mtp/qwen3_5_mtp.py`. Reading it
   confirmed five choices at zero cost (concat order `[embedding, hidden]`,
   the `(h_i, t_{i+1})` pairing, positions from 0, `full_attention_interval=1`,
   the target's `lm_head`) and independently confirmed that the head must be
   primed over the prompt. Comparing against it (`scripts/mtp_bisect.py`)
   localized the fault to the first stage, and the norm magnitudes fell out
   of that.

   Two earlier fixes were real and neither was the cause, which is worth
   knowing before reading the diff: the head's KV had never been primed
   (`encode_full_attention_block` takes its span from the `position` argument,
   so a draft at position P attended over P rows nobody wrote) and was never
   rewound after a rejected draft. Both are errors now rather than silence.
   A third, the hidden input being the trunk's post-final-norm state rather
   than its residual, is also correct-per-the-reference and also did not move
   the symptom.

   **CLOSED 2026-08-18, and the floor was much lower than "small" implied.**
   The head's per-head `q_norm`/`k_norm` are centered too and were still read
   plainly, because `encode_rms_norm_bf16w_perhead` had no centered sibling
   and both norms are resolved by NAME inside the shared attention block. The
   fix is that sibling (`rmsnorm_bf16w_perhead_centered`) plus a
   `QkNormConvention` parameter on `encode_full_attention_block`, so the two
   call sites -- trunk and head -- can disagree about tensors of the same name.

   | block | accepted/rd, 5 centered | 7 centered | speedup then | now |
   | ---: | ---: | ---: | ---: | ---: |
   | 2 | 1.35 | **1.84** | 1.37x | **1.66x** |
   | 4 | 1.94 | **3.20** | 1.03x | **1.47x** |
   | 8 | 2.05 | **4.29** | 0.67x | **1.17x** |

   Single-step acceptance 0.82 -> 0.94, top-1 23/24 -> 24/24, pearson +0.60 ->
   +0.6453. `qwen38_quality_gate` reproduces perplexity 4.9432 and both frozen
   digests exactly, which is what says the TRUNK did not move with it.

   **THIS PAGE RECORDED TWO THINGS AS FINDINGS THAT WERE ARTIFACTS OF THAT
   DEVIATION**, and both are withdrawn: that "the chain saturates at ~2.05",
   and that block 8 loses. The chain reaches 4.29 and block 8 pays 1.17x.
   Reading "23/24 top-1, so the cost is measurably small" as licence to defer
   is the mistake to carry forward -- top-1 over 24 positions was ALREADY
   saturated and could not have moved much whatever happened, while the
   quantity that decides the question is acceptance at positions 3 through 7.
   A scalar taken at the top of a curve cannot bound the curve.

4. **Only then the batched verify**, if step 3 clears the table above.
   **UNBLOCKED**: step 3 clears it at blocks 2, 4 and 8, with block 2 the
   optimum at 1.66x. Note the `break_even` column those ratios are read
   against is still a PROJECTION of what a batched verify would cost; step 4
   is what turns it into a measurement, and its target is M+1 = 3 rows.

   **RE-DERIVED AGAINST THE MEASURED CURVE, and the useful reading is against
   the CEILING column rather than the break-even one.** Break-even itself did
   not move -- `3 x c(3) + 0.03 = 1.71` needs `c(3) = 0.56`, consistent with
   the tabulated 0.54 and 0.58 -- so the whole change is on the accept side.
   Measured committed-per-round against each block's own maximum:

   | block | committed | ceiling for that block | at |
   | ---: | ---: | ---: | ---: |
   | 2 | 2.84 of 3 | 1.75x | **95%** |
   | 4 | 4.20 of 5 | 1.75x | 84% |
   | 8 | 5.29 of 9 | 1.99x | 59% |

   **Block 2 is at 95% of everything it can ever be worth**, so no drafter
   improvement can add more than 5% there and the remaining lever at that
   block is `c(M)` alone. The 2.07x asymptote lives at LONG blocks, and long
   blocks are gated on the chain surviving past position 8, which it does not.
   Whoever picks this up should build the verify at block 2 for the 1.66x that
   is already earned, and treat "why does the chain die at 8" as the separate
   question that owns the gap between 1.66x and 2.07x.

   **BUILT AND MEASURED 2026-08-18. BLOCK 2 PAYS 1.44x ON THE CLOCK**, against
   the 1.66x projected here. Numbers, rollback rates and the per-arm table are
   in `docs/MTP.md` ("The batched verify, measured end to end"); the accept
   counts come out IDENTICAL between the sequential and batched arms at every
   block, which is what says the two are the same computation.

   **THE PROJECTION WAS OPTIMISTIC BY 13% AT BLOCK 2 AND BY MUCH MORE ABOVE
   IT, AND THE MISSING TERM IS THE ROLLBACK.** Every composite on this page
   costs a round as one verify pass. On a family with a recurrent half that is
   wrong whenever a proposal is rejected: the gated-DeltaNet state cannot be
   rewound incrementally, so the round restores a whole-state snapshot and
   replays the accepted prefix as a SECOND batched pass. The probability of
   paying that rises with the block -- 10% at block 2, 84% at block 8, 98% at
   block 15 -- so batching is worse than a sequential verify at blocks 8 and
   15 while being much better at 2. A sequential verify never rolls back at
   all, because it stops at the first rejection having absorbed exactly the
   committed tokens.

   Two consequences for anyone re-costing this. **Add a rollback term before
   trusting any block-size table on a recurrent architecture**, and note it is
   a function of the per-position acceptance curve rather than of the mean.
   And the small-block answer now has three independent legs instead of two:
   verify cost scales nearly linearly in M, the accept chain decays, and the
   rollback probability rises.

   The remaining lever at block 2 is still `c(M)` alone.

5. **Make the install itself reachable without a hand-run network test, and
   without paying for the trunk twice.** Steps 1-4 above landed a working
   drafter and measured it, but getting a headed install onto disk at all
   still meant running `crates/repack/tests/qwen38_checkpoint_network.rs`'s
   ignored test by hand.

   **Catalog-driven pull: done.** `crates/catalog/src/models.json` carries a
   `qwen38-27b-mtp` row with an `mtp` field naming the official
   `Qwen/Qwen3.8-27B` repo separately from `source.repo`
   (`docs/MTP.md`'s "Installing a headed checkpoint" has the mechanism and
   the real-bytes verification). `turbospark-model pull qwen38-27b-mtp` now
   produces the install with no hand-written test in the loop, still
   streaming the full ~15 GB trunk.

   **Reusing an existing trunk: done.** `--reuse-trunk-from <alias>` reads an
   ALREADY-INSTALLED `qwen38-27b`'s resident entries back off its own
   `model_weights.bin`, byte for byte (`repack::read_resident_entries`,
   `repack::graft_qwen_gdn_dense_mtp_head`), and only the ~239 MB head
   crosses the network. `build_resident_weights_bin_mixed` does not care
   where a spec's bytes came from, which is what licenses this at all: an
   entry read back off disk and one freshly quantized from a shard are
   indistinguishable to the writer. Refused unless the named install's
   recorded repository and revision match the row's exactly, so a
   differently-pinned or unrelated install cannot silently graft the wrong
   bytes onto the wrong trunk. Measured against the real checkpoint:
   `pull qwen38-27b-mtp --reuse-trunk-from qwen38-27b` finished in 4m21s
   against the plain pull's ~30 minutes, and its `model_weights.bin` came out
   byte-for-byte identical (`cmp`, not just equal size) to the fully
   re-streamed one. Scoped to the `qwen35` dense family: the MoE half's
   routed experts live outside `model_weights.bin` entirely, in
   `packed_experts/`, which this path never touches -- grafting onto an MoE
   install would silently produce a directory missing its routed experts.

## What this changes elsewhere

`docs/BATCHED_PREFILL.md` composed its "fully batched" column at
`c(M) ~ 0.6`. **Re-weighted 2026-08-18 and it moved the most of anything
here**: at M=16 the resident projections read 0.447 and the routed-expert
proxy 0.287, which is `c(16) = 0.399` over the 78.8% of prefill GPU work
that batches, and the whole-program projection went from ~1.5x to a
1.55-1.97x band. Its step 1 result (1.22x measured) is unaffected: it
batches command buffers, not math.

**The contrast with the MoE speculative page is the reusable part.** One
kernel fix, one week, three composites: decisive on the dense family here,
worth ~0.4x of extra prefill speedup there, and worth two points on the MoE
verify. What separates them is not the kernel but what each divides by:
a verify pass divides by an accept length and keeps only the accepted
prefix, a prefill chunk keeps all M of its tokens, and a dense decode has
no un-amortizable expert term to begin with.

## Sources

- MTPLX: [github.com/youssofal/MTPLX](https://github.com/youssofal/MTPLX) (Apache-2.0)
- DFlash accept lengths: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- Per-position acceptance curve: [arXiv 2607.07409](https://arxiv.org/pdf/2607.07409)
- The MoE arm of this question: `docs/SPECULATIVE_DECODING.md`
- Why the same terms give a different answer for prefill: `docs/BATCHED_PREFILL.md`
