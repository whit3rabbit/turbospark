# Native MTP speculative decoding (worth building, after a kernel fix that halved c(M))

The question this page answers: `youssofal/MTPLX` reports 2.24x from
speculative decoding driven by a checkpoint's own multi-token-prediction
head rather than a separate draft model, on Apple Silicon, on a model this
port runs. Should this engine do the same?

Answer, measured 2026-08-17 on this machine: **yes, at a small block size,
worth roughly 1.35x to 1.58x with a drafter as good as MTPLX reports.** The
ceiling over all possible drafters is ~2.07x.

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

## What MTP is worth taking

`mtplx/` is ~15 subdirectories of MLX tensor code, a Mac app, a dashboard, a
vLLM-Metal path and an OpenAI/Anthropic server this repo already has. None of
that ports. Three ideas do, and none of them is code:

1. **MTP-head-as-drafter.** An architecture read off the checkpoint header.
2. **Exact rejection sampling with residual correction.** Published math
   (Leviathan et al., arXiv 2211.17192; Chen et al., arXiv 2302.01318).
3. **Draft-depth auto-tuning.**

**Take no MTPLX source.** Its LICENSE is verified stock Apache-2.0, so
inclusion is legal but adds NOTICE obligations to a repo that is uniformly
MIT. The head architecture comes from the safetensors header and the
acceptance algorithm from the papers.

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

`MFERENCE_PHASES=1 MFERENCE_DISPATCH_PROFILE=1` on `~/models/ternary27b.gturbo`
-- the same architecture and the same decode flow as `qwen38`, and on disk.
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

**Parity is still EXACT**, not tolerant: `dequant_int4_gemm_parity.rs` compares
the batched kernel bit-for-bit against B separate GEMV calls, which is the
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

So the exact kernel wins on both axes -- faster AND bit-identical to a
sequential decode -- and the trade `docs/SPECULATIVE_DECODING.md` worried
about does not have to be made. **Treat "use simdgroup_matrix" as closed.**
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

So that page's table was stale in BOTH directions at different times, and is
now beaten. **Its verdict is owed a re-derivation** and this page does not
attempt one: the MoE composite is dominated by a 19% un-amortizable floor and
by the routed pair, which is 26% of decode compute and still has no batched
form at all. A better GEMV does not fix either.

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

Two readings. **The optimum is a SMALL block**, which still inverts the
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
2. **The draft step.** `crates/runtime/src/families/qwen/mtp.rs` plus an
   `MtpState` owning its own one-layer KV rather than widening
   `KvCacheManager`, whose sizing is driven by `ArchConfig` and whose every
   family's oracle peak is frozen. Off by default behind
   `MFERENCE_MTP_DRAFT=<depth>`.
3. **Accept length, sequential verify.** `accept_length_probe.rs`'s shape
   with the MTP head in place of the n-gram drafter. Keep both of its
   disciplines: verify one `produce` at a time (only the ratio matters), and
   gate against a NON-speculative reference run.
4. **Only then the batched verify**, if step 3 clears the table above.

## What this changes elsewhere

`docs/BATCHED_PREFILL.md` composes its "fully batched" column at
`c(M) ~ 0.6`. The GEMV term is now ~0.46 at M=8, so that projection is
conservative and its steps 2-4 are worth more than it says. Its step 1 result
(1.22x measured) is unaffected -- it batches command buffers, not math.

## Sources

- MTPLX: [github.com/youssofal/MTPLX](https://github.com/youssofal/MTPLX) (Apache-2.0)
- DFlash accept lengths: [arXiv 2602.06036](https://arxiv.org/abs/2602.06036)
- Per-position acceptance curve: [arXiv 2607.07409](https://arxiv.org/pdf/2607.07409)
- The MoE arm of this question: `docs/SPECULATIVE_DECODING.md`
- Why the same terms give a different answer for prefill: `docs/BATCHED_PREFILL.md`
