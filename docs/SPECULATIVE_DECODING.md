# Speculative Decoding and DFlash (Measured Marginal)

The question this page answers: should this engine adopt speculative
decoding, specifically DFlash -- a small block-diffusion drafter proposes a
whole block of tokens in one parallel forward pass, the target verifies the
block in one batched pass, and the longest correct prefix is accepted?

Answer, measured 2026-08-10 on the real Qwen 3.6 35B-A3B install:
**about 1.1x at best, at a small block size, and only once a batched MoE
kernel exists that does not yet.** Not the 3.6x DFlash reaches at
concurrency 1 on datacenter GPUs, and not the 1.5x Meta measured for it on
an M4 Max with a DENSE model. Recorded here so the arithmetic is not
re-derived; it also appears in ROADMAP under the speculative-decoding item.

Nothing here refutes DFlash. It is a statement about this engine's verify
cost, and every number that would have to change for the answer to change
is named at the bottom.

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
It carries NO embedding table and NO LM head -- it reuses the target's, both
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

- B dispatches of the existing GEMV over ONE matrix cost 0.65B, not ~1 --
  the weights do not stay cached across them;
- encoding those into a CONCURRENT compute encoder rather than the engine's
  serial one recovers only 1.07-1.40x, and 8 x the resulting rate lands on
  the ~355 GiB/s the kernel saturates at, which is the proof the hardware
  genuinely moved the weight bytes eight times.

So `dequant_int4_gemm_simd` exists: one dispatch, each packed nibble read
once and multiplied into B accumulators. Parity against B separate GEMV
calls is EXACT, not tolerant, and mutation-checked three ways.

| shape | M=2 | M=4 | M=8 | M=16 |
| --- | ---: | ---: | ---: | ---: |
| expert 512x2048 | 0.77 | 0.55 | 0.44 | 0.36 |
| o_proj 2048x2048 | 1.01 | 0.80 | 0.71 | 0.67 |
| q_proj 4096x2048 | 1.01 | 0.82 | 0.78 | 0.75 |
| stacked 8192x2048 | 1.03 | 0.87 | 0.79 | 0.78 |

Nothing below M=4 is worth batching at all.

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
datacenter GPU gets. Granting a batched MoE at ~0.5, `c(8)` lands near 0.67
and a verify pass is about 1.8x cheaper than eight sequential passes.

### Accept length

Two sources, and they agree on the shape.

**An n-gram drafter, measured here** (`accept_length_probe.rs`, real Qwen,
greedy, 300 tokens per arm, sequential verify because only the ratio
matters): it fires on 10-11% of rounds and reaches 2.46 / 2.76 / 2.76
accepted-plus-bonus at blocks 4 / 8 / 16. **Refuted**, and the shape says
why: its accept length SATURATES -- block 16 accepts exactly what block 8
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

| block | DFlash accept length | break-even here | verdict |
| ---: | ---: | ---: | --- |
| 4 | 2.96 | 2.6 | **pays, 1.14x** |
| 8 | 4.26 | 4.4 | loses, 0.97x |
| 16 | 6.49 (mean) | 7.5 | loses, 0.87x |
| 16 | 7.87 (best task) | 7.5 | pays, 1.05x |

**On this engine the optimum is a SMALL block and the win is about 1.1x**,
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
NON-SPECULATIVE run rather than the other block sizes: comparing speculative
arms against each other passes even when all of them are wrong the same way.

## Standing decision

**Do not build it for 1.14x.** Three caveats all point the same way and the
margin is inside the composite's error bar:

- the break-even column assumes a batched MoE that does not exist; without
  one, every row loses;
- the per-position curve is published for Qwen3-4B rather than this target,
  and the 35B drafter is a later retrain;
- `c(M)` for the unbuilt MoE kernel is estimated, not measured.

What was built along the way is kept, because it is all independently
useful: `dequant_int4_gemm_simd` and its exact parity test, the four
measurement surfaces, and the rollback primitives
(`KvCacheManager::rewind_by`, `GdnStateManager::snapshot`/`restore`,
`RealForwardRunner::checkpoint`/`rollback`) with `rollback_probe.rs`, which
is a standing correctness test whatever happens to this phase.

## What would change the answer

In order of leverage:

1. **`c(M)`, not the drafter.** At `c(8) = 0.67` block 8 reads 0.97x, at
   0.60 it reads 1.05x, at 0.55 it reads 1.11x. An 18% kernel improvement is
   worth more than any drafter change and lifts every row at once. The first
   target is the MoE phase-1/phase-2 pair: 26% of decode compute with no
   batched form at all.
2. **`simdgroup_matrix`.** The one remaining lever on the GEMV, and its cost
   is not effort: matrix hardware reorders accumulation, so the verify pass
   would stop agreeing bit-for-bit with a sequential decode and speculative
   output would no longer be provably identical to non-speculative output.
   That is a design decision, worth taking only if an end-to-end run lands
   short with the exact path.
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
