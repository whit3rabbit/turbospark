# MLX and oMLX kernel experiments

TurboSpark's Rust text engine consumes some MLX-format checkpoints and runs
its own Metal kernels. The experiment here tests an adapted upstream kernel
directly on Metal. It does not change the Swift MLX image pipelines or any
runtime default. For compile-only preparation, see [MoE startup](MOE_STARTUP.md).

mlx-serve (`ddalcu/mlx-serve`) ships ports of the same oMLX kernels, nearly
all of them NAX-only, and reaches its M4 Max numbers through host-side verify
lanes and row-exact kernels instead. That comparison is in
[MLX_SERVE_KERNELS.md](MLX_SERVE_KERNELS.md).

## What was already investigated

| Candidate from the oMLX audit | Current evidence and decision |
| --- | --- |
| GDN prefill threadgroup staging | Already priced at an upper bound of 4.66% of the dense Qwen prefill work. Keep closed; see [reference curve and profile](BENCHMARKS.md#reference-curve-measured-not-inferred) and `crates/gpu/tests/gdn_prefill_share_bench.rs`. |
| Re-tile our handwritten INT4 MMA kernel | The wide tile and activation staging lost their gates. Preserve [measured dead ends 13, 14, 16](../ROADMAP.md#do-not-revisit-measured-dead-ends). A full upstream Steel implementation has different tile machinery and now has separate evidence below. |
| Batched prefill attention | Experimental batch partial/combine kernels and dense Llama wiring already exist. Finish the existing [attention fork's](BATCHED_PREFILL.md#the-attention-fork) model and performance gates before adding another implementation for dense Llama. The pasted audit's claim that no batch attention is wired is stale. At Qwen3.8's GQA shape that batch kernel measured 4-12x slower than the per-row GQA loop, so the dense qwen prefill runs its own multi-row GQA partial (`attention_decode_gqa_rows_partial`, 1.8-4.2x the loop at 64 rows; `docs/SPEEDUP_PLAYBOOK.md`). |
| MoE gather GEMM and weighted sum | Our route-list pair already avoids per-expert argument-buffer tiling and fuses down projection with router-rank reduction. An upstream gather port would also need expert packing/sorting, inverse routes, a separate output slab, and numerical qualification. No new gather experiment was run. |
| NAX, activation INT8, ANE, sparse GLM attention | Do not transfer results to this M4 text-engine contract. They require another device, precision/API contract, or model architecture. No port was attempted. |

The source audit is pinned to oMLX
[`68c8c09f`](https://github.com/jundot/omlx/tree/68c8c09f6f6a54fe36181f9aabba9525797d1b58/omlx/custom_kernels).
Its [Qwen quantized GEMM wrapper](https://github.com/jundot/omlx/blob/68c8c09f6f6a54fe36181f9aabba9525797d1b58/omlx/custom_kernels/qwen35_prefill/csrc/qwen35_qmm.metal)
calls Steel `qmm_t_impl`. The tested dependencies come from MLX v0.31.1,
[`ce45c525`](https://github.com/ml-explore/mlx/tree/ce45c52505c8158ea48d2a54e8caae05efd86bfe/mlx/backend/metal/kernels/steel/gemm).
Both projects use Apache-2.0. The harness retains exact upstream files,
copyright notices, licenses, commit IDs, and per-file SHA-256 identities.

## Isolated Steel probe, 2026-10-03

`scripts/omlx_steel_probe.py` downloads the pinned header closure and builds
`scripts/omlx_steel_probe.swift` as a standalone native Metal harness. It
does not install MLX, load a model, or add a product-side Python runtime.

The prototype uses upstream `qmm_t_impl`, `BlockLoader`, and `BlockMMA`,
with three `(BM, BK, BN)` variants: `(32,32,32)`, `(32,64,64)`, and
`(64,32,64)`. Each uses four SIMD groups and 128 threads. The adapted loader
reads our packed group-64 INT4 bytes and BF16 scale/bias planes, computes
dequantized weights in FP32, then stages FP16 matrix operands. Activations
and output are FP16, with FP32 matrix accumulation. This changes rounding
and reduction order relative to the current scalar affine-factored kernel.

The control compiles the current `dequant_int4.metal` plus
`dequant_int4_batch.metal`, including production function constants,
`best_row_block` selection, 256-thread dispatch, and token-major offsets.
At M=32 it runs two M=16 calls. The product's batch cap remains 16.
Compilation, allocation, fixture construction, and host encoding are
outside the GPU timestamp measurement. Buffers are reused after warmup;
each arm calibrates toward 50 ms of device work, with six alternating-order
rounds per cell and a fresh process per run.

Measured on Apple M4 Max, macOS 26.6.2, Metal 3.1, fast math. Two of six
confirmation processes passed the AC, Nominal thermal, and periodic CPU
screen (no other process at 50% CPU or more). The other four are retained
and excluded from the table. Desktop GPU contention is not measured by
that CPU screen. These are exploratory kernel results, not frozen engine
throughput rows.

Speedup is control GPU time divided by Steel GPU time. The table uses the
median of the two eligible process ratios for `(32,32,32)`:

| Matrix, output rows x input width | M=2 | M=8 | M=16 | M=32 |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.6 shared gate/up, 512x2048 | 0.083x | 0.408x | 0.672x | 1.566x |
| Qwen 3.6 QKV-sized projection, 8192x2048 | 0.208x | 0.830x | 1.465x | 2.977x |
| Dense Qwen gate/up, 17408x5120 | 0.233x | 0.820x | 1.431x | 2.871x |

At M=16 the QKV-sized projection measured 0.1709 ms for the control and
0.1166 ms for Steel; the dense gate/up measured 0.8309 ms and 0.5807 ms.
Their process speedup ranges were 1.457-1.473x and 1.417-1.445x. The larger
tile variants did not improve those cells. Small widths and the small
shared-expert projection regress, so this does not support a blanket
replacement of the existing kernel.

Numerical screening passed against an independent FP64 CPU reference at
M=1,2,3,7,16,32, and against the current GPU kernel on all timed shapes.
The fixture varies nibbles, rows, groups, signed FP16 inputs, and ragged
BF16 companions. It uses the existing MMA parity bound, 0.005 times each
output row's maximum reference magnitude. Worst normalized error was
0.000459 against CPU and 0.000818 against the GPU control. These are
synthetic bounds, not model-logit or generated-output parity.

A dtype mutation changed all 12 BF16 device-pointer declarations in a
separate generated source copy to FP16. The CPU screen failed with a
79.28 normalized error, exit 1. The unmutated source hash was unchanged.
This establishes that the fixture catches the scale/bias reinterpretation
error; it does not establish sensitivity to every possible tile defect.

## Reproduce and inspect

Use an Apple Silicon Mac with Xcode, AC power, and no competing workload:

```sh
python3 scripts/omlx_steel_probe.py \
  --output target/omlx-steel-$(date +%Y%m%d-%H%M%S) --runs 3
```

The output directory must be new. It contains the upstream originals and
licenses, adapted and control shader compositions, source identities,
compiled harness SHA-256, per-process JSONL and stderr, periodic host
samples, and `summary.json`. The summary keeps all cells and a separate
`eligible_cells` collection. An empty eligible collection is no speed
evidence. A failed or timed-out process retains its logs and host samples.

Retained local evidence is under `target/omlx-audit/` (ignored artifacts):

- `steel-confirm-3/`, eligible process 1.
- `steel-screened-3/`, eligible process 1.
- `bf16-mutation/`, applied mutation count, failure log, and original hash.
- `steel-run-1/` and `steel-run-2/`, earlier exploratory fixture/calibration
  versions, excluded from the table.

Both confirmation sets used Swift harness SHA-256
`6409906435af66ad58bd18046cc8fbfec380bee87d75fce0dbf150932252a7f1`
and adapted shader SHA-256
`a071159b1c24ad9de14685651f0092b97fa6f82b4003e75218e78e712982a60d`.
These identify the experiment sources, not the complete dirty workspace.

## Runtime promotion boundary

The result supports a targeted large-projection prefill experiment. No
runtime dispatch or default was changed by the probe itself. **Update
2026-10-11:** the kernel is now vendored from MLX's own `quantized.h` (MIT,
`scripts/vendor_mlx_steel_qmm.py`) and is the dense qwen chunked-prefill
default at a 64-row micro-batch, after the gates below ran on the real
`qwen38-27b` trunk (`docs/BATCHED_PREFILL.md`, "The dense qwen batched arm
becomes the default, on Steel").

**Update 2026-10-15:** the qwen MoE half (Qwen 3.6, Ornith 35B) now
prefills its GDN and attention projections and its shared expert as M-row
GEMMs on Steel, while the routed experts keep the per-token loop. On a
2,940-token Qwen 3.6 prompt that prefilled 1.42-1.56x faster than the
per-token arm. It holds to a MoE KL bound rather than the dense one
(`docs/BATCHED_PREFILL.md`, "The qwen MoE batched arm"). Gemma 4 does not
use Steel, and no Gemma 4 run on it exists.

A runtime port must qualify a matching chunked-prefill checkpoint through
the [model gates](../.claude/docs/model-gates.md): independent kernel
parity, real greedy and sampled output, model-quality and memory checks,
and paired total prefill measurements including preparation costs. Keep
M=1 decode and speculative verification on their current contracts. M=32
also needs separate scratch, state, and driver-cap qualification. The
synthetic M=32 ratio cannot be reported as a shipped prefill speedup.

## open-jet megakernel review, 2026-10-11

What came of it, how each change was qualified, and how to apply the same
moves to other kernels, families, and drafters: [speedup playbook](SPEEDUP_PLAYBOOK.md).

[open-jet](https://github.com/L-Forster/open-jet)'s `megakernel/` is a CUDA
engine for the same model (Qwen3.8-27B, llama.cpp arch `qwen35`: 48
Gated-DeltaNet and 16 gated-attention layers plus an MTP block) at Q4_K_M on
an RTX 3090. It runs a whole speculative cycle in one persistent kernel.
Each technique was checked against this engine's M4 Max path. Kernel timings
are synthetic and interleaved. End-to-end rows are interleaved fresh-process
pairs on the real `qwen38-27b-mtp` install, on AC, with other sessions
compiling on the CPU. They are exploratory, not frozen benchmark rows.

| open-jet technique | This engine before | Decision and evidence |
| --- | --- | --- |
| Split-K decode attention that loads each K/V tile once for all 6 Q heads of a group | `attention_decode_partial`: one threadgroup per Q head, two threadgroup barriers per position, K/V re-read per Q head. 8-25 GB/s on unique K/V bytes | **Built**: `attention_decode_gqa_partial`, opt-in per scratch, on for `QwenGdnDense`. Simdgroups take interleaved positions with 4 rows in flight and the group's Q and O in registers, no MMA. 3-13x over 512-32K positions (`crates/gpu/tests/qwen38_attention_decode_probe.rs`). End to end, decode 1.24x at 5.8K and 1.76x at 18K context, prefill 1.38x at 18K (this family prefills through decode attention). Greedy output byte-identical in those runs |
| Gated-DeltaNet rollback by lazy replay, no state snapshots | Host memcpy of the 147 MiB state into fresh `Vec`s every round: 3.7 ms with the GPU idle. Restore 3.8 ms | **Partly built**: pooled GPU blit snapshot, 0.06 ms on the host and ordered before the verify on the same queue. Restore 1.95 ms. Lazy commit itself needs a commit-row output from the verify delta kernel and the replay folded into the next verify. Not built |
| MTP drafting through a 32K-row draft vocabulary | Every logits step read the full 248K-row lm_head (~0.7 GB), including the round's last step, whose proposal is discarded | **Built**: the row-only last step goes through `prime_drafter` and skips the head. With the GPU snapshot, block-2 MTP measured 29.65 -> 32.16 tok/s (+8.5%), output byte-identical to the old binary. 32.58 tok/s (+9.9%) with the GQA kernel as well. The draft vocabulary itself was measured, not built (`docs/MTP_SPECULATIVE.md`) |
| dp4a GEMV on int8 activations, 2-4 rows per warp, `cp.async`, L2 evict hints | FP16 activations, one row per simdgroup | **Not transferable**: Apple GPUs expose no packed int8 dot product. The INT4 GEMV already reads 0.90-1.11x of its own large-shape reference at every decode shape, and gate/up runs 1.16x MLX (`docs/DECODE_BUDGET.md`) |
| One persistent kernel per cycle with a software grid barrier | One command buffer per token, ~978 dispatches | **Not attempted**: Metal has no grid-wide barrier, and mega-fusion is measured dead end 9 (`ROADMAP.md`) |
| FFN gate/up with a `silu(g) * u` epilogue, norms folded into the consumer | Separate gate, up, `silu_mul`, residual add and RMS norm dispatches (~5.6% of a token) | **Candidate**: about 1% from removed dispatches. Not built |
| Prefill GEMMs on tensor cores straight from the k-quants, chunked WY DeltaNet | Prefill ~15-42 tok/s on this family | **Largest remaining gap**: open-jet reports ~1500 tok/s on a 3090. The Steel probe above is the starting point. Chunked WY is measured dead end 15 here |

The GQA kernel changed digests only at near-ties. Qwen3.8's gate went
4.9432 -> 4.9444 with one greedy phrase swapped. Ternary-Bonsai went
6.8350 -> 6.8351 and Bonsai-2 went 5.8134 -> 5.8141, each with one sampled
phrase swapped. All three rows were re-frozen from two agreeing fresh
processes. `TURBOSPARK_ATTN_DECODE_KERNEL=barrier` still reproduces each
old row exactly. The MTP generation gate is lossless at blocks 2 and 4 on
the new code. The Qwen3.8 memory oracle peaks at 661 MiB against 660.3
frozen and a 750 ceiling.
