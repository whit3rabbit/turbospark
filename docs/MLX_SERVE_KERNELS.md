# mlx-serve: where its decode speed comes from, and what applies here

A read-only comparison of `ddalcu/mlx-serve` (a Zig server over MLX with its
own Metal kernels) against this engine, in the shape of
[KAPITSA_KERNELS.md](KAPITSA_KERNELS.md). It answers one question: how does
mlx-serve reach its decode tokens per second on Qwen3.8-27B and the other
dense Qwen installs, and which of those mechanisms are worth building or
measuring here. The adoption candidates are at the end, split into engine
work (for [SPEEDUP_PLAYBOOK.md](SPEEDUP_PLAYBOOK.md)) and evaluator work (for
`crates/lab`, `tslab`).

Nothing was run. Every number in the "their numbers" sections is mlx-serve's
own, from its `benchmarks.md`, `CHANGELOG.md`, `docs/`, and the war stories in
`docs/gotchas/engine-mlx.md`, with the machine and conditions it names. The
TurboSpark rows quoted for scale are the ones already measured on this host in
`KAPITSA_KERNELS.md`. mlx-serve has not been measured on this machine.

## Pinned source and license

- `ddalcu/mlx-serve` at `191da71de32a65f69ebceeb6cb2e91036ab50b30` (tag
  v26.10.3, committed 22 October 2026), cloned shallow on 24 October 2026.
  Its LICENSE is MIT; `LICENSE-APACHE-2.0` covers the incorporated
  third-party parts that its NOTICE lists. (An earlier draft of this line
  called the whole repository Apache-2.0.)
- Read: `src/kernels/*.metal` (20 files, 5,315 lines), `src/transformer.zig`
  (the verify lane table and NAX gates), `src/simd_qmm.zig`, `src/rowqmv.zig`,
  `src/row_attn.zig`, `src/lane_qmm.zig`, `src/dflash.zig` (block caps),
  `src/model.zig` (`rowExactDecode`), `AGENTS.md`, `docs/performance.md`,
  `docs/metal-tracing.md`, `docs/reference.md`, `benchmarks.md`,
  `CHANGELOG.md`, and `docs/gotchas/engine-mlx.md` (5,734 lines of measured
  stories). Three delegated read-only passes covered the quantized-linear
  front door and its lanes (`transformer.zig`, `qmv2.zig`, `moe_affine4.zig`
  and neighbours), the speculative loop and scheduler (`generate.zig`,
  `mtp*.zig`, `round_cost.zig`, `scheduler.zig`, `prefix_cache.zig`), and
  the attention, KV, and GDN host code (`gatedFullAttnWith`, `dec_attn.zig`,
  `kv_quant.zig`, `gdn_decode.zig`). Their findings are folded in below with
  the line-level citations left out.
- Nothing was pasted into this tree. Ideas are reimplemented or declined.
  Copied code would need a NOTICE entry here, as it does there.
- Provenance matters for reading `src/kernels/`: nearly every file is a
  port. oMLX (`jundot/omlx`, Apache-2.0) supplies the NAX attention, the NAX
  gather GEMM, the GLM MoE decode kernels, the KDA kernels, and the pipelined
  GDN prefill. MLX's steel headers (MIT) sit under those. TensorFold
  (`ashhart/TensorFold`, MIT) supplies the row-exact kernels and keyed
  sampling. MTPLX (Apache-2.0) supplies the verify split-K and NAX m16 lanes.
  Inco Splash supplies the packed-KV attention tile shape. mlx-serve's own
  contribution is the host side: lane selection, measured gates, the
  speculative loop, and the discipline recorded in its gotchas.

## The short answer

Ranked by how much of the M4 Max 27B number each explains. Their own
evidence is cited per item below.

1. **Speculation with a wide verify that costs little more than a step.**
   On the M4 Max 128 GB their Qwen3.8-27B 4-bit cell moved from 70 tok/s with
   the MTP head (v26.9.6) to 92, 109, and 106 tok/s with a DFlash2 drafter
   verifying a tree of 8 rows (v26.10.1 to v26.10.3). The verify rows ride
   two kernel families: a split-K lane for 2 to 7 rows, and, once a drafter is
   bound, row-exact matrix-unit kernels whose rows carry the one-row step's
   bits at any width. That exactness is what lets them verify a tree and
   keep output byte-identical to serial decoding.
2. **Stock MLX GEMVs for everything bandwidth-bound.** They did not write a
   4-bit decode GEMV. Their trace of a Qwen3.5-2B decode puts MLX's
   `affine_qmv_fast` at 54% of sampled GPU time, and on Flash Next the large
   projections run at the machine's bandwidth (an out-projection at 870 GB/s,
   the 636 MB lm_head in 594 us). Their rule: a custom kernel that replaces
   one dispatch with one dispatch loses to MLX's `qmv`. Custom kernels go
   where a shape is kernel-time bound or several dispatches collapse.
3. **A dependency-chain model of dispatch cost, measured with a probe.**
   `MLX_SERVE_DISPATCH_PROBE=N` injects N exact multiplies into the expert
   chain per layer and fits the slope: 1.5 to 1.7 us per chained dispatch on
   an M4 Max (Laguna, 40 layers), 2.8 us on an M5 Ultra (Flash Next). Fusions
   that shorten the chain paid (router plus SwiGLU, 2.7%). Fusions of work the
   GPU already overlapped measured nothing, three times.
4. **Prefix reuse that carries the drafter's state.** A cache hit restores
   the trunk KV and the DFlash context or MTP history with it. Without that,
   a warm hit started the drafter blind: 72 tok/s cold against 38 warm on an
   8k echo until it was fixed.
5. **Pipelining at the step boundary, not inside the step.** The host builds
   the next round while the GPU runs this one. An async-eval ladder inside
   the step, worth 9.7% in the engine it was ported from, measured worse at
   every stride here, because the overlap was already collected.
6. **Measurement discipline that catches its own mistakes.** Engagement
   lines in the server log prove an A/B arm ran. Same-boot interleaving with
   the cell order reversed on the second pass cancels a 2% first-cell bias.
   Per-silicon tables (block caps, GEMV geometry) replace single-chip
   constants after the 2-bit GEMV tuned on an M4 Max ran 0.85 to 0.9x stock
   on an M1 Ultra and an M5 Max.

Three things do not explain the number. The 4-bit decode GEMV is stock
MLX. The one-row decode attention is stock `sdpa_vector` at head dim 256.
And the NAX kernels in `src/kernels/` need an M5-class GPU, so they do not
run on the M4 Max the release table is measured on.

## Their numbers

Decode tok/s by release, their `benchmarks.md`, Apple M4 Max 128 GB,
llmprobe's code-completion prompt, warmup discarded, median of 3, "mlx-serve
at its fastest config", with the speculative mode that engaged named beside
the number:

| Model | 26.8.11 | 26.9.6 | 26.10.1 | 26.10.2 | 26.10.3 |
|-------|--------:|-------:|--------:|--------:|--------:|
| Qwen3.8 27B 4b (ddalcu) | 70 mtp | 72 mtp | 92 dflash | 109 dflash | 106 dflash |
| Qwen3.8 27B iQ-MLX 3.8bpw | . | . | . | . | 62 mtp |
| Qwen3.8 Flash-Next mixed 4-8b | 70 mtp | 82 mtp | 88 mtp | 91 mtp | . |
| Qwen3.6 35B-A3B 4b | . | 241 mtp | 238 mtp | . | 239 mtp |
| Gemma 4 E4B 4b | 117 | 118 pld | 129 pld | 129 | 127 pld |

On an M5 Ultra 256 GB the same 27B cell reads 242 tok/s (dflash) in both
26.10.2 and 26.10.3, and the 27B 3.8bpw cell 134 to 136 (mtp).

Two readings of the 27B row matter for this page:

- The MTP cells sat at 68 to 73 tok/s for seven releases. The jump came with
  the drafter and the row-exact verify, not from kernel tuning of the plain
  path. Their 26.10.1 changelog puts the 27B plus drafter at +28% on the M4
  Max and +66% on the M5 Ultra against 26.9.6 in the same session, "with
  identical results".
- The cell is a code-completion prompt, and their own per-prompt tables show
  acceptance is a property of the prompt. On the Flash Next install at depth
  3, per-position acceptance read 0.91/0.80/0.68 on code and 0.65/0.37/0.25
  on prose. Do not read the 106 as a prose number.

For scale, and nothing more, the rows this host has: TurboSpark
`qwen38-27b-mtp` on this M4 Max 36 GB decodes 22.3 tok/s with speculation
off and 34.9 to 47.8 tok/s with MTP block 2 across three prompts, and
kapitsa's own harness read 90.7 tok/s greedy on the same model here
(`KAPITSA_KERNELS.md`, "Measured: kapitsa on this machine"). mlx-serve was
not run here, its machine has 128 GB, and its prompt differs, so the gap
between 106 and 47.8 is not a measurement. The mechanisms below are what to
take from it.

## What `src/kernels/` is, file by file

The directory the question pointed at. Each row says what the kernel serves
and which hardware runs it. "NAX" means the M5-class tensor units
(`applegpu_g17`, macOS 26.2 or later), gated at runtime by a probe that
never builds the kernel object elsewhere, because pipeline creation can fail
off-G17.

| File | Lines | Origin | Serves | Hardware |
|------|------:|--------|--------|----------|
| `nax_tiles.metal` | 837 | oMLX, from MLX steel `nax.h` | Tile and fragment types the other NAX kernels include | NAX |
| `gather_qmm_nax_header.metal`, `gather_qmm_nax.metal`, `_mapped*`, `_scan` | 1,242 | oMLX `m5_gather_qmm_nax.py` | MoE gather GEMM on the tensor units: expert-sorted tiles, a scan kernel that builds tile descriptors, a mapped form with a fused gate and up epilogue | NAX |
| `nax_attention.metal` | 640 | oMLX, from MLX `steel_attention_nax.h` | Prefill attention at q/k 192 or 128 with v 128 (MiMo, Flint sliding layers): causal, sliding window with block skip, sinks, key-range passes with an fp32 carry | NAX |
| `gdn_pipelined.metal` | 609 | oMLX `gated_delta_pipelined` | GatedDeltaNet prefill recurrence: 12-token blocks, 16 value rows per threadgroup, 128 threads, fp32 state fragments in registers, the next block prefetched into registers | any |
| `kda_percore.metal`, `kda_decode.metal` | 838 | oMLX GLM-5.3 patches | Kimi Delta Attention prefill recurrence and the one-sequence decode body for up to 8 tokens | any |
| `glm5_qmv_header.metal`, `glm5_moe_gate_up.metal`, `glm5_moe_down.metal` | 546 | oMLX GLM-5.3 decode kernels | One-token MoE decode with MLX's `qmv_fast` arithmetic copied verbatim, slot-major so rows sharing an expert read it back to back, a `volatile` to keep a product from fusing into an FMA | any |
| `qsa_nax.metal`, `qsa_nax_header.metal` | 298 | oMLX Flash Next QSA | Sparse attention over selected blocks for `qwen4_exp`, 12 heads as the rows of a 16x16 fragment | NAX |
| `h3_vsa_nax.metal` | 156 | mlx-serve, on the oMLX NAX attention loop | Block-sparse video attention for MiniMax-H3 | NAX |
| `mxfp8_qmv.metal` | 36 | mlx-serve | MXFP8 GEMV for 1 to 8 rows, one simdgroup per output row, codes decoded once and applied to every row of x | any |
| `hc_prefill_norm.metal`, `hc_prefill_mix.metal` | 42 | mlx-serve, from Apple's `rms_single_row` | Flash Next hyper-connection prefill: a norm that keeps MLX's four-value reduction and bf16 rounding, and a mix that reproduces MLX's mean rounding | any |
| `ple_gather.metal`, `ple_gather_header.metal` | 71 | mlx-serve | Flash Next n-gram table gather on the GPU, including an NVFP4 decode | any |

Three things this table says about the question as asked:

- The directory is not where the M4 Max speed lives. Of 5,315 lines, about
  3,200 are NAX-only and 1,400 serve GLM, KDA, or Flash Next. The Qwen3.8
  dense 27B on an M4 Max touches `gdn_pipelined.metal` at prefill and
  nothing else here.
- The kernels that make the 27B fast on an M4 Max are Zig string literals,
  compiled through `mlx_fast_metal_kernel` at runtime: the verify lanes in
  `transformer.zig`, and the TensorFold ports in `rowqmv.zig`,
  `simd_qmm.zig`, `row_attn.zig`, and `gdn_decode.zig`. The next sections
  read those.
- Kernel source is generated per shape class by comptime Zig, and template
  arguments ride in MLX's JIT cache key (`custom_kernel_<name>_<template
  hash>_<dtypes>`). A per-token-varying value such as the prefill chunk
  width is a scalar input, never a template value. A template value that
  varies per call is a fresh JIT per value.

## How the custom kernels are built and gated

Worth recording because the same traps exist in any JIT kernel path.

- **Comptime codegen with named scalars.** The split-K verify kernel is
  generated per (M, column tile). Its first form indexed an array
  `Vec8 v[MROWS]` and spilled: 10x slower at M=6. Named scalars with literal
  accumulator indices fixed it. M=8 remained a spill and occupancy cliff
  (round time 636 ms against 115 stock), which is why the plain-SIMD lane
  stops at 7 rows.
- **Threadgroup size is per compiled kernel, by register use.** M1 and M2
  grant this kernel family 448 or 704 threads, not 1024. A kernel above 256
  threads probes on first eval and halves its simdgroups per threadgroup
  while keeping the same summation order, so the bits do not change. Under
  test a latched MLX error is invisible and poisons later tests, so each
  kernel test drops its own latch.
- **Config caches key on the full shape.** A `metal_kernel` config cached by
  row count handed a 16-slot batched tick a 16-wide verify's shape. The fix
  is a `ShapeKey` of every dimension and an LRU for borrowed handles.
- **Small inputs land in the `constant` address space.** MLX binds inputs
  under 4 KB as `constant`, so a carry buffer read through a typed `device`
  pointer compiles on one arm and fails on the other. Index directly.
- **Reproducing an MLX op means its reduction tree and accumulator.** MLX's
  bf16 sum reduces in the input dtype. `mlx_compile` is not
  output-preserving: its SiLU fusion diverged from the op chain at token 55
  in a same-binary greedy A/B. A JIT kernel runs in a different Metal math
  mode than MLX's metallib, so `1 / (1 + exp(|x|))` lands a rounding apart
  on exactly one bf16 pattern. The fused SwiGLU therefore tabulates MLX's
  own sigmoid over all 65,536 patterns once per dtype. Laguna XS went 13.12
  to 12.81 ms per forward on it.
- **Every kernel has a kill switch and an engagement line.** The log names
  the lane that ran (`[kv-attn] verify kernel engaged ...`), and the bench
  reads the mode off `[spec-stats] mode=` rather than the launch
  environment, after a zsh word-splitting bug made a "composed" arm silently
  run the fast path for half a session.

## The quantized-linear front door

One function, `Transformer.qmatmul`, routes every quantized projection, and
the order of its arms is the design. Each custom lane returns null to
decline and the caller keeps going down the list, so a lane that fires on a
shape it does not win on is treated as a bug class there, not a tuning
detail.

1. Format-specific engines first: GGUF, Sushi fp8, Hadamard packs (the
   2-bit `qmv2` and NAX `qmv_nax2` kernels), ternary.
2. The row-exact kernels, only under `rowExactDecode` (a DFlash drafter
   bound on a dense Qwen or Nemotron-H trunk).
3. `lane_qmm` for a weight re-tiled at drafter bind (NAX).
4. `simd_qmm` for batched decode of 8 to 16 rows where no `matmul2d` tile
   exists.
5. The verify lanes (`verifyQmm`) for 2 to 24 rows.
6. The dequant-and-steel GEMM from 2048 rows.
7. Stock `mlx_quantized_matmul`, which is where every one-row dense decode
   projection lands.

Three host-side facts around it:

- The per-weight (bits, group, mode) is resolved on first touch from the
  scales dtype and shapes and cached by the scales array's pointer, so a
  later call is one pointer compare. Mixed-precision packs resolve per
  weight, not per type.
- Shapes with N under 512 never take a custom lane. The per-call kernel node
  build is about 10 us and about 50 such calls ride every verify round, so a
  tiny projection gains nothing over stock `qmv` and pays the build.
- A `getenv` per layer per forward is a libc scan of the environment block,
  so every switch is read once and cached. Same rule as our `OnceLock`
  reads.

Their reference states the cost of the exact arm plainly: exact mode costs
MTP about 30%, which is why serial decode and MTP rounds stay on stock
unless a DFlash drafter is bound. That one sentence is the gate E1 below
copies.

The MoE side has a finding that transfers as a rule even though our MoE
kernels differ. MLX's `gather_qmm` at decode width costs in proportion to
the expert bank's size, not the work: 30, 50, 77, and 300 us at banks of 8,
25, 50, and 201 MB, all reading the same 10 experts. Their in-place gather
kernels read the bank with GPU-resident indices instead, one simdgroup per
(expert slot, output row). And because MLX bills a command buffer for the
`data_size` of every bound array, binding a whole 419 MB bank forced a
commit per kernel, so the kernels bind one-expert views of the bank. Our
route-list pair already avoids per-expert argument-buffer tiling
(`MLX_KERNELS.md`), so the lesson here is the measurement, not the kernel.

## The verify lanes, which are the whole story

Stock MLX picks a transposed quantized matmul by `get_qmv_batch_limit(K, N)`
(10 on these shapes): below it the weight-reuse `qmv`, at or above it a
32x32 tiled `qmm_splitk` that wastes half a tile at width 16. The rows of a
speculative verify (2 to 16) fall in a dead zone that underuses bandwidth.
mlx-serve's answer is a lane table, `vqmmLaneForTile` in `transformer.zig`,
selected per call from M, K, N, and the machine's tile class:

| Rows M | Lane | Hardware | Notes |
|-------:|------|----------|-------|
| 2 to 7 | split-K (`splitk`) | any | 4-bit affine, gs 32/64/128, K % 64, N % 4, N >= 512. A threadgroup owns 4 output columns, K is split over 2 simdgroups for N >= 4096 (else 4), one barrier. M=7 uses a 2-column tile. 5/6/8-bit forms exist, shape-gated to N >= 5120 |
| 2 to 7, N >= 100,000 | wide `msg` tile | any | 8 independent simdgroups per threadgroup, no barrier. The lm_head class, where the tiny tile's grid thrashes the scheduler. 2.1x stock in isolation, loses to split-K in context at regular N |
| 8 to 24 | `matmul2d` shader tile | M4 family (`applegpu_g16`, macOS 26.2+) | The M5 tensor-op kernel run as plain shader code, in 8-, 16-, and 24-row tiles, 11.2 TFLOP/s. A 32-row tile exceeds threadgroup memory, and stock is at its own roofline there |
| 8 to 16 | NAX m16 tile | M5-class only | MTPLX's `nax_verify.py` tile on the per-core matrix units, K % 256, N % 32. Zero-padded to 16 rows |
| 8 to 9 | crossrow | any, opt-in | One packed-weight read serves two input rows. Measured a 4% loss here (47.5 vs 49.4 tok/s) and stays off |
| 1 to 16, drafter bound | row-exact `simd_qmm` / `rowqmv` | any | The TensorFold ports, next section. Replace the lanes above when `rowExactDecode` holds |

What each lane bought, their measurements, all on an M4 Max unless marked:

- **Split-K: +34% live on DFlash, about 0 on serial decode.** The right
  shape for a verify-width kernel. A wider tile (`bn` 8) was 5 to 8% slower
  at every width although the activation-traffic arithmetic said it should
  halve the dominant read: the x rows were 66 KB and never left cache. Two K
  parts was optimal (1: 42.5, 2: 42.4, 4: 44.5, 8: 49.0 ms).
- **The 6-bit lane was worth more than the 4-bit one it was modelled on.**
  The 27B 6-bit pack is their best quality per byte for a 36 GB Mac (95.0%
  top-1 against bf16, against the 4-bit pack's 83.6%). With the lane
  shape-gated to N >= 5120: 1.10x code, 1.22x prose, 1.17x agentic. Every
  shape forced on: 1.03 to 1.05x. The 1024-wide K/V projections give back
  about half the gain, which is why adoption is a measured predicate per
  shape and not a width list. 8-bit measured 1.01 to 1.04x and ships off.
- **The shader `matmul2d` tile made four concurrent streams pay.** One
  stream cannot fill a wide block with drafts worth verifying. Four streams
  need the rows anyway: with 4 to 8 lanes in one 16-row verify, N=4 short
  went 71.9 to 109 tok/s aggregate (Inco Splash 94.6 on the same box), N=8
  105 to 129. Whole-forward qmm at 8, 16, and 24 rows: 62.6, 73.3, and
  104.3 ms against stock 84.7, 133, and 134.
- **On NAX the m16 tile pays at depth 8.** Qwen3.8-27B 8-bit at a forced
  depth of 8: 89.81 tok/s with the tile against 79.51 stock (+13%). The
  2-bit ternary MMA kernel makes the stronger claim: every fragment
  multiplies all rows, so 8 rows cost what 1 does, and an 8-row window costs
  about what 5 rows do on the M4 plan.
- **A lane is adopted per shape on its own model.** The split-K lane on the
  Gemma 4 E4B drafter at M=5 was a small net loss, so adoption is an A/B per
  (shape, model), never a width list.
- **The parity bar is fp32-dequant truth, never kernel against kernel, and
  never byte identity with stock.** The verify tile accumulates in a
  different order than stock `qmv`, so near-tie argmaxes flip. The shipped
  4-bit lane diverges from stock at character 322 of a greedy continuation,
  the 6-bit lane at 557, both fluent. Their bar: RMS error ratio against
  stock at most 3.0x, red-verified by flipping a nibble mask.

The cost curves they recorded, which are the numbers to compare against
ours. Whole-forward time per verify width, Muse-Glimmer 30B 4-bit, M4 Max,
one eval per batch of launches (a per-launch barrier is about 0.14 ms of
sync, more than a whole K/V projection), stock lanes past 7:

| M | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 10 | 16 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|---:|---:|
| ms | 33.0 | 35.9 | 34.9 | 38.5 | 42.9 | 47.0 | 55.7 | 84.4 | 138.1 | 138.4 |

And the verify-forward ladder for the Qwen3.8-27B 4-bit pack, same machine,
one boot per rung, `--no-drafter --no-mtp`: S=1 33.7, 4 39.4, 6 49.1,
8 61.3, 12 90.9, 16 89.4 ms. S=12 costs more than S=16 because a 12-row call
wastes four rows of the same two row tiles. The round model
`emitted / (verify(S) + draft)` with draft about 9 ms predicted the live
block sweep within 3%, which made the block decidable on paper: at 100%
acceptance the ceiling is 94 tok/s at block 5, 103 at 6, 113 at 8, 160 at
16. A 106 tok/s cell at block 8 therefore means near-perfect acceptance on
that prompt.

The decomposition they did of the S=16 marginal (89.4 minus 33.7 = 55.7 ms)
is the one worth copying as a method. Summing the batched microbench over
the pack's weight inventory: qmm is 35.5 ms at M=1 (the entire serial
forward, as a bandwidth-bound decode should be) and 73.8 ms at M=16, so qmm
is 69% of the marginal at 11.1 TFLOP/s, about 70% of the M4 Max fp32 peak.
The GDN recurrence is not in the marginal (12.3 ms at T=1, 8.7 at T=16, the
marginal is negative). The remaining 19 ms is attention and glue.

For the comparison: our shipped batched GEMM costs 3.1 to 3.6x a 1-row GEMV
at 8 rows on the qwen38 shapes, and the MMA8 prototype 1.25 to 1.29x
(`INT4_SMALL_M_MMA.md`). Their split-K at 5 rows is 42.9/33.0 = 1.30x the
one-row forward, whole model, and their shader tile at 16 rows is 73.3/35.5
= 2.07x for the qmm share alone. Same regime, same conclusion: the verify
width that pays is set by this curve, and both engines spent their effort
on it.

## The row-exact kernels, which make the tree verify possible

This is the mechanism behind the 26.10.x jump, and it is the one this tree
has been circling as playbook item 1 ("make decode and verify share one
arithmetic"). mlx-serve shipped it in September 2026 as a port of
TensorFold's kernels, gated on a drafter being bound.

The contract, from `simd_qmm.zig`'s header: every row runs one fixed FMA
chain per 64-input group, the groups split over S chunks combined by a fixed
tree, and S depends on the shape only (16 chunks for N <= 2048, else 8). A
row's bits therefore never depend on how many rows ride with it. Serial
decode and a verify window agree bit for bit, and so do the nodes of a draft
tree and a chain over each node's own path.

The pieces:

- **`simd_qmm`** (4/6/8-bit, 1 to 16 rows): `mma` runs the chain on 8x8
  fp32 `simdgroup_matrix` operands for 2 or more rows, with the weight
  fragment built once per group and reused across row tiles, the activation
  fragment pre-scaled by a power of two per nibble position so masked fields
  need no shift, and the scale and bias applied per group in fp32 with the
  row's group sum. `scalar` runs the same chain with scalar FMAs for one row.
  Where a chip's MMA does not reproduce the scalar chain, a per-shape probe
  (`prepare`: 8 random rows through `mma` against 8 one-row `scalar` calls)
  detects it once and sends one-row calls of that shape through `mma`. Up to
  3 rows take a row-staged scalar variant. Past that `mma` is cheaper, and
  at 6 and 8 bits from 2 rows. This is `simdgroup_matrix`, so it runs on the
  M4 Max.
- **`rowqmv`** (4/6/8-bit, 1 to 16 rows): MLX's `qmv_fast` loop with every
  input row on its own simdgroup, 4 output rows per simdgroup. The fallback
  when `simd_qmm` declines a shape or its probe fails.
- **`row_attn`** (1 to 16 rows): one threadgroup per (kv head, window row,
  128-key chunk), keys interleaved over 4 simdgroups with an online softmax
  in ascending position, simdgroups merged in order, then chunks. What rides
  in the window changes nothing a row computes. Rows carry a depth and a
  path, so a tree node attends to the cache plus its own ancestors.
- **`gdn_decode.recurTree`**: a prework launch (conv, SiLU, q/k norm once
  per head and row) and a tree walk with one value row per simdgroup,
  restart states in registers, the state rounded to its stored dtype per
  token exactly as serial decode does. The commit replays the kept path from
  the round's input state and copies the conv window in the same launch.
- **`lane_qmm`** (NAX only, 1 to 128 rows): the same contract on the M5
  tensor units. Every 64-group is one fixed 16-row `matmul2d`, scale and bias
  applied in group order in fp32, and a weight re-tiled in place at drafter
  bind is read by nothing else afterwards. On an M5 Ultra this put the 27B
  4-bit at 245 tok/s against TensorFold's own 241 in the same session.
- **`keyed_sample`**: a hash of seed, absolute position, and token id draws
  each verify row and draft, so a sampled run is byte-identical with the
  drafter on or off, and seeded output is the same on a prefix-cache hit.

The gate, `ModelConfig.rowExactDecode`: every decode-width projection fits
the kernels, a DFlash drafter is bound, the family is dense `qwen3_5`
(which spans Qwen 3.5 to 3.8 in their naming) or `nemotron_h`, and the pack
has no Hadamard rotation. Serial decode and MTP rounds without a drafter
keep the faster stock paths. The draft tree needs it (`specTreeSupported`),
plus the GDN tree kernels, which are Metal-only.

Two measured results on the M4 family:

- Qwen3.8-27B 4-bit with the DFlash2 drafter, M4 Max: 92 then 109 then 106
  tok/s across 26.10.1 to 26.10.3 (release table above), the first column
  with the row-exact lanes and the tree at `TREE_BLOCK_CAP` 8.
- Ternary Bonsai 2 27B 2-bit over z-lab's DFlash2 drafter, M4 16 GB: 22.5
  tok/s mean (code 24.6, math 29.7, prose 13.3) with the "M4 MMA lane" and
  block 8 as the default, up from 14.6 after an f16/bf16 dtype mismatch that
  had made `simd_qmm` decline was fixed.

Their own caveats, which are ours too:

- The one-row cost of the exact kernel is paid by every serial token on an
  install with a drafter bound. Their reference puts exact mode at about 30%
  on MTP, which is why MTP and serial stay stock without a drafter. They
  accept the cost where the drafter is bound because the drafter pays for
  it, and `prepare` keeps `scalar` for one-row calls wherever its bits
  already agree.
- Exactness was lost twice after it shipped, both times by a fast path that
  bypassed the exact kernel: the fused GDN verify arm required an unjoined
  projection, so a grouped verify ran the composed chain while a solo verify
  ran the kernel, and past the fused-KV crossover the packed-KV attention
  served serial steps and verify rows with different bits. Each needed a test
  that pins the row against its solo verify on both sides of the arm.
- A grid narrower than the rows it commits writes none of the rest: a
  1-node tree committed an unwritten conv row until the prework copied state
  rows striding by the tree width. The guard runs 1-, 2-, and 8-node
  prefixes of the same tree.

## The speculative round, host side

The kernels above set the verify cost. The loop around them decides how
often the host and the GPU wait on each other, and theirs waits once per
round.

**A serial step is one async eval and one host read.** The next step's lazy
graph is built while the GPU runs the current one, submitted with
`mlx_async_eval`, and the pending token is read afterwards. Sampling runs on
the GPU, including the keyed sampler. On Flash Next the traced host phases
per step are build 1.25 ms, a 0.3 to 0.5 ms flush, and a 9.3 ms submit that
is the encoder pacing itself behind the GPU, with the resolve at 0.001 ms.
The step is GPU-bound: 11.0 ms against a 5.4 ms byte floor on an M5 Ultra.

**An MTP round touches the host once.** The draft chain is dispatched as
soon as its first chunk is built. The verify input is a lazy concat of
`[t1, drafts]`, the acceptance graph computes the argmax at every position,
the accept probabilities, and a pre-sampled correction for every possible
reject position in the same graph, and one `async_eval` realizes the whole
round. The one host read is the draft ids. The next round's chain is built
from the lazy verdict before that read where the install allows it. Their
accounting of what the pipelining bought: the CPU graph build was the
recoverable overhead, the emit gap is about 0.03 ms, and on an M5 Ultra the
lazy pre-draft is +2.6% with host encode, not the tail, bounding the rest.

**Rollback is capture, not snapshot, and not a tape.** The verify forward
writes the GDN state per position (`capture_ssm_seq`) in the recurrence
kernel itself, a partial accept truncates the KV by offset and restores the
captured state at the accepted position, and nothing is re-forwarded. A
refcount-shared KV snapshot across the verify made every `slice_update`
copy the whole buffer per layer per round (about 4.3 GB per round at 64k
context), so it is forbidden. A replay tape like our item 1 was audited and
declined there: the capture share was under 1% of a round, so there was
nothing for a lazy replay to reclaim. The difference from this tree is where
the capture lives. Ours was a host-side blit per round before the tape
(`KAPITSA_KERNELS.md`, item 1), theirs is a per-position write inside the
kernel, which is also what makes a T-row window equal T one-row calls.

**Depth comes from measured costs, not a constant.** A round-cost table
keyed by (verify width, KV bucket) holds EMAs of round milliseconds and
emitted tokens, fed from the inter-round wall clock on solo rounds only,
with warmup rounds, width transitions, and interleaved prefill ticks
dropped, and with outlier defenses (a step over 1.5x the floor is
implausible, a self-spike over 3x is dropped). The planner maximizes
expected tokens per round cost over single-chunk depths from per-position
acceptance EMAs with an optimistic prior of 0.85 (a realistic prior starves
exploration of deep positions forever, because they are observed only when
a round reaches them). Caps are per silicon: 6 off NAX, 8 on NAX, lower
cold-start rows for an M1 Pro, a base M4, and a base M5, and the table may
climb past a row only up to the widest width it has trusted samples for.
A cross-request seed carries a healthy run's EMAs to the next request on
the same model (Flash Next: 1.63x to 2.04x speculative ratio).

That is the same shape as our depth replay (`KAPITSA_KERNELS.md`, lever
2), with two differences. Theirs prices rounds from the live table rather
than a modeled 67.5 and 129.5 ms, and its token column is a workload mixture
they deliberately do not plan on. Their adaptive controller also lost to a
fixed depth on prose more than once, and the fix each time was the cost
side, not the controller.

**Acceptance is exact by default, and byte-exact with a drafter.** Greedy
accepts the longest run of argmax matches. Sampled requests use Leviathan
rejection sampling with the target's filtered probabilities for all `1 + m`
positions from one batched softmax, a one-hot proposal for argmax drafts,
and the residual against the true proposal where drafts are sampled. On a
row-exact trunk with a DFlash drafter bound, acceptance becomes serial's own
draw: a keyed Gumbel-max sampler (a hash of seed, absolute position, and
token id) draws each verify row with the key serial decoding would use at
that position, and a draft stays only when it equals that draw. Drafted
output is then byte-identical to serial at any temperature, which is the
"with identical results" in their release note. Typical and TokenV3
acceptance exist as opt-in lossy modes (+16 to +27% on their 1K to 1M
ladder), and a per-draft acceptance gate at 50% after 5 rounds turns
speculation off for the rest of a request that is not paying.

**Lookup inside the MTP round is shipped.** `mtp_lookup.zig` matches a
4-gram over committed tokens, needs an 8-token suffix, drafts up to 7 (14
after a 16-token agreement, "keeps an ordinary lookup's verify at S <= 8,
inside the fused paths"), and replaces the head chain for that round when
its tokens per cost beat the chain's. A lookup round never feeds the EV or
cost state. Under typical acceptance a copied point-mass draft made agents
loop (11 of 19 runs cut by the loop guard), so lookup rounds always verify
with exact acceptance. In a tree round, a continuation backed by 24 or more
matching context tokens is verified as a chain with no assistant forward.

This is the hybrid our Kapitsa page leaves open as lever 5 and prices at
"needs an MTP replay first". Their gate is the measured tokens-per-cost
comparison per round, not a replay.

**The DFlash2 block sweep on the 27B, before the tree.** M4 Max, 4-bit,
code and prose tok/s with accepted per round in parentheses: block 4
77.9/47.4 (2.60), block 5 82.0/50.4 (3.21), block 6 81.3/45.8 (3.71), block
7 77.4/40.3 (4.26), block 8 79.8/38.5 (4.71). Accepted per round climbs with
the block and tok/s peaks at 5 on both prompt classes, which is the M4 cap.
The tree moved the cap to 8 by making those rows worth verifying.

## The draft block is a machine property

`dflash.blockCapForMachine` is a per-silicon table, and the row for the M4
is a measured sweep: Muse-Glimmer 30B 4-bit on an M4 Max, 160-token
generations, serial reference in the same boot, block 16 at 0.92x, 8 at
1.16x, 7 at 1.43x, 6 at 1.79x, 5 at 1.97x, 4 at 1.81x, 3 at 1.77x. So the
default cap off a wide lane is 5, the M3 Ultra row is 8 (from oMLX's
measurements on the same model pairing), a draft tree takes 8, and NAX
machines take the drafter's configured block. An explicit
`--draft-block-size` clamps against the config only.

The round profile at block 5 (`MLX_SERVE_DFLASH_TRACE=1`, which inserts
eval barriers): verify 50.2 ms, assistant 6.2, draft head 2.9, append 0.9,
accept 0.2, scheduler gap 0.01. Everything is the verify, which is why the
lanes above are where the work went, and why "cross-round pre-draft" is
recorded as dead: a 0.01 ms gap has nothing to overlap.

Related levers they measured and closed, all on Muse 4-bit at block 5
unless marked: a draft-only 3-bit lm_head (a win at block 16, a loss at
block 5: 55.4 tok/s and 2.19 accepted per round against 57.8 and 2.37 on the
trunk head), adaptive block size (the spread that justified it was an
artifact of the 3-bit head), sampled drafts through the full Leviathan
ratio (exact, and a loss: 1.87x against 1.98x greedy, because this
assistant's argmax tracks the trunk while its distribution shape does not),
and assistant micro-optimizations (halving its weight read was no faster at
identical acceptance).

## Dispatch cost: the probe, and what it does and does not license

The method is the part to take. `MLX_SERVE_DISPATCH_PROBE=N` adds N
elementwise multiplies by an exact 1.0 per MoE layer inside the expert
path, so MLX cannot elide them and the result is consumed. The slope is the
cost of a dispatch on the dependency chain:

| Host, model | 0 | 4 | 8 | slope |
|-------------|--:|--:|--:|------:|
| M4 Max, Laguna XS, 40 layers | 13.08 / 13.13 | 13.37 / 13.31 | 13.63 / 13.66 ms | 1.5 to 1.7 us |
| M5 Ultra, Flash Next 4.7bpw | 11.07 | 11.61 | 12.13 ms | about 2.8 us |

That slope is an upper bound that only materializes for a fusion that
shortens the chain. Their record:

- **Paid**: the MoE router and the SwiGLU, both on the chain (router to
  expert gather, gate/up to activation to down), 2.7% together, byte
  identical. A decode-width residual add fused with the next block's norm
  (`add_norm.zig`, bit-equal to `add -> fast::rms_norm`). The MoE down-path
  tail (down gather, score multiply, sum) as one kernel, 1.5%. Three fused
  GDN decode dispatches per layer, 3.9% of a Flash Next forward.
- **Null**: the attention output gate (four dispatches off the chain, the
  GPU already overlapped them with the q/k/v projections), the gated shared
  expert folded into the expert kernels (8 chained dispatches became 2 and
  the forward got slower, because each replacement kernel was slower than
  MLX's `qmv` plus elementwise), joining the four GDN in-projections into
  one qmm (59.96 vs 59.97 tok/s), the shared expert as an 11th slot of the
  gather kernels (3% worse: the separate chain was routing-independent and
  ran in parallel with the router).
- **Rule they wrote down**: a chain the GPU already overlaps is not a
  dispatch to fuse. A fusion pays only if it shortens the dependency chain.
  On a step that is kernel-time bound, a kernel that replaces one dispatch
  with one dispatch cannot win, only one that replaces several and is no
  slower itself.

Two more findings in the same family:

- **Small dense models spent a third of a token on dependent launches.**
  Spark 4B, Gemma 4 E4B, and LFM2.5 2.6B decoded at 55 to 65% of the
  bandwidth floor while the 27B sat at 88%. A 4B forward was about 940
  kernels at about 3 us of GPU idle per dependent small kernel. The fixes
  were all bit-identical: q|k|v and gate|up rows joined at load into one
  buffer whose axis-0 views the separate paths keep (decode width only),
  one residual kernel doing pre-norm, add, scale, and post-norm with MLX's
  own `rms_single_row` tree, unary activations tabulated once per dtype by
  running the op chain over all 65,536 patterns, then one multiply. Decode:
  Spark +10%, E4B +9%, Gemma MoE +7%, LFM2.5 +4%, prefill untouched.
- **Bigger command buffers lose.** Packing a Flash Next step into about 17
  buffers instead of about 100 read 10.82 vs 10.45 ms GPU: the encoder
  stopped being throttled but the GPU started later and the overlap at the
  step's head and tail went. The 5 us between kicks is not recoverable by
  packing.

Against our own numbers: the playbook derived about 1.6 us per removed
dispatch from two timing samples on Qwen3.8 prefill, and Whisper's measured
floor is 5.4 us GPU. mlx-serve's probe slope of 1.5 to 1.7 us on an M4 Max
is the same figure measured properly, and their "only chain-shortening
fusions pay" explains why the first dispatch-count estimate here overshot by
three.

## Prefill: stock, with two interventions

- **Stock `qmm_t` has a dead zone from M >= 2048.** Its 32x32x32 tile stalls
  on qwen shapes, so `prefillDqGemm` dequantizes the weight to a bf16 [N, K]
  transient (under 1 ms) and runs the dense steel GEMM near peak. Numerics
  differ by bf16 weight rounding only. On the M5 Ultra stock qmm is the
  fused NAX kernel and beats the route below about 0.9 x min(N, K) rows, so
  the floor is per call there. The chip's GEMM tile picks the crossover.
- **One long causal dispatch tripped macOS IOGPU preemption.** A single
  `msv_attn_p256` dispatch scanning the whole KV at high kv_len won every
  microbench and lost every live A/B. Capping per-dispatch work and carrying
  the online-softmax state through fp32 buffers between key-range dispatches
  was bit-identical to the single dispatch and turned the loss into a win.
- **Where the roofline is.** Against oMLX on a dense 27B, same session: 2k
  268 vs 243 tok/s (+10%), 8k and 32k ties. Their arithmetic: 2 x 27e9
  FLOP per token at MLX's measured 14.4 TFLOPS bf16 GEMM rate (89% of the M4
  Max's 16.2) puts the 8k roofline at about 30.9 s and both engines within 3
  to 6% of it. "Nobody beats anybody by 5% on a dense-27B prefill on this
  hardware." The winnable margins are short contexts and MoE or small models.
- **Verify forwards are not prefill.** Their periodic mid-graph eval, which
  bounds GB-scale transients during 2048-token chunks, fired on every verify
  width above 1 and drained the pipeline 16 times per MTP round on a 64-layer
  model. `PREFILL_EVAL_MIN_SEQ = 32` exempts them. The same class exists in
  any chunked driver that keys a prefill-only behavior on `seq_len > 1`.

The ANE prefill split they ship is M4-and-below only, opt-in, lossy (int8
rows, fp16 datapath), +12% median at 16k and +18% at 32k on the 27B 6-bit,
and a loss on both M5 Max testers (the NAX GPU prefill outruns the seam).
Ours (`SPEEDUP_PLAYBOOK.md`, "Opt-in: the ANE+Steel prefill split") is the
same conclusion from a different implementation.

## Attention and KV at decode

The attention dispatch for the Qwen 3.5 family is one function
(`gatedFullAttnWith`) whose arms, in order, are: QSA for Flash Next,
packed-KV kernels when the cache is quantized, the row-exact kernels when a
drafter is bound (serial steps included, so one row and sixteen rows run one
arithmetic), a refusal if a draft tree reaches a shape without a row-exact
kernel, the prefill kernels for any `seq_len > 1`, and stock MLX for one row.

- At head dim 256 and one row, decode attention is MLX's `sdpa_vector`.
  Nothing custom runs there on a dense bf16 cache. MLX's fused kernel
  refuses `q_len * gqa > 32`, so causal verify widths 2 to 15 are split into
  groups of `min(8, 32 / gqa)` rows and sent through the vector kernel one
  group at a time (exact, rows are independent under a per-row mask).
- `msv_attn_p256` is the head-dim-256 prefill kernel, not a decode kernel.
  It declines under 16 query rows, and dispatching verify widths through it
  once measured decode at 18 tok/s against 48. It is their own port of MLX's
  steel attention with a 64-row query tile, 32-key blocks, 8 simdgroups, a
  register-resident Q, fp32 fragments on `simdgroup_float8x8`, and exp2
  softmax. One threadgroup is one query head, so the K/V tile is re-staged
  per head of a GQA group. Long key ranges are split across dispatches with
  the online-softmax state (max, sum, unnormalized O) carried in fp32
  buffers, bit-identical to one dispatch.
- The one custom dense decode attention is `dec_attn.zig`, MiMo's global
  layers only (q/k 192, v 128). A KV group's query heads times the query
  positions (one at decode, up to 8 in a verify) are the rows of one MMA
  tile, so a key split reads its keys once for every head that shares them.
  A 4-row verify went 35.9 to 29.6 ms at 8k keys, 49.8 to 33.5 at 32k, 69.1
  to 35.0 at 64k. At one row it engages only from 8,192 keys, because MLX's
  vector kernel is as fast below that. The design is our GQA-shared decode
  partial (`SPEEDUP_PLAYBOOK.md`, first row of the table) with the verify
  rows added, and it is held to a 2x-of-stock error bar, not bit equality.
- Fused QK-norm plus RoPE at head dim 256 serves 1 to 32 rows in one
  dispatch, bit-identical to the composed chain. The recipe: cos and sin
  tables extracted by probing stock `mlx_fast_rope` with a ones row and a
  zeros row, never re-derived, the RMS norm mirroring `rms_single_row` with
  `precise::rsqrt`, the rotation in fp32 with one rounding to the output
  dtype. Our batched layers still encode the per-head norm and the RoPE as
  separate row-batched dispatches.
- Quantized KV at head dim 256 runs through Apple's `matmul2d` tensor op
  over a threadgroup tile (32-token KV pages dequantized into tile memory,
  one thread per query-row group, online softmax, split-KV partials merged
  at the end), structure after Inco Splash. It is tried first and declines
  outside a window: 4 to 8 query rows from 2048 KV tokens, 2 to 3 from 8192,
  one from 16384. Their `--kv-quant` table: about 10% slower decode at 2 to
  4k context, and at 42k the fused packed reads cut the penalty from 47% to
  18%. A verify-width eligibility bug here once collapsed MTP decode to 28
  tok/s past 8k where the dense read did 60.
- Sliding-window layers trim every attention read to the window, prompt
  chunks and speculative steps included: Muse-Glimmer 30B at 64k context
  8.6 to 21.0 tok/s.
- No KV snapshots across a verify. A refcount-shared snapshot made the next
  `slice_update` copy the whole cache per layer per round (about 4.3 GB of
  copy traffic per MTP round at 64k on Qwen3.6-27B). Rollback is scalar
  anchors, per-position SSM captures, and an offset-only truncate.
- The cache is head-major `[B, H, T, D]` with position on axis 2, views
  sliced along T out of a larger buffer, so every custom kernel takes K/V
  through strides and never forces a contiguous copy. Growth is proportional
  (a quarter of capacity, floored at one chunk, capped at 8,192 tokens).
  Sliding layers keep the full buffer and trim the view.

### GDN, three prefill kernels and a two-dispatch decode

- Prefill has three recurrence kernels behind one route. The stock per-token
  port. A blocked kernel (oMLX) that stages k, q, and v in 16-, 32-, or
  48-token blocks in threadgroup memory, keeps the state register-resident
  as (value row, 16-wide segment) fragments, and reduces the k-state and
  q-state contractions over the 8 segment lanes with shuffles, with no
  barrier in the token loop. And `gdn_pipelined.metal`, where "pipelined"
  means software pipelining at two levels: the next 12-token block's k, q,
  v, g, and beta are prefetched into registers while the current block
  computes, and inside a block step t+1's loads are issued before step t's
  reductions. The pipelined kernel sums the two 128-wide dots in a different
  fp32 order, so greedy text can move, and it is on only on NAX for the
  archs they measured it on.
- Decode is two dispatches per layer. K1 runs one head over 4 threadgroups,
  each recomputing the conv, SiLU, and q/k norm prework for its head before
  its slice of the recurrence rows, because the recompute is cheaper than a
  barrier between kernels. K2 does the per-head gated RMS norm (and the
  Hadamard rotation on those packs). Bit-identical to the composed chain.
  This is the fusion our Kapitsa page declined as item 9 at an estimated 1%
  or less. Their measured figure for the three-dispatch form on Flash Next
  was 3.9% of a forward, on a model where the GDN layers are a larger share.
- The verify form runs the same per-token prework and recurrence token after
  token for up to 16 rows, writes the per-step state capture that rollback
  reads, and rounds the state to its stored dtype after every token so a
  T-row window equals T one-row calls. The state is `[B, Hv, Dv, Dk]`, the
  capture `[T, B, Hv, Dv, Dk]`, kept in fp32 on Flash Next by default
  because a bf16 gate near one is exactly 1.0 and that head never forgets.

### Numerics rules they wrote down

- A running max initialized to a finite -3.0e38 rather than -inf, so the
  rescale factor `exp2(old - new)` can never be `exp2(nan)`. Masked scores
  still use a true -inf, and a merge skips a split no key reached.
- A composed causal mask built as `triu(ones) * -inf` produces `0 * -inf =
  NaN` that every parity loop hid, because `NaN > x` is false. Build masks
  with a select, and assert finiteness before diffing.
- `metal_kernel` JITs custom sources in Safe math mode and mlx-c offers no
  way to change it, so a JIT transcendental can land one rounding from the
  metallib's. A 16-bit activation has 65,536 inputs, so sweep them all
  rather than sampling.
- `precise::rsqrt` wherever an RMS norm must match MLX's `rms_single_row`.
- A `volatile` compiler barrier pins tile-load scheduling around the NAX
  matrix op in two kernels. The same trick as our `volatile` half for the
  f32-to-f16 round trip (`KAPITSA_KERNELS.md`, items 4 and 5), used for a
  different purpose.
- Every kernel's doc comment says whether it is bit-identical to the chain
  it replaces or not, and the ones that are not (the pipelined GDN, the
  head-dim split in the NAX attention, `dec_attn`, the packed-KV decode)
  are gated so an exact path never reads them.

## Negative results worth not re-deriving here

Each is theirs, measured, and recorded with the number. Several match
entries already on this tree's dead-end lists, which is the point of
listing them.

- Intra-step async-eval ladder (fire `async_eval` at layer boundaries):
  monotonically worse as the stride tightens (39.88 off to 41.49 ms at
  stride 1), because the step boundary already overlaps.
- Dispatching each MTP draft step as it is built: neutral (predraft lap 4.37
  to 4.09 ms in one run, 4.40 to 4.33 in the other), because the lap
  measured CPU build time, which early dispatch does not shrink.
- An 8-bit MTP head: 1.7% and 1.0% slower with acceptance +2% / -3%. Head
  weight precision is not where the round cost is.
- Crossrow M 8/9 lane (one weight read serves two rows): 4% loss, because
  stock's GEMM tile already reads each weight once at those widths.
- An `sg8` simdgroup-matrix lane for M 8 to 16 off NAX: reverted, later
  replaced by the shader-compiled `matmul2d` tile.
- A row-interleaved FMA construction inside the split-K tile: measurably
  loses to one sequential dequant-and-FMA chain per column.
- A grouped-expert NAX tile for verify-width MoE: about 9% loss in both
  variants. At verify widths a run is 1 to 2 rows, so the 16-row tile's
  amortization never engages while its cost is paid, and the sorted gather is
  already near the expert-byte floor (about 1.7 ms per row).
- A 64x64 NAX prompt tile for 2-bit: 0.87 to 0.93x stock, because the
  toolchain has `uint4b_format` but no 2-bit tensor format.
- A take-plus-batched-qmm MoE decode as a default: adopted on a 3x inflated
  measurement, lost when re-measured, opt-in now.
- Prefill attention tile variants: an 8-key ping-pong (the per-block
  rescale doubles), V fragments from device memory (too slow), a K-only
  double buffer (barrier structure unchanged), a 32-row Q tile with two
  threadgroups per core (cache thrash at long key lengths). All measured
  worse than the 64-row tile.
- The packed-KV decode kernel's first two forms: one threadgroup per query
  head with a `simd_sum` chain (5x slower), then 24 to 28 KiB of threadgroup
  memory (1.7x slower, one threadgroup per core). Occupancy, not math.
  Threadgroup memory is kept under about 10 KiB there.
- A concatenated q/k/v weight: bit-identical and a null at both widths, so it
  ships opt-in because the copy costs 1.34 GB on Laguna XS.
- Warming up at the real KV length: no effect, their metallib is compiled
  ahead of time.
- A base-3 repack of ternary codes (18.75% fewer bytes, bit-identical): 20 to
  30% slower in a dependent chain, the unaligned 26-bit fields cost two
  word loads per lane.
- Threadgroup-staged activations and register-blocked rows in a multi-row
  GEMV: both lost, the same two findings as our `dequant_int4_gemm_simd`
  header (`SPECULATIVE_DECODING.md`, "c(M), re-measured").
- A split-K small GEMV for the 150 small dense GEMVs per Flash Next step
  (router 512 rows, shared expert 640): +0.1 to +0.2 ms GPU over stock in
  two forms. Launch-latency bound, not parallelism bound.
- TurboQuant-style rotated KV (a Hadamard before `mlx_quantize`): removed.
  No fused path read it, so it dequantized and un-rotated the whole cache
  every token, 14% slower than affine at 4k and a third slower at 16k.
- Sharpened stochastic MTP drafting (temp 0.6 proposal, full Leviathan
  accept): loses to greedy argmax on low-entropy code, wins on flat prose
  rows, so the proposal is now chosen per request from the target's own
  sampling parameters.
- Prompt lookup as a general drafter: default on, but gated by a 3-gram
  repetition score on the prompt and a sticky runtime acceptance gate at 50%
  per draft, and exempted entirely for DFlash. Our replay
  (`KAPITSA_KERNELS.md`, "Prompt lookup") reached the same place.
- The DFlash yield gate calibrated on dense trunks sent a sparse target's
  winning drafter to plain decode. A sparse target takes an absolute floor of
  1.8 accepted per round.

## Decision table

Scope: single-stream decode on the dense Qwen installs this host measures,
then aggregate. "Have" means the mechanism is already in this tree.

| # | Idea | Status here | Where |
|---|------|-------------|-------|
| 1 | Row-exact small-M matrix-unit GEMM, decode and verify on one arithmetic, gated on a drafter | Converges with playbook item 1 and item 7 (`TURBOSPARK_INT4_MMA8`). Their gating and per-shape probe are the new parts | Engine, lever E1 |
| 2 | Draft tree verify over the exact kernels (DFlash2 lattice, block 8) | Replay and costing done, not built (`KAPITSA_KERNELS.md`, KFlash). Their shipped result raises the priority | Engine, lever E2 |
| 3 | GDN tree walk with restart states in registers, per-token state rounding | Prototype exists (`tv_gdn_tree` in `tree_verify_cost_bench.rs`), bit-identical | Engine, with E2 |
| 4 | Dispatch probe (exact multiplies injected on the chain) | Built: `TURBOSPARK_DISPATCH_PROBE`, `tslab profile --dispatch-probe`. Dense qwen and llama FFNs only. Read 1.87 us per chained dispatch on `qwen38-27b` (below, L1) | Lab, L1 |
| 5 | Whole-forward verify-width ladder from the weight inventory | Built: `tslab kernel --bench verify_ladder [--verify-from EXP]`, shapes and counts from the resident index | Lab, L2 |
| 6 | Engagement lines as A/B evidence | Built: `foundation::engaged!`, checked per run against the knob table and `expect_engaged` | Lab, L3 |
| 7 | In-situ stand-in ablation with sane values | Built: `TURBOSPARK_STANDIN`, `tslab profile --standin` (residual passes through; dense qwen and llama trunks). Qwen3.8 decode: FFN 61%, GDN 24%, attention 7% (below, L4) | Lab, L4 |
| 8 | Metal System Trace analyzer | Built: `tslab trace` (attach after warm-up, short window, never quotable) | Lab, L5 |
| 9 | Rows-per-simdgroup sweep in situ before deepening a multi-row kernel | Built: `rows_attn` bench arms over rows per threadgroup and Q heads per simdgroup | Lab, L6 |
| 10 | Per-silicon tables (block cap, GEMV geometry) | Single-machine repo today. Record as a rule for any default | Engine, rule |
| 11 | Fused residual add plus next norm, bit-equal | Not checked here. The Qwen batched layers encode them as separate row-batched dispatches | Engine, E3 |
| 12 | Joined q/k/v and gate/up at load (decode width), tabulated activations | Pays on small models (55 to 65% of floor). Our 27B target is at the floor like theirs. Low value for qwen38, worth a profile on the small installs | Engine, E4 |
| 13 | Prefix cache carrying drafter state (MTP history, DFlash context) | Partly: `feat/gdn-turn-snapshot` rewinds the MTP drafter on reuse. DFlash2 context is not carried | Engine, E5 |
| 13b | Lookup drafts inside an MTP round, gated per round on tokens per cost | Open as `KAPITSA_KERNELS.md` lever 5 ("needs an MTP replay"). Shipped there with a per-round cost gate and exact acceptance | Engine, E6 |
| 13c | Keyed per-position sampling so a sampled verify row reproduces serial's draw | Not here. Ours is exact rejection sampling, which preserves the distribution but not the bytes | Engine, note under E2 |
| 13d | Round-cost table measured live per (width, KV bucket) feeding the depth planner | Built: `round_cost` ledger records, `depth-replay --costs ledger:EXP` | Lab, L7 |
| 13e | Fused QK-norm plus RoPE in one dispatch at head dim 256, bit-identical | Separate row-batched dispatches here. On the chain, so it may pay; their bar was a live paired A/B | Engine, measure first |
| 13f | Two-dispatch GDN decode (prework recomputed per threadgroup) | Declined here as Kapitsa item 9 at an estimated 1% or less. Their measured 3.9% is on Flash Next, where GDN is a larger share. Re-estimate with the dispatch probe (L1) before reopening | Engine, measure first |
| 14 | Split-K lane for 2 to 7 rows | Superseded by the matrix-unit route (item 1), which they also moved to | Declined |
| 15 | Stock-MLX GEMVs for decode | Not applicable: this engine has no MLX dependency and its GEMV already dequantizes in registers at bandwidth | Not applicable |
| 16 | NAX kernels (`gather_qmm_nax`, `nax_attention`, `lane_qmm`) | Need an M5-class GPU. Re-read when one is on the bench | Deferred |
| 17 | Async-eval ladder, crossrow lane, 8-bit head, base-3 ternary repack, sampled DFlash drafts | Measured negative there, mechanism-level reasons apply here | Declined |

## How each engine lever would be done here

Ordered by expected single-stream value on `qwen38-27b`, then aggregate.
Estimates are marked. The qualification bars are the playbook's.

**E1. Gate the shared-arithmetic tile on the drafter, and probe one-row
exactness per shape.**

Why: the MMA8 tile costs 1.24 to 1.27x the GEMV at one row
(`INT4_SMALL_M_MMA.md`), which is why flag-on loses 9 to 12% on prose where
block 2 is best. mlx-serve pays the same one-row tax only where a block
drafter is bound, and keeps the scalar kernel for one-row calls whose bits
already agree with the tile.

Where: `real_forward_dispatch.rs` (route on "drafter open" rather than a
global flag) and `dequant_int4_mma8.rs` (a per-shape `prepare` that runs 8
rows through the tile against 8 one-row scalar calls and records which
shapes need the tile at one row).

Gate: bit equality between a one-row decode and the same row inside a
verify (already the item 7 bar), `mtp_generation_gate`, and the three
quality gates for the headless install, which must not move because the
route never fires without a drafter. Measure: `tslab ab` on
`qwen38-27b-mtp` arms `plain`, `mtp2~mtp:2`, `mtp4~mtp:4`, and on the
DFlash2 install `d7~dflash:7`. Expected: the prose loss disappears on the
headless install by construction, and the code gain (+16% at block 4) stays.

**E2. Build the DFlash2 tree verify at depth 7 to 8 over the flat verify.**

Why: our replay gives 1.07 to 1.12x committed per round at depth 7 and the
costing gives about 1% overhead (`KAPITSA_KERNELS.md`, "Costing the tree").
mlx-serve's shipped 27B cell went from 70 to 106 tok/s with the tree at
block 8 on an M4 Max, their verify ladder puts block 8 at 61.3 ms against
33.7 for one row, and their tree's exactness comes from the same row-exact
contract E1 provides.

Where: the four pieces the KFlash section lists (ancestor-masked verify
attention with a row-against-decode parity test, parent-state GDN rows from
the `tv_gdn_tree` prototype, per-row RoPE positions, KV and capture
compaction), then the host tree build in place of the selector's greedy
walk, scoring children only when a node is expanded. Their two traps to
design against from the start: a fast path that bypasses the exact kernel
on one side of an arm (pin every row against its solo verify on both
sides), and a grid narrower than the rows it commits.

Gate: lossless against plain greedy at depth 2 and 7, byte-identical
generated text, `tslab tree-replay` count against the live committed per
round. Measure: `tslab ab` arms `d7~dflash:7` against `d7tree~dflash:7`
under E1.

**E3. Fuse the decode-width residual add with the next norm, bit-equal.**

Why: on the chain by definition, and mlx-serve's `add_norm.zig` is
bit-equal to the composed ops. Our batched Qwen layers encode the residual
add and the post-attention norm as two row-batched dispatches per layer
(`batched_layers.rs`).

Sizing, an estimate: on the 27B decode step that is 2 x 64 dispatches at
about 1.6 us, so roughly 0.2 ms of a 45 ms step. Under 1%, below the host's
A/B floor on its own. Worth doing only bundled with E1 and E2 verify work,
where the verify round is the unit. Gate: `to_bits()` equality against the
two-dispatch form on rows of different magnitude.

**E4. Profile the small installs for launch-bound chains before any kernel
work there.**

Why: mlx-serve's 4B-class models ran at 55 to 65% of the bandwidth floor
and the fix was dispatch structure, not kernels. Our playbook item 4
("Audit the other decode families for passes and lanes") is the same
hypothesis.

Where: `tslab profile` on the smallest dense installs (Qwen3.5 2B or 4B
class, Gemma 4 E4B if installed) with the dispatch probe from L1 to get the
chained-dispatch price on this host. Then the three bit-identical structural
changes if the profile shows the gap: joined q/k/v and gate/up at decode
width, one residual kernel, tabulated activations. Gate: bit-identical
greedy transcripts, which they achieved on 17 packs.

**E5. Carry the DFlash2 context across a prefix reuse.**

Why: a restore forwards no trunk layers, so the drafter's captured hiddens
are empty on a warm hit. Their measurement was 92.6% to 66.5% per-draft
acceptance and 80.2 to 60.9 tok/s on a full-prefix hit until the context
rode the cache entry. `feat/gdn-turn-snapshot` rewinds the MTP drafter
already.

Where: the turn snapshot carries the DFlash2 capture rows for its 8-token
margin. Gate: acceptance per position on a cached follow-up equal to the
cold run, `turn_snapshot_gate` extended to the DFlash2 install.

**E6. Lookup drafts inside the MTP round, gated per round.**

Why: our replay (`KAPITSA_KERNELS.md`, "Prompt lookup") put a standalone
lookup drafter at 1.33 to 1.58x on copy-heavy output and flat or losing on
prose, about what MTP block 2 gives everywhere, and left the hybrid open
pending an MTP replay. mlx-serve ships the hybrid: a 4-gram over committed
tokens with an 8-token suffix, up to 7 drafts, replacing the head chain for
one round when its tokens per cost beat the chain's, and never feeding the
depth planner.

Where: a draft source in the speculative loop that tries the n-gram first
and otherwise runs the MTP head, with the per-round gate reading
`TURBOSPARK_SPEC_STATS` style round costs rather than a prior. Two traps
they recorded: a copied draft under any approximate acceptance loops (use
exact acceptance for lookup rounds), and lookup rounds must not update the
acceptance EMAs. Gate: greedy output identical to the sequential loop,
`mtp_generation_gate`. Measure: `tslab ab` on `qwen38-27b-mtp` with the
`copy-heavy` workload and one prose protocol case, arms `mtp2~mtp:2` and
`lookup~mtp:2`.

**Rule, not a lever: a default belongs to the engine and the chip that
measured it.** Their 2-bit GEMV geometry tuned on an M4 Max lost on an M1
Ultra and an M5 Max, their block cap 5 was an M4 number wearing a universal
constant's name, and their ANE split lost on every M5. This repo measures
on one M4 Max. Any default that encodes a geometry or a width should carry
the chip it was measured on in its comment, and the lab's ledger already
records it per run.

## What the lab should take

`crates/lab` (`tslab`) measures with fresh processes, interleaved arms, a
null floor, quality and memory gates, and per-shape kernel curves. These
are the pieces mlx-serve's gotchas show earning their keep that tslab did
not have. All seven are now built (decision table rows 4 to 9 and 13d say
where); the text below is the design they follow. L1 and L4 have a first
reading on `qwen38-27b`, under their entries. The others have none yet.

**L1. A dispatch probe.** Add `--dispatch-probe N` to `tslab profile`. The
runtime side is one knob, `TURBOSPARK_DISPATCH_PROBE`, that encodes N exact
multiplies by 1.0 on the FFN chain per layer (the result must feed the next
kernel so nothing elides it). The lab runs N in {0, 4, 8}, fits the slope,
and writes `us_per_chained_dispatch` to the ledger.

It replaces the 1.6 us figure the playbook derived from two samples, and it
gives every future fusion estimate a measured price. The probe is pure and
observable, which is why it beats "run the component twice and read the
marginal".

*Measured, 2026-10-25* (ledger `qwen38-dispatch-probe-3`): `qwen38-27b.gturbo`,
M4 Max, AC power, warm page cache, host load at most 2.5, `short-explanation`
prompt, 64 decode tokens, one warmup round and 6 measured rounds with arm
order rotated. All 24 measured runs passed the host guard and printed the
probe's engagement line.

| Probe per layer | Added dispatches per token | Decode ms/token, median | Range over 6 runs |
|---:|---:|---:|---:|
| 0 | 0 | 45.46 | 45.18-45.79 |
| 4 | 256 | 46.07 | 45.92-46.23 |
| 8 | 512 | 46.60 | 46.34-46.73 |
| 16 | 1024 | 47.40 | 47.22-47.57 |

The fit over the medians gives **1.87 us per chained dispatch** (largest
residual 0.2%). Fitted one round at a time, the slope is 1.71-1.96 us,
median 1.81. Each count's range sits above the one before it, and the
effect at 16 per layer (about 1.9 ms) is three times the widest range within
an arm. That agrees with mlx-serve's 1.5 to 1.7 us on an M4 Max and with the
playbook's two-sample 1.6 us. It is the chained price of an elementwise
dispatch in the dense decode trunk, an upper bound on what a fusion saves
(one that shortens the chain), and not a frozen `docs/BENCHMARKS.md` row.

A first run at 2 measured rounds and counts 0, 4 and 8 read 2.26 us
(`qwen38-dispatch-probe-2`): there the unprobed arm's own two runs differed by
about 1.7 ms, more than the probe's effect at 8. Use enough rounds that the
largest count's effect clears the unprobed arm's spread.

**L2. A whole-forward verify-width ladder.** `tslab kernel --bench
int4_matmul` times each shape alone. Add a `verify_ladder` bench that walks
the install's weight inventory from `ArchConfig` (every projection, with its
layer count) and sums the per-shape times at each M into one row, the way
their `MLX_SERVE_VERIFY_WIDTH_UBENCH` does. Record alongside it the measured
verify forward from `TURBOSPARK_SPEC_STATS` at the same M. The difference is
the non-GEMM marginal (attention, GDN, glue), which is the decomposition
that told them qmm was 69% of a 16-row marginal and the GDN recurrence none
of it. Report c(M) for the whole model, not per shape, because the block
decision is made on the whole model.

**L3. Engagement lines as part of an arm.** The knob registry in `knobs.rs`
names each switch. Add an optional `engagement` substring per knob (the log
line the runtime prints when the lane runs), and have the worker fail an
arm whose log lacks it. Their zsh word-splitting bug ran a "composed" arm on
the fast path for half a session and read a 2x win as neutral. Our
`TURBOSPARK_ATTN_DECODE_KERNEL=barrier` trap (it forces the trunk's verify
too) is the same class from the other side: an arm should prove what ran.

**L4. Stand-in ablation with sane values.** "Attribute by skipping" in the
playbook aborted on stale NaNs. Their `QWEN4_STANDIN=gdn,attn,mlp` replaces
a block's output with a finite stand-in (zeros collapse the downstream graph
and read as a 4 ms forward, so use the block's input or a constant of the
right scale). Add a `TURBOSPARK_STANDIN` knob per block class and a
`--standin` axis to `tslab profile` that prices each block in situ. It
sizes a block without the dispatch profiler's per-encoder inflation.

*Measured, 2026-10-25* (ledger `qwen38-standin-2`), same install, host and
workload as L1, 2 measured rounds per arm. The base arm decoded at 21.0-21.1
tok/s (47.52 ms per token, median of its host-clean runs).

| Block skipped | Decode tok/s with it skipped | Cost in place, ms/token | Share of a token |
|---|---:|---:|---:|
| FFN (dense, 64 layers) | 53.2-53.7 | 28.82 | 61% |
| Gated-DeltaNet (48 layers) | 27.4-27.8 | 11.27 | 24% |
| Full attention (16 layers) | 22.4-22.9 | 3.36 | 7% |

The three add to 43.4 of 47.5 ms. The rest, about 4 ms, is the head, the
norms and the embedding. Each arm's engagement lines confirm what was
skipped: the attention arm printed no decode-attention line and the FFN arm
no fused-FFN line. These are two-round readings for ranking blocks, never
quotable as throughput, and a skipped block also changes what the later
layers compute, so a share is the block's cost on this path rather than an
exact decomposition.

**L5. A Metal System Trace command.** `tslab trace` wraps `xcrun xctrace
record --template "Metal System Trace" --instrument "Metal GPU Counters"`
around a `turbospark-check` run and exports two tables: compute-encoder
intervals (busy percent, idle gaps, encoders per step) and the shader
profiler intervals (per-kernel share and count). Their analyzer is about 60
lines of Python over `xctrace export`.

Rules they recorded for reading a trace:

- Kernel samples are about a tenth of GPU time, so compare percentages
  between an A and a B trace, never microseconds.
- Cut the window to the decode span before reading busy percent.
- Low busy inside a pure decode window means launch-bound, high busy means
  kernel-time bound.
- Never quote tok/s from a traced run.

This complements `TURBOSPARK_DISPATCH_PROFILE`, which ranks dispatches but
gives each its own encoder and so inflates.

**L6. Rows per simdgroup as a kernel bench axis.** Their affine-4 MoE
kernels went 110.5 to 114.8 tok/s by changing rows per simdgroup from 4 to
1, after three plausible fixes lost. Our GQA rows kernel found the same by
hand (one Q head per simdgroup won, more rows per threadgroup lost).

Where a bench kind has that constant, sweep it as an arm. The int4 bench
already has arms per tile. The attention bench should take heads per
simdgroup and rows per threadgroup as arms too.

**L7. Round costs from the ledger, not from two constants.** `tslab
depth-replay` prices a round at a modeled 67.5 ms for depth 2 and 129.5 for
depth 4 and interpolates the rest. mlx-serve's planner reads a table keyed
by (verify width, KV bucket) fed from live rounds, with warmup, width
transitions, interleaved ticks, and spikes over 3x dropped. The lab already
records `ms_per_round` per arm through `TURBOSPARK_SPEC_STATS`. Fold those
into a `round_cost` record per (install, width, KV bucket) in the ledger
and have `depth-replay` read it, so a replay on a 16K prompt prices
16K-bucket rounds and not 2K ones.

Their rule about the table applies: the token column is a workload mixture,
so plan on time and take tokens from the acceptance record.

**L8. A projection and a calibration.** `tslab project` prices a kernel mode
from kernel curves, per-forward counts, a measured base and, where the mode
changes the dispatch count, the dispatch probe's price. It reports the share
of the base the benched kernels explain, so a mode whose bench covers little of
a token says so before the A/B. `tslab calibrate` joins that record to the
verdict that followed and ledgers the miss, and `tslab calibrate --all` shows
the bias per bench and arm pair. Built after the 8-row tile's small-projection
lever, where a microbench and a dispatch profile predicted a gain that the
clean wall-clock sweep did not show (`docs/KAPITSA_KERNELS.md`, lever 9).
Contract and reading: [KERNEL_MODES.md](KERNEL_MODES.md).

**L9. Exactness hints from the hashes.** `tslab kernel --hash` now also stores
the hash of the first output row alone and prints, per arm, whether its output
bits equal the first arm's and whether that first row changes with the batch
beside it. A tile that differs but is row-invariant is the shared-arithmetic
signature (decode and verify can run it together). One that differs and varies
is a drift bound. It is a hint from one seed that names which exactness class
to declare in a spec, never a substitute for the parity test.

Two disciplines to adopt without code. Reverse the cell order on the second
pass of a paired sweep, not just the arm order: their control pack that
ignored the lever measured a 1.8 to 2.8% "gain" between two identical cells
until the order was reversed (tslab rotates arm order per round, which
covers the arm half). And when a lane is kernel-time bound, require a
replacement kernel to beat the kernel it replaces on its own before
counting the dispatches it removes.

## What this page does not cover

- Their HTTP, scheduler, and continuous batching, beyond the one fact that
  the inference thread is the sole MLX caller. Their aggregate numbers (four
  streams 71.9 to 109 tok/s on the 27B) are a serving-shape result and sit
  under `KAPITSA_KERNELS.md` lever 6.
- The MoE kernels (GLM, Flash Next, Gemma MoE). The 35B-A3B MTP cell has
  been flat at 236 to 241 tok/s since 26.9.2 on their M4 Max, so there was
  no speed story to read there.
- NAX. Every `_nax` kernel needs an M5. Their M5 Ultra 27B cell (242 tok/s)
  is a different machine class and is not a target for this host.
- Media, audio, vision, ANE chat, and the two-Mac cluster.
