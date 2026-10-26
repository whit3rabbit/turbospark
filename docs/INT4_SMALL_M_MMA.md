# Small-M matrix-unit GEMM on packed weights

How to run a GEMM for 1 to 8 token rows on Apple's `simdgroup_matrix` units
straight from packed quantized weights, so that 8 rows cost about what 1 row
costs. It is the cost curve speculative verify, tree verify, and every other
small-batch path in this engine has been missing. This page covers the
technique, what was measured, and the checklist for bringing it to another
kernel, weight format or model family.

Status: production kernel wired behind `TURBOSPARK_INT4_MMA8=1` (shared
arithmetic) and `=verify` (drift-bound, verify only), default off,
shape-routed. Lossless under the flag on the real 27B MTP install.
Workload-dependent end to end: on a code prompt, block 4 under the flag
beats the shipped default by 16%. On the protocol's prose cases it is still
9-12% behind (see "Shape routing" below). Prototype
and its variants: `crates/gpu/tests/int4_mma8_reg_bench.rs`. Production:
`crates/gpu/src/dequant_int4_mma8.rs` and
`shaders/dequant_int4_mma8.metal`. Idea from
kapitsa's `src/kernels/verify8.metal` (Apache-2.0, `docs/KAPITSA_KERNELS.md`
item 7), reimplemented here, no code copied.

## The problem it solves

Our small-M paths were never weight-bandwidth-bound. The shipped batched GEMM
(`dequant_int4_gemm_simd`) reads each weight once but does its multiply-adds
on scalar lanes, so its cost grows with the row count: 8 rows cost 3.1-3.6x a
1-row GEMV at the qwen38 shapes (c(8) of 0.41-0.56,
`docs/SPECULATIVE_DECODING.md`). That one number is why only MTP block 2
pays, and why a tree verify (KFlash) has no room to work.

The earlier matrix-unit attempt (`dequant_int4_mma.metal`) lost at every
width. It dequantized each weight into threadgroup memory so that
`simdgroup_load` could read it, and that dequant work grows with rows times K
whatever the batch.

## The technique

Three changes remove the threadgroup dequant completely.

1. **Register operands.** Each lane writes its two elements of the A and B
   8x8 operands directly through `thread_elements()` and calls
   `simdgroup_multiply_accumulate`. Nothing goes through `simdgroup_load` or
   threadgroup memory. The lane layout is MLX steel's:
   `fm = (q & 4) | ((lane >> 1) & 3)`, `fn = ((q & 2) << 1) | ((lane & 1) << 1)`,
   `q = lane >> 2`. A lane holds `[fm][fn]` and `[fm][fn + 1]`.
2. **Codes as exact halves.** A 4-bit code needs no dequant FMA. Masking a
   packed word with `0x000F000F` gives two codes, one per 16-bit half, which
   are then read as halves in one of two modes:
   - **Subnormal mode (the winner):** the bare bits are the half subnormal
     `q * 2^-24`. The epilogue scales by `s * 2^24`.
   - **Offset mode:** OR'd with `0x64006400`, each half is `1024 + q`. The
     epilogue subtracts `1024 * s * sum(x)`, which cancels precision.

   Per 64-group the epilogue is `y += s' * dot + b' * sum(x)` in fp32, with
   the group sums of each token row precomputed.
3. **Permute the activations, not the weights.** A prep pass writes the 8
   token rows into an "X8 table" in fragment order. With
   `phys(f, j) = 16 (j / 2) + 8 (f / 4) + (f % 4) + 4 (j % 2)` mapping
   k-slot `j` of fragment `f` to an input inside the group, one lane's 16
   codes for a whole group are one contiguous 8-byte run of its weight row.
   The resident layout is read in place: low nibble first, groups of 64,
   BF16 scale and bias, 2-byte aligned.

One simdgroup owns 8 output rows (one fragment, `nf = 1`) over a K slice.
`ks` simdgroups split K and are summed in fixed slice order through
threadgroup memory. `cs` row blocks share a threadgroup.

## Measured, 2026-10-21

Apple M4 Max 36 GB, macOS 26.5.2, AC power, release build, 1-minute load
2.5-6.2 from other sessions. Random codes for timing, a pool of 4 copies per
shape so no call finds its weights in the SLC, 2 GiB of weight traffic per
timed arm, 5 interleaved rounds with rotated arm order, medians. The 54
configs were swept once per shape and the best one timed. The X8-table prep
dispatch is inside every timed MMA call.

| Shape | GEMV 1 row (ms) | Shipped GEMM 8 rows | MMA 8 rows | MMA 1 row | Best config |
|-------|----------------:|--------------------:|-----------:|----------:|-------------|
| gate/up 17408x5120 | 0.133 | 3.54x | 1.25x | 1.24x | sub, nf 1, ks 4 |
| down 5120x17408 | 0.141 | 3.08x | 1.26x | 1.24x | sub, nf 1, ks 8 |
| gdn_inproj 16480x5120 | 0.126 | 3.57x | 1.25x | 1.25x | sub, nf 1, ks 4 |
| packed_q 12288x5120 | 0.096 | 3.41x | 1.27x | 1.26x | sub, nf 1, ks 4 |
| o_proj 5120x6144 | 0.052 | 3.24x | 1.29x | 1.27x | sub, nf 1, ks 2, cs 4 |

Ratios are against the 1-row GEMV. Three runs agreed: 512 MiB windows gave
1.23-1.33x, and both 2 GiB runs gave 1.25-1.29x.

What it says:

- **8 rows for 1.25-1.29x one row.** The shipped 8-row GEMM is 3.1-3.6x, so
  the MMA kernel does the same work in 0.35-0.41 of its time.
- **Rows are free.** One live row costs the same as eight (1.24-1.27x). The
  remaining 25% over the GEMV is the tile kernel streaming weights less
  efficiently than the GEMV (about 80% of its bandwidth), not a per-row
  cost. Two tuning variants did not move it (see "Open work").
- **Subnormal mode wins every shape at the 2 GiB windows.** Apple's matrix
  units keep half subnormals. They are not flushed.

### Exactness

`mma8_matches_the_reference_as_closely_as_the_shipped_gemm` compares
against an f64 reference over real quantized rows at 256x2048, 44x1024 (a
partial row block) and 512x4096, with R = 1, 3 and 8 and five configs:

- In subnormal mode the max error is identical to the shipped GEMM's on every
  case, and 99.2-100% of outputs are bit-equal to the shipped GEMM. The rest
  differ at half-rounding ties from summation order.
- Offset mode is 88-90% bit-equal, with max error up to 6% above the GEMM's.
- Mutation check: swapping `c.x` for `c.y` in the table prep fails the gate
  (max error 73 against a bound of 0.037).

`a_rows_bits_do_not_depend_on_the_rows_beside_it`: a token row's output is
bit-identical whether it runs alone or with 1, 4 or 7 other rows, across
three configs including a K split. This is the property that lets decode and
verify share one arithmetic. It was not mutation-checked.

So this kernel is NOT in the bit-exact class against the current GEMV, but it
is bit-exact against ITSELF at any row count. That decides how it can ship.

## How it can ship without breaking losslessness

`docs/SPEEDUP_PLAYBOOK.md` requires decode and verify to stay bit-exact
against each other. Its "Next work" item 1 already names the route for
attention: make decode and verify share one arithmetic. Here that means:

- Route BOTH the verify rows and the plain decode row through the MMA
  kernel, for every projection in a family that speculates. Row invariance
  makes a speculative stream equal to the plain one again.
- Pay for it once: decode numerics change, so frozen digests and benchmark
  rows for that family are re-frozen by the protocol in the playbook.
- Price the plain-decode side before switching: at one row the tile is
  1.24-1.27x the GEMV today. A family with no drafter can keep the GEMV and
  its own numerics. Closing that 25% is the first tuning task, because it
  decides whether non-speculative decode pays anything.

The alternative is a drift bound for verify (the MoE batched-prefill
policy). That is a policy change and needs an explicit decision.

## The shared-arithmetic wiring (2026-10-21)

`TURBOSPARK_INT4_MMA8=1` (read once, `real_forward_dispatch.rs`) sends
every INT4 projection reaching `encode_gemv_any` (one row) or
`encode_gemm_any` (up to 16 rows, in tiles of 8) through
`encode_dequant_int4_mma8_resident`. The INT4 bypasses step aside under it.
In this first version every bypass was off whenever the flag was on. Since
shape routing (below), each fused site asks `int4_mma8_takes(rows, cols)`
for its own matrices and steps aside only when one is served: the dense
FFN (`qwen/dense.rs`), the MoE shared expert (`qwen/moe.rs`) and the fused
GDN in-proj (`qwen` and `qwen4` `attn.rs`), which falls through to its
per-tensor arm. Steel prefill is untouched: it already rounds differently
from decode, and decode-vs-verify identity does not involve it.

Kernel choices, all functions of the shape only:

- Subnormal codes, one fragment per simdgroup, one row block per
  threadgroup.
- K split from `int4_mma8_k_split(n, k)`: 4 at 1024+ row blocks, 8 at 64+,
  16 below, capped at the group count. From `mma8_k_split_sweep` (8 rows,
  prep included, best of 3, x the 1-row GEMV, M4 Max):

  | Shape | 1-row GEMV ms | ks 1 | ks 2 | ks 4 | ks 8 | ks 16 |
  |-------|--------------:|-----:|-----:|-----:|-----:|------:|
  | gate/up 17408x5120 | 0.1417 | 1.36 | 1.22 | 1.21 | 1.22 | 1.30 |
  | down 5120x17408 | 0.1437 | 1.34 | 1.27 | 1.28 | 1.25 | 1.18 |
  | packed_q 12288x5120 | 0.1053 | 1.39 | 1.23 | 1.22 | 1.21 | 1.26 |
  | o_proj 5120x6144 | 0.0519 | 1.39 | 1.28 | 1.28 | 1.29 | 1.31 |
  | k/v 1024x5120 | 0.0079 | 3.70 | 2.20 | 1.87 | 1.78 | 1.83 |
  | in_proj_a 48x5120 | 0.0030 | 8.85 | 5.18 | 3.36 | 2.67 | 2.55 |
  | mtp_fc 5120x10240 | 0.0812 | 1.40 | 1.31 | 1.31 | 1.28 | 1.28 |
  | 32768x5120 | 0.2465 | 1.37 | 1.24 | 1.23 | 1.27 | 1.30 |

  The two small rows are why shape routing keeps matrices under 4096 rows
  on the GEMV/GEMM pair: there the prep and launch cost dominate.
- The activation table and sums live in a per-`MetalContext` scratch,
  grown to the widest K seen. Passes on one context run in order, so calls
  share it. A concurrent encoder would race on it: none is wired today.

Gates, all passing:

| Gate | Result |
|------|--------|
| `dequant_int4_mma8_parity.rs` (precompiled, `-ffast-math`) | Error vs f64 within the shipped GEMM's bound at every K-split arm; row bits identical at R = 1, 2, 3, 8. Both gates mutation-checked: a swapped table coordinate fails the first, an R-dependent K split fails the second |
| Synthetic qwen35 MTP/DFlash/chunked suites, flag off | 55/55 |
| Same suites, flag on | All decode-vs-verify identity tests pass. The two frozen-digest tests fail, as they must: they pin GEMV numerics. Mutation: leaving decode on the GEMV under the flag fails `a_batched_pass_is_bit_identical_to_the_same_tokens_run_sequentially` |
| `mtp_generation_gate`, `qwen38-27b-mtp`, flag on | "lossless at every block" at blocks 2 and 4, plus the sampled arm |
| `qwen38_quality_gate`, flag on | Perplexity 4.9419, within tolerance of the recorded row. Determinism holds. Greedy and sampled golden digests change, by design |

Other families: the flag is global, so every family's INT4 GEMV and GEMM
change together. Only qwen35 dense was gated. Do not enable it on another
family without that family's quality gate.

### End to end, `qwen38-27b-mtp`, 2026-10-21

M4 Max 36 GB, AC, release build from a HEAD worktree holding only these
changes. `turbospark-bench --model ... --case <id> --speculative off|2|4|8
--shaping greedy`, one process per run, a warmup round discarded, 3
measured rounds with rotated arm order, medians with min-max. Host load
1.2-8.3 from other sessions, highest in round 1.

| Arm | short-explanation off | on | ratio | medium-review off | on | ratio |
|-----|----:|----:|----:|----:|----:|----:|
| Speculation off | 21.16 [21.06-21.81] | 17.10 [14.49-17.38] | 0.81x | 21.89 [21.46-22.07] | 16.95 [15.52-17.25] | 0.77x |
| Block 2 | 32.21 [31.46-32.70] | 28.51 [28.18-30.40] | 0.89x | 33.82 [32.70-35.61] | 30.38 [28.66-32.53] | 0.90x |
| Block 4 | 18.07 [17.14-18.25] | 27.87 [27.71-29.24] | 1.54x | 20.99 [20.80-21.65] | 31.99 [31.59-32.72] | 1.52x |
| Block 8 | 11.93 [11.80-12.13] | 15.13 [14.90-15.59] | 1.27x | 13.78 [13.54-13.89] | 18.38 [17.86-18.72] | 1.33x |

Committed tokens per round are unchanged by the flag. Block 2 commits 2.16
and 2.13 on short and 2.30 on medium, block 4 commits 2.38 and 2.37 on short
and 2.75 on medium. The
flag-on text differs from flag-off text because decode numerics changed,
and speculation-off arms emit a different stream in both modes, as
before (`docs/BENCHMARKS.md`).

What it says:

- The verify cost curve moved as designed. Block 4 went from losing to
  plain decode (0.85-0.96x) to 1.63-1.89x over flag-on plain decode, and
  block 8 improved by 27-33%.
- It is not a net win on prose. The best flag-off arm (block 2) beats the
  best flag-on arm (block 4) by 6-16%. Acceptance falls fast on these cases
  (2.37 committed at block 4 against 2.13 at block 2), so cheaper rows buy
  little, while every decode-shaped call pays the tile's overhead.
- Plain decode lost 19-23%, more than the kernel bench's 1.22x on the large
  matrices predicts once the GEMV-bound share is counted. The engine adds
  three costs the bench did not have: one prep dispatch per projection, the
  small matrices (k/v 1.8x, in_proj_a/b 2.6x the GEMV), and the lost FFN and
  GDN in-proj fusions.
- Not measured: a code-heavy case, where the replay in
  `docs/KAPITSA_KERNELS.md` showed 5.0 committed per round at block 8. That
  is where this route should pay most.

### Shape routing (2026-10-22)

The tile now serves only projections with at least 4096 output rows
(`int4_mma8_takes` in `real_forward_dispatch.rs`, a function of the shape
only). Smaller INT4 matrices keep the GEMV for decode and the shipped
batched GEMM for verify. That pair is already bit-identical, so decode and
verify still agree. The fused FFN and fused GDN in-proj step aside only when
one of their own matrices is served, so they come back for small shapes
(the MoE shared expert, for one). `TURBOSPARK_INT4_MMA8_MIN_ROWS` overrides
the threshold, read once, for tests.

Gates, all passing:

- Synthetic qwen35 suites under four settings: flag off (55/55), flag on
  with every matrix served (threshold 1), flag on with a mix (threshold 200:
  packed q, GDN qkv and the head served, everything else not), and flag on
  at the default. Every decode-vs-verify identity test passes in all four.
  The frozen digests fail wherever a matrix is served. At the default that
  is only the DFlash fixture, whose 248,100-row head is served.
- `mtp_generation_gate`, flag on: lossless at every block, six runs.
- `qwen38_quality_gate`, flag on: perplexity 4.9429 (4.9419 before shape
  routing), within tolerance. Golden digests change, by design.

Protocol A/B, same method as above, quieter host (load 0.7-1.6), tok/s
medians with min-max:

| Arm | short-explanation off | on | ratio | medium-review off | on | ratio |
|-----|----:|----:|----:|----:|----:|----:|
| Speculation off | 21.82 [21.82-21.84] | 17.62 [17.42-17.72] | 0.81x | 21.75 [21.75-21.77] | 17.56 [17.50-17.70] | 0.81x |
| Block 2 | 33.56 [33.31-33.68] | 30.66 [30.35-31.34] | 0.91x | 35.21 [35.15-35.42] | 32.01 [31.78-32.87] | 0.91x |
| Block 4 | 18.75 [18.73-19.23] | 29.39 [29.38-29.87] | 1.57x | 21.67 [21.66-22.08] | 31.96 [31.84-32.04] | 1.47x |
| Block 8 | 12.24 [12.20-12.38] | 16.20 [16.20-16.57] | 1.32x | 14.13 [14.12-14.29] | 18.40 [18.38-18.76] | 1.30x |

Code case: `mtp_generation_gate`'s code prompt, 96 greedy tokens, two runs
per flag alternating, through the wired speculative loop. Lossless in all
four runs.

| Flag | Sequential | Block 2 | Block 4 |
|------|-----------:|--------:|--------:|
| Off | 22.39, 22.34 | 43.58, 43.07 | 34.38, 34.17 |
| On | 18.28, 18.28 | 39.92, 39.97 | 50.08, 50.11 |

What it says:

- **It is workload-dependent, and on code it wins.** Code has high
  acceptance, so block 4 under the flag (50.1 tok/s) beats the shipped
  default, block 2 without it (43.3), by 16%. On prose the best flag-on arm
  stays 9-12% behind the default (29.4 against 33.6, 32.0 against 35.2).
- Shape routing helped a little: plain decode went from 0.77-0.81x to
  0.81x, block 2 from 0.89-0.90x to 0.91x. The remaining loss is the large
  matrices' 1.22x one-row cost, which shape routing cannot touch.
- The 96-token code runs are short. Treat 16% as a direction, not a frozen
  number.

One-prep-per-input was not built: the prep is 2.6-3.6 us and the served
projections share inputs in only two places per layer, so the saving is
under 1% of a token.

### What would make it pay everywhere

1. **Pick the block per request, or per round.** Under the flag the best
   block depends on acceptance: 4 for code, 2 for prose. The adaptive depth
   controller (`speculative_adaptive.rs`, `docs/KAPITSA_KERNELS.md` row 13)
   is the mechanism. That would take the code gain without the prose loss,
   apart from the decode and block-2 penalty.
2. **The one-row gap does not close with this design (measured
   2026-10-22).** `mma8_ablation` in the bench times the base kernel with
   one cost removed at a time, at R = 1, 8-aligned weights for every arm:

   | Variant (x the 1-row GEMV) | gate/up | down | packed_q | o_proj |
   |---|---:|---:|---:|---:|
   | Base | 1.21 | 1.24 | 1.21 | 1.23 |
   | No MMAs (FMAs instead) | 1.17 | 1.18 | 1.19 | 1.13 |
   | Weights from cache | 0.99 | 1.01 | 1.01 | 1.02 |
   | Table from cache | 1.20 | 1.23 | 1.21 | 1.25 |
   | Compute only (both cached) | 0.96 | 0.96 | 0.98 | 0.99 |
   | Persistent accumulator | 1.21 | 1.24 | 1.22 | 1.24 |
   | One 8-byte load per lane | 1.22 | 1.24 | 1.24 | 1.24 |

   Every pipeline reports 1024 threads per threadgroup, so occupancy is
   not register-limited. The compute alone already costs what the whole
   GEMV does: an 8x8x8 MMA does 512 multiply-adds, of which 64 are useful
   at one row, and its operand setup is paid per call. Perfect overlap
   with the weight stream would land near 1.0x at best, and none of five
   variants (two loop shapes, persistent accumulator, wide loads, cached
   table) improved overlap. Kapitsa's own one-row tile is about 10% slower
   than its decode kernel. So under shared arithmetic, plain decode pays
   about 20% on the large matrices, and that is a property of the
   approach, not of this implementation.
3. A larger code-heavy corpus before any default changes.

### The drift-bound arm (2026-10-22)

`TURBOSPARK_INT4_MMA8=verify` sends only batched calls (verify, and
prefill batches under 32 rows) to the tile, for the same 4096+ row
shapes. One-row decode and the fused decode kernels keep GEMV arithmetic,
so verify rows differ from decode in the last bits by design.

**Synthetic suites, every matrix served:** 48 of 55 pass. The 7 failures
are all bit-equality tests (batched passes against sequential decode,
including chunked prefill and head priming) plus one frozen digest. That is
the expected signature of this arm, and nothing else fails.

**Drift, real 27B** (`int4_mma8_drift_probe.rs`): 256 greedy tokens per
prompt are decoded one at a time, then replayed through batched passes of 3
and 5 rows. Each pass is committed in full, so drift compounds through the
state as it would in real speculation. KL(decode || verify) in nats per
row, mean and max; top-1 agreement. Pass widths 3 and 5 give identical
numbers, as row-independent kernels should.

| Prompt | Flag off: mean / max KL, top-1 | Verify arm: mean / max KL, top-1 |
|--------|------------------------|--------------------------|
| short-explanation | 8.5e-6 / 1.1e-4, 99.61% | 8.2e-6 / 6.9e-5, 98.82% |
| medium-review | 6.8e-6 / 9.7e-5, 100% | 5.8e-6 / 4.8e-5, 100% |
| code | 3.0e-7 / 1.2e-5, 100% | 7.4e-7 / 3.8e-5, 100% |

**The flag-off column was a bug, since fixed.** On the real install the
shipped batched path was not bit-identical to decode: 0 of 255 rows matched
in either arm. A later bisect found why (2026-10-23, `docs/DFLASH2.md`,
losslessness caveat): the GDN conv's decode and prefill kernels summed
their taps in different orders. With that fixed, the flag-off arm reads KL
0 and 255 of 255 rows bit-equal on all three prompts. The verify-arm
column above predates the fix and was not re-run. Every value is far below
the MoE batched-prefill bound (1e-2 nats mean KL).

**Stream identity, wired loop, 256 greedy tokens** (tokens matching the
sequential stream before the first departure):

| Prompt | Flag off, block 2 / 4 | Verify arm, block 2 / 4 |
|--------|----------------------:|------------------------:|
| short-explanation | 166 / 166 | 198 / 198 |
| medium-review | 256 / 256 | 256 / 256 |
| code | 256 / 256 | 256 / 256 |

With the flag off, greedy speculation left the sequential stream at token
166 on short-explanation. Two causes, both now explained
(`docs/DFLASH2.md`): the GDN conv summation order (fixed), and the
speculative loop's chunked prefill, which is not bit-exact by design. With
the conv fixed and per-token prefill on both arms, all three prompts match
for 256 tokens at blocks 2 and 4.

**Speed, protocol prose A/B** (flag-off block 2 is today's default;
3 measured rounds after a warmup, rotated, load up to 3.9, medians with
min-max). Several verify runs hit a stall (one at 0.71 tok/s) from host
contention, so the minima are not meaningful. Read the medians.

| Arm | short-explanation | medium-review |
|-----|------------------:|--------------:|
| Default (flag off, block 2) | 34.59 [34.38-35.20] | 37.13 [37.12-37.15] |
| Verify arm, block 2 | 33.20 [29.04-33.20] | 34.83 [24.36-34.83] |
| Verify arm, block 4 | 31.76 [0.71-31.80] | 34.29 [28.68-34.57] |
| Verify arm, block 8 | 17.93 [17.20-17.95] | 20.17 [19.89-20.20] |

Code prompt, from the probe's wired loop (one run each, different
processes): default block 2 40.26 tok/s, verify arm block 2 35.08 and
block 4 46.27. The verify-arm probe also read a lower sequential speed
(17.7 against 20.3-20.7) although its decode path is unchanged, so that run
was likely on a busier host and its numbers read low.

What it says:

- On prose the drift-bound arm is 4-6% behind the default at its best
  (block 2). Shared arithmetic was 9-12% behind.
- On code, block 4 under it is about 15% ahead of the default, the same
  gain shared arithmetic gave.
- Measured before the conv fix, its drift was indistinguishable from what
  shipped speculation then had. Shipped verify is now bit-exact, so this
  arm would ADD drift of about 1e-5 nats mean KL where there is none, and
  that weakens its case against shared arithmetic.
- Neither arm beats the default on prose, so neither is a default. The
  choice between them is policy: shared arithmetic keeps decode and verify
  bit-exact against each other but slows plain decode by about 20%. The
  drift-bound arm leaves decode untouched and matches today's level of
  drift.

Known break, unrelated to this flag: `mtp_accept_length_probe`'s batched
arm diverges from its reference within a few tokens with the flag OFF too.
It drives `checkpoint`, a batched pass and `rollback` by hand without the
lazy-tape commit the speculative loop now does. Its batched numbers are void
until it is updated.

## Where it applies

Any path that multiplies 2 to 8 rows by a packed weight, in rough order of
value:

1. **Speculative verify** (MTP and DFlash2, Qwen dense). The block sweep in
   `docs/KAPITSA_KERNELS.md` was limited by verify cost. Re-run it with this
   kernel before deciding the serving block.
2. **Tree verify (KFlash).** The replay there found a tree only pays at
   blocks above 2. With rows this cheap, those blocks may pay.
3. **The DFlash2 drafter's own block forward** (8 rows per round).
4. **Small prefill chunks** below Steel's 32-row floor and chunk tails.
5. **Multi-request batched decode** (one row per request).
6. **MoE.** The shared expert takes it as-is. Routed experts need rows
   grouped per expert first, and kapitsa measured 1.56x for 8 rows there
   because extra rows bring in extra experts.

## Applying it to another weight format

The kernel needs a code to become an exact half with integer ops, and an
epilogue that is affine per group:

| Format | Fit | Notes |
|--------|-----|-------|
| INT4 affine, group 64 (ours, MLX) | Measured | This page |
| INT8 / other affine bit widths | Direct | Codes up to 10 bits fit a half subnormal. 8-bit codes take one byte per slot, and kapitsa instantiates a `b8` variant |
| Affine with group 32 or 128 | Direct | 4 or 16 fragments per group instead of 8; adjust `phys` and the table stride |
| Symmetric (no bias) | Direct | Drop the `b * sum(x)` term |
| GGUF k-quants (Q4_K, Q5_K, Q6_K) | Possible, unmeasured | Sub-blocks of 32 with 6-bit scales and mins: per-sub-block epilogue and a scale decode |
| Lookup quants (IQ*) | No | Codes are not linear in the value |
| BF16 / FP16 weights | Different problem | No dequant to remove; kapitsa reports bf16 verify is memory-bound |

## Checklist for bringing it to a new kernel or family

1. **Layout first.** Confirm group size, nibble or byte order, scale and
   bias dtypes, and row alignment against the resident format. The table
   permutation absorbs packing order, but only if one lane's codes for a
   group are contiguous.
2. **Start from the bench shader.** Port `x8_table` and `mma8_impl` from
   `int4_mma8_reg_bench.rs`. In a production shader, bake `N` and `K` as
   function constants and put every constant (`nf`, `ks`, `cs`, mode) in the
   pipeline cache key. A missing constant silently reuses the wrong
   pipeline.
3. **Correctness gate.** Copy the f64-reference test: remainder row blocks,
   R = 1 to 8, every config the dispatcher can pick, error no worse than the
   kernel being replaced.
4. **Row-invariance gate.** Copy the R-invariance test. Without it, decode
   and verify cannot share arithmetic.
5. **Mutation-check** the layout (swap a coordinate in the prep) and confirm
   the gate fails before trusting a pass.
6. **Ratio bench.** AC power, a weight pool that defeats the SLC, 2 GiB or
   more per arm, interleaved rounds, against both the 1-row GEMV and the
   kernel being replaced. Report the one-row arm too.
7. **Wire behind an opt-in flag**, read once, with the old path one flag
   away.
8. **Model gates.** Re-freeze digests by the playbook protocol. Then
   `mtp_generation_gate` lossless at several blocks, the family's quality
   gate, the memory oracle, and an interleaved end-to-end A/B on AC power.

## Runbook: files, flags and how to run them

Code and tests from this work. Real-install runs need
`TURBOSPARK_MTP_INSTALL_DIR=~/models/qwen38-27b-mtp.gturbo` (or the
variable named in the file header). Run one 27B process at a time: two at
once, or one beside a heavy build, has stalled this 36 GB machine.

| What | Where | Run |
|------|-------|-----|
| Production kernel | `crates/gpu/src/dequant_int4_mma8.rs`, `shaders/dequant_int4_mma8.metal` | via the flags below |
| Kernel gates (accuracy, row invariance, refusals) | `crates/gpu/tests/dequant_int4_mma8_parity.rs` | `cargo test -p turbospark-gpu --release --test dequant_int4_mma8_parity` |
| Prototype, variants, ratio bench, K-split sweep, ablation | `crates/gpu/tests/int4_mma8_reg_bench.rs` | add `-- --ignored --nocapture` for the benches |
| Decode-vs-verify kernel parity at real shapes (GEMV vs GEMM at 3 and 5 rows, fused FFN, fused GDN in-proj) | `crates/gpu/tests/qwen38_decode_verify_parity.rs` | `cargo test -p turbospark-gpu --release --test qwen38_decode_verify_parity` |
| GDN conv summation-order regression (committed, `7b37fd36`) | `crates/gpu/tests/gdn_conv_order_parity.rs` | `cargo test -p turbospark-gpu --release --test gdn_conv_order_parity` |
| Runtime routing | `crates/runtime/src/real_forward_dispatch.rs` (`int4_mma8_mode`, `int4_mma8_takes`), fused-site guards in `qwen/dense.rs`, `qwen/moe.rs`, `qwen/attn.rs`, `qwen4/attn.rs` | `TURBOSPARK_INT4_MMA8=1` (shared) or `=verify` (drift-bound); `TURBOSPARK_INT4_MMA8_MIN_ROWS` for tests |
| Drift and wired-loop probe (teacher-forced KL, top-1, bit-equal rows; stream identity and tok/s) | `crates/bench/tests/int4_mma8_drift_probe.rs` | `TURBOSPARK_INT4_MMA8=0\|1\|verify ... --test int4_mma8_drift_probe --release -- --ignored --nocapture`; add `TURBOSPARK_BATCHED_GEMV=0` to give both arms the same prefill |
| Lossless gate and quality gate | `crates/bench/tests/mtp_generation_gate.rs`, `qwen38_quality_gate.rs` | as in their headers, with the flag set |
| Protocol A/B | `turbospark-bench --model <install> --case <id> --speculative off\|2\|4\|8 --shaping greedy` | one process per run, warmup discarded, arms rotated, load checked |

The bisect that found the GDN conv bug used temporary hooks that were not
kept. The recipe is in `.claude/docs/diagnostics.md`
("Decode-vs-verify bisect").

## Open questions

Recorded so nobody has to reconstruct them. Each names what would answer it.

**Policy (yours to decide)**

1. **Should the speculative loop prefill the way sequential decode does?**
   After the conv fix, the only thing that still parts the default greedy
   speculative stream from a sequential one is prefill: the speculative
   loop primes the drafter through the chunked driver (Steel at 32+ rows,
   not bit-exact). `TURBOSPARK_BATCHED_GEMV=0` makes both arms match for 256
   tokens on three prompts, at a prefill-speed cost that was not measured.
   Options: accept the documented divergence, make the speculative loop's
   prefill bit-exact (or match the sequential path's choice), or offer a
   reproducible mode.
2. **Which INT4 MMA8 arm, if either, ever becomes a default?** Shared
   arithmetic is bit-exact but slows plain decode about 20% on the large
   matrices. The drift-bound arm leaves decode alone but, now that shipped
   verify is exact, adds about 1e-5 nats of drift where there is none.
   Neither beats the default on prose. Both win about 15% on a code prompt.

**Measurements not yet run**

3. The drift-bound arm's drift and speed after the conv fix (its table
   above predates it).
4. Real-model gates for the conv change beyond qwen38 decode: chunked
   prefill quality, and qwen4, whose dilated PLE conv uses the same prefill
   kernel. Only the synthetic and gpu suites covered them.
5. A larger code-heavy corpus for the INT4 MMA8 A/B. The +15-16% code
   result is two 96-token runs per arm.
6. The full-vocab head (248320x5120) through the MMA8 kernel was not timed
   (about 715 MB per pool copy). Its GEMV-vs-GEMM parity was checked.
7. Other chips. Only the M4 Max was measured.
8. Other families under the flag. It is global and only qwen35 dense was
   gated.

**Engineering not done**

9. **Adaptive block under the flag.** Bounded by data already in hand:
   under the flag blocks 2 and 4 are within 4% on prose, so a per-round
   chooser lands near 0.91-0.93x of the default there. It would need a bench
   harness (the bench binary has no adaptive arm) and retuned climb
   thresholds, which were set for expensive rows.
10. **The one-row gap is structural** (`mma8_ablation`, "What would make it
    pay everywhere"). Five variants did not move it. Reopen only with a
    different design, not another loop variant.

    **The different design was tried, 2026-10-25: a two-pass split-K. It
    does not pay.** Pass 1 is the shipped K loop with the barrier and the
    slice-0 reduction tail removed; each simdgroup stores its `float2` to a
    device partials buffer, so KS is no longer capped at 16. Pass 2 sums the
    slices from zero in ascending order, the shipped tail's order, so at the
    shipped KS the output is bit-identical to the production tile
    (`two_pass_split_k_is_bit_equal_to_the_shipped_tile`; a reversed combine
    fails it). The fragments are already k-slices of one 8x8 tile, so the
    split costs nothing in exactness, only the partials round trip and a
    second dispatch.

    `int4_mma8_reg_bench.rs`'s `two_pass_split_k_sweep`, M4 Max on AC, two
    runs at load 3.5-4.0 (above the 2.5 bar, so read it as directional).
    Two-pass time, prep and both passes included, over the shipped tile at
    the same R:

    | shape | R | KS 4 | KS 8 | KS 16 | KS 32 |
    |---|---:|---:|---:|---:|---:|
    | gate/up 17408x5120 | 8 | 1.042-1.046 | 1.016 | 1.060-1.064 | 1.273-1.285 |
    | down 5120x17408 | 8 | 1.007-1.027 | 0.993-1.006 | 0.949-0.980 | 0.949-0.985 |
    | gdn_inproj 16480x5120 | 8 | 1.039-1.040 | 1.035-1.053 | 1.057-1.072 | 1.254-1.258 |
    | packed_q 12288x5120 | 8 | 1.030-1.042 | 1.038-1.052 | 1.055-1.059 | 1.187-1.200 |
    | o_proj 5120x6144 | 8 | 1.030-1.042 | 1.035-1.047 | 1.050-1.056 | 1.025-1.071 |

    R=1 reads the same within 2%. Only `down` (the longest K, 272 groups
    over 640 row blocks) gains, 1.5-5.1% at KS 16-32, which is more K
    parallelism and not the tail; every other shape loses at every KS.
    That fails the promotion rule (5% at R=8 on the large shapes), so it is
    not wired. Consistent with the ablation's "weights from cache 0.99-1.02x"
    row: the tile is bound by streaming the weights, and the tail it removes
    was never the floor. This closes the question again with the new data
    point; a per-shape KS for `down` alone would buy about 1% of a verify
    round and is not worth a second kernel path.
11. **One prep per input** (q/k/v, gate/up): under 1% of a token by the
    measured prep cost, not built.
12. **16 rows as two tiles** from one weight load (kapitsa's `qmm8t2`).
13. **KFlash tree verify on DFlash2.** The MTP replay declined it, but a
    DFlash2 lattice is not capped at one extra token per round. Needs a
    DFlash2 install and a lattice dump (`docs/KAPITSA_KERNELS.md`).

**Known breaks and stale text found along the way**

14. `mtp_accept_length_probe`'s batched arm diverges with every flag at its
    default: it rolls back by hand without the lazy-tape commit. Its batched
    numbers are void until it follows `run_speculative`'s round contract.
15. Resolved: `real_forward_init::tests::mapped_residency_is_served_on_gemma4_and_refused_by_name_everywhere_else`
    failed on 2026-10-22 (a family-count assertion, 15 against 16, after
    the Flint family landed) and passes on `main` at `2666539a`.
