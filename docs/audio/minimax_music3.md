# MiniMax Music 3: speed, MLX parity, and what not to redo

This is the engineering memory for `crates/audio/src/music/minimax_music3/` and
its Metal kernels. The family README owns the commands and the per-change
measurements; this page owns the conclusions, the method, and the mistakes, so
the next person does not rediscover them. Written 2026-10-17 on an Apple M4 Max
(`applegpu_g16s`) with the pinned 4-bit snapshot
`mlx-community/MiniMax-Music3-4bit` at revision `c7ea32923b24`.

Run `scripts/verify-music3.sh` before and after touching a Music 3 kernel. It
fails if an "exact" change moved a bit. `--mlx MODEL_DIR` adds the diff against
MLX itself, which is the only check that covers dense and convolution kernels.

## Status

| Question | Answer |
|---|---|
| Is the pinned 4-bit profile bit-identical to MLX 0.32.3? | **Yes, end to end**: hiddens, logits, sampled codes, flow noise, DiT, vocoder and the waveform. Verified for 3 frames at 2 flow steps and for 25 frames at 4 flow steps (all 88,064 samples equal). |
| Does that hold at product scale? | Yes. 8 s, 200 frames, 30 flow steps: our 16-bit WAV equals MLX's output in 0 of 705,536 samples. A 210-frame request (two chunks, 2 flow steps) matches in all 740,352 samples, with identical codes and hiddens. |
| Which kernels are diffed against MLX bit for bit? | Dense bf16 (depth projection and heads, rows 1 to 16), 4-bit affine (depth layers rows 1 to 16, the DiT's `proj_in` and `proj_out` rows 16 to 691), and convolutions (condition encoder, vocoder input, residual, 1x1 and transposed convs). 0 differing values. |
| Are the other profiles (6-bit, 8-bit, MXFP8, MXFP4, NVFP4) verified exact? | **No.** Only the affine 4-bit profile has been compared. The kernels they use are shared, but nothing has been checked. |
| What does the app, server, FFI and CLI run by default? | `FlowProfile::Fast`, which is not MLX's trajectory by design (see below). |
| Has anyone listened to the fast profile? | No. Every quality number here is a drift metric on one prompt and one seed. |

## Two flow profiles

`Music3Runner` opens in `FlowProfile::Fast`; `Model` (the library) opens in
`Exact`. A runner can be switched with `set_flow_profile`, and the CLI takes
`--flow-profile fast|exact`. Everything that compares with MLX or a CPU
reference must use `Exact`.

| | Exact | Fast |
|---|---|---|
| Solver | Euler | Adams-Bashforth 2 |
| Unconditional guidance branch | every step | reused, interval 3 after warmup 2 |
| Steps when a request names none | 30 | 21 |
| DiT forwards for one chunk | 60 | 30 |
| Matches MLX | yes (4-bit verified) | no, by design |

Measured on one 8 s prompt, scored against the exact 30-step audio
(`ab_music3_flow.py`, `spectral_distance.py`, single runs, taken before the
convolution and dense-dispatch fixes below, which move the exact reference by
only a few bf16 steps):

| Setting | DiT forwards | Correlation | Log-band L1 |
|---|---|---|---|
| Euler 30 (reference) | 60 | 1.0000 | 0 |
| Euler 30, reuse 3/2 | 42 | 0.9947 | 1.13 dB |
| Euler 21 | 42 | 0.9548 | 1.48 dB |
| AB2 21 | 42 | 0.9730 | 1.33 dB |
| **AB2 21, reuse 3/2 (Fast)** | **30** | **0.9742** | **1.30 dB** |
| Euler 21, reuse 3/2 | 30 | 0.9605 | 1.50 dB |
| AB2 15 | 30 | 0.7831 | 2.01 dB |

Fifteen AB2 steps do **not** match 30 Euler steps here, whatever the CUDA port
audio.cpp-lowend-gpu reports. A spectral metric can reject a change but never
approve one: that project found a flow chunk hop of 150 passed every spectral
check and was audibly broken, so anything touching the trajectory needs ears on
a full song.

## Speed: what the time is and what did not help

The kernel work, all exact (hash oracles and WAV hashes unchanged at the time):

- `music3_linear_tiled` was **decode-bound, not MMA-bound**. Ablating the 691-row
  DiT projections gave decode about 44%, activation staging 16%, matrix multiply
  15%. Hoisting the per-word scale, offset and index out of the per-nibble loop
  and staging activations four wide cut the 2048x16384 projection from 18.3 to
  10.3 ms. After that, decode is about 2.7 ms and the MMAs about 3.4 ms (near
  the f32 ALU rate) of 10.7, with about 4.5 ms of staging, barriers and epilogue.
- End to end on the 8 s clip (medians of 3, interleaved, byte-identical WAVs,
  before the MLX-dispatch fixes): 30 steps 124.6 s to 85.6 s, DiT stage 95.3 s to
  59.3 s; 1 step 36.9 s to 33.1 s.
- Not worth repeating: taller or wider tiles (64x32, 32x64, 64x64: 64x64 was 2.3x
  slower), remapping lanes so neighbours read contiguous words, aliasing the
  result tile onto the input tiles (1.5%, noise), and a bf16-operand MMA (the
  MMA is not the limit). The Steel INT4 kernel used for Qwen prefill rounds
  weights to FP16 and cannot be exact.
- The depth decoder (hidden 4096, 4 layers, about 12 to 15 s per 8 s clip) is a
  per-row GEMV below 13 rows. Its cost grows with rows because weights are
  re-read per row, and the kernel choice by row count is part of the MLX contract.
- audio.cpp's remaining idea is AR guidance truncation (conditional-only after
  frame 50, halving the AR rows). Not implemented: it changes the AR trajectory.

Timing after the MLX-dispatch fixes: see the README's Metal performance section.

## Real MLX is faster than we are

Measured 2026-10-17 on the same M4 Max, 8 s clip, seed 7, the 4-bit profile, MLX
0.32.3 (Python, `mlx-audio` `feb25a37`, its own `generate` pipeline) against our
native Metal path. AC power; MLX numbers are one run per row after a warmup, ours
are the AC runs from the README ("Exactness cost"). MLX is deterministic: two
1-step runs hashed identically.

| | MLX total | MLX AR | MLX flow incl. vocoder | Ours exact | Ours Fast (21 steps) |
|---|---|---|---|---|---|
| 1 step | 11.0 s, 11.7 s | 9.3 s, 9.9 s | 1.7 s, 1.9 s | 29.0 s | not measured |
| 30 steps | **36.6 s** | 10.9 s | 25.8 s | 85.6 s | about 63 s (busy host) |

So MLX is about 2.4x faster than our exact path and about 1.7x faster than our
Fast profile, for the same audio (bit-identical at 30 steps). Where the gap is,
by stage (ours from the 30-step run, mean): AR about 20 s against MLX's 11 s
(depth 13 s, LM decode 6 s); the DiT about 59 s against about 24 s; and the
vocoder about 6.7 s against roughly 1 s (MLX's flow stage at 1 step, vocoder
included, is only 1.7 to 1.9 s). The vocoder is the largest ratio (about 6x) and
the DiT the largest absolute gap.

**Vocoder as one submission per block.** Each vocoder block (transposed conv plus
three residual units) now records into a single command buffer
(`Sequence::convolution` and `Sequence::snake`; bf16 device weights only, the
per-op path otherwise and under a trace observer). Values are bit-identical to
the per-op path (`vocoder_block_sequences_match_the_per_op_decode_bitwise`, all
seven tiny profiles) and `verify-music3.sh --mlx` stays green. Vocoder time at
8 s drops from about 6.7 s to about 4.0 s (`--flow-only`, 1 step, flow 6.05 s
with the DiT about 2.0 s, load average 8 from another session's build, so
indicative). 

**Parallel split-K does not apply to the vocoder.** It was the first guess for
the remaining 4 s, and per-conv timing refuted it: MLX splits K only for small
outputs, and at 8 s every transposed conv is far above that threshold
(`conv_split` is 1). The real cost was that the tiled kernel walked all
`kernel` taps of a stride-`s` transposed conv although only `kernel / s` touch
input; the rest are blocks of exact zero products. `music3_convt_phase` tiles
over output phase and runs only the taps a phase can see (same ascending
order, `ic % 8 == 0` so each 8-wide MMA block sits inside one tap, and an
all-zero block leaves the accumulator unchanged), so the bits are identical.
Per-half transposed convs: 189, 371, 204 and 73 ms become 21, 40, 76 and 60 ms;
the vocoder drops to about 2.6 s (busy host, indicative). Guards: the conv
golden gained long transposed cases recorded with the old kernel, and the MLX
harness diffs all four blocks' transposed convs at lengths up to 300, 0 differing.
Remaining time is the k7 residual convs (43 to 108 ms each at the large sizes),
the k1 convs, Snake, and the phase kernel's strided output stores.

This does not make MLX the right default for TurboSpark: it would put a Python and
MLX runtime inside a native app. It does mean there is real headroom in our
kernels, and MLX is now a trustworthy oracle for finding it, because the outputs
are identical.

## The MLX gaps, in the order they were found

The ignored test `real_music3_pinned_4bit_native_matches_mlx_and_resets` failed at
3.82 against its 3.0 tolerance, and the end-to-end waveform correlated only 0.47
to 0.68 with MLX. Each explanation along the way was wrong until a bit-level
comparison replaced the theory. **Every one of these looked like "noise" until it
was isolated.**

1. **Wrong: "a kernel threshold mismatch."** MLX's qmv batch limits for this chip
   (13, 15, 13) match `linear_plan`.
2. **Wrong: "sampling noise, so exactness is impossible."** The old test compared
   codebooks 4 to 7 after the sampler had already picked a different code at
   codebook 3 (ours 585, MLX 594). Concluding "inexact by nature" was the mistake:
   **one differing bit amplified by a top-k draw looks exactly like noise.**
3. **Dense matmul dispatch (`cc9344ee`).** `depth.projection` and
   `depth.audio_heads` are dense bf16, and MLX applies each to both CFG rows in
   one matmul. We ran two single-row calls. Fixed with `music3_gemv_wide`, dense
   split-K, and batching.
4. **Convolution K order.** MLX's implicit-GEMM conv loader visits the taps inside
   each 16-channel chunk, so K runs chunk-major, not tap-major. Our conv summed
   tap-major, which differed in about 8 of 10,000 values.
5. **Transposed convolution.** MLX runs it as an unfold plus a split-K matmul,
   sums the taps in ascending weight order, and sees both stereo halves as one
   batch (so `M = 2 x output positions`, which changes the partition count).
6. **Quantized split-K reduction order.** The partials are summed by a generic
   column reduce whose order changes with the partition count.

After 3 to 6 the first differing tensor disappears: against the natural MLX
reference the whole pipeline is identical, with MLX's own noise.

### MLX dense `x @ W^T` dispatch (M3 and later; `matmul.cpp`, `gemv.h` at `v0.32.3`)

| Rows (M) | Kernel | Notes |
|---|---|---|
| 1 | gemv | already exact in ours |
| 2 to 15 | `gemv_wide` | needs K % 4 == 0, bf16 or f16, N > 1. float4 `dot` chunks, unroll 8, lane block sums folded in order, halving `simd_shuffle_down`. 32 lanes per row when `ceil(M/5) == 1` or N <= 64, else 16. |
| 16 and up, small M x N | split-K GEMM | when `ceil(M/16) * ceil(N/16) <= 2048` (Max and Ultra, else 1024), `K/16 >= 8`, `K >= max(M,N)`. Partitions = `clamp(next_pow2((K/16) / (ceil(M/32) * ceil(N/32))), 2, 32)`. Float32 partials summed in partition order, rounded once. |
| otherwise | steel GEMM, single K chain | |

### MLX convolution dispatch (`conv.cpp`)

| Convolution | MLX path | What we match |
|---|---|---|
| groups 1, non-transposed, C and O 16-aligned (or small) | implicit-GEMM kernel | tap-inside-16-channel-chunk K order (`p[13]` in `music3_conv_tiled`) |
| transposed (input-dilated) | unfold, then `steel_matmul(M = batch x outlen, N = oc, K = taps x ic)` | the dense split-K rule above, ascending tap order, and a `batch` hint (the vocoder passes 2) through `DeviceWeight::convolution_batched` |

### MLX quantized matmul dispatch (`quantized.cpp`)

For `M >= vector_limit` with transposed weights and one batch, `qmm_splitk` runs
`split_k = max(1, 512 / (ceil(M/32) * ceil(N/32)))` partitions (capped by
`K / max(group_size, 32)` and reduced until it divides K), each partial written
in the activation dtype. The partials are summed by a column reduce:

| Partitions R | Order |
|---|---|
| 2 to 8 | one row per thread, ordered fold: the plain sequential bf16 sum |
| 9 to 16 | `min(8, R)` threads, thread `t` sums rows `t, t+8, ...` in bf16, then the thread totals are folded in order |
| 32 | float sum of all partials (`simd_sum`), rounded once |
| other | not modelled (falls back to the sequential sum) |

Implemented in `music3_linear_tiled` (the `lanes` accumulators) and
`crates/gpu/src/music3.rs` (`mlx_splitk_partitions`, `linear_plan`).

## How to find an MLX mismatch (the method)

1. **Compare one operation, not a pipeline.** `crates/audio/tools/
   gen_music3_mlx_linear_cases.py` runs the MLX operation on the real checkpoint
   weights at the row counts and lengths that matter and dumps inputs and
   outputs; `crates/gpu/examples/music3_mlx_linear_cmp.rs --require-exact` diffs
   our kernel bit for bit. It prints where the first few misses sit.
2. **Teacher-force.** `probe_minimax_music3_reference.py --force-warmup-codes
   '[...]'` replaces MLX's draws with ours so every depth step sees the same input.
3. **Walk the trace to the first difference, and count differing values, not the
   maximum.** `--trace --trace-max-calls N` on MLX versus
   `TURBOSPARK_MUSIC3_TRACE=1` on `music3_backend_probe`. A stage that matches
   proves everything upstream of it is exact; the first mismatch names the
   operation.
4. **Read the pinned MLX source** for the dispatch rule before guessing. The
   heuristics (`get_qmv_batch_limit`, `gemv_wide_config`, the split-K triggers,
   the conv dispatcher, `strided_reduce_general_dispatch`) are device-generation
   dependent and live in `matmul.cpp`, `quantized.cpp`, `conv.cpp`, `reduce.cpp`.
5. **A mismatch rate tells you what kind of difference it is.** About 1 in 10,000
   is a different fp32 summation order. A few percent is a different rounding
   point. Near 100% is a different algorithm. A mismatch that only appears at some
   lengths is a heuristic that depends on the shape (partition counts, batch).

## Pitfalls checklist

- **`Music3Runner` opens in the Fast profile.** Any parity probe or test that
  compares with MLX or a CPU reference must call
  `set_flow_profile(FlowProfile::Exact)`. `music3_bench` and `ab_music3_flow.py`
  default to `exact` for that reason; pass `--flow-profile fast` to measure the
  default.
- **A loose tolerance hides a real bug.** The 4-bit gate sat at 3.0 on a 1e-3
  quantity and later at "while the draws match". It is now zero tolerance for
  `affine4`: hiddens, logits, codes and the waveform must be bit-identical.
  Treat a tolerance that needed raising as a finding, not a fix.
- **MLX heuristics see the batch.** The vocoder's two stereo halves are one MLX
  call, so its split-K rule sees twice the rows. Run per half without the hint and
  the partition count, and so the rounding, differs.
- **The hash oracles cover affine linear kernels only.** `hashes` and `dit_hashes`
  are checked against golden files by `crates/gpu/tests/music3_oracle.rs`. Dense
  and convolution kernels are covered only by the MLX diff, so changing one
  without running `scripts/verify-music3.sh --mlx` is blind. The goldens
  changed once on purpose (221 `hashes` lines) when the quantized split-K
  reduction began following MLX for more than 8 partitions; the 192 real-shape
  lines did not move.
- **Output bits depend on the chip.** The kernels reproduce MLX's choices, and
  several depend on the GPU: the quantized split-K limit by chip name
  (`Ultra`, `M5`, `M3` or `M4`, other) and the dense split-K threshold in
  `mlx_splitk_partitions` (`Max` or `Ultra`). The first CI run of the oracle,
  on a runner GPU, matched the DiT-shape and convolution goldens but not the
  affine grid, which passes on an M4 Max in debug and release. The affine
  golden is therefore a property of the M4 Max that produced it, and a golden
  per chip would be needed for CI to guard it. Which differing lines exist on
  the runner, and whether the name branches are the whole cause, were not
  looked at. `Music3Device::with_device_name` plans kernels as if on another
  chip while running on the real GPU, so one machine can test the plans of the
  others. Use it in any test whose result depends on a split choice.
- **The CI runner's trig differs from the M4 Max.** The runner's GPU reports
  itself as "Apple Paravirtual device" (no Apple GPU family). On the tiny
  `affine4` profile its stage trace is bit-identical to the M4 Max through
  every norm and QKV projection and first differs at `ar.0.q_rope`: the
  `music3_rope` cos and sin differ in the last bit, for `metal::fast::` and for
  `metal::precise::` alike (31 of 32 probed rope hashes differ, with the
  exp2 variant irrelevant). On the M4 Max `fast` and `precise` agree and are
  not correctly rounded either (`sin(1.0)` is one ulp high), so no portable
  rewrite reproduces the reference; MLX's own rope on the runner would move
  too. A one-ulp angle change only sometimes flips a bf16 rounding, which is
  why bf16 and affine8 stay exact and affine4 does not. Explicit `fma` in the
  quantized kernels was tried and changed nothing on the runner. The zero
  tolerance `affine4` comparison in
  `seven_native_tiny_profiles_match_pinned_mlx_and_reset` therefore runs only
  on `Apple M4 Max` (or with `TURBOSPARK_MUSIC3_ORACLE_ANY_DEVICE=1`); other
  devices still run it twice and check reset, finiteness and sizes. The other
  six profiles keep their tolerances everywhere. Not explained: the runner
  also differs from the affine hash golden on rows 2 to 13 at K = 512 for
  every bit width (387 of 1021 lines), a path the tiny profiles do not reach.
- **A row window must plan from the whole matrix.** Below M3 a dense bf16
  projection of 2 to 15 rows takes the tiled kernel and picks its split-K from
  the output width. The LM head window (a slice of the head) planned from its own
  width, so at K = 4096 the 3016-row window split in two and the 5000-row head
  did not, and a logit differed in the last bf16 bit (-19.625 against -19.75 at
  column 2499 on the CI runner). `WeightData::planned_rows` now carries the
  full head's rows and `Music3Weight::planned_as_rows` applies it.
  `lm_head_window_matches_full_head_under_every_chips_plan` covers M1 to M5
  names. M3 and later take the wide GEMV, which has no split, and were never
  affected. Any other slice of a larger matrix needs the same treatment.
- **Python 3.15 only offered `mlx` 0.32.4**, and the probes refuse any other
  MLX version. Python 3.12 (and 3.14) list 0.32.3; the script uses 3.12.
- **The reference probe does not write `"encoding"`** into `request.json`; the
  test's tolerance table reads it. The verify script adds `"affine4"`.
- **Benchmark only on AC power with a quiet host.** A timing loop that started on
  AC silently continued on battery and the same binary slowed by 2x partway
  through; check `pmset -g batt` before and after. A 1-minute load average above 2
  usually means another session is running.
- **The `#[ignore]`d MLX tests never run in CI.** They run only through
  `scripts/verify-music3.sh --mlx`, so run it after any Music 3 change.
- **Exact output changes the recorded WAV hashes.** After the dense, convolution
  and reduction fixes the exact profile writes `46243600...` (8 s, 1 step) and
  `1cbb8ce0...` (8 s, 30 steps), no longer `16c5db73...` and `361ad4d6...`.
- **Verify a mutation actually applied.** One check this session passed because a
  `sed` expression silently did nothing; assert the edit changed the file.

## Open

- **The other profiles.** 6-bit, 8-bit, MXFP8, MXFP4 and NVFP4 share kernels with
  the verified one but have not been compared with MLX.
- **Re-time the depth stage and a full song.** See the README performance section.
- **Listening.** Intelligibility, transients and chunk seams for the Fast profile.
- **Closing the speed gap to MLX** (see above): the vocoder first, then the DiT and
  the AR stage. Any change must keep `scripts/verify-music3.sh --mlx` green.
