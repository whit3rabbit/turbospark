# MiniMax Music 3: speed, MLX parity, and what not to redo

This is the engineering memory for `crates/audio/src/music/minimax_music3/` and
its Metal kernels. The family README owns the commands and the per-change
measurements; this page owns the conclusions, the method, and the mistakes, so
the next person does not rediscover them. Written 2026-10-17 on an Apple M4 Max
(`applegpu_g16s`) with the pinned 4-bit snapshot
`mlx-community/MiniMax-Music3-4bit` at revision `c7ea32923b24`.

Run `scripts/verify-music3.sh` before and after touching a Music 3 kernel. It
fails if an "exact" change moved a bit. `--mlx MODEL_DIR` adds the diff against
MLX itself.

## Status

| Question | Answer |
|---|---|
| Are the affine 4-bit kernels bit-exact with MLX? | Yes, rows 1 to 16, on the real depth weights (65 of 65 cases with the dense ones below) |
| Are the dense bf16 kernels bit-exact with MLX? | Yes since `cc9344ee`, rows 1 to 16, for the depth projection and heads |
| Is the 4-bit AR path (hiddens, logits, sampled codes) identical to MLX? | Yes: zero differing values across all 8 codebooks, every code, all final hiddens, on the natural reference |
| Is the end-to-end waveform identical to MLX? | **No.** With identical hiddens and identical noise the flow output correlates 0.68 with MLX. The flow half is untraced. |
| What does the app, server, FFI and CLI run by default? | `FlowProfile::Fast`, which is not MLX's trajectory (see below) |
| Has anyone listened to the fast profile? | No. Every quality number here is a drift metric on one prompt and one seed. |

## Two flow profiles

`Music3Runner` opens in `FlowProfile::Fast`; `Model` (the library) opens in
`Exact`. A runner can be switched with `set_flow_profile`, and the CLI takes
`--flow-profile fast|exact`.

| | Exact | Fast |
|---|---|---|
| Solver | Euler | Adams-Bashforth 2 |
| Unconditional guidance branch | every step | reused, interval 3 after warmup 2 |
| Steps when a request names none | 30 | 21 |
| DiT forwards for one chunk | 60 | 30 |
| Matches MLX | yes (AR exact, flow open) | no, by design |

Measured on one 8 s prompt, scored against the exact 30-step audio
(`ab_music3_flow.py`, `spectral_distance.py`, single runs):

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
  before the dense dispatch fix): 30 steps 124.6 s to 85.6 s, DiT stage 95.3 s to
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

## The 4-bit MLX gap: what it actually was

The ignored test `real_music3_pinned_4bit_native_matches_mlx_and_resets` failed at
3.82 against its 3.0 tolerance. The sequence of wrong and right explanations is
the lesson.

1. **Wrong: "a kernel threshold mismatch."** MLX's qmv batch limits for this chip
   (13, 15, 13) match `linear_plan`.
2. **Wrong: "sampling noise, so exactness is impossible."** The test compared
   codebooks 4 to 7 after the sampler had already picked a different code at
   codebook 3 (ours 585, MLX 594), so those rows were incomparable. That
   explained the size of the error but not the flip. Concluding "inexact by
   nature" here was the mistake: **one differing bit amplified by a top-k draw
   looks exactly like noise.**
3. **Right: the first differing tensor.** Comparing per-operation traces with
   identical inputs (teacher forcing) showed 2 of 16384 values differing at the
   input to depth layer 0, one bf16 step. A bf16 add of equal operands cannot
   differ, so the dense `depth.projection` feeding it did.
4. **Cause.** `depth.projection` and `depth.audio_heads` are **dense bf16**, not
   4-bit. MLX 0.32.3 applies each to both CFG rows in one matmul (M = 2), which is
   a different kernel from two single-row calls. We called them one row at a time.

MLX 0.32.3 dense `x @ W^T` dispatch on M3 and later (source: `mlx/backend/metal/
matmul.cpp`, `kernels/gemv.h` at `v0.32.3`):

| Rows (M) | Kernel | Notes |
|---|---|---|
| 1 | gemv | already exact in ours |
| 2 to 15 | `gemv_wide` | needs K % 4 == 0, bf16 or f16, N > 1. float4 `dot` chunks, unroll 8, lane block sums folded in order, halving `simd_shuffle_down`. 32 lanes per row when `ceil(M/5) == 1` or N <= 64, else 16. The result for a vector does not depend on how vectors are grouped. |
| 16 and up, small M x N | split-K GEMM | when `ceil(M/16) * ceil(N/16) <= 2048` (Max and Ultra, else 1024), `K/16 >= 8`, `K >= max(M,N)`. Partitions = `clamp(next_pow2((K/16) / (ceil(M/32) * ceil(N/32))), 2, 32)`. Float32 partials, summed in partition order, rounded once. |
| otherwise | steel GEMM, single K chain | |

Implemented in `crates/gpu/src/shaders/music3.metal` (`music3_gemv_wide`, the
`p[13]` dense split-K fold) and `crates/gpu/src/music3.rs` (`linear_plan`). The
three projections and the head are batched over both CFG rows in
`minimax_music3/mod.rs` and `depth.rs` (including the device-sequence head path).

Result: 65 of 65 linear cases exact against MLX; on the natural reference all
eight codebooks' hiddens and logits, every sampled code, and all 98,304 final
hiddens are bit-identical.

## How to find an MLX mismatch (the method)

1. **Compare one operation, not a pipeline.** `crates/audio/tools/
   gen_music3_mlx_linear_cases.py` runs the MLX operation on the real checkpoint
   weights at rows 1 to 16 and dumps inputs and outputs;
   `crates/gpu/examples/music3_mlx_linear_cmp.rs --require-exact` diffs our
   kernel bit for bit. No sampling or model code is involved.
2. **Teacher-force.** `probe_minimax_music3_reference.py --force-warmup-codes
   '[...]'` replaces MLX's draws with ours so every depth step sees the same input.
   Without it a one-ulp difference flips a draw and everything downstream is
   incomparable.
3. **Walk the trace to the first difference.** `--trace --trace-max-calls N` on
   MLX versus `TURBOSPARK_MUSIC3_TRACE=1` on `music3_backend_probe`. Compare in
   order and count differing values, not just the maximum.
4. **Read the pinned MLX source** for the dispatch rule before guessing. The
   heuristics (`get_qmv_batch_limit`, `gemv_wide_config`, the split-K trigger)
   are device-generation dependent.

## Pitfalls checklist

- **`Music3Runner` opens in the Fast profile.** Any parity probe or test that
  compares with MLX or a CPU reference must call
  `set_flow_profile(FlowProfile::Exact)`. `music3_bench` and `ab_music3_flow.py`
  default to `exact` for that reason; pass `--flow-profile fast` to measure the
  default.
- **A loose tolerance hides a real bug.** The 4-bit gate sat at 3.0 on a 1e-3
  quantity. Treat a tolerance that needed raising as a finding, not a fix.
- **The hash oracles cover affine kernels only.** `hashes` and `dit_hashes` are
  checked against golden files by `crates/gpu/tests/music3_oracle.rs`. Dense
  kernels are covered only by the MLX linear diff, so changing one without
  running `scripts/verify-music3.sh --mlx` is blind.
- **Python 3.15 only offered `mlx` 0.32.4**, and the probes refuse any other
  MLX version. Python 3.12 (and 3.14) list 0.32.3; the script uses 3.12.
- **The reference probe does not write `"encoding"`** into `request.json`; the
  test's tolerance table reads it. The verify script adds `"affine4"`.
- **Benchmark only on AC power with a quiet host.** A timing loop that started on
  AC silently continued on battery and the same binary slowed by 2x partway
  through; `pmset -g batt` before and after. A 1-minute load average above 2
  usually means another session is running.
- **The `#[ignore]`d MLX tests never run in CI.** They run only through
  `scripts/verify-music3.sh --mlx`, so run it after any Music 3 change.
- **Changing the exact path changes the recorded WAV hashes.** After
  `cc9344ee` the exact profile writes `bf3d26d5...` (8 s, 1 step) and
  `e87ffd16...` (8 s, 30 steps), no longer `16c5db73...` and `361ad4d6...`.

## Open

- **The flow half.** With identical hiddens and identical noise (supplied through
  `--noise-file` on both sides) the waveform correlation against MLX is 0.68. The
  DiT and vocoder have not been traced stage by stage the way the depth decoder
  was. Until they are, "matches MLX" holds for the AR stage only.
- **Re-time after the dense fix.** The speed numbers above predate it; the timing
  run was invalidated by a switch to battery power.
- **Listening.** Intelligibility, transients and chunk seams for the Fast profile.
- **MLX itself as a speed reference.** Not benchmarked against ours on this
  clip. The README's 56 s MLX figure is from an earlier session on MLX 0.31.2.
