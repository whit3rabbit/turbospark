# 8-bit Z-Image-Turbo MLX speed investigation

Status: exploratory as of 2026-09-26. The goal is about 30 seconds for a
1024 x 1024 image on the Apple M4 Max. No denoise optimization has been
promoted to the Swift app. The current nine-forward output remains the
reference for this investigation.

## Workload and evidence boundary

The local runs used checkout `2c946d1f51e78457033ec0bae1fc6b0cc55e66e6`,
release `ZImageMLXBenchmark`, `mlx-swift` 0.30.6, macOS 26.6.2, and an Apple
M4 Max with 36 GB unified memory. The source was the pinned
[`andrevp/Z-Image-Turbo-MLX-8bit`](https://huggingface.co/andrevp/Z-Image-Turbo-MLX-8bit)
revision `c9f70995562299b1eda9b9145a94dd7a5a1ae0d6`, with 8-bit
group-64 transformer weights. The fixed harness prompt is a tiny red cabin
beside a frozen lake, seed 42, guidance 0. The weight files were already in
the Hugging Face cache; a symlink-only MLX snapshot supplied the component
layout expected by the app. No new weight payload was downloaded.

These are fresh-process standalone pipeline runs. The timer includes local
model loading, text encoding, denoising, VAE decoding, PNG encoding, and
output write. It excludes model download, package compilation, SwiftUI, and
app packaging. Each row below is one run under varying host load. The
[frozen three-run nine-forward median](ZIMAGE_TURBO.md#recommended-model-and-benchmark-record)
is 57.61 seconds; do not treat a ratio between that median and a single row
here as a controlled speedup. `peak_phys_gb` is a progress-event sample of
process `phys_footprint`, not a qualified memory floor.
The maintained benchmark command is in [BENCHMARKING.md](BENCHMARKING.md#image-generation-macos-mlx).
Add `--steps 8` or `--steps 4` to reproduce those schedule variants. The
cache run needs the removed experimental patch.

The [compile-only JIT warmup experiment](MOE_STARTUP.md#compile-only-warmup-results-2026-10-03)
found no repeatable total-startup benefit and remains opt-in.
`TURBOSPARK_METAL_KERNEL_WARMUP=1` prepares the Rust/Metal Qwen MoE text
runner using MLX-derived weights; it does not configure this Swift MLX image
pipeline. Image warmup needs separate startup, output-quality, and memory
measurements, with preparation included in the total elapsed time.

## End-to-end screening runs

| Run | Full transformer forwards | Elapsed | Denoise | Text encode | VAE decode | Peak phys | Output versus nine-forward image |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [Current nine-forward baseline](experiments/z-image-mlx-8bit-speed-2026-09-26/nine-forward.png) | 9 | 68.29 s | 62.33 s | 2.43 s | 3.30 s | 18.47 GiB | Reference |
| [Eight-step scheduler](experiments/z-image-mlx-8bit-speed-2026-09-26/eight-forward.png) | 8 | 56.58 s | 50.92 s | 2.20 s | 3.36 s | 17.07 GiB | Close, changed pixels |
| [Four-step scheduler](experiments/z-image-mlx-8bit-speed-2026-09-26/four-forward.png) | 4 | 30.62 s | 25.48 s | 2.21 s | 2.81 s | 18.47 GiB | Visible composition and detail changes |
| [Nine-step residual cache](experiments/z-image-mlx-8bit-speed-2026-09-26/five-compute-cache.png) | 5 of 9 | 42.73 s | 37.31 s | 2.18 s | 3.12 s | 18.47 GiB | Darker and softer in this scene |

The nine-forward baseline PNG has SHA-256
`7a3a63b042e082cc0dc4e9026aee4eae25e8edde64e98225944fcfd3078a5b19`.
The other PNG hashes, in table order, are
`13a7e1359dc0dd3d1b90ce9c299d241c4105c430f0d955024dd0fd9111623748`,
`b6882b1d768cd0e09700d8071dbff1676f897c090267d8c51c6e89e6308b8582`,
and `add5067c7b93d0bc5ce3327eebfccb3b903fc80007e68462a6e2c73084739d72`.
The four-forward row reaches the requested time by changing the sampling
schedule. It is a speed/quality tradeoff, not an engine speedup.

## Eight-forward prompt screen

The release benchmark now accepts `--prompt` and `--seed`, so the eight-step
variant can generate user-chosen images without editing source for each prompt.
Three new 1024 x 1024 cases compared eight and nine steps with the same
prompt and seed.
They used the same HEAD and model as the first screen, with only the benchmark
CLI prompt and seed flags added.
The linked PNGs and per-run logs are the review artifacts. Each row is one
fresh-process run, not a paired quiet-host speed measurement.

| Case and seed | Nine-forward PNG | Eight-forward PNG | Nine / eight elapsed | RGB cosine / MAE | Visual review |
| --- | --- | --- | --- | --- | --- |
| Portrait, 20260926 | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/portrait-9-step.png) | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/portrait-8-step.png) | 66.15 / 68.55 s | 0.99549 / 0.01949 | Face, cup, and hands remain close; details change. |
| Bookstore, 20260927 | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/typography-9-step.png) | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/typography-8-step.png) | 97.09 / 79.70 s | 0.97606 / 0.06230 | Both show BOOKS and two bicycles; framing changes. |
| Small objects, 20260928 | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/objects-9-step.png) | [PNG](experiments/z-image-mlx-8bit-speed-2026-09-26/objects-8-step.png) | 81.65 / 72.55 s | 0.99562 / 0.02224 | Both duplicate the requested apple; eight steps show one key where nine shows two. |

Run logs: portrait [nine](experiments/z-image-mlx-8bit-speed-2026-09-26/portrait-9-step.txt)
and [eight](experiments/z-image-mlx-8bit-speed-2026-09-26/portrait-8-step.txt),
bookstore [nine](experiments/z-image-mlx-8bit-speed-2026-09-26/typography-9-step.txt)
and [eight](experiments/z-image-mlx-8bit-speed-2026-09-26/typography-8-step.txt),
small objects [nine](experiments/z-image-mlx-8bit-speed-2026-09-26/objects-9-step.txt)
and [eight](experiments/z-image-mlx-8bit-speed-2026-09-26/objects-8-step.txt).

The exact prompts are recorded in
[prompts.txt](experiments/z-image-mlx-8bit-speed-2026-09-26/prompts.txt),
and all six PNG hashes are in
[SHA256SUMS](experiments/z-image-mlx-8bit-speed-2026-09-26/SHA256SUMS).
The RGB measures compare each eight-step output to its own nine-step output;
they are not a quality score. Manual review found no obvious new prompt
failure in these three cases, but the bookstore composition moved. Host load
rose during the sequence: one eight-step run took longer than nine steps.
These timings cannot establish a speed ratio. The earlier cabin screen shows
that eliminating a forward can save denoise time on a
quieter run. The repository's nine-forward parity and app default stay in
place; the `--steps 8` CLI invocation is an opt-in quality tradeoff.

For a narrow pixel check, normalized RGB arrays from the saved PNGs were
compared to the nine-forward PNG. Eight, four, and cached-five runs had raw
pixel cosine 0.99759, 0.98009, and 0.98074; mean absolute error was
0.00949, 0.04271, and 0.06185. These metrics measure agreement with one
seeded image, not prompt fidelity or general image quality. The later
multi-prompt screen above covers only the eight-forward candidate. The cached
result had the largest mean absolute error and did not justify a default
cache mode.

The cache run reused the residual across the 30 main transformer blocks on
zero-based steps 3, 4, 6, and 7. It still ran the current-step prelude and
final projection. This is the mechanism used by
[LeMiCa's Z-Image example](https://github.com/UnicomAI/LeMiCa/blob/main/LeMiCa4Z-Image/inference_zimage.py),
but the local schedule was not calibrated or searched. The temporary Swift
patch was removed. The cache row cannot be reproduced from the current
checkout without reapplying an experimental implementation.

## Denoiser profile and rejected changes

Temporary per-layer timing in a two-step run put the 30 main blocks at
roughly 0.19-0.21 seconds each on the second forward. One block spent about
0.085 seconds in attention and 0.103 seconds in its feed-forward path;
elementwise work was below 0.005 seconds. These are synchronized wall-clock
instrumentation results, not a GPU counter trace. A Metal System Trace
attempt stalled during processing and produced no usable trace.

| Experiment | Focused observation | Decision |
| --- | --- | --- |
| Dequantize one block's quantized linears to BF16 | Block time about 0.191 to 0.173 s; PNG bytes changed and memory grew | Too small and output-changing for promotion |
| Compile one block's feed-forward path with MLX | Feed-forward time about 0.103 to 0.101 s; focused PNG remained byte-identical | Negligible gain |
| Concatenate quantized Q/K/V projections | Focused PNG remained byte-identical; denoise was slightly slower and sampled peak rose from 18.47 to 19.96 GiB | Reverted |
| Try `mlx-swift` 0.31.6 | Two-step denoise 11.66 s versus about 12.06-12.91 s on 0.30.6; PNG bytes changed | No demonstrated nine-step gain; reverted |

These short runs were screening probes, not paired performance studies. The
source and release benchmark were restored to `mlx-swift` 0.30.6. A post-
revert two-step PNG, and a later run after adding the benchmark's prompt and
seed flags, both matched the prior SHA-256
`fac534c374ab29c78dffc2eef130f3663b85132658b7ef1f65421a084d1ef1e3`.

## What the supplied zimgturbo source adds

The supplied `zimgturbo` 0.1.1 source files match the published source
archive (SHA-256
`1d896b98a14621e10e9aec73fb038eb60398bf3356e90081fc13f30cb6bc7d69`).
Its `BOTTLENECK.md`, `OPTIMIZATION_GUIDE.md`, `kernels.py`, and
architecture diagram describe an M5 Pro 16-core GPU engine using Metal 4
tensor operations, per-output-channel symmetric int8 weights, per-token
int8 activations, a fused int8 GEMM, and int8 flash attention. The author's
measurements put denoise at about 12.6 seconds for eight forwards. Their
end-to-end numbers vary between 13.5 and 14.0 seconds across those files
and the [package page](https://pypi.org/project/zimgturbo/0.1.1/). These are
author-reported M5 measurements, not TurboSpark M4 results.

The author's block breakdown attributes about 69% to GEMM, 21% to attention,
and 10% to small kernels on that M5 implementation. The supplied kernel uses
`mpp::tensor_ops::matmul2d` with `int8_t` inputs and int32 accumulation. Its
conversion scheme and calibrated scales differ from this repository's
group-64 MLX-affine 8-bit checkpoint. Using that engine would require a
weight conversion and an independent quality gate. The source also warns
that large SwiGLU and attention-output intermediates need FP32 or BF16 range.
An FP16 shortcut needs numerical validation before use.

A local shape-only probe ran the supplied int8 GEMM kernel through MLX 0.32.2
on the M4 Max. It used synthetic int8 values in `[-8, 8]`, fixed 0.001
row/column scales, zero bias, and best-of-seven synchronized timings.

| GEMM | M x K x N | Best | Effective throughput |
| --- | --- | ---: | ---: |
| QKV | 4608 x 3840 x 11520 | 37.69 ms | 10.82 TOPS |
| Output | 4608 x 3840 x 3840 | 12.93 ms | 10.51 TOPS |
| W1/W3 | 4608 x 3840 x 20480 | 65.76 ms | 11.02 TOPS |
| W2 | 4608 x 10240 x 3840 | 34.26 ms | 10.58 TOPS |

The M5 author reports about 49 TOPS on those shapes. The local probe used
neither converted model weights nor a complete pipeline. The published M5
throughput did not transfer in this probe. This result does not establish a
hardware ceiling or predict full-image speed.

A separate synthetic MLX 0.32.2 probe over the same four shapes summed the
best per-shape times to 158.45 ms for group-64 8-bit matmul, 140.14 ms for
dense BF16, and 145.48 ms for dense FP16. This is a shape comparison, not an
integrated block timing, and it does not include the extra memory needed to
keep dense weights resident.

The supplied guide's tile choice is tuned for M5. A
[follow-up M4 Max sweep](experiments/z-image-mlx-8bit-speed-2026-09-26/m4-gemm-tile-sweep.txt)
used its QKV shape (4608 x 3840 x 11520), the same synthetic int8 inputs and
scales, and five synchronized timings per configuration. All successful tile
variants produced the same FP16 output as the supplied 64 x 128 x 4 tile.
That tile measured 37.75 ms best; 32 x 128 x 4 reached 37.00 ms (about 2%
faster). The other six configurations ranged from 38.61 to 70.93 ms best.
One shape-only 2% improvement does not justify converting the group-64 model
to this kernel's different weight layout. The M4 probe remains around 11 TOPS
for this QKV shape, far from the guide's M5 figure.

## Remaining decision and gates

The current nine-forward path is unchanged. The older
[upstream Z-Image pipeline](https://github.com/Tongyi-MAI/Z-Image/blob/main/src/zimage/pipeline.py)
skips a final zero-timestep forward in its nine-timestep recipe, while the
Swift scheduler executes nine nonzero timesteps and appends zero afterward.
The eight-step row above uses a different schedule, so it does not establish
parity with that upstream recipe. The repository's nine-forward native
contract remains separate. Changing it needs an explicit parity and quality
review.

The user accepts quality-checked pixel changes. The three-prompt screen above
supports an opt-in eight-forward CLI run, but does not qualify a faster app
default. Promotion still needs paired release timings on a quiet host, broader
prompt review, and the Swift real-model and packaged-app gates. No current
result establishes a quality-preserving 30-second default.
