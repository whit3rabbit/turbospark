# ZImageMLXBenchmark

## Overview
A standalone command-line harness (a single `main.swift` with `@main`) that generates one 1024 x 1024 image through `ZImagePipeline` from an MLX model directory and prints per-stage wall-clock time and peak memory. Its stage bucket contract is the reference that the sibling [QwenImageMLXBenchmark](../QwenImageMLXBenchmark/EGM.md) follows. It is a measurement tool and not part of the shipped app. Any number it prints still follows the benchmark protocol in `docs/BENCHMARKING.md` before being quoted.

## Interface
- CLI: `ZImageMLXBenchmark <mlx-model-directory> [output.png] [--steps N] [--guidance F] [--prompt TEXT] [--seed N]`. Defaults are 9 steps, guidance 0, seed 42, fixed 1024 x 1024, residency policy `.warm`, and an output of `zimage-mlx-benchmark.png` in the temporary directory [main.swift:ZImageMLXBenchmark].
- `StageBreakdown`: accumulates seconds per bucket (`load_weights_s`, `text_encode_s`, `denoise_s`, `vae_decode_s`, `png_encode_s`, `other_s`) [main.swift:StageBreakdown].
- `PeakMemory`: `currentFootprintBytes()` and `lifetimePeakFootprintBytes()` via `task_info(TASK_VM_INFO)` [main.swift:PeakMemory].
- Output lines on stdout: `PROGRESS`, `RESULT` and `BREAKDOWN` (the last carries `peak_phys_gb`) [main.swift:emit].
- Child directories: none.

## Invariants
- "Encoding text" fires twice (tokenizer and text-encoder weight load during `loadModel`, then the real prompt encode in `generateCore`) and both land in `text_encode_s`. The three weight-loading stages ("Loading model", "Loading transformer", "Loading VAE") all land in `load_weights_s`. An unknown stage falls into `other_s` [main.swift:StageBreakdown].
- The quotable peak is the kernel lifetime peak, because a sampled value misses transients such as the VAE decode's float32 activations. The reported `peak_phys_gb` is the maximum of the progress-event samples, the final sample and the lifetime peak. Unlike the Qwen harness, there is no 20 ms per-stage sampler here, so no per-stage peaks are reported [main.swift:lifetimePeakFootprintBytes] [main.swift:ZImageMLXBenchmark].
- Argument errors terminate with `fatalError`. There must be one or two positional arguments. `--guidance` must be a non-negative number [main.swift:ZImageMLXBenchmark].
- Resolution and residency are fixed in code (1024 x 1024, `.warm`), so a result from here is comparable only to runs with the same settings.

## Side Effects
- Runs the full image pipeline on the GPU through MLX and loads the model from the given directory.
- Disk: writes the PNG to the output path [main.swift:ZImageMLXBenchmark].
- Writes to stdout through `FileHandle.standardOutput`.
- None for network, Keychain, UserDefaults, notifications or extra threads.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- The harness itself is run by hand with the usage line above, with the protocol in `docs/BENCHMARKING.md`. No test in this directory was read.
