# QwenImageMLXBenchmark

## Overview
A standalone command-line harness (a single `main.swift` with `@main`) that generates one image through `QwenImage21Pipeline` from an MLX model directory and prints per-stage wall-clock time and memory figures. It mirrors the stage bucket contract of the sibling [ZImageMLXBenchmark](../ZImageMLXBenchmark/EGM.md), so the two report comparable fields. It is a measurement tool and not part of the shipped app. Any number it prints still follows the benchmark protocol in `docs/BENCHMARKING.md` before being quoted.

## Interface
- CLI: `QwenImageMLXBenchmark <mlx-model-directory> [output.png] [--steps N] [--prompt TEXT] [--seed N] [--width N] [--height N] [--residency warm|staged] [--vae-dtype float32|bfloat16]`. Defaults are 40 steps, seed 42, 1024 x 1024, `warm` residency, `float32` VAE, and an output of `qwen-image-mlx-benchmark.png` in the temporary directory [main.swift:QwenImageMLXBenchmark].
- `StageBreakdown`: accumulates seconds per bucket (`load_weights_s`, `text_encode_s`, `denoise_s`, `vae_decode_s`, `png_encode_s`, `other_s`) from progress stage names [main.swift:StageBreakdown].
- `PeakMemory`: `currentFootprintBytes()` (live `phys_footprint`) and `lifetimePeakFootprintBytes()` (kernel `ledger_phys_footprint_peak`) via `task_info(TASK_VM_INFO)` [main.swift:PeakMemory].
- `StagePeakSampler`: a 20 ms polling thread recording the maximum footprint seen per pipeline stage [main.swift:StagePeakSampler].
- Output lines on stdout, flushed per line: `PROGRESS`, `RESULT`, `BREAKDOWN`, `STAGE_PEAK` and `PEAK_MEM` [main.swift:emit].
- Child directories: none.

## Invariants
- The quotable peak memory is the kernel lifetime peak, not a sampled value, because the VAE decode allocates float32 activations and returns them to the OS between samples. The reported `PEAK_MEM` is the maximum of the sampled, final and lifetime figures [main.swift:lifetimePeakFootprintBytes] [main.swift:QwenImageMLXBenchmark].
- The per-stage sampler exists because lazy MLX weight materialization and VAE activations rise and fall between progress events, so event-time sampling can miss the stage that sets the peak. `STAGE_PEAK` values come from the sampler [main.swift:StagePeakSampler].
- Stage bucket names must stay aligned with the ZImage harness (they are keyed by the pipeline's stage strings such as "Loading model", "Denoising" and "Decoding"). An unknown stage falls into `other_s` [main.swift:bucketByStage].
- Argument errors terminate with `fatalError`. There must be one or two positional arguments (model, optional output path) [main.swift:QwenImageMLXBenchmark].
- `--residency staged` and `warm` are different memory and speed trade-offs, so a recorded result must name the residency and VAE dtype (the `RESULT` line prints both).

## Side Effects
- Runs the full image pipeline on the GPU through MLX and loads the model from the given directory.
- Disk: writes the PNG to the output path [main.swift:QwenImageMLXBenchmark].
- Spawns a polling thread (stopped at the end) and writes to stdout.
- None for network, Keychain, UserDefaults or notifications in this file.

## Verification
- Run: `swift test --package-path swift/TurboSparkApp`
- The harness itself is run by hand with the usage line above, with the protocol in `docs/BENCHMARKING.md`. No test in this directory was read.
