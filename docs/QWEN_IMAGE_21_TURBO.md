# Qwen-Image-2.1-Turbo MLX: Swift app implementation and evidence

Status: ships in the Swift app on the vendored Swift + MLX pipeline.
Text-to-image only, batch one, CFG off, 512 to 1024 pixel sides in multiples
of 32. The native Rust/Metal runtime in `crates/image` stays Z-Image-only. A
native Qwen runtime is a separate, gated phase (see the end of this page).

This page records what was built, what was measured and under which
conditions, and what is still open. The reusable bring-up process for image
families is in [ZIMAGE_TURBO.md](ZIMAGE_TURBO.md). The non-Turbo model is
recorded in [QWEN_IMAGE_21_MLX.md](QWEN_IMAGE_21_MLX.md).

## Pinned inputs

- Repository: `Sawfwair/Qwen-Image-2.1-Turbo-MLX-Mixed-4bit`, revision
  `d8276b786f94a4e8944189cc181e839e31759ab5`, with no gating on the download. An unofficial MLX
  quantization (no fine-tuning) of `Qwen/Qwen-Image-2.1-Turbo`, the 8-step
  distillation of `Qwen/Qwen-Image-2.1`. The conversion's own metadata marks
  `quality_qualified: false`: only weight-reconstruction error is recorded,
  not image quality.
- Catalog row `qwen-image-2.1-turbo-mlx-mixed-4bit`, family
  `qwen-image-2.1-turbo`, 16 required files. The 16 downloaded files match the
  repository's published `SHA256SUMS` (checked locally). Weight payload is
  14.24 GB, the packed install is 14.25 GB, and the retained source tree the
  MLX runtime reads is another 14.24 GB, so an install needs about 28.5 GB.
- License: Qwen Research License Agreement, non-commercial (research and
  evaluation use only). `LICENSE` and `Notice` are required files and stay in
  the retained source tree. The app shows the terms beside the download.
  Derivative products must state "Built with Qwen".

## What differs from the Qwen-Image-2.1 port

The architecture is unchanged (32-layer single-stream DiT, Qwen3-VL text
encoder, 64-channel VAE, prefix KV cache). Four things differ and each one
broke loading or sampling until handled:

- Sampling: a fixed 8-step ladder, `sample_sigmas` in `model_index.json`
  (1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
  then 0), with `use_dynamic_shifting: false` and no terminal stretch. The
  scheduler uses the ladder verbatim and the pipeline refuses any other step
  count, because a different count would skip or invent sigmas.
- Mixed quantization: the transformer blocks are MLX affine Q4 and the text
  encoder is Q8, group 64, everything else BF16. The component configs carry
  one `quantization` block each and no per-tensor map. The loader previously
  hard-coded 4 bits for every layer with a `.scales` tensor. It now infers
  each layer's width from the tensor shapes
  (`bits = 32 * packedLast / (64 * scalesLast)`) and rejects a layout that is
  not an exact MLX affine one.
- Tensor names: the text encoder ships in the transformers-5 layout
  (`model.language_model.*`, vision tower under `model.visual.*`) while the
  module tree uses the mlx-vlm layout (`language_model.model.*`). The
  transformer uses Diffusers names for the shared modulation
  (`modulation.1`) and the timestep MLP (`time_text_embed.timestep_embedder`).
  The VAE stores RMS-norm `gamma` as `[C,1,1,1]`. Each is mapped explicitly
  and unit-tested against names copied from the safetensors headers. The
  vision tower (about 1 GiB at Q8) is no longer read.
- Layout: Diffusers shard names for the transformer, and single-file text
  encoder and VAE with no index. The packer already handled both.

A loader defect found while doing this is worth recording: a module parameter
with no checkpoint tensor only logged a warning and kept its random
initialization, so generation ran to completion and returned noise. The first
real-weights run hit exactly that (three parameters). It is now a hard error.
This applies to every component and to the older 4-bit checkpoint too. That
checkpoint's source tree was not available here, so its load with the strict
check has not been run.

## Memory design

The pipeline takes a residency policy. The app uses `staged`: encode the
prompt, release the 9.4 GB encoder, load the transformer, denoise, release it,
load the VAE, decode, release it. `warm` keeps all three resident (the old
behavior). Warm residency was not measured in this work.

The first real runs showed that the VAE decode, not the denoiser, set the
lifetime peak. Two changes, both measured:

- Evaluating each decoder block and clearing the allocator cache between
  blocks bounded the live set to one block. Output is byte-identical to the
  unchanged decode (same md5).
- Decoding in bfloat16, the VAE's stored dtype and the one the Diffusers
  reference decodes in, roughly halves the remaining decode memory. It
  changes pixels slightly (see the table). The app session uses it, and the
  library default remains float32.

## Measurements

Machine: M4 Max, 36 GB, AC power, release build, fresh process per run,
fixed prompt ("A tiny red cabin beside a frozen lake at blue hour, pine
forest, stars, cinematic lighting, wide composition, no text."), seed 42,
8 steps, `staged` residency.

Conditions to read these under: this was not a quiet host. Other sessions were
running on the machine and the load average was 7 to 11 during these runs.
Each row is a single run. They are bring-up observations, not frozen rows, and
they are not paired against any other model. The 1024 bfloat16 run is
the clearest case: its denoise took 127 s against 77 s in the float32 run
with identical denoiser code, so its wall time is contention, not a property
of the decode dtype.

| Case | Wall (s) | Peak phys_footprint | Stage peaks (GiB) |
| --- | --- | --- | --- |
| 512, float32 decode, no per-block eval | 22.4 to 26.0 (3 runs) | 18.24 to 18.29 GiB | decode 18.23, encode 8.27, denoise 6.63 |
| 512, float32 decode, per-block eval | 20.3 | 8.31 GiB | decode 7.89, encode 8.28, denoise 6.63 |
| 512, bfloat16 decode | 31.7 (contended) | 8.30 GiB | decode 4.39, encode 8.30 |
| 1024, float32 decode, per-block eval | 95.7 | 20.52 GiB | decode 20.52, encode 8.25, denoise 7.68 |
| 1024, bfloat16 decode | 153.1 (contended) | 11.20 GiB | decode 11.20, encode 8.26, denoise 7.69 |

Stage times for the 95.7 s run: denoise 77.1 s, weight loading 10.9 s, text
encode 1.4 s, VAE decode 6.2 s.

Image difference, bfloat16 against float32 decode, same seed, 8-bit RGB:
512 x 512 PSNR 48.2 dB (mean squared error 0.99, max pixel error 33), and 1024 x
1024 PSNR 50.9 dB (mean squared error 0.53, max pixel error 35). Both are
below one grey level on average, and the maximum is a few fine-detail pixels.

Determinism: four fresh-process float32 runs at 512 with the same seed
produced the same PNG (md5 prefix `60b094d63432`), including runs before and
after the per-block evaluation change.

External reference (not a pass criterion): the model's publisher reports about
97 s and 20.6 GiB at 1024 and 9.1 GiB at 512 for this checkpoint on their own
runtime. The float32 rows above land in the same range, which is consistent
but is not an equivalence claim.

## Image review

The 512 and 1024 outputs are coherent and follow the prompt (red cabin,
frozen lake, pine forest, blue-hour sky). The 1024 image shows the stars the
prompt asks for, and the 512 image does not. No noise, banding, or
overexposure was seen in either. That is a visual check on one prompt and
seed, not a quality qualification. Full-image parity against Diffusers is
unverified, as it is for the non-Turbo model, and BF16 equivalence is not
claimed.

## Verification record

- `make swift-test-qwenimage`: 20 tests, including the ladder values, the
  `model_index.json` parser and its rejections, per-layer bit inference from
  real tensor shapes, a mixed 8-bit, 4-bit and dense module round trip,
  every key mapping above, and the strict missing-parameter error. The
  mixed-width round trip was mutation-checked: forcing the old fixed 4-bit
  width makes it fail with the shape mismatch the bug produced.
- `cargo test -p turbospark-image -p turbospark-catalog -p turbospark-ffi`
  passes. New: the Turbo layout through the full install builder, the family
  prefix scoped to this repository (not its publisher), the catalog row, and
  file selection against the repository's real 26-file listing.
- Real install: the pinned files were packed with the production builder.
  The manifest records family `qwen-image-2.1-turbo`, 8 steps, guidance 1.0,
  prompt bound 1024, latent `[1,64,64,64]`.
- `make swift-test-app`: 3,144 tests, 0 failures (9 skipped, none of them
  these).
- App path on the real install: `ServedImageMetalTests`, run with
  `TURBOSPARK_TEST_SERVED_IMAGE_ALIAS=qwen-image-2.1-turbo-mlx-mixed-4bit`,
  passed. `AppModel` found the install through the FFI, attached it to the
  embedded server, and `POST /v1/images/generations` at the manifest's
  1024 x 1024 returned HTTP 200 and a valid PNG in 132.7 s, including server
  start and a cold load on the same contended host. The Images screen and
  the model manager UI were not exercised by hand.

## Models in other folders

Image installs are found in the folders listed under Settings > Models >
Additional Model Folders, the same list the text models use. The scan looks at
each folder and its immediate subdirectories, and accepts one install bundle
chosen directly. A found model appears in the Images model menu if it is one of
the two supported families. Removing it from the list excludes the path and
leaves the files on disk, and Settings > Models restores it. Only installs in
the shared store are deleted. An install copied to another drive runs only if
its retained source tree (`.<alias>.<revision>.image-source`) was copied beside
it, since the MLX runtime reads that tree and not the packed copy.

Before this change the text-model scan treated an image install as a text
model, because both are bundles with a `manifest.json`. It now skips bundles
whose manifest declares `image-generation`.

## Findings in adjacent code

- `turbospark-model pull-image --repo` used its own copy of the file
  selection rules and had no `processor/` rule, so every Qwen-Image row
  failed to pack from the CLI. It now calls
  `catalog::materialize_image_source`, the function the app's installer uses.
  The CLI still deletes the downloaded source tree after packing, so a CLI
  install does not leave the tree the MLX runtime reads, while the app's installer
  keeps it.
- The catalog's quantization label comes from the alias suffix and the
  installed row's label from the transformer's observed widths, so both say
  4-bit for this model. The Q8 text encoder is not in that label.

## Open items

- Warm residency was not measured, and no quiet-host, repeated-run frozen row
  exists for this model. No paired comparison against Z-Image Turbo was run
  on the same day, so no speed ranking is claimed. The recommendation order
  puts Z-Image Turbo 8-bit first on the strength of its recorded row only.
- The bfloat16 decode default for the app is a judgment from the PSNR above,
  not a quality gate. Float32 is one argument away if that is preferred.
- Only 512 and 1024 were exercised. 2048 pixel output, reference images, and
  transparency are not supported or tested.
- The model manager's heading still reads "Curated Z-Image models".
- The pinned revision came from a repository renamed on its first day, so the
  pin and the hash check are the protection against a later change.

## Native runtime (not started)

Entry criteria, because the native Z-Image path recorded in
[BENCHMARKS.md](BENCHMARKS.md#image-generation) is about 63 times slower than
the MLX pipeline:

1. A quantized GEMM (simdgroup matrix, fused affine dequantization, mixed
   4 and 8 bit) and a fused attention kernel in `shaders/image.metal`,
   benchmarked on Z-Image against the Swift MLX row first.
2. Then a Qwen runtime: single-stream DiT with shared modulation and the
   prefix KV cache, block-causal attention, three-axis RoPE (16, 56, 56), the
   Qwen3-VL text path, the 64-channel VAE decode with per-channel statistics,
   and a staged memory plan. The Swift pipeline is the parity oracle.
3. A new resource contract and the IG0 and IG1 evidence repeated for this
   family, per ZIMAGE_TURBO.md.
4. The native path is pinned in the app only when it matches or beats MLX on
   wall time and peak memory under the same protocol.
