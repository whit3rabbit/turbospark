# Qwen-Image-2.1 MLX: Swift app implementation and evidence

Status: first-cut MLX port shipped behind the app's image family dispatch.
Text-to-image only, 1024x1024 bounded presets, batch one, true-CFG off.
This page records what was built, what is verified, and the measured
head-to-head against the Z-Image Turbo MLX rows on the same machine. The
reusable bring-up process for image families lives in
[ZIMAGE_TURBO.md](ZIMAGE_TURBO.md); this page is the Qwen-Image record.

## What shipped

- Catalog row `qwen-image-2.1-mlx-4bit` pinning
  `mlx-community/Qwen-Image-2.1-MLX-4bit` at revision
  `4db4e8c0c0e7a1debf0320415bec8388e888494c` (about 10.5 GB), with a
  per-family install envelope: 40 supported steps, one transformer forward
  per step, guidance 1.0 (true-CFG off), 1024 prompt tokens, 64-channel
  latents, 512-to-1024 bounded sides in multiples of 32.
- `swift/TurboSparkApp/Vendor/QwenImage`, an original Swift + MLX port of
  the diffusers `QwenImage21Pipeline` stack (see its `UPSTREAM.md` for the
  source files it follows). No public Swift or Python MLX implementation of
  this model existed; LM Studio and Radiant Canvas ship proprietary engines.
- App integration: `QwenImageGenerationSession` behind the same
  `ImageGenerationSession` protocol as the Z-Image adapter, family dispatch
  in `AppModel+ImageGeneration`, the "Qwen-Image 2.1" family row in the
  model menus, and the 32 GiB memory tier for the download recommendation.
  The retained MLX source tree loads directly; no snapshot staging or
  `quantization.json` synthesis is needed because each component config
  embeds its quantization metadata.
- `QwenImageMLXBenchmark`, mirroring the `ZImageMLXBenchmark`
  PROGRESS/RESULT/BREAKDOWN/PEAK_MEM contract with `--steps`, `--prompt`,
  `--seed`, `--width`, and `--height` flags.

## Architecture facts that shaped the port

These were read from the diffusers sources at the pinned reference and are
load-bearing; each one broke images when gotten wrong during bring-up.

- Single-stream DiT: 32 layers, inner 4096 (32 heads x 128), 64 latent
  channels in and out, patch size 1, so a 1024x1024 image is 4096 tokens
  (four times Z-Image's token count at the same resolution).
- One shared modulation projection (SiLU then Linear 4096 -> 16384) is read
  identically by every block, sliced into [scale1, gate1, scale2, gate2].
  Scale-only adaLN: residual branches are `x + tanh(gate) * branch(scale *
  x)`.
- `causal_condition` makes the text prefix timestep-independent: text tokens
  modulate from t = 0, so their keys and values are computed once on step 0,
  extracted into a prefix KV cache, and reused by every later step. Cached
  steps attend image queries against the cached prefix concatenated with the
  current image keys.
- Attention is block-causal: the text prefix is causal over itself, and the
  target image attends to everything.
- 3-axis RoPE with axes (16, 56, 56), theta 10000. Image h/w grids are
  centered on zero; the frame axis freezes image tokens at the text length.
  Angles are `pos * theta^(-2i/dim)` computed in float32.
- Text encoder: the Qwen3-VL-8B language model only. The conditioning
  vector is the last decoder hidden state BEFORE the final norm. The prompt
  is framed with the raw template string, not `apply_chat_template`, and the
  system-segment tokens are dropped from the hidden states after encoding.
- Timestep input: the pipeline passes `sigma` and the embedding multiplies
  by `time_factor = 1000`.
- VAE `AutoencoderKLQwenImage21`: decode-only path, float32 weights, 16x
  spatial upsampling with per-channel `latents * std + mean`
  denormalization, RMS-norm residual blocks (L2 over channels), single-head
  spatial attention as 1x1 convolutions, and parameter-free DupUp3D
  channel-duplication shortcuts.
- Scheduler: flow-match Euler with dynamic exponential time shift
  (`exp(mu) / (exp(mu) + 1/t - 1)`, mu linear in image sequence length from
  base 256 to max 8192 with shifts 0.5 to 0.9), stretched so the last sigma
  lands on terminal 0.02, then a trailing zero. The base ladder runs from
  sigma 1 down to `1 / num_train_timesteps` (0.001), not `1 / steps`; the
  first port used the wrong lower endpoint and the parity test now freezes
  the corrected values.
- Quantization: MLX affine 4-bit group 64 on 2D linear weights and
  embeddings, loaded from each component config; norms, convs, and the VAE
  stay dense.

## Verification record

- End-to-end: the 20-step and 40-step 1024x1024 generations produce
  coherent, prompt-matching images (fixed cabin-by-the-lake scene used for
  visual checks during bring-up).
- Determinism: same seed in fresh processes produces byte-identical PNGs.
  The three same-seed 40-step benchmark runs double as this gate; their
  MD5s are recorded with the benchmark rows.
- Parity: a PyTorch/diffusers reference run was infeasible on this checkout
  (no torch environment). The fallback is component-level checks frozen in
  `Vendor/QwenImage/Tests/QwenImageTests/QwenImageParityTests.swift`,
  computed independently from the diffusers formulas:
  - sigma schedules at 8 and 40 steps against recorded reference values
    (mutation-checked: reverting the base-ladder endpoint fails the test at
    7e-3 absolute),
  - the exact T2I prompt template and system segment strings,
  - RoPE position grids (centered h/w axes, frame axis frozen at text
    length) and frequency values against `pos * theta^(-2i/dim)`.
  Full-image parity against diffusers remains unverified; the visual checks
  are the only image-level evidence.
- Memory: peak `phys_footprint` at the 1024x1024 envelope measured 25.9
  GiB, which sets the 32 GiB tier for the download recommendation.

## Benchmark record

Machine: M4 Max development machine, release build, fresh process per run,
fixed prompt and seed 42 at 1024x1024, per
[BENCHMARKING.md](BENCHMARKING.md#image-generation-macos-mlx). Timing blocks
were run back-to-back on a quiet machine (no competing compile or test jobs)
between 13:08 and 13:47 local. The Z-Image-Turbo row is the installed 8-bit
MLX variant (`andrevp/Z-Image-Turbo-MLX-8bit`, revision
`c9f70995562299b1eda9b9145a94dd7a5a1ae0d6`, staged snapshot) at its native
nine steps; the Qwen rows use the installed 4-bit variant at its native 40
steps plus low-step probes.

| Row | Runs (s) | Median (s) | Denoise median | Peak phys_footprint |
| --- | --- | --- | --- | --- |
| Z-Image Turbo 8-bit, 9 steps | 78.68 / 86.80 / 88.03 | 86.80 | 70.58 s (7.8 s/step) | 12.73-12.80 GiB |
| Qwen-Image-2.1 4-bit, 40 steps | 481.93 / 542.84 / 574.24 | 542.84 | 522.16 s (13.1 s/step) | 25.87-25.90 GiB |
| Qwen-Image-2.1 4-bit, 20 steps (probe) | 303.15 | - | 279.83 s | 25.87 GiB |
| Qwen-Image-2.1 4-bit, 12 steps (probe) | 200.60 | - | 175.80 s | 25.90 GiB |

Other stages (medians): Z-Image text encode 10.5 s, VAE decode 6.2 s;
Qwen text encode 12.1 s, VAE decode 5.9 s. Single-run probes are
exploratory, not frozen rows.

Determinism: the three 40-step Qwen runs and the three 9-step Z-Image runs
each produced byte-identical PNGs across fresh processes (Qwen MD5
`54b4d12461e9c4e7d953df6ca3890b2a`, Z-Image MD5
`6328c2ad29a33cfb27e4cc0e2f2c6f35`).

Reading the per-step economics: fitting the denoise times as one prefill
step plus N-1 cached steps gives a cached step near 12 s and a first step
near 45 s. The first step carries the text-prefix forward, full-sequence
attention, and the lazy materialization of the quantized weights, so it is
roughly four cached steps by itself.

Head-to-head verdict: at native settings Z-Image Turbo is about 6.3x
faster end to end (86.8 s versus 542.8 s median) and fits in half the
memory (12.8 versus 25.9 GiB peak). Even the 12-step Qwen probe (200.6 s)
is more than twice the Z-Image native time. This matches the architecture:
the Qwen DiT is larger and processes four times the image tokens at the
same 1024x1024 output, and its native schedule needs 40 steps versus
Turbo's 9. The Qwen family is not the speed pick on this hardware; its
value is the different output character and text rendering, at a 32 GiB
memory tier.

Comparison limits: this block is one machine on one day; the frozen
Z-Image 8-bit row recorded elsewhere in these docs measured 57.61 s median
on a different day and machine state, consistent with the documented
tens-of-percent wall-time swing. The head-to-head ratio above is only
valid within this block. The 20-step and 12-step probes change the sigma
schedule and are not quality-equivalent to the 40-step native setting.

## Scope limits of the first cut

- Text-to-image only. Reference images, masks, and the editing path are not
  ported; the pipeline errors rather than guessing if asked for them.
- Batch one, no true CFG. The `trueCfgScale` request field exists but the
  first cut fixes it at 1.0; enabling guidance doubles forwards and needs
  its own memory envelope before it can ship.
- Local snapshot directories only: the app always runs from the retained
  install source. Installs that predate source retention fail with a
  reinstall message instead of falling back to a Hub fetch.
- The native Rust Metal image runtime in `crates/image` stays Z-Image-only;
  this family runs through the vendored MLX package, like the app's MLX
  Z-Image route.

The [Qwen MoE text JIT warmup experiment](MOE_STARTUP.md#compile-only-warmup-results-2026-10-03)
remains opt-in after finding no repeatable startup gain.
`TURBOSPARK_METAL_KERNEL_WARMUP=1` applies to the Rust/Metal text runner,
including MLX-derived text weights, not this Swift MLX image pipeline.
An image warmup default needs its own paired total-startup, image-quality,
and memory gates.

## License note

The upstream weights carry the `qwen-research` license (non-commercial
research). The vendored package contains no weights; app installs download
at explicit user action like the other image rows.
