---
uuid: "09ef101c-e821-4ff5-95ff-3bb53b5e05a3"
title: "turbospark-gpu: one model can use two norm conventions"
summary: "A normalization convention (plain vs centered RMS norm) is a property of the TENSOR, not the model family. One checkpoint can carry both"
tags: ["crate", "gpu", "gotchas"]
source: "crates/gpu/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Why does this crate have two RMS norm kernels that look almost identical?

`rmsnorm_bf16w` (plain, `out = x * inv * w`) and `rmsnorm_bf16w_centered`
(`out = x * inv * (1 + w)`) are separate kernels, not one kernel with a
mode flag, because the convention is a property of the individual tensor
being normalized, not of the model family dispatching it. `muse_glimmer`
carries both in one forward pass: four per-layer norms are centered while
its final `model.norm.weight` is plain. Asking "which norm does this
family use" has no answer. Only "which norm does this TENSOR use" does.

The `+1` is applied on the FP32 accumulator at dispatch time, never baked
into the stored weight at repack time, because baking is lossy exactly
where it matters: a centered weight sits near zero, and BF16's absolute
resolution near 1.0 is 2^-8, so a weight of 0.01 would lose about 39% of
its own magnitude.

## Don't

- Don't assume a normalization convention transfers from one tensor to a
  same-shaped, same-named tensor elsewhere in the same checkpoint. The
  `qwen3_5` trunk and its MTP head both carry `self_attn.q_norm.weight` at
  the same shape through the same shared attention function, and the
  head's copy is centered while the trunk's is plain, because the
  converter baked the `+1` into the trunk weights and left the head's
  alone. `QkNormConvention` is a parameter on that call site for exactly
  this reason.
- Don't build or trust a parity fixture whose norm weights sit near zero.
  `x * w` and `x * (1 + w)` converge as `w -> 0`, so a fixture there cannot
  tell the two kernels apart even if one is silently substituted for the
  other. Assert the fixture discriminates before trusting the parity case.
- Don't add the `+1` as a function constant on the plain kernel. A
  specialization byte that misses the pipeline cache's `constants_key`
  means the two norms of one model can silently collapse to the same
  compiled function, whichever variant happened to compile first.
- Don't assume this generalizes to every family: Gemma's `+1` was a real
  candidate and was refuted by measurement (its GGUF and MLX installs'
  resident cores are bit-identical without it, despite llama.cpp's own
  converter adding one). Re-derive per tensor and per family rather than
  applying this pattern by analogy.
- Don't expect a whole-block composition test (e.g. the vision tower's) to
  catch every kernel-selection mistake inside it. Two nearly-identical
  functions (like the tower's two GELU forms) can differ by less than the
  accumulated floating-point error the block already tolerates, making the
  wrong choice invisible at that test's bound. Pin the choice with a
  dedicated per-kernel case instead.
