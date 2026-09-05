---
uuid: "03e2bfe7-3eee-4165-94cc-967cf2c3f869"
title: "turbospark-vision-io"
summary: "Portable vision preprocessing for the qwen3_5 tower: decode, PIL-bicubic resize, patchify, position tables. Plain Vec<f32> arithmetic, no Metal, builds off macOS on purpose"
tags: ["crate", "vision-io"]
source: "crates/vision-io/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-vision-io do?

Image decode, PIL-bicubic smart resize, normalize, patchify, and the three
vision position tables (interpolated position embedding, vision rope
frequency rows, mRoPE triples) for the `qwen3_5` vision tower. Plain
arithmetic over `Vec<f32>`, no Metal, no macOS dependency, no model
install. It's deliberately one of the crates the root cross-target
`cargo check` covers, so a `cfg` or dependency that drops it from that
portable list is a bug in this crate, not an accepted limitation.

Every fixture under `tests/generated/` is PROBED from the vendored
mlx-vlm reference, never hand-transcribed, so a fixture can't encode this
port's own misreading of the source. Regenerate with
`uv run --python 3.12 --with mlx --with mlx-vlm --with numpy --with pillow
scripts/qwen3vl_vision_oracle.py all`.

## Don't

- Don't assume the patch row's inner feature order is the reference's
  `(C, T, P_h, P_w)`. It's `(T, P_h, P_w, C)` here, on purpose, matching
  the tower's own `patch_embed.proj.weight` layout so repack can copy the
  weight verbatim. A mismatch here is silent wrong numerics: the GEMM's
  shape stays valid, the tower runs, and the model just reads a different
  image.
- Don't reach for `tvF.resize` (torchvision) semantics without checking
  which library the checkpoint's own processor actually calls. This
  family's `preprocessor_config.json` declares `resample: 3`, i.e. PIL
  bicubic, and PIL and torchvision differ by whole 8-bit coefficient
  levels, four orders of magnitude over this crate's parity bar.
- Don't source a new golden fixture from a JPEG. Pillow and the `image`
  crate disagree on individual samples of the same JPEG (different IDCT
  and chroma upsampling), so a JPEG-sourced fixture would compare
  arithmetic PLUS an undiagnosable decoder difference. Every fixture starts
  from synthetic pixels embedded directly in the fixture file.
- Don't rewrite a float formula (like `linspace`) to its algebraically
  equivalent form if a parity test is failing by a ULP or two. Three
  spellings can each disagree with the reference's own in the last f32 bit.
  Match the reference's SPELLING first, then look at tolerances.
- Don't fall back to the generic Qwen2-VL pixel-budget defaults when a
  checkpoint's config seems to be missing one. This family's
  `size: {shortest_edge, longest_edge}` is a factor of 16 different from
  the generic default, so a silent fallback resizes every image to a
  fraction of its intended resolution and produces a plausible-looking,
  degraded result. `from_preprocessor_config_json` refuses instead when
  neither spelling is present.
- Don't call `splice_image_placeholders` and `mrope_position_triples`
  separately by hand with independently-derived counts and grids. Use
  `splice_and_walk`: nothing else forces the placeholder count to agree
  with the grid, and a mismatch there succeeds silently while producing a
  wrong-length placeholder run and misplaced position spans.
