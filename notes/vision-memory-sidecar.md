---
uuid: "15fd89a7-e368-490a-9a60-5df7cd77f897"
title: "Vision memory sidecar"
summary: "A <alias>.gturbo-vision/ directory installs the qwen3_5 vision tower alone (RealForwardRunner::attach_vision_sidecar, turbospark-model pull-vision, --vision-sidecar), so a text-only trunk gains vision without a second 14+ GB download"
tags: ["vision", "runtime", "ffi", "memory"]
source: "docs/VISION.md"
created: "2026-09-06"
updated: "2026-09-06"
---

## What is the vision memory sidecar?

A standalone install format (`<alias>.gturbo-vision/`: a degenerate
zero-layer `manifest.json` plus a `vision_sidecar.json` record naming the
pairing family and hidden size) so the `qwen3_5` vision tower installs and
attaches to any compatible text-only trunk at runtime, instead of every
vision-capable install duplicating its 14+ GB trunk for ~0.9 GiB of tower.

- `RealForwardRunner::attach_vision_sidecar(dir)`: binds one after `open()`,
  before the tower's lazy first-image open, so a text-only session pays
  nothing until an image actually arrives.
- `turbospark-model pull-vision <row>`: fetches a tower directly from a repo,
  bypassing `catalog::gate`'s MLX-quantization check (a legitimately-BF16
  tower repo has no `quantization` block for that gate to find).
- `--vision-sidecar <PATH>` reaches `turbospark-check`, `turbospark-server`,
  and the FFI's `OpenOptions`.
- `RealForwardRunner::release_vision_tower()` (FFI: `ts_session_release_vision`)
  frees the open tower's streamer slots (or mapped-residency buffer),
  position table, and a sidecar's own resident weights/mmap, WITHOUT
  forgetting the sidecar attachment -- the next image reopens it from the
  same place.
- `resolve_vision_pixel_budget` (`crates/runtime/src/vision/budget.rs`)
  clamps the checkpoint's declared `max_pixels` to what this session's
  `--load-guard` tier can afford on top of committed weights and KV --
  a binary search against `VisionShape::scratch_bytes`.

Verified on real hardware: a pulled sidecar attached to a text-only
`qwen38-27b.gturbo` reproduces byte-for-byte identical transcription
against the combined `qwen38-27b-vision.gturbo` install.

## Don't

- Don't re-derive the load guard, physical memory, committed bytes, or KV
  bytes a second time for a pixel-budget clamp. Every front end threads the
  SAME four values `--max-context`/`ts_session_open` already resolved, or a
  hub's `recommend` verdict could silently disagree with its own `open`
  refusal (`crates/ffi/CLAUDE.md` Gotcha 12's rule, one capability over).
- Don't assume `vision_tower_parity.rs` or `vision_memory_oracle.rs` cover a
  sidecar-attached run. Both still read only the combined install's env
  var; a sidecar-aware arm is a scoped-out follow-up.
- Don't build a release test against a combined install (trunk + its own
  tower): it can't tell "reopened the sidecar" apart from "forgot the
  sidecar and silently ran the trunk's own tower instead", since both pass
  byte-identity trivially in the second case. Use a text-only trunk with a
  standalone sidecar attached, so a forgotten attachment refuses outright.
