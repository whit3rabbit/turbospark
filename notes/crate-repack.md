---
uuid: "b5145f1e-69f3-4604-aa3a-94b2707d05e4"
title: "turbospark-repack"
summary: "Parses safetensors and GGUF headers, streams checkpoints over ranged HTTP, quantizes to INT4/INT8, and writes the .gturbo install format. #![forbid(unsafe_code)]"
tags: ["crate", "repack"]
source: "crates/repack/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-repack do?

It is the intake and packaging layer: pure safetensors and GGUF header
parsers, a ranged HTTP downloader (`ranged_download/`), the INT4/INT8
quantization repack pipelines, the `.gturbo` directory writer
(`gturbo_writer/`), per-family checkpoint converters (`gemma4_checkpoint/`,
`hf_checkpoint.rs`, `gguf_checkpoint/`), and every synthetic install builder
the rest of the workspace tests against. `#![forbid(unsafe_code)]` is
enforced. It streams multi-gigabyte checkpoints a layer at a time and never
writes the whole thing to disk at once.

## Don't

- Don't assume the ranged downloader's speed comes from concurrency alone.
  Every checkpoint URL here redirects to a CDN bridge that speaks HTTP/2,
  and `reqwest` will multiplex all concurrent chunk GETs onto ONE
  connection unless the client forces `http1_only()`. Concurrency only
  helps a walk whose tensors exceed the chunk cap, which in practice means
  MoE routed-expert tensors, not dense ones.
- Don't make a new family-extension manifest field conditional in the
  writer. `arch_validation` resolves an omitted field against the GEMMA
  baseline no matter what family the manifest claims, so leaving one out
  makes that family's installs permanently unloadable.
- Don't trust a synthetic install's generated text or router behavior as
  correctness evidence. Synthetic weights are deterministic pseudo-random
  noise (real but semantically meaningless output), and synthetic MoE
  routers are near-uniform, so expert-slot permutation bugs are invisible
  on synthetic fixtures alone.
- Don't assume a checkpoint that repacks successfully will run. Whether the
  walk can PARSE a block type and whether the install can DECODE it are two
  separate gates (`model_io::validate_quant` against the manifest, and
  `RealForwardRunner::open` against the resident index), asserted in both
  directions by `gguf_install_refused.rs`.
- Don't re-run a full repack to verify a byte-transform fix (V-head
  ordering, a sign convention, a narrowing decision). Tensors sit at fixed
  offsets and lengths, and `open()` runs no checksum, so patching the
  install in place and reopening it is a seconds-long loop against a
  ~20-minute re-stream.
- Don't add a manifest field and stop after the writer and the validator.
  There's a third consumer, the manifest PEEKER (`manifest_peek.rs`,
  what `crates/cli` and `crates/bench` use to reconstruct an `ArchConfig`
  from an installed directory), and it fails silently rather than loudly:
  a real vision install once opened, decoded text correctly, and refused
  every image as headless because the peeker never learned the new fields.
- Don't treat a 429 or 5xx from the download the same as a 404. A missing
  range is permanent and should fail fast. A 429/5xx means "not now" and
  needs backoff instead. This walk cannot resume, so a rate limit treated
  as fatal restarts a 16 GB stream from zero.
