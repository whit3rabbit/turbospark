---
uuid: "a59a0c0d-c823-4c52-a2dd-6a7f6884e1ed"
title: "turbospark-catalog"
summary: "Curated model table, header-only HF probe, and install driver behind turbospark-model. Sidecars are fetched and test-rendered before any weight byte streams"
tags: ["crate", "catalog"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e05"]
source: "crates/catalog/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-catalog do?

Backs the `turbospark-model` binary in `crates/cli`: the curated table
(`models.json`, embedded via `include_str!`), the header-only Hugging Face
probe (`probe/gguf.rs`, `probe/safetensors.rs`), and the install driver
(`install.rs`) that turns either into a `.gturbo` directory. `hf.rs` only
does KB-scale metadata endpoints (file lists, small GETs). Actual multi-GB
weight streaming belongs to `crates/repack::HttpRangeSource`, never here. A
second HTTP client here fetching a full weight body would silently lose the
range chunking, retry ladder, and the Xet bridge's `http1_only()` setting.
`#![forbid(unsafe_code)]`.

## Don't

- Don't move sidecar fetching next to the other file writes in `install.rs`.
  Sidecars are fetched, loaded, and test-rendered (a one-turn conversation
  encoded through `MfTokenizer`) BEFORE any weight byte streams. Qwen3.8-27B's
  bring-up once failed on a 404 for `merges.txt` after a 20-minute stream had
  already written a perfectly good install.
- Don't widen `catalog_network`'s 2% size tolerance to fit a hand-typed
  `download_bytes` figure. Run
  `cargo test -p turbospark-catalog --test catalog_network --release --
  --ignored --nocapture` after editing a row and paste its own number back.
  It's the only fingerprint a `main`-pinned GGUF row has.
- Don't let a probe gate inherit a normal caller's convenient parser
  default. `repack::parse_gemma4_quantization` defaults to 4-bit when
  `config.json` carries no `quantization` block, correct for its usual
  callers and catastrophic for a probe whose whole question is whether a
  checkpoint is quantized at all. `probe/safetensors.rs` checks the key's
  presence separately, before parsing.
- Don't let a later probe gate overwrite an earlier refusal.
  `ProbeReport::refuse` keeps the FIRST one, because gates run
  cheapest-and-most-fundamental first. A second true-but-useless refusal
  (e.g. "no sidecars" after "architecture unsupported") sends the reader
  chasing the wrong fix.
- Don't trust a probe's "no chat template found" verdict on a gated Hugging
  Face repo without `HF_TOKEN` set. `get_optional`'s pattern match discards
  a 401 the same way it discards a genuine absence, so both print the
  identical warning. Export `HF_TOKEN` before believing that verdict.
- Don't read a measured peak (`fit.rs`'s `counted`) as portable across
  context or slot count. `counted` (slot cache + KV) and `mapped` (the whole
  install on disk) answer different questions, and even `counted` is only
  valid at the exact `(context, slots)` pair it was measured at.
