---
uuid: "398e0e03-758c-41bf-9612-33731fa700b5"
title: "turbospark-model-io"
summary: "Manifest parsing, ArchConfig validation, and the mmap resident weight index. A family baseline is per ARCHITECTURE and validated field by field, not filled in as a placeholder"
tags: ["crate", "model-io"]
source: "crates/model-io/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-model-io do?

Parses and validates `manifest.json` (`ArchConfig`, quantization scheme
checks), decodes the packed-expert layout for streamed MoE
(`PackedExpertsLayout`), reads the resident tensor index and wraps the
mmap'd weight buffer (`ResidentBuffer`, zero-copy), verifies SHA-256 and
install receipts, and holds the two sizing policies (`context_policy.rs`,
`expert_cache_policy.rs`) shared by `crates/runtime` and `crates/catalog`.
`unsafe` is restricted specifically to the `mmap` call in
`resident_buffer.rs`.

Architecture baselines (`arch_baselines/`) are one per `ModelFamily`
(Gemma 4, Qwen 3.6, DeepSeek-V4, `llama`/Mixtral, `qwen3moe`, `gptOss`,
`qwen35`), and `arch_validation.rs` compares an install's declared config
against its family's baseline field by field.

## Don't

- Don't add a family baseline as a placeholder "to get something building."
  `arch_validation` compares field by field against it, so an invented
  baseline validates real installs against fiction. A baseline is earned by
  a decode flow or a name table, never assumed ahead of one.
- Don't assume an omitted family-extension field (like `attnOutputGate` or
  `ropeNeoxSubdim`) means "off." A manifest that omits one is validated
  against GEMMA's value for it, whatever family it actually claims, because
  every optional field resolves via `.unwrap_or(gemma_defaults.<field>)`.
  The writer side has to emit these fields unconditionally for every
  non-Gemma family.
- Don't compare a manifest float with `==` after any float arithmetic.
  `arch_validation` uses `!=` against a parser accurate to only ~1 ULP, so
  an invented value like `32^-0.5` fails to round-trip where the real
  families' binary-fraction constants (1.0, 0.0625, 2^-4.5) are fine.
- Don't confuse the two `expert_stride` fields. `PackedExpertsLayout`'s is
  the model-wide MAXIMUM a slot is sized from. `LayerLayout`'s is what one
  layer's file is actually padded to. Address or size a layer with the
  second, or a mixed-quantization install (different layers at different
  widths) over-reads every layer to the size of the largest one.
- Don't move `LoadGuard::Relaxed`'s numbers casually. Every frozen memory
  peak in `docs/BENCHMARKS.md` and every `measured` row in `models.json`
  describes an engine budgeting at exactly today's reserve and context
  fraction. Changing the default doesn't fail a build or redden a gate. It
  just leaves every published number quietly describing a configuration
  the engine no longer opens with.
- Don't derive a new tier's numbers from first principles. State them
  RELATIVE to `Relaxed`, since nothing has independently measured them, and
  assert the tiers stay DISTINGUISHABLE as well as ordered (an ordering
  sweep alone stays green even if every tier collapses onto one number).
