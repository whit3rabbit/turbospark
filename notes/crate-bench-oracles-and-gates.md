---
uuid: "35d68e3f-6f31-4c93-9e9a-8b6f2ee54a01"
title: "turbospark-bench: oracle and quality-gate conventions"
summary: "Memory oracles and quality gates take the window/budget/vocab size from the install's own manifest, never a shared constant or the tokenizer. A per-family assistant prefix is required for perplexity to mean anything"
tags: ["crate", "bench", "testing"]
source: "crates/bench/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What are the rules for writing or reading a memory oracle / quality gate?

Each per-family oracle and quality gate in `crates/bench/tests/` is an
`#[ignore]`d test gated on a `TURBOSPARK_<FAMILY>_INSTALL_DIR` env var (see
[[ignored-tests-and-real-model-gates]] for how to run them, and
[[crate-bench]] for the harness they sit in). These are the conventions
that make their numbers comparable across families and runs.

## Don't

- Don't compare a memory ceiling across families without checking the
  context window each one ran at. A ceiling is a ceiling AT ONE window.
  Mistral's dense oracle runs at 8,192 (not the shared 4,096) because its
  `long-synthesis` case doesn't fit 4,096 tokens under that tokenizer, and
  raising the shared constant would move every other family's already-
  frozen peak. The window is printed beside every ceiling for this reason.
- Don't assume every family shares the 1,024-token generation budget
  either. It's a property of the MODEL, not its tokenizer. gpt-oss needs
  3,072 (its Harmony reasoning channel runs before the answer) and Muse
  Glimmer needs 2,048. A generation that hits the shared budget and stops
  on `maxTokens` produces an invalid, non-comparable run.
- Don't read a `vocab_size()` off the tokenizer for a real-model harness.
  Use `RealForwardRunner::vocab_size()` (the model's own padded head
  width) instead. The tokenizer's number is a per-dialect constant that
  was only ever correct while exactly one model used that dialect.
- Don't splice a reference answer straight into the assistant slot when
  measuring perplexity on a family whose chat framing is structured
  (Harmony's generation prompt ends at `<|start|>assistant`, and the next
  real token has to be `<|channel|>`). Skipping the family's assistant
  prefix reads perplexity in the hundred-thousands, indistinguishable from
  a genuinely broken model, even though the actual generations are fluent.
- Don't score perplexity over the whole prompt. Both supported families
  are instruction-tuned, so training masked the loss on user-turn text.
  Teacher-force only a fixed reference ANSWER into the assistant slot, the
  same way `quality_common` does.
- Don't freeze a digest for a template that reads the system clock. Pin
  `tokenizer::CHAT_DATE_ENV` before rendering, or the digest and the
  perplexity both expire at midnight and read like a numerics regression
  the next morning.
- Don't set a throughput floor off a single reading. Peak memory
  reproduces tightly run to run, but tok/s can move over 10% between two
  back-to-back runs on the same install. Take at least two readings and
  set the floor from the slower one.
