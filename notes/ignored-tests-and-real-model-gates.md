---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e07"
title: "The #[ignore]d tests and real-model gates"
summary: "147 ignored tests are opt-in: checkpoint downloads, per-family memory oracles/quality gates, cross-engine KL dumps. Run --ignored, point TURBOSPARK_<FAMILY>_INSTALL_DIR at a real .gturbo"
tags: ["testing", "day-one"]
depends_on: ["b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e08"]
source: "docs/TESTING.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What are the #[ignore]d tests, and when do I run one?

They're opt-in because they need a real, multi-gigabyte model checkpoint or
take minutes: checkpoint downloads, per-family memory oracles and quality
gates, the cross-engine KL divergence dumps (against mlx-lm and
llama.cpp), and real-install behaviour gates like prefix-KV reuse.

```sh
cargo test --workspace -- --ignored     # runs ALL of them; some download GBs
```

Each one takes an env var naming a local `.gturbo` install directory, e.g.:

```sh
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-bench --test memory_oracle --release -- --ignored --nocapture
```

Full family list and per-family quirks (which context window, which
budget, which install each needs) live in AGENTS.md's command block and
`docs/TESTING.md`. The full env var table is `docs/ENV.md`.

## When do I need to run one?

- **A change could move memory or decode throughput** -> run that family's
  memory oracle.
- **A change could move numerics** (a kernel, the output head, quantization,
  the sampler) -> run that family's quality gate. Coherence-by-eye alone
  cannot see the few-percent drift a subtly-wrong quantization change
  produces.
- **A change touches the decode path, output head, KV cache, or a Metal
  encode loop** -> run the real-model smoke tests too (greedy AND sampled,
  see [[real-model-smoke-tests]]), per family the change touches, not once
  for the workspace. **"Touches" includes moving code.** A pure file split
  once shipped a Gemma-sandwich-norm bug into Qwen and produced a reference
  perplexity of 255,409 against a frozen 6.2536, with the whole workspace
  suite green.

## Don't

- Don't treat a passing standard suite as proof a numerics-affecting change
  is safe. It structurally cannot see this class of bug (see above).
- Don't run the quality-sensitivity test as routine verification. It's not
  part of the regular loop. Run it only when the quality gate's own
  credibility is in question (after changing the corpus, the scoring, or
  the quantization path itself).
- Don't toggle a `TURBOSPARK_*` env var between tests inside one test file.
  Each `tests/*.rs` file is one binary, and its `#[test]`s run as threads
  sharing that process's environment, and `std::env::set_var` is global to
  the file with no ordering guarantee between cases.
