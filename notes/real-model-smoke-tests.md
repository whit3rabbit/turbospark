---
uuid: "b6f1a0a1-1e3a-4c1e-9c2a-1a2b3c4d5e08"
title: "Real-model smoke tests: greedy is not enough"
summary: "Run BOTH greedy (--temperature 0.0001 --top-k 1) and sampled (CLI defaults) generation on any decode/head/KV/Metal-encode change. Greedy alone cannot see a broken sampler"
tags: ["testing", "day-one"]
source: "AGENTS.md, docs/DEVELOPMENT.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## Why do I need to run two smoke tests, not one?

```sh
cargo build --release -p turbospark-cli
printf '[{"role":"user","content":"Explain how coastal wetlands reduce flood damage."}]' > /tmp/p.json

# 1. Greedy. Catches broken math.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 1 --temperature 0.0001 --top-k 1

# 2. SAMPLED, at CLI defaults (T=0.2, top-k 64, top-p 0.95). Catches
#    distribution bugs greedy cannot see.
./target/release/turbospark-check --model ~/models/gemma4.gturbo \
  --messages-file /tmp/p.json --max-new 400 --seed 20260721
```

Greedy generation is `argmax`, which is invariant under every monotone
transform of the distribution. A bug that only distorts the *shape* of the
distribution (not the ranking of the top token) is completely invisible to
greedy decoding: the output looks correct every time.

This actually happened. A producer that ran `softmax` twice (once in the
model head, once in the sampler) collapsed a 262K-vocab distribution to
near-uniform. Greedy still picked the right top token every time (softmax
is monotone), and top-p/top-k still ranked correctly. Only the temperature
reweight was destroyed, degenerating sampling into a coin flip among the
surviving top-k tokens. Only the sampled run could show it.

## Don't

- Don't run only the greedy smoke test and call a decode-path change
  verified. Run both.
- Don't use a bare `--prompt` for this. On an instruction-tuned model it
  babbles because the chat template is missing, which reads like a broken
  decode when it's actually a missing template. Use `--messages-file`.
- Don't run this once for the whole workspace. Run it per model FAMILY the
  change touches (`crates/runtime/src/families/<name>/`), since a
  family-specific bug (e.g. a norm convention difference) won't show on a
  different family's install.
