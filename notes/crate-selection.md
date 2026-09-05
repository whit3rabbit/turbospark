---
uuid: "c119671d-ab85-476d-9d3b-0ed636eb50d1"
title: "turbospark-selection"
summary: "Token sampling: shaping, top-k/top-p truncation, penalties, choose. select() takes raw LOGITS, never pre-softmaxed probabilities"
tags: ["crate", "selection"]
source: "crates/selection/CLAUDE.md"
created: "2026-09-04"
updated: "2026-09-04"
---

## What does turbospark-selection do?

Candidate token selection (`select`, `select_from_logits`) from
per-candidate logits: temperature, top-k, top-p, min-p, presence/frequency
penalty, seed, and distribution guards (NaN/Inf, empty candidates). It runs
once per decoded token at the FULL vocabulary and sits entirely outside
`TURBOSPARK_PHASES=1` and every dispatch profiler, since those only cover
`LogitProducer::produce`, not what happens after it returns.

## Don't

- Don't pass pre-softmaxed probabilities into `select`. It applies softmax
  internally, so double-softmaxing collapses top-k/top-p toward uniform and
  silently ruins sampling while greedy generation still looks fine (softmax
  is monotone, so `argmax` doesn't notice).
- Don't assume `--temperature 0.0001` exercises the fast path. `is_deterministic`
  is `temperature == 0.0` exactly, so anything else, including a
  near-zero value meant to emulate greedy, runs the full sampled pipeline.
- Don't add a new whole-vocabulary pass here without measuring it. This
  crate WAS the entire 1.5x decode gap against the Swift original: a full
  sort in the old top-k step cost 18.9 ms/token at V=262144, more than the
  whole GPU forward pass, invisible to every profiler because nothing here
  is inside `produce`. Use `cargo test -p turbospark-selection --release
  --test rank_top_k -- --ignored --nocapture` to price a change.
- Don't validate a throughput fix against the greedy smoke alone. The
  ranking win here is a function of `top_k`: at `--top-k 1` the old and new
  paths are both effectively O(n), so greedy shows no difference at all
  while the CLI's actual sampled defaults (top-k 64) see the whole 6.5-7.0x
  effect. Same failure mode as the sampler's own softmax bug (greedy tests
  a different code path here, not a weaker one).
- Don't treat `min_p() == None` and `min_p() == Some(0.0)` as equivalent.
  `with_min_p` treats `0.0` as disabled (`None`), not as a no-op filter
  value, so a downstream check for "is min-p active" has to match on the
  `Option`, not compare a float.
- Don't assume presence/frequency penalties see the whole context the way
  OpenAI's API does. They only see the GENERATED suffix of `history`, never
  the prompt (llama.cpp's convention, not OpenAI's), and min-p composes
  AFTER the top-k/top-p cut, never able to reintroduce an already-excluded
  candidate.
