# turbospark-selection

Candidate token selection (`select`) from candidate logit vectors under shaping configuration (temperature, top-k, top-p, min-p, repetition penalty, frequency and presence penalties, seed determinism, step position, rejection sampling, and distribution guards).

Downstream workspace crates import this package via the `selection` alias:

```toml
[dependencies]
selection = { package = "turbospark-selection", path = "../selection" }
```

## Purpose & Role

`turbospark-selection` converts raw unnormalized logit vectors emitted by the model's language head into the next decoded token ID. It executes token penalties, probability distribution truncation (top-k, top-p nucleus, min-p), temperature scaling, and seeded categorical sampling or greedy argmax selection.

## Safety

- The crate contains no `unsafe` code, but `lib.rs` does not set `#![forbid(unsafe_code)]`, so the compiler does not enforce this.
- Pure CPU computation with zero device-specific assembly or external side effects.

## Key Modules

- `shaping.rs`: `ShapingConfig` holding validated sampling parameters (temperature, top_k, top_p, min_p, repetition_penalty, frequency_penalty, presence_penalty, seed) and typed error `SelectionError`.
- `choose.rs`: Primary entry point `select`. Dispatches greedy argmax, categorical sampling, or rejection sampling.
- `penalty.rs`: Penalizes repeated tokens based on historical token frequency and occurrence.
- `truncation.rs`: Truncation algorithms including top-k partial ranking, nucleus top-p filtering, and min-p thresholding.
- `distribution.rs`: The materialized shaped distribution (`ShapedDistribution`), the speculative-decoding residual sampler (`residual`), and seeded inverse-transform draws over the surviving support.
- `derive.rs`: Derivation of the per-step draw value from the optional seed and step position (seeded reproducibility, clock fallback).

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-selection
cargo test -p turbospark-selection
```

## Tests

This crate contains 11 integration test files in `tests/`:
- `determinism.rs`: Validates deterministic sampling repeatability across fixed seeds.
- `distribution.rs`: Tests deterministic-selection behavior and temperature spread across candidates.
- `domain_guards.rs`: Verifies rejection of non-finite score entries and empty candidate domains, plus oversized top-k and extreme penalty saturation.
- `host_sampler_cost.rs`: Profiles host-side sampling latency by component at the real Gemma 4 vocabulary (V=262,144).
- `min_p.rs`: Tests min-p candidate truncation relative to top token probability.
- `penalty.rs` & `presence_frequency.rs`: Verifies repetition, presence, and frequency penalty mathematics.
- `rank_top_k.rs`: Validates partial-sorting top-k algorithms without full array sorting.
- `rejection_sampling.rs`: Tests the distributions rejection sampling consumes, verifying draw-for-draw consistency with `select`.
- `shaping_validation.rs`: Validates `ShapingConfig` constructor boundary constraints.
- `truncation_order.rs`: Verifies the truncation composition order: probability-mass truncation first against the full distribution, then the rank-based cap.

## Crate Gotchas

1. **Logits Input Contract**: `selection::select` accepts raw unnormalized LOGITS and computes softmax internally. Passing pre-normalized probabilities collapses the distribution toward uniform when temperatures differ from 1.0.
2. **Deterministic Step Seeding**: Selection uses a deterministic pseudorandom generator seeded per generation step. This guarantees bit-exact reproducible token sequences across identical runs.
3. **Hot-Path Partial Ranking**: In models with large vocabularies (e.g. Gemma 4 with V=262,144), full sorting of logits would dominate decode latency. `truncation.rs` uses partial ranking, a sequential kth-largest cut followed by sorting only the admitted set, to extract top-k candidates without a full sort.
