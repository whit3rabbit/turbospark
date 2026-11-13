# turbospark-selection (src)

## Overview
Token selection after the producer returns logits: penalties, shaping (temperature, top-k, top-p, min-p), truncation, a shaped distribution, and deterministic seeded draws. Sampling owns normalization.

## Interface
- Selection entry points: [choose.rs:select], [choose.rs:shaped_distribution].
- Shaping config and errors: [shaping.rs:ShapingConfig] (builder `new`, `with_presence_penalty`, `with_frequency_penalty`, `with_min_p`), [shaping.rs:SelectionError], [shaping.rs:DEFAULT_REPETITION_PENALTY], [shaping.rs:MAX_TOP_K].
- Distribution: [distribution.rs:ShapedDistribution] (`new`, `point_mass`, `support`, `probability`, `draw`), and [distribution.rs:residual] (rejection-sampling correction distribution from p and q; `None` when the residual mass underflows).
- Penalties: [penalty.rs:apply_repetition_penalty], [penalty.rs:apply_presence_and_frequency_penalty].
- Truncation and ranking: [truncation.rs:softmax], [truncation.rs:rank_indices], [truncation.rs:rank_top_k], [truncation.rs:rank_indices_u32_into], [truncation.rs:rank_top_k_u32_into], [truncation.rs:truncate_by_probability_mass], [truncation.rs:truncate_by_rank], [truncation.rs:truncate_by_min_p].
- The live path is [choose.rs:shape_candidates], which inlines softmax, top-p, and min-p and uses only `rank_indices_u32_into` and `rank_top_k_u32_into`. `softmax`, `rank_indices`, `rank_top_k` and the `truncate_by_*` helpers are not called by `select`; they are reference implementations that tests exercise, so a change to one must be mirrored in the other. tests/reference_drift.rs composes the reference helpers and requires the live support from `shaped_distribution` to match (penalties off).
- Seeded derivation: [derive.rs:derive_step_value], [derive.rs:to_unit_interval].

## Invariants
- `ShapingConfig::new` validates temperature (finite, non-negative), top_k (0..=MAX_TOP_K), top_p (in (0, 1]), and repetition penalty (finite, > 0). The builders validate the rest at call time: [shaping.rs:ShapingConfig::with_presence_penalty] and [shaping.rs:ShapingConfig::with_frequency_penalty] (finite, in [-2, 2]), [shaping.rs:ShapingConfig::with_min_p] (finite, in [0, 1); 0 means disabled).
- Candidate domain must not be empty and scores must be finite [choose.rs:select].
- `ShapedDistribution::new` requires parallel, non-empty support and probabilities [distribution.rs:ShapedDistribution].
- Ranking uses a TOTAL order (NaN-safe) [truncation.rs:rank_indices].
- Pipeline order in [choose.rs:shape_candidates]: repetition penalty, presence/frequency penalty, softmax, rank (top-k cut when enabled), top-p against full-distribution mass, top-k cap on the survivors, min-p against `ranked[0]`, then reweight and draw. Pinned by tests/min_p.rs and tests/truncation_order.rs.
- Penalty scope: the repetition penalty covers the whole `history` (prompt included, each distinct id once). The presence/frequency penalty covers only the generated suffix `history[len - position..]`, with `position` clamped to `history.len()`. Callers must pass `position` as the generated-token count.
- Deterministic mode: `temperature == 0.0` ([shaping.rs:ShapingConfig::is_deterministic]) returns the argmax, ignoring penalties, truncation and seed. Ties go to the lowest index [choose.rs:argmax]. [choose.rs:shaped_distribution] returns the matching point mass.
- Fallbacks keep results defined: an empty survivor set yields the argmax; [choose.rs:reweight] returns uniform when the weights underflow to zero; draws that run past the tail return the last survivor.
- Panics, not errors: [distribution.rs:ShapedDistribution::new] asserts on mismatched or empty input, and [truncation.rs:assert_u32_domain] asserts when the domain exceeds u32::MAX.
- [penalty.rs:apply_repetition_penalty] clamps the penalty to the f32 range and saturates scores to f32::MIN/MAX so non-finite values cannot reach the softmax.
- Draws are deterministic for a given `Some(seed)` and position [derive.rs:derive_step_value], [distribution.rs:ShapedDistribution]. With `seed: None` the value is clock-derived and not reproducible.

## Side Effects
- Uses a thread-local scratch buffer to avoid per-call allocation [choose.rs:SCRATCH]. Buffers are vocabulary-sized (multi-megabyte at V=262,144) and live as long as the thread.
- Pure given a seed. Reads `SystemTime::now()` when `seed` is `None` [derive.rs:derive_step_value], so the README's bit-exact reproducibility claim holds only with `seed: Some`.

## Verification
- Run: `cargo test -p turbospark-selection`
