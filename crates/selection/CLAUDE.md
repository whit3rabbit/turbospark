# mrefrust-selection

Candidate token selection (`select`, `select_from_logits`) from per-candidate logit vectors under a validated shaping configuration (temperature, top-k, top-p, repetition penalty, seed, step position, distribution guards).

## Directory & File Structure

```
crates/selection/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root
|   +-- shaping.rs          # Main select and select_from_logits entry points
|   +-- choose.rs           # Categorical distribution sampling and argmax selection
|   +-- penalty.rs          # Frequency and repetition penalty application logic
|   +-- truncation.rs       # Top-k and top-p (nucleus) candidate truncation
|   \-- derive.rs           # Derived selection parameters and helper functions
\-- tests/
    +-- determinism.rs      # Verifies seed determinism across selection runs
    +-- distribution.rs     # Verifies temperature and logit distribution behavior
    +-- domain_guards.rs    # Edge case guards (NaN/Inf logits, empty candidates)
    +-- penalty.rs          # Repetition penalty calculation unit tests
    +-- shaping_validation.rs# Parameter range validation unit tests
    \-- truncation_order.rs # Top-k and top-p truncation order unit tests
```

## Key Modules

- `shaping.rs`: Main entry points `select` and `select_from_logits`.
- `choose.rs`: Categorical sampling from probability distributions.
- `penalty.rs`: Repetition penalty application based on token history.
- `truncation.rs`: Top-k and top-p (nucleus) filtering logic.
- `derive.rs`: Derived selection parameters and helper functions.

## Development & Test Commands

```sh
# Run tests for mrefrust-selection
cargo test -p mrefrust-selection
```

## Crate Gotchas

1. **Logits in, Probs Out Internally**: `selection::select` accepts raw unnormalized LOGITS and applies softmax internally. Never pass pre-softmaxed probabilities into `select`. Passing probabilities causes double-softmaxing (`softmax(softmax(z))`), collapsing top-k/top-p distributions toward uniform and ruining sampling.
2. **Deterministic Parity**: Selection uses a deterministic pseudorandom generator seeded per run to ensure exact reproducibility across runs when a seed is provided.
3. **This crate runs per decoded token at the FULL vocabulary, and no profiler in this repo sees it.** `select` is called from `run_raw_completion` after `LogitProducer::produce` returns, so it is outside every `MFERENCE_PHASES=1` bucket and every dispatch ranking (AGENTS.md Gotcha 23). At Gemma 4's V=262144 an O(n log n) step here costs more than the entire GPU forward pass: the full sort `rank_top_k` replaced was 18.9 ms/token against a ~25 ms pass, and was the whole of this port's one-time 1.5x decode gap against Swift (`docs/BENCHMARKS.md`). Treat any new whole-domain pass as a throughput change and measure it with `cargo test -p mrefrust-selection --release --test rank_top_k -- --ignored --nocapture`.
4. **Only a prefix of the ranked order is observable.** Probability-mass truncation keeps a prefix, and rank truncation caps that prefix, so with `top_k > 0` nothing past rank `top_k` can change the result -- which is what licenses the partial ranking. The full sort remains the fallback for `top_k == 0`. `tests/rank_top_k.rs` pins the usize reference and the `_u32_into` scratch variants to agree, ties and all; a tie-break difference would change sampled output while every throughput number improved, and greedy would not notice (Gotcha 16's failure mode).
5. **The hot path never materializes the probability vector.** `select` keeps thread-local scratch (`working`, `exps`, `ranked`) and ranks unnormalized `exp(s - max)` keys via `rank_top_k_u32_into`; division by the positive normalizer is monotone, so the order matches ranking probabilities, and `exps[i] / sum` reproduces the retired full vector's entries bit for bit where they are read (top-p cumsum, reweight). Any change here must keep the real-model sampled smoke md5-identical, not just the greedy one.
6. **`--temperature 0.0001` is not the fast path.** `is_deterministic` is `temperature == 0.0` exactly, so the repo's "greedy" smoke runs the full sampled pipeline. Only an exact `0` takes the `argmax` early return.
