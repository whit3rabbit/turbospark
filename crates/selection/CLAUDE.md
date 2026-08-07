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
