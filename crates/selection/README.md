# turbospark-selection

Candidate token selection (`select`, `select_from_logits`) from candidate logit vectors under shaping configuration (temperature, top-k, top-p, repetition penalty, seed determinism, step position, and distribution guards).

Downstream workspace crates import this package via the `selection` alias:

```toml
[dependencies]
selection = { package = "turbospark-selection", path = "../selection" }
```

## Key Modules

- `shaping.rs`: Main entry points `select` and `select_from_logits`.
- `choose.rs`: Categorical distribution sampling and argmax selection.
- `penalty.rs`: Frequency and repetition penalty application based on token history.
- `truncation.rs`: Top-k and top-p (nucleus) candidate truncation algorithms.
- `derive.rs`: Derived selection parameters and helper functions.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-selection
cargo test -p turbospark-selection
```

## Crate Gotchas

1. **Logits Input Contract**: `selection::select` accepts raw unnormalized LOGITS and applies softmax internally. Passing pre-normalized probabilities causes double-softmaxing (`softmax(softmax(z))`), collapsing top-k/top-p distributions toward uniform.
2. **Deterministic Seed Parity**: Selection uses a deterministic pseudorandom generator seeded per generation step to guarantee reproducible sampling across runs.
3. **Hot-Path Optimization**: `select` runs per decoded token across the full vocabulary (V=262144 on Gemma 4). It avoids full sorting by partial-ranking top-k candidates and avoids allocating full probability vectors.
