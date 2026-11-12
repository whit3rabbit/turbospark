# turbospark-window-fit

## Overview
Deterministic conversation-window fitting. Package `turbospark-window-fit`, library name `turbospark_window_fit` [Cargo.toml]. It has no dependencies. `crates/cli` and `crates/ffi` import it under the alias `window_fit` [crates/cli/Cargo.toml, crates/ffi/Cargo.toml]. Integration tests are `tests/fit_rules.rs` (eviction order, boundaries, edge cases) and `tests/outcome_contract.rs` (outcome accessors and determinism). Existing `README.md` and `AGENTS.md` here are human-maintained and are not modified by this file.

## Subdirectories & Boundaries
- src: [./src/EGM.md](./src/EGM.md) - `fit_conversation_window` and `WindowFitOutcome`.
