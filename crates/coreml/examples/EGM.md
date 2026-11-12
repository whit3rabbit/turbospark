# coreml examples

## Overview
Two command-line examples for inspecting Core ML models. They are developer tools, not public API.

## Interface
- [coreml_describe.rs:main]: prints inputs, outputs and states of a model; loads on CPU so no ANE compile is triggered. Usage: `cargo run -p turbospark-coreml --example coreml_describe -- <model> [function] [cache_dir]`; an empty or `-` function means none.
- [coreml_plan.rs:main]: prints per-device op placement from `MLComputePlan` (macOS 14.4+). Flags: `--units cpu|gpu|ane|all`, `--function`, `--cache`, `--ops`.

## Invariants
- `coreml_describe` loads with CPU compute units only [coreml_describe.rs:main].
- `coreml_plan` defaults to `--units all`, matching an app load [coreml_plan.rs:main].

## Side Effects
- A `.mlpackage` argument is compiled into the cache dir (default `.compiled` next to the package) [coreml_describe.rs:main], [coreml_plan.rs:main].
- `--units ane` or `all` can trigger a first Neural Engine compile [coreml_plan.rs:main].

## Verification
- Run: `cargo build -p turbospark-coreml --examples`
