# mrefrust-core

Shared primitives, error types, runtime configuration, allowed numeric sets, automatic chunk-size resolution, and prefill chunking logic.

Downstream workspace crates import this package via the `foundation` alias (`foundation = { package = "mrefrust-core", path = "../core" }`).

## Safety

- Contains no `unsafe` code, though `#![forbid(unsafe_code)]` is not explicitly set in `lib.rs` (see root `AGENTS.md` Gotcha 9).

## Directory & File Structure

```
crates/core/
+-- Cargo.toml              # Crate manifest declaring package mrefrust-core
+-- src/
|   +-- lib.rs              # Library root re-exporting core modules
|   +-- primitives.rs       # Shared primitive types (TokenId = i32, LogitValue, LogitsView)
|   +-- runtime_config.rs   # RuntimeConfig, RuntimeConfigBuilder, ALLOWED_* const sets
|   +-- chunk_sizing.rs     # Automatic chunk-size resolution algorithm
|   +-- prefill.rs          # Prefill chunking primitives and iterator logic
|   \-- error.rs            # CoreError enum declaration
\-- tests/
    +-- chunk_sizing.rs     # Unit tests for chunk-size resolution
    \-- runtime_config.rs   # Unit tests for RuntimeConfig builder validation
```

## Key Modules

- `primitives.rs`: Defines workspace-wide primitives including `pub type TokenId = i32` and `LogitValue`.
- `runtime_config.rs`: Holds system parameters and allowed sets (`ALLOWED_CACHE_SLOTS = [8, 16, 24, 32]`, `ALLOWED_CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096]`).
- `chunk_sizing.rs`: Implements 3-state resolution rule turning input prompt lengths into allowed chunk sizes.
- `prefill.rs`: Handles splitting long input token sequences into executable prefill chunks.
- `error.rs`: Central error type for core initialization failures.

## Development & Test Commands

```sh
# Run unit and integration tests for mrefrust-core
cargo test -p mrefrust-core
```

## Crate Gotchas

1. **Numeric Setters Panic on Invalid Inputs**: Runtime configuration numeric setters abort construction by panicking when a value is outside its allowed set (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`). Callers accepting unvalidated input must validate first or wrap with `std::panic::catch_unwind`.
2. **Token Interchange Width**: Token IDs cross crate boundaries as signed 32-bit integers (`pub type TokenId = i32`). Keep this interchange width when wiring downstream crates.
3. **Workspace Import Alias**: Downstream crates import `mrefrust-core` as `foundation`. Always use `foundation::...` when importing from core in other crates.
