# mrefrust-core

Shared primitives, error types, runtime configuration, allowed numeric parameter sets, automatic chunk-size resolution, and prefill chunking primitives for the mrefrust workspace.

Downstream workspace crates import this package via the `foundation` alias:

```toml
[dependencies]
foundation = { package = "mrefrust-core", path = "../core" }
```

## Safety

- Contains no `unsafe` code.

## Key Modules

- `primitives.rs`: Defines workspace-wide primitive types including `pub type TokenId = i32`, `LogitValue`, and `LogitsView`.
- `runtime_config.rs`: System parameters and allowed parameter sets (`ALLOWED_CACHE_SLOTS = [8, 16, 24, 32]`, `ALLOWED_CHUNK_SIZES = [128, 256, 512, 1024, 2048, 4096]`).
- `chunk_sizing.rs`: Automatic three-state chunk-size resolution algorithm mapping prompt length to concrete allowed chunk size.
- `prefill.rs`: Prefill chunking primitives and chunk iterator logic for splitting long input token sequences.
- `error.rs`: Central `CoreError` enum declaration.

## Development & Test Commands

```sh
# Run unit tests for mrefrust-core
cargo test -p mrefrust-core
```

## Crate Gotchas

1. **Numeric Setters Panic on Invalid Inputs**: Runtime configuration setters panic when values are outside their allowed constant sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`). Callers taking unvalidated input must validate first or catch panics with `std::panic::catch_unwind`.
2. **Token Interchange Width**: Token IDs cross crate boundaries as signed 32-bit integers (`pub type TokenId = i32`).
3. **Workspace Import Alias**: Downstream crates import `mrefrust-core` using the `foundation` alias (`foundation::...`).
