# turbospark-core

Shared primitives, error types, runtime configuration, allowed numeric parameter sets, automatic chunk-size resolution, and prefill chunking primitives for the turbospark workspace.

Downstream workspace crates import this package via the `foundation` alias:

```toml
[dependencies]
foundation = { package = "turbospark-core", path = "../core" }
```

## Safety

- Contains no `unsafe` code.

## Key Modules

- `primitives.rs`: Defines workspace-wide primitive types including `pub type TokenId = i32`, `LogitValue`, and `LogitsView`.
- `runtime_config.rs`: System parameters and allowed parameter sets (`ALLOWED_CACHE_SLOTS = [8, 16, 24, 32, 48, 64, 96, 128]`, `ALLOWED_CHUNK_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]`).
- `chunk_sizing.rs`: Automatic three-state chunk-size resolution algorithm mapping prompt length to concrete allowed chunk size.
- `prefill.rs`: Prefill chunking primitives and chunk iterator logic for splitting long input token sequences.
- `steering.rs`: `SteeringMode` (`Ablate`/`Add`/`Clamp`/`Renorm`), the directional-steering edit's four modes, shared by the CPU reference (`turbospark_compute::steering`) and the Metal dispatch (`turbospark_gpu::encode_steer_direction`).
- `error.rs`: Central `Error` enum declaration.

## Development & Test Commands

```sh
# Run unit tests for turbospark-core
cargo test -p turbospark-core
```

## Crate Gotchas

1. **Numeric Setters Panic on Invalid Inputs**: Runtime configuration setters panic when values are outside their allowed constant sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`). Callers taking unvalidated input must validate first or catch panics with `std::panic::catch_unwind`.
2. **Token Interchange Width**: Token IDs cross crate boundaries as signed 32-bit integers (`pub type TokenId = i32`).
3. **Workspace Import Alias**: Downstream crates import `turbospark-core` using the `foundation` alias (`foundation::...`).
