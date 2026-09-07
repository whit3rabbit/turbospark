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
- `runtime_config.rs`: Allowed numeric sets and documented defaults (`ALLOWED_CACHE_SLOTS = [8, 16, 24, 32, 48, 64, 96, 128]`, `ALLOWED_CHUNK_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]`), pinned against each other at compile time by `const` assertions.
- `chunk_sizing.rs`: Automatic three-state chunk-size resolution algorithm mapping prompt length to concrete allowed chunk size.
- `prefill.rs`: Prefill chunking primitives and chunk iterator logic for splitting long input token sequences.
- `steering.rs`: `SteeringMode` (`Ablate`/`Add`/`Clamp`/`Renorm`), the directional-steering edit's four modes, shared by the CPU reference (`turbospark_compute::steering`) and the Metal dispatch (`turbospark_gpu::encode_steer_direction`).

## Development & Test Commands

```sh
# Run unit tests for turbospark-core
cargo test -p turbospark-core
```

## Crate Gotchas

1. **Allowed Sets, Not Panicking Setters**: Runtime knobs are validated against const allowed-value sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`) rather than through a builder. Compile-time `const` assertions pin each documented default inside its own set and each set sorted ascending.
2. **Token Interchange Width**: Token IDs cross crate boundaries as signed 32-bit integers (`pub type TokenId = i32`).
3. **Workspace Import Alias**: Downstream crates import `turbospark-core` using the `foundation` alias (`foundation::...`).
