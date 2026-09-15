# turbospark-core

Shared primitives, error types, runtime configuration, allowed numeric parameter sets, automatic chunk-size resolution, prefill chunking primitives, and directional steering mode definitions for the turbospark workspace.

Downstream workspace crates import this package via the `foundation` alias:

```toml
[dependencies]
foundation = { package = "turbospark-core", path = "../core" }
```

## Purpose & Role

`turbospark-core` is the root leaf crate of the workspace. It defines the universal primitive types and configuration invariants shared by all compute, runtime, GPU, and packaging crates, with zero internal workspace dependencies.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Pure, deterministic logic with no filesystem or network I/O.

## Key Modules

- `primitives.rs`: Defines workspace-wide primitive types:
  - `pub type TokenId = i32`: Signed 32-bit token identifier used across tokenizers, KV cache, and runtime dispatch.
  - `LogitValue = f32`: Raw floating-point logit type.
  - `LogitsView<'a>`: Non-owning slice view over raw unnormalized logit buffers.
- `runtime_config.rs`: Canonical allowed parameter sets and defaults:
  - `ALLOWED_CACHE_SLOTS = [8, 16, 24, 32, 48, 64, 96, 128]` (default: 16)
  - `ALLOWED_CHUNK_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]` (default: 512)
  - Enforces ascending order and valid defaults at compile time via `const` assertions.
- `chunk_sizing.rs`: Pure 3-state chunk-size resolution algorithm mapping prompt length to the nearest allowed chunk size.
- `prefill.rs`: Prefill chunking iterator and splitting logic for partitioning long prompt token sequences without heap allocation; defines `MAX_CHUNK_TOKENS = 4096`.
- `steering.rs`: `SteeringMode` enum (`Ablate`, `Add`, `Clamp`, `Renorm`), the directional steering edit modes shared between CPU reference kernels (`turbospark_compute::steering`) and Metal dispatch (`turbospark_gpu::encode_steer_direction`).

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-core
cargo test -p turbospark-core
```

## Tests

- `tests/chunk_sizing.rs`: Verifies 3-state chunk-size resolution bounds, rounding, and edge cases.
- `tests/runtime_config.rs`: Validates allowed set boundaries, default values, and const constraint invariants.

## Crate Gotchas

1. **Allowed Sets, Not Panicking Setters**: Runtime configuration parameters are checked against explicit sorted allowed sets (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`) rather than open integer ranges.
2. **Token Interchange Width**: Token IDs are signed 32-bit integers (`pub type TokenId = i32`). Negative values are reserved for sentinel error codes or unassigned positions.
3. **Workspace Import Alias**: Workspace crates import `turbospark-core` using `foundation` as the dependency alias (`foundation::...`).
4. **Steering Renorm Invariant**: `SteeringMode::Renorm` projects out the steered direction and then rescales the residual row back to its original L2 norm. The norm repair is computed analytically without a second reduction pass.
