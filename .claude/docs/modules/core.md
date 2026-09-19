# Archived module guide: core

This is the detailed module guide moved from `crates/core/AGENTS.md`. The active entry point is the short guide at `crates/core/AGENTS.md`; this page preserves the deeper directory map, commands, gotchas, and evidence notes for on-demand reading.

# turbospark-core

Shared primitives, error types, runtime configuration, allowed numeric sets, automatic chunk-size resolution, and prefill chunking logic.

Downstream workspace crates import this package via the `foundation` alias (`foundation = { package = "turbospark-core", path = "../core" }`).

## Safety

- Contains no `unsafe` code, and `#![forbid(unsafe_code)]` is set in `lib.rs`.

## Directory & File Structure

```
crates/core/
+-- Cargo.toml              # Crate manifest declaring package turbospark-core
+-- src/
|   +-- lib.rs              # Library root re-exporting core modules
|   +-- primitives.rs       # Shared primitive types (TokenId = i32, LogitValue, LogitsView)
|   +-- runtime_config.rs   # Allowed numeric sets, documented defaults, const assertions
|   +-- chunk_sizing.rs     # Automatic chunk-size resolution algorithm
|   +-- prefill.rs          # Prefill chunking primitives and iterator logic
|   \-- steering.rs         # SteeringMode: the directional-steering edit's four modes
\-- tests/
    +-- chunk_sizing.rs     # Unit tests for chunk-size resolution
    \-- runtime_config.rs   # Unit tests for the allowed-set contract
```

## Key Modules

- `primitives.rs`: Defines workspace-wide primitives including `pub type TokenId = i32` and `LogitValue`.
- `runtime_config.rs`: Allowed numeric sets and documented defaults (`ALLOWED_CACHE_SLOTS = [8, 16, 24, 32, 48, 64, 96, 128]`, `ALLOWED_CHUNK_SIZES = [32, 64, 128, 256, 512, 1024, 2048, 4096]`), plus `const fn` helpers and `const _: () = assert!(...)` blocks that pin each default inside its own set and each set sorted ascending, at compile time.
- `chunk_sizing.rs`: Implements 3-state resolution rule turning input prompt lengths into allowed chunk sizes.
- `prefill.rs`: Handles splitting long input token sequences into executable prefill chunks; also the home of `MAX_CHUNK_TOKENS`, derived from `ALLOWED_CHUNK_SIZES`'s own maximum.
- `steering.rs`: `SteeringMode` (`Ablate` / `Add` / `Clamp` / `Renorm`), the
  four edits
  the directional-steering kernel applies to a residual stream row. **`Renorm`
  is `Ablate` followed by a rescale of the row back to its original `||x||`**,
  and it attacks `docs/OBLITERATION.md`'s collapse from the opposite side to
  the alpha ceiling that page derives: the ceiling AVOIDS the damage, this
  REPAIRS it. Two things about it are worth knowing before touching either
  implementation. It needs NO second reduction and NO second pass -- `||x||^2`
  fuses into the loop that already computes the coefficient, and the POST-edit
  norm is analytic (`||x'||^2 = ||x||^2 - alpha*(2 - alpha)*c_hat^2`, exactly,
  because the projection is orthogonal). And its rescale factor is exactly
  `1.0` in the other three modes, where `1.0 * v == v` in IEEE-754, so it
  lands in the SAME write loop and leaves them bit-identical rather than
  merely within tolerance -- which is what the real-model null control
  confirms. `can_grow()` is FALSE for it, like `Ablate` and unlike the other
  two: it restores a magnitude the row already carried, so no element can
  exceed a value that was already representable. **`alpha * (2 - alpha)`
  equals `alpha` at exactly 1.0**, so any test of the rescale at full strength
  alone is blind to the difference between them -- and full strength is the
  natural value to reach for, since making it usable is the mode's whole
  purpose. Both parity suites sweep alpha for that reason. It lives
  in this leaf crate rather than beside either implementation because it is
  the one thing both of them name and NEITHER can reach the other:
  `turbospark_compute::steering` is the numerical contract and
  `turbospark_gpu::encode_steer_direction` selects on it, and `crates/gpu`
  carried `turbospark-compute` as a DEV-dependency only when this was
  written -- TurboQuant's KV-cache quantization (`673341e`) later added it as
  a real dependency for the unrelated `kv_quant_tables.rs`, so the
  unnameable-from-the-dispatch-module argument no longer holds structurally,
  though the enum still lives here and nothing outside `compute` consumes it
  today. Its discriminants
  are the wire values the MSL kernel's `kSteerMode*` constants switch on, so a
  reorder here silently swaps two edits that both decode fluently; the four
  parity cases (`steer_ablate_matches_compute_reference`,
  `steer_add_matches_compute_reference`, `steer_clamp_matches_compute_reference`,
  `steer_renorm_matches_compute_reference`) in
  `crates/gpu/tests/utility_and_pass.rs` are what catch that
  across the boundary, and `steering_mode_codes_are_pinned` catches it on this
  side.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-core
cargo test -p turbospark-core
```

## Crate Gotchas

1. **Numeric Setters Panic on Invalid Inputs**: Runtime configuration numeric setters abort construction by panicking when a value is outside its allowed set (`ALLOWED_CACHE_SLOTS`, `ALLOWED_CHUNK_SIZES`). This is an intentional fatal precondition failure: there is no `Result`-returning variant and no clamping. Callers accepting unvalidated input must validate first or wrap with `std::panic::catch_unwind`. Read the allowed values from the const arrays in `src/runtime_config.rs` rather than re-hardcoding the literals. Automatic chunk-size resolution (the three-state rule turning an unknown-or-known input length into one allowed chunk size) lives in `src/chunk_sizing.rs` and reads those same constants rather than redeclaring them.
2. **Token Interchange Width**: Token IDs cross crate boundaries as signed 32-bit integers (`pub type TokenId = i32`). Keep this interchange width when wiring downstream crates.
3. **Workspace Import Alias**: Downstream crates import `turbospark-core` as `foundation`. Always use `foundation::...` when importing from core in other crates.
