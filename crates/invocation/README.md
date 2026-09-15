# turbospark-invocation

Pure CLI argument parsing, command-line request assembly (`InvocationRequest`), option definitions (`OPTIONS`), diagnostic error formatting, typed failures (`ParseFailure`), and usage rendering.

Performs no filesystem, environment, or process I/O.

Downstream workspace crates import this package via the `invocation` alias:

```toml
[dependencies]
invocation = { package = "turbospark-invocation", path = "../invocation" }
```

## Purpose & Role

`turbospark-invocation` provides deterministic, IO-free command-line argument parsing for `turbospark-check` and peer tools. It validates CLI syntax, enforces argument dependencies and constraints, and packs flags into a typed `InvocationRequest` structure.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Pure parsing with no side effects or environment reads.

## Key Modules

- `options.rs`: Master declaration table of all supported command-line flags and options (`OPTIONS`).
- `parser.rs`: Pure two-pass flag parser and value translator converting string slices into validated data types.
- `request.rs`: Fully assembled `InvocationRequest` struct holding validated configuration (model path, sampling options, steering vectors, cache sizing, vision sidecar paths).
- `failure.rs`: `ParseFailure` enum for typed parse errors, missing arguments, and usage violations.
- `diagnostics.rs`: Formats user-friendly, terminal-rendered diagnostic errors with actionable remediation advice.
- `usage.rs`: Generates formatted help and option usage documentation.

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-invocation
cargo test -p turbospark-invocation
```

## Tests

This crate contains 5 integration test files in `tests/`:
- `parse_failures.rs`: Validates error handling on unrecognized flags, missing values, and conflicting arguments.
- `parse_ordering.rs`: Verifies parser invariance regardless of argument order.
- `parse_outcomes.rs`: Tests successful parsing across complex flag combinations (multi-vector steering, vision flags, sampling presets).
- `request_defaults.rs`: Confirms default values match documented runtime configuration defaults.
- `usage_and_status.rs`: Asserts option table invariants and help text rendering.

## Crate Gotchas

1. **Five-Place Rule for Flag Changes**: Adding or modifying a command-line flag requires synchronized changes across five distinct places:
   1. The `OPTIONS` table in `src/options.rs`.
   2. Parser dispatch `match` in `src/parser.rs` (first pass).
   3. Parser value extraction `match` in `src/parser.rs` (second pass).
   4. The `InvocationRequest` struct definition and field initializers in `src/request.rs`.
   5. Option-count assertions in `tests/usage_and_status.rs`.
   Missing any match arm triggers a compiler warning or runtime panic due to `unreachable!`.
2. **IO Isolation**: `InvocationRequest` does not check if model directories or message files exist; filesystem existence is verified downstream by `turbospark-cli` or `turbospark-runtime`.
