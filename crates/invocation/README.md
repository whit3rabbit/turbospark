# turbospark-invocation

Pure CLI argument parsing, command-line request assembly (`InvocationRequest`), option definitions (`OPTIONS`), exit-status and stream-routing diagnostics, typed failures (`ParseFailure`), and usage rendering.

Performs no filesystem, environment, or process I/O.

Downstream workspace crates import this package via the `invocation` alias:

```toml
[dependencies]
invocation = { package = "turbospark-invocation", path = "../invocation" }
```

## Purpose & Role

`turbospark-invocation` provides deterministic, IO-free command-line argument parsing for `turbospark-check` (the binary in `crates/cli`) and peer tools. It validates CLI syntax, enforces argument dependencies and constraints, and packs flags into a typed `InvocationRequest` structure.

## Safety

- The crate contains no `unsafe` code, but `lib.rs` does not set `#![forbid(unsafe_code)]`, so the compiler does not enforce this.
- Pure parsing with no side effects or environment reads.

## Key Modules

- `options.rs`: Master declaration table of all supported command-line flags and options (`OPTIONS`).
- `parser.rs`: Pure flag parser. One left-to-right token scan converts values into validated data types, then fixed-order whole-invocation checks run.
- `request.rs`: Fully assembled `InvocationRequest` struct holding validated configuration (model path, sampling options, steering vectors, cache sizing, vision sidecar paths).
- `failure.rs`: `ParseFailure` enum for typed parse errors, missing arguments, and usage violations.
- `diagnostics.rs`: Maps a parse outcome to a process exit status (success, or a distinct invalid-invocation status) and a stream routing decision (primary vs diagnostic stream).
- `usage.rs`: Generates formatted help and option usage documentation.

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-invocation
cargo test -p turbospark-invocation
```

## Tests

This crate contains 5 integration test files in `tests/`:
- `parse_failures.rs`: Validates error handling on unrecognized flags, missing required options, and conflicting arguments.
- `parse_ordering.rs`: Verifies ordering, precedence, and purity: literal value consumption, short-circuit position, whole-invocation check order, and repeatable parses.
- `parse_outcomes.rs`: Tests successful parsing across complex flag combinations (multi-vector steering, vision flags, sampling presets).
- `request_defaults.rs`: Confirms default values match documented runtime configuration defaults.
- `usage_and_status.rs`: Asserts option table invariants and help text rendering.

## Crate Gotchas

1. **Five-Place Rule for Flag Changes**: Adding or modifying a command-line flag requires synchronized changes across five distinct places:
   1. The `OPTIONS` table in `src/options.rs`.
   2. No-value flag `match` in `src/parser.rs` (the `if !decl.takes_value` branch).
   3. Value-taking flag `match` in `src/parser.rs` (after the value lookup).
   4. The `InvocationRequest` struct definition in `src/request.rs` and its field initializer in `src/parser.rs`'s `Success` construction.
   5. Option-count assertions in `tests/usage_and_status.rs`.
   A flag in `OPTIONS` with no `match` arm panics at runtime through `unreachable!`. The compiler does not catch it.
2. **IO Isolation**: `InvocationRequest` does not check if model directories or message files exist; filesystem existence is verified downstream by `turbospark-cli` or `turbospark-runtime`.
