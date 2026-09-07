# turbospark-invocation

Pure CLI argument parsing, command-line request assembly (`InvocationRequest`), option definitions (`OPTIONS`), diagnostic error formatting, typed failures (`ParseFailure`), and usage rendering.

Performs no filesystem, environment, or process I/O.

Downstream workspace crates import this package via the `invocation` alias:

```toml
[dependencies]
invocation = { package = "turbospark-invocation", path = "../invocation" }
```

## Key Modules

- `options.rs`: Table of supported command-line options (`OPTIONS`).
- `parser.rs`: Pure command-line flag parser and option value translator.
- `request.rs`: `InvocationRequest` structure holding validated command parameters.
- `failure.rs`: `ParseFailure` enum for typed parse and usage errors.
- `diagnostics.rs`: Formats user-facing diagnostic error messages.
- `usage.rs`: Renders command-line usage documentation.

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-invocation
cargo test -p turbospark-invocation
```

## Crate Gotchas

1. **Five-Place Rule for Flag Changes**: Adding or modifying a command-line flag requires updating five distinct places:
   1. The `OPTIONS` table in `src/options.rs`.
   2. Parser dispatch `match` in `src/parser.rs` (first pass).
   3. Parser value extraction `match` in `src/parser.rs` (second pass).
   4. The `InvocationRequest` struct definition and literal instantiation.
   5. Option-count assertions in `tests/usage_and_status.rs`.
   Missing a match arm leads to a runtime panic due to `unreachable!`.
