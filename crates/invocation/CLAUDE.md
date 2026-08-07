# mrefrust-invocation

Pure CLI argument parsing, command-line request assembly (`InvocationRequest`), options definition (`OPTIONS`), diagnostics, typed failures (`InvocationFailure`), usage rendering, and outcome routing decisions.

Performs no filesystem, environment, or process I/O.

## Directory & File Structure

```
crates/invocation/
+-- Cargo.toml              # Crate manifest
+-- src/
|   +-- lib.rs              # Library root
|   +-- options.rs          # OPTIONS table declaring supported command-line flags
|   +-- parser.rs           # Pure command-line flag parser and value translator
|   +-- request.rs          # InvocationRequest struct holding validated parameters
|   +-- failure.rs          # InvocationFailure enum for typed parse/usage errors
|   +-- diagnostics.rs      # Diagnostic error message formatter
|   \-- usage.rs            # Command-line usage renderer
\-- tests/
    +-- parse_failures.rs   # Unit tests for invalid flag combinations and syntax failures
    +-- parse_ordering.rs   # Unit tests verifying flag ordering independence
    +-- parse_outcomes.rs   # Unit tests verifying parsed options to InvocationRequest
    +-- request_defaults.rs # Unit tests checking default value assignments
    \-- usage_and_status.rs # Tests asserting option-count consistency & exit status codes
```

## Key Modules

- `options.rs`: Table of supported command-line options (`OPTIONS`).
- `parser.rs`: Pure command-line flag parser and value translator.
- `request.rs`: `InvocationRequest` structure holding validated parameters.
- `failure.rs`: `InvocationFailure` enum for typed parse or usage errors.
- `diagnostics.rs`: Formats diagnostic and error messages.
- `usage.rs`: Renders command-line usage text.

## Development & Test Commands

```sh
# Run unit and integration tests for mrefrust-invocation
cargo test -p mrefrust-invocation
```

## Crate Gotchas

1. **5-Place Rule for Adding Flags**: Adding a new command-line option touches five distinct places:
   1. The `OPTIONS` table in `src/options.rs`.
   2. Parser dispatch `match` in `src/parser.rs` (first pass).
   3. Parser value extraction `match` in `src/parser.rs` (second pass).
   4. The `InvocationRequest` struct definition and literal instantiation.
   5. `tests/usage_and_status.rs`'s hardcoded option-count assertion.
   *Note: Because parser matches end in `unreachable!`, missing a match arm results in a runtime panic rather than a compile error.*
