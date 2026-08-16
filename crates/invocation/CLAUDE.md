# turbospark-invocation

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
# Run unit and integration tests for turbospark-invocation
cargo test -p turbospark-invocation
```

## Crate Gotchas

1. **5-Place Rule for Adding Flags**: Adding a new command-line option touches five distinct places:
   1. The `OPTIONS` table in `src/options.rs`.
   2. Parser dispatch `match` in `src/parser.rs` (first pass).
   3. Parser value extraction `match` in `src/parser.rs` (second pass).
   4. The `InvocationRequest` struct definition and literal instantiation.
   5. `tests/usage_and_status.rs`'s hardcoded option-count assertion.
   *Note: Because parser matches end in `unreachable!`, missing a match arm results in a runtime panic rather than a compile error.*

2. **Two flags take an `auto` keyword, and only one of them defaults to it.** `--prefill-chunk` parses `auto` into `PrefillChunk::Auto` but defaults to `Fixed(DEFAULT_CHUNK_SIZE)`; `--expert-cache-slots` parses `auto` into `ExpertCacheSlots::Auto` and **defaults to it**, because a slot count that is right for one machine is wrong for the next and this crate may not look at either the machine or the install. Both enums cross into `crates/runtime` unresolved, which is the same division `PowerProfile` takes -- and note that enum is DUPLICATED rather than shared (this crate depends only on `foundation`, so `crates/cli` maps between the two spellings). Adding a third such flag means a third mapping, not a new dependency edge.

   The consequence for a test: `request_defaults.rs` asserts `ExpertCacheSlots::Auto`, not a number, so the documented default is a POLICY. What that policy resolves to, and the floor guaranteeing it never resolves below the count that used to be hardcoded here, live in `crates/runtime/src/expert_cache_policy.rs`.
