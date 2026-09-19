# turbospark-invocation

Pure CLI argument parsing, command-line request assembly (`InvocationRequest`), options definition (`OPTIONS`), diagnostics, typed failures (`ParseFailure`), usage rendering, and outcome routing decisions.

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
|   +-- failure.rs          # ParseFailure enum for typed parse/usage errors
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
  It also carries the pure MIRRORS of types the engine owns -- `MaxContext`,
  `ExpertCacheSlots`, `PowerProfile`, and (with the load-guard work)
  `LoadGuard`. Each exists because this crate may not read a machine or an
  install and every one of those is a claim about one; `crates/cli` maps
  between the two spellings in exactly one place per type. A mirror is not a
  duplicate to be tidied away: deleting it would mean depending on
  `turbospark-model-io`, which is what keeps this crate pure.
- `failure.rs`: `ParseFailure` enum for typed parse or usage errors.
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

2b. **`--version` is a THIRD parse outcome, not a variant of `Help`.** Both
   exit 0 on the primary stream, but a caller printing the usage table where a
   version was asked for is its own wrong answer, so `ParseOutcome::Version`
   is a sibling and `stream_routing` gives it `render_version()`. It
   short-circuits at the token it is reached at, exactly as `--help` does, so
   whichever comes FIRST wins and neither needs `--model` to be present. The
   string comes from `CARGO_PKG_VERSION` through `usage::VERSION`: every crate
   here inherits `version.workspace = true`, which is what lets one pure
   library answer `--version` for all three binaries. A hand-maintained
   constant would be a second place to forget on a release, and the failure
   mode is a binary confidently reporting the wrong version.

2. **THREE flags take an `auto` keyword, and two of them default to it.** `--prefill-chunk` parses `auto` into `PrefillChunk::Auto` but defaults to `Fixed(DEFAULT_CHUNK_SIZE)`; `--expert-cache-slots` parses `auto` into `ExpertCacheSlots::Auto` and **defaults to it**, because a slot count that is right for one machine is wrong for the next and this crate may not look at either the machine or the install. Both enums cross into `crates/runtime` unresolved, which is the same division `PowerProfile` takes -- and note that enum is DUPLICATED rather than shared (this crate depends only on `foundation`, so `crates/cli` maps between the two spellings). Adding another such flag means another mapping, not a new dependency edge -- and `--reasoning` is now an additional duplicated enum (`ReasoningEffort`, duplicated here and in `tokenizer`, joined by `crates/cli`'s `map_reasoning_effort`). Unlike the two above it resolves to nothing downstream: its value is rendered into the PROMPT by the checkpoint's own chat template, so what it means is the checkpoint's business and this crate validates only the spelling.

   `--max-context` is the third, and it also defaults to `Auto`. It differs
   from the other two in taking an arbitrary positive integer rather than a
   member of a published allowed set: a context window is a per-token KV
   allocation, so every positive value is legal and the only real bound is
   what memory holds. Zero is refused rather than read as `auto`, since the
   flag already has a spelling for "you decide" and a window of zero admits
   no prompt.

   **This is the first `auto` on an axis that is NOT throughput-only.** The
   slot count cannot move a digest (output is byte-identical across
   8/16/24/32 since `crates/runtime/CLAUDE.md` Gotcha 8's fix), where a context window
   decides how much KV is allocated and how long a prompt is admitted. What
   licenses the sensing default anyway is the resolver's rule that an install
   declaring no trained context resolves to `DEFAULT_MAX_CONTEXT` -- which is
   every install written before that field existed, so nothing already on
   disk moves. `crates/model-io/src/context_policy.rs` owns it (re-exported
   from `crates/runtime`, not defined there).

   The consequence for a test: `request_defaults.rs` asserts `ExpertCacheSlots::Auto`, not a number, so the documented default is a POLICY. What that policy resolves to, and the floor guaranteeing it never resolves below the count that used to be hardcoded here, live in `crates/model-io/src/expert_cache_policy.rs` (also re-exported from `crates/runtime`).

3. **Some flags here are parsed, validated, printed and consumed by NOTHING.** `--rdadvise` round-trips through `InvocationRequest` and `main.rs`'s resolved-request block, and no binary reads it. Grep for `request.<field>` outside `main.rs` before building a new flag, or a `TURBOSPARK_*` seam, for something that may already have a surface -- chunked prefill was wired to an env var in 2026-08-16 and `--prefill-chunk` was found afterwards, in the printed block, in the same unwired state `--rdadvise` is still in.

   `--prefill-chunk` itself is no longer an example of this: it was wired 2026-08-26 (`crates/cli/CLAUDE.md` Gotcha 7). `resolve_chunk_tokens` in `crates/cli/src/generate/mod.rs` reads `request.prefill_chunk.resolved()` and feeds the chunked prefill path only when `session.runner.supports_chunked_prefill()` says the install's family can serve it, else falls back to the sequential path silently -- so finding an unconsumed flag is still not the end of the question, but the resolution here was a capability check rather than a new `PrefillChunk` off-state variant.
