# turbospark-core (src)

## Overview
Portable, dependency-light primitives shared by the runtime, invocation, CLI and server crates: token and logit types, allowed-value sets for runtime knobs, prefill chunk planning, steering modes, and the humanizer prompt contract.

## Interface
- Token and logit types: [primitives.rs:TokenId], [primitives.rs:LogitValue], [primitives.rs:LogitsView].
- Runtime knob sets and defaults: [runtime_config.rs:ALLOWED_CACHE_SLOTS], [runtime_config.rs:ALLOWED_SPECULATION_BLOCKS], [runtime_config.rs:ALLOWED_CHUNK_SIZES], [runtime_config.rs:DEFAULT_CHUNK_SIZE], [runtime_config.rs:DEFAULT_CACHE_SLOTS], [runtime_config.rs:DEFAULT_ADAPTIVE_CAP], [runtime_config.rs:MIN_ADAPTIVE_CAP], [runtime_config.rs:DEFAULT_MAX_CONTEXT], const fn [runtime_config.rs:max_of] (which [prefill.rs:MAX_CHUNK_TOKENS] derives from).
- Chunk sizing: [chunk_sizing.rs:InputLength], [chunk_sizing.rs:resolve_automatic_chunk_size].
- Prefill planning: [prefill.rs:MAX_CHUNK_TOKENS], [prefill.rs:PrefillChunkSpan], [prefill.rs:prefill_chunk_spans] (clamps `chunk_tokens` to `[1, MAX_CHUNK_TOKENS]`, empty plan for zero tokens, panics if `start_position + offset` overflows), [prefill.rs:PrefillChunkCommitState], [prefill.rs:PrefillError].
- Steering: [steering.rs:SteeringMode] (`as_u32`, `parse`, `as_str`, `can_grow`; derives `Default`, which is Ablate), [steering.rs:STEERING_MODE_NAMES].
- Humanizer: [humanizer.rs:HUMANIZER_INSTRUCTION], [humanizer.rs:HUMANIZER_SEPARATOR], [humanizer.rs:PROMPT_FINGERPRINT], [humanizer.rs:build_humanize_prompt], [humanizer.rs:audit_rewrite], [humanizer.rs:RewriteAudit], [humanizer.rs:Script], [humanizer.rs:HUMANIZER_CONTEXT], [humanizer.rs:default_max_new] (output budget clamped to 256..=2048, saturating), [humanizer.rs:REUSE_FLAG_THRESHOLD] (0.22), `RewriteAudit::warnings` (stderr report lines), sampling constants `HUMANIZER_TEMPERATURE/TOP_K/TOP_P/REPETITION_PENALTY`.
- Engagement flag: [engagement.rs:enabled], [engagement.rs:line], and the exported `engaged!` macro (callers use `foundation::engaged!`; prints `[engaged] NAME` to stderr once per call site when enabled).
- Module wiring: [lib.rs] declares chunk_sizing, engagement, humanizer, prefill, primitives, runtime_config, steering.

## Invariants
- ALLOWED_CACHE_SLOTS and ALLOWED_CHUNK_SIZES are non-empty, strictly ascending, and contain their defaults, enforced by const assertions [runtime_config.rs:ALLOWED_CACHE_SLOTS]. ALLOWED_SPECULATION_BLOCKS and the adaptive caps have no compile-time check.
- The state only tracks what the caller reports: the caller calls `mark_dirty` before writing a chunk and `mark_committed` after; if the chunk errors or panics the state stays dirty, and `require_clean` refuses further work until `reset()`. It detects no error itself [prefill.rs:PrefillChunkCommitState].
- `mark_dirty` requires a positive token count [prefill.rs:PrefillChunkCommitState].
- Steering mode integer encodings are fixed (Ablate 0, Add 1, Clamp 2, Renorm 3) [steering.rs:SteeringMode].
- The humanizer wrapper is byte-exact and its prompt hash is pinned by [humanizer.rs:PROMPT_FINGERPRINT].
- Automatic chunk size resolves within the allowed chunk size set [chunk_sizing.rs:resolve_automatic_chunk_size].

## Side Effects
- Reads env var `TURBOSPARK_ENGAGEMENT` once and caches it in a `OnceLock` [engagement.rs:enabled].
- Otherwise pure; no disk or network access. None beyond the above.

## Verification
- Run: `cargo test -p turbospark-core`
