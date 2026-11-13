# turbospark-invocation (src)

## Overview
Pure command-line grammar for the engine: option table, parser, validated request, failure types, and a mapping from parse outcome to exit status and stream routing. It emits nothing and terminates nothing.

## Interface
- Parsing: [parser.rs:parse], [parser.rs:ParseOutcome] (Success, Help, Version, Failure).
- Option table: [options.rs:OPTIONS], [options.rs:OptionDecl].
- Request: [request.rs:InvocationRequest], [request.rs:Mode], [request.rs:ReadAheadMode], [request.rs:PowerProfile], [request.rs:ReasoningEffort], [request.rs:PrefillChunk], [request.rs:ExpertCacheSlots], [request.rs:ExpertResidency], [request.rs:Speculation], [request.rs:SpeculativeDrafter], [request.rs:MaxContext], [request.rs:LoadGuard], [request.rs:KvBits], defaults `DEFAULT_MAX_NEW`, `DEFAULT_MAX_CONTEXT`, `DEFAULT_TEMPERATURE`, `DEFAULT_TOP_K`, `MAX_TOP_K`, `DEFAULT_TOP_P`, `DEFAULT_REPETITION_PENALTY`, plus [request.rs:steering_knob].
- Failures: [failure.rs:ParseFailure] (UnknownOption, MissingValue, InvalidValue, MissingRequired, MutuallyExclusive, NoModeSelected).
- Diagnostics: [diagnostics.rs:ExitStatus] (`code`), [diagnostics.rs:StreamRouting], [diagnostics.rs:exit_status], [diagnostics.rs:stream_routing].
- Usage text: [usage.rs:VERSION], [usage.rs:render_version], [usage.rs:render_usage].

## Invariants
- A value-taking option consumes the very next token whatever its text; a help token short-circuits ahead of all whole-invocation checks; those checks run in one fixed order and only the first failure is reported [parser.rs:parse].
- Help and Version are distinct outcomes, both exit 0 on stdout [parser.rs:ParseOutcome].
- `--help` and `--version` act only when the scan reaches them: an earlier unknown option, missing value or invalid value wins over a later help or version token [parser.rs:parse].
- `--system` values are joined with a newline in supplied order. `--stop`, `--image` and the steering lists are repeatable and order-preserving: the i-th knob value pairs with the i-th `--steering` path, and a shorter knob list extends by its last value [parser.rs:parse, request.rs:steering_knob].
- `--vision-sidecar` is an opaque path; the literal `auto` is read as a path, not a keyword. A sub-1.0 `--top-p` with `--top-k 0` is accepted [parser.rs:parse].
- Only two exit statuses: Success is 0 and InvalidInvocation is 2 [diagnostics.rs:ExitStatus].
- `DEFAULT_MAX_CONTEXT`, the adaptive-cap bounds, `ALLOWED_SPECULATION_BLOCKS`, `SteeringMode`, the accepted sets `ALLOWED_CACHE_SLOTS` and `ALLOWED_CHUNK_SIZES`, and the prefill-chunk default `DEFAULT_CHUNK_SIZE` come from `foundation` (the turbospark-core crate); the other defaults are literals in this crate. Keep the literals in step with the CLI docs [request.rs:DEFAULT_MAX_NEW].
- Mode and pairing: `--model` is required (MissingRequired). Exactly one of `--prompt`, `--messages-file`, `--humanize-file` or `--chat` is required (NoModeSelected), and any two are MutuallyExclusive. `--system` is accepted only in chat mode (InvalidValue) [parser.rs:parse].
- Steering knobs are refused, not ignored: a `--steering-mode`, `--steering-scale` or `--steering-layers` without `--steering`, or `--steering-target` or `--steering-gate` given explicitly without it, more mode, scale or layers values than paths, or an inverted layer range fails the whole invocation [parser.rs:parse].
- A failure reaches the diagnostic stream as its `Debug` text followed by usage, not a formatted message [diagnostics.rs:stream_routing].

## Side Effects
- None; pure data mapping [diagnostics.rs:exit_status].

## Verification
- Run: `cargo test -p turbospark-invocation`
