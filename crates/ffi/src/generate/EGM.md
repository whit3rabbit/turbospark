# ffi generate

## Overview
One generation turn over the C ABI: decode wire messages, render the chat template, tokenize and fit the window, attach vision, stream events, and report. Also the humanizer rewrite path.

## Interface
- Event kinds: [mod.rs:TS_EVENT_PREFILL] (0), [mod.rs:TS_EVENT_CONTENT] (1), [mod.rs:TS_EVENT_REASONING] (2), [mod.rs:TS_EVENT_TOOL] (3), [mod.rs:TS_EVENT_FINISH].
- Humanize: [humanize.rs:HumanizeOptions], [humanize.rs:AuditWire], [humanize.rs:HumanizeResult], [humanize.rs:output_budget].
- Prompt building: private helpers [prompt.rs:decode_messages], [prompt.rs:role_of], [prompt.rs:historical_call]; crate-private [prompt.rs:ToolOffer].
- Vision (crate-private): [vision.rs:VisionScope], [vision.rs:attach_images], [vision.rs:SCRIPTED_HAS_NO_TOWER].

## Invariants
- The humanizer is a RAW COMPLETION rewriter and must not go through the chat template path [humanize.rs] (module doc).
- Message roles are validated; an unknown role is an error, not a default [prompt.rs:role_of].

## Side Effects
- Streams tokens to a host callback while holding the session engine lock [mod.rs] (see header rule 4 in [../../include/EGM.md](../../include/EGM.md)).
- Vision attach may load or release a vision tower on the session [vision.rs:VisionScope].

## Verification
- Run: `cargo test -p turbospark-ffi`
