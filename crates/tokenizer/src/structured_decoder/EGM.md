# tokenizer structured_decoder

## Overview
Streaming assistant-output decoder: splits generated text and tokens into visible content and parsed tool calls, with one decoder arm per dialect.

## Interface
- [mod.rs:StructuredAssistantDecoder]: `new` (takes `prompt_ids`; starts in the Thought channel when the prompt ends inside an open think frame, [mod.rs:prompt_opens_thought]), `consume`, `consume_flushed_text`, `drain`, `finish`, `has_tool_calls`, `take_failed_span_text`.
- [mod.rs:StructuredAssistantEvent].
- Dialect arms (crate-private): [chatml.rs] (Qwen), [deepseek.rs] (DSML buffering), [glm.rs], [harmony.rs] (channels), [kimi.rs], [mistral.rs] (`[TOOL_CALLS]`), [muse.rs] (recipient frames).

## Invariants
- GLM and Kimi tool markers are ADDED but NOT SPECIAL tokens in the checkpoint table, so their arms match on text [glm.rs], [kimi.rs] (module docs).
- Mistral's tool format has one special token and no closing one [mistral.rs] (module doc).
- Muse Glimmer reasons on every turn regardless of `--reasoning`, so its arm is built unconditionally like Harmony's [muse.rs] (module doc).
- `finish` returns `ToolCallParserError` when a tool call is left incomplete, except Harmony and Mistral, which have no in-stream closing token: `finish` closes and emits their pending call, so a consumer that skips `finish` drops it [mod.rs:StructuredAssistantDecoder::finish].
- After any parser error the decoder stays failed and every later `consume` returns `Malformed` [mod.rs:StructuredAssistantDecoder::consume].

## Side Effects
- None (in-memory state machine).

## Verification
- Run: `cargo test -p turbospark-tokenizer`
