# turbospark-tokenizer (src)

## Overview
Tokenizer wrapper, chat-template rendering (checkpoint Jinja first, dialect fallbacks second), streaming detokenizer, stop matching, reasoning-effort handling, and tool-call parsing for the supported `ChatDialect` set.

## Interface
- Public re-exports in [lib.rs]: `Message`, `Role`, `ContentPart`, `FunctionDefinition`, `HistoricalToolCall`, `MfDetokenizer`, `ChatDialect`, `MfTokenizer`, `ToolCallSupport`, `NO_SUCH_TOKEN_ID`, `TokenizerError`, `ToolCallParserError`, `render_generic_chat_template`, `CHAT_DATE_ENV`, `JsonValue`, `ReasoningEffort`, `ReasoningSupport`, `StreamingStopMatcher`, `StructuredAssistantDecoder`, `StructuredAssistantEvent`, `Tokenizer`, `TokenId`, and the tool-call parsers `DeepseekToolCallParser`, `GemmaToolCallParser`, `GlmToolCallParser`, `KimiToolCallParser`, `MistralToolCallParser`, `QwenToolCallParser`, `ParsedToolCall` (re-exported from [tool_call/mod.rs]).
- [detokenizer.rs:MfDetokenizer]: `push`, `flush`; handles byte-fallback tokens split across pushes. Each `push` re-decodes the whole stable id history, so total decode work is O(n^2) in generated length.
- [stop_matcher.rs:StreamingStopMatcher]: `push`, `finish`, `is_stopped`.
- [reasoning.rs:ReasoningEffort] (Off, Low, Medium, High, XHigh; `parse`, `as_str`, `level`, `enable_thinking`), [reasoning.rs:ReasoningSupport]; `MfTokenizer::reasoning_support` and `MfTokenizer::accepted_reasoning_levels` probe the template for supported levels [reasoning.rs:accepted_reasoning_levels].
- [jinja_chat_template.rs:render_generic_chat_template], `encode_generic_tool_chat`; crate-private [jinja_compat.rs:rewrite_generation_tags], [jinja_compat.rs:parenthesize_conditional_kwargs], [jinja_date.rs:strftime_now].
- [json_value.rs:JsonValue] (`parse`, `encoded`, `to_serde_json`, `gemma_schema_normalized`), [json_value.rs:MAXIMUM_DEPTH].
- [error.rs:TokenizerError], [error.rs:ToolCallParserError].
- Children: [chat_template](./chat_template/EGM.md), [dialect](./dialect/EGM.md), [structured_decoder](./structured_decoder/EGM.md), [tool_call](./tool_call/EGM.md).

## Invariants
- The reasoning level lives in the checkpoint template, not the weights; a level only matters if the template reads the relevant key [reasoning.rs:ReasoningSupport].
- JSON values nest at most 64 levels [json_value.rs:MAXIMUM_DEPTH].
- Token IDs are signed 32-bit ([crates/core/src/primitives.rs:TokenId], re-exported by [lib.rs]).
- A pinned chat date is YYYY-MM-DD or YYYY-MM-DDTHH:MM[:SS] (seconds default to 0), else rendering errors. Day is range-checked only as 1..=31, so 2026-02-31 is accepted [jinja_date.rs:parse_pinned].

## Side Effects
- Reads env var `TURBOSPARK_CHAT_DATE` when rendering `strftime_now`; when unset it reads the system clock in UTC, so output is non-deterministic [jinja_date.rs:strftime_now].
- Tokenizer loading reads files from the model directory [dialect/mod.rs:MfTokenizer].

## Verification
- Run: `cargo test -p turbospark-tokenizer`
