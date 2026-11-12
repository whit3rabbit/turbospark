# tokenizer tool_call

## Overview
Tool-call body parsers per dialect, each turning model-emitted markup into [mod.rs:ParsedToolCall].

## Interface
- [mod.rs:ParsedToolCall], [mod.rs:MAXIMUM_BYTES], [mod.rs:deepseek_dsml_mark].
- Parsers (each with `new` and `parse`): [qwen.rs:QwenToolCallParser], [gemma.rs:GemmaToolCallParser] (also `is_representable_object_key`), [deepseek.rs:DeepseekToolCallParser] (`tool_calls_open_mark`, `tool_calls_close_mark`, `DSML_MARK`), [mistral.rs:MistralToolCallParser], [glm.rs:GlmToolCallParser] (`GLM_*_MARK` consts), [kimi.rs:KimiToolCallParser] (`KIMI_*_MARK` consts).

## Invariants
- A tool-call body is capped at 256 KiB [mod.rs:MAXIMUM_BYTES].
- Function names must be non-empty, at most 64 bytes, ASCII alphanumeric plus `_` and `-` [mod.rs:is_valid_function_name].
- Grammars are read off each checkpoint's own chat template (GLM-4.7-Flash, Kimi-K2.5) [glm.rs], [kimi.rs] (module docs).
- Parse failures return `ToolCallParserError` [error.rs:ToolCallParserError].

## Side Effects
- None (pure parsing).

## Verification
- Run: `cargo test -p turbospark-tokenizer`
