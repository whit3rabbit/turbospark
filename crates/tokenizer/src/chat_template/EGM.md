# tokenizer chat_template

## Overview
Message types and the per-dialect fallback chat-template renderers. The checkpoint's own Jinja template is preferred by `apply_chat_template`; these renderers serve checkpoints with no template and DeepSeek, whose native tool chat is hand-composed strings.

## Interface
- Types: [mod.rs:Role], [mod.rs:Message] (`new`, `with_parts`, `image_count`), [mod.rs:ContentPart], [mod.rs:HistoricalToolCall], [mod.rs:FunctionDefinition].
- `MfTokenizer` methods defined here: `apply_chat_template`, `apply_chat_template_with_reasoning`, `apply_dialect_chat_template`, `encode_text_continuation`, `verify_image_markers`, `verify_bare_image_marker`, `verify_bare_image_placeholder`, `encode_deepseek_tool_chat` [mod.rs] (helpers in [deepseek.rs]).
- Fallback renderers (crate-private): [chatml.rs] (Qwen ChatML), [deepseek.rs] (V4 template and native tool chat), [gemma.rs], [llama3.rs], [mistral.rs].

## Invariants
- Per-dialect renderers are the FALLBACK; the installed checkpoint template wins when present [mod.rs] (module doc).
- Image marker verification fails with `InvalidVisionMarkers` when `vision_start == image_pad` or either id is absent from the vocabulary, then unless the template renders exactly one start and one pad for a single image [mod.rs:verify_image_markers].

## Side Effects
- Fallback renderers are pure. The checkpoint-template path registers `strftime_now`, which reads `TURBOSPARK_CHAT_DATE` or the system clock, so a template that calls it is not deterministic [jinja_date.rs:strftime_now].

## Verification
- Run: `cargo test -p turbospark-tokenizer`
