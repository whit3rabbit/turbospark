---
description: "Tokenizer loading, chat-dialect resolution, chat-template rendering, streaming detokenization, stop matching, and tool-call parsing for the supported model families."
icon: language
---

# tokenizer

`crates/tokenizer` (package `turbospark-tokenizer`) owns everything between
text and token ids: loading a Hugging Face tokenizer directory, resolving its
chat dialect and special-token contract, rendering prompts through the
checkpoint's own chat template, streaming detokenization, stop-string
matching, reasoning-effort controls, and tool-call parsing. It is pure Rust
with `#![forbid(unsafe_code)]`, and it is the piece that makes prompt
rendering part of model behavior rather than a string formatting detail.

## What it does

| Responsibility | Where |
| --- | --- |
| Tokenizer loading, dialect resolution, encode/decode | `src/dialect/` (`MfTokenizer`, `ChatDialect`) |
| Chat message types and per-dialect fallback renderers | `src/chat_template/` (`Message`, `Role`, `ContentPart`) |
| Jinja rendering of the checkpoint's own template | `src/jinja_chat_template.rs` (`render_generic_chat_template`) |
| Incremental detokenization | `src/detokenizer.rs` (`MfDetokenizer`) |
| Stop-string matching across chunk boundaries | `src/stop_matcher.rs` (`StreamingStopMatcher`) |
| Structured assistant output: reasoning channels and tool calls | `src/structured_decoder/` (`StructuredAssistantDecoder`) |
| Per-dialect tool-call parsers | `src/tool_call/` (`QwenToolCallParser`, `GemmaToolCallParser`, ...) |
| Reasoning effort and per-checkpoint support | `src/reasoning.rs` (`ReasoningEffort`, `ReasoningSupport`) |
| Error taxonomy | `src/error.rs` (`TokenizerError`, `ToolCallParserError`) |

The supported dialect set in `ChatDialect` covers Gemma, ChatML (Qwen),
DeepSeek (V3/V4 line and the V2 line), Mistral, Harmony (gpt-oss), Muse
Glimmer, Llama 3, Spark, MiniMax, GLM, and Kimi.

## Where it sits

`turbospark-tokenizer` depends on `turbospark-core` (aliased `foundation`)
and wraps the Hugging Face `tokenizers` crate plus `minijinja` for template
rendering. Nearly every surface consumes it: `runtime`, `cli`, `server`,
`catalog`, `ffi`, `bench`, and the `audio` and `image` crates, which resolve
their fixture token ids through a loaded tokenizer.

## Public API tour

Root re-exports from `crates/tokenizer/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `MfTokenizer::load_from_dir(&Path)` | `src/dialect/mod.rs` | Loads `tokenizer.json`, `tokenizer_config.json`, and `generation_config.json`; detects the dialect |
| `MfTokenizer::encode(text, add_bos)` / `decode(ids, skip_special)` | `src/dialect/mod.rs` | UTF-8 to `Vec<i32>` and back |
| `MfTokenizer::token_to_id` / `id_to_token` | `src/dialect/mod.rs` | Single-token resolution |
| `MfTokenizer::apply_chat_template(&[Message])` | `src/chat_template/mod.rs` | Prefers the checkpoint's own installed Jinja template; errors rather than falling back on a render failure |
| `MfTokenizer::accepted_reasoning_levels()` / `reasoning_support()` | `src/reasoning.rs` | What the installed template can express |
| `dialect`, `bos_id`, `eos_id`, `stop_token_ids`, `think_start_id`, `vocab_size`, ... | `src/dialect/mod.rs` | The resolved special-token contract (public fields) |
| `ChatDialect`, `ToolCallSupport`, `NO_SUCH_TOKEN_ID` | `src/dialect/mod.rs` | The framing dialect and whether its markup carries parseable tool calls natively |
| `render_generic_chat_template(tokenizer, messages, tools, add_generation_prompt, reasoning)` | `src/jinja_chat_template.rs` | Direct render through the installed template source |
| `Message`, `Role`, `ContentPart`, `FunctionDefinition`, `HistoricalToolCall` | `src/chat_template/mod.rs` | Chat inputs; `Message::with_parts` builds multimodal messages whose images are placeholder markers |
| `MfDetokenizer::new(&tokenizer)`, `push(id)`, `flush()` | `src/detokenizer.rs` | Streaming decode that reproduces full decode exactly once flushed |
| `StreamingStopMatcher::new(stops)`, `push`, `finish`, `is_stopped` | `src/stop_matcher.rs` | Withholds a partial stop match across chunks, then reports the stop |
| `StructuredAssistantDecoder`, `StructuredAssistantEvent` | `src/structured_decoder/mod.rs` | Splits tokens into `Content`, `Reasoning`, and `ToolCall` events per dialect |
| `ParsedToolCall` and the per-family parsers | `src/tool_call/` | `id`, `name`, `arguments` (`JsonValue`), and `arguments_json` |
| `ReasoningEffort` | `src/reasoning.rs` | `Off` (default), `Low`, `Medium`, `High`, `XHigh` |
| `JsonValue`, `TokenizerError`, `ToolCallParserError` | `src/json_value.rs`, `src/error.rs` | Minimal JSON values and the two error types |

## Examples

### Load a tokenizer and round-trip text

Adapted from `crates/tokenizer/tests/chatml_dialect.rs`, which loads a real
vendored BPE fixture.

```rust
use std::path::PathBuf;
use turbospark_tokenizer::{ChatDialect, MfTokenizer};

let dir = PathBuf::from("tests/fixtures/ChatMLTokenizer");
let tok = MfTokenizer::load_from_dir(&dir).unwrap();

// The dialect was detected from the tokenizer's own special tokens.
assert_eq!(tok.dialect, ChatDialect::ChatMl);

let ids = tok.encode("hello there", false);
let text = tok.decode(&ids, true);
assert_eq!(text, "hello there");

// Resolve a special token through the loaded table, never from JSON.
let eos = tok.token_to_id("<|im_end|>");
assert!(eos.is_some());
```

### Render a chat prompt

From `crates/tokenizer/tests/jinja_chat_template.rs`. Rendering goes through
the checkpoint's installed template; the fixture's Qwen ChatML template
frames turns as `<|im_start|>role ... <|im_end|>` and ends with the
assistant generation prompt.

```rust
use turbospark_tokenizer::{
    render_generic_chat_template, Message, MfTokenizer, ReasoningEffort, Role,
};

let tok = MfTokenizer::load_from_dir(&dir).unwrap();
let messages = vec![
    Message::new(Role::System, "be concise"),
    Message::new(Role::User, "hi"),
];
let rendered = render_generic_chat_template(
    &tok, &messages, &[], true, ReasoningEffort::Off,
).unwrap();

assert!(rendered.contains("<|im_start|>user\nhi<|im_end|>\n"));
// The fixture's template prefills a closed think frame after the
// assistant header at the default effort.
assert!(rendered.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
```

The exact ending depends on the installed template and the reasoning level
passed. Tool schemas join through the `tools` argument as
`FunctionDefinition` values, and `tok.apply_chat_template(&messages)` is the
shorter route at the default effort.

### Stream tokens through a detokenizer and stop matcher

Adapted from the streaming tests in `crates/tokenizer/tests/chatml_dialect.rs`.

```rust
use turbospark_tokenizer::{MfDetokenizer, MfTokenizer, StreamingStopMatcher};

let tok = MfTokenizer::load_from_dir(&dir).unwrap();

// Streaming decode reproduces full decode exactly, once flushed.
let ids = tok.encode("hello there friend", false);
let mut detok = MfDetokenizer::new(&tok);
let mut streamed = String::new();
for &id in &ids {
    streamed += &detok.push(id); // only complete pieces come out
}
streamed += &detok.flush();
assert_eq!(streamed, tok.decode(&ids, true));

// A partial stop match is withheld across chunks, then reported.
let mut matcher = StreamingStopMatcher::new(vec!["STOP".to_string()]);
let mut out = matcher.push("hello ST");
assert!(!matcher.is_stopped());
out += &matcher.push("OP world");
assert!(matcher.is_stopped());
assert_eq!(out, "hello "); // "STOP" was withheld, not emitted

// finish() releases text held short of a match.
let mut matcher = StreamingStopMatcher::new(vec!["STOP".to_string()]);
let mut out = matcher.push("hello wor");
out += &matcher.finish();
assert_eq!(out, "hello wor");
```

For structured output, wrap the same token stream in a
`StructuredAssistantDecoder::new(&tok, allowed_tools, id_generator,
&prompt_ids)` and match the returned `StructuredAssistantEvent::Content`,
`Reasoning`, and `ToolCall` values; `ChatDialect::tool_call_support()`
tells you whether the dialect's markup carries tool calls natively or only
through prompting.

## Gotchas

{% hint style="warning" %}
Resolve added-token ids from a loaded tokenizer (`token_to_id`,
`encode`), never from fixture JSON. The tokenizer library may renumber added
tokens relative to the JSON, and a hardcoded id silently targets the wrong
token. This is a workspace-wide safety invariant for token IDs.
{% endhint %}

{% hint style="warning" %}
`generation_config.json` is the authority for a checkpoint's full
end-of-sequence set. `tokenizer_config.json` carries only one `eos_token`
string, so a multi-stop checkpoint looks single-stop without it. Preserve
template whitespace, reasoning controls, tool delimiters, and assistant
prefixes exactly: prompt rendering is part of model behavior.
{% endhint %}

One more from the source: the entire Kimi K2 line ships `tiktoken.model`
with no `tokenizer.json`, so no K2 install is loadable by this engine today;
the dialect's test fixture carries the real added tokens with a disclosed
synthetic BPE body.

## Related pages

- [core](core.md)
- [Model Families](../models/model-families.md)
- [Tool Calling on the Local API](../command-line-and-local-api/call-the-local-api.md)
- [Audio Overview](../audio/overview.md)
