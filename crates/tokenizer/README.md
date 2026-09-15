# turbospark-tokenizer

Tokenizer wrapper around Hugging Face `tokenizers` (`MfTokenizer`), chat dialect resolution (Gemma 4, ChatML/Qwen, DeepSeek-V4, Mistral, Harmony/gpt-oss, Muse Glimmer, Llama-3, MiniMax), chat template rendering (text-only and `minijinja` + `pycompat`), streaming detokenization (`MfDetokenizer`), stop condition matching (`StreamingStopMatcher`), tool call DSL parsers, reasoning effort controls, and streaming structured decoder (`StructuredAssistantDecoder`).

Downstream workspace crates import this package via the `tokenizer` alias:

```toml
[dependencies]
tokenizer = { package = "turbospark-tokenizer", path = "../tokenizer" }
```

## Purpose & Role

`turbospark-tokenizer` handles all prompt encoding, token decoding, conversation formatting, and structured output extraction. It translates chat messages into concrete token sequences using the model's native Jinja template or hardcoded dialect fallback, parses incoming streams for stop tokens and tool calls, and handles multi-channel reasoning traces.

## Safety

- `#![forbid(unsafe_code)]` is enforced in `lib.rs`.
- Zero raw pointer manipulation or unsafe memory conversions.

## Key Modules

- `dialect/`: Resolves model-specific special tokens, channel delimiters, and formatting conventions for supported families:
  - `resolvers.rs` / `resolve.rs`: Architecture-based dialect resolution.
  - `minimax.rs`: MiniMax dialect special tokens.
  - `config.rs`: Generation configuration dialect mapping.
- `chat_template/`: Built-in text chat template rendering used as a fallback when `chat_template.jinja` is absent.
- `jinja_chat_template.rs`: High-fidelity Jinja template engine wrapper (`minijinja` + `pycompat`) executing checkpoint-provided `chat_template.jinja`.
- `jinja_compat.rs` & `jinja_date.rs`: Python-compatible Jinja runtime filters and date/time functions required by modern chat templates.
- `json_value.rs`: Lightweight JSON value representation used inside Jinja template contexts.
- `reasoning.rs`: Reasoning effort configuration (`low`, `medium`, `high`) and reasoning token formatting.
- `detokenizer.rs`: `MfDetokenizer` for incremental UTF-8 token decoding without garbled multibyte characters.
- `stop_matcher.rs`: `StreamingStopMatcher` for evaluating dynamic stop strings, EOS token sets, and maximum generation boundaries.
- `structured_decoder/`: `StructuredAssistantDecoder` for streaming JSON validation and structured schema constraints.
- `tool_call/`: Dialect-specific tool call DSL parsers (Gemma, Qwen, DeepSeek XML/JSON formats).

## Development & Test Commands

```sh
# Run all unit and integration tests for turbospark-tokenizer
cargo test -p turbospark-tokenizer
```

## Tests

This crate includes 14 comprehensive test suites in `tests/`:
- Dialect parity: `chatml_dialect.rs`, `deepseek_dialect.rs`, `harmony_dialect.rs`, `llama3_dialect.rs`, `minimax_dialect.rs`.
- Chat templates: `installed_template.rs`, `jinja_chat_template.rs`.
- Streaming and stop conditions: `generation_config_eos.rs`, `harmony_channels.rs`, `vision_markers.rs`.
- Structured decoding and tool calling: `structured_decoder.rs`, `tool_call_support.rs`, `tool_calls.rs`.
- Reasoning traces: `reasoning_effort.rs`.

## Crate Gotchas

1. **Dynamic Added Token IDs in Test Fixtures**: Test fixtures embed placeholder added token IDs in their `added_tokens` JSON lists. The `tokenizers` loader renumbers added tokens sequentially starting after the base vocabulary. Always resolve token IDs at runtime from a loaded `MfTokenizer` (using `token_to_id`, `end_of_turn_id`) rather than hardcoding numbers from fixture JSON.
2. **Bounded EOS Parsing**: Generation configurations in `generation_config.json` can declare multiple EOS token IDs as either integers or lists. The loader explicitly bounds and validates EOS candidate sets to prevent out-of-range token matches.
3. **Incremental Detokenization**: Multibyte UTF-8 characters (like emojis or non-Latin glyphs) may be split across multiple tokens. `MfDetokenizer` buffers incomplete UTF-8 sequences and emits text only when valid codepoints are formed.
