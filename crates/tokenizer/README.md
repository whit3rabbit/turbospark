# turbospark-tokenizer

Tokenizer wrapper around Hugging Face `tokenizers` (`MfTokenizer`), chat dialect resolution (Gemma 4, ChatML/Qwen, DeepSeek-V4), chat template rendering (text-only and `minijinja` + `pycompat`), streaming detokenization (`StreamingDetokenizer`), stop condition matching (`StopMatcher`), tool call DSL parsers, and streaming structured decoder (`StructuredDecoder`).

Downstream workspace crates import this package via the `tokenizer` alias:

```toml
[dependencies]
tokenizer = { package = "turbospark-tokenizer", path = "../tokenizer" }
```

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Key Modules

- `dialect.rs`: Resolves dialect special tokens and chat formatting rules for supported model families.
- `chat_template.rs`: Built-in text chat template rendering.
- `jinja_chat_template.rs`: Jinja template engine wrapper (`minijinja` + `pycompat`) for rendering `chat_template.jinja`.
- `detokenizer.rs`: `StreamingDetokenizer` for incremental UTF-8 token decoding.
- `stop_matcher.rs`: `StopMatcher` for evaluating stop sequences and EOS token sets.
- `structured_decoder.rs`: `StructuredDecoder` for streaming JSON and structured output parsing.
- `tool_call/`: Dialect-specific tool call DSL parsers (Gemma, Qwen, DeepSeek).

## Development & Test Commands

```sh
# Run unit and integration tests for turbospark-tokenizer
cargo test -p turbospark-tokenizer
```

## Crate Gotchas

1. **Dynamic Added Token IDs in Test Fixtures**: Test fixtures embed placeholder added token IDs in their `added_tokens` JSON lists. The `tokenizers` loader renumbers added tokens sequentially starting after the base vocabulary. Always resolve token IDs at runtime from a loaded `MfTokenizer` (using `token_to_id`, `end_of_turn_id`) rather than hardcoding numbers from fixture JSON.
