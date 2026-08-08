# turbospark-tokenizer

Tokenizer wrapper around HF `tokenizers` (`MfTokenizer`), chat dialect resolution (Gemma 4, ChatML/Qwen, DeepSeek-V4), chat template rendering (text-only and `minijinja` + `pycompat`), streaming detokenization (`StreamingDetokenizer`), stop condition matching (`StopMatcher`), tool call DSL parsers, and streaming structured decoder (`StructuredDecoder`).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/tokenizer/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root re-exporting MfTokenizer and dialect APIs
|   +-- dialect.rs                  # Resolves dialect special tokens and chat formatting rules
|   +-- chat_template.rs            # Built-in text chat template renderer
|   +-- jinja_chat_template.rs      # minijinja + pycompat wrapper rendering chat_template.jinja
|   +-- detokenizer.rs              # StreamingDetokenizer for incremental UTF-8 token decoding
|   +-- stop_matcher.rs             # StopMatcher for evaluating stop sequences and EOS token sets
|   +-- structured_decoder.rs       # StructuredDecoder for streaming JSON / structured output
|   +-- json_value.rs               # JSON value helper types for tool parameter encoding
|   +-- error.rs                    # TokenizerError enum definition
|   \-- tool_call/                  # Dialect-specific tool call DSL parsers
|       +-- mod.rs                  # Module root for tool call parsers
|       +-- gemma.rs                # Gemma tool call DSL parser
|       +-- qwen.rs                 # Qwen tool call DSL parser
|       \-- deepseek.rs             # DeepSeek tool call DSL parser
\-- tests/
    +-- chatml_dialect.rs           # ChatML dialect encoding & detokenization unit tests
    +-- deepseek_dialect.rs         # DeepSeek-V4 dialect formatting unit tests
    +-- generation_config_eos.rs    # EOS token array resolution unit tests
    +-- jinja_chat_template.rs      # Jinja template rendering unit tests
    +-- structured_decoder.rs       # Streaming structured decoder unit tests
    +-- tool_calls.rs               # Tool call DSL parser unit tests across Gemma/Qwen/DeepSeek
    \-- fixtures/                   # Vendored toy tokenizer fixture directories
        +-- ChatMLTokenizer/        # Toy ChatML tokenizer.json fixture
        \-- DeepseekTokenizer/      # Toy DeepSeek tokenizer.json fixture
```

## Key Modules

- `dialect.rs`: Resolves dialect special tokens and chat formatting rules for supported model families.
- `chat_template.rs`: Built-in text chat template rendering.
- `jinja_chat_template.rs`: Jinja template engine wrapper (`minijinja` + `pycompat`) for rendering `chat_template.jinja`.
- `detokenizer.rs`: `StreamingDetokenizer` for incremental UTF-8 token decoding.
- `stop_matcher.rs`: `StopMatcher` for evaluating stop sequences and EOS token sets.
- `structured_decoder.rs`: `StructuredDecoder` for streaming JSON / structured output parsing.
- `tool_call/`: Dialect-specific tool call DSL parsers (Gemma, Qwen, DeepSeek).

## Development & Test Commands

```sh
# Run tests for turbospark-tokenizer
cargo test -p turbospark-tokenizer
```

## Crate Gotchas

1. **Dynamic Added Token IDs in Test Fixtures**: Vendored test fixtures under `crates/*/tests/fixtures/{ChatMLTokenizer,DeepseekTokenizer}` embed placeholder added token IDs (e.g. `248044`) in their `added_tokens` JSON lists. The `tokenizers` loader renumbers added tokens sequentially starting right after the base vocabulary. NEVER hardcode token IDs by reading fixture JSON directly; always resolve token IDs at runtime from a loaded `MfTokenizer` (e.g., using `token_to_id`, `end_of_turn_id`).
