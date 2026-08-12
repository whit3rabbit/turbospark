# turbospark-tokenizer

Tokenizer wrapper around HF `tokenizers` (`MfTokenizer`), chat dialect resolution (Gemma 4, ChatML/Qwen, DeepSeek-V4, Mistral, Harmony/gpt-oss), chat template rendering (text-only and `minijinja` + `pycompat`), streaming detokenization (`StreamingDetokenizer`), stop condition matching (`StopMatcher`), tool call DSL parsers, and streaming structured decoder (`StructuredDecoder`).

## Safety

- `#![forbid(unsafe_code)]` is enforced in this crate.

## Directory & File Structure

```
crates/tokenizer/
+-- Cargo.toml                      # Crate manifest
+-- src/
|   +-- lib.rs                      # Library root re-exporting MfTokenizer and dialect APIs
|   +-- dialect.rs                  # Resolves dialect special tokens and chat formatting rules
|   +-- chat_template.rs            # Per-dialect text chat renderers (the FALLBACK path)
|   +-- jinja_chat_template.rs      # minijinja + pycompat wrapper rendering the checkpoint's own template
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
    +-- harmony_dialect.rs          # gpt-oss: the three-member stop set, and no fallback renderer
    +-- installed_template.rs       # Checkpoint template beats dialect; per-family agreement guard
    +-- jinja_chat_template.rs      # Jinja template rendering unit tests
    +-- structured_decoder.rs       # Streaming structured decoder unit tests
    +-- tool_calls.rs               # Tool call DSL parser unit tests across Gemma/Qwen/DeepSeek
    \-- fixtures/                   # Vendored toy tokenizer fixture directories
        +-- ChatMLTokenizer/        # Toy ChatML tokenizer.json fixture
        +-- DeepseekTokenizer/      # Toy DeepSeek tokenizer.json fixture
        +-- HarmonyTokenizer/       # gpt-oss's special-token NAMES + a minimal Harmony template
        \-- ZephyrTokenizer/        # Mistral's token table + an embedded Zephyr template
```

## Key Modules

- `dialect.rs`: Resolves dialect special tokens and chat formatting rules for supported model families.
- `chat_template.rs`: Per-dialect text chat rendering, plus DeepSeek's hand-rolled native tool chat. The FALLBACK for a checkpoint that ships no template (see Gotcha 1).
- `jinja_chat_template.rs`: Jinja template engine wrapper (`minijinja` + `pycompat`) rendering the checkpoint's own template, for plain text chat as well as tool chat.
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

1. **The Checkpoint's Template Beats the Dialect, and the Template Hides in
   Two Places** (AGENTS.md Gotcha 41). `ChatDialect` is resolved from the
   special-token table and decides IDS and the STOP SET only; it is not
   evidence about chat framing. TinyLlama-1.1B-Chat and Mistral-7B-Instruct
   present the identical `<unk>`/`<s>`/`</s>` table (Zephyr's `<|user|>` is
   plain text and never enters it), resolve to the same
   `ChatDialect::Mistral`, and are trained on different framing. So
   `apply_chat_template` renders the checkpoint's own Jinja template when it
   ships one and only falls back to `chat_template.rs`'s per-dialect
   renderers otherwise (`apply_dialect_chat_template` is that fallback,
   public so the guard test can compare the two). `load_from_dir` reads the
   template from EITHER a standalone `chat_template.jinja` OR
   `tokenizer_config.json`'s older `chat_template` key, which may itself be
   a string or a named list -- real installs on this machine split across
   both conventions and the split does not follow family. The one axis on
   which the two renders differ is `trim`: `chat_template.rs` strips
   surrounding whitespace from content unconditionally, a template only
   where it says `| trim`, and that single trailing newline re-froze
   `qwen3moe_quality_gate`'s row. Before touching any of this, run
   `tests/installed_template.rs --ignored` with the install vars set: it
   pins the trim behaviour per family and is what says whether
   `crates/bench`'s frozen digests are about to move. Note it compares the
   REAL protocol prompt via `include_str!`, not a retyped one -- on a tidy
   one-line string `trim` is a no-op and the guard sees nothing.

2. **Harmony's stop set has THREE members and its naming is inverted** (ROADMAP M5). `ChatDialect::Harmony` covers `gpt-oss`, and it is the one dialect here whose stop set is not obvious from its EOS token: generation ends at `<|return|>` when the model has answered AND at `<|call|>` when it is invoking a tool, with `<|endoftext|>` as the base end-of-sequence. Dropping `<|call|>` does not error -- the model generates straight past its own tool call, which reads as a rambling model rather than as a stop-set bug. `<|end|>` is deliberately NOT a stop: it closes the SYSTEM and USER turns inside a rendered prompt, so stopping on it ends generation at the first token of a well-formed reply. The naming inverts every other dialect here: `<|endoftext|>` is the PAD token and `<|return|>` is the turn end. There is NO fallback renderer and `apply_dialect_chat_template` REFUSES for this dialect, because Harmony's real template is 17 KB of system preamble, reasoning-effort knob and TypeScript tool namespace, and a partial re-implementation is Gotcha 1's failure mode exactly; a real install always ships the template, so the refusal is what a MALFORMED install gets. `StructuredDecoder` passes Harmony content through UNDECODED, so the `analysis` channel reaches a caller as text -- a stated limitation, and strictly better than the alternative, since Harmony's `channel_start_id` has no closing counterpart and the Gemma arm would open a channel label and never close it, swallowing the whole reply.

3. **Dynamic Added Token IDs in Test Fixtures**: Vendored test fixtures under `crates/*/tests/fixtures/{ChatMLTokenizer,DeepseekTokenizer}` embed placeholder added token IDs (e.g. `248044`) in their `added_tokens` JSON lists. The `tokenizers` loader renumbers added tokens sequentially starting right after the base vocabulary. NEVER hardcode token IDs by reading fixture JSON directly; always resolve token IDs at runtime from a loaded `MfTokenizer` (e.g., using `token_to_id`, `end_of_turn_id`).
