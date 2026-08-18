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
|   +-- dialect/                    # Resolves dialect special tokens and chat formatting rules
|   |   +-- mod.rs                  # ChatDialect enum & public API
|   |   +-- config.rs               # DialectConfig table & properties
|   |   \-- resolve.rs              # Special token probing & dialect resolution
|   +-- chat_template/              # Per-dialect text chat renderers (the FALLBACK path)
|   |   +-- mod.rs                  # Entry point & fallback dispatcher
|   |   +-- chatml.rs               # ChatML chat template renderer
|   |   +-- deepseek.rs             # DeepSeek chat template & tool call renderer
|   |   +-- gemma.rs                # Gemma chat template renderer
|   |   \-- mistral.rs              # Mistral [INST] chat template renderer
|   +-- jinja_chat_template.rs      # minijinja + pycompat wrapper rendering the checkpoint's own template
|   +-- detokenizer.rs              # StreamingDetokenizer for incremental UTF-8 token decoding
|   +-- stop_matcher.rs             # StopMatcher for evaluating stop sequences and EOS token sets
|   +-- structured_decoder/         # StructuredDecoder for streaming JSON / structured output
|   |   +-- mod.rs                  # StructuredDecoder state machine & event types
|   |   +-- chatml.rs               # ChatML thought & tool parsing
|   |   +-- deepseek.rs             # DeepSeek tool parsing
|   |   \-- harmony.rs              # Harmony channel & reasoning parser
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
    +-- harmony_channels.rs         # gpt-oss: the channel frame, and the tool call inside its header
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

- `dialect/`: Resolves dialect special tokens and chat formatting rules for supported model families. **`detect_dialect`'s ORDER is load-bearing and its Harmony arm requires THREE markers, not two** -- see Gotcha 6.
- `chat_template/`: Per-dialect text chat rendering, plus DeepSeek's hand-rolled native tool chat. The FALLBACK for a checkpoint that ships no template (see Gotcha 1).
- `jinja_chat_template.rs`: Jinja template engine wrapper (`minijinja` + `pycompat`) rendering the checkpoint's own template, for plain text chat as well as tool chat. Carries `parenthesize_conditional_kwargs`, a REMOVABLE minijinja compatibility shim (AGENTS.md Gotcha 53).
- `detokenizer.rs`: `StreamingDetokenizer` for incremental UTF-8 token decoding.
- `stop_matcher.rs`: `StopMatcher` for evaluating stop sequences and EOS token sets.
- `structured_decoder/`: `StructuredDecoder`, splitting generated output into visible content, reasoning (Harmony only, see Gotcha 4) and parsed tool calls.
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

2. **Harmony's stop set has THREE members and its naming is inverted** (ROADMAP M5). `ChatDialect::Harmony` covers `gpt-oss`, and it is the one dialect here whose stop set is not obvious from its EOS token: generation ends at `<|return|>` when the model has answered AND at `<|call|>` when it is invoking a tool, with `<|endoftext|>` as the base end-of-sequence. Dropping `<|call|>` does not error -- the model generates straight past its own tool call, which reads as a rambling model rather than as a stop-set bug. `<|end|>` is deliberately NOT a stop: it closes the SYSTEM and USER turns inside a rendered prompt, so stopping on it ends generation at the first token of a well-formed reply. The naming inverts every other dialect here: `<|endoftext|>` is the PAD token and `<|return|>` is the turn end. There is NO fallback renderer and `apply_dialect_chat_template` REFUSES for this dialect, because Harmony's real template is 17 KB of system preamble, reasoning-effort knob and TypeScript tool namespace, and a partial re-implementation is Gotcha 1's failure mode exactly; a real install always ships the template, so the refusal is what a MALFORMED install gets.

3. **Dynamic Added Token IDs in Test Fixtures**: Vendored test fixtures under `crates/*/tests/fixtures/{ChatMLTokenizer,DeepseekTokenizer}` embed placeholder added token IDs (e.g. `248044`) in their `added_tokens` JSON lists. The `tokenizers` loader renumbers added tokens sequentially starting right after the base vocabulary. NEVER hardcode token IDs by reading fixture JSON directly; always resolve token IDs at runtime from a loaded `MfTokenizer` (e.g., using `token_to_id`, `end_of_turn_id`).

4. **Harmony's frame is a HEADER/BODY pair, not a bracketing token pair, and that is why it has its own arm.** `<|channel|>` opens a header, `<|message|>` ends the header and opens the body, `<|end|>` closes the body: `channel_end_id` is `NO_SUCH_TOKEN_ID` for this dialect precisely so the Gemma-shaped bracketing arm stays unreachable (it would open a channel label on the first `<|channel|>` and never close it, swallowing the whole reply). `message_start_id` / `message_end_id` carry the two ids the state machine needs, and are `NO_SUCH_TOKEN_ID` everywhere else. Three things about `consume_harmony` worth knowing before changing it. It keys on TOKEN IDS, never on text, so it does not care whether the detokenizer renders special tokens or how the header's words tokenize. It EMITS the analysis channel as `StructuredAssistantEvent::Reasoning`, and **since `--reasoning` landed the ChatML and Gemma arms do too** -- that used to be the one asymmetry here, justified by the fact that no knob could turn those channels on, and the knob is what made it false. A body those arms discard is not hidden but DESTROYED, which is the wrong answer for a caller who asked to see the model think. The three arms now agree; what still differs is the FRAME (bracketing pair, `<think>` pair, header/body triple). And anything that is not the `final` channel is reasoning, including an unrecognized name, because a new channel misreported as reasoning shows up in the wrong place while one misreported as the answer corrupts the reply.

5. **A HARMONY TOOL CALL COMES OUT OF `finish`, NOT OUT OF `consume`, and every other dialect is the other way round.** Harmony frames a call as a `to=functions.NAME` recipient inside the channel header the state machine already parses, with a raw-JSON body -- so there is no fourth parser beside the Gemma / Qwen / DeepSeek three, just `JsonValue::parse` plus the existing allowed-tools check. What is genuinely different is WHEN it can be emitted: `<|call|>` terminates the call, `<|call|>` is in the dialect's stop set, and `run_raw_completion` breaks before the progress callback, so the decoder never sees the token that ends the span it is parsing. **A consumer that drives only `consume` therefore gets every call silently dropped** -- no error, no markup leaking, just a turn with nothing in it. That is what `crates/server`'s `stream_blocking` did until this landed, and it is why `finish` is now called there. Three smaller decisions worth not re-litigating. Only the `functions` namespace is a caller tool: builtins (`browser`, `python`) live in their own, `python`'s body is source code rather than JSON, and treating one as a call would fail to parse and poison the stream rather than degrade. A recipient the caller did not OFFER is not an error either but an ordinary body, reported by the channel rule -- unlike every other dialect, this decoder is built for every Harmony generation rather than only for requests carrying tools (server crate Gotcha 12), so an empty allowlist is the normal case and failing on it would cost the CLI a turn. And the arguments must parse to an OBJECT, matching what all three DSL parsers build; a truncated body (a generation that hit its token budget mid-JSON) is `Malformed`, which is the same verdict the Gemma arm reaches on an unterminated tool span.

6. **`ChatDialect::Harmony` and `ChatDialect::MuseGlimmer` SHARE `<|start|>`
   and `<|message|>` and share nothing else, so the probe order and the marker
   count are both load-bearing** (AGENTS.md Gotcha 52). Harmony is tested
   first and requires `<|channel|>` as well; Muse Glimmer is tested after it
   on `<|start|>` plus `<|eot|>`. The third Harmony marker was added because
   the two-marker probe resolved Muse Glimmer to Harmony and then failed to
   load on a missing `<|startoftext|>` -- a probe that can pass where its own
   resolver will fail. Both are checked in each arm rather than one, because a
   single token decides nothing here: `<|eot|>` alone is a Llama-3-family
   spelling that says nothing about the frame.
   Muse Glimmer's frame is `<|start|>role<|message|>content<|eot|>`, with
   `<|eom|>` for a message that hands off rather than ends the turn -- so
   `<|eom|>` is NOT in the stop set (stopping on it truncates a turn the model
   intends to continue) and `message_end_id` carries it instead. Its stop set
   is `<|end_of_text|>` and `<|eot|>`, which is what `generation_config.json`
   declares and NOT what `tokenizer_config.json`'s single `eos_token` says.
   Like Harmony it has NO fallback renderer: its template carries an
   image/video content macro and an `<atem:function_calls>` tool DSL, and that
   DSL is PLAIN TEXT rather than special tokens, so no tool-call parsing is
   wired for it and every tool marker id is `NO_SUCH_TOKEN_ID`.
