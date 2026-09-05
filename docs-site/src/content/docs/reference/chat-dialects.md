---
title: Chat Dialects, Templates, and Structured Decoding
description: The tokenizer-layer contract in turbospark-tokenizer: chat dialect detection and token resolution, Jinja template rendering, reasoning-effort handling, streaming structured decoding, tool-call parsing, and stop matching.
---

Source of record: `crates/tokenizer/src/dialect/` (`mod.rs`, `resolve.rs`,
`resolvers.rs`), `jinja_chat_template.rs` (plus the `jinja_compat.rs` shim it
calls), `reasoning.rs`, `structured_decoder/mod.rs`, `tool_call/mod.rs`, and
`stop_matcher.rs`.

## What a dialect is, and where it is resolved

A **chat dialect** (`crate::dialect::ChatDialect`) is an enum resolved from
the loaded tokenizer's special-token table when `MfTokenizer::load_from_dir`
opens a tokenizer directory. The dialect decides exactly two things:

1. The **special token IDs** (turn markers, tool-call markers, channel
   markers, thinking markers) exposed as public fields on `MfTokenizer`.
2. The **stop set** (`stop_token_ids`), plus `tool_call_stop_id`, the one
   stop that means "the model is invoking a tool" rather than "the turn is
   over".

The dialect does **not** decide chat framing. Framing is a property of the
checkpoint, which ships its own Jinja template; `MfTokenizer::apply_chat_template`
renders that template when present and falls back to the per-dialect text
renderers in `chat_template/` only when the checkpoint ships none.

`load_from_dir` reads, in order of authority:

| File | Required | Used for |
|---|---|---|
| `tokenizer.json` | yes | The HF `tokenizers` backend; dialect detection runs against its table. |
| `tokenizer_config.json` | no | Gemma's `bos_token`/`eos_token` names; the older embedded `chat_template` key (string or named list). |
| `chat_template.jinja` | no | Standalone template (the newer HF convention). Read first; `tokenizer_config.json`'s embedded key is the fallback. |
| `generation_config.json` | no | The checkpoint's full EOS set (e.g. an `eos_token_id` array); every non-negative id is merged into `stop_token_ids`. |

Which template convention a checkpoint uses is a property of its converter's
vintage, not of its family: HF moved the template out of
`tokenizer_config.json` into a standalone `chat_template.jinja` partway
through, while llama.cpp's GGUF converter still writes the older embedded
key. Reading only the standalone file makes every pre-move checkpoint look
template-less, which is how TinyLlama-1.1B-Chat (Zephyr framing) came to be
fed Mistral's `[INST]` fallback.

### Resolved ID fields on `MfTokenizer`

| Field | Meaning |
|---|---|
| `bos_id`, `eos_id`, `pad_id` | Base sequence tokens. |
| `end_of_turn_id` | The turn-end marker read by the completion loop. |
| `tool_call_start_id` / `tool_call_end_id` | Bracketing pair around an emitted tool call, where the dialect has one. |
| `tool_response_id` / `tool_response_end_id` | Bracketing pair around an injected tool result. |
| `tool_call_stop_id` | The stop-set member meaning "invoking a tool". Read by `run_raw_completion`'s stop ladder only. |
| `channel_start_id` / `channel_end_id` | Channel frame pair (bracketing dialects) or header opener (header/body dialects, with end as the sentinel). |
| `message_start_id` / `message_end_id` | Header/body pair for header-shaped frames; sentinel everywhere else. |
| `think_start_id` / `think_end_id` | `Option<i32>` thought-frame pair. |
| `stop_token_ids` | Full stop set (`BTreeSet<i32>`), including ids merged from `generation_config.json`. |
| `vocab_size` | The model's padded lm_head row count, not the tokenizer's actual vocabulary. Logits buffers use this; callers holding a `RealForwardRunner` should read that runner's `vocab_size()` instead. |
| `bos_prefix_id` (private) | What `encode(_, add_bos: true)` prepends; `None` for dialects whose template or fallback renderer emits BOS itself (ChatML, Harmony, Muse Glimmer, Llama 3). |

`NO_SUCH_TOKEN_ID` (`-1`) is the sentinel for roles a dialect frames as plain
text rather than a special token. It is never a valid token ID; resolvers
assign it rather than inventing IDs for markup a checkpoint cannot emit.

## Dialect index

| Variant | Checkpoints | Frame | Fallback renderer | Stop set |
|---|---|---|---|---|
| `Gemma` | Gemma 4 | Turn/channel contract; `<turn|>` ends a turn; `<|channel>` / `<channel|>` bracket channels. | yes | `{eos, eot, tool_response}` |
| `ChatMl` | Qwen-style ChatML | `<|im_start|>` / `<|im_end|>`; `<think>` / `</think>`; `<tool_call>` / `</tool_call>`; `<tool_response>` / `</tool_response>`. | yes | `{im_end, end_of_text}` |
| `Deepseek` | DeepSeek-V4 | Fullwidth-bar sentence marks (see entry). | yes | `{eos}` |
| `Mistral` | Mistral / Mixtral; also any table with only `<s>`/`</s>` (Zephyr) | `[INST] ... [/INST]` in plain text; only `<s>` and `</s>` are special. | yes | `{eos}` |
| `Harmony` | gpt-oss | `<|start|>role<|message|>content<|end|>`, with `<|channel|>` opening the analysis/final split. | **no** (refuses) | `{return, call, endoftext}` |
| `MuseGlimmer` | muse_glimmer | `<|start|>role<|message|>content<|eot|>`; `<|eom|>` hands off; tool DSL is plain text. | **no** (refuses) | `{end_of_text, eot}` |
| `Llama3` | Meta Llama 3 base/Instruct | `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>`, opened by `<|begin_of_text|>`. | yes | `{end_of_text, eot_id}` |

## Detection rules

`detect_dialect` probes the loaded tokenizer in this fixed order and returns
the first arm whose witnesses are all present:

| # | Arm | Witnesses (all required) |
|---|---|---|
| 1 | `Deepseek` | `<\u{FF5C}User\u{FF5C}>` present |
| 2 | `Harmony` | `<|start|>` AND `<|message|>` AND `<|channel|>` |
| 3 | `MuseGlimmer` | `<|start|>` AND `<|eot|>` |
| 4 | `Llama3` | `<|start_header_id|>` AND `<|eot_id|>` |
| 5 | `ChatMl` | `<|im_end|>` |
| 6 | `Mistral` | `<turn|>` ABSENT, `<s>` present, `</s>` present |
| 7 | `Gemma` | fallback for everything unrecognized |

Rules the ordering and the marker counts encode:

- **A detection probe must not pass where its own resolver will fail.** The
  Harmony arm requires a third marker (`<|channel|>`, the format's defining
  feature) because a two-marker `<|start|>` + `<|message|>` probe resolved
  `mlx-community/Muse-Glimmer-30B-4bit` to Harmony and then failed to load on
  a missing `<|startoftext|>`: refused rather than run, which is still the
  wrong answer.
- **Both markers are checked in each arm.** A single token decides nothing
  here: `<|eot|>` alone is a Llama-3-family spelling that says nothing about
  the frame.
- **Shared tokens are probed around, not on.** Harmony and Muse Glimmer share
  `<|start|>` and `<|message|>` (Harmony is probed first, on the token Muse
  Glimmer lacks). Llama 3 and Muse Glimmer share `<|begin_of_text|>` and
  `<|end_of_text|>` (Llama 3 is keyed on its own header markers instead).
  Mistral's arm requires Gemma's `<turn|>` to be absent because `</s>` alone
  is far too common a token to decide a dialect on.
- **Lookup round-trips.** `special_token_id` resolves a name to an id only if
  mapping the id back returns the same string, rejecting the unk-token
  fallback some tokenizers substitute for out-of-vocabulary names.
- **Resolvers over-resolve on purpose.** Harmony, Muse Glimmer, and Llama 3
  resolve more markers than they store, so a checkpoint whose frame is only
  half present fails at load rather than at the first rendered prompt.
- **Gemma is the fallback**, deliberately: every fixture and install that
  predates the other dialects lands there, and moving the default would
  change what an unrecognized tokenizer does.

## Dialect entries

### Gemma

`resolve_gemma` reads the BOS/EOS **names** from `tokenizer_config.json`
(`bos_token` / `eos_token`), then resolves them by name. Required tokens:

| Token | Resolves to |
|---|---|
| config `bos_token` | `bos_id` (also `bos_prefix_id`) |
| config `eos_token` | `eos_id` |
| `<pad>` | `pad_id` |
| `<turn|>` | `end_of_turn_id` |
| `<\|tool_response>` | `tool_response_id`, and `tool_call_stop_id` |
| `<\|tool_call>` / `<tool_call\|>` | `tool_call_start_id` / `tool_call_end_id` |
| `<tool_response\|>` | `tool_response_end_id` |
| `<\|channel>` / `<channel\|>` | `channel_start_id` / `channel_end_id` |

- Gemma hands over to the caller by emitting the tool-**response** marker,
  which is why `tool_call_stop_id` is `tool_response_id` here.
- Channels bracket, so the pair above says everything; `message_*_id` are
  sentinels and there is no header to end.
- No thinking markers (`think_*_id` are `None`).
- `vocab_size` 262,144.

### ChatML (`ChatMl`)

| Token | Resolves to |
|---|---|
| `<\|im_start\|>` | required (resolved, stored in `_` binding) |
| `<\|im_end\|>` | `end_of_turn_id` |
| `<\|endoftext\|>` | `bos_id`, `eos_id`, `pad_id` |
| `<tool_call>` / `</tool_call>` | `tool_call_start_id` / `tool_call_end_id` |
| `<tool_response>` / `</tool_response>` | `tool_response_id` / `tool_response_end_id` |
| `<think>` / `</think>` | `channel_start_id` / `channel_end_id`, and `think_start_id` / `think_end_id` |

- No BOS prefix: the template frames turns itself, so `encode` with
  `add_bos: true` adds nothing.
- `tool_call_stop_id` is the sentinel: ChatML closes a call with
  `</tool_call>` and then ends the turn with `<|im_end|>`, so no stop token
  of its own means "tool".
- `vocab_size` 248,320 (the padded head row count, not the tokenizer's
  actual vocabulary).

### DeepSeek (`Deepseek`)

DeepSeek-V4's marks use U+FF5C (fullwidth vertical bar) and U+2581 (lower
one eighth block), shown here in Rust escape form exactly as `resolve.rs`
defines them:

| Constant (escape form) | Resolves to |
|---|---|
| `<\u{FF5C}begin\u{2581}of\u{2581}sentence\u{FF5C}>` | `bos_id` (also `bos_prefix_id`) |
| `<\u{FF5C}end\u{2581}of\u{2581}sentence\u{FF5C}>` | `eos_id`, `pad_id`, `end_of_turn_id` |
| `<\u{FF5C}User\u{FF5C}>` | required (detection witness) |
| `<\u{FF5C}Assistant\u{FF5C}>` | required |
| `<think>` / `</think>` | `channel_start_id` / `channel_end_id`, and `think_start_id` / `think_end_id` |

- The thought channel brackets; there is no header (`message_*_id`
  sentinels).
- Tool-call markers are plain text in this dialect, so every tool id is the
  sentinel; DeepSeek's hand-rolled native tool chat lives in
  `chat_template/deepseek.rs`.
- Stop set `{eos}`; `vocab_size` 129,280.

### Mistral

| Token | Resolves to |
|---|---|
| `<s>` | `bos_id` (also `bos_prefix_id`) |
| `</s>` | `eos_id`, `pad_id`, `end_of_turn_id` |

- Turns are framed in **plain text** (`[INST] user [/INST] assistant</s>`),
  so there is no instruction marker in the special-token table to key on;
  the sentence pair is the only reliable witness.
- Mixtral 8x7B-Instruct v0.1 has exactly three special tokens (`<unk>`,
  `<s>`, `</s>`) and no tool-calling or thinking markup: every such id is
  the sentinel, so `StructuredDecoder` never hunts for markup the model
  cannot emit.
- End of turn IS `</s>`, not a separate marker. `vocab_size` 32,000.
- **Zephyr is this dialect.** TinyLlama-1.1B-Chat and the vendored
  `ZephyrTokenizer` fixture carry the identical `<unk>`/`<s>`/`</s>` table
  (Zephyr's `<|user|>` is plain text and never enters it) and resolve to
  `ChatDialect::Mistral`, while their framing comes from the checkpoint's
  own embedded Zephyr template.

### Harmony (`Harmony`, gpt-oss)

| Token | Resolves to |
|---|---|
| `<\|startoftext\|>` | `bos_id` |
| `<\|endoftext\|>` | `pad_id` (not the turn end) |
| `<\|return\|>` | `eos_id` and `end_of_turn_id` |
| `<\|call\|>` | `tool_call_stop_id` |
| `<\|end\|>` | `message_end_id` (required; deliberately NOT a stop) |
| `<\|message\|>` | `message_start_id` (required) |
| `<\|channel\|>` | `channel_start_id` (required; detection witness) |

- **The stop set has three members**: `<|return|>` (answered),
  `<|call|>` (invoking a tool), and `<|endoftext|>` (base end-of-sequence).
  Dropping `<|call|>` does not error; the model generates straight past its
  own tool call.
- `<|end|>` is deliberately not a stop: it closes the system and user turns
  inside a rendered prompt, so stopping on it would end generation at the
  first token of a well-formed reply.
- Naming inverts every other dialect here: `<|endoftext|>` is the PAD token
  and `<|return|>` is the turn end.
- `channel_end_id` is the sentinel: `<|channel|>` opens a header that
  `<|message|>` ends and `<|end|>` closes the body. A non-sentinel end would
  make the Gemma-shaped bracketing arm reachable for this dialect.
- Tool markup ids stay sentinel even though Harmony has tool calling:
  a call is a channel plus a recipient in the message header, not a
  bracketing token pair (see structured decoding below).
- No BOS prefix (the template emits `<|start|>` itself). `vocab_size`
  201,088.
- **No fallback renderer.** Harmony's real template is 17 KB of system
  preamble, reasoning-effort knob, and a TypeScript tool namespace; a
  hand-rolled renderer would be a large second implementation of something
  the checkpoint ships. The dialect variant exists for the ids and the stop
  set.

### Muse Glimmer (`MuseGlimmer`)

| Token | Resolves to |
|---|---|
| `<\|begin_of_text\|>` | `bos_id` |
| `<\|end_of_text\|>` | `eos_id` |
| `<\|eot\|>` | `end_of_turn_id` |
| `<\|finetune_right_pad\|>` | `pad_id` |
| `<\|start\|>` | `channel_start_id` (required) |
| `<\|message\|>` | `message_start_id` (required) |
| `<\|eom\|>` | `message_end_id` (required) |

- Stop set is `{<|end_of_text|>, <|eot|>}`, which is what
  `generation_config.json` declares (`eos_token_id: [200001, 200008]`) and
  NOT what `tokenizer_config.json`'s single `eos_token` says. Resolving only
  the latter loses the end-of-turn stop.
- `<|eom|>` is deliberately NOT a stop: it ends a message that is handing
  off (a tool call), so stopping on it would truncate a turn the model
  intends to continue. It is carried as `message_end_id` instead.
- `<|start|>` opens a message header, the same job Harmony gives
  `<|channel|>`, so it goes in `channel_start_id`; `channel_end_id` stays
  the sentinel for Harmony's reason (header/body triple, not a bracketing
  pair).
- Tool calls are `<atem:function_calls>` **plain text**, not special tokens:
  every tool marker id is the sentinel and no tool-call span parsing is
  wired for this dialect.
- No BOS prefix. `vocab_size` 202,048.
- **No fallback renderer**: the checkpoint's template carries an
  image/video content macro and the ATEM tool DSL.

### Llama 3 (`Llama3`)

| Token | Resolves to |
|---|---|
| `<\|begin_of_text\|>` | `bos_id` |
| `<\|end_of_text\|>` | `eos_id`, `pad_id` |
| `<\|eot_id\|>` | `end_of_turn_id` |
| `<\|start_header_id\|>` | required (detection witness) |
| `<\|end_header_id\|>` | required |

- End of turn is `<|eot_id|>`, not `<|end_of_text|>`: the checkpoint closes
  every assistant turn with the former and reserves the latter for the raw
  end of a document.
- No dedicated `<pad>`; `<|end_of_text|>` is reused, as ChatML and Mistral
  reuse their own EOS.
- No BOS prefix: the fallback renderer emits `<|begin_of_text|>` itself,
  matching the real template's `{{- bos_token }}` at the top, so the encoder
  must not prepend a second one.
- No tool-calling or thinking markup in the base 8B-Instruct table this was
  built against: every such id is the sentinel. A 3.1-family checkpoint's
  `<|eom_id|>` / `<|python_tag|>` pair is out of scope and untested here.
- Stop set `{eos, eot}`; `vocab_size` 128,256 (128,000 base BPE merges plus
  256 reserved special-token slots).

## Jinja template rendering

`jinja_chat_template.rs::render_generic_chat_template` is the primary render
path for plain text chat as well as tool chat, using `minijinja` as the
engine. Signature:

```rust
pub fn render_generic_chat_template(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    tools: &[FunctionDefinition],
    add_generation_prompt: bool,
    reasoning: ReasoningEffort,
) -> Result<String, TokenizerError>
```

`MfTokenizer::encode_generic_tool_chat(messages, tools, reasoning)` wraps
it: renders, then encodes with `add_bos = false`.

The template context matches the shape HF's own call site passes:

| Key | Value |
|---|---|
| `messages` | Array of message objects: `role`; `content` as a bare string for text-only messages, or HF's content-part list (`[{"type": "image"}, {"type": "text", "text": ...}]`) for multimodal ones; optional `tool_call_id`, `name`, and `tool_calls` (each call as `{id, type: "function", function: {name, arguments}}`). |
| `tools` | Present **only when non-empty** (absent otherwise, so a template's `{% if tools %}` branch behaves as upstream). Each tool as `{type: "function", function: {name, description, parameters}}`. |
| `add_generation_prompt` | Bool. |
| `enable_thinking` | `reasoning.enable_thinking()` (see reasoning below). |
| `reasoning_effort`, `reasoning_strength` | The level string, **only when a level is asked for**. Absent rather than null when unset: a template resolves its key with `\|default('xhigh')`, and a present-but-null key defeats that default instead of taking it. |
| `add_vision_id` | Always `false`. The flag only controls an optional `"Picture N: "` text prefix and defaults falsy upstream. |
| `bos_token`, `eos_token` | The token strings for `bos_id` / `eos_id`. |

The template environment registers two functions `minijinja` has no built-in
equivalent for: `raise_exception` (HF's chat-template global for malformed
input; surfaces to the caller as `TokenizerError::InvalidChatTemplate`) and
`strftime_now` (transformers' own global, needed by gpt-oss's system
preamble). `minijinja_contrib`'s `pycompat` callback is set as the unknown
method handler.

Render failures come back as `TokenizerError::InvalidChatTemplate`: no
installed template, a `raise_exception` rejection by the template itself, or
a template feature `minijinja` does not support.

### The conditional-keyword-argument shim

Before the source is handed to `minijinja`,
`jinja_compat::parenthesize_conditional_kwargs` rewrites every conditional
expression used as a keyword argument, `k=EXPR` to `k=(EXPR)`:

- **Why it exists.** minijinja 2.22.0 rejects `f(k=a if c else d)` with a
  syntax error; Jinja2's grammar for a keyword-argument value is a full
  expression, conditional included. Muse Glimmer's template has
  `namespace(name=tcid if tcid else '')`, and without the rewrite the whole
  template fails to parse and no prompt can be rendered at all.
- **Why it is safe.** The parenthesized form is exactly Jinja2's own
  precedence for a keyword-argument value, so no semantics change by
  construction.
- **How the scan stays out of prose.** It enters only `{{ ... }}` and
  `{% ... %}` blocks and steps over `{# ... #}` comments; `==`, `!=`, `<=`,
  `>=` are not counted as `=`; a `=` counts only when the preceding
  character is part of an identifier and the following one is not another
  `=`. String literals are tracked so a `,` or `)` inside quotes cannot end
  an argument early, and nesting is tracked so a call inside a call is one
  value.
- **It is meant to be deleted** when a stable minijinja handles conditional
  keyword arguments directly. Returns a borrowed `Cow` (no copy) when
  nothing needed rewriting, which is the case for every other template here.

## Reasoning effort

`reasoning.rs::ReasoningEffort` is the knob behind `--reasoning` (CLI) and
`reasoning_effort` (server). The level lives in the checkpoint's template,
not in the weights: a reasoning model reasons harder because its rendered
system preamble tells it to.

| Variant | Spelling | Notes |
|---|---|---|
| `Off` | `off` | Default. `enable_thinking` renders false and **no effort key is inserted at all**; byte-identical to the prompt rendered before the type existed. |
| `Low` | `low` | |
| `Medium` | `medium` | |
| `High` | `high` | Harmony's and Muse Glimmer's top setting. **Qwen 3.8 rejects this spelling.** |
| `XHigh` | `xhigh` | Qwen 3.8's top setting, and its own default when nothing is passed. |

API: `parse(&str) -> Option<Self>` and `as_str()` are inverses over the five
spellings; `ALL` lists them ascending with `Off` first;
`level() -> Option<&'static str>` is `None` for `Off`; `enable_thinking()`
is `self != Off`.

Rules the implementation encodes:

- **A level implies `enable_thinking: true`.** Every template that has both
  reads the effort key inside the thinking gate, so a level with thinking
  off would set a key nothing reads: a flag that does nothing and looks
  like it worked.
- **Both spellings are set.** `EFFORT_KEYS = ["reasoning_effort",
  "reasoning_strength"]`: Qwen 3.8 and Harmony read the former, Muse Glimmer
  the latter. A template reads the one it knows and ignores the other, so
  deciding per family would be a table that rots.
- **The accepted set is the checkpoint's own, not this port's.** Validation
  is left to the template, which names its set in its own error (Qwen 3.8
  raises on `high` with "Unexpected reasoning effort high. Supported types
  are xhigh (default), medium, and low", surfaced through
  `TokenizerError::InvalidChatTemplate`).

### Reasoning support classes

`MfTokenizer::reasoning_support() -> ReasoningSupport` reads the installed
template **source** (a substring scan: a Jinja variable is read by name, so
a template that never writes the identifier provably cannot honour it; the
error direction is toward permitting):

| Class | Meaning | Examples |
|---|---|---|
| `None` | The template names none of the keys; a level changes no byte of the prompt. | Gemma's dialect fallback; every template-less synthetic install. |
| `ToggleOnly` | The template reads `enable_thinking` and no effort key: thinking toggles, the level is ignored. | Qwen3.5-era checkpoints (`bonsai27b`, `ternary27b`). |
| `Level` | The template reads an effort key, so every level is expressible. | Qwen 3.8, Harmony, Muse Glimmer. |

`MfTokenizer::accepted_reasoning_levels() -> Vec<ReasoningEffort>` refines
this for UI building: it renders a one-message conversation at each of the
five spellings and reports, ascending and always starting at `Off`:

- Levels whose rendered prompt is byte-identical to an earlier one are
  **collapsed**, because "did it raise" alone is the wrong question: a
  template that ships but reads no reasoning key renders happily at every
  level, and a raise-only probe would report five accepted levels of which
  four are silent no-ops. `None` reports `[Off]`; `ToggleOnly` reports
  `Off` plus one on-level (its four renders are the same prompt).
- A `ToggleOnly` on-level is `Low` **by position** and carries no meaning as
  a label; read `reasoning_support()` and say "On" rather than printing the
  spelling.
- The result never comes back empty: it fails open, because an empty set
  would hide a control the checkpoint may well honour.

Both are open-time calls (five renders of two lines), never per-turn.

## Structured output decoding

`structured_decoder/mod.rs::StructuredAssistantDecoder` splits a generated
stream into events per dialect:

```rust
pub enum StructuredAssistantEvent {
    Content(String),    // text for display
    Reasoning(String),  // scratchpad, separated from the answer
    ToolCall(ParsedToolCall),
}
```

Construction takes the turn's prompt IDs:

```rust
StructuredAssistantDecoder::new(
    tokenizer: &MfTokenizer,
    allowed_tools: HashSet<String>,
    id_generator: impl FnMut() -> String,
    prompt_ids: &[i32],
)
```

### The initial state comes from the rendered prompt

`prompt_ids` is an argument because a generation prompt can open a frame the
model then never emits the opening token for, and the decoder must start
where the prompt left the model:

- `prompt_opens_thought` scans the prompt IDs **backwards** for the first
  `think_start_id` / `think_end_id` and reports which it met. Backwards,
  because a tool preamble puts balanced `<think></think>` pairs in its
  instructions and only the tail decides. Qwen's template ends
  `<|im_start|>assistant\n<think>\n` with thinking on, so the first
  generated token is already scratchpad and `<think>` never arrives; a
  decoder starting in `Channel::Visible` stays there and reports the whole
  scratchpad as the answer. Always false where `think_*_id` are `None`
  (Gemma, Harmony, Muse Glimmer).
- `muse_state_for` does the same for Muse Glimmer's header shape: the
  checkpoint's `add_generation_prompt` emits `<|start|>assistant` and stops,
  so a real generation begins inside a header and no `<|start|>` ever
  arrives. The returned `MuseState` (`Header` seeded with the decoded
  header text, `Body`, `Between`, or `Unframed`) is what stops the header
  remainder (` to=self`, the scratchpad recipient) passing through as prose.
- The prompt is the source, **not the reasoning level**: keying on
  `reasoning != Off` would open a frame on a checkpoint whose template
  enables thinking without prefilling the tag, and hand the caller an empty
  reply.
- Passing `&[]` means "nothing was prefilled" and reproduces the
  unconditional-`Visible` behaviour exactly.

### Per-dialect arms of `consume`

`consume(token_id, delta)` keys every transition on the token **ID**, never
on text, which makes the state machine independent of whether the
detokenizer renders special tokens and of how header words tokenize.

| Dialect | Arm |
|---|---|
| `Gemma` | Bracketing channels: `channel_start_id` enters `Label`, `channel_end_id` returns to `Visible`. `Label` accumulates until a newline: the first line lowercased is the channel name (`final` or `answer` goes visible, anything else, including an unrecognized name, is reasoning), the rest is content. `<\|tool_call>` opens a span; every token id inside is accumulated (up to the shared byte ceiling, `Oversized` past it); `<tool_call\|>` closes it, decodes the span with specials intact, and runs `GemmaToolCallParser`. Tool-response markers are accepted only after at least one emitted call and with no open span, else `Malformed`. |
| `ChatMl` | The `<think>` pair via `channel_start_id` / `channel_end_id`; the body is emitted as `Reasoning`. |
| `Deepseek` | DSML markers, via the tool-call module below. |
| `MuseGlimmer` | The header/body state machine seeded by `muse_state_for`; `<\|eot\|>` never reaches it (see the hazard below). |
| `Harmony` | A header/body triple: `<\|channel\|>` opens a header, `<\|message\|>` ends it and opens the body, `<\|end\|>` closes the body. Anything that is not the `final` channel is reasoning, including an unrecognized name. |
| `Mistral`, `Llama3` | Passthrough: empty deltas yield nothing, everything else is a `Content` event. No markup to decode; all channel/tool ids are the sentinel. |

A real Harmony assistant turn and a tool call read:

```text
<|channel|>analysis<|message|>REASONING<|end|>
<|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
```

```text
<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>{"city":"Oslo"}<|call|>
```

`consume_flushed_text(text)` drives the same machine from text chunks
(internally `consume(-1, text)`; empty text is a no-op). `has_tool_calls()`
reports whether at least one call has been emitted.

### Finish semantics and the stop-token-swallowing hazard

`run_raw_completion` breaks out of its generation loop on a stop token
**before** the progress callback, so stop tokens never reach `consume`. Two
consequences:

- **A Harmony tool call is emitted from `finish()` and nowhere else in
  practice.** `<|call|>` terminates the call and `<|call|>` is in the stop
  set, so the one token that closes the span is the token the decoder is
  structurally unable to see. A consumer that drives only `consume` gets
  the reasoning and the content and silently drops every call.
- For the other dialects, an open tool span at `finish()` is the error it
  looks like: their terminator is an ordinary token that either arrived or
  did not, and `finish()` returns `Malformed`.

`finish()` also releases anything held back: `drain()` returns withheld
text (a potential DSL-open prefix) as a final `Content` event. Muse
Glimmer's `<|eot|>` is absent from its state machine on purpose for the
same reason: it is a stop token, so only `<|eom|>`, which hands off rather
than ending the turn, ever reaches `consume`.

## Tool-call parsers

`tool_call/` provides one parser per dialect with a bracketed, text-DSL call
shape:

| Export | Dialect | Shape |
|---|---|---|
| `GemmaToolCallParser` | Gemma | Gemma's custom DSL inside `<\|tool_call>` ... `<tool_call\|>`. |
| `QwenToolCallParser` | Qwen/ChatML | ChatML `<function=...>` framing inside `<tool_call>` ... `</tool_call>`. |
| `DeepseekToolCallParser` | DeepSeek | DSML markers; `deepseek_dsml_mark()` exposes the mark. |

All three produce:

```rust
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonValue,
    pub arguments_json: String,
}
```

Shared rules:

- `MAXIMUM_BYTES = 256 * 1024`, the byte ceiling on an accumulated span.
- `is_valid_function_name`: non-empty, at most 64 bytes, every character
  ASCII alphanumeric, `_`, or `-`.

Harmony is deliberately **not** a fourth parser: a Harmony call is a
`to=functions.NAME` recipient in the channel header the state machine
already parses, with a raw-JSON body, so there is no DSL grammar to read.

## Stop-string matching

`stop_matcher.rs::StreamingStopMatcher` matches caller-supplied stop
**strings** across streaming text chunks (token-id stops are handled by the
dialect's stop set, above):

```rust
StreamingStopMatcher::new(stops: Vec<String>)  // empty strings filtered out
m.push(text) -> String   // emittable prefix; withholds the rest
m.finish() -> String     // releases pending text that never matched
m.is_stopped() -> bool
```

- On each `push`, if any stop string matches, everything before the earliest
  match is returned and the matcher latches stopped.
- Otherwise the longest suffix of the accumulated text that is a proper
  prefix of some stop string is withheld (computed in characters, not bytes,
  so multi-byte boundaries are safe), and the rest is returned. A stop split
  across two chunks is still caught.

## See also

- [HTTP API](/reference/http-api) for how decoded tool calls surface on the
  server's chat endpoints, including the tool-call guardrails and
  `finish_reason` mapping that consume `tool_call_stop_id`.
