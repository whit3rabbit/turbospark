# Tool calling: wire formats, native decoding, and the rescue tier

Which tool-call wire formats this repo understands, at which of the three
independent implementations they are handled, and how to add a new format at
either tier. Read this page before touching tool-call parsing, before adding
a dialect, and before believing a claim about a format's grammar.

## The three independent implementations

There is no single tool-call parser in this repo. There are three, and they
exist for different reasons:

**(a) The native tier: one streaming decoder per `ChatDialect`.**
`crates/tokenizer/src/tool_call/` holds one parser per format (Gemma's DSL,
Qwen's `<function=>` XML, DeepSeek's DSML), and
`crates/tokenizer/src/structured_decoder/` holds the per-dialect streaming
state machines that bracket tokens off the raw stream and hand bodies to
those parsers. Keyed on the dialect the checkpoint resolves to, never on
tensor naming or text (AGENTS.md Gotcha 12). Used by `turbospark-server`,
`turbospark-check` and the FFI alike through `runtime::TurnSplitter`
(`docs/STREAMING.md`).

**(b) The rescue tier: markup recovered from a completed turn.**
`crates/server/src/guardrails.rs` plus `forge-guardrails` and this repo's
local additions in `crates/server/src/guardrails/extra_formats.rs`. It runs
only when the native decoder parsed no call, only on the buffered path, and
it validates what it recovers against the schema the request sent
(`docs/FORGE_GUARDRAILS.md` for the verdict, the one retry, and section 0b
for why there are two enforcement points and which requests get buffered).

**(c) The Swift app: its own copy of both.** `TurboSparkApp`'s native chat
never reaches the Rust server, so it has its own parser
(`Tools/Core/ToolCallParser.swift`) and its own rescue engine
(`Tools/Guardrails/ForgeGuardrailsEngine.swift`). The two rescue engines are
kept in parity by MIRRORED FIXTURES: every format test in
`crates/server/src/guardrails/tests.rs` has a sibling in
`Tests/TurboSparkAppTests/ForgeGuardrailsTests.swift` with the same markup,
so drift shows up as one side failing a test the other passes.

## Coverage table

| Family | Wire format | Native (Rust) | Rescue (Rust) | Rescue (Swift) | Verified against |
|---|---|---|---|---|---|
| Gemma 4 | `<\|tool_call>call:NAME{k:v,...}<tool_call\|>` | yes | `extra_formats` | engine Pattern 1d | real installed `gemma4.gturbo` |
| Qwen 3.5 / 3.6 / 3.8 | `<tool_call>` special tokens wrapping `<function=NAME><parameter=K>` XML | yes | `extra_formats` | engine Pattern 1 (+ parameter extraction) | real installed `qwen38-27b.gturbo` |
| Qwen2 / Qwen2.5 | Same ChatML `<tool_call>` / `<function=NAME>` XML; `<think>` and `<tool_response>` may remain ordinary text | yes | `extra_formats` | engine Pattern 1 (+ parameter extraction) | real Qwen2.5 MLX text smoke; tool-call gate pending |
| gpt-oss (Harmony) | `<\|channel\|>commentary to=functions.NAME ... <\|message\|>{json}<\|call\|>` | yes | n/a | n/a | real installed checkpoint |
| DeepSeek-V4 | `<dsml:tool_calls>` text markers | yes | n/a | n/a | bundled fixture |
| Muse Glimmer | `<atem:function_calls>` DSL | no parser (body routes to reasoning) | bare JSON only | bare JSON only | real installed checkpoint |
| Mistral | `[TOOL_CALLS] [{...}]` | no | forge | forge | unit fixtures |
| GLM | name + `<arg_key>K</arg_key>` / `<arg_value>V</arg_value>` pairs | no | `extra_formats` | engine Pattern 1b | published GLM-4.6 `chat_template.jinja` |
| MiniMax (and the Anthropic invoke shape generally) | `<invoke name="N"><parameter name="K">V</parameter></invoke>` | no | `extra_formats` | engine Pattern 1 (+ parameter extraction) | published MiniMax-M2 `chat_template.jinja` |
| Kimi K2 | `<\|tool_call_begin\|>functions.NAME:IDX<\|tool_call_argument_begin\|>{json}<\|tool_call_end\|>` | no | `extra_formats` | engine Pattern 1c | Moonshot's `tool_call_guidance.md` + vLLM's `kimi_tool_parser` |
| Longcat | `<longcat_tool_call>{"name":...,"arguments":{...}}</longcat_tool_call>` | no | forge's JSON scan, ZERO new code | engine Pattern 3b (tag strip) | vLLM's `longcat_tool_parser` (Hermes JSON body) |
| OpenAI / Hermes-style bare JSON | `{"name": ..., "arguments": {...}}` | no | forge | engine Pattern 4 | unit fixtures |

**RESCUE-TIER ROWS ARE NOT VALIDATED AGAINST A REAL INSTALL**, the same way
`docs/FORGE_GUARDRAILS.md` section 0c flags Muse Glimmer. None of GLM,
MiniMax, Kimi K2 or Longcat is installed on this machine, and the repo's
convention is to build a native `ChatDialect` + streaming decoder only
against a checkpoint that can be smoke-tested. Their grammars were read out
of each family's own published `chat_template.jinja` and Moonshot's tool-use
guide (2026-09-06), NOT out of the compressed cross-engine table OLMX keeps
-- which mislabels Gemma's format besides (next section).

## Two compressed-table claims corrected

**Qwen 3.8 did not change the grammar.** The real installed
`~/models/qwen38-27b.gturbo/chat_template.jinja` (line 68) renders the
identical `<tool_call>` / `<function=NAME>` / `<parameter=K>` instructions
Qwen 3.5 and 3.6 use; the checkpoint is architecture `qwen35` under the
hood. One dialect (`ChatDialect::ChatMl`), one parser
(`crates/tokenizer/src/tool_call/qwen.rs`), nothing to change. This
paragraph exists so nobody re-asks the next time a point release lands.

**Qwen2.5 uses the same ChatML tool-call grammar but a smaller special-token
contract.** The official Qwen2.5 tokenizer registers `<tool_call>` and
`</tool_call>`, while some published sidecars do not register `<think>` or
`<tool_response>` as special tokens. The ChatML resolver therefore treats
those two pairs as optional and leaves them as ordinary text when absent.

**Gemma's real format is the colon/brace DSL, not `<start_function_call>`.**
The installed `gemma4.gturbo` template (lines 244-258) renders
`<|tool_call>call:NAME{key:value,...}<tool_call|>` inside the special-token
pair. This port's `GemmaToolCallParser` matches it exactly. A cross-engine
table's one-cell label is not a grammar; read the template.

## The special-token wrinkle: what the rescue actually receives

**A WRAPPER TAG THAT IS A SPECIAL TOKEN NEVER REACHES THE RESCUE.** The
detokenizer renders every special token to the empty string (AGENTS.md
Gotcha 44), so markup the template shows as `<tool_call>...</tool_call>`
arrives at the rescue layer as the INTERIOR only, whenever the checkpoint's
vocabulary resolves those tokens under a dialect this engine knows. The
route is: the native decoder buffers the interior as a tool span, its
parser refuses the body, the failed span's body is RELEASED as text
(`StructuredAssistantDecoder::take_failed_span_text`, emitted by
`TurnSplitter::feed` on the error path -- before that release existed the
body was dropped and the call was lost twice over, once to the parser and
once to the rescue), and the rescue parses what is left.

This is why the GLM strategy keys on the `<arg_key>`/`<arg_value>` pair
markup rather than on the `<tool_call>` wrapper the template teaches: the
pairs survive both routes (literal text and wrapper-stripped), the wrapper
survives only one. Before writing a wrapper-anchored rescue pattern, ask
what the text looks like AFTER this engine's own decode.

## Adding a native dialect

`docs/NEW_MODEL.md` is the end-to-end checklist;
`crates/tokenizer/CLAUDE.md` Gotcha 11 names the four exhaustive matches a
new `ChatDialect` variant touches. A native tier needs a real install to
smoke-test against -- that is the bar that keeps the tier honest, and the
reason the four families below it are rescue-only.

## Adding a rescue format

Add a strategy in `crates/server/src/guardrails/extra_formats.rs`, tried
before `forge_guardrails::rescue_tool_call` and falling through to it when
nothing matches. The rules, each one load bearing:

- **The grammar comes from the checkpoint's own published template**, read
  raw, not from a summary table. Where a format names its calls through an
  id convention (Kimi K2's `functions.NAME:IDX`), the vendor's tool-use
  documentation is the source, not memory.
- **The allowlist gates every strategy.** A recovered name not among the
  tools the request offered rescues nothing; recovered markup must not mint
  a call to a tool the caller never granted.
- **Anchor on the signature that survives the decode**, per the wrinkle
  above.
- **Test at `inspect`** (the only place the tier runs), with the fixture
  mirrored into the Swift test file, and mutation-check the pattern so it
  reddens only its own cases.
- **Before writing any strategy, check forge's JSON scan first.** Longcat
  needed zero Rust code: `{"name", "arguments"}` inside any wrapper is
  already covered, and the test pinning that is the thing that keeps the
  zero-code claim true instead of assumed.

## What the rescue tier cannot do

It runs on COMPLETED turns only (a request carrying tools is buffered when
guardrails are on, `docs/FORGE_GUARDRAILS.md` section 5); it cannot feed a
streaming consumer; it has never run against a real GLM/MiniMax/Kimi/Longcat
install; and its regexes are matched against model output that may vary
around the template's own rendering, which is why each strategy accepts the
family's shape rather than its exact bytes.

## See also

- `docs/FORGE_GUARDRAILS.md`: the verdict, the retry, the buffering, and
  what is deliberately not taken from `forge-guardrails`.
- `docs/STREAMING.md`: the `TurnSplitter` the native tier streams through.
- `swift/docs/SWIFT_TOOLS.md`: the Swift app's tool execution, which the Swift
  rescue engine feeds.
