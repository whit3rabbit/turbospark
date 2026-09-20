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
| Mistral | `[TOOL_CALLS]` special token, then a JSON array of `{name, arguments}` running to end of turn | yes, since 2026-09-15 (`MistralToolCallParser`; span emitted from `finish` because `</s>` is the only terminator and it is a stop token) | forge | forge | real installed `mistral7b-dense.gturbo`; earliest tables carry no marker and keep the passthrough |
| GLM | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value></tool_call>`; string argument values RAW, everything else JSON (`tojson`) | yes, since 2026-09-19 (`GlmToolCallParser`; text-marker arm -- the markup is ADDED but not special in the real table, ids 154843-154850, so it survives detokenization; `tool_call_support` = `Native`) | `extra_formats` | engine Pattern 1b | `zai-org/GLM-4.7-Flash`'s `tokenizer.json` + `chat_template.jinja`, read raw 2026-09-19; real-install witness still gated on the deepseek2 `q_lora_rank > 0` descope |
| GLM | name + `<arg_key>K</arg_key>` / `<arg_value>V</arg_value>` pairs | no | `extra_formats` | engine Pattern 1b | published GLM-4.6 `chat_template.jinja` |
| MiniMax (and the Anthropic invoke shape generally) | `<minimax:tool_call>` wrapper (a NON-special added token in the published M2 table, so it survives detokenization as text) around `<invoke name="N"><parameter name="K">V</parameter></invoke>` | no, and the 2026-09-15 probe of the published tokenizer found two blockers to revisit before building the arm: the wrapper tokens are not special (so a native arm is a text-marker arm like DeepSeek's, not an id-bracket arm), and this port's dialect probe strings (`]~!b[` et al.) do not match the published M2 table's bos (`]!p~[`) -- the dialect may have been built off a different generation's tokenizer | `extra_formats` | engine Pattern 1 (+ parameter extraction) | published MiniMax-M2 `chat_template.jinja` + `tokenizer_config.json` |
| Kimi K2 | `<|tool_calls_section_begin|>` wrapper, then per call `<|tool_call_begin|>functions.NAME:IDX<|tool_call_argument_begin|>{json}<|tool_call_end|>` | yes, since 2026-09-19 (`KimiToolCallParser`; text-marker arm -- the section/call markers are ADDED but not special in the real table, ids 163595-163599; the checkpoint's own `functions.NAME:IDX` id is kept verbatim on the parsed call) | `extra_formats` | engine Pattern 1c | `moonshotai/Kimi-K2.5`'s `tokenizer_config.json` added-token list + `chat_template.jinja`, read raw 2026-09-19; **the whole K2 line is tiktoken-only** -- no `tokenizer.json` exists anywhere in the ecosystem (K2, K2.5, K2.6, K2.7, Thinking, the mlx/ISTA/nvidia builds all checked), so no K2 install is loadable by this engine and the witness stays unreachable independent of the models' 1T-class size |
| Longcat | `<longcat_tool_call>{"name":...,"arguments":{...}}</longcat_tool_call>` | no | forge's JSON scan, ZERO new code | engine Pattern 3b (tag strip) | vLLM's `longcat_tool_parser` (Hermes JSON body) |
| OpenAI / Hermes-style bare JSON | `{"name": ..., "arguments": {...}}` | no | forge | engine Pattern 4 | unit fixtures |

**TWO OF THE RESCUE-TIER-ONLY ROWS HAVE SINCE GONE NATIVE, WITHOUT AN
INSTALL.** GLM and Kimi K2 now carry native dialects derived from their
real published tables (2026-09-19), but neither is validated against a
real generation yet, and the blocks are precise:

- **GLM**: the checkpoint that carries the dialect (`zai-org/GLM-4.7-Flash`,
  a `Glm4MoeLiteForCausalLM` MLA MoE whose unsloth GGUF reports
  `general.architecture = deepseek2`) is gated at deepseek2 intake by the
  recorded `q_lora_rank > 0` descope (`DEVIATIONS.md`'s deepseek2 section)
  -- its GGUF carries the full q-lora trio (`attn_q_a` / `attn_q_a_norm` /
  `attn_q_b`), SPLIT `attn_k_b` / `attn_v_b` projections, `exp_probs_b.bias`
  noaux_tc routing, and MXFP4 expert tensors, four lift items before a
  witness install can even be attempted.
- **Kimi K2**: no tokenizer sidecar exists to install at all -- the line is
  tiktoken-only -- and every K2 checkpoint is 1T-class, past this machine at
  any quantization.

The native decoder tests therefore run over fixtures built from the REAL
tables (GLM: reduced from the real `tokenizer.json`; Kimi: the real
added-token list over a disclosed synthetic BPE body), which is what the
dialect tier can honestly claim. Longcat remains rescue-only with no table
derived at all.

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
`TurnSplitter::feed` on the error path as `TurnEvent::ReleasedToolSpan` --
before that release existed the body was dropped and the call was lost
twice over, once to the parser and once to the rescue), and the rescue
parses what is left.

**THE RELEASED BODY BYPASSES THE RAW-TEXT ELIGIBILITY GATE, and that is
not a loophole.** `inspect`'s bare-JSON rescue refuses raw prose unless the
whole visible response is one JSON value, because scanning prose for an
embedded object can erase a warning or a quoted example. A released span
body is not prose: the model bracketed it in tool-call markup that the
detokenizer rendered away, so it carries no marker the eligibility gate
could find and may carry any amount of ordinary prose around the call. The
server records the released bodies beside the reply (`Generated::
released_span_text`, fed by the `ReleasedToolSpan` variant) and tries them
FIRST, on the bracketing's own authority, before the protected raw-text
path. ChatML is the case that pins this: a bare-JSON body inside a
`tool_call` span is native-parser-refused, released, and rescued.

This is why the GLM strategy keys on the `<arg_key>`/`<arg_value>` pair
markup rather than on the `<tool_call>` wrapper the template teaches: the
pairs survive both routes (literal text and wrapper-stripped), the wrapper
survives only one. Before writing a wrapper-anchored rescue pattern, ask
what the text looks like AFTER this engine's own decode.

## Adding a native dialect

`docs/NEW_MODEL.md` is the end-to-end checklist;
`crates/tokenizer/AGENTS.md` Gotcha 11 names the four exhaustive matches a
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
