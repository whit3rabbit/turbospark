# turbospark-tokenizer

Tokenizer wrapper around HF `tokenizers` (`MfTokenizer`), chat dialect resolution (Gemma 4, ChatML/Qwen, DeepSeek-V4, Mistral, Harmony/gpt-oss, Muse Glimmer, Llama-3), chat template rendering (text-only and `minijinja` + `pycompat`), streaming detokenization (`MfDetokenizer`), stop condition matching (`StreamingStopMatcher`), tool call DSL parsers, and streaming structured decoder (`StructuredAssistantDecoder`).

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
|   |   +-- minimax.rs              # MiniMax-M2 dialect and token resolution
|   |   +-- resolve.rs              # Special token probing & dialect resolution
|   |   \-- resolvers.rs            # Per-dialect token resolution routines
|   +-- chat_template/              # Per-dialect text chat renderers (the FALLBACK path)
|   |   +-- mod.rs                  # Entry point & fallback dispatcher
|   |   +-- chatml.rs               # ChatML chat template renderer
|   |   +-- deepseek.rs             # DeepSeek chat template & tool call renderer
|   |   +-- gemma.rs                # Gemma chat template renderer
|   |   +-- llama3.rs               # Llama-3 header-frame chat template renderer
|   |   \-- mistral.rs              # Mistral [INST] chat template renderer
|   +-- jinja_chat_template.rs      # minijinja + pycompat wrapper rendering the checkpoint's own template
|   +-- jinja_compat.rs             # Jinja compatibility syntax rewriter
|   +-- jinja_date.rs               # Standalone pure-ASCII UTC date formatting algorithms
|   +-- detokenizer.rs              # MfDetokenizer for incremental UTF-8 token decoding
|   +-- stop_matcher.rs             # StreamingStopMatcher for evaluating stop sequences and EOS token sets
|   +-- structured_decoder/         # StructuredAssistantDecoder for streaming JSON / structured output
|   |   +-- mod.rs                  # StructuredAssistantDecoder state machine & event types
|   |   +-- chatml.rs               # ChatML thought & tool parsing
|   |   +-- deepseek.rs             # DeepSeek tool parsing
|   |   +-- harmony.rs              # Harmony channel & reasoning parser
|   |   +-- mistral.rs              # Mistral tool call parser
|   |   \-- muse.rs                 # Muse Glimmer recipient-frame parser
|   +-- json_value.rs               # JSON value helper types for tool parameter encoding
|   +-- reasoning.rs                # Reasoning effort configuration and parameter definitions
|   +-- error.rs                    # TokenizerError enum definition
|   \-- tool_call/                  # Dialect-specific tool call DSL parsers
|       +-- mod.rs                  # Module root for tool call parsers
|       +-- gemma.rs                # Gemma tool call DSL parser
|       +-- mistral.rs              # Mistral tool call DSL parser
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
    +-- llama3_dialect.rs           # Llama-3 dialect resolution & fallback renderer tests
    +-- minimax_dialect.rs          # MiniMax-M2 dialect resolution & formatting tests
    +-- mistral_tool_calls.rs       # Mistral tool call DSL and structured decoder tests
    +-- reasoning_effort.rs         # Reasoning effort parameter parsing and template tests
    +-- structured_decoder.rs       # Streaming structured decoder unit tests
    +-- tool_call_support.rs        # Links tool_call_support to what each decoder arm can emit
    +-- tool_calls.rs               # Tool call DSL parser unit tests across Gemma/Qwen/DeepSeek
    +-- vision_markers.rs           # verify_image_markers vision marker id validation tests
    \-- fixtures/                   # Vendored toy tokenizer fixture directories
        +-- ChatMLTokenizer/        # Toy ChatML tokenizer.json fixture
        +-- DeepseekTokenizer/      # Toy DeepSeek tokenizer.json fixture
        +-- GemmaTokenizer/         # Toy Gemma tokenizer.json fixture
        +-- HarmonyTokenizer/       # gpt-oss's special-token NAMES + a minimal Harmony template
        +-- Llama3Tokenizer/        # Llama-3's special-token NAMES, no chat_template (fallback path)
        +-- MuseGlimmerTokenizer/   # muse_glimmer's special-token NAMES + an embedded template
        +-- ReasoningEffortTokenizer/ # Reasoning effort tokenizer fixture
        \-- ZephyrTokenizer/        # Mistral's token table + an embedded Zephyr template
```

## Key Modules

- `dialect/`: Resolves dialect special tokens and chat formatting rules for supported model families. **`detect_dialect`'s ORDER is load-bearing and its Harmony arm requires THREE markers, not two** -- see Gotcha 6.
- `chat_template/`: Per-dialect text chat rendering, plus DeepSeek's hand-rolled native tool chat. The FALLBACK for a checkpoint that ships no template (see Gotcha 1).
- `jinja_chat_template.rs`: Jinja template engine wrapper (`minijinja` + `pycompat`) rendering the checkpoint's own template, for plain text chat as well as tool chat. Calls `jinja_compat::parenthesize_conditional_kwargs`, a REMOVABLE minijinja compatibility shim (AGENTS.md Gotcha 53).
- `detokenizer.rs`: `MfDetokenizer` for incremental UTF-8 token decoding.
- `stop_matcher.rs`: `StreamingStopMatcher` for evaluating stop sequences and EOS token sets.
- `structured_decoder/`: `StructuredAssistantDecoder`, splitting generated output into visible content, reasoning (Harmony, ChatML, Gemma and Muse Glimmer, see Gotcha 4) and parsed tool calls.
- `tool_call/`: Dialect-specific tool call DSL parsers (Gemma, Qwen, DeepSeek).

## Development & Test Commands

```sh
# Run tests for turbospark-tokenizer
cargo test -p turbospark-tokenizer
```

## Crate Gotchas

1. **The Checkpoint's Template Beats the Dialect, and the Template Hides in
   Two Places**. `ChatDialect` is resolved from the
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

7. **A CHATML GENERATION PROMPT OPENS THE `<think>` FRAME ITSELF, so
   `StructuredAssistantDecoder::new` takes the PROMPT and not just the
   tokenizer.** Qwen's own template ends `<|im_start|>assistant\n<think>\n`
   when thinking is on and `<|im_start|>assistant\n<think>\n\n</think>\n\n`
   when it is off. In the ON case the model's FIRST generated token is already
   scratchpad and `think_start_id` never arrives, so a decoder that always
   started in `Channel::Visible` stayed there -- the `</think>` that comes
   later flips Visible to Visible, a no-op -- and reported the whole
   scratchpad as the answer with the reasoning stream EMPTY. Measured on the
   real `qwen38-27b` install at `--reasoning low`: 1,413 bytes of answer and 0
   of reasoning before, 676 and 737 after, the same tokens either way.
   That is Gotcha 4's asymmetry closing one layer down. The ChatML arm was
   taught to EMIT reasoning when `--reasoning` landed, and emitting is
   necessary but not sufficient: the arm also has to be ENTERED, and on this
   dialect the prompt is what enters it.

   **THE PROMPT IS THE SOURCE, NOT THE REASONING LEVEL**, and the difference
   is a failure mode. `prompt_opens_thought` scans the rendered prompt ids
   backwards for the first `think_start_id`/`think_end_id` and reports which
   it met; keying on `reasoning != Off` instead would open the frame on a
   checkpoint whose template enables thinking WITHOUT prefilling the tag, and
   then the model's own `<think>` is a no-op, its `</think>` closes a frame
   that was never its, and the answer arrives as reasoning -- an EMPTY reply,
   which is worse than the bug being fixed. The scan is inert on Gemma,
   Harmony and Muse Glimmer: their `think_*_id` are `None`.

   Backwards, not forwards, because a tool preamble puts balanced
   `<think></think>` pairs in its instructions ("use the `<think></think>`
   block to plan your next tool call") and only the TAIL decides. Note the
   fixture that pins this needs a CLOSED tail to discriminate -- with an open
   tail a forward scan agrees by coincidence, since a balanced pair starts
   with `<think>` too, and the test passes against both directions
   (AGENTS.md Gotchas 48, 50, 51 on a fourth axis; the mutation check is what
   caught the first version of it).

   Found while reviewing `froggeric/Qwen-Fixed-Chat-Templates`, a community
   Qwen template rewrite, and it is a bug in THIS port rather than anything
   that repo fixes -- its generation prompt ends the same way Qwen's does.
   That template renders cleanly through `minijinja` + `pycompat` with no new
   shim (`[::-1]`, `[:n]`, `startswith`, `split`, `lstrip` all work) and its
   default `xml` tool format is what `QwenToolCallParser` already expects, so
   nothing here blocks adopting it; the reasons not to are architectural
   (Gotcha 1: the checkpoint's template wins, and this port ships
   none) rather than technical.

   **THE SWEEP IS THE REUSABLE PART.**
   `every_dialects_decoder_starts_where_its_own_prompt_left_the_model` renders
   every bundled fixture at every level its template accepts, scans the prompt
   ids independently, and asserts the decoder's first event agrees. It is not
   `#[ignore]`d and needs no install, so a NEW dialect gets SWEPT by existing
   -- which is the whole reason the ChatML case survived: the question had only
   ever been asked per dialect, by hand. **What it checks is the `<think>`
   pair's invariant specifically**, so a dialect framing its reasoning
   otherwise is included without being tested; Muse Glimmer passes it by having
   no think ids at all, and its own frame is pinned by the four `muse_*` cases
   instead. Reverting `new` to an
   unconditional `Channel::Visible` reddens it. Measured across the seven
   fixtures: ChatML OPEN at every level and closed at `off`, Gemma opening no
   channel at a level and pre-closing an empty one at `off`, Harmony ending at
   `<|start|>assistant` outside any frame, DeepSeek pre-closing with
   `</think>`, Mistral with no thought frame at all. **ChatML was the only one
   wrong**, which is worth knowing before hunting for siblings.

   **THE SWEEP FOUND A SECOND BUG, IN A DIFFERENT MECHANISM.** `muse_glimmer`
   had no decoder arm at all (`consume` grouped it with Mistral) and no caller
   built one for it, so its `to=self` scratchpad printed as the reply. It is
   NOT the ChatML shape: what was missing was the whole frame, not an initial
   state. See Gotcha 8.

8. **MUSE GLIMMER REASONS ON EVERY TURN, so its decoder arm is built
   unconditionally like Harmony's rather than behind `--reasoning`.** Its
   template calls `render_reasoning()` from the system message with no gate and
   defaults the strength to `high` when the caller sets nothing, so a plain
   `--messages-file` run with NO reasoning flag already asks for reasoning.
   Before this arm existed, EVERY museGlimmer generation this port produced
   printed the scratchpad as the reply: measured on the real 30B install with
   no flag, stdout opened ` to=selfExplain how coastal wetlands reduce flood
   damage.` followed by the scratch work, with 0 bytes on the reasoning stream.
   After: 1,737 bytes of reasoning on stderr and an answer that starts at
   `Coastal wetlands are a natural flood defense`.

   **THE FRAME IS `<|start|>ROLE to=RECIPIENT<|message|>BODY<|eom|>`, and the
   RECIPIENT IS THE CHANNEL.** That is the one structural difference from
   Harmony, whose first header word is a channel NAME with the recipient an
   optional extra. `to=user` (and a header naming no recipient, which the
   template defaults to `user`) is the answer; `to=self` is the scratchpad;
   anything else is reasoning, which is Harmony's default direction and routes
   this dialect's `to=<toolname>` calls too -- the `<atem:function_calls>` body
   is PLAIN TEXT with no parser wired, so unparseable markup on the reasoning
   stream beats it appearing as the reply.

   Three things that follow. **`<|start|>` lives in `channel_start_id`**, which
   is the field Harmony gives `<|channel|>` for the same job; `channel_end_id`
   stays the sentinel so the Gemma-shaped bracketing arm cannot become
   reachable. **`<|eot|>` is absent from the state machine on purpose** -- it is
   a stop token, so the loop breaks before the callback and it never arrives
   (AGENTS.md Gotcha 49); only `<|eom|>`, which hands off rather than ending the
   turn, reaches `consume`. And **the initial state comes from the PROMPT**,
   exactly as Gotcha 7 requires for ChatML: `add_generation_prompt` emits
   `<|start|>assistant` and STOPS, so a real generation begins inside a header
   and no `<|start|>` ever arrives. Starting unframed passes the header
   remainder (` to=self`) through as prose and then the whole scratchpad as the
   reply -- which is precisely the bug, so the two fixes share one rule.

9. ~~**THERE IS NO LLAMA-3 DIALECT, so no Llama-3 checkpoint loads.**~~
   **LANDED, see Gotcha 10.** `detect_dialect`'s fallback used to be Gemma,
   and Llama-3's table (`<|begin_of_text|>` / `<|start_header_id|>` /
   `<|eot_id|>`, no `<s>`, no `<|im_end|>`) matched no positive probe -- so it
   landed on Gemma and `resolve_gemma` failed on a missing `<pad>`. A
   tokenizer gap, not an architecture one: `turbospark-model probe` reports
   `Meta-Llama-3-8B-Instruct` RUNNABLE (32 layers, hidden 4096, Q4_K/Q6_K, no
   `rope_freqs.weight`). It failed before any weight byte streamed, sidecars
   being verified first, so it cost seconds rather than a re-stream.
   NUMBERED 9 AND NOT 7 on purpose: 7 and 8 were in flight in another
   session's working copy when this landed, so the gap was transient and
   closed when that commit arrived.

10. **`ChatDialect::Llama3` CLOSES GOTCHA 9, AND ITS SHARED-MARK TRAP IS THE
   SAME ONE GOTCHA 6 NAMES FOR HARMONY/MUSE GLIMMER, ARRIVING ON A THIRD
   PAIR.** Meta's Llama-3 family (base and Instruct) shares
   `<|begin_of_text|>` and `<|end_of_text|>` with `muse_glimmer` and nothing
   else, so `detect_dialect` keys on this family's OWN frame markers
   (`<|start_header_id|>` and `<|eot_id|>`) rather than the shared pair --
   neither string collides with Muse Glimmer's `<|start|>` / `<|message|>` /
   `<|eot|>` (no `_header_id` / `_id` suffix on any of those three), so the
   two probes stay disjoint whatever order they run in.

   No tool-calling or thinking markup: the base 8B-Instruct table this was
   built and probed against (`meta-llama/Meta-Llama-3-8B-Instruct`, per
   `docs/OBLITERATION.md`'s Open section, which is what this unblocks) has
   none, so every such id is `NO_SUCH_TOKEN_ID`, Mistral's sentinel for the
   same reason. A 3.1-family checkpoint's `<|eom_id|>` (message handoff,
   mirroring Muse Glimmer's `<|eom|>`) and `<|python_tag|>` (built-in tool
   call) would need their own arm; nothing here has been measured against
   one, and no 3.1 checkpoint has been probed.

   **UNLIKE HARMONY AND MUSE GLIMMER, THIS DIALECT GETS A FALLBACK RENDERER**
   (`chat_template/llama3.rs`), on the same reasoning Mistral's has one: the
   reference template is a handful of markers around the content (one
   `<|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>` block per
   message, no system preamble, no tool namespace), not a large second
   implementation of something complex. It DOES literally emit
   `<|begin_of_text|>`, unlike Mistral's fallback (that file's own Gotcha:
   "it emits no `<s>`... this gap is inert" because Mistral's real template
   always wins) -- `resolve_llama3` sets `bos_prefix_id: None` to match, so
   `encode(_, add_bos: true)` never doubles it. A real install still always
   ships its own template and takes the Jinja path, so
   this fallback is what a malformed install gets, same as every other
   dialect's.

11. **ADDING A `ChatDialect` VARIANT TOUCHES FOUR EXHAUSTIVE MATCHES, all
    compiler-enforced but one.** `dialect::resolve::resolve_dialect`,
    `chat_template::apply_dialect_chat_template`,
    `chat_template::encode_text_continuation`, and
    `structured_decoder::consume`'s dialect match are all non-wildcard, so a
    missing arm is a build error rather than a runtime panic.
    **NOT COMPILER-ENFORCED, AND THERE IS EXACTLY ONE OF THEM SINCE
    2026-09-05:** `runtime::turn_stream::TurnSplitter::new` keys on the
    dialect through `matches!(dialect, A | B)`, which compiles fine with a
    new variant matching neither arm -- decide by hand whether the new
    dialect belongs in those unions. `ChatDialect::Llama3` needed none of
    them (no tool-calling or thinking markup to decode), which is why it is
    not the worked example for that half.

    **THIS ENTRY USED TO NAME THREE SITES** --
    `server::handler::exec::needs_decoder`, `cli::generate::format`'s
    `ChannelSplit` and `ffi::generate`'s port of it -- which is what the
    `TurnSplitter` refactor collapsed. Worth knowing because the failure
    mode of a stale list here is the quiet one: a reader follows it to three
    files, finds no dialect match in any of them, correctly concludes
    nothing needs doing, and misses the one place that does.

12. **A MULTIMODAL MESSAGE RENDERS THROUGH A CONTENT-PART LIST, AND A
    TEXT-ONLY ONE STILL RENDERS THROUGH THE BARE STRING** (ROADMAP M-V6).
    `Message::content_parts` is EMPTY on every text message, and
    `message_to_json` branches on that: empty takes the `content is string`
    arm every template has always taken, non-empty emits HF's
    `[{"type": "image"}, {"type": "text", ...}]` list. That is what leaves
    every frozen digest in `crates/bench` where it is, and
    `a_text_only_message_renders_identically_through_both_constructors` pins
    it.

    **The shape is the TEMPLATE's, not this port's invention.** qwen3_5's
    `render_content` macro branches on `content is string` first and falls
    through to `content is iterable`, testing each item for an `image` key or
    `item.type == 'image'`. Emitting a bare string with the markup spelled
    into it would take the TEXT arm and tokenize the angle brackets rather
    than the special token -- a prompt of roughly the right length carrying
    none of the right ids.

    **The parts are ORDERED and the order is load-bearing.** The template
    emits the marker run where the part sits, so
    `[Image, Text(q)]` and `[Text(q), Image]` are different prompts and the
    second moves every mRoPE position past the image.

    **`add_vision_id` stays hardcoded `false` and that is Phase 0's finding
    rather than an omission**: it controls only an optional `"Picture N: "`
    prefix and defaults to falsy upstream, so `false` is what the reference
    sends for the unlabeled case. Threading a context variable would change
    the rendered bytes of every image prompt to match no reference.

    **THE FALLBACK RENDERERS REFUSE AN IMAGE RATHER THAN DROPPING IT.** None
    of them emits a vision marker, so a multimodal message would render as its
    text alone -- and then the id sequence has no `<|image_pad|>` for the
    splice to expand, `PromptVision` gets spans that do not exist, and the
    model answers about a picture it never saw. Every real vision install
    ships its own template, so this is what a MALFORMED install gets, exactly
    as the Harmony and Muse Glimmer refusals are.

    Adding a `ContentPart` variant (video is the obvious next one) touches
    `message_to_json`'s match and nothing else that is compiler-enforced;
    `Message::image_count` and the fallback refusal both key on `Image`
    specifically, so decide by hand whether a new variant belongs in them.

13. **`accepted_reasoning_levels` DEDUPES BY RENDERED BYTES, AND "DID IT
    RAISE" ALONE WOULD BE THE WRONG QUESTION.** A GUI cannot build a level
    menu from `reasoning_support` (three coarse states) and must not build one
    from the family (root Gotcha 56), so this ASKS the template: render a
    one-message conversation at each of `ReasoningEffort::ALL` and report what
    came back. Qwen 3.8 answers `[off, low, medium, xhigh]` and raises on
    `high`, gpt-oss and Muse Glimmer answer `[off, low, medium, high]`.

    **The raise is only half the signal.** A template that SHIPS but names no
    reasoning key renders happily at every level and produces IDENTICAL bytes
    -- `HarmonyTokenizer` and `ZephyrTokenizer` are that shape -- so a
    raise-only probe reports five accepted levels of which four are silent
    no-ops, which is precisely the failure `ReasoningSupport` exists to
    prevent, re-introduced by the thing meant to refine it. Collapsing levels
    whose render is byte-identical to an earlier one is one rule that lands
    every support shape correctly: `None` reports `[Off]` whether it has a
    template or not, `ToggleOnly` reports `Off` plus ONE on-level, and `Level`
    reports what the template really distinguishes.

    Three things for a caller. The result is never empty and always opens at
    `Off` (it fails OPEN, in the direction `reasoning_support`'s own doc
    argues for). `ToggleOnly`'s on-level is `Low` BY POSITION and is not a
    label. And it is an OPEN-time call, five renders of two lines, never a
    per-turn one.

    **`ReasoningEffort::ALL` is what the sweep iterates, so a spelling missing
    from it is never offered by any template -- and no fixture here can see
    that.** Found by mutation: dropping `High` from `ALL` left every case in
    `tests/reasoning_effort.rs` green, because `ReasoningEffortTokenizer` is
    the only `Level` fixture and it REJECTS `high`. The spelling only matters
    on Harmony and Muse Glimmer, which are not fixtures here.
    `the_probe_sweeps_every_spelling_the_parser_accepts` ties `ALL` to `parse`
    for that reason rather than restating five names.

14. **CHAT TEMPLATES ARE UNTRUSTED PROGRAMS WITH OUTPUT TOO.** A tokenizer
    sidecar can consume unbounded template instructions or emit unbounded text
    before the model ever runs. Set a VM fuel limit and render into a bounded
    writer, then keep a regression for both an output flood and a nested-loop
    workload. Do not rely on the output cap alone, because a loop can burn CPU
    without producing many bytes.

15. **THE JINJA COMPAT SHIM SCANS BYTES OF PROSE, SO IT OWES TWO PROPERTIES.**
    `jinja_compat.rs::parenthesize_conditional_kwargs` walks the template as
    bytes, and templates are mostly multibyte prose (Spark 2.5's ships a
    non-ASCII version comment like `{#- 0826<banben> -#}` and fullwidth-bar markers). The scan must only
    ever slice at `{` block boundaries -- `{` is ASCII, so `i` stays on a
    char boundary and whole-span copies are safe; a rewrite that advances by
    "one character" computed from a byte turns every non-ASCII byte it
    touches into mojibake, and that regression is invisible on ASCII
    fixtures. Second, the pass must stay LAZY: it exists for one template
    shape (a conditional keyword argument minijinja rejects), so it returns
    `Cow::Borrowed` and allocates nothing for every template that needs no
    rewrite. Making the scan eagerly copy every render taxes all of them for
    the sake of one, and nothing but a benchmark will notice. Keep both
    regressions: a multibyte template asserting the shim is byte-preserving,
    and a no-rewrite template asserting the borrowed path.
