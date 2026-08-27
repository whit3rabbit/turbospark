# turbospark-server

HTTP server (`turbospark-server`) on Axum, speaking two wire formats against one
local backend: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and
`/v1/models`. Both generation endpoints support non-streaming and Server-Sent
Events (SSE) streaming output.

The wire types are `anyllm_translate`'s (crates.io 0.16, default features:
pure, IO-free), not hand-rolled. An Anthropic request is translated into the
OpenAI request the existing path already understands, run through the shared
generation core in `handler/`, and translated back.

## Directory & File Structure

```
crates/server/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- main.rs                 # Binary entry point for turbospark-server
|   +-- args.rs                 # Command line parsing and host binding resolution
|   +-- main_tests.rs           # Unit tests for CLI args and host binding
|   +-- lib.rs                  # Library root: router, re-exported wire types
|   +-- handler/                # /v1/chat/completions + /v1/models, and the shared generation core
|   |   +-- mod.rs              # The two Axum handlers and the router wiring
|   |   +-- plan.rs             # `plan`: chat template, encode, shaping config
|   |   +-- exec.rs             # `run_full` and `stream_blocking`
|   |   \-- tests.rs            # Unit tests for the two above
|   +-- messages.rs             # Anthropic /v1/messages: translate in, generate, translate out
|   +-- guardrails.rs           # Tool-call rescue, argument validation, the retry loop
|   |   \-- tests.rs            # Unit tests for the verdict (pure, no model)
|   +-- model.rs                # ChatModel trait and the ScriptedChatModel backend
|   +-- real_model.rs           # RealChatModel: RealForwardRunner backend (macOS only)
|   +-- response.rs             # Constructors for the OpenAI response & SSE chunk envelopes
|   \-- response_tests.rs       # Unit tests for response and chunk serialization
\-- tests/
    +-- chat_completions.rs     # Integration tests for the OpenAI endpoint
    +-- messages.rs             # Integration tests for /v1/messages, /v1/models, wider OpenAI shapes
    +-- harmony_channels.rs     # gpt-oss reasoning -> thinking/reasoning_content, and its tool calls
    +-- reasoning_channels.rs   # Streaming & non-streaming reasoning channel translation tests
    +-- guardrails.rs           # Rescue/validate/retry end to end, both endpoints, no model
    +-- real_backend.rs         # Gated real-model end-to-end test (macOS, #[ignore]d)
    \-- fixtures/               # Test tokenizer fixtures for integration tests
```

## Key Modules

- `main.rs`: Server binary entry point and CLI option handling (`--model` real mode, legacy positional scripted mode, port, `--bind loopback|tailnet`).
- `handler/`: the `/v1/chat/completions` and `/v1/models` handlers, plus the generation core both endpoints share -- `plan.rs` (chat template, encode, shaping config) and `exec.rs`'s `run_full` and `stream_blocking` (which owns the `StructuredAssistantDecoder` when a request carries tools).
- `messages.rs`: the Anthropic `/v1/messages` handler, wrapping the same core in `translate_request` / `translate_response` / `new_stream_translator`.
- `guardrails.rs`: tool-call rescue parsing, argument validation against the request's own schema, and the one-retry loop, over `forge-guardrails` (see Gotcha 18). `inspect` is the pure verdict; `run_guarded` is the loop that acts on it.
- `model.rs`: the `ChatModel` trait and `ScriptedChatModel`, bridging Axum handlers to `turbospark-runtime`. The trait owns WHICH decode loop runs (`run_completion`, Gotcha 17), not just which producer.
- `real_model.rs`: `RealChatModel`, a `RealForwardRunner` behind the same trait (macOS only).
- `response.rs`: constructors for `anyllm_translate::openai`'s response and SSE chunk envelopes, filling the many fields this server never populates in one place.

## Development & Test Commands

```sh
# Run tests for turbospark-server
cargo test -p turbospark-server

# Smoke the two generation endpoints against a running --model server.
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":120,"messages":[{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":40,"stream":true,"messages":[{"role":"user","content":"hi"}]}'

# The point of /v1/messages: an Anthropic-native client, no proxy.
ANTHROPIC_BASE_URL=http://127.0.0.1:8080 ANTHROPIC_API_KEY=unused \
  CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=true claude

# Launch against a real .gturbo install (macOS; use --release, a debug
# build decodes far too slowly to be usable). `--model` takes a directory
# or a `turbospark-model` alias.
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo [--port N] [--max-context N|auto] [--expert-cache-slots auto|N] \
  [--bind loopback|tailnet] [--power-profile performance|balanced|efficiency] \
  [--max-tokens-per-sec R] [--speculative off|auto|N] [--speculative-drafter auto|mtp|dflash] \
  [--guardrails on|off]
cargo run --release -p turbospark-server --bin turbospark-server -- --model gemma4

# Launch the portable scripted server (tokenizer only, canned completions).
cargo run -p turbospark-server --bin turbospark-server -- <tokenizer-dir> [port]

# The gated real-model end-to-end test.
TURBOSPARK_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p turbospark-server --test real_backend --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Two backends, and the real one is serialized.** `RealChatModel` (macOS only) owns ONE `RealForwardRunner` behind a `Mutex`: it costs a multi-gigabyte mapping plus a Metal pipeline compile to open and takes `&mut self`, so concurrent requests queue rather than run. `ScriptedChatModel` (`ScriptedLogitProducer`, canned completions) stays the portable backend the integration tests drive, and is the only one on non-macOS. That is why `ChatModel` exposes `with_producer` (lend a producer) rather than a `new_producer` factory. **`ScriptedChatModel` can force an EXACT token sequence**, which is how a dialect's markup is proven end to end with no model at all: one `one_hot(vocab, id)` per step, with the first `prompt_ids.len() - 1` steps consumed by prefill (`produce_prefill`), so the decode steps must start at exactly that offset or the generation begins mid-sequence. `tests/harmony_channels.rs` is the worked example, and it reconstructs the prompt length the same way `plan` does (the checkpoint's own template, `encode(&prompt, false)`).
2. **`encode(&prompt, false)`.** The chat template already emits the literal `<bos>`, so encoding the rendered prompt with a BOS prefix doubles it. Invisible on the ChatML test fixture (its `bos_token` is null); on a real Gemma install it degrades output.
3. **`top_k` defaults to 64, not 0.** `ShapingConfig::new` rejects `top_p < 1.0` when `temperature > 0` and `top_k == 0`, so a 0 default would 400 every standard OpenAI request that sets only `top_p`.
4. **The `anyllm_translate` `middleware` feature is NOT usable here, and the reason is not obvious.** It looks like exactly the right tool (it ships an `anthropic_compat_router` that adds `POST /v1/messages`), but `AnthropicCompatConfig` requires a `backend_url` and forwards over `reqwest` -- this server's backend is in-process, so enabling it would mean HTTP-hopping to ourselves on loopback. It also depends on **axum 0.8** while this crate is on **0.7**, which would put two incompatible `Router` / `Sse` / `Event` types in one binary. Use default features and call `translate_request` / `translate_response` / `new_stream_translator` directly. The consequence: `stream_event_to_sse` lives behind that feature and returns an axum-0.8 `Event`, so `messages.rs` hand-rolls the same ten-line variant-to-name match. **Corollary: READ THE VENDORED SOURCE before implementing a wire shape.** `~/.cargo/registry/src/*/anyllm_translate-0.16.0/src/` answers "does it already do this" in one grep, and for Harmony channel decoding the answer was yes end to end -- `Delta.reasoning_content`, `ChatMessage.reasoning_content`, and both mappings onto `thinking` blocks already existed -- so a planned wire-format implementation became filling one field.
5. **`seed` has no field on the OpenAI request type; it lands in `extra`.** `anyllm_translate`'s `ChatCompletionRequest` gives explicit fields only to what needs translating, and sweeps the rest (`seed`, `logprobs`, `n`, `logit_bias`, ...) into a `#[serde(flatten)] extra` map. `build_config` reads it back out with `request.extra.get("seed")`. Drop that lookup and every seeded request silently becomes unseeded -- no deserialization error, no test failure unless one asserts determinism.
6. **Anthropic SSE is not OpenAI SSE with different JSON.** Clients dispatch on the `event:` line, which the OpenAI stream never sets, and the stream ends at `message_stop` with **no `[DONE]` sentinel**. Also, `StreamingTranslator` is a state machine over the OpenAI chunk *sequence*: it opens the message on the first chunk carrying a role and closes the content block on `finish()`. Feed it chunks in a different order than the chat route emits (role chunk, content chunks, finish chunk) and the Anthropic event structure comes out malformed rather than erroring.
7. **A request with `tools` takes a DIFFERENT prompt path, and the split is deliberate.** `plan` renders through the checkpoint's own `chat_template.jinja` (`encode_generic_tool_chat`) when `tools` is non-empty, and through `apply_chat_template` otherwise. Only the Jinja template can express tools, and it opens a thought channel in its generation prompt -- which is why the same condition also switches `stream_blocking` onto `StructuredAssistantDecoder`, whose channel tracking swallows that. Wire one without the other and either the tools go unrendered or the model's thought text streams to the client as content. The text-only path is byte-identical to what it was before tools existed; `handler::tests::no_tools_keeps_the_text_only_template` is the guard.
8. **`StopReason::ToolCalls` needs a dialect whose STOP SET says "tool", and only two have one.** It fires on `tokenizer.tool_call_stop_id`: Gemma's `<|tool_response>` and Harmony's `<|call|>`. It used to read `tool_response_id` directly, which is Gemma's alone, so a `gpt-oss` call reached clients as `finish_reason: "stop"` -- a wrong answer that looks like nothing, since the `tool_use` block was there beside it. ChatML has none (`[im_end, endoftext]`; it closes a call with `</tool_call>` and then ends the turn normally), so the ChatML fixture can prove the decoder parses a call out of the stream while the finish reason it reaches is `EndOfTurn`. `tests/harmony_channels.rs` covers the Harmony half end to end with no model. Gemma's half still belongs to the real-model gate:
   ```sh
   curl -s localhost:8123/v1/messages -H 'content-type: application/json' -d '{
     "model":"claude-sonnet-4-6","max_tokens":300,"temperature":0.2,
     "tools":[{"name":"get_weather","description":"Get the current weather in a given city",
               "input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
     "messages":[{"role":"user","content":"What is the weather in Oslo right now? Use the tool."}]}'
   # -> content[0].type "tool_use", stop_reason "tool_use"
   ```
   Feed the result back as a `tool_result` block to check the other half: that turn only renders correctly because `plan` carries `tool_call_id` across.
10. **Rate control is process-level, and that is a decision rather than an omission.** `--power-profile` / `--max-tokens-per-sec` are resolved ONCE in `main.rs` (which is also the only place this process asks the OS about Low Power Mode) and reach `GenerationConfig` through `ChatModel::rate_control`, applied in `plan` after `build_config`. There is deliberately no per-request field: there is one runner per process, a power setting is a property of the machine rather than of a caller's prompt, and a request that could pick its own rate would let any client opt out of the machine's power policy. The consequence to know is that a cap lengthens the runner mutex hold in proportion, so a capped server queues concurrent requests for longer -- acceptable only because Gotcha 1's queue is already serial.

15. **`--help` and `--version` are handled BEFORE the flag loop, and a
   flag-led invocation can no longer fall through to the scripted mode.** Both
   short-circuits take no value while that loop advances two tokens per flag,
   so reaching them there would consume whatever followed. The mode test was
   also `args[0] == "--model"` exactly, which meant
   `turbospark-server --port 8080 --model X` was read as a positional
   TOKENIZER DIRECTORY named `--port` and died with
   `failed to load tokenizer: ... No such file or directory` -- a filesystem
   error about a path nobody typed. Anything starting with `-` now belongs to
   `parse_model_args`, so a mis-ordered or misspelled flag gets the usage
   text. `a_flag_in_any_position_stays_out_of_the_scripted_mode` pins both
   directions, including that a real positional path still reaches the
   scripted mode.

   The version string comes from `CARGO_PKG_VERSION` rather than from
   depending on `turbospark-invocation` for its `render_version`: every crate
   inherits `version.workspace = true`, so the two are the same string, and
   `the_version_line_matches_the_clis` is what keeps them spelled the same.

16. **`--max-context` defaults to `auto`, and what makes an
   environment-sensing default safe is that an install declaring no trained
   context resolves to 4,096.** `RealChatModel::open` takes it as a POLICY
   (`Option<u32>`, `None` meaning auto) exactly as it takes the slot count,
   resolves it through `runtime::resolve_max_context` BEFORE allocating, and
   exposes the result as `context_plan()` for the startup line. Two failure
   modes are deliberately different: past the checkpoint's trained context is
   a WARNING at startup (RoPE extrapolates rather than failing, and an install
   written before that field existed declares none, so refusing would be
   enforced on some installs and not others), while past what memory holds is
   a REFUSAL carrying the whole subtraction -- otherwise it surfaces as a
   Metal allocation failure with no number in it pointing at the flag.
   `tests/real_backend.rs` pins BOTH sized knobs rather than letting either
   sense the machine, for AGENTS.md Gotcha 35's reason.

11. **`--bind tailnet` fails rather than widening, and is not authentication.** It binds only the single address `tailscale ip -4` reports, and only when that address is a dotted-quad inside 100.64.0.0/10; zero, several, IPv6, out-of-range, or malformed output is an error, never a fall back to loopback or a wildcard. Every device the Tailnet ACL admits gets unauthenticated access to the full API (no auth, no TLS). The flag exists only in `--model` mode; the scripted `<tokenizer-dir>` mode always binds loopback.

12. **The decoder is built for THREE independent reasons, and keeping them independent is the point.** `stream_blocking` used to construct `StructuredAssistantDecoder` exactly when a request carried tools, which is the same condition `plan` uses to pick the tool template (Gotcha 7). `needs_decoder` now also fires for `ChatDialect::Harmony`, because `gpt-oss` writes its reasoning into an `analysis` channel BEFORE its answer and an undecoded stream sends the reasoning and the frame markup to the client as the reply. The THIRD condition is `reasoning_effort` on a ChatML or Gemma request: those dialects have a thought channel that is unreachable until a level is asked for, and reachable the moment one is. It is keyed on the REQUEST rather than the dialect, which is what keeps every existing call on the pass-through path it has always had. **The prompt path stays keyed on `tools` alone.** Coupling them again would render the tool template for every gpt-oss request, which changes the prompt rather than the presentation. The split reaches the wire as one field: `assistant_message` fills OpenAI `reasoning_content` and `reasoning_delta` fills the streaming equivalent, and `anyllm_translate`'s existing mappings turn both into an Anthropic `thinking` block. `thinking_blocks` stays `None` deliberately: that field carries Anthropic's SIGNED blocks, and a local model has nothing to sign with. One inherited ordering constraint, recorded in `reasoning_delta`: the streaming translator opens a thinking block on the first reasoning delta and closes it on the first content delta without reopening, so a backend that interleaved the two would need upstream work rather than a reordering here. Harmony emits analysis before final, so this server's sequence is well formed. `tests/harmony_channels.rs` proves the whole chain with the scripted backend and no model; mutation-checked by reverting `needs_decoder` to the tools-only condition, which reddens all three cases.

14. **`stream_blocking` MUST call `decoder.finish()`, and for a year it did not.** A Harmony tool call is terminated by `<|call|>`, which is a stop token, so `run_raw_completion` breaks before the progress callback and the decoder never sees the token ending the span it is parsing -- the call comes out of `finish` (tokenizer crate Gotcha 5). Skipping it dropped every `gpt-oss` call silently: no error, no markup on the wire, just an assistant turn with nothing in it. The call is emitted AFTER `run_raw_completion` returns, which is why `tests/harmony_channels.rs` asserts the streamed `tool_calls` delta arrives BEFORE the finish chunk rather than only that it arrives. Two side notes. It is gated on the run having SUCCEEDED, since a failed request reports the error rather than a partial turn. And it also releases the tail DeepSeek's arm withholds as a possible tool-marker prefix, which this loop used to truncate -- a fix that came free with the call and touches no other dialect (the Gemma, ChatML and Harmony arms never fill `held_text`).

13. **`--model` resolves a catalog alias, and `catalog` is a macOS-only dependency on purpose.** `open_real_model` passes the argument through `catalog::resolve_model_arg` -- the same one behind `turbospark-check --model` -- so an install answers to one name whichever binary opens it, with an existing directory always winning over an alias (a bare name that preferred an alias would serve a DIFFERENT model than the command line named, and this binary runs unattended). Two things follow. The dependency is gated to `cfg(target_os = "macos")` beside `repack`, which costs nothing because `--model` is refused before resolution on every other platform, and because `catalog`'s dependency set is a subset of `repack`'s plus `tokenizer` -- adding it pulls in **no new external crate**. And the guard test (`an_alias_resolves_to_its_install_directory`) asserts on a name that is NOT a directory, deliberately: for a real path `resolve_model_arg` and `PathBuf::from` agree, so every path-shaped case passes under the mutation that removes resolution entirely. It is also the only test in this binary that touches `TURBOSPARK_HOME`, which is what makes it safe under parallel test threads.

17. **SPECULATION IS PROCESS-LEVEL LIKE THE RATE CAP, BUT ITS SECOND
   CONDITION IS PER REQUEST -- and that is the one thing this server cannot
   inherit from the CLI.** `--speculative` and `--speculative-drafter` are
   parsed in `args.rs`, resolved ONCE in `RealChatModel::open` through
   `runtime::speculation_policy` (the same three functions `open_session`
   calls), and reported on the startup line. That much is Gotcha 10's shape
   exactly: there is one runner per process and the drafter's state is
   allocated at open, so there is nothing a request could switch.

   What differs is the OTHER input. Acceptance is `argmax(target) ==
   proposal`, exact only at temperature 0. On the CLI that is a property of
   the process, so `open_session` refuses once and is done. Here every
   request carries its own temperature, so `open` resolves the INSTALL half
   as though the request were deterministic and `run_completion` applies the
   per-request half. **A sampled request falls back to the sequential loop
   SILENTLY** -- it is the normal case (OpenAI and Anthropic clients send a
   non-zero temperature by default), and a per-request warning for the normal
   case is noise that trains an operator to ignore the startup line that
   matters. Refusing would be worse: a 400 for a setting the caller never
   sent.

   **The practical consequence to tell an operator: a server started with
   `--speculative` speculates on a MINORITY of its traffic.** That is a
   limitation rather than a bug, and it is the same refusal the CLI makes.

   **`ChatModel::run_completion` exists because the speculative loop cannot be
   reached through `with_producer` at all.**
   `run_raw_completion_speculative` is generic over
   `runtime::SpeculativeProducer`, which carries an associated `Checkpoint`
   type and is therefore not object-safe, while `with_producer` hands out a
   `&mut dyn LogitProducer`. Only a backend holding the CONCRETE runner can
   call it, so the loop CHOICE belongs to the backend. The trait method is
   DEFAULTED to the sequential loop, which is why `ScriptedChatModel` needed
   no change and every integration test kept its exact path;
   `with_producer` stays as the primitive that default is written on.

   `tests/real_backend.rs` pins speculation OFF rather than letting `auto`
   sense the install, for AGENTS.md Gotcha 35's reason and alongside the two
   sized knobs Gotcha 16 already pins: `Auto` reads the resident index, so a
   gate left on it would decode speculatively or sequentially depending on
   which drafter the install that env var points at happens to carry.

18. **TOOL-CALL GUARDRAILS ARE ON BY DEFAULT, AND A REQUEST CARRYING TOOLS IS
   BUFFERED RATHER THAN STREAMED WHILE THEY ARE.** `src/guardrails.rs` wraps
   `forge-guardrails` (crates.io, `default-features = false`): it rescues a
   call the `StructuredAssistantDecoder` could not parse out of the raw text,
   checks a parsed call's arguments against the schema the request sent, and
   re-asks ONCE with a nudge. `--guardrails off` restores the previous path
   exactly.

   **The buffering is inherent, not an implementation shortcut.** A verdict
   needs the whole turn: a call worth rescuing is one the decoder did not
   parse, so it is indistinguishable from prose until the turn ends, and a
   retry re-generates from scratch. Emit deltas live and both repairs are
   already on the wire. So `stream_response` forks -- a tool-carrying request
   goes to `buffered_stream_response`, everything else keeps the live
   `stream_blocking` path byte for byte. The cost is time-to-first-token
   becoming time-to-last-token for a tool turn (~5-10 s at 20-45 tok/s), taken
   knowingly: the alternative does nothing for Claude Code, which sends
   `stream: true` WITH tools, and a `forge-guardrails-proxy` in front would
   buffer identically. The condition is keyed on the REQUEST carrying tools,
   the same shape as Gotchas 7 and 12.

   **THE RETRY RE-RENDERS THE PROMPT rather than appending tokens**, which is
   why `run_guarded` takes the whole `ChatCompletionRequest` and not the
   planned `prompt_ids`. A nudge has to arrive as a `user` turn after the
   failed `assistant` turn, through the checkpoint's own template; appended
   tokens land inside whatever channel the model was last writing in and are
   wrong on every dialect. The budget is ONE, because Gotcha 1's queue is
   serial and a second generation doubles the worst-case mutex hold.

   **RESCUE COMPOSES WITH VALIDATION, and getting that backwards was a real
   bug this crate's own test caught.** The first version returned a rescued
   call unchecked, which is exactly inverted: a call recovered from markup the
   decoder could not parse is the one MOST likely to have arguments the model
   also got wrong. `inspect` rescues first and then validates the result, and
   a rescued call that fails validation is re-asked rather than sent --
   `an_invalid_call_is_retried_and_the_second_answer_wins` is the guard, and it
   failed on the first run by returning a call with empty arguments.

   **A retry cannot be tested with `ScriptedChatModel`**, and the reason is
   worth knowing before someone tries: `with_producer` rebuilds the producer
   from the same steps on every call, so it replays one answer forever.
   `tests/guardrails.rs` carries `TwoTurnModel`, whose producer also overrides
   `produce_prefill` to a no-op -- that decouples the script from the PROMPT
   LENGTH, which matters because a retry's prompt is longer than the first
   (it carries the failed turn plus the nudge) and a `ScriptedLogitProducer`
   sized for the first generation runs out mid-prefill on the second.

   `--guardrails` is process-level like the rate cap and speculation, on the
   same reasoning (Gotchas 10 and 17) plus one of its own: a per-request field
   would let any client opt its own traffic out of the repair the deployment
   chose. `tests/real_backend.rs` pins it OFF for Gotcha 35's reason.

   **THIS IS THE ONE CRATE IN THE WORKSPACE THAT OVERRIDES `rust-version`.**
   `forge-guardrails` declares 1.87 and the workspace floor is 1.82, so
   `crates/server/Cargo.toml` spells `rust-version = "1.87"` literally rather
   than inheriting. Overriding here rather than raising the workspace keeps
   the claim true for the other fifteen crates, none of which has that floor.
   Nothing in this repo was taking 1.82 literally anyway -- `rust-toolchain.toml`
   pins `stable` -- but a declared floor a crate cannot honour is the kind of
   silent-and-wrong this file exists to prevent. Note the 1.87 belongs to the
   `guardrails` feature: that crate's DEFAULT features need 1.95, which is
   exactly why `default-features = false` is not merely a size decision.

19. **CHUNKED PREFILL IS AUTOMATIC HERE SINCE 2026-08-26, WITH NO PER-REQUEST
   OR PROCESS FLAG AT ALL.** `RealChatModel::run_completion`'s non-speculative
   branch locks the runner and checks `RealForwardRunner::supports_chunked_prefill()`
   before deciding which loop to run: `run_raw_completion_chunked` at
   `foundation::DEFAULT_CHUNK_SIZE` when the open install's family can serve
   it (Gemma 4, or the dense half of `llama`), `run_raw_completion` otherwise.
   Unlike rate control, speculation and guardrails (Gotchas 10, 17, 18), this
   one genuinely needed no flag: those three are policy choices an operator
   might want to disable, where prefill shape is a pure throughput axis with
   the SAME losslessness guarantee `crates/cli`'s `--prefill-chunk` wiring
   relies on (`crates/runtime/CLAUDE.md` Gotcha 14's byte-identity contract),
   so there is nothing for a flag to trade off.

   Checked ORDER matters: speculation is resolved first (its own `match` arm
   above this one), so the two seams are not composable, exactly as the CLI's
   `stream_turn` documents for the same pair. Decode's per-token progress
   callback is unaffected either way, since only prefill routes differently
   and the two loops share it downstream.

   Verified against the real Gemma 4 install end to end (`tests/real_backend.rs`,
   `--ignored`, needs `TURBOSPARK_GEMMA4_INSTALL_DIR`): both the streaming and
   non-streaming cases pass with this dispatch live, which is the first time
   that test has exercised anything but `run_raw_completion` on this backend.
