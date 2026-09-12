# turbospark-server

HTTP server (`turbospark-server`) on Axum, speaking four wire formats against
one or more local backends: OpenAI `/v1/chat/completions`, legacy
`/v1/completions`, and `/v1/responses`; Anthropic `/v1/messages` and
`/v1/messages/count_tokens`; Ollama
`/api/{tags,version,show,chat,generate,embeddings,embed}`; `/v1/models` and
`/v1/models/:model`; `/v1/embeddings`; and a lock-free `GET /health`. Which backend serves a request
is `registry.rs`'s decision (Gotcha 28). Every generation endpoint
supports non-streaming and Server-Sent Events (SSE) streaming output;
`count_tokens` never generates at all. `--api-key` adds opt-in
`x-api-key`/`Authorization: Bearer` auth on every route except `/health`
(Gotcha 26); with no flag the server has no auth at all, same as before it
existed.

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
|   +-- bind.rs                 # Host and bind address resolution (--bind loopback|tailnet)
|   +-- main_tests.rs           # Unit tests for CLI args and host binding
|   +-- lib.rs                  # Library root: router, re-exported wire types
|   +-- cancel.rs               # Cancel/CancelOnDrop/CancelGuard: client-disconnect cancellation (Gotcha 25)
|   +-- queue.rs                # GenerationQueue/run_gated: FIFO admission for the one-runner gate (Gotcha 1)
|   +-- auth.rs                 # x-api-key/Bearer middleware for --api-key (Gotcha 26)
|   +-- handler/                # /v1/chat/completions + /v1/models + /health, and the shared generation core
|   |   +-- mod.rs              # The Axum handlers and the router wiring
|   |   +-- plan.rs             # `plan`: chat template, encode, shaping config
|   |   +-- exec.rs             # `run_full` and `stream_blocking`
|   |   \-- tests.rs            # Unit tests for the two above
|   +-- messages.rs             # Anthropic /v1/messages + /v1/messages/count_tokens: translate in, generate (or just plan), translate out
|   +-- completions.rs          # OpenAI legacy /v1/completions: raw prompt, no chat template
|   +-- responses/              # OpenAI /v1/responses: item-shaped request/response, typed SSE event sequence
|   |   +-- mod.rs              # The Axum handler and streaming coordinators
|   |   +-- map.rs              # Request folding, tool translation, and warning checks
|   |   +-- sse.rs              # Response building and typed SSE event sequence
|   |   \-- tests.rs            # Unit tests for mapping and serialization
|   +-- guardrails.rs           # Tool-call rescue, argument validation, the retry loop
|   |   +-- extra_formats.rs    # Rescue strategies for GLM/MiniMax/Kimi/Qwen-XML/Gemma tool-call markup
|   |   \-- tests.rs            # Unit tests for the verdict (pure, no model)
|   +-- registry.rs             # ModelRegistry: which backend serves a request (Gotcha 28)
|   +-- observe.rs              # ServerEvent/ServerObserver, and the run_completion decorator (Gotcha 29)
|   +-- ollama.rs               # Ollama-compatible routes; NDJSON rather than SSE (Gotcha 30)
|   +-- embeddings.rs           # /v1/embeddings + Ollama /api/{embeddings,embed}: encoding format, dimensions, size caps
|   +-- model.rs                # ChatModel trait and the ScriptedChatModel backend
|   +-- real_model.rs           # RealChatModel: RealForwardRunner backend (macOS only)
|   +-- encoder_model.rs        # RealEncoderModel: embedding-only backend over a distinct EncoderRunner, not RealForwardRunner (macOS only)
|   +-- response.rs             # Constructors for the OpenAI response & SSE chunk envelopes
|   +-- response_tests.rs       # Unit tests for response and chunk serialization
|   \-- vision.rs               # Image decoding and vision token injection mapping
\-- tests/
    +-- auth.rs                 # 401/200 both header spellings, /health exempt, build_router untouched
    +-- cancellation.rs         # Drop a streaming response mid-generation, assert it stopped short
    +-- chat_completions.rs     # Integration tests for the OpenAI endpoint
    +-- completions.rs          # Integration tests for /v1/completions, incl. the not-templated assertion
    +-- embeddings_api.rs       # Integration tests for /v1/embeddings and the Ollama embedding routes
    +-- guardrails.rs           # Rescue/validate/retry end to end, both endpoints, no model
    +-- harmony_channels.rs     # gpt-oss reasoning -> thinking/reasoning_content, and its tool calls
    +-- images.rs               # Integration tests for vision and image endpoints
    +-- messages.rs             # Integration tests for /v1/messages, count_tokens, /v1/models, wider OpenAI shapes
    +-- observe.rs              # the event sequence, incl. a request auth rejected
    +-- ollama.rs               # /api/* shapes and the NDJSON framing, asserted line by line
    +-- real_backend.rs         # Gated real-model end-to-end test (macOS, #[ignore]d)
    +-- reasoning_channels.rs   # Streaming & non-streaming reasoning channel translation tests
    +-- registry.rs             # routing by model id, and the single-model fallback
    +-- responses.rs            # Integration tests for /v1/responses, incl. the exact SSE event-order assertion
    +-- streaming_error_framing.rs  # A mid-stream failure is framed as an SSE/NDJSON error event, never a bare drop
    +-- system_prompt.rs        # The deployment-wide default system prompt, and what suppresses it per endpoint
    \-- fixtures/               # Test tokenizer fixtures for integration tests
```

## Key Modules

- `main.rs`: Server binary entry point and CLI option handling (`--model` real mode, legacy positional scripted mode, port, `--bind loopback|tailnet`).
- `handler/`: the `/v1/chat/completions`, `/v1/models`, and `/health` handlers, plus the generation core both generation endpoints share -- `plan.rs` (chat template, encode, shaping config, and `openai_request_warnings`/`merge_degradation` for the OpenAI-side half of `x-anyllm-degradation`, Gotcha 22) and `exec.rs`'s `run_full` and `stream_blocking` (which owns the `StructuredAssistantDecoder` when a request carries tools).
- `messages.rs`: the Anthropic `/v1/messages` handler, wrapping the same core in `translate_request` / `translate_response` / `new_stream_translator`, plus `count_tokens`, which runs `translate_request` + `plan` and stops there -- no generation, so no `ChatModel` call past that point.
- `completions.rs`: the legacy OpenAI `/v1/completions` handler -- a hand-rolled request type (`anyllm_translate` has none for this endpoint), `tokenizer.encode(_, add_bos: true)` with no chat template, and its own minimal `run_full`/`stream_response` with no decoder (Gotcha 23).
- `responses/`: the OpenAI `/v1/responses` handler (`mod.rs`, `map.rs`, `sse.rs`, `tests.rs`) -- `map::responses_to_chat_request` folds `anyllm_translate::openai::responses::ResponsesRequest` (which the vendored crate ships but never maps onto Chat Completions itself) down onto the same `ChatCompletionRequest` `handler::plan` renders, and `sse.rs` emits a hand-built typed `ResponsesStreamEvent` sequence (Gotcha 24).
- `guardrails.rs`: tool-call rescue parsing, argument validation against the request's own schema, and the one-retry loop, over `forge-guardrails` (see Gotcha 18). `inspect` is the pure verdict; `run_guarded` is the loop that acts on it.
- `registry.rs`: `ModelRegistry` and the resolution policy -- exact id, else the single attached model whatever the name, else a 404 naming what is there (Gotcha 28). `SingleModel` is what every pre-registry caller gets; `StaticRegistry` is a fixed set; the FFI implements its own over locked storage so a running server can gain and lose models.
- `observe.rs`: `ServerEvent`, `ServerObserver`, and `ReportingModel` -- a `ChatModel` decorator that reports each generation from `run_completion`, the one choke point every path already goes through (Gotcha 29).
- `ollama.rs`: the Ollama-compatible routes, hand-rolled shapes and NDJSON framing (Gotcha 30).
- `model.rs`: the `ChatModel` trait and `ScriptedChatModel`, bridging Axum handlers to `turbospark-runtime`. The trait owns WHICH decode loop runs (`run_completion`, Gotcha 17), not just which producer.
- `real_model.rs`: `RealChatModel`, a `RealForwardRunner` behind the same trait (macOS only).
- `embeddings.rs`: `/v1/embeddings` (OpenAI-shaped) and the Ollama `/api/embeddings`/`/api/embed` routes -- request size caps (`MAX_EMBEDDING_INPUTS`, `MAX_EMBEDDING_BYTES`) and the `dimensions` truncation check against the model's own width.
- `encoder_model.rs`: `RealEncoderModel`, a `ChatModel` impl over `runtime::encoder::EncoderRunner` -- a distinct encoder-only runner, not `RealForwardRunner` (macOS only).
- `cancel.rs`: `Cancel` (the `Arc<AtomicBool>` threaded through every async call chain), `CancelOnDrop` (wraps an SSE stream, sets it when axum drops the stream), and `CancelGuard` (sets it if a non-streaming handler's own future is dropped before `.defuse()`) -- the client-disconnect cancellation mechanism (Gotcha 25).
- `queue.rs`: `GenerationQueue` (one-permit FIFO admission over a `tokio::sync::Semaphore`) and `run_gated` (acquire-then-`spawn_blocking`, holding the permit for the body's whole duration), reached through `ChatModel::generation_queue` (Gotcha 1).
- `auth.rs`: `require_api_key`, an `axum::middleware::from_fn_with_state` layer applied to a sub-router in `lib.rs::build_router_with_options` -- everything except `GET /health`. Accepts `x-api-key` or `Authorization: Bearer`; compares with a local constant-time byte loop rather than the `subtle` crate (Gotcha 26).
- `response.rs`: constructors for `anyllm_translate::openai`'s response and SSE chunk envelopes, filling the many fields this server never populates in one place.

## Development & Test Commands

```sh
# Run tests for turbospark-server
cargo test -p turbospark-server

# The Ollama-compatible routes. NDJSON, so read them line by line rather
# than expecting SSE framing; `stream` defaults to TRUE here, unlike every
# OpenAI-shaped route above.
curl -s localhost:8080/api/tags
curl -s localhost:8080/api/version
curl -s localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","stream":false,"messages":[{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/api/chat -H 'content-type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}'

# Smoke the two generation endpoints against a running --model server.
curl -s localhost:8080/health
curl -s localhost:8080/v1/models
curl -s localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":120,"messages":[{"role":"user","content":"hi"}]}'
curl -sN localhost:8080/v1/messages -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","max_tokens":40,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8080/v1/messages/count_tokens -H 'content-type: application/json' \
  -d '{"model":"claude-sonnet-4-6","messages":[{"role":"user","content":"hi"}]}'
curl -s localhost:8080/v1/completions -H 'content-type: application/json' \
  -d '{"model":"m","prompt":"The capital of France is","max_tokens":10}'
curl -s localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"model":"m","input":"hi","max_output_tokens":40}'
curl -sN localhost:8080/v1/responses -H 'content-type: application/json' \
  -d '{"model":"m","input":"hi","max_output_tokens":40,"stream":true}'

# --api-key: run the server with `--api-key sk-test` (or export
# TURBOSPARK_API_KEY=sk-test first) to try these. Every route above 401s
# without one of the two headers below once that flag is set.
curl -s localhost:8080/v1/models -H 'x-api-key: sk-test'
curl -s localhost:8080/v1/models -H 'authorization: Bearer sk-test'
curl -s -o /dev/null -w '%{http_code}\n' localhost:8080/v1/models   # 401, no key
curl -s localhost:8080/health                                       # 200, /health is exempt

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

1. **Two backends, and the real one is serialized behind a FIFO gate.** `RealChatModel` (macOS only) owns ONE `RealForwardRunner` behind a `Mutex`: it costs a multi-gigabyte mapping plus a Metal pipeline compile to open and takes `&mut self`, so concurrent requests queue rather than run. Since ROADMAP P1 item 5 that queueing is EXPLICIT: `ChatModel::generation_queue` (default `None`) hands out a one-permit `GenerationQueue` (`queue.rs`), and every generation path acquires it in ARRIVAL order before touching the runner -- the runner mutex alone gives no fairness guarantee, and every waiter used to park a tokio BLOCKING thread spinning on it. The acquisition points are a CLOSED, NON-NESTED set (`run_guarded` -- around its whole retry loop, so a guardrail retry keeps its queue position; the five live-stream spawn sites via `run_gated`; ollama's `run`; `completions::run_full`; embeddings): acquiring twice on one request would deadlock on the single permit, so `handler::exec::run_full` itself never acquires. A request whose client disconnects while QUEUED releases without generating (the same silent-discard shape a mid-generation cancel takes); one that disconnects mid-generation still holds the permit until the generation actually stops. `tests/generation_queue.rs` pins both contracts with a hold-until-released model. `ScriptedChatModel` (`ScriptedLogitProducer`, canned completions) stays the portable backend the integration tests drive, keeps NO gate (the default `None`), and is the only one on non-macOS. That is why `ChatModel` exposes `with_producer` (lend a producer) rather than a `new_producer` factory. **`ScriptedChatModel` can force an EXACT token sequence**, which is how a dialect's markup is proven end to end with no model at all: one `one_hot(vocab, id)` per step, with the first `prompt_ids.len() - 1` steps consumed by prefill (`produce_prefill`), so the decode steps must start at exactly that offset or the generation begins mid-sequence. `tests/harmony_channels.rs` is the worked example, and it reconstructs the prompt length the same way `plan` does (the checkpoint's own template, `encode(&prompt, false)`).
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

12. **The decoder is built for THREE independent reasons, and keeping them independent is the point.** `stream_blocking` used to construct `StructuredAssistantDecoder` exactly when a request carried tools, which is the same condition `plan` uses to pick the tool template (Gotcha 7). The decision also fires for `ChatDialect::Harmony`, because `gpt-oss` writes its reasoning into an `analysis` channel BEFORE its answer and an undecoded stream sends the reasoning and the frame markup to the client as the reply. The THIRD condition is `reasoning_effort` on a ChatML or Gemma request: those dialects have a thought channel that is unreachable until a level is asked for, and reachable the moment one is. It is keyed on the REQUEST rather than the dialect, which is what keeps every existing call on the pass-through path it has always had. **The prompt path stays keyed on `tools` alone.** Coupling them again would render the tool template for every gpt-oss request, which changes the prompt rather than the presentation. The split reaches the wire as one field: `assistant_message` fills OpenAI `reasoning_content` and `reasoning_delta` fills the streaming equivalent, and `anyllm_translate`'s existing mappings turn both into an Anthropic `thinking` block. `thinking_blocks` stays `None` deliberately: that field carries Anthropic's SIGNED blocks, and a local model has nothing to sign with. One inherited ordering constraint, recorded in `reasoning_delta`: the streaming translator opens a thinking block on the first reasoning delta and closes it on the first content delta without reopening, so a backend that interleaved the two would need upstream work rather than a reordering here. Harmony emits analysis before final, so this server's sequence is well formed. **SINCE 2026-09-05 THE DECISION IS NOT THIS CRATE'S.** `needs_decoder` was one of three drifting copies (the CLI's `ChannelSplit` and the FFI's port of it were the others) and is now `runtime::turn_stream::TurnSplitter::new`, which `stream_blocking` wraps its loop with -- so the three reasons above are still the reasons, stated one file over. `docs/STREAMING.md` is the home. `tests/harmony_channels.rs` proves the whole chain with the scripted backend and no model; mutation-checked by reverting the splitter's condition to the tools-only one, which reddens all three cases.

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

20. **IMAGES REACH BOTH ENDPOINTS THROUGH ONE DECODER, AND THAT IS THE
   VENDORED CRATE'S DOING RATHER THAN THIS CRATE'S** (ROADMAP M-V8).
   `/v1/messages` translates into the OpenAI request the chat route already
   understands, and `anyllm_translate`'s `message_map/request.rs` maps an
   Anthropic `ContentBlock::Image` onto `ChatContentPart::ImageUrl`: a base64
   source becomes `data:<media_type>;base64,<data>`, a URL source passes
   through. So `src/vision.rs` handles one shape and both endpoints work.

   Gotcha 4's rule paid again, and it nearly did not: a first grep of
   `mapping/message_map.rs` (16 lines, a module stub) found no `Image` arm and
   suggested the Anthropic path lost images before this server saw them. The
   arm is in `mapping/message_map/request.rs`. Grep the DIRECTORY, not the
   file that shares its name.

   **A REMOTE URL IS REFUSED RATHER THAN FETCHED.** Fetching would make this
   server an HTTP client driven by request content -- an SSRF surface, a
   timeout budget and a redirect policy, none of which belongs in a local
   inference server.

   **THE URL SHAPE IS VALIDATED WHATEVER THE BACKEND CAN SERVE, and the
   payload is decoded only when it can.** A remote URL is a malformed request
   for this server however it is configured, so refusing it on a vision
   install and accepting it on a text-only one would leave a client unable to
   tell which problem it had. Splitting `split_data_url` from
   `decode_data_url` is what makes that free: a text-only backend pays nothing
   for a multi-megabyte data URL it will discard. The first version validated
   only when a tower was present and returned 200 on a remote URL, which the
   integration tests caught.

   **AN IMAGE THIS SERVER CANNOT SERVE IS REPORTED** on
   `x-anyllm-degradation`, now on BOTH routes -- OpenAI's spec has no such
   field, so that half is an extension rather than a translation. This
   module's own header used to record that `compute_request_warnings` knows
   nothing about images, which was true and was the gap. The turn still
   SUCCEEDS: the text half is answerable, and refusing would break every
   client that sends an incidental image.

21. **`run_completion` TAKES THE IMAGES BECAUSE THE ENCODE AND THE GENERATION
   MUST SHARE ONE LOCK.** A backend serializes on its one runner per call
   (Gotcha 1), so `set_prompt_vision` followed by `run_completion` would be
   two locks with a GAP: a second request landing in it overwrites the map,
   and the first generation then prefills the second request's picture. Both
   answer fluently and no test of either request alone can see it.
   `RealChatModel::run_with_images` takes the lock once and holds it across
   the encode, the injection and the decode, then CLEARS the map whether the
   generation succeeded or not.

   Speculation and chunked prefill are both skipped on that path, deliberately:
   the qwen family this tower belongs to serves neither, so composing them
   would be untested code on an unreachable path.

   **THE GATE ASSERTS ON OUTPUT, NOT ON SHAPE, and that is M-V7's lesson
   rather than a preference.** `tests/images.rs` covers every refusal with a
   scripted backend and no model; it structurally cannot tell whether a
   picture reaches the model. That is exactly the gap M-V5's injection bug
   lived in for two milestones -- lengths and counts all agreed while the
   model answered about a page it had never seen.
   `real_backend.rs`'s `real_backend_reads_an_image_sent_over_both_endpoints`
   asserts the transcription contains the page's own line numbers, and both
   endpoints return byte-identical text on the real install.

22. **`/health` READS ONLY A FIELD, NOT THE RUNNER, ON PURPOSE.**
    `handler::health` calls `ChatModel::model_id()` alone, which every
    backend answers from a plain `&str` field (`RealChatModel::model_id`
    never touches its `Mutex<RealForwardRunner>`). A liveness probe that
    queued behind the one generation this process can run at a time would
    answer the wrong question -- "is the process alive" needs to stay true
    while a request is mid-decode, not just between requests.

    **THE OpenAI HALF OF `x-anyllm-degradation` HAS NO VENDORED
    COUNTERPART, BECAUSE `anyllm_translate::compute_request_warnings` ONLY
    EVER SEES THE ANTHROPIC-SHAPED REQUEST `/v1/messages` TRANSLATES
    FROM.** A field like `response_format: {"type": "json_object"}` or
    `presence_penalty` arrives on `/v1/chat/completions` directly, with no
    Anthropic translation step to warn about it. `handler::plan::
    openai_request_warnings` is that gap's fix, built on the same
    `anyllm_translate::TranslationWarnings` type (comma-joined items) rather
    than a hand-rolled string, so the two halves of the header read alike.
    `merge_degradation` (`"; "`-joined) replaces the ad hoc match both
    `chat_completions` and `messages::messages` used to write inline for
    combining it with a dropped-image note -- one function, both call sites.

    Every warning here is conditioned on the DEFAULT the server already
    produces matching the field's own OpenAI default: `n` warns past 1, not
    at 1 (this server already returns one choice), `response_format` warns
    past `"text"`, not at it. `presence_penalty` and `frequency_penalty`
    warn UNCONDITIONALLY whenever present, because neither reaches
    `selection::shaping` yet -- delete those two arms the day they do.

    **SSE keep-alive (`Sse::keep_alive`, 15s interval) is on every SSE
    construction in the crate** (`handler::mod`'s two, `messages`'s two,
    `completions`'s one -- five as of Gotcha 23) for the same reason a long
    prefill needs it most: the gap between a request landing and its first
    token can run several seconds at this engine's decode rates, long
    enough for a loopback proxy or an idle-conservative client to give up on
    a connection that has sent nothing yet.

23. **`/v1/completions` HAS NO CHAT TEMPLATE, WHICH IS WHY IT NEEDS ITS OWN
    `run_full`/`stream_response` RATHER THAN `handler::exec`'S.** Every other
    generation path in this crate goes through `handler::plan::plan`, which
    renders the checkpoint's Jinja template and therefore always has a
    prompt shaped like a chat turn. This endpoint's whole reason to exist is
    a prompt that is NOT one -- `tokenizer.encode(prompt, add_bos: true)`,
    the same convention `turbospark-check --prompt` uses (`crates/cli`) and
    the ONE call site in this crate that passes `true` rather than `false`
    (every chat path sets `add_bos: false` because the template emits its
    own `<bos>`; nothing emits one here, so the tokenizer has to). Sharing
    `handler::exec::run_full` would have meant threading a "skip the
    template" flag through `plan` for a caller that needs none of what
    `plan` does past encoding -- no images, no tools, no reasoning, no
    `StructuredAssistantDecoder`. A second, smaller `run_full` costs less
    than that flag would.

    **THE ONE THING WORTH SHARING IS SHAPING, AND IT MOVED OUT OF `plan.rs`
    RATHER THAN BEING COPIED.** `handler::plan::build_shaping` (temperature,
    top_p, top_k, repetition_penalty, seed) takes primitives rather than a
    `ChatCompletionRequest`, because `/v1/completions`'s hand-rolled request
    type has no such struct to hand it -- both callers pass their own
    `extra` flatten map. `stop_strings` moved `pub(crate)` for the same
    reason: one OpenAI-shaped `stop` field (bare string or array), one
    parser, two callers.

    **THIS ENDPOINT IS WHERE OpenAI's `n`, `suffix`, `logprobs`, `best_of`,
    AND `echo` ACTUALLY LIVE ON THE WIRE**, unlike the chat endpoint where
    most of that set is `/v1/chat/completions`-flavoured. `n > 1` and
    `suffix` (fill-in-the-middle, which no kernel here implements) are
    refused with a 400 rather than silently answering the wrong shape;
    `logprobs`, `best_of`, and `echo` are accepted and reported on
    `x-anyllm-degradation` via a second, endpoint-local
    `completion_warnings` -- NOT `handler::plan::openai_request_warnings`,
    because the two endpoints don't share a request type to read the fields
    off of, only the pattern (`anyllm_translate::TranslationWarnings`).

    **PROVING "NO TEMPLATE" NEEDED AN INDIRECT TEST, because the scripted
    backend cannot report what it received.** `tests/completions.rs`'s
    `the_prompt_is_not_chat_templated` sends the same text to both endpoints
    and asserts `/v1/completions`'s `usage.prompt_tokens` comes out LOWER --
    a ChatML wrapper adds `<|im_start|>user\n...<|im_end|>\n<|im_start|>
    assistant\n` on top of the content, so a leaked template would inflate
    the raw endpoint's count rather than leave it alone. A shape-only
    assertion (200, right `object` field) cannot see a template leak at all.

    **`count_tokens` (`messages.rs`) IS THE MIRROR CASE: A CALLER THAT NEEDS
    `plan` TO RUN AND STOP THERE.** `MessageCreateRequest::max_tokens` is a
    required `u32` on the wire type -- correct for `/v1/messages`, where a
    real generation needs a budget, and wrong for `count_tokens`, whose own
    Anthropic spec accepts a request with none. The handler takes
    `Json<serde_json::Value>` rather than `Json<MessageCreateRequest>`
    directly, injects `"max_tokens": 1` ONLY when the key is absent, then
    deserializes and runs `translate_request` + `plan` exactly as
    `/v1/messages` would -- so the count is definitionally what a real call
    on this request would prefill, not a separately-maintained estimate that
    can drift from it.

24. **`anyllm_translate` SHIPS THE RESPONSES WIRE TYPES AND NEITHER HALF OF
    THE MAPPING THIS SERVER NEEDS.** `openai::responses::{ResponsesRequest,
    ResponsesResponse, ResponsesUsage}` and
    `mapping::responses_streaming_map::ResponsesStreamEvent` exist because
    the crate translates Anthropic<->Responses
    (`responses_message_map::{anthropic_to_responses_request,
    responses_to_anthropic_response}`, `translate_request_responses` /
    `translate_response_responses` in `translate.rs`) -- there is no
    Responses<->Chat-Completions mapping anywhere in it, because nothing
    upstream needed one. This server generates through Chat Completions
    shape only (`handler::plan` renders and shapes THAT type, for all three
    endpoints), so `responses/` hand-writes both directions itself, reusing
    only the wire TYPES.

    **THE ANTHROPIC-FACING MAPPING WAS STILL WORTH READING FIRST** (crate
    Gotcha 4's rule), because Responses' `input`/`output` shapes are the same
    whichever OTHER API is on the far end: a tool call and its result are
    ROOT-LEVEL items (`function_call`, `function_call_output`), not content
    blocks nested on a message, which `responses_message_map::
    convert_blocks_to_items` and `extract_output_item` show without having
    to derive it from OpenAI's docs. `item_to_message` and `output_items`-
    equivalent construction here read the same three item types that pair
    exercises, in the same flattened shape.

    **A Responses TOOL IS FLAT WHERE A CHAT COMPLETIONS ONE IS NESTED.**
    `{"type":"function","name":...,"parameters":...}` against
    `{"type":"function","function":{"name":...}}` -- one extra layer,
    `flat_tool_to_chat_tool`'s whole job.

    **`top_p`, `tool_choice`, AND `stop` HAVE NO FIELD ON `ResponsesRequest`
    AT ALL**, unlike Chat Completions' explicit ones -- they arrive only in
    Responses' own `extra` flatten map. `responses_to_chat_request` pulls
    them OUT of a clone of that map into the `ChatCompletionRequest` fields
    `handler::plan` and `build_config` actually read, and removes them from
    what is forwarded so nothing downstream reads a Responses-shaped value
    under the same key twice. Everything else in `extra` (`top_k`,
    `repetition_penalty`, `seed`, `reasoning_effort`, `n`, ...) passes
    through unchanged, because it is already the map shape those readers
    expect.

    **`previous_response_id` IS REFUSED RATHER THAN IGNORED.** This server
    keeps no prior turn on disk to continue; silently starting a fresh
    conversation under a client-supplied continuation id would answer a
    DIFFERENT question than the one the client thinks it asked, and do so
    with no error anywhere. A stateless server that ignored the field would
    be indistinguishable from a stateful one that lost the turn.

    **REASONING IS AN OUTPUT ITEM ON THE NON-STREAMING PATH AND DROPPED ON
    THE STREAMING ONE**, and that asymmetry is not an oversight: OpenAI
    defines no delta EVENT for a Responses reasoning summary (there is a
    `reasoning` item type but no `response.reasoning_summary.delta` in the
    documented event set this crate's reference,
    `mapping::responses_streaming_map.rs`, consumes), so streaming one would
    be an invented shape on both the producer and any real client's parser.
    `may_produce_reasoning` -- NOT `runtime::TurnSplitter`'s own build-a-decoder
    condition, whose first term (`!tools.is_empty()`) fires for call-parsing
    reasons that have nothing to do with whether reasoning is actually
    produced -- decides
    ahead of the stream whether THIS request's dialect would have separated
    one out, and reports it on `x-anyllm-degradation` before the body starts,
    which is the only point in a streaming response headers can still be set.

    **THE TYPED EVENT SEQUENCE IS THE ACTUAL CONTRACT, so the test that
    matters asserts the exact ORDER, not just that the right names
    appeared.** `tests/responses.rs`'s
    `streaming_response_emits_events_in_the_documented_order` compares the
    full `Vec<&str>` of `event:` lines against a literal sequence; mutating
    `close_message_events` to emit `output_item.done` before
    `content_part.done` passed every other test in the file and reddened
    only that one. A client's own state machine (this crate's Anthropic-
    Anthropic case, `messages::StreamingTranslator`, is the local example of
    exactly that kind of consumer) would misparse the swapped order the same
    way.

    **THE LIVE AND BUFFERED STREAMING PATHS ARE SEPARATE FUNCTIONS ON THE
    SAME REASON `handler::mod` and `messages.rs` HAVE PAIRS OF THEIR OWN**
    (Gotcha 12's third condition, Gotcha 18): a tool-carrying request under
    active guardrails buffers the whole generation before framing it as
    events, so a rescued or re-asked call never streams the syntax error it
    was rescued from. `buffered_stream_response` re-frames a `Generated` the
    same event shapes the live path builds incrementally, just one delta per
    item instead of one per token.

25. **A CLIENT DISCONNECT NOW SHORTENS THE GENERATION IT WAS WAITING ON, AND
    THE MECHANISM IS SPLIT ACROSS AN OWNED FLAG AND A BORROWED PREDICATE
    BECAUSE THE TWO CANNOT CROSS THE SAME BOUNDARY.**
    `runtime::CancelFlag<'a>` (`&'a dyn Fn() -> bool`, polled once per
    prefill and decoded token inside `run_raw_completion_cancellable` and
    its chunked/speculative siblings) is not `Send`, so it cannot be
    captured by a `spawn_blocking` `move` closure -- only the
    `Arc<AtomicBool>` it reads from can. `crates/server/src/cancel.rs`'s
    `Cancel` type alias is that `Arc`, threaded through every async-visible
    signature in the crate (`guardrails::run_guarded`, `handler::exec::
    run_full`, every handler and `stream_response`/`buffered_stream_response`
    pair); `as_cancel_flag` builds the actual `CancelFlag` closure exactly
    ONCE, on the blocking thread the generation runs on, and `&flag` is what
    gets passed down through `ChatModel::run_completion` into
    `RealChatModel`'s four internal dispatch sites (images, chunked,
    sequential, speculative -- all four switched to their `_cancellable`
    runtime entry point in the same commit) and `model.rs`'s default
    sequential-loop impl.

    **TWO INDEPENDENT DETECTORS SET THE FLAG, AND NEITHER SUBSUMES THE
    OTHER'S REASON FOR EXISTING EVEN THOUGH EITHER ALONE CAUGHT THIS CRATE'S
    OWN TEST.** `CancelOnDrop` (wrapping the SSE stream `Sse::new` returns,
    on every streaming construction in the crate) fires when axum DROPS that
    stream, which is the disconnect signal itself and fires independent of
    whether any chunk was ever in flight -- the only signal a BUFFERED path
    has at all, since nothing is sent until the whole generation is done.
    The `tx.send(...).is_err()` check inside each live path's `send` closure
    is the faster, redundant detector for the per-token path: measured with
    `tests/cancellation.rs`'s `CountingModel` (a `LogitProducer` that counts
    every `produce` call rather than replaying a fixed script, so it can run
    past any scripted step budget), disabling EITHER detector alone left the
    test passing -- dropping the plain (unwrapped) receiver stream still
    drops the inner `mpsc::Receiver`, so the very next sequential path's
    send fails anyway -- and only disabling BOTH reddened it, turning a
    0.3s test into one that still had not finished after three minutes.
    That timing gap, not a numeric assertion, is the proof: an abandoned
    `max_tokens: 1_000_000` request left running for real minutes is exactly
    the cost this feature removes, and `CancelGuard` (a local guard that
    sets the same flag if dropped before `.defuse()`, covering the
    non-streaming and buffered paths where it is the HANDLER's own future
    that gets dropped, not the SSE stream) closes the same gap for the two
    request shapes that never open one.

    **CANCELLING IS NOT AN ERROR AND NOTHING HERE REPORTS ONE.** The runtime
    returns a normal `RawDecodeResult` with `StopReason::Cancelled` and
    whatever it had generated; every caller checks that reason and discards
    the result silently rather than sending a finish chunk, an error event,
    or (in `guardrails::run_guarded`) treating a truncated turn as one worth
    a guardrail retry -- the client that would read any of those is the
    same one already gone. The lock `RealChatModel`'s one runner serializes
    on is still held until the cancelled generation actually stops, so a
    disconnect shortens a queued waiter's wait rather than skipping it
    outright.

26. **`--api-key` IS A SUB-ROUTER AND A LAYER, NOT A PER-HANDLER CHECK, AND
    THE ORDER `lib.rs::build_router_with_options` DOES THINGS IN IS THE
    WHOLE REASON `/health` STAYS EXEMPT.** `protected` (every route except
    `/health`) is built, `.layer(axum::middleware::from_fn_with_state(...))`
    is applied to IT ALONE when `options.api_key` is `Some`, and only THEN
    does `Router::new().route("/health", ...).merge(protected)` combine the
    two -- `/health` was never a member of the router the layer wrapped, so
    no request to it can reach `auth::require_api_key` however the merge is
    ordered. Registering `/health` inside `protected` (even innocuously,
    alongside the other routes, before the `.layer()` call) puts it back
    under the layer and reddens `tests/auth.rs`'s
    `health_is_exempt_from_the_api_key_requirement` with a 401 -- the
    mutation this crate's tests were checked against. Registering it a
    SECOND time on the merged router is worse: axum panics at router-build
    time (`Overlapping method route`) rather than silently preferring
    either one, which is a good failure mode to know about rather than
    stumble into while refactoring this function.

    **`build_router` (no options) IS THE DEFAULT, AND EVERY EXISTING CALLER
    KEEPS IT.** It is `build_router_with_options(state, RouterOptions::
    default())` -- `RouterOptions::api_key` defaults to `None`, `protected`
    skips the `.layer()` call entirely, and the router this crate served
    before `--api-key` existed is exactly what every test file except
    `tests/auth.rs` still gets, unauthenticated. This matters beyond the
    test suite: `RouterOptions` is the shape the FFI's future in-process
    server (the omlx-parity plan's F1) is meant to pass its own
    `apiKey: Option<String>` through as, so `build_router`'s zero-option
    case staying inert is what keeps that integration from having to
    special-case "no key" on its own side.

    **`x-api-key` IS CHECKED BEFORE `Authorization: Bearer`, AND THAT ORDER
    IS DELIBERATE RATHER THAN ARBITRARY.** Anthropic-native clients --
    Claude Code among them, via `ANTHROPIC_API_KEY` /
    `ANTHROPIC_BASE_URL` -- send `x-api-key`, which is the walkthrough this
    crate's own docs point at (`docs/CLI.md`'s "The point of `/v1/messages`"
    example). `Authorization: Bearer` is accepted as the generic/OpenAI
    fallback for clients that only know that spelling, checked second so
    neither header shadows the other when a caller sends both (a stray
    default `Authorization` header from an HTTP client library, say) --
    `presented_key` reads `x-api-key` first and never looks at
    `Authorization` once it has found one.

    **THE COMPARISON IS A LOCAL EIGHT-LINE FUNCTION RATHER THAN THE
    `subtle` CRATE, AND THE REASON IS SCOPE RATHER THAN NOT-INVENTED-HERE.**
    `subtle`'s `ConstantTimeEq` is built for comparing against a TABLE of
    secrets (its `Choice` type composes across multiple comparisons without
    ever branching on any one of them); this process holds exactly ONE key
    for its whole lifetime, so there is nothing to compose. `constant_time_eq`
    still refuses to short-circuit on the first differing byte -- `diff |= x
    ^ y` folds the whole comparison into one accumulator read only once, at
    the end -- which is the actual property worth keeping: a naive `==`
    would let a timing side channel narrow the key character by character.
    **THE LENGTH CHECK IS A SEPARATE FAST PATH AND A TEST HAS TO ACCOUNT FOR
    IT SEPARATELY**, or a wrong-key test proves nothing about the loop
    beneath it: `constant_time_eq` returns `false` on `a.len() != b.len()`
    before the loop ever runs, so a wrong key of a DIFFERENT length from the
    real one is refused by that check alone and would still be refused if
    the loop's body were deleted outright. `tests/auth.rs`'s
    `a_wrong_key_of_the_same_length_is_refused` sends a key matching
    `"sk-correct"`'s own length for exactly this reason -- mutating the loop
    to `true` reddened that test and left every different-length case
    passing, which is what proved the length check and the loop are two
    independent things a test has to cover separately, not one.

27. **`presence_penalty`/`frequency_penalty`/`min_p` REACH `ShapingConfig`
    THROUGH ONE SHARED FUNCTION, AND `/v1/completions` DELIBERATELY CALLS
    IT WITH TWO OF THE THREE HELD BACK.** Added 2026-08-30.
    `handler::plan::build_shaping` now takes explicit
    `presence_penalty: Option<f32>, frequency_penalty: Option<f32>`
    parameters and reads `min_p` out of the request's `extra` flatten map
    itself (the same map `top_k` and `repetition_penalty` already use --
    see the sampling-knob-surface entry in `DEVIATIONS.md`), then applies
    all three through `ShapingConfig::with_presence_penalty`/
    `with_frequency_penalty`/`with_min_p`. `messages.rs::build_config`
    passes `request.presence_penalty` and `request.frequency_penalty`
    straight through (both are explicit fields on `anyllm_translate`'s
    OpenAI request type). `completions.rs::build_config` passes `None,
    None` for those two params WITH A COMMENT, because the legacy
    `/v1/completions` wire shape this port hand-rolled
    (`completions::CompletionRequest`) never declared either field to
    begin with -- there is nothing on the request to read, so passing
    `None` is not a scope cut, it is the honest value. `min_p` still
    reaches `/v1/completions` through the same `extra` path `build_shaping`
    already reads for every caller.
    **`openai_request_warnings` NO LONGER REPORTS EITHER PENALTY ON
    `x-anyllm-degradation`.** Before this change both were accepted on the
    wire and silently ignored, which is what the warning existed to
    surface; now both are honored, so keeping the warning would have meant
    telling a caller their request was degraded when it was not. The two
    `if request.presence_penalty...` / `if request.frequency_penalty...`
    arms were deleted from `plan.rs::openai_request_warnings` rather than
    left dead, and `tests/chat_completions.rs`'s
    `unsupported_openai_fields_are_reported_on_the_degradation_header`
    was repointed at `logprobs` (still genuinely unsupported) so the test
    keeps discriminating instead of asserting a warning that no longer
    fires. See `crates/selection/CLAUDE.md`'s matching Gotcha for the
    sampler-side semantics (generated-suffix-only penalties, min-p's
    composition order, the `[-2, 2]`/`[0, 1)` bounds) -- this crate's half
    of the change is wiring, not policy.

28. **A REQUEST IS ROUTED TO A MODEL NOW, AND THE FALLBACK THAT KEEPS EVERY
    PRE-REGISTRY CLIENT WORKING IS THE LOAD-BEARING PART.** Added 2026-08-30
    (`src/registry.rs`). `ChatModel::model_id`'s doc used to say a request's
    `model` field was echoed back rather than routed on; that is still true
    of a server with ONE model and no longer true in general.

    The rule, in order: an exact id match wins; failing that, **if exactly
    one model is attached it serves the request whatever name was asked
    for**; only with two or more attached and no match is it a 404 naming
    what IS available. The middle clause is not a courtesy. `docs/CLI.md`'s
    own Anthropic walkthrough sends `"model":"claude-sonnet-4-6"` at an
    install named nothing of the sort, Claude Code's gateway discovery does
    the same, and every OpenAI SDK sends its own default -- all of which
    worked precisely because the name was ignored. Routing strictly would
    have 404'd the documented client on the first day multi-model shipped.
    `tests/registry.rs`'s `one_model_serves_a_name_it_does_not_know` is that
    invocation turned into an assertion, and deleting `resolve_among`'s
    `[only]` arm reddens it and nothing else.

    **`handler::AppState` KEPT ITS NAME AND ITS MEANING, WHICH IS WHY THIS
    WAS EIGHT FUNCTION HEADS RATHER THAN A SWEEP.** It is still
    `Arc<dyn ChatModel>` -- ONE resolved backend -- and `plan`, `run_full`,
    `stream_blocking`, `may_produce_reasoning` and `run_guarded` are untouched
    (`needs_decoder` was in this list until the `TurnSplitter` refactor moved
    that decision to `crates/runtime`; see Gotcha 12). What changed is that axum's `State` now
    carries a `ServerState` (registry plus observer), and each of the eight
    entry points calls `handler::resolve_backend` as its first act. Past
    that line the crate is single-model again.
    `build_router(state: impl Into<ServerState>)` with a `From<Arc<dyn
    ChatModel>>` impl is what left every existing call site -- this crate's
    whole integration suite -- compiling verbatim.

    **`/v1/models/:model` DELIBERATELY DOES NOT TAKE THE FALLBACK.** It
    matches against `registry.rows()` rather than through `resolve`, because
    a lookup asks whether this exact id exists where the fallback answers a
    different question. Routing it through `resolve` reddens
    `a_model_lookup_does_not_take_the_single_model_fallback` alone.

    **AN EMPTY REGISTRY IS 503, NOT 404.** A server with nothing attached is
    a real state a host starts one in (`crates/ffi`'s `ts_server_start`
    takes a null session), and "the request is fine, the server is not
    ready" is a different thing for a client's retry logic than "that model
    does not exist".

29. **THIS CRATE REPORTS WHAT IT DOES NOW, AND THE FACTS COME FROM TWO
    PLACES BECAUSE NEITHER EMITTER CAN SEE THE OTHER'S.** Added 2026-08-30
    (`src/observe.rs`). `RouterOptions.observer` is `None` by default and
    then nothing is recorded and no event is even BUILT (`observe::record`
    takes a closure), so every pre-observer caller keeps the exact path it
    had -- the standalone binary passes `None` deliberately: its equivalent
    is stdout, and a second structured channel nothing reads is overhead per
    request for no reader.

    An axum middleware layer knows the method, path, status and wall
    duration for EVERY request including the ones no handler ran. The
    generation paths know the token counts and phase timings. So a request
    produces `RequestStarted` and `RequestFinished` from the layer,
    `RequestRouted` from `resolve_backend`, and `Generated` from the
    generation, tied together by an id the layer mints into the request
    extensions.

    **THE LAYER'S PLACEMENT AND ITS ORDER ARE TWO SEPARATE PROPERTIES WITH
    TWO SEPARATE TESTS, and one mutation does not catch both.** WHICH
    ROUTER it is applied to decides `/health` coverage -- moving it onto
    `protected` (which reads as tidier, since the auth layer goes there)
    leaves `/health` unobserved, and reddens
    `health_is_observed_even_though_it_is_exempt_from_auth` and NOTHING
    else, the 401 case included, because a later `.layer()` still wraps an
    earlier one. WHEN it is applied relative to `auth` decides the 401
    coverage: applied BEFORE the auth layer it sits inside, the rejection
    returns from the outer layer, and this code never runs. Measured both
    ways; both mutations are recorded in `lib.rs`'s own comment.

    **`Generated` IS EMITTED BY A DECORATOR ON `ChatModel::run_completion`
    RATHER THAN BY A REPORTER THREADED THROUGH THE CORE.** That method IS
    the choke point already (Gotcha 17: it exists because the speculative
    loop is not object-safe, so the trait method owns which decode loop
    runs), and every generation in this crate goes through exactly one call
    to it. `observe::ReportingModel` wraps the resolved backend and
    **delegates rather than reimplementing** -- calling `with_producer` in
    that override instead would silently downgrade every speculative or
    chunked request to the sequential loop the moment an observer was
    configured, with nothing failing.

    **THERE IS NO TIME-TO-FIRST-TOKEN FIELD, and its absence is the honest
    answer rather than a gap.** A caller means "request in, first token
    out", which includes the wait behind the one-runner-per-model lock
    (Gotcha 1), and the decorator only starts counting once it already holds
    the runner. What is reported is `prefillSeconds` and `decodeSeconds`
    off `RawDecodeResult` plus the middleware's `durationMs`, which DOES
    include the queue; subtracting gives the wait. A field named `ttftMs`
    filled from `prefillSeconds` would read as the first number and be the
    second. Token counts come off `RawDecodeResult` and never off a count of
    deltas -- `swift/CLAUDE.md` Gotcha 7 is the worked example of how far
    apart those two are.

    A request the guardrails re-asked emits TWO `Generated` events. That is
    the useful reading rather than a duplicate: the retry is real work the
    machine did, and a consumer that replaced rather than accumulated would
    under-report exactly the requests that cost the most.

30. **THE OLLAMA ROUTES ARE THE ONE NON-SSE STREAM IN THIS CRATE.** Added
    2026-08-30 (`src/ollama.rs`): `/api/tags`, `/api/version`, `/api/show`,
    `/api/chat`, `/api/generate`. `anyllm_translate` ships no Ollama types
    (it does ship Gemini, deliberately not taken), so the shapes are
    hand-rolled the way `completions.rs`'s legacy ones are.

    The reason to carry a fourth wire format at all: the other three are
    what a developer picks deliberately, and Ollama's is what a lot of
    tooling speaks by DEFAULT with no way to change it, so for those clients
    it is this shim or nothing.

    **NDJSON, NOT SSE.** One bare JSON object per line over a plain body: no
    `event:` lines, no `data:` prefix, no `[DONE]` sentinel, and the LAST
    object carries `"done": true` plus the counters. So this module builds
    its own response rather than taking a parameter on the shared SSE one,
    and Gotcha 22's note that keep-alive is on "every SSE construction in
    the crate" stays true by not applying here.
    `tests/ollama.rs`'s `a_streaming_chat_is_ndjson_ending_in_a_done_object`
    parses every line as a complete object, which an SSE body cannot
    satisfy; returning the stream through `Sse::new` reddens it alone.

    **`stream` DEFAULTS TO TRUE**, unlike every OpenAI-shaped endpoint here
    where an absent `stream` means false. That is Ollama's own default, and
    a client that omits the field is expecting a stream.

    Three honest refusals worth not re-deriving. `num_ctx` in the options
    bag is DROPPED rather than honoured: the context window is fixed when
    the model is opened and the KV cache is already allocated at it, so a
    request cannot change it. `size` in `/api/tags` reports 0 rather than an
    estimate, because this server downloaded nothing and an install's
    on-disk bytes are not what it committed (AGENTS.md Gotcha 58).
    `/api/version` reports THIS crate's version rather than borrowing an
    Ollama one, because clients gate features on that range and a borrowed
    number would promise endpoints that do not exist here. And reasoning and
    tool pieces are dropped on the Ollama path -- its wire shape has nowhere
    for either, and emitting a scratchpad as the answer is AGENTS.md Gotcha
    56's failure.

31. **`--prefix-reuse` IS PROCESS-LEVEL LIKE THE RATE CAP, SPECULATION AND
    GUARDRAILS, AND IT IS THE ONE FLAG HERE THAT DEFAULTS TO ON RATHER THAN
    TO A SAFE NO-OP.** Added 2026-09-01 (ROADMAP.md section 4). `RealChatModel::open`
    calls `RealForwardRunner::set_prefix_reuse` once, right after the runner
    opens -- the mechanism itself (`crates/runtime/CLAUDE.md` Gotcha 30) is
    unmodified and was already wired into `run_raw_completion_chunked`, which
    is the loop this server actually takes for any chunked-prefill-capable
    family (Gotcha 19).

    **THE DEFAULT IS SAFE TO FLIP BECAUSE THE MECHANISM IS PROVABLY LOSSLESS,
    NOT BECAUSE THE TRADE-OFF DISAPPEARS.** A prompt that does not extend the
    previous request's KV always falls back to a full reset before prefill
    (`raw_completion.rs`'s `if reused == 0 { producer.reset(); }`, mirrored in
    the chunked driver) -- there is no path where a mismatched request reads
    stale KV from an unrelated conversation. What is real is a traffic-mix
    cost: **THIS SERVER HAS EXACTLY ONE RUNNER SERVING EVERY CLIENT'S EVERY
    CONVERSATION** (Gotcha 1), with no per-conversation identity anywhere in
    the request path, so two interleaved unrelated conversations each discard
    the other's reusable prefix and the optimization simply does not fire --
    at no correctness cost, but also at no benefit, while still paying the one
    real price: `KvCacheManager::reset`'s `advise_dontneed` calls are skipped
    between turns when reuse is on, so pages that would normally be released
    stay resident, raising the IDLE FLOOR (never the peak) for the life of the
    process.

    **THERE IS NO SERVER-SIDE STDERR LINE THE WAY THE CLI'S `--chat` HAS ONE,
    SO `ServerEvent::Generated` CARRIES `reusedPrefixTokens` INSTEAD.** The
    CLI's `[prefix-reuse] N/M` line exists because this class of feature has
    already shipped silently inert once (`crates/runtime/CLAUDE.md` Gotcha
    30: "measured in the real chat REPL at 0/33 ... through two rounds of
    apparently-working implementation"). A server has no equivalent terminal
    an operator is watching, so the observable has to travel with whatever
    DOES watch traffic -- an attached `ServerObserver`. `ReportingModel`
    reads `RawDecodeResult::reused_prefix_tokens` off the same value it
    already reads `prompt_tokens`/`new_tokens` from, so this cost nothing
    beyond one field; a request the guardrails re-asked still produces TWO
    `Generated` events for one HTTP call (Gotcha 29), and each carries its
    OWN count. `tests/real_backend.rs`'s
    `real_backend_reuses_kv_across_two_chat_turns` is the guard: it asserts
    turn 2's event reads a NONZERO count, not merely that both requests
    returned 200, which is the same class of trivially-passing test Gotcha
    30 in the runtime crate warns against.

    **`tests/real_backend.rs`'s two PRE-EXISTING open calls pin this `false`**,
    for AGENTS.md Gotcha 35's reason: a gate that let a process-level policy
    sense its own default would assert against a different configuration than
    the one it was written for. The vision test's `false` is additionally
    inert rather than merely conservative -- `set_prompt_vision` taints
    `kv_prefix` on every call (`crates/runtime/CLAUDE.md` Gotcha 30's own
    taint list), so that path could not reuse a prefix whatever the flag said.

32. **`--session-slots` (ROADMAP section 4's Option 3) IS THE FLAG THAT
    ACTUALLY FIXES GOTCHA 31's STATED LIMITATION, AND IT NEEDED A SECOND
    CORRECTNESS FIX FOUND ONLY BY THE REAL-INSTALL GATE.** Gotcha 31 shipped
    `--prefix-reuse on|off` and documented its own gap plainly: "this
    server has exactly one runner serving every client's every
    conversation, with no per-conversation identity anywhere in the
    request path, so two interleaved unrelated conversations each discard
    the other's reusable prefix." `--session-slots N` (default 1, no pool)
    is the fix: `RealChatModel::open` threads it into
    `RealForwardRunner::open_with_slot_policy_speculation_steering_and_sessions`,
    a NEW sibling of `open_with_slot_policy_speculation_and_steering` rather
    than a widened form of it -- that six-argument function has roughly a
    dozen existing callers across `crates/runtime`'s own tests, `crates/cli`
    and `crates/ffi`, none of which has any reason to acquire a session
    pool, and `session_slots <= 1` costs nothing extra
    (`model_io::session_pool_bytes`), so widening it would be a no-op diff
    at every one of those call sites for no benefit. `turbospark-server` is
    the only front end whose one runner serves more than one conversation
    at a time, which is why it is the only caller of the new form. The
    session-pool MECHANISM itself (`SessionSlot`/`SessionPool`, the swap in
    `RealForwardRunner::select_session`, the park-and-promote in `reset()`)
    lives entirely in `crates/runtime`; see `crates/runtime/CLAUDE.md`
    Gotcha 32 for its design and for the two real bugs a real-install test
    found in it (a destructive shallow-match rewind, then an
    overly-strict discriminator that undid the pool's own correct swap
    decision) -- neither bug was visible to the synthetic
    `crates/runtime/tests/session_pool.rs` file written alongside the
    feature, which is why `crates/server/tests/real_backend.rs`'s
    `real_backend_reuses_kv_across_two_interleaved_conversations` exists as
    its own real-install gate rather than being considered redundant with
    it.

    **Real committed memory, not a floor**, which is why this is `--expert-
    cache-slots`'s explicit-opt-in shape rather than `--prefix-reuse`'s
    safe-default-on one: a parked slot is allocated at `open()`, before any
    request ever needs it, unlike prefix reuse's only cost (a raised idle-
    memory floor between requests). `RealChatModel::open` refuses an
    over-committed value BEFORE `open_inner` allocates a single parked
    slot, using `model_io::session_pool_bytes` and following the exact
    "refuse with the whole subtraction shown" pattern `--max-context`
    already established (`crates/model-io/src/context_policy.rs`'s
    `ContextTooLarge`) -- the alternative is an unattributed Metal
    allocation error with no number in it pointing back at the flag. No
    `auto`, unlike `--expert-cache-slots`: there is no measured throughput/
    footprint trade a formula could resolve from machine size alone, only
    a policy choice about expected concurrent-conversation count that only
    the operator running this deployment knows.

    **The orphan-flag refusal follows the `--steering-*`-without-
    `--steering` precedent exactly** (`args.rs`'s six steering flags):
    `--session-slots N > 1` while `--prefix-reuse off` is a parse-time
    error, because a parked session is never reused without prefix reuse
    enabled and a command line naming a pool while explicitly disabling the
    one thing that could ever populate it says one thing while the server
    does another. Unlike the steering flags' orphan check, this one needs
    no separate "was it explicit" bookkeeping: `1` is the only value
    reachable without naming the flag at all, so `session_slots > 1` alone
    already means the caller asked for a pool on purpose.

    **Observability follows `reused_prefix_tokens`'s exact precedent, one
    field over**: `RawDecodeResult::session_slot_evicted` (set inside
    `crate::session_pool`'s `reset()` path, `crates/runtime/CLAUDE.md`
    Gotcha 32) reaches `ServerEvent::Generated.sessionSlotEvicted` through
    the same `ReportingModel` choke point `reused_prefix_tokens` already
    uses. It answers a narrower and more actionable question than "is
    pooling on": whether it is being CHURNED under, which is what an
    operator actually needs to decide whether to raise `--session-slots`.
    The startup line only prints when the resolved pool holds more than
    the one live session (`RealForwardRunner::session_pool_size() > 1`),
    matching the guardrails/prefix-reuse lines' always-print shape for a
    fixed toggle but the reasoning-line's say-nothing-when-off shape for a
    flag whose default genuinely has nothing to report.
