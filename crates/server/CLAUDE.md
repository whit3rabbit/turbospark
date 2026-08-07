# mrefrust-server

HTTP server (`mference-server`) on Axum, speaking two wire formats against one
local backend: OpenAI `/v1/chat/completions`, Anthropic `/v1/messages`, and
`/v1/models`. Both generation endpoints support non-streaming and Server-Sent
Events (SSE) streaming output.

The wire types are `anyllm_translate`'s (crates.io 0.16, default features:
pure, IO-free), not hand-rolled. An Anthropic request is translated into the
OpenAI request the existing path already understands, run through the shared
generation core in `handler.rs`, and translated back.

## Directory & File Structure

```
crates/server/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- main.rs                 # Binary entry point for mference-server
|   +-- lib.rs                  # Library root: router, re-exported wire types
|   +-- handler.rs              # /v1/chat/completions + /v1/models, and the shared generation core
|   +-- messages.rs             # Anthropic /v1/messages: translate in, generate, translate out
|   +-- model.rs                # ChatModel trait and the ScriptedChatModel backend
|   +-- real_model.rs           # RealChatModel: RealForwardRunner backend (macOS only)
|   \-- response.rs             # Constructors for the OpenAI response & SSE chunk envelopes
\-- tests/
    +-- chat_completions.rs     # Integration tests for the OpenAI endpoint
    +-- messages.rs             # Integration tests for /v1/messages, /v1/models, wider OpenAI shapes
    +-- real_backend.rs         # Gated real-model end-to-end test (macOS, #[ignore]d)
    \-- fixtures/               # Test tokenizer fixtures for integration tests
```

## Key Modules

- `main.rs`: Server binary entry point and CLI option handling (`--model` real mode, legacy positional scripted mode, port, `--bind loopback|tailnet`).
- `handler.rs`: the `/v1/chat/completions` and `/v1/models` handlers, plus the generation core both endpoints share -- `plan` (chat template, encode, shaping config), `run_full`, and `stream_blocking`.
- `messages.rs`: the Anthropic `/v1/messages` handler, wrapping the same core in `translate_request` / `translate_response` / `new_stream_translator`.
- `model.rs`: the `ChatModel` trait and `ScriptedChatModel`, bridging Axum handlers to `mrefrust-runtime`.
- `real_model.rs`: `RealChatModel`, a `RealForwardRunner` behind the same trait (macOS only).
- `response.rs`: constructors for `anyllm_translate::openai`'s response and SSE chunk envelopes, filling the many fields this server never populates in one place.

## Development & Test Commands

```sh
# Run tests for mrefrust-server
cargo test -p mrefrust-server

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
# build decodes far too slowly to be usable).
cargo run --release -p mrefrust-server --bin mference-server -- \
  --model ~/models/gemma4.gturbo [--port N] [--max-context N] [--expert-cache-slots N] \
  [--bind loopback|tailnet]

# Launch the portable scripted server (tokenizer only, canned completions).
cargo run -p mrefrust-server --bin mference-server -- <tokenizer-dir> [port]

# The gated real-model end-to-end test.
MREFRUST_GEMMA4_INSTALL_DIR=~/models/gemma4.gturbo \
  cargo test -p mrefrust-server --test real_backend --release -- --ignored --nocapture
```

## Crate Gotchas

1. **Two backends, and the real one is serialized.** `RealChatModel` (macOS only) owns ONE `RealForwardRunner` behind a `Mutex`: it costs a multi-gigabyte mapping plus a Metal pipeline compile to open and takes `&mut self`, so concurrent requests queue rather than run. `ScriptedChatModel` (`ScriptedLogitProducer`, canned completions) stays the portable backend the integration tests drive, and is the only one on non-macOS. That is why `ChatModel` exposes `with_producer` (lend a producer) rather than a `new_producer` factory.
2. **`encode(&prompt, false)`.** The chat template already emits the literal `<bos>`, so encoding the rendered prompt with a BOS prefix doubles it. Invisible on the ChatML test fixture (its `bos_token` is null); on a real Gemma install it degrades output.
3. **`top_k` defaults to 64, not 0.** `ShapingConfig::new` rejects `top_p < 1.0` when `temperature > 0` and `top_k == 0`, so a 0 default would 400 every standard OpenAI request that sets only `top_p`.
4. **The `anyllm_translate` `middleware` feature is NOT usable here, and the reason is not obvious.** It looks like exactly the right tool (it ships an `anthropic_compat_router` that adds `POST /v1/messages`), but `AnthropicCompatConfig` requires a `backend_url` and forwards over `reqwest` -- this server's backend is in-process, so enabling it would mean HTTP-hopping to ourselves on loopback. It also depends on **axum 0.8** while this crate is on **0.7**, which would put two incompatible `Router` / `Sse` / `Event` types in one binary. Use default features and call `translate_request` / `translate_response` / `new_stream_translator` directly. The consequence: `stream_event_to_sse` lives behind that feature and returns an axum-0.8 `Event`, so `messages.rs` hand-rolls the same ten-line variant-to-name match.
5. **`seed` has no field on the OpenAI request type; it lands in `extra`.** `anyllm_translate`'s `ChatCompletionRequest` gives explicit fields only to what needs translating, and sweeps the rest (`seed`, `logprobs`, `n`, `logit_bias`, ...) into a `#[serde(flatten)] extra` map. `build_config` reads it back out with `request.extra.get("seed")`. Drop that lookup and every seeded request silently becomes unseeded -- no deserialization error, no test failure unless one asserts determinism.
6. **Anthropic SSE is not OpenAI SSE with different JSON.** Clients dispatch on the `event:` line, which the OpenAI stream never sets, and the stream ends at `message_stop` with **no `[DONE]` sentinel**. Also, `StreamingTranslator` is a state machine over the OpenAI chunk *sequence*: it opens the message on the first chunk carrying a role and closes the content block on `finish()`. Feed it chunks in a different order than the chat route emits (role chunk, content chunks, finish chunk) and the Anthropic event structure comes out malformed rather than erroring.
7. **`--bind tailnet` fails rather than widening, and is not authentication.** It binds only the single address `tailscale ip -4` reports, and only when that address is a dotted-quad inside 100.64.0.0/10; zero, several, IPv6, out-of-range, or malformed output is an error, never a fall back to loopback or a wildcard. Every device the Tailnet ACL admits gets unauthenticated access to the full API (no auth, no TLS). The flag exists only in `--model` mode; the scripted `<tokenizer-dir>` mode always binds loopback.
