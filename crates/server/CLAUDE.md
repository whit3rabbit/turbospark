# mrefrust-server

OpenAI-compatible HTTP server (`mference-server`) implementing the `/v1/chat/completions` REST endpoint using Axum. Supports non-streaming and Server-Sent Events (SSE) streaming output.

## Directory & File Structure

```
crates/server/
+-- Cargo.toml                  # Crate manifest
+-- src/
|   +-- main.rs                 # Binary entry point for mference-server
|   +-- lib.rs                  # Library root re-exporting router and handlers
|   +-- handler.rs              # /v1/chat/completions HTTP endpoint handler (Axum & SSE)
|   +-- model.rs                # ChatModel trait and the ScriptedChatModel backend
|   +-- real_model.rs           # RealChatModel: RealForwardRunner backend (macOS only)
|   +-- request.rs              # OpenAI Chat Completions API request payload structs
|   \-- response.rs             # OpenAI Chat Completions API response & SSE event serializers
\-- tests/
    +-- chat_completions.rs     # Integration tests for non-streaming and SSE streaming endpoints
    +-- real_backend.rs         # Gated real-model end-to-end test (macOS, #[ignore]d)
    \-- fixtures/               # Test tokenizer fixtures for integration tests
```

## Key Modules

- `main.rs`: Server binary entry point and CLI option handling (`--model` real mode, legacy positional scripted mode, port, `--bind loopback|tailnet`).
- `handler.rs`: `/v1/chat/completions` Axum request handler routing non-streaming and SSE stream responses.
- `model.rs`: the `ChatModel` trait and `ScriptedChatModel`, bridging Axum handlers to `mrefrust-runtime`.
- `real_model.rs`: `RealChatModel`, a `RealForwardRunner` behind the same trait (macOS only).
- `request.rs`: OpenAI Chat Completions JSON request payload deserialization.
- `response.rs`: OpenAI Chat Completions JSON response and SSE chunk serialization.

## Development & Test Commands

```sh
# Run tests for mrefrust-server
cargo test -p mrefrust-server

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
4. **`--bind tailnet` fails rather than widening, and is not authentication.** It binds only the single address `tailscale ip -4` reports, and only when that address is a dotted-quad inside 100.64.0.0/10; zero, several, IPv6, out-of-range, or malformed output is an error, never a fall back to loopback or a wildcard. Every device the Tailnet ACL admits gets unauthenticated access to the full API (no auth, no TLS). The flag exists only in `--model` mode; the scripted `<tokenizer-dir>` mode always binds loopback.
