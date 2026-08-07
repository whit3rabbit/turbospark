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
|   +-- model.rs                # ScriptedChatModel backend adapter for mrefrust-runtime
|   +-- request.rs              # OpenAI Chat Completions API request payload structs
|   \-- response.rs             # OpenAI Chat Completions API response & SSE event serializers
\-- tests/
    +-- chat_completions.rs     # Integration tests for non-streaming and SSE streaming endpoints
    \-- fixtures/               # Test tokenizer fixtures for integration tests
```

## Key Modules

- `main.rs`: Server binary entry point and CLI option handling (tokenizer directory, port).
- `handler.rs`: `/v1/chat/completions` Axum request handler routing non-streaming and SSE stream responses.
- `model.rs`: `ScriptedChatModel` backend bridging Axum handlers to `mrefrust-runtime`.
- `request.rs`: OpenAI Chat Completions JSON request payload deserialization.
- `response.rs`: OpenAI Chat Completions JSON response and SSE chunk serialization.

## Development & Test Commands

```sh
# Run tests for mrefrust-server
cargo test -p mrefrust-server

# Launch local mference-server
cargo run -p mrefrust-server --bin mference-server -- <tokenizer-dir> [port]
```

## Crate Gotchas

1. **Scripted Backend Only**: `mference-server` currently uses `ScriptedChatModel` powered by `ScriptedLogitProducer`. It returns scripted token completions for API layout validation and integration testing; real weight forward runner wiring is future roadmap work.
