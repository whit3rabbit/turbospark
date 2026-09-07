# turbospark-server

HTTP server (`turbospark-server`) built on Axum, implementing OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/embeddings`; Anthropic `/v1/messages` and `/v1/messages/count_tokens`; Ollama-compatible `/api/{tags,version,show,chat,generate,embeddings,embed}`; `/v1/models`; and a `/health` liveness probe. Every generation endpoint supports non-streaming JSON and Server-Sent Events (SSE) streaming formats (Ollama's is NDJSON instead).

The server uses `anyllm_translate` wire types for IO-free translation between Anthropic and OpenAI request and response formats.

## Binary Execution

```sh
# Launch server against a real .gturbo model install (macOS, release mode required)
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo [--port 8080] [--bind loopback|tailnet]

# Launch scripted server (canned completions, for integration testing)
cargo run -p turbospark-server --bin turbospark-server -- <tokenizer-dir> [port]
```

## Key Modules

- `main.rs` / `args.rs`: Server binary entry point and command-line configuration (`--model`, `--bind`, `--port`, and many more; `--help` lists them all).
- `handler/`: Handlers for `/v1/chat/completions`, `/v1/models`, and `/health`, plus the shared generation core (`plan`, `run_full`, `stream_blocking`).
- `messages.rs`: Handler for Anthropic `/v1/messages` and `/v1/messages/count_tokens`, translating requests/responses to/from the OpenAI pipeline.
- `completions.rs`: Handler for the legacy OpenAI `/v1/completions` endpoint (raw prompt, no chat template).
- `responses/`: Handler for the OpenAI `/v1/responses` endpoint.
- `embeddings.rs` / `encoder_model.rs`: `/v1/embeddings` and the Ollama embedding routes, and the encoder backend behind them.
- `ollama.rs`: The Ollama-compatible `/api/*` routes (NDJSON, not SSE).
- `guardrails.rs`: Tool-call rescue, argument validation, and the retry loop.
- `registry.rs`: Which attached model serves a request.
- `model.rs`: `ChatModel` trait and `ScriptedChatModel` backend.
- `real_model.rs`: `RealChatModel` driving `RealForwardRunner` on macOS.
- `response.rs`: OpenAI response envelope and SSE chunk constructors.

## Development & Test Commands

```sh
# Run integration tests for turbospark-server
cargo test -p turbospark-server

# Test endpoint accessibility against a running server
curl -s http://127.0.0.1:8080/v1/models
```

## Crate Gotchas

1. **Serialized Real Backend**: `RealChatModel` owns a single `RealForwardRunner` behind a `Mutex`. Requests are processed serially because forward passes require mutable model state and Metal resources.
2. **Anthropic SSE Framing**: Anthropic SSE streams require explicit `event:` headers and end at `message_stop` without an OpenAI `[DONE]` sentinel.
3. **Tailnet Binding**: `--bind tailnet` binds exclusively to the Tailscale IPv4 interface (100.64.0.0/10) without TLS or application authentication; Tailnet ACLs provide the sole access control.
