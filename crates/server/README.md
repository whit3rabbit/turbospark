# turbospark-server

High-performance local HTTP inference server (`turbospark-server`) built on Axum. Implements OpenAI `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, and `/v1/embeddings`; Anthropic `/v1/messages` and `/v1/messages/count_tokens`; Ollama-compatible `/api/{tags,version,show,chat,generate,embeddings,embed}`; `/v1/models`; and `/health` liveness probes.

Supports both non-streaming JSON responses and Server-Sent Events (SSE) / NDJSON streaming formats. Uses zero-allocation translations between Anthropic and OpenAI protocols.

## Purpose & Role

`turbospark-server` exposes local Apple Silicon inference to third-party tools, IDE extensions (Continue, Cursor, Copilot), and agent frameworks (Claude Code, OpenCode, Hermes) through standardized OpenAI and Anthropic HTTP APIs.

## Binary Execution

```sh
# Launch server against a real .gturbo model install (macOS, release mode required)
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo [--port 8080] [--bind loopback|tailnet]

# Launch server with round-robin multi-model concurrency pool
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo --pool-size 2

# Attach distinct installs and route requests by model id or alias
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model ~/models/gemma4.gturbo --model ~/models/qwen38-27b.gturbo

# Launch scripted server (canned completions, for client integration testing)
cargo run -p turbospark-server --bin turbospark-server -- <tokenizer-dir> [port]
```

## Key Modules

- `main.rs` & `args.rs`: Binary entry point and command-line flag parser (`--model`, repeatable for distinct installs, `--bind`, `--port`, `--pool-size`, `--api-key`, `--max-queue-depth`).
- `handler/`: Core handlers for `/v1/chat/completions`, `/v1/models`, and `/health`, and generation orchestration (`plan`, `run_full`, `stream_blocking`).
- `messages.rs`: Handler for Anthropic `/v1/messages` and `/v1/messages/count_tokens`, translating Anthropic messages to/from the OpenAI pipeline.
- `completions.rs`: Handler for legacy OpenAI `/v1/completions` (raw prompts without chat templates).
- `responses/`: Handler for the OpenAI `/v1/responses` endpoint.
- `embeddings.rs` & `encoder_model.rs`: `/v1/embeddings` and Ollama embedding routes backed by BERT/XLM-RoBERTa encoder models.
- `ollama.rs`: Ollama-compatible `/api/*` endpoints with NDJSON streaming.
- `guardrails.rs` & `guardrails/`: Tool-call rescue parser, schema repair, and validation retry loop.
- `registry.rs`: Model routing registry, including `PoolRegistry` for round-robin dispatch across multiple loaded model instances.
- `queue.rs`: Request FIFO generation queue managing concurrent request buffering and queue timeouts.
- `observe.rs`: Prometheus metrics collector and request latency instrumentation.
- `vision.rs`: Multimodal image URL decoding, download limits, and vision tensor injection.
- `bind.rs`: Network interface binding (loopback vs Tailscale tailnet).
- `auth.rs`: Bearer token and API key validation.
- `cancel.rs`: Client disconnect cancellation handling via cancellation tokens.
- `model.rs` & `real_model.rs`: `ChatModel` abstraction and `RealChatModel` driver.
- `response.rs`: Response envelopes and SSE chunk formatting.

## Development & Test Commands

```sh
# Run all server unit and integration tests
cargo test -p turbospark-server

# Test endpoint accessibility against a running server
curl -s http://127.0.0.1:8080/v1/models
```

## Tests

This crate contains 22 integration test suites in `tests/`:
- Protocol compatibility: `chat_completions.rs`, `completions.rs`, `messages.rs`, `responses.rs`, `ollama.rs`, `embeddings_api.rs`.
- Concurrency and queues: `generation_queue.rs`, `registry.rs`, `cancellation.rs`.
- Security and transport: `auth.rs`, `streaming_error_framing.rs`.
- Tool calling and guardrails: `guardrails.rs`, `system_prompt.rs`, `harmony_channels.rs`, `reasoning_channels.rs`.
- Multimodal and real model tests: `images.rs`, `real_backend.rs`, `observe.rs`.

## Crate Gotchas

1. **Serialized Real Backend & Concurrency Pools**: A single `RealChatModel` instance processes requests serially because Metal forward passes require mutable GPU buffers and KV cache state. For concurrent request processing, configure `--pool-size N` to instantiate multiple runner instances managed by `PoolRegistry`.
2. **Anthropic SSE Framing Differences**: Anthropic SSE streams use explicit `event:` header lines (e.g. `content_block_delta`, `message_delta`) and terminate with a `message_stop` event rather than the OpenAI `data: [DONE]` sentinel.
3. **Tailnet Binding Security**: Setting `--bind tailnet` binds exclusively to the Tailscale IPv4 carrier grade interface (100.64.0.0/10) with mandatory API key authentication.
