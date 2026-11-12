# turbospark-server src

## Overview
Library and binary for the OpenAI-, Anthropic- and Ollama-compatible generation server. `lib.rs` builds the axum router; models are resolved per request through a registry; generation is serialized through a FIFO queue. The standalone binary lives in `main.rs` and `args.rs`; the app embeds the library through FFI.

## Interface
- [lib.rs:build_router], [lib.rs:build_router_with_options], [lib.rs:RouterOptions], [lib.rs:ServerState], [lib.rs:SERVER_ROUTE_PATHS].
- Routes: `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/messages`, `/v1/messages/count_tokens`, `/v1/models`, `/v1/models/{model}`, `/v1/embeddings`, `/v1/images/generations` (`/v1/images/edits` refuses), Ollama `/api/*`, `/health`, and the audio routes when a provider is attached ([lib.rs:SERVER_ROUTE_PATHS], [lib.rs:build_router_with_options]).
- Backends: [model.rs:ChatModel], [model.rs:ScriptedChatModel], [real_model.rs:RealChatModel], [encoder_model.rs:RealEncoderModel], [coreml_embedding_model.rs:CoreMlEmbeddingModel].
- Registry: [registry.rs:ModelRegistry], [registry.rs:ModelLoader], [registry.rs:PoolRegistry], [registry.rs:StaticRegistry], [registry.rs:SingleModel], [registry.rs:ModelLoaderRegistry].
- Queue: [queue.rs:GenerationQueue], [queue.rs:QueueTicket], [queue.rs:GenerationPermit], [queue.rs:QueueFull]; waiters are capped at [queue.rs:DEFAULT_MAX_QUEUED] (64, `--max-queued`, 0 is unlimited) and a request past the cap gets 429 with `Retry-After`.
- Resolution: [lib.rs:ServerState::with_max_concurrent_resolutions] bounds blocking model resolutions (default 4); a saturated executor returns a retryable 503.
- Observability: [observe.rs:ServerObserver], [observe.rs:ServerEvent], [observe.rs:RequestIds].
- Providers: [images.rs:ImageProvider], [audio_real.rs] (real audio provider), [vision.rs] (vision input handling).
- Networking: [auth.rs:ApiKey], [bind.rs:BindMode], [host_check.rs:is_loopback_host], [cancel.rs:new_cancel].
- Endpoints: [completions.rs:completions], [messages.rs:messages], [embeddings.rs:embeddings], [ollama.rs:chat], [images.rs:generations].
- Subdirectories: [audio](./audio/EGM.md), [guardrails](./guardrails/EGM.md), [handler](./handler/EGM.md), [responses](./responses/EGM.md); `guardrails.rs` is the parent module of the guardrails directory.

## Invariants
- Bearer / `x-api-key` auth applies to every route except `GET /health` ([lib.rs:build_router_with_options], [auth.rs:ApiKey]).
- With no API key, every route except `/health` refuses a non-loopback `Host` ([host_check.rs:is_loopback_host]). The loopback set also admits `host.docker.internal` so a Docker Desktop container can reach the server.
- A Tailnet bind has no path to a wildcard or LAN address ([bind.rs:BindMode]).
- Request bodies are size-limited: [lib.rs:MAX_REQUEST_BODY_BYTES] (25 MiB) by default, [images.rs:MAX_IMAGE_REQUEST_BODY_BYTES] on `/v1/images/generations`, [audio/mod.rs:MAX_AUDIO_UPLOAD_BYTES] on the audio router.
- No audio route is registered unless an audio provider is attached ([lib.rs:ServerState::with_audio_provider]).
- With no observer, no event is built ([lib.rs:RouterOptions]).

## Side Effects
- Binds a TCP listener and serves HTTP/SSE/NDJSON/WebSocket ([main.rs]).
- Loads and evicts models, GPU and memory use, through the registry ([registry.rs:ModelRegistry]).
- Spawns blocking generation tasks ([handler/exec.rs:run_full]).

## Verification
- Run: `cargo test -p turbospark-server`
