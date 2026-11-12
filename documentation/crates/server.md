---
description: "The turbospark-server crate: an embeddable axum library and binary serving OpenAI, Anthropic, and Ollama-compatible routes against in-process model backends."
icon: server
---

# server

`crates/server` (package `turbospark-server`) is both a library (crate name
`turbospark_server`) and the `turbospark-server` binary. It builds one axum
router that speaks several wire formats against in-process backends: an
OpenAI-compatible surface, an Anthropic-compatible surface translated with
`anyllm_translate`, Ollama-compatible routes with NDJSON framing, and an
optional audio surface. Applications embed the router through the library;
end users start it through the binary, whose operational flags are
documented in [Server Configuration](../command-line-and-local-api/server.md).

## What it does

| Route group | Summary | Framing |
| --- | --- | --- |
| OpenAI-compatible | Chat completions, legacy completions, responses, embeddings, model listing | SSE or JSON |
| Anthropic-compatible | Messages plus a count-only token endpoint (no generation) | SSE (no `[DONE]` sentinel) or JSON |
| Ollama-compatible | Tags, version, show, chat, generate, embeddings | NDJSON (`stream` defaults to true here) |
| Image | One image generation route; an edits route answers unsupported | JSON |
| Audio | Transcription, speech, music, and job routes behind an `AudioProvider` | JSON, multipart uploads, WebSocket realtime |
| Liveness | `GET /health`, registry read only, never queued behind a generation | JSON |

The authoritative route list is `SERVER_ROUTE_PATHS` in `src/lib.rs` and
`build_router_with_options` itself; per-endpoint request and response shapes
live in the API Reference space and in
[Calling the Local API](../command-line-and-local-api/call-the-local-api.md).

Cross-cutting pieces:

- **Auth**: `--api-key` (or `$TURBOSPARK_API_KEY`) requires `x-api-key` or
  `Authorization: Bearer` on every route except `/health`, via a middleware
  layer applied to the protected sub-router only.
- **Guardrails**: on by default; rescues a tool call the decoder could not
  parse, validates arguments against the request's schema, and re-asks once.
  A request carrying tools is buffered rather than streamed while this is on.
- **Registry**: requests route by model id; with exactly one model attached,
  it serves every request whatever name was asked for.

## Where it sits

The library depends on `runtime` (the raw-completion loop), `selection`
(sampler shaping), `tokenizer`, `core`, and the portable `audio` and
`vision-io` crates, plus `axum`, `tokio`, `anyllm_translate`, and
`forge-guardrails`. `catalog` and `repack` are macOS-only dependencies,
gated beside the real backend: `--model` resolution goes through
`catalog::resolve_model_arg`, the same contract the CLI uses. The crate
overrides `rust-version` to 1.87 (the workspace floor is 1.82) because of
`forge-guardrails`; it is the one crate in the workspace that does.

Consumers: the `turbospark-server` binary, `crates/ffi` (which embeds the
router in-process), and `crates/cli`'s `serve` command, which execs the
binary.

## Public API tour

Root re-exports from `crates/server/src/lib.rs`:

| Item | Path | Meaning |
| --- | --- | --- |
| `build_router(state)` / `build_router_with_options(state, options)` | `src/lib.rs` | The axum `Router`; the route list in its doc comment is authoritative |
| `ServerState` | `src/lib.rs` | Registry plus optional image, audio, and observer attachments; `From<Arc<dyn ChatModel>>` wraps a single model |
| `RouterOptions` | `src/lib.rs` | `api_key: Option<String>` and `observer: Option<Arc<dyn ServerObserver>>` |
| `ChatModel`, `ScriptedChatModel` | `src/model.rs` | The backend trait (which decode loop runs) and the portable canned-logits backend |
| `RealChatModel` (macOS only) | `src/real_model.rs` | A `RealForwardRunner` behind the same trait; `open` takes the process-level knobs |
| `registry::{ModelRegistry, SingleModel, StaticRegistry}` | `src/registry.rs` | Resolution policy: exact id, else the single attached model, else a 404 |
| `GenerationQueue`, `GenerationPermit` | `src/queue.rs` | One-permit FIFO admission for the serialized real backend |
| `GuardrailConfig` | `src/guardrails.rs` | Rescue parsing, argument validation, retry budget |
| `observe::{ServerObserver, ServerEvent}` | `src/observe.rs` | Request and generation events reported from one choke point |
| Wire types (`ChatCompletionRequest`, `ChatCompletionResponse`, `MessageCreateRequest`, ...) | re-exported from `anyllm_translate` | The envelopes every route reads and writes |
| `AudioProvider`, `ImageProvider` and friends | `src/audio.rs`, `src/images.rs` | The provider traits that attach the optional route groups |

`ChatModel` is the seam between the wire and the engine: handlers plan
(chat template, encode, shaping config) and then call
`ChatModel::run_completion`. `ScriptedChatModel` replays a fixed logit
script and is what the whole integration suite drives; `RealChatModel`
holds the one real runner per process.

## Examples

### Embed the router in a Rust process

Adapted from `crates/server/tests/chat_completions.rs` (simplified: the
test generates scripted one-hot steps from a fixture tokenizer):

```rust
use std::sync::Arc;
use turbospark_server::{build_router, ChatModel, ScriptedChatModel};

async fn spawn(model: Arc<dyn ChatModel>) -> String {
    let router = build_router(model);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}
```

`build_router` takes anything that converts into `ServerState`, so a single
backend needs no registry of its own. Add options where a deployment needs
them:

```rust
use turbospark_server::{build_router_with_options, RouterOptions, ServerState};

let options = RouterOptions {
    api_key: Some("sk-local".to_string()),
    observer: None,
};
let router = build_router_with_options(ServerState::from(model), options);
```

Every route except `GET /health` then requires `x-api-key` or
`Authorization: Bearer`.

### Run the real server binary

From the crate's module guide (macOS; `--model` takes a `.gturbo` directory
or a `turbospark-model` alias):

```sh
cargo run --release -p turbospark-server --bin turbospark-server -- \
  --model gemma4 --port 8080 --api-key "$TURBOSPARK_API_KEY"
```

### Run the portable scripted server

A tokenizer-only mode in which every response comes from
`ScriptedChatModel`, not a forward pass. It always binds loopback:

```sh
cargo run -p turbospark-server --bin turbospark-server -- <tokenizer-dir> [port]
```

## Gotchas

{% hint style="warning" %}
The real backend is one runner per process. Generation requests serialize
behind a one-permit FIFO queue (`GenerationQueue`), and a capped
`--max-tokens-per-sec` lengthens each hold. `/health` deliberately reads
only the registry so a liveness probe never queues behind a decode; do not
move it onto the generation path.
{% endhint %}

{% hint style="warning" %}
The server provides no TLS. `--bind tailnet` binds only a verified
Tailscale IPv4 address and refuses to start without `--api-key` or a
non-empty `$TURBOSPARK_API_KEY`, but it never encrypts traffic; put a TLS
terminator in front for anything beyond a trusted tailnet. The audio routes
exist only on a state with an attached `AudioProvider`, so embedders of
embedding-only hosts keep a smaller route surface.
{% endhint %}

## Related pages

- [runtime](runtime.md)
- [catalog](catalog.md)
- [ffi](ffi.md)
- [Server Configuration and Deployment](../command-line-and-local-api/server.md)
- [Calling the Local API](../command-line-and-local-api/call-the-local-api.md)
