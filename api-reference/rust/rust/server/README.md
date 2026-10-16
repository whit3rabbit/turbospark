---
description: >-
  Embed TurboSpark's local inference routes in an existing Axum service with the
  turbospark-server crate.
icon: box
---

# HTTP server

`turbospark-server` returns the HTTP routes that `turbospark serve` runs as an Axum `Router`. Use it when your application already runs Axum and you want one process to serve your own routes next to TurboSpark's OpenAI, Anthropic, and Ollama-compatible endpoints. If you only need a local server, run the `turbospark-server` binary or `turbospark serve` and skip the library.

The crate does not bind a port, terminate TLS, or pick a model for you. Your code does all three. It also decides whether requests need an API key. The [HTTP API reference](../../../http/http.md) documents every endpoint, request, and response shape, so this page covers only the Rust side.

## Install

The package is `turbospark-server`, and Rust code imports it as `turbospark_server`. Inside the TurboSpark workspace, source files import some crates under short aliases such as `foundation` and `tokenizer`. Those aliases are internal to the workspace, so use the package names.

Add the package from crates.io once the version you need is published. Until then, use a path or git dependency from a checkout:

```toml
[dependencies]
turbospark-server = { path = "../turbospark/crates/server" }
turbospark-tokenizer = { path = "../turbospark/crates/tokenizer" }
axum = "0.7"
tokio = { version = "1", features = ["full"] }
```

Two version constraints apply. The router type comes from axum 0.7. A service on axum 0.8 cannot merge it directly. The crate declares `rust-version = "1.87"`, which is higher than the workspace floor of 1.82.

## What the crate exposes

The public surface is small:

* `build_router(state)` returns a `Router` with no authentication.
* `build_router_with_options(state, options)` adds an API key and an event observer through `RouterOptions`.
* `ServerState` holds the model registry, plus optional image and audio providers.

Chat, completions, responses, messages, embeddings, Ollama, and model listing routes are always registered. The audio routes and `/v1/metrics` exist only when you attach an audio provider. `/v1/images/generations` always exists, but it returns 503 until you attach an image provider. The [router options page](router-options.md) covers those settings in detail.

## Smallest working example

This program merges the TurboSpark router into an app with one route of its own. It uses `ScriptedChatModel`, which replays a fixed logit sequence and needs no model weights. That makes it useful for wiring and tests, not for real output.

```rust
use std::path::PathBuf;
use std::sync::Arc;

use axum::routing::get;
use axum::Router;
use turbospark_server::{build_router_with_options, ChatModel, RouterOptions, ScriptedChatModel};
use turbospark_tokenizer::MfTokenizer;

async fn hello() -> &'static str {
    "hello from the host app"
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_dir = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or("usage: embed_check <tokenizer-dir>")?,
    );
    let tokenizer = MfTokenizer::load_from_dir(&tokenizer_dir)?;

    // Scripted backend: serves listing and auth routes with no generation.
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedChatModel::new(tokenizer, 4096, Vec::new()));
    let api = build_router_with_options(
        model,
        RouterOptions {
            api_key: Some(std::env::var("APP_API_KEY")?),
            ..Default::default()
        },
    );

    // api_key guards only the routes in `api`. Routes added to `app` here are open.
    let app = Router::new().route("/app/hello", get(hello)).merge(api);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8080").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

Run it with a tokenizer directory. The ChatML fixture in the repository works: `crates/server/tests/fixtures/ChatMLTokenizer`.

```sh
APP_API_KEY=sk-local cargo run -- /path/to/tokenizer-dir
curl -s http://127.0.0.1:8080/health
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/v1/models
curl -s http://127.0.0.1:8080/v1/models -H 'x-api-key: sk-local'
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/app/hello
```

The health check returns `{"status":"ok","state":"ready"}`. The second request returns `401` because it sends no key, and the third lists the attached model under two ids, `scripted` and `claude-turbospark-scripted`. The last request returns `200` with no key, because the route you added is outside the key check. Do not send generation requests to this example, because the scripted backend has no logit steps to replay.

For a real backend on macOS, construct a `RealChatModel` and pass it to the same functions. `RealChatModel::open` takes a model directory plus several runtime options, so check its [docs.rs entry](https://docs.rs/turbospark-server/latest/turbospark_server/struct.RealChatModel.html) before you call it.

## Security boundary

The library provides no network security. Without an `api_key`, every route accepts requests from anyone who can reach the listener. The crate never adds TLS. The binary binds `127.0.0.1` by default, but an embedded router uses whatever address your listener binds.

{% hint style="warning" %}
`build_router` applies no authentication. If the listener is reachable from other machines, set `RouterOptions::api_key`, put TLS in front of the listener, or bind to loopback.
{% endhint %}

The key covers only the routes on the TurboSpark router. A route you merge in with `Router::new().route(...)` is open unless you add your own auth layer to it, as the example's last comment says. The binary reads `--api-key` and forwards it to `RouterOptions`, and its source notes that the server "provides no TLS." Your application owns the bind address, TLS termination, and any user-level authorization beyond the shared key.

## Errors

`build_router` and `build_router_with_options` return a `Router` and cannot fail. Failures happen earlier, while you open a tokenizer or a model. Handle those where you open the backend. `MfTokenizer::load_from_dir` returns `TokenizerError`, `RealChatModel::open` returns `Result<Self, String>`, and `StaticRegistry::new` returns `Result<Self, String>` when two backends share a public id.

After the router is running, errors arrive as HTTP responses with JSON bodies. The main cases are:

| Status | When                                                           | Body                                                          |
| ------ | -------------------------------------------------------------- | ------------------------------------------------------------- |
| 401    | Missing or wrong API key on any route except `GET /health`     | `error.type` is `authentication_error`                        |
| 404    | Several models are attached and the request names none of them | `error.code` is `model_not_found`, with the available ids     |
| 503    | A requested installed model failed to load                     | `error.code` is `model_load_refused`, with `retryable: true`  |
| 503    | No attached model can serve this request kind                  | A message saying no capable model is loaded (no error code)   |
| 503    | Too many model resolutions are already running                 | Retryable, per `ServerState::with_max_concurrent_resolutions` |
| 503    | An image request arrives with no image provider attached       | A message only                                                |

For generation and embedding requests, a server with one capable model serves that model whatever name the client sent, so the 404 appears only when several capable models are attached. The `GET /v1/models/{id}` lookup is different. It matches the id exactly and returns 404 for any other name. Requests above the body limit get a `413` from Axum's body limit layer. Chat routes allow 25 MiB, and audio and image routes have their own limits.

## Where to go next

* [Router options and server state](router-options.md): API keys, observers, registries, and the audio and image providers.
* [HTTP API](../../../http/http.md): endpoints, request and response shapes, and the OpenAPI specifications.
* [docs.rs reference](https://docs.rs/turbospark-server/latest/turbospark_server/): every public type and method.
* [Streaming](https://github.com/whit3rabbit/turbospark/blob/main/docs/STREAMING.md), [tool calling](https://github.com/whit3rabbit/turbospark/blob/main/docs/TOOL_CALLING.md), and the [audio API](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) for behavior the wire contract does not cover.
