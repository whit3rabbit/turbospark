---
description: >-
  Configure the turbospark-server router with an API key, an event observer, a
  model registry, and optional image and audio providers.
icon: sliders
---

# Router options and state

`RouterOptions` holds the two settings for the router: an API key and an observer. `ServerState` holds what the router serves: the model registry, the optional image and audio providers, and the limit on concurrent model resolutions. Start with `build_router` when you have one model and no auth needs. Move to `build_router_with_options` and `ServerState` when you need a key, event logging, or more than one backend.

## RouterOptions

`RouterOptions` derives `Default` and `Clone`, so you can set the fields you need and leave the rest at their defaults:

```rust
use turbospark_server::RouterOptions;

let options = RouterOptions {
    api_key: Some("sk-local-test".to_string()),
    ..Default::default()
};
```

The fields are:

* `api_key: Option<String>`: requires `x-api-key: <key>` or `Authorization: Bearer <key>` on every route in this router except `GET /health`. An empty `x-api-key` header counts as absent, and the `Bearer` scheme matches case-insensitively. The key comparison runs in constant time. `None` leaves the router unauthenticated.
* `observer: Option<Arc<dyn observe::ServerObserver>>`: receives a `ServerEvent` for each request and generation. `None` records nothing, and the router does not build the event values at all.

The key guards only the routes this router registers. If you merge the router into an app that has its own routes, those routes stay open unless you wrap them in your own auth layer. Requests rejected by the key check still reach the observer. Auth failures appear as finished requests with a 401 status, which keeps a console honest about what was refused.

## Observers

`ServerObserver` has one method, `record`, which takes an owned `ServerEvent`. The router calls it from request threads, including threads that run a generation, so keep it fast and do not call back into the router. The example below logs the status and duration of each finished request:

```rust
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;

use turbospark_server::observe::{ServerEvent, ServerObserver};
use turbospark_server::registry::StaticRegistry;
use turbospark_server::{build_router_with_options, ChatModel, RouterOptions, ScriptedChatModel, ServerState};
use turbospark_tokenizer::MfTokenizer;

struct LogFinished;

impl ServerObserver for LogFinished {
    fn record(&self, event: ServerEvent) {
        if let ServerEvent::RequestFinished { status, duration_ms, .. } = event {
            eprintln!("request finished: status={status} duration_ms={duration_ms}");
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tokenizer_dir = PathBuf::from(std::env::args().nth(1).ok_or("usage: options <tokenizer-dir>")?);
    let tokenizer = MfTokenizer::load_from_dir(&tokenizer_dir)?;
    let model: Arc<dyn ChatModel> = Arc::new(ScriptedChatModel::new(tokenizer, 4096, Vec::new()));

    let registry = StaticRegistry::new(vec![model])?;
    let state = ServerState::new(Arc::new(registry))
        .with_max_concurrent_resolutions(NonZeroUsize::new(2).expect("nonzero"));

    let _router = build_router_with_options(
        state,
        RouterOptions {
            api_key: Some("sk-local-test".to_string()),
            observer: Some(Arc::new(LogFinished)),
        },
    );
    Ok(())
}
```

The events cover the request lifecycle and model changes. The variants are `RequestStarted`, `RequestRouted`, `Generated`, `RequestFinished`, `ModelAttached`, and `ModelDetached`. The enum is `serde`-serializable with a camelCase `kind` tag, so a host can forward events as JSON without a mapping layer. Check `docs.rs` for the field list of each variant before you match on one.

## ServerState

`ServerState` is the value you pass as the first argument. You rarely build it field by field. Use one of these constructors:

* `ServerState::new(registry)` takes an `Arc<dyn ModelRegistry>`.
* `ServerState::from(model)` takes an `Arc<dyn ChatModel>` and wraps it in a single-model registry. `build_router(model)` does this for you.
* `with_image_provider(provider)` attaches an `Arc<dyn ImageProvider>`.
* `with_audio_provider(provider)` attaches an `Arc<dyn AudioProvider>`.
* `with_max_concurrent_resolutions(n)` sets how many model resolutions can run at once. The default is 4. A request that arrives while the limit is reached gets a retryable 503. Resolution can open an installed model synchronously, so this limit keeps those opens off the request workers.

The audio routes and `/v1/metrics` exist only on a state with an audio provider. A host that attaches no audio provider keeps the chat-only route surface.

## Model registries

A registry decides which backend serves each request. The default rule is an exact match on the requested model id. If the id does not match and exactly one model is attached, that model serves the request whatever name the client sent.

Three registries ship with the crate:

* `SingleModel::new(model)` serves one backend. `ServerState::from` uses it.
* `StaticRegistry::new(models)` serves a fixed list. It returns `Err(String)` when two backends share a public id, rather than letting one shadow the other.
* `PoolRegistry::new(members)` serves several runners that advertise the same id. Use it when you want more than one runner for one model.

For a registry of your own, implement `ModelRegistry`. It needs `resolve(&self, requested: Option<&str>) -> Resolution` and `rows(&self) -> Vec<ModelRow>`. The `resolve_embedding` method has a default that calls `resolve`. A host that loads models while it runs can keep its own locked storage behind this trait, as the FFI host does.

{% hint style="info" %}
`ScriptedChatModel` reports the id `scripted` for every instance. Two scripted models in one `StaticRegistry` therefore conflict. Use a custom `ChatModel` with its own `model_id` when you need several backends in one registry.
{% endhint %}

## Concurrency on the real backend

A real `RealChatModel` on macOS runs one generation at a time. It holds one runner, and that runner needs exclusive access to its GPU buffers and KV cache state. Concurrent requests wait in a first-in, first-out queue, and a request whose client disconnects while queued is dropped without generating.

The `ChatModel` trait exposes `generation_queue`, which returns an optional `GenerationQueue`. Its default is `None`, and a real backend supplies a queue. For more parallel runners, attach several models through a `PoolRegistry` rather than raising the resolution limit. The resolution limit controls model opens, not generation throughput.

## Image and audio providers

Attach these only when your host can serve them. Both are traits with `Send + Sync` bounds, and both are re-exported from the crate root.

`ImageProvider` has two methods. `models` lists the image model ids, and `generate` takes an `ImageGenerateRequest` with `model`, `prompt`, `width`, `height`, and `seed` fields. It returns a boxed future that resolves to the PNG bytes or an `ImageError`, which is `Busy`, `Cancelled`, or `Failed(String)`. Without a provider, or with a provider that lists no models, `/v1/images/generations` returns 503.

`AudioProvider` covers transcription, speech, music generation, and model listing. Its required methods are `models`, `transcribe`, `speak`, and `generate`. `load` and `unload` have default implementations. Errors arrive as `AudioError`, which has the variants `Busy`, `Cancelled`, `NotFound`, `Invalid`, `Unsupported`, and `Failed`. The [audio API reference](https://github.com/whit3rabbit/turbospark/blob/main/docs/AUDIO_API.md) describes the wire shapes for those routes.

## Route list

`turbospark_server::SERVER_ROUTE_PATHS` lists every path a full server registers, audio routes included. The crate's contract tests use it to check the OpenAPI specifications. It does not reflect which optional routes a particular state enables, so check the state you built before you rely on a path.

## Limits to plan around

* Chat and text routes accept request bodies up to 25 MiB. Image and audio routes have their own, separate limits.
* The default resolution limit is 4 concurrent resolutions.
* `MAX_IMAGE_PROMPT_BYTES` is public, and an image prompt over that size is refused.

For the wire behavior behind these limits, see the [HTTP API reference](../../../http/http.md).
