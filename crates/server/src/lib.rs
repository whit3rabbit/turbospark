//! OpenAI- and Anthropic-compatible generation server on loopback:
//! request/response envelopes, SSE streaming framing, and the axum router,
//! wired to `turbospark-runtime`'s raw-completion loop. Ported from the intent
//! of `Sources/MferenceServer` (an OpenAI-compatible
//! `/v1/chat/completions` endpoint), plus an Anthropic `/v1/messages`
//! endpoint that renders the same generation through `anyllm_translate` so
//! Anthropic-native clients need no proxy in between. The same core also
//! serves OpenAI's legacy `/v1/completions` and `/v1/responses`, an
//! OpenAI-shaped `/v1/embeddings`, and Ollama-compatible `/api/*` routes
//! (NDJSON rather than SSE); `build_router`'s own route list is the
//! authoritative one. Two backends implement
//! [`ChatModel`]: [`ScriptedChatModel`] (portable, fixed logit sequence, what
//! the tests drive) and [`RealChatModel`] (macOS only, a real
//! `RealForwardRunner` forward pass against a `.gturbo` install). Model
//! dialect auto-selection is the tokenizer's job here, not the server's.

mod auth;
mod cancel;
mod completions;
mod embeddings;
#[cfg(target_os = "macos")]
mod encoder_model;
mod guardrails;
mod handler;
mod messages;
mod model;
pub mod observe;
mod ollama;
mod queue;
#[cfg(target_os = "macos")]
mod real_model;
#[cfg(target_os = "macos")]
#[cfg(test)]
#[path = "real_model_tests.rs"]
mod real_model_tests;
pub mod registry;
mod response;
mod responses;
pub mod vision;

/// The header both routes report a degraded request on.
///
/// `anyllm_translate`'s own spelling, kept for `/v1/messages` where it
/// originated, and reused on `/v1/chat/completions` since ROADMAP M-V8 --
/// one client-visible mechanism rather than two names for the same signal.
pub(crate) const DEGRADATION_HEADER: &str = "x-anyllm-degradation";

/// axum 0.7's own default (2 MiB) is smaller than a single base64-encoded
/// photo: every image this server accepts arrives as JSON body text, so a
/// realistic multi-image chat request routinely exceeds it and gets a bare
/// plain-text 413 from the framework -- neither this crate's own JSON error
/// shape nor the `x-anyllm-degradation` header a too-large-to-serve image
/// gets everywhere else. 25 MiB is generous for several photos in one
/// request while still bounding the worst case; `vision::MAX_IMAGES_PER_REQUEST`
/// bounds image COUNT separately, since a body limit alone does not stop a
/// request from splitting its budget across an unreasonable number of tiny
/// images.
const MAX_REQUEST_BODY_BYTES: usize = 25 * 1024 * 1024;

#[cfg(target_os = "macos")]
/// Embedding model running encoder forward passes for embeddings.
pub use encoder_model::RealEncoderModel;
/// Tool-call guardrail configuration: rescue parsing, argument validation,
/// and the retry budget.
pub use guardrails::GuardrailConfig;
/// Shared application state for request handling.
pub use handler::AppState;
/// Trait and canned test backend for chat generation.
pub use model::{ChatModel, ScriptedChatModel};
pub use queue::{GenerationPermit, GenerationQueue};
#[cfg(target_os = "macos")]
/// GPU-backed chat model running real forward passes against `.gturbo` installs.
pub use real_model::RealChatModel;

// The wire envelopes are `anyllm_translate`'s types, re-exported under the
// names this crate has always used so callers see one surface.
pub use anyllm_translate::anthropic::{MessageCreateRequest, MessageResponse};
pub use anyllm_translate::openai::streaming::ChatCompletionChunk;
pub use anyllm_translate::openai::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, ChatUsage, Choice,
};

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;

/// Options [`build_router_with_options`] takes beyond the shared
/// [`ServerState`]. A struct rather than a bare `Option<String>` parameter so
/// a future option (a CORS policy, say) has somewhere to land without another
/// signature change; `Default` gives `build_router` its zero-option case for
/// free.
#[derive(Default, Clone)]
pub struct RouterOptions {
    /// Require this key on every request except `GET /health`. `None`
    /// (the default) leaves the router exactly as unauthenticated as it was
    /// before this option existed.
    pub api_key: Option<String>,
    /// Where request and generation events go. `None` (the default) means
    /// nothing is recorded and no event is even BUILT
    /// ([`observe::record`]), which is what keeps every pre-observer caller
    /// -- the whole integration suite among them -- on the exact path it had.
    pub observer: Option<Arc<dyn observe::ServerObserver>>,
}

/// What axum's `State` carries: which models are attached, and who is
/// watching.
///
/// **[`handler::AppState`] KEEPS ITS NAME AND ITS MEANING**, which is what
/// makes routing by model id a small change rather than a sweep. It is still
/// `Arc<dyn ChatModel>` -- one resolved backend -- and every internal
/// function that takes one (`plan`, `run_full`, `stream_blocking`,
/// `needs_decoder`, `run_guarded`, ...) is untouched. Only the axum entry
/// points take a `ServerState`, and each one's first act is to resolve it
/// down to an `AppState`.
#[derive(Clone)]
pub struct ServerState {
    pub(crate) registry: Arc<dyn registry::ModelRegistry>,
    pub(crate) observer: Option<Arc<dyn observe::ServerObserver>>,
    pub(crate) ids: Arc<observe::RequestIds>,
}

impl ServerState {
    pub fn new(registry: Arc<dyn registry::ModelRegistry>) -> Self {
        Self {
            registry,
            observer: None,
            ids: Arc::new(observe::RequestIds::default()),
        }
    }
}

/// So `build_router(model)` keeps compiling for every caller that predates
/// the registry, this crate's integration tests included.
impl From<Arc<dyn ChatModel>> for ServerState {
    fn from(model: Arc<dyn ChatModel>) -> Self {
        Self::new(Arc::new(registry::SingleModel::new(model)))
    }
}

impl From<Arc<dyn registry::ModelRegistry>> for ServerState {
    fn from(registry: Arc<dyn registry::ModelRegistry>) -> Self {
        Self::new(registry)
    }
}

/// [`build_router`] with no options: no auth. Every existing caller of this
/// function -- the integration tests in `tests/`, and the FFI's future
/// in-process server -- keeps the exact behavior it had before
/// [`RouterOptions`] existed.
///
/// See [`build_router_with_options`] for the route list.
pub fn build_router(state: impl Into<ServerState>) -> Router {
    build_router_with_options(state, RouterOptions::default())
}

/// Builds the Axum router bound to `state` with the following routes:
///
/// - `GET /health`: liveness/readiness probe, no generation lock taken,
///   and NEVER behind `options.api_key` -- an auth check answers "who are
///   you", and a liveness probe answering "no" to an unauthenticated caller
///   would be indistinguishable from the process being down.
/// - `POST /v1/chat/completions`: OpenAI-compatible chat completion endpoint
/// - `POST /v1/completions`: OpenAI's legacy raw-prompt completion endpoint
/// - `POST /v1/responses`: OpenAI's Responses API endpoint
/// - `POST /v1/messages`: Anthropic-compatible messages endpoint
/// - `POST /v1/messages/count_tokens`: Anthropic's count-only endpoint (no generation)
/// - `GET /v1/models`: OpenAI-compatible list of available models
/// - `GET /v1/models/:model`: OpenAI-compatible model detail endpoint
///
/// With `options.api_key` set, every route above `/health` requires
/// `x-api-key: <key>` or `Authorization: Bearer <key>` (`crates/server/src/
/// auth.rs`, `crates/server/CLAUDE.md` Gotcha 26); the layer is applied to a
/// sub-router BEFORE it merges with the unauthenticated `/health` route, so
/// merging cannot leak it onto that one.
pub fn build_router_with_options(state: impl Into<ServerState>, options: RouterOptions) -> Router {
    let mut state = state.into();
    state.observer = options.observer.clone();

    let protected = Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(messages::messages))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route("/v1/models", get(handler::models))
        .route("/v1/models/:model", get(handler::model_detail))
        .route("/v1/embeddings", post(embeddings::embeddings))
        .route("/api/tags", get(ollama::tags))
        .route("/api/version", get(ollama::version))
        .route("/api/show", post(ollama::show))
        .route("/api/chat", post(ollama::chat))
        .route("/api/generate", post(ollama::generate))
        .route("/api/embeddings", post(embeddings::ollama_embeddings))
        .route("/api/embed", post(embeddings::ollama_embed))
        // Explicit rather than left at axum's 2 MiB default -- see
        // `MAX_REQUEST_BODY_BYTES`'s own doc. `/health` is unaffected: it
        // carries no body and is never a member of this router.
        .layer(axum::extract::DefaultBodyLimit::max(MAX_REQUEST_BODY_BYTES));
    let protected = match options.api_key {
        Some(key) => protected.layer(axum::middleware::from_fn_with_state(
            auth::ApiKey(key.into()),
            auth::require_api_key,
        )),
        None => protected,
    };

    let router = Router::new()
        .route("/health", get(handler::health))
        .merge(protected);

    // **APPLIED TO THE MERGED ROUTER, AFTER THE AUTH LAYER, AND THOSE ARE
    // TWO SEPARATE PROPERTIES WITH TWO SEPARATE TESTS.** A console that
    // cannot show a REJECTED request is missing the rows somebody opened it
    // to find, and neither half alone gets there.
    //
    // WHICH ROUTER decides `/health` coverage. Moving this onto `protected`
    // (which reads as tidier, since that is where the other layer goes)
    // leaves `/health` unobserved: it was never a member of that router.
    // Mutation-checked -- that change reddens
    // `health_is_observed_even_though_it_is_exempt_from_auth` and NOTHING
    // else, the 401 case included.
    //
    // WHEN, relative to `auth`, decides the 401 coverage, and it survives
    // the move above because a later `.layer()` still wraps an earlier one.
    // Applying this one BEFORE the auth layer puts it inside: the rejection
    // returns from the outer layer and never reaches this code. That
    // mutation reddens `a_request_rejected_by_auth_is_still_recorded`.
    //
    // With no observer configured the layer is not added at all, so every
    // pre-observer caller pays nothing.
    let router = match options.observer {
        Some(_) => router.layer(axum::middleware::from_fn_with_state(
            state.clone(),
            observe_layer,
        )),
        None => router,
    };

    router.with_state(state)
}

/// Mints a request id, records the HTTP-level facts around a request, and
/// puts the id where a handler can find it.
///
/// It cannot see anything about generation -- see [`observe`]'s header for
/// why the counters come from the handlers instead.
async fn observe_layer(
    axum::extract::State(state): axum::extract::State<ServerState>,
    mut request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    let id = state.ids.next();
    let method = request.method().to_string();
    let path = request.uri().path().to_string();
    request.extensions_mut().insert(observe::RequestTag(id));

    observe::record(&state.observer, || observe::ServerEvent::RequestStarted {
        id,
        at_ms: observe::now_ms(),
        method,
        path,
    });

    let started = std::time::Instant::now();
    let mut response = next.run(request).await;
    let status = response.status().as_u16();
    let duration_ms = started.elapsed().as_millis() as u32;

    let error = if status >= 400 {
        let (parts, body) = response.into_parts();
        match axum::body::to_bytes(body, 64 * 1024).await {
            Ok(bytes) => {
                let err = extract_error_message(&bytes);
                response =
                    axum::response::Response::from_parts(parts, axum::body::Body::from(bytes));
                err
            }
            Err(_) => {
                response = axum::response::Response::from_parts(parts, axum::body::Body::empty());
                None
            }
        }
    } else {
        None
    };

    observe::record(&state.observer, || observe::ServerEvent::RequestFinished {
        id,
        status,
        duration_ms,
        error,
    });
    response
}

fn extract_error_message(bytes: &[u8]) -> Option<String> {
    if bytes.is_empty() {
        return None;
    }
    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(bytes) {
        if let Some(err_obj) = val.get("error") {
            if let Some(msg) = err_obj.get("message").and_then(|m| m.as_str()) {
                return Some(msg.to_string());
            }
            if let Some(msg) = err_obj.as_str() {
                return Some(msg.to_string());
            }
        }
        if let Some(msg) = val.get("message").and_then(|m| m.as_str()) {
            return Some(msg.to_string());
        }
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            let capped: String = trimmed.chars().take(1024).collect();
            return Some(capped);
        }
    }
    None
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
