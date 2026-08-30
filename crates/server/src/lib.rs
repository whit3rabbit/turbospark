//! OpenAI- and Anthropic-compatible generation server on loopback:
//! request/response envelopes, SSE streaming framing, and the axum router,
//! wired to `turbospark-runtime`'s raw-completion loop. Ported from the intent
//! of `Sources/MferenceServer` (an OpenAI-compatible
//! `/v1/chat/completions` endpoint), plus an Anthropic `/v1/messages`
//! endpoint that renders the same generation through `anyllm_translate` so
//! Anthropic-native clients need no proxy in between. Two backends implement
//! [`ChatModel`]: [`ScriptedChatModel`] (portable, fixed logit sequence, what
//! the tests drive) and [`RealChatModel`] (macOS only, a real
//! `RealForwardRunner` forward pass against a `.gturbo` install). Model
//! dialect auto-selection is the tokenizer's job here, not the server's.

mod auth;
mod cancel;
mod completions;
mod guardrails;
mod handler;
mod messages;
mod model;
#[cfg(target_os = "macos")]
mod real_model;
mod response;
mod responses;
pub mod vision;

/// The header both routes report a degraded request on.
///
/// `anyllm_translate`'s own spelling, kept for `/v1/messages` where it
/// originated, and reused on `/v1/chat/completions` since ROADMAP M-V8 --
/// one client-visible mechanism rather than two names for the same signal.
pub(crate) const DEGRADATION_HEADER: &str = "x-anyllm-degradation";

/// Tool-call guardrail configuration: rescue parsing, argument validation,
/// and the retry budget.
pub use guardrails::GuardrailConfig;
/// Shared application state for request handling.
pub use handler::AppState;
/// Trait and canned test backend for chat generation.
pub use model::{ChatModel, ScriptedChatModel};
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

use axum::routing::{get, post};
use axum::Router;

/// Options [`build_router_with_options`] takes beyond the shared
/// [`AppState`]. A struct rather than a bare `Option<String>` parameter so a
/// future option (a CORS policy, say) has somewhere to land without another
/// signature change; `Default` gives `build_router` its zero-option case for
/// free.
#[derive(Default, Clone)]
pub struct RouterOptions {
    /// Require this key on every request except `GET /health`. `None`
    /// (the default) leaves the router exactly as unauthenticated as it was
    /// before this option existed.
    pub api_key: Option<String>,
}

/// [`build_router`] with no options: no auth. Every existing caller of this
/// function -- the integration tests in `tests/`, and the FFI's future
/// in-process server -- keeps the exact behavior it had before
/// [`RouterOptions`] existed.
///
/// See [`build_router_with_options`] for the route list.
pub fn build_router(state: AppState) -> Router {
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
pub fn build_router_with_options(state: AppState, options: RouterOptions) -> Router {
    let protected = Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/responses", post(responses::responses))
        .route("/v1/messages", post(messages::messages))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route("/v1/models", get(handler::models))
        .route("/v1/models/:model", get(handler::model_detail));
    let protected = match options.api_key {
        Some(key) => protected.layer(axum::middleware::from_fn_with_state(
            auth::ApiKey(key.into()),
            auth::require_api_key,
        )),
        None => protected,
    };

    Router::new()
        .route("/health", get(handler::health))
        .merge(protected)
        .with_state(state)
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
