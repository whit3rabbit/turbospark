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

mod completions;
mod guardrails;
mod handler;
mod messages;
mod model;
#[cfg(target_os = "macos")]
mod real_model;
mod response;
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

/// Builds the Axum router bound to `state` with the following routes:
///
/// - `GET /health`: liveness/readiness probe, no generation lock taken
/// - `POST /v1/chat/completions`: OpenAI-compatible chat completion endpoint
/// - `POST /v1/completions`: OpenAI's legacy raw-prompt completion endpoint
/// - `POST /v1/messages`: Anthropic-compatible messages endpoint
/// - `POST /v1/messages/count_tokens`: Anthropic's count-only endpoint (no generation)
/// - `GET /v1/models`: OpenAI-compatible list of available models
/// - `GET /v1/models/:model`: OpenAI-compatible model detail endpoint
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handler::health))
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/v1/completions", post(completions::completions))
        .route("/v1/messages", post(messages::messages))
        .route("/v1/messages/count_tokens", post(messages::count_tokens))
        .route("/v1/models", get(handler::models))
        .route("/v1/models/:model", get(handler::model_detail))
        .with_state(state)
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
