//! OpenAI- and Anthropic-compatible generation server on loopback:
//! request/response envelopes, SSE streaming framing, and the axum router,
//! wired to `mrefrust-runtime`'s raw-completion loop. Ported from the intent
//! of `Sources/MferenceServer` (an OpenAI-compatible
//! `/v1/chat/completions` endpoint), plus an Anthropic `/v1/messages`
//! endpoint that renders the same generation through `anyllm_translate` so
//! Anthropic-native clients need no proxy in between. Two backends implement
//! [`ChatModel`]: [`ScriptedChatModel`] (portable, fixed logit sequence, what
//! the tests drive) and [`RealChatModel`] (macOS only, a real
//! `RealForwardRunner` forward pass against a `.gturbo` install). Model
//! dialect auto-selection is the tokenizer's job here, not the server's.

mod handler;
mod messages;
mod model;
#[cfg(target_os = "macos")]
mod real_model;
mod response;

pub use handler::AppState;
pub use model::{ChatModel, ScriptedChatModel};
#[cfg(target_os = "macos")]
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

/// Builds the router: `POST /v1/chat/completions`, `POST /v1/messages`, and
/// `GET /v1/models`, all bound to `state`.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .route("/v1/messages", post(messages::messages))
        .route("/v1/models", get(handler::models))
        .with_state(state)
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
