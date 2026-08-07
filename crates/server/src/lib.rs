//! OpenAI-compatible Chat Completions server on loopback: request/response
//! envelopes, SSE streaming framing, and the axum router, wired to
//! `mrefrust-runtime`'s raw-completion loop. Ported from the intent of
//! `Sources/MferenceServer` (an OpenAI-compatible `/v1/chat/completions`
//! endpoint). Two backends implement [`ChatModel`]: [`ScriptedChatModel`]
//! (portable, fixed logit sequence, what the tests drive) and
//! [`RealChatModel`] (macOS only, a real `RealForwardRunner` forward pass
//! against a `.gturbo` install). Model dialect auto-selection is the
//! tokenizer's job here, not the server's.

mod handler;
mod model;
#[cfg(target_os = "macos")]
mod real_model;
mod request;
mod response;

pub use handler::AppState;
pub use model::{ChatModel, ScriptedChatModel};
#[cfg(target_os = "macos")]
pub use real_model::RealChatModel;
pub use request::{ChatCompletionRequest, ChatMessage};
pub use response::{ChatCompletionChunk, ChatCompletionResponse, Choice, Usage};

use axum::routing::post;
use axum::Router;

/// Builds the router: `POST /v1/chat/completions`, bound to `state`.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(handler::chat_completions))
        .with_state(state)
}

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
