//! OpenAI and Ollama-compatible embedding endpoints.
//!
//! - `POST /v1/embeddings`: OpenAI-compatible embeddings
//! - `POST /api/embeddings`: Ollama legacy embeddings
//! - `POST /api/embed`: Ollama batch embeddings

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};

use crate::handler::error_response;
use crate::registry::Resolution;
use crate::ServerState;

/// Input can be a single string or an array of strings.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Single(String),
    Multiple(Vec<String>),
}

impl EmbeddingInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbeddingInput::Single(s) => vec![s],
            EmbeddingInput::Multiple(v) => v,
        }
    }
}

/// OpenAI embedding request shape.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct EmbeddingRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub encoding_format: Option<String>,
    #[serde(default)]
    pub dimensions: Option<usize>,
    #[serde(default)]
    pub user: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

/// `POST /v1/embeddings`.
pub async fn embeddings(
    State(state): State<ServerState>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let model_str = req.model.as_deref();
    let model = match state.registry.resolve_embedding(model_str) {
        Resolution::Model(m) => m,
        Resolution::Unknown {
            requested,
            available,
        } => {
            let msg = format!(
                "model '{requested}' is not attached; available: {}",
                available.join(", ")
            );
            return error_response(StatusCode::NOT_FOUND, msg);
        }
        Resolution::Empty => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "server has no attached model".to_string(),
            );
        }
    };

    let texts = req.input.into_vec();
    if texts.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty".to_string(),
        );
    }

    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let raw_embeddings = match model.encode(&text_refs) {
        Ok(embs) => embs,
        Err(err) => return error_response(StatusCode::BAD_REQUEST, err),
    };

    let mut data = Vec::with_capacity(raw_embeddings.len());
    let mut total_tokens = 0;
    for (index, mut emb) in raw_embeddings.into_iter().enumerate() {
        // Approximate token count from character length
        total_tokens += texts[index].len().div_ceil(4);

        if let Some(dim) = req.dimensions {
            if dim < emb.len() {
                emb.truncate(dim);
                let norm: f32 = emb.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-12);
                for x in &mut emb {
                    *x /= norm;
                }
            }
        }

        data.push(EmbeddingData {
            object: "embedding",
            index,
            embedding: emb,
        });
    }

    let response = EmbeddingResponse {
        object: "list",
        data,
        model: model.model_id().to_string(),
        usage: EmbeddingUsage {
            prompt_tokens: total_tokens,
            total_tokens,
        },
    };

    Json(response).into_response()
}

/// Ollama legacy embedding request shape (`/api/embeddings`).
#[derive(Debug, Deserialize)]
pub struct OllamaEmbeddingRequest {
    pub model: Option<String>,
    pub prompt: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbeddingResponse {
    pub embedding: Vec<f32>,
}

/// `POST /api/embeddings`.
pub async fn ollama_embeddings(
    State(state): State<ServerState>,
    Json(req): Json<OllamaEmbeddingRequest>,
) -> Response {
    let model = match state.registry.resolve_embedding(req.model.as_deref()) {
        Resolution::Model(m) => m,
        Resolution::Unknown {
            requested,
            available,
        } => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!(
                    "model '{requested}' not found; available: {}",
                    available.join(", ")
                ),
            );
        }
        Resolution::Empty => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "no model attached".to_string(),
            );
        }
    };

    match model.encode(&[&req.prompt]) {
        Ok(mut embs) => {
            let embedding = embs.pop().unwrap_or_default();
            Json(OllamaEmbeddingResponse { embedding }).into_response()
        }
        Err(err) => error_response(StatusCode::BAD_REQUEST, err),
    }
}

/// Ollama batch embedding request shape (`/api/embed`).
#[derive(Debug, Deserialize)]
pub struct OllamaEmbedRequest {
    pub model: Option<String>,
    pub input: EmbeddingInput,
}

#[derive(Debug, Serialize)]
pub struct OllamaEmbedResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f32>>,
}

/// `POST /api/embed`.
pub async fn ollama_embed(
    State(state): State<ServerState>,
    Json(req): Json<OllamaEmbedRequest>,
) -> Response {
    let model = match state.registry.resolve_embedding(req.model.as_deref()) {
        Resolution::Model(m) => m,
        Resolution::Unknown {
            requested,
            available,
        } => {
            return error_response(
                StatusCode::NOT_FOUND,
                format!(
                    "model '{requested}' not found; available: {}",
                    available.join(", ")
                ),
            );
        }
        Resolution::Empty => {
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "no model attached".to_string(),
            );
        }
    };

    let texts = req.input.into_vec();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();

    match model.encode(&text_refs) {
        Ok(embeddings) => Json(OllamaEmbedResponse {
            model: model.model_id().to_string(),
            embeddings,
        })
        .into_response(),
        Err(err) => error_response(StatusCode::BAD_REQUEST, err),
    }
}
