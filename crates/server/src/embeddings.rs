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

use crate::handler::{error_response, resolve_embedding_backend};
use crate::model::ChatModel;
use crate::observe::RequestTag;
use crate::ServerState;

/// `encoding_format`'s two OpenAI-documented spellings. Anything else is
/// refused rather than silently treated as `float`: a caller who sent a
/// typo'd or future value has no way to learn their request was not
/// honoured otherwise.
enum EncodingFormat {
    Float,
    Base64,
}

fn parse_encoding_format(value: Option<&str>) -> Result<EncodingFormat, String> {
    match value {
        None | Some("float") => Ok(EncodingFormat::Float),
        Some("base64") => Ok(EncodingFormat::Base64),
        Some(other) => Err(format!(
            "encoding_format must be float or base64, not {other:?}"
        )),
    }
}

const BASE64_ALPHABET: &[u8; 64] =
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding, the mirror of `vision::base64_decode`.
/// Hand-rolled for the same reason that one is: it is a dozen lines, and
/// this is the only encoder this workspace needs.
fn base64_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(BASE64_ALPHABET[(n >> 18 & 0x3F) as usize] as char);
        out.push(BASE64_ALPHABET[(n >> 12 & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 {
            BASE64_ALPHABET[(n >> 6 & 0x3F) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            BASE64_ALPHABET[(n & 0x3F) as usize] as char
        } else {
            '='
        });
    }
    out
}

/// A vector as OpenAI's own base64 shape: the raw little-endian `f32` bytes,
/// standard-base64-encoded -- what the stock Python SDK sends
/// `encoding_format` as by default and base64-decodes on the way back.
fn embedding_value(v: &[f32], format: &EncodingFormat) -> serde_json::Value {
    match format {
        EncodingFormat::Float => serde_json::json!(v),
        EncodingFormat::Base64 => {
            let mut bytes = Vec::with_capacity(v.len() * 4);
            for x in v {
                bytes.extend_from_slice(&x.to_le_bytes());
            }
            serde_json::json!(base64_encode(&bytes))
        }
    }
}

/// Above this many inputs (OpenAI's own documented ceiling) or this many
/// total bytes of text, a request is refused before it ever reaches the
/// encoder. `spawn_blocking` (below) keeps a real forward pass off the
/// async runtime, but nothing here streams partial results or cancels a
/// batch mid-flight, so an unbounded one would still tie up a blocking
/// thread for an unreasonable span on nothing but request size.
const MAX_EMBEDDING_INPUTS: usize = 2048;
const MAX_EMBEDDING_BYTES: usize = 1 << 20;

fn check_embedding_input_size(texts: &[String]) -> Result<(), String> {
    if texts.len() > MAX_EMBEDDING_INPUTS {
        return Err(format!(
            "too many inputs: {} exceeds the {MAX_EMBEDDING_INPUTS}-item limit",
            texts.len()
        ));
    }
    let total_bytes: usize = texts.iter().map(String::len).sum();
    if total_bytes > MAX_EMBEDDING_BYTES {
        return Err(format!(
            "input too large: {total_bytes} bytes exceeds the {MAX_EMBEDDING_BYTES}-byte limit"
        ));
    }
    Ok(())
}

/// Runs `model.encode` on a blocking thread, matching every generation
/// path's own `spawn_blocking` (`handler::exec::run_full`,
/// `completions::run_full`): a real embedding model's forward pass is
/// synchronous CPU/GPU work, not I/O, and running it inline here parked a
/// tokio worker for its whole duration with no cancellation and no
/// `CancelGuard` -- enough concurrent embedding requests could starve
/// `/health` the same way an un-`spawn_blocking`ed generation would
/// (Gotcha 22). `texts` is threaded through and back so the caller keeps
/// ownership after the `.await` for its own token-count accounting.
async fn encode_blocking(
    model: std::sync::Arc<dyn ChatModel>,
    texts: Vec<String>,
) -> (Vec<String>, Result<Vec<Vec<f32>>, String>) {
    match tokio::task::spawn_blocking(move || {
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let result = model.encode(&text_refs);
        (texts, result)
    })
    .await
    {
        Ok(pair) => pair,
        Err(e) => (Vec::new(), Err(format!("embedding task panicked: {e}"))),
    }
}

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
    /// An array of `f32` for `encoding_format: "float"` (the default), or a
    /// base64 string for `"base64"` -- one field, two shapes, matching
    /// OpenAI's own wire contract.
    pub embedding: serde_json::Value,
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
    tag: Option<axum::Extension<RequestTag>>,
    Json(req): Json<EmbeddingRequest>,
) -> Response {
    let model = match resolve_embedding_backend(&state, tag.map(|t| t.0), req.model.as_deref()) {
        Ok(m) => m,
        Err(response) => return response,
    };
    let format = match parse_encoding_format(req.encoding_format.as_deref()) {
        Ok(f) => f,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };

    let texts = req.input.into_vec();
    if texts.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input must not be empty".to_string(),
        );
    }
    if let Err(e) = check_embedding_input_size(&texts) {
        return error_response(StatusCode::BAD_REQUEST, e);
    }
    if req.dimensions == Some(0) {
        return error_response(
            StatusCode::BAD_REQUEST,
            "dimensions must be greater than 0".to_string(),
        );
    }

    let (texts, encoded) = encode_blocking(model.clone(), texts).await;
    let raw_embeddings = match encoded {
        Ok(embs) => embs,
        Err(err) => return error_response(StatusCode::BAD_REQUEST, err),
    };
    // The backend's own contract (`ChatModel::encode`'s doc) is one vector
    // per input text, in order. A backend that returned a different count
    // would otherwise panic on `texts[index]` below -- a bug in a backend
    // this crate does not control (any FFI host implementing the trait),
    // reported as a 500 rather than crashing the request.
    if raw_embeddings.len() != texts.len() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "embedding backend returned {} vectors for {} inputs",
                raw_embeddings.len(),
                texts.len()
            ),
        );
    }
    if let Some(dim) = req.dimensions {
        if let Some(width) = raw_embeddings.first().map(Vec::len) {
            if dim > width {
                return error_response(
                    StatusCode::BAD_REQUEST,
                    format!("dimensions {dim} exceeds this model's own width of {width}"),
                );
            }
        }
    }

    let mut data = Vec::with_capacity(raw_embeddings.len());
    let mut total_tokens = 0;
    for (index, mut emb) in raw_embeddings.into_iter().enumerate() {
        // The real tokenizer this server already holds, not a character-count
        // estimate: `usage.prompt_tokens` is what a caller bills against.
        total_tokens += model.tokenizer().encode(&texts[index], false).len();

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
            embedding: embedding_value(&emb, &format),
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
    tag: Option<axum::Extension<RequestTag>>,
    Json(req): Json<OllamaEmbeddingRequest>,
) -> Response {
    let model = match resolve_embedding_backend(&state, tag.map(|t| t.0), req.model.as_deref()) {
        Ok(m) => m,
        Err(response) => return response,
    };

    if let Err(e) = check_embedding_input_size(std::slice::from_ref(&req.prompt)) {
        return error_response(StatusCode::BAD_REQUEST, e);
    }
    let (_, encoded) = encode_blocking(model, vec![req.prompt]).await;
    match encoded {
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
    tag: Option<axum::Extension<RequestTag>>,
    Json(req): Json<OllamaEmbedRequest>,
) -> Response {
    let model = match resolve_embedding_backend(&state, tag.map(|t| t.0), req.model.as_deref()) {
        Ok(m) => m,
        Err(response) => return response,
    };

    let texts = req.input.into_vec();
    if let Err(e) = check_embedding_input_size(&texts) {
        return error_response(StatusCode::BAD_REQUEST, e);
    }

    let (texts, encoded) = encode_blocking(model.clone(), texts).await;
    let embeddings = match encoded {
        Ok(embs) => embs,
        Err(err) => return error_response(StatusCode::BAD_REQUEST, err),
    };
    if embeddings.len() != texts.len() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!(
                "embedding backend returned {} vectors for {} inputs",
                embeddings.len(),
                texts.len()
            ),
        );
    }
    Json(OllamaEmbedResponse {
        model: model.model_id().to_string(),
        embeddings,
    })
    .into_response()
}
