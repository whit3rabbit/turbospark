//! Prompt-to-PNG Images API. The generator lives in the embedding host.

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::ServerState;

static NEXT_SEED: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenerateRequest {
    pub model: String,
    pub prompt: String,
    pub width: u32,
    pub height: u32,
    pub seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageError {
    Busy,
    Cancelled,
    Failed(String),
}

pub trait ImageProvider: Send + Sync {
    fn models(&self) -> Vec<String>;
    fn generate(
        &self,
        request: ImageGenerateRequest,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, ImageError>> + Send + '_>>;
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ImageCreateRequest {
    model: String,
    prompt: String,
    n: Option<u8>,
    size: Option<String>,
    seed: Option<u64>,
    stream: Option<bool>,
    response_format: Option<String>,
    output_format: Option<String>,
    quality: Option<String>,
}

fn error(status: StatusCode, message: impl Into<String>, param: Option<&str>) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {
            "message": message.into(),
            "type": "invalid_request_error",
            "param": param,
        }})),
    )
        .into_response()
}

fn size(raw: Option<&str>) -> Result<(u32, u32), &'static str> {
    let raw = raw.unwrap_or("1024x1024");
    let raw = if raw == "auto" { "1024x1024" } else { raw };
    let Some((width, height)) = raw.split_once('x') else {
        return Err("size must be WIDTHxHEIGHT");
    };
    let (Ok(width), Ok(height)) = (width.parse::<u32>(), height.parse::<u32>()) else {
        return Err("size must be WIDTHxHEIGHT");
    };
    if !(512..=1024).contains(&width)
        || !(512..=1024).contains(&height)
        || width % 16 != 0
        || height % 16 != 0
        || u64::from(width) * u64::from(height) > 1024 * 1024
    {
        return Err("unsupported image size");
    }
    Ok((width, height))
}

pub async fn edits_unsupported() -> Response {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "image edits are unsupported; use /v1/images/generations",
        None,
    )
}

pub async fn generations(
    State(state): State<ServerState>,
    Json(input): Json<serde_json::Value>,
) -> Response {
    let input: ImageCreateRequest = match serde_json::from_value(input) {
        Ok(input) => input,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string(), None),
    };
    let Some(provider) = state.image_provider else {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "image generation is unavailable",
            None,
        );
    };
    let models = provider.models();
    if models.is_empty() {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "no image model is attached",
            Some("model"),
        );
    }
    if !models.contains(&input.model) {
        return error(
            StatusCode::NOT_FOUND,
            "image model is not attached",
            Some("model"),
        );
    }
    if input.prompt.trim().is_empty() {
        return error(
            StatusCode::BAD_REQUEST,
            "prompt must not be empty",
            Some("prompt"),
        );
    }
    let n = input.n.unwrap_or(1);
    if !(1..=4).contains(&n) {
        return error(
            StatusCode::BAD_REQUEST,
            "n must be between 1 and 4",
            Some("n"),
        );
    }
    let (width, height) = match size(input.size.as_deref()) {
        Ok(size) => size,
        Err(message) => return error(StatusCode::BAD_REQUEST, message, Some("size")),
    };
    if input.stream == Some(true) {
        return error(
            StatusCode::BAD_REQUEST,
            "streaming images are unsupported",
            Some("stream"),
        );
    }
    if input
        .response_format
        .as_deref()
        .is_some_and(|format| format != "b64_json")
    {
        return error(
            StatusCode::BAD_REQUEST,
            "only b64_json is supported",
            Some("response_format"),
        );
    }
    if input
        .output_format
        .as_deref()
        .is_some_and(|format| format != "png")
    {
        return error(
            StatusCode::BAD_REQUEST,
            "only PNG output is supported",
            Some("output_format"),
        );
    }
    if input.quality.is_some() {
        return error(
            StatusCode::BAD_REQUEST,
            "quality is selected by the installed model",
            Some("quality"),
        );
    }
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let seed = input
        .seed
        .unwrap_or_else(|| (created.as_nanos() as u64) ^ NEXT_SEED.fetch_add(1, Ordering::Relaxed));
    let mut data = Vec::with_capacity(n as usize);
    for index in 0..n {
        let request = ImageGenerateRequest {
            model: input.model.clone(),
            prompt: input.prompt.clone(),
            width,
            height,
            seed: seed.wrapping_add(u64::from(index)),
        };
        match provider.generate(request).await {
            Ok(png) if png.starts_with(b"\x89PNG\r\n\x1a\n") => data.push(serde_json::json!({
                "b64_json": base64::engine::general_purpose::STANDARD.encode(png)
            })),
            Ok(_) => {
                return error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "image generator returned invalid PNG data",
                    None,
                )
            }
            Err(ImageError::Busy) => {
                return error(StatusCode::TOO_MANY_REQUESTS, "image queue is full", None)
            }
            Err(ImageError::Cancelled) => {
                return error(
                    StatusCode::SERVICE_UNAVAILABLE,
                    "image generation was cancelled",
                    None,
                )
            }
            Err(ImageError::Failed(message)) => {
                return error(StatusCode::INTERNAL_SERVER_ERROR, message, None)
            }
        }
    }
    let mut response =
        Json(serde_json::json!({"created": created.as_secs(), "data": data})).into_response();
    response.headers_mut().insert(
        "x-turbospark-seed",
        HeaderValue::from_str(&seed.to_string()).expect("integer header"),
    );
    response
}
