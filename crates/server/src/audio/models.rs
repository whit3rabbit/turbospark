//! `GET/POST /v1/audio/models`, `DELETE /v1/audio/models/{id}`.
//!
//! Kept off `/v1/models`, which is the OpenAI chat listing (audio models are
//! also appended there, with a `capabilities` field, like image models).
//! Loading takes an installed catalog alias only; a filesystem path would let
//! any API caller probe directories.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use super::{audio_error, error, timed, unavailable, AudioModelInfo};
use crate::ServerState;

pub(crate) fn model_json(m: &AudioModelInfo) -> serde_json::Value {
    serde_json::json!({
        "id": m.id,
        "object": "model",
        "owned_by": "turbospark",
        "task": m.task.as_str(),
        "capabilities": [m.task.as_str()],
    })
}

pub(crate) async fn list_models(State(state): State<ServerState>) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let data: Vec<_> = audio.provider.models().iter().map(model_json).collect();
    Json(serde_json::json!({"object": "list", "data": data})).into_response()
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct LoadRequest {
    #[serde(alias = "model")]
    model_id: String,
}

pub(crate) async fn load_model(
    State(state): State<ServerState>,
    Json(input): Json<serde_json::Value>,
) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let handle = audio.clone();
    timed(&handle, "models_load", async move {
        let request: LoadRequest = match serde_json::from_value(input) {
            Ok(r) => r,
            Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string(), None),
        };
        let alias = request.model_id.trim().to_string();
        if alias.is_empty() {
            return error(
                StatusCode::BAD_REQUEST,
                "model_id must not be empty",
                Some("model_id"),
            );
        }
        match audio.provider.load(alias).await {
            Ok(info) => (StatusCode::CREATED, Json(model_json(&info))).into_response(),
            Err(e) => audio_error(e),
        }
    })
    .await
}

pub(crate) async fn delete_model(
    State(state): State<ServerState>,
    Path(id): Path<String>,
) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let handle = audio.clone();
    timed(&handle, "models_unload", async move {
        if !audio.provider.models().iter().any(|m| m.id == id) {
            return error(
                StatusCode::NOT_FOUND,
                format!("audio model {id:?} is not attached"),
                Some("id"),
            );
        }
        // An unload would orphan the job's result; make the client cancel or
        // wait instead.
        if audio.jobs.running_for_model(&id) > 0 {
            return error(
                StatusCode::CONFLICT,
                "the model has running jobs; cancel them or wait",
                Some("id"),
            );
        }
        match audio.provider.unload(id.clone()).await {
            Ok(()) => Json(serde_json::json!({"id": id, "deleted": true})).into_response(),
            Err(e) => audio_error(e),
        }
    })
    .await
}
