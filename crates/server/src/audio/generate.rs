//! `POST /v1/audio/generate`: text prompt to music.
//!
//! Backed by MiniMax Music 3, which takes a caption and lyrics (use
//! `[instrumental]` for none) and runs for minutes or longer, so `async=true`
//! (202 plus a job id) is the practical mode. Fields the model has no knob
//! for (`audio_start`, `guidance_scale`, ...) are refused by name through
//! `deny_unknown_fields`, never ignored.

use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use super::jobs::{JobFailure, JobOutput, SubmitError};
use super::{
    audio_error, audio_error_parts, error, timed, unavailable, wav_bytes, AudioError, AudioTask,
    GenerateRequest, GeneratedAudio,
};
use crate::ServerState;

const MAX_AUDIO_LENGTH_SECONDS: f64 = 360.0;
const MAX_STEPS: usize = 30;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct GenerateInput {
    model: String,
    prompt: String,
    lyrics: Option<String>,
    audio_length: Option<f64>,
    num_inference_steps: Option<usize>,
    seed: Option<u64>,
    response_format: Option<String>,
    #[serde(rename = "async")]
    asynchronous: Option<bool>,
}

fn encode(audio: GeneratedAudio) -> Result<JobOutput, String> {
    Ok(JobOutput {
        content_type: "audio/wav",
        body: wav_bytes(audio.sample_rate, audio.channels, audio.samples)?,
    })
}

pub(crate) async fn generate(
    State(state): State<ServerState>,
    Json(input): Json<serde_json::Value>,
) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let handle = audio.clone();
    timed(&handle, "generate", async move {
        let input: GenerateInput = match serde_json::from_value(input) {
            Ok(i) => i,
            Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string(), None),
        };
        if input.prompt.trim().is_empty() {
            return error(
                StatusCode::BAD_REQUEST,
                "prompt must not be empty",
                Some("prompt"),
            );
        }
        if let Some(len) = input.audio_length {
            if !len.is_finite() || len <= 0.0 || len > MAX_AUDIO_LENGTH_SECONDS {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("audio_length must be in (0, {MAX_AUDIO_LENGTH_SECONDS}] seconds"),
                    Some("audio_length"),
                );
            }
        }
        if let Some(steps) = input.num_inference_steps {
            if !(1..=MAX_STEPS).contains(&steps) {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("num_inference_steps must be between 1 and {MAX_STEPS}"),
                    Some("num_inference_steps"),
                );
            }
        }
        if input.response_format.as_deref().is_some_and(|f| f != "wav") {
            return error(
                StatusCode::BAD_REQUEST,
                "response_format must be wav",
                Some("response_format"),
            );
        }
        if let Err(r) = audio.model_for(&input.model, AudioTask::Music) {
            return r;
        }
        let request = GenerateRequest {
            model: input.model.clone(),
            caption: input.prompt,
            // The runner rejects empty lyrics; "no vocals" is spelled this way.
            lyrics: input
                .lyrics
                .filter(|l| !l.trim().is_empty())
                .unwrap_or_else(|| "[instrumental]".to_string()),
            duration_seconds: input.audio_length,
            steps: input.num_inference_steps,
            seed: input.seed,
        };
        let provider = audio.provider.clone();

        if input.asynchronous.unwrap_or(false) {
            let work = Box::pin(async move {
                let generated = provider.generate(request).await.map_err(|e| {
                    let (status, message) = audio_error_parts(&e);
                    JobFailure { status, message }
                })?;
                encode(generated).map_err(|message| JobFailure {
                    status: StatusCode::INTERNAL_SERVER_ERROR,
                    message,
                })
            });
            return match audio.jobs.submit("generation", input.model, work) {
                Ok(id) => (
                    StatusCode::ACCEPTED,
                    Json(serde_json::json!({
                        "job_id": id,
                        "id": id,
                        "object": "audio.job",
                        "status": "running",
                        "poll_url": format!("/v1/audio/jobs/{id}"),
                    })),
                )
                    .into_response(),
                Err(SubmitError::Full) => audio_error(AudioError::Busy),
            };
        }

        match provider.generate(request).await {
            Ok(generated) => match encode(generated) {
                Ok(out) => {
                    let mut response = (StatusCode::OK, out.body).into_response();
                    response.headers_mut().insert(
                        header::CONTENT_TYPE,
                        HeaderValue::from_static(out.content_type),
                    );
                    response
                }
                Err(m) => audio_error(AudioError::Failed(m)),
            },
            Err(e) => audio_error(e),
        }
    })
    .await
}
