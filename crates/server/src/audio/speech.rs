//! `POST /v1/audio/speech`: text in, audio out.
//!
//! Kokoro is the only TTS family, English only, with one voice pack, so the
//! voice list is the provider's to enforce and the route only bounds shape.
//! `wav` is the default (OpenAI defaults to mp3, which this server cannot
//! encode, and refuses by name). `stream=true` is `pcm` only: a chunked body
//! with one chunk per synthesized segment, so the first audio leaves before
//! the last segment is generated.

use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use serde::Deserialize;

use super::{
    audio_error, error, pcm_s16le, timed, unavailable, wav_bytes, AudioError, AudioTask,
    SpeechRequest,
};
use crate::ServerState;

const MAX_INPUT_CHARS: usize = 4096;
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpeechInput {
    model: String,
    #[serde(alias = "text")]
    input: String,
    voice: Option<String>,
    speed: Option<f32>,
    response_format: Option<String>,
    stream: Option<bool>,
}

pub(crate) async fn speech(
    State(state): State<ServerState>,
    axum::Json(input): axum::Json<serde_json::Value>,
) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let handle = audio.clone();
    timed(&handle, "speech", async move {
        let input: SpeechInput = match serde_json::from_value(input) {
            Ok(i) => i,
            Err(e) => return error(StatusCode::BAD_REQUEST, e.to_string(), None),
        };
        if input.input.trim().is_empty() {
            return error(
                StatusCode::BAD_REQUEST,
                "input must not be empty",
                Some("input"),
            );
        }
        if input.input.chars().count() > MAX_INPUT_CHARS {
            return error(
                StatusCode::BAD_REQUEST,
                format!("input is limited to {MAX_INPUT_CHARS} characters"),
                Some("input"),
            );
        }
        let speed = input.speed.unwrap_or(1.0);
        if !speed.is_finite() || !(MIN_SPEED..=MAX_SPEED).contains(&speed) {
            return error(
                StatusCode::BAD_REQUEST,
                format!("speed must be between {MIN_SPEED} and {MAX_SPEED}"),
                Some("speed"),
            );
        }
        let format = input.response_format.as_deref().unwrap_or("wav");
        let pcm = match format {
            "wav" => false,
            "pcm" => true,
            other => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("response_format {other:?} is not supported; use wav or pcm"),
                    Some("response_format"),
                )
            }
        };
        let stream = input.stream.unwrap_or(false);
        if stream && !pcm {
            return error(
                StatusCode::BAD_REQUEST,
                "stream=true requires response_format=pcm",
                Some("stream"),
            );
        }
        if let Err(r) = audio.model_for(&input.model, AudioTask::TextToSpeech) {
            return r;
        }
        let request = SpeechRequest {
            model: input.model,
            text: input.input,
            voice: input.voice.unwrap_or_else(|| "af_heart".to_string()),
            speed,
        };
        let mut synthesized = match audio.provider.speak(request).await {
            Ok(s) => s,
            Err(e) => return audio_error(e),
        };
        let rate = synthesized.sample_rate;

        if stream {
            // Pull the first chunk before committing to a 200, so a request
            // that fails at once (unknown voice, busy) still gets its status.
            let first = match synthesized.chunks.recv().await {
                Some(Ok(c)) => Some(c),
                Some(Err(e)) => return audio_error(e),
                None => None,
            };
            let rest = synthesized.chunks;
            let body = futures::stream::unfold((first, rest), |(first, mut rest)| async move {
                let next = match first {
                    Some(c) => Some(Ok(c)),
                    None => rest.recv().await,
                };
                // A mid-stream failure can only end the body: the status is
                // already sent. The truncated chunked body is the signal.
                match next {
                    Some(Ok(samples)) => Some((
                        Ok::<_, std::io::Error>(axum::body::Bytes::from(pcm_s16le(&samples))),
                        (None, rest),
                    )),
                    Some(Err(e)) => {
                        Some((Err(std::io::Error::other(format!("{e:?}"))), (None, rest)))
                    }
                    None => None,
                }
            });
            let mut response = Response::new(Body::from_stream(body));
            let headers = response.headers_mut();
            headers.insert(header::CONTENT_TYPE, HeaderValue::from_static("audio/pcm"));
            headers.insert(
                "x-audio-sample-rate",
                HeaderValue::from_str(&rate.to_string()).expect("digits are a valid header"),
            );
            return response;
        }

        let mut samples = Vec::new();
        while let Some(chunk) = synthesized.chunks.recv().await {
            match chunk {
                Ok(c) => samples.extend_from_slice(&c),
                Err(e) => return audio_error(e),
            }
        }
        if samples.is_empty() {
            return audio_error(AudioError::Failed("the model produced no audio".into()));
        }
        let (content_type, body) = if pcm {
            ("audio/pcm", pcm_s16le(&samples))
        } else {
            match wav_bytes(rate, 1, samples) {
                Ok(b) => ("audio/wav", b),
                Err(m) => return audio_error(AudioError::Failed(m)),
            }
        };
        let mut response = (StatusCode::OK, body).into_response();
        let headers = response.headers_mut();
        headers.insert(header::CONTENT_TYPE, HeaderValue::from_static(content_type));
        headers.insert(
            "x-audio-sample-rate",
            HeaderValue::from_str(&rate.to_string()).expect("digits are a valid header"),
        );
        response
    })
    .await
}
