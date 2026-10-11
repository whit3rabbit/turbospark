//! `POST /v1/audio/speech`: text in, audio out.
//!
//! The provider enforces each family's voices and options; the route bounds
//! shape and applies the rules that depend only on the family. Voice cloning
//! (`reference_audio`, a base64 WAV) is refused for a family that cannot
//! clone, and MegaTTS3 requires both the reference and `consent: true`, the
//! caller's statement that it has the right to use the voice.
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
    audio_error, clones_voices, error, pcm_s16le, timed, unavailable, wav_bytes, AudioError,
    AudioTask, SpeechRequest,
};
use crate::ServerState;

const MAX_INPUT_CHARS: usize = 4096;
const MIN_SPEED: f32 = 0.5;
const MAX_SPEED: f32 = 2.0;
/// Base64 of a 30 s reference: 48 kHz stereo f32 WAV is about 15.4 MB.
const MAX_REFERENCE_BASE64: usize = 24 << 20;

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
    language: Option<String>,
    reference_audio: Option<String>,
    reference_text: Option<String>,
    instruction: Option<String>,
    consent: Option<bool>,
    steps: Option<usize>,
    seed: Option<u64>,
    p_w: Option<f32>,
    t_w: Option<f32>,
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
        let info = match audio.model_for(&input.model, AudioTask::TextToSpeech) {
            Ok(info) => info,
            Err(r) => return r,
        };
        let family = info.family.as_deref();
        if let Some(reference) = &input.reference_audio {
            if family.is_some_and(|f| !clones_voices(f)) {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!(
                        "model {:?} does not clone voices; reference_audio needs a voice-cloning model",
                        input.model
                    ),
                    Some("reference_audio"),
                );
            }
            if reference.len() > MAX_REFERENCE_BASE64 {
                return error(
                    StatusCode::BAD_REQUEST,
                    "reference_audio is limited to a 30 second WAV",
                    Some("reference_audio"),
                );
            }
        }
        if family == Some("megatts3") {
            if input.reference_audio.is_none() {
                return error(
                    StatusCode::BAD_REQUEST,
                    "voice cloning needs reference_audio: a base64 WAV of 1 to 30 seconds",
                    Some("reference_audio"),
                );
            }
            if input.consent != Some(true) {
                return error(
                    StatusCode::BAD_REQUEST,
                    "voice cloning needs consent: true, stating you have the right to use the reference voice",
                    Some("consent"),
                );
            }
        }
        let request = SpeechRequest {
            model: input.model,
            text: input.input,
            voice: input.voice.unwrap_or_else(|| "af_heart".to_string()),
            speed,
            language: input.language,
            reference_audio: input.reference_audio,
            reference_text: input.reference_text,
            instruction: input.instruction,
            consent: input.consent.unwrap_or(false),
            steps: input.steps,
            seed: input.seed,
            p_w: input.p_w,
            t_w: input.t_w,
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
