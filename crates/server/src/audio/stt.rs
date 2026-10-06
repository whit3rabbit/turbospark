//! `POST /v1/audio/transcriptions`: WAV in, transcript out.
//!
//! Accepts OpenAI-style `multipart/form-data` (`file`, `model`, ...) or a raw
//! `audio/wav` body with the options in the query string. Only WAV decodes
//! (the audio crate has no other container), so anything else is a 415.
//! Options this stack cannot honour are refused with a 400 naming the field
//! rather than silently ignored.

use std::convert::Infallible;

use axum::body::Body;
use axum::extract::{FromRequest, Multipart, Query, Request, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use tokio_stream::wrappers::ReceiverStream;
use turbospark_audio::conversion::{to_mono_resampled, MonoResampleStrategy};
use turbospark_audio::error::AudioError as WavError;
use turbospark_audio::resample::SincHannOptions;

use super::jobs::{JobFailure, JobOutput, SubmitError};
use super::{
    audio_error, audio_error_parts, error, timed, unavailable, AudioState, AudioTask,
    TranscribeRequest, Transcription, MAX_AUDIO_UPLOAD_BYTES, MAX_TRANSCRIBE_SECONDS,
};
use crate::ServerState;

/// Every STT runner consumes this rate.
pub(crate) const STT_SAMPLE_RATE: u32 = 16_000;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Format {
    Json,
    VerboseJson,
    Text,
    Srt,
    Vtt,
}

impl Format {
    pub(crate) fn parse(raw: &str) -> Option<Self> {
        Some(match raw {
            "json" => Self::Json,
            "verbose_json" => Self::VerboseJson,
            "text" => Self::Text,
            "srt" => Self::Srt,
            "vtt" => Self::Vtt,
            _ => return None,
        })
    }
}

pub(crate) struct Params {
    pub(crate) model: Option<String>,
    pub(crate) language: Option<String>,
    pub(crate) format: Format,
    pub(crate) stream: bool,
    pub(crate) asynchronous: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            model: None,
            language: None,
            format: Format::Json,
            stream: false,
            asynchronous: false,
        }
    }
}

fn parse_bool(name: &str, value: &str) -> Result<bool, Response> {
    match value.trim().to_ascii_lowercase().as_str() {
        "true" | "1" => Ok(true),
        "false" | "0" | "" => Ok(false),
        _ => Err(error(
            StatusCode::BAD_REQUEST,
            format!("{name} must be true or false"),
            Some(name),
        )),
    }
}

pub(crate) fn parse_language(value: &str) -> Result<Option<String>, Response> {
    let v = value.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("auto") {
        return Ok(None);
    }
    if v.len() > 16
        || !v
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(error(
            StatusCode::BAD_REQUEST,
            "language must be a short tag such as \"en\" or \"auto\"",
            Some("language"),
        ));
    }
    Ok(Some(v.to_ascii_lowercase()))
}

impl Params {
    fn set(&mut self, name: &str, value: &str) -> Result<(), Response> {
        let unsupported = |what: &str| {
            error(
                StatusCode::BAD_REQUEST,
                format!("{what} is not supported by this server"),
                Some(name),
            )
        };
        match name {
            "model" => self.model = Some(value.trim().to_string()),
            "language" => self.language = parse_language(value)?,
            "response_format" => {
                self.format = Format::parse(value.trim()).ok_or_else(|| {
                    error(
                        StatusCode::BAD_REQUEST,
                        "response_format must be json, verbose_json, text, srt or vtt",
                        Some("response_format"),
                    )
                })?
            }
            "stream" => self.stream = parse_bool(name, value)?,
            "async" => self.asynchronous = parse_bool(name, value)?,
            // Runners take no prompt or sampling controls. An empty prompt and
            // temperature 0 are what clients send by default, so they pass.
            "prompt" if value.trim().is_empty() => {}
            "prompt" => return Err(unsupported("a transcription prompt")),
            "temperature" => match value.trim().parse::<f64>() {
                Ok(0.0) => {}
                _ => return Err(unsupported("a non-zero temperature")),
            },
            "timestamp_granularities[]" | "timestamp_granularities" => {
                if value.trim() != "segment" {
                    return Err(unsupported("word-level timestamps"));
                }
            }
            other => return Err(unsupported(&format!("the field {other:?}"))),
        }
        Ok(())
    }
}

fn multipart_error(e: axum::extract::multipart::MultipartError) -> Response {
    let status = e.status();
    let message = if status == StatusCode::PAYLOAD_TOO_LARGE {
        format!(
            "request body too large; audio uploads are limited to {} MiB",
            MAX_AUDIO_UPLOAD_BYTES / (1024 * 1024)
        )
    } else {
        e.body_text()
    };
    error(status, message, None)
}

/// The WAV bytes and options of a request, from either encoding.
async fn read_input(state: &ServerState, request: Request) -> Result<(Params, Vec<u8>), Response> {
    let content_type = request
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let mut params = Params::default();

    if content_type.starts_with("multipart/form-data") {
        let mut multipart = Multipart::from_request(request, state)
            .await
            .map_err(|e| error(StatusCode::BAD_REQUEST, e.body_text(), None))?;
        let mut file: Option<Vec<u8>> = None;
        while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
            let name = field.name().unwrap_or("").to_string();
            if name == "file" {
                if file.is_some() {
                    return Err(error(
                        StatusCode::BAD_REQUEST,
                        "send exactly one file part",
                        Some("file"),
                    ));
                }
                file = Some(field.bytes().await.map_err(multipart_error)?.to_vec());
            } else {
                let value = field.text().await.map_err(multipart_error)?;
                params.set(&name, &value)?;
            }
        }
        let file = file.ok_or_else(|| {
            error(
                StatusCode::BAD_REQUEST,
                "missing the file part",
                Some("file"),
            )
        })?;
        return Ok((params, file));
    }

    let raw_ok = content_type.is_empty()
        || [
            "audio/wav",
            "audio/x-wav",
            "audio/wave",
            "audio/vnd.wave",
            "application/octet-stream",
        ]
        .iter()
        .any(|t| content_type.starts_with(t));
    if !raw_ok {
        return Err(error(
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "send multipart/form-data, or a raw audio/wav body; only WAV is supported",
            None,
        ));
    }
    let Query(pairs) = Query::<Vec<(String, String)>>::try_from_uri(request.uri())
        .map_err(|e| error(StatusCode::BAD_REQUEST, e.body_text(), None))?;
    for (name, value) in pairs {
        params.set(&name, &value)?;
    }
    let body = axum::body::to_bytes(request.into_body(), MAX_AUDIO_UPLOAD_BYTES)
        .await
        .map_err(|_| {
            error(
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "request body too large; audio uploads are limited to {} MiB",
                    MAX_AUDIO_UPLOAD_BYTES / (1024 * 1024)
                ),
                None,
            )
        })?;
    Ok((params, body.to_vec()))
}

/// WAV bytes to the mono 16 kHz f32 every runner takes. CPU-bound, so callers
/// run it on the blocking pool.
pub(crate) fn decode_wav_16k(bytes: &[u8]) -> Result<Vec<f32>, (StatusCode, String)> {
    let wave = turbospark_audio::wav::read_wav_f32_bytes(bytes).map_err(|e| {
        let status = match e {
            WavError::NotWav { .. } | WavError::UnsupportedWavFormat { .. } => {
                StatusCode::UNSUPPORTED_MEDIA_TYPE
            }
            _ => StatusCode::BAD_REQUEST,
        };
        (status, format!("{e}; only WAV (PCM or float) is supported"))
    })?;
    if wave.frame_count() == 0 {
        return Err((
            StatusCode::BAD_REQUEST,
            "the audio contains no samples".into(),
        ));
    }
    if wave.duration_seconds() > MAX_TRANSCRIBE_SECONDS {
        return Err((
            StatusCode::BAD_REQUEST,
            format!(
                "audio is {:.0} s; the limit is {:.0} s",
                wave.duration_seconds(),
                MAX_TRANSCRIBE_SECONDS
            ),
        ));
    }
    // Sinc-Hann for the usual 44.1/48 kHz downsample: linear interpolation
    // aliases and costs recognition accuracy.
    to_mono_resampled(
        &wave,
        STT_SAMPLE_RATE,
        &MonoResampleStrategy::SincHann(SincHannOptions::default()),
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))
}

pub(crate) async fn decode_blocking(bytes: Vec<u8>) -> Result<Vec<f32>, Response> {
    tokio::task::spawn_blocking(move || decode_wav_16k(&bytes))
        .await
        .map_err(|_| {
            error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "audio decode failed",
                None,
            )
        })?
        .map_err(|(status, message)| error(status, message, Some("file")))
}

fn timestamp(seconds: f64, comma: bool) -> String {
    let total_ms = (seconds.max(0.0) * 1000.0).round() as u64;
    let (h, m, s, ms) = (
        total_ms / 3_600_000,
        total_ms / 60_000 % 60,
        total_ms / 1000 % 60,
        total_ms % 1000,
    );
    let sep = if comma { ',' } else { '.' };
    format!("{h:02}:{m:02}:{s:02}{sep}{ms:03}")
}

pub(crate) fn render(t: &Transcription, format: Format, duration: f64) -> JobOutput {
    const TEXT: &str = "text/plain; charset=utf-8";
    match format {
        Format::Json => JobOutput {
            content_type: "application/json",
            body: serde_json::json!({"text": t.text}).to_string().into_bytes(),
        },
        Format::VerboseJson => {
            let segments: Vec<_> = t
                .segments
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    serde_json::json!({
                        "id": i,
                        "start": s.start_seconds,
                        "end": s.end_seconds,
                        "text": s.text,
                    })
                })
                .collect();
            JobOutput {
                content_type: "application/json",
                body: serde_json::json!({
                    "task": "transcribe",
                    "language": t.language,
                    "duration": duration,
                    "text": t.text,
                    "segments": segments,
                })
                .to_string()
                .into_bytes(),
            }
        }
        Format::Text => JobOutput {
            content_type: TEXT,
            body: format!("{}\n", t.text).into_bytes(),
        },
        Format::Srt => {
            let mut out = String::new();
            for (i, s) in t.segments.iter().enumerate() {
                out.push_str(&format!(
                    "{}\n{} --> {}\n{}\n\n",
                    i + 1,
                    timestamp(s.start_seconds, true),
                    timestamp(s.end_seconds, true),
                    s.text.trim()
                ));
            }
            JobOutput {
                content_type: TEXT,
                body: out.into_bytes(),
            }
        }
        Format::Vtt => {
            let mut out = String::from("WEBVTT\n\n");
            for s in &t.segments {
                out.push_str(&format!(
                    "{} --> {}\n{}\n\n",
                    timestamp(s.start_seconds, false),
                    timestamp(s.end_seconds, false),
                    s.text.trim()
                ));
            }
            JobOutput {
                content_type: "text/vtt; charset=utf-8",
                body: out.into_bytes(),
            }
        }
    }
}

fn output_response(out: JobOutput) -> Response {
    let mut response = (StatusCode::OK, out.body).into_response();
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(out.content_type),
    );
    response
}

pub(crate) async fn translations_unsupported() -> Response {
    error(
        StatusCode::NOT_IMPLEMENTED,
        "audio translation is unsupported; use /v1/audio/transcriptions",
        None,
    )
}

pub(crate) async fn transcriptions(State(state): State<ServerState>, request: Request) -> Response {
    let Some(audio) = state.audio.clone() else {
        return unavailable();
    };
    let metrics_handle = audio.clone();
    timed(
        &metrics_handle,
        "transcriptions",
        run(audio, state, request),
    )
    .await
}

async fn run(audio: AudioState, state: ServerState, request: Request) -> Response {
    let (params, bytes) = match read_input(&state, request).await {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(model) = params.model.clone().filter(|m| !m.is_empty()) else {
        return error(StatusCode::BAD_REQUEST, "model is required", Some("model"));
    };
    if params.stream && params.asynchronous {
        return error(
            StatusCode::BAD_REQUEST,
            "stream and async cannot be combined",
            Some("stream"),
        );
    }
    if let Err(r) = audio.model_for(&model, AudioTask::SpeechToText) {
        return r;
    }
    let samples = match decode_blocking(bytes).await {
        Ok(s) => s,
        Err(r) => return r,
    };
    let duration = samples.len() as f64 / f64::from(STT_SAMPLE_RATE);
    let request = TranscribeRequest {
        model: model.clone(),
        samples,
        language: params.language,
    };
    let format = params.format;
    let provider = audio.provider.clone();

    if params.asynchronous {
        let work = Box::pin(async move {
            match provider.transcribe(request).await {
                Ok(t) => Ok(render(&t, format, duration)),
                Err(e) => {
                    let (status, message) = audio_error_parts(&e);
                    Err(JobFailure { status, message })
                }
            }
        });
        return match audio.jobs.submit("transcription", model, work) {
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
            Err(SubmitError::Full) => audio_error(super::AudioError::Busy),
        };
    }

    if params.stream {
        return ndjson(provider, request, duration);
    }

    match provider.transcribe(request).await {
        Ok(t) => output_response(render(&t, format, duration)),
        Err(e) => audio_error(e),
    }
}

/// Segment-granular NDJSON. The runner returns a whole transcript, so lines
/// for segments arrive together when it finishes; `start` goes out at once so
/// a proxy sees bytes during a long decode. A client that disconnects while
/// the request waits drops the provider future, which cancels queued work.
fn ndjson(
    provider: std::sync::Arc<dyn super::AudioProvider>,
    request: TranscribeRequest,
    duration: f64,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<String, Infallible>>(16);
    let model = request.model.clone();
    tokio::spawn(async move {
        let line = |v: serde_json::Value| Ok::<_, Infallible>(format!("{v}\n"));
        if tx
            .send(line(serde_json::json!({"type": "start", "model": model})))
            .await
            .is_err()
        {
            return;
        }
        let result = tokio::select! {
            r = provider.transcribe(request) => r,
            _ = tx.closed() => return,
        };
        match result {
            Ok(t) => {
                for (i, s) in t.segments.iter().enumerate() {
                    let _ = tx
                        .send(line(serde_json::json!({
                            "type": "segment", "id": i,
                            "start": s.start_seconds, "end": s.end_seconds, "text": s.text,
                        })))
                        .await;
                }
                let _ = tx
                    .send(line(serde_json::json!({
                        "type": "complete", "text": t.text,
                        "language": t.language, "duration": duration,
                    })))
                    .await;
            }
            Err(e) => {
                let (status, message) = audio_error_parts(&e);
                let _ = tx
                    .send(line(serde_json::json!({
                        "type": "error",
                        "error": {"status": status.as_u16(), "message": message},
                    })))
                    .await;
            }
        }
    });
    let mut response = Response::new(Body::from_stream(ReceiverStream::new(rx)));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}
