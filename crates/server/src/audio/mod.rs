//! Audio API: speech-to-text, text-to-speech and music generation.
//!
//! Shaped like `images.rs`: this module owns the routes, request validation
//! and wire formats; an [`AudioProvider`] supplied by the host owns inference.
//! The standalone binary supplies `audio_real::RealAudioProvider`; the FFI
//! host supplies none, and then NO audio route is registered at all (so the
//! app's route surface is unchanged and an unconfigured server answers 404).
//!
//! Runners behind the provider are not incremental. "Streaming" here is
//! therefore segment-granular (NDJSON) or utterance-granular (WebSocket
//! commit), never live partial text; the docs say so rather than imitate it.

// Every handler here returns `Result<_, Response>` for early exits, the same
// shape `handler/mod.rs` allows per function. Boxing the response would only
// add an allocation to an error path.
#![allow(clippy::result_large_err)]

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;

mod generate;
mod jobs;
mod metrics;
mod models;
mod speech;
mod stt;
mod ws;

pub(crate) use generate::generate;
pub(crate) use jobs::{delete_job, get_job, get_job_result, JobStore};
pub(crate) use metrics::{metrics_endpoint, AudioMetrics};
pub(crate) use models::{delete_model, list_models, load_model};
pub(crate) use speech::speech;
pub(crate) use stt::{transcriptions, translations_unsupported};
pub(crate) use ws::realtime;

/// Upper bound on an uploaded audio body. A WAV decodes to f32, so the
/// transient in-memory cost is up to 4x this (8-bit input), which is why the
/// cap sits well below the 1 GiB the WAV reader itself tolerates.
pub(crate) const MAX_AUDIO_UPLOAD_BYTES: usize = 128 * 1024 * 1024;

/// Concurrent realtime WebSockets.
const MAX_REALTIME_SOCKETS: usize = 8;

/// Longest decoded clip a transcription request may carry. Qwen3-ASR refuses
/// past 20 minutes on its own; this is the server-wide ceiling.
pub(crate) const MAX_TRANSCRIBE_SECONDS: f64 = 30.0 * 60.0;

/// Routes registered when (and only when) an audio provider is attached.
/// `tests` and the OpenAPI contract test compare against this list.
pub const AUDIO_ROUTE_PATHS: &[&str] = &[
    "/v1/audio/transcriptions",
    "/v1/audio/translations",
    "/v1/audio/transcriptions/realtime",
    "/v1/audio/speech",
    "/v1/audio/generate",
    "/v1/audio/jobs/{id}",
    "/v1/audio/jobs/{id}/result",
    "/v1/audio/models",
    "/v1/audio/models/{id}",
    "/v1/metrics",
];

type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// What a loaded model does. One model has exactly one task.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioTask {
    SpeechToText,
    TextToSpeech,
    Music,
}

impl AudioTask {
    pub fn as_str(self) -> &'static str {
        match self {
            AudioTask::SpeechToText => "speech_to_text",
            AudioTask::TextToSpeech => "text_to_speech",
            AudioTask::Music => "music_generation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioModelInfo {
    pub id: String,
    pub task: AudioTask,
}

/// Mono 16 kHz f32 PCM, already decoded and resampled by the route.
#[derive(Debug, Clone)]
pub struct TranscribeRequest {
    pub model: String,
    pub samples: Vec<f32>,
    /// `None` is auto-detect.
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TranscribedSegment {
    pub start_seconds: f64,
    pub end_seconds: f64,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Transcription {
    pub text: String,
    pub language: Option<String>,
    pub segments: Vec<TranscribedSegment>,
}

#[derive(Debug, Clone)]
pub struct SpeechRequest {
    pub model: String,
    pub text: String,
    pub voice: String,
    pub speed: f32,
}

/// Synthesized audio arriving one segment at a time. Dropping the receiver
/// tells the provider to stop synthesizing.
pub struct SpeechStream {
    pub sample_rate: u32,
    pub chunks: tokio::sync::mpsc::Receiver<Result<Vec<f32>, AudioError>>,
}

#[derive(Debug, Clone)]
pub struct GenerateRequest {
    pub model: String,
    pub caption: String,
    pub lyrics: String,
    pub duration_seconds: Option<f64>,
    pub steps: Option<usize>,
    pub seed: Option<u64>,
}

/// Interleaved f32 samples.
#[derive(Debug, Clone)]
pub struct GeneratedAudio {
    pub sample_rate: u32,
    pub channels: u16,
    pub samples: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioError {
    /// The provider's queue is full; retry later.
    Busy,
    Cancelled,
    NotFound(String),
    /// The request is well formed but names something this model cannot do.
    Invalid(String),
    Unsupported(String),
    Failed(String),
}

/// Host-side audio inference. Implementations must be cheap to call from many
/// requests at once: serialize model work internally (a worker thread per
/// model is the standalone provider's choice) and answer [`AudioError::Busy`]
/// rather than queueing without bound.
pub trait AudioProvider: Send + Sync {
    fn models(&self) -> Vec<AudioModelInfo>;

    fn transcribe(
        &self,
        request: TranscribeRequest,
    ) -> BoxFuture<'_, Result<Transcription, AudioError>>;

    fn speak(&self, request: SpeechRequest) -> BoxFuture<'_, Result<SpeechStream, AudioError>>;

    fn generate(
        &self,
        request: GenerateRequest,
    ) -> BoxFuture<'_, Result<GeneratedAudio, AudioError>>;

    /// Attach an installed model by catalog alias. Never a filesystem path:
    /// this is reachable over HTTP.
    fn load(&self, _alias: String) -> BoxFuture<'_, Result<AudioModelInfo, AudioError>> {
        Box::pin(async {
            Err(AudioError::Unsupported(
                "model loading is not available".into(),
            ))
        })
    }

    fn unload(&self, _id: String) -> BoxFuture<'_, Result<(), AudioError>> {
        Box::pin(async {
            Err(AudioError::Unsupported(
                "model unloading is not available".into(),
            ))
        })
    }
}

/// Everything the audio routes share. Present on `ServerState` only when a
/// provider is attached.
#[derive(Clone)]
pub(crate) struct AudioState {
    pub(crate) provider: Arc<dyn AudioProvider>,
    pub(crate) jobs: Arc<JobStore>,
    pub(crate) metrics: Arc<AudioMetrics>,
    /// Open realtime sockets are capped; each holds up to 300 s of PCM.
    pub(crate) realtime_slots: Arc<tokio::sync::Semaphore>,
}

impl AudioState {
    pub(crate) fn new(provider: Arc<dyn AudioProvider>) -> Self {
        Self {
            provider,
            jobs: Arc::new(JobStore::default()),
            metrics: Arc::new(AudioMetrics::default()),
            realtime_slots: Arc::new(tokio::sync::Semaphore::new(MAX_REALTIME_SOCKETS)),
        }
    }

    /// The model named `id`, which must exist and do `task`.
    pub(crate) fn model_for(&self, id: &str, task: AudioTask) -> Result<AudioModelInfo, Response> {
        let models = self.provider.models();
        match models.into_iter().find(|m| m.id == id) {
            None => Err(error(
                StatusCode::NOT_FOUND,
                format!("audio model {id:?} is not attached"),
                Some("model"),
            )),
            Some(m) if m.task != task => Err(error(
                StatusCode::BAD_REQUEST,
                format!(
                    "model {id:?} is a {} model, not {}",
                    m.task.as_str(),
                    task.as_str()
                ),
                Some("model"),
            )),
            Some(m) => Ok(m),
        }
    }
}

/// Same envelope as the image routes: `{"error":{message,type,param}}`.
pub(crate) fn error(
    status: StatusCode,
    message: impl Into<String>,
    param: Option<&str>,
) -> Response {
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

pub(crate) fn audio_error(e: AudioError) -> Response {
    let (status, message) = audio_error_parts(&e);
    let param = matches!(e, AudioError::NotFound(_)).then_some("model");
    let mut response = error(status, message, param);
    if matches!(e, AudioError::Busy) {
        response
            .headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("2"));
    }
    response
}

/// Status and message for a failure that has no `Response` yet (jobs, sockets).
pub(crate) fn audio_error_parts(e: &AudioError) -> (StatusCode, String) {
    match e {
        AudioError::Busy => (
            StatusCode::TOO_MANY_REQUESTS,
            "the audio queue is full; retry shortly".to_string(),
        ),
        AudioError::Cancelled => (
            StatusCode::SERVICE_UNAVAILABLE,
            "request cancelled".to_string(),
        ),
        AudioError::NotFound(m) => (StatusCode::NOT_FOUND, m.clone()),
        AudioError::Invalid(m) => (StatusCode::BAD_REQUEST, m.clone()),
        AudioError::Unsupported(m) => (StatusCode::NOT_IMPLEMENTED, m.clone()),
        AudioError::Failed(m) => (StatusCode::INTERNAL_SERVER_ERROR, m.clone()),
    }
}

/// 503 for a route reached through a state that somehow has no provider.
/// Unreachable through the router (routes are conditional); kept so a handler
/// never panics if that invariant is broken.
pub(crate) fn unavailable() -> Response {
    error(
        StatusCode::SERVICE_UNAVAILABLE,
        "audio is unavailable",
        None,
    )
}

pub(crate) fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 16-bit PCM WAV bytes for interleaved f32 samples.
pub(crate) fn wav_bytes(
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
) -> Result<Vec<u8>, String> {
    let wave = turbospark_audio::waveform::Waveform::new(sample_rate, channels, samples)
        .map_err(|e| e.to_string())?;
    Ok(turbospark_audio::wav::write_wav_i16(&wave))
}

/// Signed 16-bit little-endian PCM for f32 samples in [-1, 1].
pub(crate) fn pcm_s16le(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for s in samples {
        let v = (s.clamp(-1.0, 1.0) * 32767.0).round() as i16;
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

/// Runs `fut` and records its outcome in the audio metrics.
pub(crate) async fn timed<F>(state: &AudioState, endpoint: &'static str, fut: F) -> Response
where
    F: Future<Output = Response>,
{
    let _inflight = state.metrics.begin();
    let started = std::time::Instant::now();
    let response = fut.await;
    state
        .metrics
        .record(endpoint, response.status().as_u16(), started.elapsed());
    response
}
