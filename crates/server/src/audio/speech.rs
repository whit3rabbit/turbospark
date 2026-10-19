//! `POST /v1/audio/speech`: text in, audio out.
//!
//! The provider enforces each family's voices and options; the route bounds
//! shape and applies the rules that depend only on the family. Voice cloning
//! (`reference_audio`, a base64 WAV) is refused for a family that cannot
//! clone, and MegaTTS3 requires both the reference and `consent: true`, the
//! caller's statement that it has the right to use the voice. OmniVoice and
//! Drama 3 clone only when a reference is sent; Drama 3 also needs the
//! reference's transcript (`reference_text`) and takes the four sampler
//! fields, which every other family refuses.
//! `wav` is the default (OpenAI defaults to mp3, which this server cannot
//! encode, and refuses by name). `stream=true` is `pcm` only: a chunked body
//! with one chunk per synthesized segment, so the first audio leaves before
//! the last segment is generated.
//! A Drama 3 request without `seed` draws a fresh one, and every Drama 3
//! response names the seed it used in `x-turbospark-seed`, as image
//! generation does.

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

/// The seed a Drama 3 request runs with: the caller's, or a fresh one. The
/// runtime takes every seed literally, so a default of 0 gave one take for
/// every request, and some seeds go silent after the first sentence.
fn drama3_seed(requested: Option<u64>) -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Two requests in the same clock tick still draw different seeds.
    static DRAWN: AtomicU64 = AtomicU64::new(0);
    requested.unwrap_or_else(|| {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        nanos ^ DRAWN.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed)
    })
}

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
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    max_frames: Option<usize>,
}

/// The Drama 3 sampler fields a request set, by name.
fn first_sampler_field(input: &SpeechInput) -> Option<&'static str> {
    if input.temperature.is_some() {
        Some("temperature")
    } else if input.top_p.is_some() {
        Some("top_p")
    } else if input.top_k.is_some() {
        Some("top_k")
    } else if input.max_frames.is_some() {
        Some("max_frames")
    } else {
        None
    }
}

fn bad_param(message: &str, param: &'static str) -> Response {
    error(StatusCode::BAD_REQUEST, message, Some(param))
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
        // OmniVoice speaks without a reference too; a reference makes the
        // request a clone, which needs consent.
        if family == Some("omnivoice") && input.reference_audio.is_some() && input.consent != Some(true)
        {
            return error(
                StatusCode::BAD_REQUEST,
                "voice cloning needs consent: true, stating you have the right to use the reference voice",
                Some("consent"),
            );
        }
        // Sampler settings are Drama 3's. Another family would ignore them, and
        // a setting that does nothing looks like it worked.
        if family.is_some_and(|f| f != "fish_drama3") {
            if let Some(param) = first_sampler_field(&input) {
                return bad_param(
                    &format!("{param} is a Drama 3 sampler setting; this model does not take it"),
                    param,
                );
            }
        }
        // Drama 3 speaks without a reference too; a reference makes the
        // request a clone, which needs consent and the reference's transcript.
        // The engine has no steps, speed, instruction or guidance weights, so
        // each is refused by name rather than ignored.
        if family == Some("fish_drama3") {
            if input.steps.is_some() {
                return bad_param("Drama 3 has no steps setting; omit steps", "steps");
            }
            if input.speed.is_some_and(|s| s != 1.0) {
                return bad_param("Drama 3 has no speed control; leave speed at 1.0", "speed");
            }
            if input.instruction.as_deref().is_some_and(|s| !s.trim().is_empty()) {
                return bad_param(
                    "Drama 3 takes delivery cues inline, such as [whispers], not an instruction",
                    "instruction",
                );
            }
            if input.p_w.is_some() {
                return bad_param("p_w is a MegaTTS3 setting; Drama 3 does not take it", "p_w");
            }
            if input.t_w.is_some() {
                return bad_param("t_w is a MegaTTS3 setting; Drama 3 does not take it", "t_w");
            }
            if input.reference_audio.is_some() {
                if input.consent != Some(true) {
                    return bad_param(
                        "voice cloning needs consent: true, stating you have the right to use the reference voice",
                        "consent",
                    );
                }
                if input
                    .reference_text
                    .as_deref()
                    .is_none_or(|t| t.trim().is_empty())
                {
                    return bad_param(
                        "Drama 3 voice cloning needs reference_text, the reference audio's transcript",
                        "reference_text",
                    );
                }
            }
        }
        let seed = if family == Some("fish_drama3") {
            Some(drama3_seed(input.seed))
        } else {
            input.seed
        };
        let seed_header = seed
            .filter(|_| family == Some("fish_drama3"))
            .map(|s| HeaderValue::from_str(&s.to_string()).expect("digits are a valid header"));
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
            seed,
            p_w: input.p_w,
            t_w: input.t_w,
            temperature: input.temperature,
            top_p: input.top_p,
            top_k: input.top_k,
            max_frames: input.max_frames,
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
            if let Some(seed) = seed_header {
                headers.insert("x-turbospark-seed", seed);
            }
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
        if let Some(seed) = seed_header {
            headers.insert("x-turbospark-seed", seed);
        }
        response
    })
    .await
}
