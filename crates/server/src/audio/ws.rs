//! `GET /v1/audio/transcriptions/realtime?model=ID[&encoding=s16le|f32le][&language=en]`
//!
//! WebSocket, utterance-granular. The client streams 16 kHz mono PCM in
//! binary frames and sends `{"type":"commit"}` when an utterance ends; the
//! server decodes everything buffered since the last commit and answers
//! `{"type":"complete","segment_id":N,"text":..,"segments":[..]}`. There is
//! no server-side endpointing and no partial text: the runners decode a whole
//! clip at a time. Closing the socket discards uncommitted audio.
//!
//! Auth rides the ordinary Bearer / `x-api-key` headers on the upgrade
//! request. Browsers cannot set them on a WebSocket, and a key in the query
//! string would end up in logs, so this route is for non-browser clients.

use std::time::Duration;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::Response;
use serde::Deserialize;

use super::stt::{parse_language, STT_SAMPLE_RATE};
use super::{
    audio_error_parts, error, timed, unavailable, AudioState, AudioTask, TranscribeRequest,
};
use crate::ServerState;

/// Audio buffered between commits. Bounds memory per socket.
const MAX_BUFFER_SECONDS: usize = 300;
/// Largest single binary frame accepted.
const MAX_FRAME_BYTES: usize = 1024 * 1024;
/// A socket that sends nothing for this long is closed.
const IDLE_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RealtimeQuery {
    model: String,
    encoding: Option<String>,
    language: Option<String>,
}

#[derive(Clone, Copy)]
enum Encoding {
    S16le,
    F32le,
}

pub(crate) async fn realtime(
    State(state): State<ServerState>,
    query: Result<Query<RealtimeQuery>, axum::extract::rejection::QueryRejection>,
    ws: WebSocketUpgrade,
) -> Response {
    let Some(audio) = state.audio else {
        return unavailable();
    };
    let handle = audio.clone();
    timed(&handle, "realtime", async move {
        let Query(query) = match query {
            Ok(q) => q,
            Err(e) => return error(StatusCode::BAD_REQUEST, e.body_text(), None),
        };
        let encoding = match query.encoding.as_deref().unwrap_or("s16le") {
            "s16le" => Encoding::S16le,
            "f32le" => Encoding::F32le,
            other => {
                return error(
                    StatusCode::BAD_REQUEST,
                    format!("encoding {other:?} is not supported; use s16le or f32le"),
                    Some("encoding"),
                )
            }
        };
        let language = match query.language.as_deref().map(parse_language).transpose() {
            Ok(l) => l.flatten(),
            Err(r) => return r,
        };
        if let Err(r) = audio.model_for(&query.model, AudioTask::SpeechToText) {
            return r;
        }
        let Ok(permit) = audio.realtime_slots.clone().try_acquire_owned() else {
            return super::audio_error(super::AudioError::Busy);
        };
        let model = query.model;
        ws.max_message_size(MAX_FRAME_BYTES)
            .on_upgrade(move |socket| async move {
                let _permit = permit;
                session(socket, audio, model, language, encoding).await;
            })
    })
    .await
}

fn decode_frame(bytes: &[u8], encoding: Encoding) -> Option<Vec<f32>> {
    match encoding {
        Encoding::S16le => {
            let chunks = bytes.chunks_exact(2);
            if !chunks.remainder().is_empty() {
                return None;
            }
            Some(
                chunks
                    .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32768.0)
                    .collect(),
            )
        }
        Encoding::F32le => {
            let chunks = bytes.chunks_exact(4);
            if !chunks.remainder().is_empty() {
                return None;
            }
            let samples: Vec<f32> = chunks
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // NaN/inf would reach the runner's own non-finite check as a 500.
            samples.iter().all(|s| s.is_finite()).then_some(samples)
        }
    }
}

async fn send_json(socket: &mut WebSocket, value: serde_json::Value) -> bool {
    socket.send(Message::Text(value.to_string())).await.is_ok()
}

async fn session(
    mut socket: WebSocket,
    audio: AudioState,
    model: String,
    language: Option<String>,
    encoding: Encoding,
) {
    let max_samples = MAX_BUFFER_SECONDS * STT_SAMPLE_RATE as usize;
    let mut buffer: Vec<f32> = Vec::new();
    let mut segment_id: u64 = 0;

    loop {
        let message = match tokio::time::timeout(IDLE_TIMEOUT, socket.recv()).await {
            Err(_) => {
                let _ = send_json(
                    &mut socket,
                    serde_json::json!({"type": "error", "message": "idle timeout"}),
                )
                .await;
                break;
            }
            Ok(None) | Ok(Some(Err(_))) => break,
            Ok(Some(Ok(m))) => m,
        };
        match message {
            Message::Binary(bytes) => {
                let Some(samples) = decode_frame(&bytes, encoding) else {
                    let _ = send_json(
                        &mut socket,
                        serde_json::json!({
                            "type": "error",
                            "message": "frame is not whole, finite samples of the chosen encoding",
                        }),
                    )
                    .await;
                    break;
                };
                if buffer.len() + samples.len() > max_samples {
                    let _ = send_json(
                        &mut socket,
                        serde_json::json!({
                            "type": "error",
                            "message": format!("more than {MAX_BUFFER_SECONDS} s buffered without a commit"),
                        }),
                    )
                    .await;
                    break;
                }
                buffer.extend_from_slice(&samples);
            }
            Message::Text(text) => {
                let kind = serde_json::from_str::<serde_json::Value>(&text)
                    .ok()
                    .and_then(|v| v.get("type").and_then(|t| t.as_str().map(str::to_owned)));
                match kind.as_deref() {
                    Some("commit") => {
                        if buffer.is_empty() {
                            if !send_json(
                                &mut socket,
                                serde_json::json!({"type": "error", "message": "nothing to commit"}),
                            )
                            .await
                            {
                                break;
                            }
                            continue;
                        }
                        let samples = std::mem::take(&mut buffer);
                        let request = TranscribeRequest {
                            model: model.clone(),
                            samples,
                            language: language.clone(),
                        };
                        let reply = match audio.provider.transcribe(request).await {
                            Ok(t) => {
                                let segments: Vec<_> = t
                                    .segments
                                    .iter()
                                    .map(|s| {
                                        serde_json::json!({
                                            "start": s.start_seconds,
                                            "end": s.end_seconds,
                                            "text": s.text,
                                        })
                                    })
                                    .collect();
                                serde_json::json!({
                                    "type": "complete",
                                    "segment_id": segment_id,
                                    "text": t.text,
                                    "language": t.language,
                                    "segments": segments,
                                })
                            }
                            Err(e) => {
                                let (status, message) = audio_error_parts(&e);
                                serde_json::json!({
                                    "type": "error",
                                    "status": status.as_u16(),
                                    "message": message,
                                })
                            }
                        };
                        segment_id += 1;
                        if !send_json(&mut socket, reply).await {
                            break;
                        }
                    }
                    Some("reset") => buffer.clear(),
                    _ => {
                        if !send_json(
                            &mut socket,
                            serde_json::json!({
                                "type": "error",
                                "message": "send binary PCM frames and {\"type\":\"commit\"}",
                            }),
                        )
                        .await
                        {
                            break;
                        }
                    }
                }
            }
            Message::Close(_) => break,
            // axum answers pings itself.
            Message::Ping(_) | Message::Pong(_) => {}
        }
    }
}
