//! Ollama-compatible routes: `/api/tags`, `/api/version`, `/api/show`,
//! `/api/chat` and `/api/generate`.
//!
//! **WHY A FOURTH WIRE FORMAT.** The three this crate already speaks are the
//! ones a developer picks deliberately. Ollama's is the one a lot of tooling
//! speaks by DEFAULT and offers no way to change -- desktop clients, editor
//! plugins, home-automation integrations -- so for those the choice is this
//! shim or nothing. `anyllm_translate` ships no Ollama types (it does ship
//! Gemini, deliberately not taken here), so the shapes below are
//! hand-rolled, the same call `completions.rs` made for the legacy endpoint.
//!
//! **THE FRAMING IS NDJSON, AND IT IS THE ONE NON-SSE STREAM IN THIS
//! CRATE.** Every other streaming path here is `Sse` -- `event:` lines,
//! `data:` lines, a keep-alive comment, and for OpenAI a `[DONE]` sentinel.
//! Ollama sends one bare JSON object per line over a plain body with no
//! sentinel at all: the LAST object carries `"done": true` and the counters,
//! and a client reads until the stream closes. So this module builds its own
//! response rather than taking a parameter on the shared SSE one, and
//! `crates/server/CLAUDE.md`'s note that SSE keep-alive is on "every SSE
//! construction in the crate" stays true by not applying here.
//!
//! **`stream` DEFAULTS TO TRUE**, unlike every OpenAI-shaped endpoint in
//! this crate, where an absent `stream` means false. That is Ollama's own
//! default and a client that omits the field is expecting a stream; getting
//! a single object instead reads as the server hanging until the whole turn
//! finishes.

use std::collections::HashSet;

use anyllm_translate::openai::{ChatCompletionRequest, ChatMessage, ChatRole};
use axum::extract::State;
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;

use crate::handler::{error_response, plan, run_full, status_for, AppState, GenError, Piece};
use crate::observe::RequestTag;
use crate::ServerState;

/// Ollama timestamps are RFC 3339. Nothing in this workspace depends on
/// `chrono`, and a client reads this field for display rather than
/// arithmetic, so it is formatted from the Unix seconds the rest of the
/// crate already uses rather than by taking a dependency for it.
fn created_at() -> String {
    let secs = crate::handler::now_unix();
    // 1970-01-01 plus `secs`, via the civil-from-days algorithm. Exact for
    // every value this can hold, and 20 lines shorter than a date crate.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let (h, m, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let mo = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if mo <= 2 { y + 1 } else { y };
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}Z")
}

fn model_entry(row: &crate::registry::ModelRow) -> serde_json::Value {
    serde_json::json!({
        "name": row.id,
        "model": row.id,
        "modified_at": created_at(),
        // **REPORTED AS ZERO RATHER THAN ESTIMATED.** Ollama clients show
        // this as a download size, and this server did not download
        // anything -- an install's on-disk bytes are not what it committed
        // (AGENTS.md Gotcha 58: those are different numbers and the whole
        // point of the two-column display in `turbospark-model recommend`).
        // A fabricated figure here would be a guess rendered as a fact.
        "size": 0,
        "digest": "",
        "details": {
            "format": "gturbo",
            "family": "turbospark",
            "families": ["turbospark"],
            "parameter_size": "",
            "quantization_level": "",
        },
        // Not Ollama's, additive for the same reason `/v1/models` carries it.
        "context_window": row.max_context,
    })
}

/// `GET /api/tags`. What an Ollama client's model list reads.
pub async fn tags(State(state): State<ServerState>) -> Response {
    Json(serde_json::json!({
        "models": state.registry.rows().iter().map(model_entry).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// `GET /api/version`. Reports THIS crate's version, not an Ollama one.
///
/// Some clients gate features on a version range and would be misled by a
/// borrowed Ollama number into expecting endpoints that do not exist here.
pub async fn version() -> Response {
    Json(serde_json::json!({ "version": env!("CARGO_PKG_VERSION") })).into_response()
}

#[derive(Deserialize)]
pub(crate) struct ShowRequest {
    #[serde(alias = "name")]
    model: Option<String>,
}

/// `POST /api/show`. Details for one model.
pub async fn show(State(state): State<ServerState>, Json(request): Json<ShowRequest>) -> Response {
    let wanted = request.model.unwrap_or_default();
    // Matched against the rows rather than through `registry.resolve`, for
    // `handler::model_detail`'s reason: a lookup asks whether this exact id
    // exists, where the single-model fallback answers a different question.
    match state.registry.rows().into_iter().find(|r| r.id == wanted) {
        Some(row) => Json(serde_json::json!({
            "details": model_entry(&row)["details"],
            "model_info": { "general.architecture": "turbospark",
                            "turbospark.context_length": row.max_context },
            "capabilities": ["completion"],
        }))
        .into_response(),
        None => error_response(StatusCode::NOT_FOUND, format!("model '{wanted}' not found")),
    }
}

#[derive(Deserialize)]
pub(crate) struct OllamaMessage {
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Deserialize)]
pub(crate) struct ChatRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    messages: Vec<OllamaMessage>,
    stream: Option<bool>,
    #[serde(default)]
    options: serde_json::Map<String, serde_json::Value>,
}

#[derive(Deserialize)]
pub(crate) struct GenerateRequest {
    #[serde(default)]
    model: String,
    #[serde(default)]
    prompt: String,
    #[serde(default)]
    system: Option<String>,
    stream: Option<bool>,
    #[serde(default)]
    options: serde_json::Map<String, serde_json::Value>,
}

/// Ollama's `options` bag onto the fields `handler::plan` and
/// `build_shaping` read.
///
/// Only the knobs that MAP are carried. `num_ctx` is silently dropped
/// rather than honoured, and that is the honest handling: the context window
/// is fixed when the model is opened and the KV cache is already allocated
/// at it (`crates/ffi/CLAUDE.md`'s note on `open.rs`), so a request cannot
/// change it and pretending otherwise would be worse than ignoring it.
fn apply_options(
    options: &serde_json::Map<String, serde_json::Value>,
    request: &mut ChatCompletionRequest,
) {
    if let Some(v) = options.get("temperature").and_then(|v| v.as_f64()) {
        request.temperature = Some(v as f32);
    }
    if let Some(v) = options.get("top_p").and_then(|v| v.as_f64()) {
        request.top_p = Some(v as f32);
    }
    if let Some(v) = options.get("num_predict").and_then(|v| v.as_i64()) {
        // Ollama's -1 means "until the context is full", which is this
        // server's own behaviour when `max_tokens` is absent.
        if v >= 0 {
            // SATURATING, not a raw `as u32`: a value past `u32::MAX` would
            // otherwise wrap to an ARBITRARY smaller (or, past a second
            // wrap, effectively unbounded-looking) budget via silent
            // truncation of the high bits, rather than clamping to the
            // largest budget this server can actually express.
            request.max_tokens = Some(u32::try_from(v).unwrap_or(u32::MAX));
        }
    }
    // `top_k`, `seed` and `repeat_penalty` have no explicit field on
    // `ChatCompletionRequest` and are read out of its `extra` flatten map by
    // `build_shaping`, under the OpenAI spellings (crate Gotcha 5).
    for (theirs, ours) in [
        ("top_k", "top_k"),
        ("seed", "seed"),
        ("repeat_penalty", "repetition_penalty"),
    ] {
        if let Some(v) = options.get(theirs) {
            request.extra.insert(ours.to_string(), v.clone());
        }
    }
}

fn role_of(role: &str) -> ChatRole {
    match role {
        "system" => ChatRole::System,
        "assistant" => ChatRole::Assistant,
        "tool" => ChatRole::Tool,
        _ => ChatRole::User,
    }
}

fn chat_message(role: ChatRole, content: String) -> ChatMessage {
    ChatMessage {
        role,
        content: Some(anyllm_translate::openai::ChatContent::Text(content)),
        name: None,
        tool_calls: None,
        tool_call_id: None,
        refusal: None,
        reasoning_content: None,
        thinking_blocks: None,
    }
}

/// A `ChatCompletionRequest` carrying only what an Ollama request can say.
///
/// Spelled out field by field rather than through `..Default::default()`
/// because the type has no `Default` -- which is the useful failure mode:
/// a field added upstream is a compile error here rather than silently
/// defaulting to something this endpoint never decided.
fn base_request(model: String, messages: Vec<ChatMessage>) -> ChatCompletionRequest {
    ChatCompletionRequest {
        model,
        messages,
        max_tokens: None,
        max_completion_tokens: None,
        temperature: None,
        top_p: None,
        stop: None,
        tools: None,
        tool_choice: None,
        stream: None,
        stream_options: None,
        presence_penalty: None,
        frequency_penalty: None,
        response_format: None,
        user: None,
        parallel_tool_calls: None,
        extra: Default::default(),
    }
}

/// One NDJSON object of a `/api/chat` stream, or its final one.
fn chat_object(model: &str, content: &str, done: Option<&Generated>) -> serde_json::Value {
    let mut object = serde_json::json!({
        "model": model,
        "created_at": created_at(),
        "message": { "role": "assistant", "content": content },
        "done": done.is_some(),
    });
    if let Some(g) = done {
        merge_counters(&mut object, g);
    }
    object
}

fn generate_object(model: &str, response: &str, done: Option<&Generated>) -> serde_json::Value {
    let mut object = serde_json::json!({
        "model": model,
        "created_at": created_at(),
        "response": response,
        "done": done.is_some(),
    });
    if let Some(g) = done {
        merge_counters(&mut object, g);
    }
    object
}

/// **THE COUNTS COME OFF `RawDecodeResult`, NEVER OFF A COUNT OF DELTAS.**
/// `swift/CLAUDE.md` Gotcha 7 is the worked example of what a delta count
/// measures instead, and these fields are exactly the ones an Ollama client
/// divides to show tokens per second.
///
/// The four `*_duration` fields (nanoseconds, Ollama's own unit) are what
/// that division actually reads: without them a client has counts and no
/// denominator, and cannot show a tok/s figure at all. `load_duration` is
/// honestly `0` -- this server holds one already-open runner for its whole
/// process lifetime, so there is no per-request load to report, unlike
/// Ollama's own server which may load a model per request.
fn merge_counters(object: &mut serde_json::Value, g: &Generated) {
    let map = object.as_object_mut().expect("object literal");
    map.insert(
        "done_reason".to_string(),
        serde_json::json!(crate::response::ollama_done_reason(
            g.decode.reason,
            !g.calls.is_empty()
        )),
    );
    map.insert(
        "prompt_eval_count".to_string(),
        serde_json::json!(g.decode.prompt_tokens),
    );
    map.insert(
        "eval_count".to_string(),
        serde_json::json!(g.decode.new_tokens),
    );
    let nanos = |secs: f64| (secs * 1_000_000_000.0).round() as u64;
    map.insert("load_duration".to_string(), serde_json::json!(0));
    map.insert(
        "prompt_eval_duration".to_string(),
        serde_json::json!(nanos(g.decode.prefill_seconds)),
    );
    map.insert(
        "eval_duration".to_string(),
        serde_json::json!(nanos(g.decode.decode_seconds)),
    );
    map.insert(
        "total_duration".to_string(),
        serde_json::json!(nanos(g.decode.prefill_seconds + g.decode.decode_seconds)),
    );
}

/// An NDJSON body: one JSON object per line, no sentinel, done when the
/// stream closes.
///
/// Wrapped in `CancelOnDrop` for the same reason every SSE construction in
/// this crate is (Gotcha 25): a disconnect during PREFILL, or during a
/// reasoning span this wire shape has nowhere to put (both piece kinds this
/// module drops without a `send` call), means nothing is ever sent before
/// the client gives up -- the `tx.send(...).is_err()` check inside `run`'s
/// own `send` closure is a real detector, but only for the case where
/// something WAS about to be sent. Without this, a client that closes the
/// connection mid-prefill on a long prompt leaves the generation running to
/// completion for nobody.
fn ndjson(
    lines: tokio::sync::mpsc::UnboundedReceiver<String>,
    cancel: crate::cancel::Cancel,
) -> Response {
    use futures::StreamExt;
    let stream = crate::cancel::CancelOnDrop::new(
        tokio_stream::wrappers::UnboundedReceiverStream::new(lines),
        cancel,
    )
    .map(|line| Ok::<_, std::convert::Infallible>(axum::body::Bytes::from(line)));
    (
        [(header::CONTENT_TYPE, "application/x-ndjson")],
        axum::body::Body::from_stream(stream),
    )
        .into_response()
}

fn one_object(object: serde_json::Value) -> Response {
    Json(object).into_response()
}

use crate::handler::Generated;

/// `POST /api/chat`.
pub async fn chat(
    State(state): State<ServerState>,
    tag: Option<axum::Extension<RequestTag>>,
    Json(request): Json<ChatRequest>,
) -> Response {
    let streaming = request.stream.unwrap_or(true);
    let model = match crate::handler::resolve_backend(
        &state,
        tag.map(|t| t.0),
        Some(request.model.as_str()),
        streaming,
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };

    let mut chat_request = base_request(
        request.model.clone(),
        request
            .messages
            .iter()
            .map(|m| chat_message(role_of(&m.role), m.content.clone()))
            .collect(),
    );
    apply_options(&request.options, &mut chat_request);

    run(state, model, chat_request, request.model, streaming, false).await
}

/// `POST /api/generate`.
///
/// Ollama's raw-prompt endpoint. It goes through the SAME chat template as
/// `/api/chat` rather than `completions.rs`'s no-template path, because
/// Ollama itself applies the model's template here -- a client sending
/// `"prompt": "why is the sky blue"` to an instruction-tuned model expects
/// an answer, and the untemplated path would babble (AGENTS.md's note on a
/// bare `--prompt`).
pub async fn generate(
    State(state): State<ServerState>,
    tag: Option<axum::Extension<RequestTag>>,
    Json(request): Json<GenerateRequest>,
) -> Response {
    let streaming = request.stream.unwrap_or(true);
    let model = match crate::handler::resolve_backend(
        &state,
        tag.map(|t| t.0),
        Some(request.model.as_str()),
        streaming,
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };

    let mut messages = Vec::new();
    if let Some(system) = request.system.filter(|s| !s.is_empty()) {
        messages.push(chat_message(ChatRole::System, system));
    }
    messages.push(chat_message(ChatRole::User, request.prompt.clone()));
    let mut chat_request = base_request(request.model.clone(), messages);
    apply_options(&request.options, &mut chat_request);

    run(state, model, chat_request, request.model, streaming, true).await
}

/// The shared body of both endpoints: plan, generate, frame.
///
/// `raw` picks which of the two object shapes to emit -- `/api/chat` nests
/// the text under `message.content` and `/api/generate` puts it flat on
/// `response`. Everything else about the two is identical, which is why
/// they share this rather than each carrying a copy of the streaming setup.
async fn run(
    state: ServerState,
    model: AppState,
    chat_request: ChatCompletionRequest,
    model_name: String,
    streaming: bool,
    raw: bool,
) -> Response {
    let planned = match plan(&model, &chat_request) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    let effort = model.default_reasoning();
    let cancel = crate::cancel::new_cancel();
    let _ = &state;

    if !streaming {
        // The FIFO gate's acquisition point for the non-streaming arm
        // (`queue.rs`'s closed set); the streaming arm below acquires
        // through `run_gated`. Acquired per arm rather than once above the
        // fork so the permit is moved into exactly the arm that runs.
        let _gate = match model.generation_queue() {
            Some(queue) => match queue.acquire(&cancel).await {
                Some(permit) => Some(permit),
                // Cancelled while queued: the same `done` object a
                // mid-generation cancel would have produced, into a
                // connection nobody is reading.
                None => {
                    let g = Generated {
                        text: String::new(),
                        reasoning: String::new(),
                        calls: Vec::new(),
                        decode: crate::handler::cancelled_before_start(),
                    };
                    let object = if raw {
                        generate_object(&model_name, &g.text, Some(&g))
                    } else {
                        chat_object(&model_name, &g.text, Some(&g))
                    };
                    return one_object(object);
                }
            },
            None => None,
        };
        let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
        let result = run_full(
            model,
            planned.prompt_ids,
            planned.config,
            planned.images,
            HashSet::new(),
            effort,
            cancel,
        )
        .await;
        guard.defuse();
        return match result {
            Ok(g) => {
                let object = if raw {
                    generate_object(&model_name, &g.text, Some(&g))
                } else {
                    chat_object(&model_name, &g.text, Some(&g))
                };
                one_object(object)
            }
            Err(GenError::Runtime(e)) => error_response(status_for(&e), e.to_string()),
            Err(GenError::Join(m)) => error_response(StatusCode::INTERNAL_SERVER_ERROR, m),
        };
    }

    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<String>();
    let cancel_for_task = cancel.clone();
    // The FIFO gate's acquisition point for the streaming arm. See
    // `handler::stream_response`'s longer note.
    let gate = model.generation_queue();
    let cancel_for_gate = cancel.clone();
    tokio::spawn(async move {
        let _ = crate::queue::run_gated(gate, &cancel_for_gate, move || {
            let send = |object: serde_json::Value| -> bool {
                // A failed send is the client having gone. Setting the flag here
                // is the fast detector; nothing else watches this body, so unlike
                // the SSE paths there is no `CancelOnDrop` behind it.
                if tx.send(format!("{object}\n")).is_err() {
                    cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
                    return false;
                }
                true
            };
            let flag = crate::cancel::as_cancel_flag(&cancel_for_task);
            let mut emit = |piece: Piece| {
                if let Piece::Text(delta) = piece {
                    let object = if raw {
                        generate_object(&model_name, &delta, None)
                    } else {
                        chat_object(&model_name, &delta, None)
                    };
                    send(object);
                }
                // Reasoning and tool pieces are DROPPED rather than folded into
                // the text: Ollama's wire shape has nowhere for either, and
                // emitting a model's scratchpad as its answer is the failure
                // AGENTS.md Gotcha 56 records. A client that wants them has
                // three other endpoints on this server that carry them.
            };
            let result = crate::handler::stream_blocking(
                &model,
                &planned.prompt_ids,
                &planned.config,
                planned.images.as_ref(),
                &HashSet::new(),
                effort,
                &flag,
                &mut emit,
            );
            // The final object closes the stream. On an error there is no
            // envelope to report through -- the body has already started and its
            // status line is long gone -- so the turn ends with `done: true` and
            // whatever reason the runtime gave, which is what a client can
            // actually read.
            match result {
                // Discarded silently, the same contract every other endpoint's
                // streaming path holds (Gotcha 25): the client that would read
                // this `done: true` object is the one already gone, and building
                // one to send into a closed channel is both pointless and (per
                // `send`, above) itself detected as a failed send.
                Ok(decode) if decode.reason == runtime::StopReason::Cancelled => (),
                Ok(decode) => {
                    let g = Generated {
                        text: String::new(),
                        reasoning: String::new(),
                        calls: Vec::new(),
                        decode,
                    };
                    let object = if raw {
                        generate_object(&model_name, "", Some(&g))
                    } else {
                        chat_object(&model_name, "", Some(&g))
                    };
                    send(object);
                }
                Err(e) => {
                    send(serde_json::json!({
                        "model": model_name,
                        "created_at": created_at(),
                        "done": true,
                        "done_reason": "error",
                        "error": e.to_string(),
                    }));
                }
            }
        })
        .await;
    });

    ndjson(rx, cancel)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ChatCompletionRequest {
        base_request("m".to_string(), Vec::new())
    }

    /// F23: `num_predict` past `u32::MAX` used to wrap via a raw `as u32`
    /// cast -- silently truncating the high bits into an ARBITRARY smaller
    /// budget -- rather than clamping to the largest one this server can
    /// actually express.
    #[test]
    fn num_predict_past_u32_max_saturates_rather_than_wraps() {
        let mut request = base();
        let options = serde_json::json!({"num_predict": (u32::MAX as i64) + 1000})
            .as_object()
            .unwrap()
            .clone();
        apply_options(&options, &mut request);
        assert_eq!(request.max_tokens, Some(u32::MAX));
    }

    #[test]
    fn num_predict_within_range_passes_through_unchanged() {
        let mut request = base();
        let options = serde_json::json!({"num_predict": 128})
            .as_object()
            .unwrap()
            .clone();
        apply_options(&options, &mut request);
        assert_eq!(request.max_tokens, Some(128));
    }

    /// Ollama's own "until the context is full" sentinel must still be
    /// left alone by the saturating cast.
    #[test]
    fn num_predict_negative_one_is_still_left_absent() {
        let mut request = base();
        let options = serde_json::json!({"num_predict": -1})
            .as_object()
            .unwrap()
            .clone();
        apply_options(&options, &mut request);
        assert_eq!(request.max_tokens, None);
    }
}
