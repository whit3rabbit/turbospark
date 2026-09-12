//! The legacy OpenAI `POST /v1/completions` endpoint: a raw prompt with NO
//! chat template applied, generating from wherever the prompt's own text
//! ends.
//!
//! `anyllm_translate` ships no types for this endpoint -- it predates Chat
//! Completions and every modern client speaks that one instead -- so the
//! request shape here is hand-rolled, reusing only what already matches
//! (`anyllm_translate::openai::Stop`, `handler::plan`'s shared shaping and
//! stop-string parsing).
//!
//! Scope: no chat template, no tools, no reasoning, no
//! `StructuredAssistantDecoder`, no guardrails. `suffix` (fill-in-the-middle)
//! and `n > 1` are refused with a 400; `logprobs`, `best_of`, and `echo` are
//! accepted on the wire but reported on `x-anyllm-degradation` rather than
//! honoured, the same contract `/v1/chat/completions` uses for its own
//! unsupported fields.

use std::time::Duration;

use anyllm_translate::openai::Stop;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::{GenerationConfig, RawDecodeProgress, RawDecodeResult};
use serde::Deserialize;

use crate::handler::{
    build_shaping, error_response, now_unix, request_id, status_for, stop_strings, AppState,
};
use crate::response::finish_reason;

const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

/// A bare string, or a single-element array of one -- the two shapes real
/// clients send. A multi-prompt batch is refused rather than silently
/// answering only the first: this server has one runner per process, so a
/// batched request already looks like N sequential ones to a caller and
/// answering as if it were one would drop N-1 answers without saying so.
#[derive(Deserialize)]
#[serde(untagged)]
enum PromptInput {
    Single(String),
    Multiple(Vec<String>),
}

fn single_prompt(input: PromptInput) -> Result<String, String> {
    match input {
        PromptInput::Single(s) => Ok(s),
        PromptInput::Multiple(v) if v.len() == 1 => Ok(v.into_iter().next().expect("len checked")),
        PromptInput::Multiple(v) => Err(format!(
            "this server accepts exactly one prompt per request, got {}",
            v.len()
        )),
    }
}

#[derive(Deserialize)]
pub(crate) struct CompletionRequest {
    model: String,
    prompt: PromptInput,
    #[serde(default)]
    suffix: Option<String>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    stop: Option<Stop>,
    #[serde(default)]
    stream: Option<bool>,
    /// `n`, `logprobs`, `best_of`, `echo`, `seed`, `top_k`,
    /// `repetition_penalty`: everything with no explicit field above.
    #[serde(flatten)]
    extra: serde_json::Map<String, serde_json::Value>,
}

fn build_config(request: &CompletionRequest) -> Result<GenerationConfig, String> {
    // `min_p` reaches this endpoint through `extra`, same as `top_k` and
    // `repetition_penalty` above it; `presence_penalty`/`frequency_penalty`
    // do not (no explicit field on `CompletionRequest` to source them from,
    // out of scope for this endpoint -- see `DEVIATIONS.md`).
    let shaping = build_shaping(
        request.temperature,
        request.top_p,
        None,
        None,
        &request.extra,
    )?;
    // A budget of 0 admits no generated token at all -- refused up front
    // rather than passed through to a wasted prefill-only round trip,
    // matching the chat endpoints' own `max_tokens` handling.
    if request.max_tokens == Some(0) {
        return Err("max_tokens must be greater than 0".to_string());
    }
    Ok(GenerationConfig {
        shaping,
        max_new_tokens: request.max_tokens.unwrap_or(16),
        stop_strings: stop_strings(request.stop.as_ref()),
        extra_stop_tokens: Vec::new(),
        rate: Default::default(),
    })
}

/// `logprobs`, `best_of`, and `echo` are accepted on the wire (real clients,
/// including old SDKs and evaluation harnesses, send them by default) and
/// reported rather than silently ignored -- the same contract
/// `handler::plan::openai_request_warnings` uses for the chat endpoint's own
/// unsupported fields.
fn completion_warnings(request: &CompletionRequest) -> Option<String> {
    let mut warnings = anyllm_translate::TranslationWarnings::default();
    if request.extra.contains_key("logprobs") {
        warnings.add("logprobs");
    }
    if request.extra.contains_key("best_of") {
        warnings.add("best_of");
    }
    if request
        .extra
        .get("echo")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        warnings.add("echo");
    }
    warnings.as_header_value()
}

/// Runs a generation to completion on a blocking task (synchronous CPU/GPU
/// work, matching `handler::exec::run_full`'s reason for one), collecting
/// every text delta with no decoder in front of it: this endpoint has no
/// tool or reasoning markup to separate out.
///
/// The FIFO gate's acquisition point for this endpoint's NON-streaming
/// path (`queue.rs`'s closed set): acquired here rather than at the
/// handler so the whole call, plan to result, holds one permit.
async fn run_full(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    cancel: crate::cancel::Cancel,
) -> Result<(String, RawDecodeResult), crate::handler::GenError> {
    let _gate = match model.generation_queue() {
        Some(queue) => match queue.acquire(&cancel).await {
            Some(permit) => Some(permit),
            None => return Ok((String::new(), crate::handler::cancelled_before_start())),
        },
        None => None,
    };
    let joined =
        tokio::task::spawn_blocking(move || {
            let mut text = String::new();
            let flag = crate::cancel::as_cancel_flag(&cancel);
            let result = model.run_completion(&prompt_ids, &config, None, &flag, &mut |progress| {
                match progress {
                    RawDecodeProgress::Token { delta, .. } => text.push_str(&delta),
                    RawDecodeProgress::Tail(tail) => text.push_str(&tail),
                    RawDecodeProgress::Prefill { .. } => {}
                }
            });
            (result, text)
        })
        .await;
    match joined {
        Ok((Ok(decode), text)) => Ok((text, decode)),
        Ok((Err(e), _)) => Err(crate::handler::GenError::Runtime(e)),
        Err(e) => Err(crate::handler::GenError::Join(e.to_string())),
    }
}

fn text_completion_chunk(
    id: &str,
    created: u64,
    model: &str,
    text: String,
    finish: Option<&'static str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": finish,
        }],
    })
}

fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    model_name: String,
    cancel: crate::cancel::Cancel,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = request_id("cmpl-");
    let created = now_unix();
    let cancel_for_task = cancel.clone();

    // The FIFO gate's acquisition point for this endpoint's streaming path.
    // See `handler::stream_response`'s longer note.
    let gate = model.generation_queue();
    let cancel_for_gate = cancel.clone();
    tokio::spawn(async move {
        let _ = crate::queue::run_gated(gate, &cancel_for_gate, move || {
            let send = |text: String, finish: Option<&'static str>| {
                let chunk = text_completion_chunk(&id, created, &model_name, text, finish);
                // Fast-path disconnect detection: `CancelOnDrop` below catches
                // the same event even with no chunk in flight (a long prefill
                // sends none), so this is a speed-up rather than the only
                // detector.
                if tx.send(Event::default().data(chunk.to_string())).is_err() {
                    cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            };
            let flag = crate::cancel::as_cancel_flag(&cancel_for_task);
            let result = model.run_completion(&prompt_ids, &config, None, &flag, &mut |progress| {
                let text = match progress {
                    RawDecodeProgress::Token { delta, .. } => delta,
                    RawDecodeProgress::Tail(tail) => tail,
                    RawDecodeProgress::Prefill { .. } => return,
                };
                if !text.is_empty() {
                    send(text, None);
                }
            });
            match result {
                // Discarded silently: the client that would read the finish
                // chunk is the one already gone.
                Ok(r) if r.reason == runtime::StopReason::Cancelled => return,
                // OpenAI's legacy shape reports the finish reason with no final
                // text delta rather than an empty one; the chunk above already
                // sent the last piece of content.
                Ok(r) => {
                    let reason = match finish_reason(r.reason) {
                        anyllm_translate::openai::FinishReason::Length => "length",
                        anyllm_translate::openai::FinishReason::ToolCalls => "tool_calls",
                        _ => "stop",
                    };
                    send(String::new(), Some(reason));
                }
                // A failed run is not a completed one: report it as a FRAMED
                // `error` event, matching the chat endpoints' contract -- never
                // a bare `data:` line followed by the "finished normally"
                // `[DONE]` sentinel.
                Err(e) => {
                    let body = serde_json::json!({
                        "error": {"message": e.to_string(), "type": "server_error"}
                    });
                    let _ = tx.send(Event::default().event("error").data(body.to_string()));
                    return;
                }
            }
            let _ = tx.send(Event::default().data("[DONE]"));
        })
        .await;
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(
        crate::cancel::CancelOnDrop::new(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            cancel,
        )
        .map(Ok),
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(SSE_KEEP_ALIVE))
        .into_response()
}

/// `POST /v1/completions`. The legacy raw-prompt endpoint: no chat template,
/// so `tokenizer.encode(prompt, add_bos: true)` is the one call site in this
/// crate that mirrors `turbospark-check --prompt`'s convention rather than
/// the chat endpoints' `add_bos: false` (a chat template emits its own
/// `<bos>`; nothing does here, so the tokenizer has to).
pub async fn completions(
    State(state): State<crate::ServerState>,
    tag: Option<axum::Extension<crate::observe::RequestTag>>,
    Json(request): Json<CompletionRequest>,
) -> Response {
    let model = match crate::handler::resolve_backend(
        &state,
        tag.map(|t| t.0),
        Some(request.model.as_str()),
        request.stream.unwrap_or(false),
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };
    if request.suffix.is_some() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "suffix (fill-in-the-middle) is not supported".to_string(),
        );
    }
    let n = request.extra.get("n").and_then(|v| v.as_u64()).unwrap_or(1);
    if n > 1 {
        return error_response(StatusCode::BAD_REQUEST, format!("n must be 1, got {n}"));
    }
    let config = match build_config(&request) {
        Ok(c) => c,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    let degraded = completion_warnings(&request);
    let prompt = match single_prompt(request.prompt) {
        Ok(p) => p,
        Err(e) => return error_response(StatusCode::BAD_REQUEST, e),
    };
    let prompt_ids = model.tokenizer().encode(&prompt, true);
    let cancel = crate::cancel::new_cancel();

    let mut response = if request.stream.unwrap_or(false) {
        stream_response(model, prompt_ids, config, request.model, cancel)
    } else {
        // Set if THIS future is dropped (client gone) before `run_full`
        // resolves; defused right after, whatever it returned.
        let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
        let result = run_full(model, prompt_ids, config, cancel).await;
        guard.defuse();
        match result {
            Ok((text, decode)) => Json(text_completion_full(
                request_id("cmpl-"),
                now_unix(),
                request.model,
                text,
                decode,
            ))
            .into_response(),
            Err(crate::handler::GenError::Runtime(e)) => {
                return error_response(status_for(&e), e.to_string())
            }
            Err(crate::handler::GenError::Join(m)) => {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, m)
            }
        }
    };
    if let Some(note) = degraded.and_then(|v| v.parse().ok()) {
        response
            .headers_mut()
            .insert(crate::DEGRADATION_HEADER, note);
    }
    response
}

fn text_completion_full(
    id: String,
    created: u64,
    model: String,
    text: String,
    decode: RawDecodeResult,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "text_completion",
        "created": created,
        "model": model,
        "choices": [{
            "text": text,
            "index": 0,
            "logprobs": null,
            "finish_reason": finish_reason(decode.reason),
        }],
        "usage": {
            "prompt_tokens": decode.prompt_tokens,
            "completion_tokens": decode.new_tokens,
            "total_tokens": decode.prompt_tokens + decode.new_tokens,
        }
    })
}

/// Nothing here needs a model: `PromptInput` collapsing, `suffix`/`n`
/// refusal, and the warning set are all pure request-shape decisions.
#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: serde_json::Value) -> CompletionRequest {
        serde_json::from_value(body).expect("request body should deserialize")
    }

    #[test]
    fn a_single_string_prompt_passes_through() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi"}));
        assert_eq!(single_prompt(r.prompt).unwrap(), "hi");
    }

    #[test]
    fn a_one_element_array_prompt_unwraps() {
        let r = request(serde_json::json!({"model": "m", "prompt": ["hi"]}));
        assert_eq!(single_prompt(r.prompt).unwrap(), "hi");
    }

    #[test]
    fn a_multi_element_array_prompt_is_refused() {
        let r = request(serde_json::json!({"model": "m", "prompt": ["a", "b"]}));
        let err = single_prompt(r.prompt).unwrap_err();
        assert!(err.contains('2'), "{err}");
    }

    #[test]
    fn an_empty_array_prompt_is_refused() {
        let r = request(serde_json::json!({"model": "m", "prompt": []}));
        assert!(single_prompt(r.prompt).is_err());
    }

    #[test]
    fn max_tokens_defaults_to_sixteen() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi"}));
        assert_eq!(build_config(&r).unwrap().max_new_tokens, 16);
    }

    #[test]
    fn zero_max_tokens_is_refused() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi", "max_tokens": 0}));
        let err = build_config(&r).unwrap_err();
        assert!(
            err.contains("max_tokens"),
            "expected max_tokens in err, got: {err}"
        );
    }

    #[test]
    fn max_tokens_is_read_when_present() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi", "max_tokens": 5}));
        assert_eq!(build_config(&r).unwrap().max_new_tokens, 5);
    }

    /// F23: a budget of 0 admits no generated token at all and used to pass
    /// straight through to a wasted prefill-only round trip.
    #[test]
    fn max_tokens_zero_is_refused() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi", "max_tokens": 0}));
        assert!(build_config(&r).is_err());
    }

    #[test]
    fn a_plain_request_has_no_warnings() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi"}));
        assert_eq!(completion_warnings(&r), None);
    }

    #[test]
    fn logprobs_best_of_and_echo_true_are_all_reported() {
        let r = request(serde_json::json!({
            "model": "m", "prompt": "hi",
            "logprobs": 5, "best_of": 2, "echo": true
        }));
        let header = completion_warnings(&r).unwrap();
        assert!(header.contains("logprobs"), "{header}");
        assert!(header.contains("best_of"), "{header}");
        assert!(header.contains("echo"), "{header}");
    }

    /// `echo: false` is the field's own default (unset behaves the same
    /// way), so it must not be reported -- the discriminating half of the
    /// case above.
    #[test]
    fn echo_false_is_not_reported() {
        let r = request(serde_json::json!({"model": "m", "prompt": "hi", "echo": false}));
        assert_eq!(completion_warnings(&r), None);
    }
}
