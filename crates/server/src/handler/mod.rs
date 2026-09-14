//! The `/v1/chat/completions` and `/v1/models` handlers, plus the generation
//! core both this endpoint and `/v1/messages` are built on: [`plan`] turns a
//! request into prompt tokens and a generation config, [`run_full`] runs it to
//! a `String`, and [`stream_blocking`] runs it emitting deltas as they arrive.
//!
//! Generation runs on a blocking task (it is synchronous CPU/GPU work, not
//! I/O). The request envelope is `anyllm_translate`'s OpenAI type rather than
//! a local struct; see `response.rs` for why.

mod exec;
mod plan;

#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use anyllm_translate::openai::ChatCompletionRequest;
use axum::extract::{Path, State};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::stream::{Stream, StreamExt};
use runtime::{GenerationConfig, RuntimeError};
use tokenizer::ReasoningEffort;

/// How often an idle SSE stream sends a comment line to keep loopback
/// proxies and clients from timing out a connection that has nothing to
/// say yet -- most visibly the gap between the request landing and the
/// first token, which at this engine's decode rates can run several
/// seconds on a long prompt.
pub(crate) const SSE_KEEP_ALIVE: Duration = Duration::from_secs(15);

pub(crate) use exec::*;
pub use plan::AppState;
pub(crate) use plan::*;

use crate::guardrails::run_guarded;
use crate::response::{
    completion_chunk, completion_response, reasoning_delta, role_delta, text_delta, tool_call_delta,
};

/// Turns the router's [`crate::ServerState`] into the one backend that will
/// serve this request, and records which one it picked.
///
/// **EVERY AXUM ENTRY POINT'S FIRST ACT, AND NOTHING FURTHER IN.** Past this
/// line the crate is single-model again: `plan`, `run_full`,
/// `stream_blocking`, `needs_decoder` and `run_guarded` all still take an
/// [`AppState`] and never learn that a registry exists. That is what keeps
/// routing by model id a change to eight function heads rather than a sweep
/// through the generation core.
///
/// `requested` is the request's own `model` field. `tag` is the id the
/// observing layer minted, absent when no observer is configured.
///
/// The `Err` variant is an `axum::Response`, which `result_large_err` reads
/// as too big to return by value. Boxing it would allocate on the refusal
/// path only to deref at the caller's `return`, and every handler in this
/// crate already returns exactly this type by value -- the lint is aimed at
/// an error type that is large by accident, and this one is the crate's own
/// return type by design.
#[allow(clippy::result_large_err)]
pub(crate) fn resolve_backend(
    state: &crate::ServerState,
    tag: Option<crate::observe::RequestTag>,
    requested: Option<&str>,
    stream: bool,
) -> Result<AppState, Response> {
    finish_resolution(
        state,
        state.registry.resolve(requested),
        tag,
        requested,
        stream,
    )
}

/// [`resolve_backend`]'s embedding-request counterpart, resolving through
/// [`crate::registry::ModelRegistry::resolve_embedding`] instead of
/// `resolve` -- the capability-aware path (`crates/server/CLAUDE.md`'s
/// registry Gotcha), so an embedding request naming a CHAT model's exact id
/// is refused rather than routed to a backend whose `encode` is the
/// trait-default error.
///
/// **Not merely a style match with `resolve_backend`.** Before this
/// existed, `embeddings.rs` called `state.registry.resolve_embedding`
/// directly, which meant an embedding request never passed through the
/// observer: no `RequestRouted` event, and any observed server (the FFI
/// host) got a BARE model with no `ReportingModel` wrapper -- harmless only
/// because `ReportingModel` did not forward `supports_embeddings`/`encode`
/// either (fixed alongside this), so the wrapper would have refused
/// embeddings outright had it ever been reached.
#[allow(clippy::result_large_err)]
pub(crate) fn resolve_embedding_backend(
    state: &crate::ServerState,
    tag: Option<crate::observe::RequestTag>,
    requested: Option<&str>,
) -> Result<AppState, Response> {
    finish_resolution(
        state,
        state.registry.resolve_embedding(requested),
        tag,
        requested,
        false,
    )
}

/// The observer-wrapping and error-mapping tail both resolvers share.
#[allow(clippy::result_large_err)]
fn finish_resolution(
    state: &crate::ServerState,
    resolution: crate::registry::Resolution,
    tag: Option<crate::observe::RequestTag>,
    requested: Option<&str>,
    stream: bool,
) -> Result<AppState, Response> {
    use crate::registry::Resolution;
    match resolution {
        Resolution::Model(model) => {
            // With no observer, or no id to tie events to, the caller gets
            // the bare model back and nothing is wrapped -- which is every
            // pre-observer caller, this crate's whole integration suite
            // included, on the exact path it always had.
            let (Some(crate::observe::RequestTag(id)), Some(observer)) = (tag, &state.observer)
            else {
                return Ok(model);
            };
            let served = model.model_id().to_string();
            observer.record(crate::observe::ServerEvent::RequestRouted {
                id,
                requested: requested.map(str::to_string),
                served,
                stream,
            });
            Ok(Arc::new(crate::observe::ReportingModel::new(
                model,
                Arc::clone(observer),
                id,
            )))
        }
        // 404 with `model_not_found`, matching what `model_detail` already
        // returns for the same question and what an OpenAI client's error
        // handling keys on. The available ids are in the message because a
        // caller who guessed wrong has no other way to find the right one.
        Resolution::Unknown {
            requested,
            available,
        } => Err((
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!(
                        "The model '{requested}' does not exist. This server has {} models \
                         attached: {}",
                        available.len(),
                        available.join(", ")
                    ),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response()),
        // A server can run with nothing CAPABLE of this request attached --
        // either nothing at all, or only models of the wrong kind (a chat
        // request against an embedding-only server, or vice versa). 503
        // rather than 404 or 400: the request itself is fine and this
        // deployment cannot serve it, which is a different thing for a
        // client's retry logic to see than a bad request or an unknown name.
        Resolution::Empty => Err(error_response(
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            "no model capable of this request is loaded; attach one before sending requests"
                .to_string(),
        )),
    }
}

pub(crate) fn error_response(status: axum::http::StatusCode, message: String) -> Response {
    (
        status,
        Json(serde_json::json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

pub(crate) fn status_for(e: &RuntimeError) -> axum::http::StatusCode {
    match e {
        RuntimeError::EmptyPrompt
        | RuntimeError::ZeroBudget
        | RuntimeError::ContextOverflow { .. }
        | RuntimeError::Selection(_) => axum::http::StatusCode::BAD_REQUEST,
        // 500 rather than 400: nothing in the REQUEST asks for speculation
        // (it is a process-level setting, like the rate cap in Gotcha 10), so
        // a caller cannot have sent anything to avoid this and must not be
        // told they did.
        RuntimeError::Producer(_) | RuntimeError::SpeculationUnavailable(_) => {
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

pub(crate) fn gen_error_response(e: GenError) -> Response {
    match e {
        GenError::Runtime(e) => error_response(status_for(&e), e.to_string()),
        GenError::Join(m) => error_response(axum::http::StatusCode::INTERNAL_SERVER_ERROR, m),
    }
}

/// `GET /health`. Reads only the registry's in-memory row count, so it answers
/// even while a generation holds the runner's lock. It deliberately omits
/// model identities and build metadata because this route is exempt from
/// authentication and exists only as a liveness/readiness probe.
pub async fn health(State(state): State<crate::ServerState>) -> Response {
    let rows = state.registry.rows();
    Json(serde_json::json!({
        "status": "ok",
        "state": if rows.is_empty() { "empty" } else { "ready" },
    }))
    .into_response()
}

/// `GET /v1/models`. One entry per public model identity. This is what an
/// OpenAI client's model picker (and Claude Code's gateway model discovery)
/// reads.
pub async fn models(State(state): State<crate::ServerState>) -> Response {
    let created = now_unix();
    Json(serde_json::json!({
        "object": "list",
        "data": state.registry.rows().into_iter().flat_map(|row| {
            let max_context = row.max_context;
            row.ids().map(|id| serde_json::json!({
                "id": id,
                "display_name": id,
                "object": "model",
                "created": created,
                "owned_by": "mference",
                // Not OpenAI's, and deliberately additive: a picker that can
                // show the window a model was OPENED at saves a user guessing,
                // and the number is per-session rather than per-checkpoint
                // (AGENTS.md Gotcha 55) so nothing else can state it.
                "context_window": max_context,
            })).collect::<Vec<_>>()
        }).collect::<Vec<_>>(),
    }))
    .into_response()
}

/// `GET /v1/models/:model`. Returns model details if `:model` names an
/// attached model.
pub async fn model_detail(
    State(state): State<crate::ServerState>,
    Path(model_id): Path<String>,
) -> Response {
    // Matched against the ROWS rather than through `registry.resolve`, which
    // would hand back the only model for any name at all under the
    // single-model fallback (`registry.rs`) -- correct for serving a
    // generation and wrong for a lookup, where the whole question is whether
    // this exact id exists.
    if let Some((id, max_context)) = state.registry.rows().into_iter().find_map(|row| {
        let max_context = row.max_context;
        row.ids()
            .find(|id| *id == model_id)
            .map(|id| (id.to_string(), max_context))
    }) {
        Json(serde_json::json!({
            "id": id,
            "object": "model",
            "created": now_unix(),
            "owned_by": "mference",
            "context_window": max_context,
        }))
        .into_response()
    } else {
        (
            axum::http::StatusCode::NOT_FOUND,
            Json(serde_json::json!({
                "error": {
                    "message": format!("The model '{model_id}' does not exist"),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found"
                }
            })),
        )
            .into_response()
    }
}

/// `POST /v1/chat/completions`. OpenAI-compatible chat completions endpoint,
/// supporting both non-streaming responses and SSE token streaming.
pub async fn chat_completions(
    State(state): State<crate::ServerState>,
    tag: Option<axum::Extension<crate::observe::RequestTag>>,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let model = match resolve_backend(
        &state,
        tag.map(|t| t.0),
        Some(request.model.as_str()),
        request.stream.unwrap_or(false),
    ) {
        Ok(m) => m,
        Err(response) => return response,
    };
    // Planned here as well as inside `run_guarded` so an unparseable request
    // is still refused with a 400 before any generation starts. The guarded
    // path re-plans because a retry turn changes the messages.
    let planned = match plan(&model, &request) {
        Ok(p) => p,
        Err(e) => return error_response(axum::http::StatusCode::BAD_REQUEST, e),
    };
    let (prompt_ids, config, images, dropped_images) = (
        planned.prompt_ids,
        planned.config,
        planned.images,
        planned.dropped_images,
    );

    let tools = tool_names(&request);
    // Re-read rather than threaded out of `plan`: `plan` already REFUSED an
    // unparseable value, so this cannot fail by the time it is reached, and
    // the pairing with `tool_names` above keeps both request-derived inputs
    // to the decoder in one place.
    let effort = reasoning_effort(&request, model.default_reasoning())
        .unwrap_or_else(|_| model.default_reasoning());
    let include_usage = request
        .stream_options
        .as_ref()
        .map(|o| o.include_usage)
        .unwrap_or(false);

    // **AN IMAGE THIS SERVER COULD NOT SERVE, OR AN OpenAI FIELD THIS
    // SERVER DOES NOT HONOUR, IS REPORTED** (ROADMAP M-V8, then widened
    // here) -- on the same header `/v1/messages` uses. Neither has a place
    // in OpenAI's own response shape, so this is an extension rather than a
    // translation, and the alternative is what shipped before: a plausible
    // answer with nothing anywhere saying part of the request was ignored.
    // Computed once, before either branch, so it applies identically to the
    // streaming and non-streaming responses below.
    let degraded = merge_degradation(openai_request_warnings(&request), dropped_images);
    let cancel = crate::cancel::new_cancel();

    let mut response = if request.stream.unwrap_or(false) {
        stream_response(
            model,
            prompt_ids,
            config,
            images,
            tools,
            effort,
            &request,
            include_usage,
            cancel,
        )
    } else {
        // Set if THIS async fn's own future is dropped (client gone) before
        // `run_guarded` resolves -- the generation it `.await`s keeps
        // running on a detached blocking task otherwise. Defused right
        // after the await, whatever it returned: there is nothing left a
        // later drop of this function could usefully cancel.
        let mut guard = crate::cancel::CancelGuard::new(cancel.clone());
        let result = run_guarded(model, &request, effort, cancel).await;
        guard.defuse();
        let generated = match result {
            Ok(r) => r,
            Err(e) => return gen_error_response(e),
        };
        Json(completion_response(
            request_id("chatcmpl-"),
            now_unix(),
            request.model,
            generated.text,
            generated.reasoning,
            generated.calls,
            generated.decode.reason,
            generated.decode.prompt_tokens as u32,
            generated.decode.new_tokens as u32,
        ))
        .into_response()
    };
    if let Some(note) = degraded.and_then(|v| v.parse().ok()) {
        response
            .headers_mut()
            .insert(crate::DEGRADATION_HEADER, note);
    }
    response
}

/// **A TOOL-CARRYING REQUEST IS BUFFERED WHEN GUARDRAILS ARE ON, and every
/// other request streams exactly as it always did.**
///
/// The guardrails need the whole generation before they can reach a verdict: a
/// call to rescue is one the decoder did not parse, so it is indistinguishable
/// from prose until the turn ends, and a retry re-generates from scratch. Emit
/// deltas live and both repairs become unavailable -- the markup is already on
/// the wire.
///
/// So a request carrying tools generates to completion, is inspected, and is
/// then re-framed as the SSE sequence it would have produced. The cost is
/// real and is taken knowingly: time-to-first-token becomes time-to-last-token
/// for that turn, roughly 5-10 s at this engine's 20-45 tok/s. The alternative
/// is guardrails that do nothing for the client that motivated them -- Claude
/// Code sends `stream: true` WITH tools -- and a `forge-guardrails-proxy` in
/// front would buffer identically.
///
/// The condition is keyed on the request carrying tools, which is what keeps
/// ordinary chat traffic on the live path byte for byte (the same shape as
/// Gotcha 12's third condition and Gotcha 7's prompt split).
#[allow(clippy::too_many_arguments)]
fn stream_response(
    model: AppState,
    prompt_ids: Vec<foundation::TokenId>,
    config: GenerationConfig,
    images: Option<crate::vision::RequestImages>,
    tools: HashSet<String>,
    effort: ReasoningEffort,
    request: &ChatCompletionRequest,
    include_usage: bool,
    cancel: crate::cancel::Cancel,
) -> Response {
    if !tools.is_empty() && model.guardrails().active() {
        return buffered_stream_response(model, request.clone(), effort, include_usage, cancel);
    }
    let model_name = request.model.clone();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = request_id("chatcmpl-");
    let created = now_unix();
    let cancel_for_task = cancel.clone();

    // The FIFO gate's acquisition point for this endpoint's live-stream
    // path (the non-streaming and buffered arms acquire inside
    // `run_guarded`). See `queue.rs`.
    let gate = model.generation_queue();
    let cancel_for_gate = cancel.clone();
    tokio::spawn(async move {
        let _ = crate::queue::run_gated(gate, &cancel_for_gate, move || {
            let send = |chunk: anyllm_translate::openai::streaming::ChatCompletionChunk| {
                // The second, faster disconnect signal: a chunk send failing
                // means the receiver -- and so the `CancelOnDrop`-wrapped
                // stream around it -- is already gone. `CancelOnDrop` catches
                // the same event independent of whether a send was ever
                // attempted (a long prefill sends none), so this is a speed-up
                // for the common case rather than the only detector.
                if tx
                    .send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()))
                    .is_err()
                {
                    cancel_for_task.store(true, std::sync::atomic::Ordering::Relaxed);
                }
            };
            send(completion_chunk(
                id.clone(),
                created,
                model_name.clone(),
                role_delta(),
                None,
            ));

            let mut call_index = 0u32;
            let flag = crate::cancel::as_cancel_flag(&cancel_for_task);
            let result = stream_blocking(
                &model,
                &prompt_ids,
                &config,
                images.as_ref(),
                &tools,
                effort,
                &flag,
                &mut |piece| {
                    let delta = match piece {
                        Piece::Text(text) => text_delta(text),
                        Piece::Reasoning(text) => reasoning_delta(text),
                        Piece::Tool(call) => {
                            call_index += 1;
                            tool_call_delta(call_index - 1, call)
                        }
                    };
                    send(completion_chunk(
                        id.clone(),
                        created,
                        model_name.clone(),
                        delta,
                        None,
                    ));
                },
            );

            match result {
                // A cancelled generation is discarded silently: the client that
                // would read a finish chunk or an error event is the one
                // already gone, and sending either into a dropped receiver is
                // both pointless and (per `send`, above) itself an error.
                Ok(r) if r.reason == runtime::StopReason::Cancelled => return,
                Ok(r) => {
                    send(completion_chunk(
                        id.clone(),
                        created,
                        model_name.clone(),
                        Default::default(),
                        Some(crate::response::finish_reason_for(r.reason, call_index > 0)),
                    ));
                    if include_usage {
                        send(crate::response::usage_chunk(
                            id.clone(),
                            created,
                            model_name.clone(),
                            r.prompt_tokens as u32,
                            r.new_tokens as u32,
                        ));
                    }
                }
                // A failed run is not a completed one: report it as a FRAMED
                // `error` event -- a client dispatches on the `event:` line, and
                // a bare `data:` line is indistinguishable from a chunk it does
                // not recognise. `[DONE]` follows it below, which is worse than
                // silence: it is the sentinel that says the stream finished
                // NORMALLY, so a client reading past the error would see success.
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

/// The guarded generation, re-framed as the SSE sequence the live path would
/// have produced.
///
/// **THE CHUNK ORDER IS LOAD BEARING and is not cosmetic.** Anthropic's
/// `StreamingTranslator` is a state machine over this exact sequence (crate
/// Gotcha 6): it opens the message on the first chunk carrying a role, opens a
/// thinking block on the first reasoning delta and closes it on the first
/// content delta without reopening. Reordering these -- content before
/// reasoning, or a finish chunk before the tool calls -- produces malformed
/// Anthropic events rather than an error, so `/v1/messages` inherits its
/// correctness from this function keeping role, reasoning, content, tool
/// calls, finish.
fn buffered_stream_response(
    model: AppState,
    request: ChatCompletionRequest,
    effort: ReasoningEffort,
    include_usage: bool,
    cancel: crate::cancel::Cancel,
) -> Response {
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let id = request_id("chatcmpl-");
    let created = now_unix();
    let model_name = request.model.clone();
    let cancel_for_stream = cancel.clone();

    tokio::spawn(async move {
        let send = |chunk: anyllm_translate::openai::streaming::ChatCompletionChunk| {
            let _ =
                tx.send(Event::default().data(serde_json::to_string(&chunk).unwrap_or_default()));
        };
        let chunk = |delta, finish| {
            completion_chunk(id.clone(), created, model_name.clone(), delta, finish)
        };

        // Nothing is sent until the whole generation is done, so `send`'s
        // own failure detection cannot fire during it -- `CancelOnDrop`
        // below is the ONLY signal this path has that the client is gone,
        // which is exactly why the plan calls this path out separately.
        match run_guarded(model, &request, effort, cancel).await {
            // Discarded silently, same contract as the live path: the
            // client that would read this turn is the one already gone.
            Ok(generated) if generated.decode.reason == runtime::StopReason::Cancelled => return,
            Ok(generated) => {
                send(chunk(role_delta(), None));
                if !generated.reasoning.is_empty() {
                    send(chunk(reasoning_delta(generated.reasoning), None));
                }
                if !generated.text.is_empty() {
                    send(chunk(text_delta(generated.text), None));
                }
                let has_calls = !generated.calls.is_empty();
                for (index, call) in generated.calls.into_iter().enumerate() {
                    send(chunk(tool_call_delta(index as u32, call), None));
                }
                send(chunk(
                    Default::default(),
                    Some(crate::response::finish_reason_for(
                        generated.decode.reason,
                        has_calls,
                    )),
                ));
                if include_usage {
                    send(crate::response::usage_chunk(
                        id.clone(),
                        created,
                        model_name.clone(),
                        generated.decode.prompt_tokens as u32,
                        generated.decode.new_tokens as u32,
                    ));
                }
            }
            // Same contract as the live path: a failed run is not a
            // completed one, so it is a FRAMED `error` event -- never a bare
            // `data:` line followed by the "finished normally" sentinel.
            Err(e) => {
                let message = match e {
                    GenError::Runtime(e) => e.to_string(),
                    GenError::Join(m) => m,
                };
                let body = serde_json::json!({
                    "error": {"message": message, "type": "server_error"}
                });
                let _ = tx.send(Event::default().event("error").data(body.to_string()));
                return;
            }
        }
        let _ = tx.send(Event::default().data("[DONE]"));
    });

    let stream: std::pin::Pin<
        Box<dyn Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(
        crate::cancel::CancelOnDrop::new(
            tokio_stream::wrappers::UnboundedReceiverStream::new(rx),
            cancel_for_stream,
        )
        .map(Ok),
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(SSE_KEEP_ALIVE))
        .into_response()
}
