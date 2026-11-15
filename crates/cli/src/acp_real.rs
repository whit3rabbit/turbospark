//! The real chat backend for `turbospark acp`: a `generate::Session` driven
//! through the SAME open, chat-template, window-fit and context-clamp path as
//! `turbospark run` / `--chat` (`generate/session.rs`, `chat.rs`), with the
//! output going to the ACP host instead of stdout.
//!
//! * stdout is the ACP protocol channel. Nothing here may print to it:
//!   `generate::stream_turn` is NOT reused because it writes the reply to
//!   stdout. Session-open diagnostics go to stderr, as they do for `run`.
//! * The prompt always goes through the checkpoint's chat template
//!   (`render_prompt`); there is no raw `--prompt` path.
//! * Producers return logits and the runtime's sampler owns normalization; this
//!   file only passes `session.shaping` through, so there is no second softmax.
//! * Speculative decoding is forced off. The cancellable speculative loops
//!   exist but are a separate, separately gated path; the first ACP backend
//!   uses the plain and chunked cancellable loops the server also uses.
//! * No Metal command is encoded here. Every per-token encode happens inside
//!   the runner's own `gpu::autorelease_pool` scopes
//!   (`real_forward_traits.rs`), so this loop adds no pool of its own.
//! * Tools: the four ACP file tools are rendered into the prompt through the
//!   checkpoint's own chat template (`encode_generic_tool_chat`, the path
//!   `turbospark-server` uses) and the splicer's allowlist is exactly those
//!   names, so a parsed call comes back as a `ToolReq`. Executing it, the
//!   permission gate and the round caps live in `acp_engine.rs`; nothing here
//!   runs a tool. A checkpoint whose template cannot render tools falls back
//!   to the text-only prompt for the rest of the process, with a note on
//!   stderr; it never gets a tool call it could not have been offered.

use std::collections::HashSet;
use std::path::PathBuf;

use invocation::InvocationRequest;
use runtime::{
    run_raw_completion_cancellable, run_raw_completion_chunked_cancellable, GenerationConfig,
    StopReason as RtStop, TurnEvent, TurnSplitter,
};
use tokenizer::{
    FunctionDefinition, HistoricalToolCall, JsonValue, Message as TokMessage, Role as TokRole,
};

use crate::acp_engine::{tool_schemas, ChatBackend, ChatMsg, Finish, ToolReq};
use crate::acp_lock::{self, ModelLock};
use crate::generate::{clamp_max_new, map_reasoning_effort, open_session, render_prompt, Session};

pub struct RealBackend {
    session: Session,
    request: InvocationRequest,
    /// False once the checkpoint's template failed to render the tools.
    tools_render: bool,
    // Declared last: dropped after the session, so the model is unmapped
    // before another process can take the lock.
    _lock: ModelLock,
}

/// A chat-mode request with everything at its default except the model and
/// speculation (off). Built through the same parser `turbospark run` uses, so
/// the defaults cannot drift from it.
fn request_for(model: &str) -> Result<InvocationRequest, String> {
    let tokens: Vec<String> = ["--model", model, "--chat", "--speculative", "off"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    match invocation::parse(&tokens) {
        invocation::ParseOutcome::Success(r) => Ok(r),
        _ => Err(format!(
            "could not build a generation request for model {model:?}"
        )),
    }
}

impl RealBackend {
    pub fn open(model: &str) -> Result<Self, String> {
        let request = request_for(model)?;
        if let Some(message) = catalog::not_connected_message(model, catalog::ModelModality::Text) {
            return Err(message);
        }
        let model_dir: PathBuf = catalog::resolve_model_arg(model);
        let root = catalog::default_root().ok_or_else(|| {
            "neither TURBOSPARK_HOME nor HOME is set; cannot place the model lock".to_string()
        })?;
        let run_dir = root.join("run");
        // Refuse before the multi-gigabyte open, not after.
        acp_lock::refuse_if_daemon_holds(
            &run_dir,
            &model_dir,
            catalog::resolve_model_arg,
            acp_lock::pid_alive,
        )?;
        let lock = acp_lock::acquire(&run_dir, &model_dir)?;
        let mut session = open_session(&request)?;
        // Re-rendering the transcript each turn usually shares a prefix with
        // the last one; a divergent turn re-prefills and is byte-identical to
        // a cold run (`runtime::kv_prefix`), same as `--chat`.
        session.runner.set_prefix_reuse(true);
        Ok(RealBackend {
            session,
            request,
            tools_render: true,
            _lock: lock,
        })
    }
}

fn to_tokenizer(history: &[ChatMsg]) -> Vec<TokMessage> {
    history
        .iter()
        .map(|m| match m {
            ChatMsg::User(t) => TokMessage::new(TokRole::User, t.clone()),
            ChatMsg::Assistant { text, calls } => {
                let mut msg = TokMessage::new(TokRole::Assistant, text.clone());
                if text.is_empty() && !calls.is_empty() {
                    msg.content = None;
                }
                msg.tool_calls = calls
                    .iter()
                    .map(|c| HistoricalToolCall {
                        id: c.id.clone(),
                        name: c.name.clone(),
                        arguments: JsonValue::parse(&c.args.to_string()).unwrap_or(JsonValue::Null),
                    })
                    .collect();
                msg
            }
            ChatMsg::Tool {
                call_id,
                name,
                text,
            } => {
                let mut msg = TokMessage::new(TokRole::Tool, text.clone());
                msg.tool_call_id = Some(call_id.clone());
                msg.name = Some(name.clone());
                msg
            }
        })
        .collect()
}

fn tool_definitions() -> Vec<FunctionDefinition> {
    tool_schemas()
        .into_iter()
        .map(|(name, description, parameters)| FunctionDefinition {
            name: name.to_string(),
            description: description.to_string(),
            parameters: JsonValue::parse(&parameters.to_string()).unwrap_or(JsonValue::Null),
        })
        .collect()
}

/// Renders `messages` for the model: through the tool-aware template path
/// when `defs` is `Some`, else the plain chat template (`turbospark run`'s).
fn render(
    tokenizer: &tokenizer::MfTokenizer,
    messages: &[TokMessage],
    defs: Option<&[FunctionDefinition]>,
    reasoning: tokenizer::ReasoningEffort,
) -> Result<Vec<i32>, String> {
    match defs {
        Some(defs) => tokenizer
            .encode_generic_tool_chat(messages, defs, reasoning)
            .map_err(|e| format!("chat template with tools: {e}")),
        None => render_prompt(tokenizer, messages, reasoning),
    }
}

impl ChatBackend for RealBackend {
    fn generate(
        &mut self,
        history: &[ChatMsg],
        tools: bool,
        cancel: &dyn Fn() -> bool,
        delta: &mut dyn FnMut(&str),
        call: &mut dyn FnMut(ToolReq),
    ) -> Result<Finish, String> {
        let turn = to_tokenizer(history);
        if turn.is_empty() {
            return Err("the conversation has no user or assistant message".to_string());
        }
        let reasoning = map_reasoning_effort(self.request.reasoning);
        let defs = tool_definitions();
        // Tools are offered only while the checkpoint's template renders them.
        // The first render of this turn is the probe: if it fails, fall back
        // to text-only for good and say so once.
        if tools && self.tools_render {
            if let Err(e) = render(&self.session.tokenizer, &turn, Some(&defs), reasoning) {
                eprintln!(
                    "turbospark-acp-agent: this checkpoint's chat template cannot render tools \
                     ({e}); continuing text-only"
                );
                self.tools_render = false;
            }
        }
        let use_tools = tools && self.tools_render;
        let session = &mut self.session;
        let defs_ref = use_tools.then_some(defs.as_slice());
        let measure = |messages: &[TokMessage]| match render(
            &session.tokenizer,
            messages,
            defs_ref,
            reasoning,
        ) {
            Ok(ids) => ids.len() as u64,
            Err(_) => u64::MAX,
        };
        let fitted =
            window_fit::fit_conversation_window(&turn, false, session.max_context as u64, measure);
        if !fitted.has_room_for_generation() {
            if fitted.measured_length() == u64::MAX {
                return Err("the conversation failed to render through the chat template".into());
            }
            return Err(format!(
                "the message needs {} tokens and does not fit the context of {}",
                fitted.measured_length(),
                session.max_context
            ));
        }
        // Trimming drops the oldest turns first, which can leave a tool result
        // whose call was dropped at the front; a template rejects that orphan.
        let mut retained: Vec<TokMessage> = fitted.retained_turns().to_vec();
        let orphans = retained
            .iter()
            .take_while(|m| m.role == TokRole::Tool)
            .count();
        retained.drain(..orphans);
        if retained.is_empty() {
            return Err("the conversation does not fit the context window".to_string());
        }
        let prompt_ids = render(&session.tokenizer, &retained, defs_ref, reasoning)?;
        let max_new = clamp_max_new(session, &self.request, prompt_ids.len())?;

        let config = GenerationConfig {
            shaping: session.shaping,
            max_new_tokens: max_new,
            stop_strings: self.request.stop.clone(),
            extra_stop_tokens: Vec::new(),
            rate: session.rate,
        };
        let vocab_size = session.runner.vocab_size();
        let allow: HashSet<String> = if use_tools {
            tool_schemas()
                .iter()
                .map(|(n, _, _)| n.to_string())
                .collect()
        } else {
            HashSet::new()
        };
        let mut next_id = 0usize;
        let mut split = TurnSplitter::new(
            &session.tokenizer,
            &allow,
            reasoning,
            move || {
                next_id += 1;
                format!("model_{next_id}")
            },
            &prompt_ids,
        );
        let mut emit = |event: TurnEvent| match event {
            TurnEvent::Content(text) | TurnEvent::ReleasedToolSpan(text) => delta(&text),
            TurnEvent::ToolCall(c) => call(ToolReq {
                id: c.id,
                name: c.name,
                args: c.arguments.to_serde_json(),
            }),
            // Reasoning and prefill progress never reach the editor.
            TurnEvent::Reasoning(_) | TurnEvent::Prefill { .. } => {}
        };
        let on_progress = |event| split.feed(event, &mut emit);
        let chunk = if session.runner.supports_chunked_prefill() {
            Some(self.request.prefill_chunk.resolved() as usize)
        } else {
            None
        };
        let result = match chunk {
            Some(chunk) => run_raw_completion_chunked_cancellable(
                &mut session.runner,
                &session.tokenizer,
                &prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                chunk,
                cancel,
                on_progress,
            ),
            None => run_raw_completion_cancellable(
                &mut session.runner,
                &session.tokenizer,
                &prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                cancel,
                on_progress,
            ),
        }
        .map_err(|e| format!("generation failed: {e}"))?;
        // Flushes a withheld tail (a stop-terminated span, DeepSeek's tail),
        // and for Harmony emits the call the stop token ended.
        let _ = split.finish(&mut emit);
        Ok(match result.reason {
            RtStop::Cancelled => Finish::Cancelled,
            RtStop::MaxTokens => Finish::MaxTokens,
            _ => Finish::EndTurn,
        })
    }
}
