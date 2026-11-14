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
//! * Tools are not wired (text-only, see docs/ACP.md): the splicer gets an
//!   empty tool allowlist, so a tool-call span stays reasoning and is dropped,
//!   never executed and never shown as answer text.

use std::collections::HashSet;
use std::path::PathBuf;

use invocation::InvocationRequest;
use runtime::{
    run_raw_completion_cancellable, run_raw_completion_chunked_cancellable, GenerationConfig,
    StopReason as RtStop, TurnEvent, TurnSplitter,
};
use tokenizer::{Message as TokMessage, Role as TokRole};
use turbospark_acp::{Message, Role};

use crate::acp_engine::{ChatBackend, Finish};
use crate::acp_lock::{self, ModelLock};
use crate::generate::{clamp_max_new, map_reasoning_effort, open_session, render_prompt, Session};

pub struct RealBackend {
    session: Session,
    request: InvocationRequest,
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
            _lock: lock,
        })
    }
}

fn to_tokenizer(history: &[Message]) -> Vec<TokMessage> {
    history
        .iter()
        .filter_map(|m| match m.role {
            Role::User => Some(TokMessage::new(TokRole::User, m.text.clone())),
            Role::Assistant => Some(TokMessage::new(TokRole::Assistant, m.text.clone())),
            // Tools are not wired, so a tool message cannot be a faithful
            // turn; dropping it is safer than inventing a role mapping.
            Role::Tool => None,
        })
        .collect()
}

impl ChatBackend for RealBackend {
    fn generate(
        &mut self,
        history: &[Message],
        cancel: &dyn Fn() -> bool,
        delta: &mut dyn FnMut(&str),
    ) -> Result<Finish, String> {
        let turn = to_tokenizer(history);
        if turn.is_empty() {
            return Err("the conversation has no user or assistant message".to_string());
        }
        let reasoning = map_reasoning_effort(self.request.reasoning);
        let session = &mut self.session;
        let measure = |messages: &[TokMessage]| match render_prompt(
            &session.tokenizer,
            messages,
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
        let prompt_ids = render_prompt(&session.tokenizer, fitted.retained_turns(), reasoning)?;
        let max_new = clamp_max_new(session, &self.request, prompt_ids.len())?;

        let config = GenerationConfig {
            shaping: session.shaping,
            max_new_tokens: max_new,
            stop_strings: self.request.stop.clone(),
            extra_stop_tokens: Vec::new(),
            rate: session.rate,
        };
        let vocab_size = session.runner.vocab_size();
        let tools = HashSet::new();
        let mut split = TurnSplitter::new(
            &session.tokenizer,
            &tools,
            reasoning,
            String::new,
            &prompt_ids,
        );
        let mut emit = |event: TurnEvent| match event {
            TurnEvent::Content(text) | TurnEvent::ReleasedToolSpan(text) => delta(&text),
            // Reasoning and prefill progress never reach the editor; with an
            // empty allowlist no ToolCall event can arrive.
            TurnEvent::Reasoning(_) | TurnEvent::Prefill { .. } | TurnEvent::ToolCall(_) => {}
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
        // Flushes a withheld tail (a stop-terminated span, DeepSeek's tail).
        let _ = split.finish(&mut emit);
        Ok(match result.reason {
            RtStop::Cancelled => Finish::Cancelled,
            RtStop::MaxTokens => Finish::MaxTokens,
            _ => Finish::EndTurn,
        })
    }
}
