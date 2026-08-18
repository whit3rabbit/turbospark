//! Interactive multi-turn chat (`--chat`), ported from the Swift original's
//! `MferenceCLI/Run.swift` `runChat`. The model is loaded once and every
//! turn re-renders the whole history through the tokenizer's own chat
//! template, so the REPL follows whichever dialect the loaded checkpoint
//! uses. Each turn starts from a reset KV cache (`run_raw_completion` resets
//! the producer itself), so no state leaks between turns.

use std::io::BufRead;

use invocation::InvocationRequest;
use tokenizer::{Message, MfTokenizer, ReasoningEffort, Role};

use crate::generate::{
    clamp_max_new, map_reasoning_effort, open_session, print_footer, print_phases, render_prompt,
    role_name, stream_turn, Session,
};

pub fn run(request: &InvocationRequest) {
    let mut session = match open_session(request) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    let opening: Vec<Message> = request
        .system
        .as_ref()
        .map(|text| vec![Message::new(Role::System, text.clone())])
        .unwrap_or_default();
    let has_leading_instruction = !opening.is_empty();
    let mut history = opening.clone();

    eprintln!("Interactive chat. Commands: /clear, /history, /quit.");
    let stdin = std::io::stdin();
    let mut lines = stdin.lock().lines();
    loop {
        eprint!("\nyou> ");
        // A `None` line is EOF (Ctrl-D): leave the loop and exit cleanly.
        let Some(line) = lines.next() else {
            eprintln!();
            return;
        };
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("\nnote: stdin: {e}");
                return;
            }
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        if input.starts_with('/') {
            match input {
                "/quit" | "/exit" => return,
                "/clear" => {
                    history = opening.clone();
                    eprintln!("history cleared");
                }
                "/history" => {
                    for message in &history {
                        eprintln!(
                            "[{}] {}",
                            role_name(message.role),
                            message.content.as_deref().unwrap_or("")
                        );
                    }
                }
                other => {
                    eprintln!("unknown command {other}; try /clear, /history, or /quit");
                }
            }
            continue;
        }

        let mut turn = history.clone();
        turn.push(Message::new(Role::User, input));
        match take_turn(
            &mut session,
            request,
            &turn,
            has_leading_instruction,
            &mut history,
        ) {
            Ok(()) => {}
            Err(e) => eprintln!("{e}"),
        }
        print_phases(&session);
    }
}

/// Fit `turn` into the context, generate one reply, and commit both the
/// fitted history and the reply into `history`.
fn take_turn(
    session: &mut Session,
    request: &InvocationRequest,
    turn: &[Message],
    has_leading_instruction: bool,
    history: &mut Vec<Message>,
) -> Result<(), String> {
    // The SAME reasoning level the turn will actually render with. A level
    // adds a system-preamble sentence, so measuring without it under-counts
    // the prompt this window fit is deciding about.
    let reasoning = map_reasoning_effort(request.reasoning);
    let measure = |messages: &[Message]| measure_prompt(&session.tokenizer, messages, reasoning);
    let fitted = window_fit::fit_conversation_window(
        turn,
        has_leading_instruction,
        request.max_context as u64,
        measure,
    );
    if !fitted.has_room_for_generation() {
        // `measure_prompt`'s render-failure sentinel, which is a length no
        // conversation has: printing it reads as an absurd token count.
        if fitted.measured_length() == u64::MAX {
            return Err(
                "error: the conversation failed to render through the chat template".to_string(),
            );
        }
        return Err(format!(
            "error: message needs {} tokens and does not fit max_context {}; \
             shorten it or raise --max-context",
            fitted.measured_length(),
            request.max_context
        ));
    }
    if fitted.removed_turn_count() > 0 {
        eprintln!(
            "note: dropped {} oldest message(s) to fit the context",
            fitted.removed_turn_count()
        );
    }
    // NOT committed to `history` yet. A render or generation failure here
    // must leave the history as it was: committing first strands the new user
    // message in it, so the natural retry sends two consecutive `user` turns,
    // which some dialects (Mistral's `[INST]`) render as a malformed prompt.
    let fitted_history = fitted.retained_turns().to_vec();

    let prompt_ids = render_prompt(&session.tokenizer, &fitted_history, reasoning)?;
    let max_new = clamp_max_new(request, prompt_ids.len())?;
    let (reply, result) =
        stream_turn(session, request, &prompt_ids, max_new).map_err(|e| format!("error: {e}"))?;
    print_footer(&result, request.quiet);
    *history = fitted_history;
    if !reply.is_empty() {
        history.push(Message::new(Role::Assistant, reply));
    }
    Ok(())
}

/// The window-fit measurement: how many tokens the whole conversation
/// renders to, generation prompt included. A render failure measures as
/// "does not fit", which drops the oldest turn and re-measures rather than
/// aborting the REPL.
fn measure_prompt(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    reasoning: ReasoningEffort,
) -> u64 {
    match render_prompt(tokenizer, messages, reasoning) {
        Ok(ids) => ids.len() as u64,
        Err(_) => u64::MAX,
    }
}
