//! Real generation, wired to `RealForwardRunner` (macOS/GPU only). Loads a
//! `.gturbo` install from `request.model`, reconstructs its full
//! `ArchConfig` from `manifest.json`'s own `arch` object (shape fields
//! read as written; family-extension fields fall back to Gemma 4's
//! baseline values, the same fallback rule `arch_validation` applies to
//! omitted manifest fields), and opens it with `RealForwardRunner`. This
//! covers both the synthetic short-name installs and real Gemma 4
//! installs repacked by `repack::write_gemma4_install` (verbatim
//! checkpoint tensor naming, MoE, mixed SWA/full attention); anything the
//! runner does not support (linear/compressed layers, non-Gemma families)
//! is rejected with a clear error, not a crash. The tokenizer is expected
//! to live alongside the `.gturbo` files in the same directory (the usual
//! HF checkpoint bundling convention).
//!
//! All three invocation modes generate: `--prompt` encodes its text
//! verbatim (no templating, matching the Swift original), `--messages-file`
//! renders a JSON conversation through the tokenizer's own chat template,
//! and `--chat` runs the interactive REPL in [`crate::chat`].

use std::collections::HashSet;
use std::io::Write;

use invocation::{InvocationRequest, Mode};
use runtime::{
    run_raw_completion, run_raw_completion_chunked, run_raw_completion_speculative,
    GenerationConfig, RawDecodeResult, TurnSplitter,
};
use tokenizer::{Message, MfTokenizer, ReasoningEffort, ReasoningSupport};

pub(crate) mod format;
pub(crate) mod session;
pub(crate) mod vision;

pub(crate) use format::{
    map_reasoning_effort, parse_messages_file, print_footer, print_phases, role_name,
};
pub(crate) use runtime::SpeculationPlan;
pub(crate) use session::{open_session, Session};

pub fn try_generate(request: &InvocationRequest) {
    match &request.mode {
        Mode::Prompt(text) => run_prompt(request, text),
        Mode::MessagesFile(path) => run_messages_file(request, path),
        Mode::Chat => crate::chat::run(request),
    }
}

fn run_prompt(request: &InvocationRequest, prompt: &str) {
    let mut session = match open_session(request) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    // Verbatim, with a BOS prefix and no chat template: this mode's contract,
    // matching the Swift original. An instruction-tuned model babbles here;
    // that is the missing markup, not a decode bug.
    //
    // **`--image` IS REFUSED HERE RATHER THAN SERVED**, and that is this
    // mode's contract rather than an omission: it encodes the prompt VERBATIM
    // with no template, so there is no `<|vision_start|><|image_pad|>` run for
    // the splice to expand and nowhere for the tower's rows to land. Serving
    // it would mean this binary inventing framing the checkpoint was not
    // trained on -- fluent output that ignores the picture (AGENTS.md Gotcha
    // 41). `--messages-file` applies the template and is where images go.
    if !request.images.is_empty() {
        eprintln!(
            "note: not attempting real generation: --image needs a chat template and --prompt \
             encodes verbatim; use --messages-file (its content parts take image entries) or \
             --chat"
        );
        return;
    }
    let prompt_ids = session.tokenizer.encode(prompt, true);
    // Clamp rather than let `check_admission` refuse the whole run: a long
    // raw prompt generates into whatever room is left, the same as the other
    // two modes.
    let max_new = match clamp_max_new(&session, request, prompt_ids.len()) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    println!("generating (real forward pass):");
    // Shared with the other two modes rather than reimplemented: this is
    // where the withheld `Tail` is printed (dropping it truncates the reply)
    // and where Harmony's channels are split.
    match stream_turn(&mut session, request, &prompt_ids, max_new) {
        Ok((_, result)) => print_footer(&result, request.quiet),
        Err(e) => eprintln!("generation failed: {e}"),
    }
    print_phases(&session);
}

fn run_messages_file(request: &InvocationRequest, path: &str) {
    // Decode the conversation before loading the model: a typo in the JSON
    // should not cost a multi-gigabyte install open first.
    let (messages, file_images) = match parse_messages_file(path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };
    // TWO SOURCES, and they may not both be used. A file's parts already say
    // WHERE each image goes; `--image` says only that there is one. Accepting
    // both would leave the pairing between a path and a marker run ambiguous,
    // which is the silent mispairing every refusal in this feature exists to
    // prevent.
    if !file_images.is_empty() && !request.images.is_empty() {
        eprintln!(
            "note: not attempting real generation: {path} already carries image content parts, \
             so --image would be ambiguous about which marker run each path pairs with; use one \
             or the other"
        );
        return;
    }
    // A file that already carries parts cannot be BATCHED: it places its own
    // markers inside the conversation, so "one page per turn" has no meaning
    // -- which of the file's parts would each turn keep?
    if request.image_batch && !file_images.is_empty() {
        eprintln!(
            "note: not attempting real generation: --image-batch runs the prompt once per \
             --image, and {path} places its images itself; drop --image-batch, or move the paths \
             to --image"
        );
        return;
    }
    // The messages are built PER TURN below, because under `--image-batch`
    // each turn carries one marker rather than all of them. A file's messages
    // already carry their parts and are used as they are.
    let from_file = !file_images.is_empty();
    let images = if from_file {
        file_images
    } else {
        request.images.clone()
    };
    let mut session = match open_session(request) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("note: not attempting real generation: {e}");
            return;
        }
    };

    // ONE PASS PER PAGE under `--image-batch`, otherwise one pass carrying
    // every image. The batch arm is the bulk-OCR shape the whole vision design
    // exists for: ONE open runner, the tower's scratch allocated and dropped
    // per page, and `reset()` between turns -- so peak memory is flat in the
    // page count rather than growing with it.
    let turns: Vec<Vec<String>> = if request.image_batch {
        if images.is_empty() {
            eprintln!(
                "note: not attempting real generation: --image-batch needs at least one --image"
            );
            return;
        }
        images.iter().map(|p| vec![p.clone()]).collect()
    } else {
        vec![images]
    };
    let batched = turns.len() > 1 || request.image_batch;

    for (page, turn_images) in turns.iter().enumerate() {
        // CONSUME the previous page's map before building this one. `reset`
        // deliberately does NOT do this -- it runs at the START of the
        // generation the map belongs to, so clearing there destroys the map
        // before prefill (`crates/runtime/src/real_forward_traits.rs`). The
        // KV cache is rewound by `run_raw_completion`'s own reset; only the
        // map is this loop's to release.
        session.runner.clear_prompt_vision();
        if batched {
            println!(
                "--- image {} of {}: {}",
                page + 1,
                turns.len(),
                turn_images[0]
            );
        }
        // THIS TURN's messages carry THIS turn's image parts. Under
        // `--image-batch` that is one marker per turn, not all of them, which
        // is what the splice then expands -- rendering three markers and
        // supplying one grid is a mismatch the walk refuses by name.
        let turn_messages = if from_file {
            messages.clone()
        } else {
            match prepend_images_to_last_user(messages.clone(), turn_images) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("note: not attempting real generation: {e}");
                    return;
                }
            }
        };
        if let Err(e) = run_one_turn(&mut session, request, &turn_messages, turn_images) {
            eprintln!("note: {e}");
            return;
        }
    }
    print_phases(&session);
}

/// Render, attach any images, and stream one turn.
fn run_one_turn(
    session: &mut Session,
    request: &InvocationRequest,
    messages: &[Message],
    images: &[String],
) -> Result<(), String> {
    let rendered = render_prompt(
        &session.tokenizer,
        messages,
        map_reasoning_effort(request.reasoning),
    )?;

    // The rendered ids carry ONE `<|image_pad|>` per image; the spliced ones
    // carry `merged_tokens` copies and are what the model sees.
    let prompt_ids = if images.is_empty() {
        rendered
    } else {
        if !session.runner.has_vision_tower() {
            return Err(format!(
                "not attempting real generation: {} image(s) were given but this install \
                 declares no vision tower; re-stream the checkpoint with its vision_tower.* \
                 tensors",
                images.len()
            ));
        }
        let params = vision::preprocess_params(session)?;
        let prepared = vision::prepare_images(images, &params)?;
        vision::attach(session, &rendered, &prepared, &params)?
    };

    let max_new = clamp_max_new(session, request, prompt_ids.len())?;
    println!("generating (real forward pass, chat template applied):");
    match stream_turn(session, request, &prompt_ids, max_new) {
        Ok((_, result)) => print_footer(&result, request.quiet),
        Err(e) => eprintln!("generation failed: {e}"),
    }
    Ok(())
}

/// Put `--image` paths into the LAST user turn, before its text.
///
/// **PREPEND, not append**, and that is matched to the reference rather than
/// chosen: `apply_chat_template(processor, config, question, num_images=1)`
/// builds `[image, text]`, so the marker run comes first and the question
/// follows. Appending would move every mRoPE position past the image and
/// produce a different prompt for the same request.
///
/// A conversation with no user turn is REFUSED rather than growing one: the
/// caller said where the question goes, and inventing a turn to hang the
/// picture on is this binary deciding framing it has no basis for.
fn prepend_images_to_last_user(
    mut messages: Vec<Message>,
    images: &[String],
) -> Result<Vec<Message>, String> {
    if images.is_empty() {
        return Ok(messages);
    }
    let Some(index) = messages
        .iter()
        .rposition(|m| m.role == tokenizer::Role::User)
    else {
        return Err(
            "--image needs a user turn to attach to, and this conversation has none".to_string(),
        );
    };
    let target = &mut messages[index];
    let mut parts: Vec<tokenizer::ContentPart> = images
        .iter()
        .map(|_| tokenizer::ContentPart::Image)
        .collect();
    if target.content_parts.is_empty() {
        if let Some(text) = target.content.clone() {
            parts.push(tokenizer::ContentPart::Text(text));
        }
    } else {
        parts.extend(target.content_parts.iter().cloned());
    }
    *target = Message::with_parts(target.role, parts);
    Ok(messages)
}

/// Render `messages` through the loaded tokenizer's own dialect template.
///
/// `add_bos` is false on purpose: the Gemma template emits the literal
/// `<bos>` mark itself, so encoding with a BOS prefix would double it.
///
/// **A LEVEL THIS CHECKPOINT CANNOT SPELL IS REPORTED, NOT SWALLOWED.** A
/// template with no effort key still honours `enable_thinking`, so the
/// request is half-served and the half that was dropped is invisible in the
/// output -- exactly the silent no-op `--reasoning` exists to avoid being.
pub(crate) fn render_prompt(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    reasoning: ReasoningEffort,
) -> Result<Vec<i32>, String> {
    if reasoning != ReasoningEffort::Off
        && tokenizer.reasoning_support() == ReasoningSupport::ToggleOnly
    {
        eprintln!(
            "note: this checkpoint's chat template has no reasoning-effort knob, so \
             --reasoning {} turns thinking ON but sets no level",
            reasoning.as_str()
        );
    }
    let rendered = tokenizer
        .apply_chat_template_with_reasoning(messages, reasoning)
        .map_err(|e| format!("chat template: {e}"))?;
    Ok(tokenizer.encode(&rendered, false))
}

/// The per-turn generation budget: never more than `--max-new`, never more
/// than the context leaves room for.
///
/// Takes the window from the SESSION rather than the request, because under
/// `--max-context auto` the request carries no number and the KV cache has
/// already been allocated at the resolved one.
pub(crate) fn clamp_max_new(
    session: &Session,
    request: &InvocationRequest,
    prompt_len: usize,
) -> Result<u32, String> {
    if prompt_len >= session.max_context as usize {
        return Err(format!(
            "context overflow: prompt {prompt_len} reaches max_context {}",
            session.max_context
        ));
    }
    let room = session.max_context - prompt_len as u32;
    Ok(request.max_new.min(room))
}

/// Generate one turn, streaming deltas to stdout, and return the text that
/// was streamed so an interactive caller can append it to its history.
pub(crate) fn stream_turn(
    session: &mut Session,
    request: &InvocationRequest,
    prompt_ids: &[i32],
    max_new: u32,
) -> Result<(String, RawDecodeResult), runtime::RuntimeError> {
    let config = GenerationConfig {
        shaping: session.shaping,
        max_new_tokens: max_new,
        stop_strings: request.stop.clone(),
        extra_stop_tokens: Vec::new(),
        rate: session.rate,
    };
    let vocab_size = session.runner.vocab_size();

    let mut reply = String::new();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let tools = HashSet::new();
    let mut split = TurnSplitter::new(
        &session.tokenizer,
        &tools,
        map_reasoning_effort(request.reasoning),
        String::new,
        prompt_ids,
    );
    let mut emit = |turn: runtime::TurnEvent| match turn {
        // No prefill progress on stdout, and no tool rendering: the
        // allowlist is empty, so a call stays reasoning and no
        // TurnEvent::ToolCall can arrive.
        runtime::TurnEvent::Prefill { .. } | runtime::TurnEvent::ToolCall(_) => {}
        // Reasoning goes to stderr so redirecting stdout captures the
        // ANSWER alone, and only the answer becomes the assistant turn.
        runtime::TurnEvent::Reasoning(reasoning) => eprint!("{reasoning}"),
        runtime::TurnEvent::Content(answer) | runtime::TurnEvent::ReleasedToolSpan(answer) => {
            let _ = write!(out, "{answer}");
            let _ = out.flush();
            reply.push_str(&answer);
        }
    };
    let on_progress = |event| {
        split.feed(event, &mut emit);
    };
    let chunk_tokens = resolve_chunk_tokens(session, request);
    let result = match (&session.speculation, chunk_tokens) {
        // Speculation wins over the chunked-prefill seam when both are on:
        // the speculative loop has its own prefill (it primes the drafter as
        // it walks) and no chunked variant. They are not composable today and
        // pretending otherwise would silently run one of them.
        (SpeculationPlan::Enabled { block }, _) => run_raw_completion_speculative(
            &mut session.runner,
            &session.tokenizer,
            prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            *block,
            on_progress,
        )?,
        (SpeculationPlan::Disabled { .. }, chunk_tokens) => match chunk_tokens {
            Some(chunk) => run_raw_completion_chunked(
                &mut session.runner,
                &session.tokenizer,
                prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                chunk,
                on_progress,
            )?,
            None => run_raw_completion(
                &mut session.runner,
                &session.tokenizer,
                prompt_ids,
                &config,
                session.max_context,
                vocab_size,
                on_progress,
            )?,
        },
    };
    // Flushes a stop-token-terminated tool call or DeepSeek's withheld tail,
    // same as crates/server/src/handler/exec.rs. A no-op today: the empty
    // tool set above means no dialect ever enters a state finish() would
    // need to close (DEVIATIONS.md).
    let _ = split.finish(&mut emit);
    let _ = writeln!(out);
    Ok((reply, result))
}

/// Whether, and at what chunk size, this turn's prefill should route
/// through the chunked driver.
///
/// `TURBOSPARK_PREFILL_CHUNK` (or legacy `TURBOSPARK_PREFILL_CHUNK`) still wins
/// when set: it is the existing A/B seam (`docs/BATCHED_PREFILL.md` step 1),
/// spelled like `TURBOSPARK_SHARED_CB` and `TURBOSPARK_ROUTED_PIPELINE` beside
/// it, and both its arms must produce identical tokens -- an explicit ask that
/// the family can't serve stays a hard error via `prefill_chunk`'s own
/// refusal, unchanged from before this flag was wired.
///
/// `--prefill-chunk` is different on purpose: it carries a default
/// (`Fixed(128)`) on every invocation, whether or not the caller typed it,
/// so routing it through the chunked driver only when
/// `RealForwardRunner::supports_chunked_prefill` says this install can serve
/// it -- and falling back to the sequential path with no error otherwise --
/// is what keeps a caller who never named the flag from seeing a family it
/// never asked about.
fn resolve_chunk_tokens(session: &Session, request: &InvocationRequest) -> Option<usize> {
    if let Some(env_chunk) = std::env::var("TURBOSPARK_PREFILL_CHUNK")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
    {
        return Some(env_chunk);
    }
    if !session.runner.supports_chunked_prefill() {
        return None;
    }
    Some(request.prefill_chunk.resolved() as usize)
}
