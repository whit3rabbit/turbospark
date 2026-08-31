//! One turn: render, decode, stream, and report.

use std::collections::HashSet;
use std::sync::atomic::Ordering;

use runtime::{GenerationConfig, RawDecodeProgress, RawDecodeResult, StopReason};
use tokenizer::{
    ChatDialect, ContentPart, Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role,
    StructuredAssistantDecoder, StructuredAssistantEvent,
};

use crate::session::{Engine, Session};
use crate::wire::{GenerateOptions, GenerateResult, WireMessage, WirePart};

/// Prefill progress event kind.
pub const TS_EVENT_PREFILL: i32 = 0;
/// Content text delta event kind.
pub const TS_EVENT_CONTENT: i32 = 1;
/// Reasoning / thought text delta event kind.
pub const TS_EVENT_REASONING: i32 = 2;

/// Splits a token's text into the answer and the reasoning that preceded it.
///
/// Ported from `crates/cli/src/generate.rs::ChannelSplit`, and the two
/// conditions are the same because the reasons are:
///
/// - `gpt-oss` ALWAYS needs a decoder. Harmony puts the model's reasoning in
///   an `analysis` channel before its answer whatever the caller asked for,
///   so without this the reasoning and the frame markup arrive as the reply.
/// - ChatML and Gemma need one only when a reasoning LEVEL was asked for.
///   Their thought channels are unreachable otherwise, so building a decoder
///   unconditionally would route shipped families through a state machine
///   with nothing to do.
///
/// Skipping it is not cosmetic. Measured on the real Gemma 4 install the
/// first time a level was asked for, the reply began with a bare `thought`
/// (the channel LABEL, as prose), then the model's scratch work, then its
/// answer, all as one run of content.
struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    /// `prompt_ids` is the rendered generation prompt: a ChatML template opens
    /// the `<think>` frame itself when thinking is on, and the decoder cannot
    /// tell without being shown (`StructuredAssistantDecoder::new`).
    fn new(tokenizer: &'a MfTokenizer, reasoning: ReasoningEffort, prompt_ids: &[i32]) -> Self {
        let wanted = matches!(
            tokenizer.dialect,
            ChatDialect::Harmony | ChatDialect::MuseGlimmer
        ) || (reasoning != ReasoningEffort::Off
            && matches!(tokenizer.dialect, ChatDialect::ChatMl | ChatDialect::Gemma));
        Self {
            decoder: wanted.then(|| {
                // An EMPTY tool allowlist. This binding has no way to run a
                // tool, so a Harmony `commentary` body stays reasoning
                // rather than being parsed as a call the caller cannot
                // service. Tools belong to the server surface.
                StructuredAssistantDecoder::new(tokenizer, HashSet::new(), String::new, prompt_ids)
            }),
        }
    }

    /// One token's `(answer, reasoning)`. Either may be empty.
    fn push(&mut self, id: i32, text: &str) -> (String, String) {
        let Some(decoder) = self.decoder.as_mut() else {
            return (text.to_string(), String::new());
        };
        let (mut answer, mut reasoning) = (String::new(), String::new());
        match decoder.consume(id, text) {
            Ok(events) => {
                for event in events {
                    match event {
                        StructuredAssistantEvent::Content(c) => answer.push_str(&c),
                        StructuredAssistantEvent::Reasoning(r) => reasoning.push_str(&r),
                        // Unreachable with an empty allowlist, and dropping
                        // beats inventing a rendering for it.
                        StructuredAssistantEvent::ToolCall(_) => {}
                    }
                }
            }
            // Losing a caller's output to a decoder error would be the worst
            // outcome available: pass the text through.
            Err(_) => answer.push_str(text),
        }
        (answer, reasoning)
    }
}

fn role_of(name: &str) -> Result<Role, String> {
    match name {
        "system" => Ok(Role::System),
        "developer" => Ok(Role::Developer),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        other => Err(format!("unknown role {other:?}")),
    }
}

/// Maps the wire shapes onto the tokenizer's own message type.
///
/// A message with parts becomes a MULTIMODAL message and renders through
/// `content_parts`; a plain string takes the text path every caller that
/// predates images already took, so nothing about a text-only render moves.
/// An image part becomes a bare `ContentPart::Image` placeholder: the pixels
/// travel separately and the two halves meet at the id sequence
/// (`crate::vision`).
fn decode_messages(messages: &[WireMessage]) -> Result<Vec<Message>, String> {
    messages
        .iter()
        .map(|m| {
            let role = role_of(&m.role)?;
            match m.parts() {
                None => Ok(Message::new(role, m.text())),
                Some(parts) => {
                    let mapped: Vec<ContentPart> = parts
                        .iter()
                        .map(|p| match p {
                            WirePart::Text { text } => ContentPart::Text(text.clone()),
                            WirePart::Image { .. } => ContentPart::Image,
                        })
                        .collect();
                    Ok(Message::with_parts(role, mapped))
                }
            }
        })
        .collect()
}

/// Every image part in the conversation, in order, with its SHAPE checked.
///
/// Cheap and pure, so a malformed part costs a message rather than the engine
/// lock -- the same split `crates/server/src/vision.rs` makes between
/// `validate_urls` (before the model) and the decode (inside the lock). What
/// it cannot check here is the bytes, which need a decoder.
///
/// Order is load-bearing: the nth image pairs with the nth marker run the
/// template renders, so this flattens across messages without sorting.
fn collect_image_parts(messages: &[WireMessage]) -> Result<Vec<&WirePart>, String> {
    let parts: Vec<&WirePart> = messages.iter().flat_map(|m| m.image_parts()).collect();
    for part in &parts {
        part.image_source()?;
    }
    Ok(parts)
}

/// Encode this turn's images and return the SPLICED prompt.
///
/// **Everything happens under the caller's one lock**, which is the same
/// contract `crates/server/src/model.rs` states for `run_completion`: encode
/// and generate cannot be separate acquisitions, or two turns interleave and
/// each prefills the other's picture, both fluently and with no error.
///
/// With no images this is `rendered` unchanged and touches nothing, so a
/// text-only turn takes the byte-identical path it always did.
#[cfg(target_os = "macos")]
fn attach_images(
    engine: &mut Engine,
    session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<Vec<i32>, String> {
    if parts.is_empty() {
        return Ok(rendered.to_vec());
    }
    let Engine::Real(runner) = engine else {
        return Err(SCRIPTED_HAS_NO_TOWER.to_string());
    };
    if !runner.has_vision_tower() {
        return Err(format!(
            "this install declares no vision tower, so it cannot accept an image: {}",
            session.info.model_path
        ));
    }
    let params = crate::vision::preprocess_params(runner, &session.model_dir)?;
    let images = crate::vision::prepare(parts, &params)?;
    crate::vision::attach(runner.as_mut(), rendered, &images, &params)
}

/// The portable half: off macOS there is no runner to encode with.
#[cfg(not(target_os = "macos"))]
fn attach_images(
    _engine: &mut Engine,
    _session: &Session,
    rendered: &[i32],
    parts: &[&WirePart],
) -> Result<Vec<i32>, String> {
    if parts.is_empty() {
        return Ok(rendered.to_vec());
    }
    Err("the engine is macOS-only, so no vision tower can run here".to_string())
}

/// Drops this turn's injection map. Called on BOTH exits from generation.
#[cfg(target_os = "macos")]
fn clear_vision(engine: &mut Engine) {
    if let Engine::Real(runner) = engine {
        runner.clear_prompt_vision();
    }
}

#[cfg(not(target_os = "macos"))]
fn clear_vision(_engine: &mut Engine) {}

/// Named once so both the scripted refusal and its test read the same words.
pub(crate) const SCRIPTED_HAS_NO_TOWER: &str =
    "this session decodes through a scripted producer, which has no vision tower; \
     open a real install to send images";

fn stop_reason_name(reason: StopReason) -> &'static str {
    match reason {
        StopReason::EndOfTurn => "endOfTurn",
        StopReason::ToolCalls => "toolCalls",
        StopReason::Eos => "eos",
        StopReason::StopString => "stopString",
        StopReason::MaxTokens => "maxTokens",
        StopReason::Cancelled => "cancelled",
    }
}

/// Renders the conversation and encodes it.
///
/// The checkpoint's own chat template, never a raw concatenation: an
/// instruction-tuned model fed unrendered text babbles, and that is missing
/// markup rather than a decode bug.
fn render(
    tokenizer: &MfTokenizer,
    messages: &[WireMessage],
    reasoning: ReasoningEffort,
) -> Result<(Vec<i32>, Option<String>), String> {
    let mut note = None;
    if reasoning != ReasoningEffort::Off {
        match tokenizer.reasoning_support() {
            // Thinking turns ON but the LEVEL is dropped. Reported rather
            // than silently honoured: a caller who asked for `low` and got
            // the checkpoint's own default should know which they got.
            ReasoningSupport::ToggleOnly => {
                note = Some(
                    "this checkpoint's chat template has no reasoning-effort knob, so \
                     thinking is ON but no level is set"
                        .to_string(),
                )
            }
            // No template at all: a level is REFUSED rather than dropped,
            // because there is nothing to express it with and silence is the
            // failure mode worth avoiding.
            ReasoningSupport::None => {
                return Err(
                    "this checkpoint ships no chat template, so a reasoning level cannot \
                     be expressed; send reasoning \"off\""
                        .to_string(),
                )
            }
            ReasoningSupport::Level => {}
        }
    }
    let decoded = decode_messages(messages)?;
    let rendered = tokenizer
        .apply_chat_template_with_reasoning(&decoded, reasoning)
        .map_err(|e| format!("chat template: {e}"))?;
    // `false`: the template emits its own BOS as text, so a BOS prefix here
    // doubles it -- invisible on a toy fixture, and it degrades output on a
    // real install.
    Ok((tokenizer.encode(&rendered, false), note))
}

/// The per-turn generation budget: never more than asked, never more than
/// the context leaves room for.
///
/// Clamped rather than refused, so a long conversation generates into
/// whatever room is left instead of failing outright. The window comes from
/// the SESSION, because under `auto` the caller named no number and the KV
/// cache was allocated at the resolved one.
fn clamp_max_new(session: &Session, asked: u32, prompt_len: usize) -> Result<u32, String> {
    if prompt_len >= session.max_context as usize {
        return Err(format!(
            "context overflow: the rendered prompt is {prompt_len} tokens and the \
             resolved window is {}",
            session.max_context
        ));
    }
    Ok(asked.min(session.max_context - prompt_len as u32))
}

/// The block THIS TURN drafts by: the session's, kept only when the turn is
/// deterministic.
///
/// **THE SECOND HALF OF THE SPECULATION DECISION IS PER TURN, and it is the
/// one thing this binding cannot settle at open.** Acceptance is
/// `argmax(target) == proposal`, which is exact speculative decoding at
/// temperature 0 and biased at any other, so a sampled turn takes the
/// sequential loop however the session resolved. That is the SERVER's shape
/// rather than the CLI's, and for the server's reason: the CLI has one
/// shaping per process and can settle both halves at open, while here the
/// temperature belongs to the request.
///
/// It falls back SILENTLY rather than failing. This binding's own sampling
/// default is T=0.2, so sampled is the NORMAL case -- a GUI would be told
/// off once per turn for a setting it never sent, and refusing would turn a
/// valid request into an error. What the caller is owed instead is the
/// session-level answer, and that is in `sessionInfo.speculation`, said
/// once.
///
/// A free function taking both inputs rather than a method, so the decision
/// is pinnable without a session and without a 14 GB install.
fn turn_block(session_block: Option<usize>, deterministic: bool) -> Option<usize> {
    session_block.filter(|_| deterministic)
}

/// Evaluates the prompt token count without running generation.
pub(crate) fn count_tokens(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
) -> Result<u32, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;
    let (prompt_ids, _note) = render(&session.tokenizer, messages, reasoning)?;
    Ok(prompt_ids.len() as u32)
}

/// Evaluates the token count of a raw text string using the session tokenizer.
pub(crate) fn count_text_tokens(session: &Session, text: &str, add_special: bool) -> u32 {
    session.tokenizer.encode(text, add_special).len() as u32
}

/// Fits a conversation transcript into a context budget using `turbospark-window-fit`.
///
/// **THE MEASUREMENT IS A FLOOR ON A CONVERSATION CARRYING IMAGES.** `render`
/// emits ONE `<|image_pad|>` per picture whatever its size, and the expansion
/// to that page's merged-token count happens in `crate::vision::attach`'s
/// splice, which needs the preprocessed grid and therefore the GPU. So an
/// image turn is undercounted here by roughly a page's worth of positions,
/// and a conversation this says fits can still be refused by `clamp_max_new`.
/// That refusal is loud and carries both numbers, which is why the undercount
/// is documented rather than paid for on every keystroke.
pub(crate) fn fit_window(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
    max_tokens: u32,
) -> Result<crate::wire::WindowFitOutcome, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;

    let bound = if max_tokens == 0 {
        session.max_context as u64
    } else {
        max_tokens as u64
    };

    let has_leading_instruction = messages
        .first()
        .map(|m| m.role == "system" || m.role == "developer")
        .unwrap_or(false);

    let measure = |slice: &[WireMessage]| -> u64 {
        match render(&session.tokenizer, slice, reasoning) {
            Ok((ids, _)) => ids.len() as u64,
            Err(_) => u64::MAX,
        }
    };

    let outcome =
        window_fit::fit_conversation_window(messages, has_leading_instruction, bound, measure);

    Ok(crate::wire::WindowFitOutcome {
        retained: outcome.retained_turns().to_vec(),
        measured_tokens: outcome.measured_length(),
        removed_turn_count: outcome.removed_turn_count(),
        has_room_for_generation: outcome.has_room_for_generation(),
    })
}

/// Formats a conversation transcript into raw prompt text using the session's
/// chat template and reasoning effort setting.
pub(crate) fn render_prompt(
    session: &Session,
    messages: &[WireMessage],
    reasoning_str: &str,
) -> Result<String, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;
    let decoded = decode_messages(messages)?;
    session
        .tokenizer
        .apply_chat_template_with_reasoning(&decoded, reasoning)
        .map_err(|e| format!("chat template: {e}"))
}

/// Encodes raw text into token IDs using the session tokenizer.
pub(crate) fn tokenize(session: &Session, text: &str, add_special: bool) -> Vec<i32> {
    session.tokenizer.encode(text, add_special)
}

/// Decodes token IDs into a text string using the session tokenizer.
pub(crate) fn detokenize(session: &Session, token_ids: &[i32], skip_special: bool) -> String {
    session.tokenizer.decode(token_ids, skip_special)
}

/// Runs one turn, calling `emit(kind, text, a, b)` per event.
pub(crate) fn generate(
    session: &Session,
    messages: &[WireMessage],
    options: &GenerateOptions,
    mut emit: impl FnMut(i32, &str, u32, u32),
) -> Result<GenerateResult, String> {
    let reasoning = ReasoningEffort::parse(&options.reasoning)
        .ok_or_else(|| format!("unknown reasoning level {:?}", options.reasoning))?;
    let (rendered_ids, _note) = render(&session.tokenizer, messages, reasoning)?;
    // Shape-checked before the lock, decoded inside it (see `attach_images`).
    let image_parts = collect_image_parts(messages)?;

    // Armed BEFORE the lock is taken, so a Stop pressed between turns cannot
    // cancel the next one before it has produced a token.
    let cancel = session.arm();
    let mut engine = session
        .engine
        .lock()
        .map_err(|_| "the session is poisoned by an earlier panic".to_string())?;

    // **THE PROMPT IS NOT FINAL UNTIL THE SPLICE HAS RUN.** The template
    // renders ONE `<|image_pad|>` per image whatever its size, and
    // `splice_and_walk` expands each marker to that image's merged-token
    // count -- so the budget clamp and every length below have to be taken
    // AFTER this, or a page's worth of positions is missing from both.
    let prompt_ids = attach_images(&mut engine, session, &rendered_ids, &image_parts)?;
    let max_new = clamp_max_new(session, options.max_new_tokens, prompt_ids.len())?;

    let config = GenerationConfig {
        shaping: session.shaping(options)?,
        max_new_tokens: max_new,
        stop_strings: options.stop.clone(),
        extra_stop_tokens: options
            .stop_tokens
            .iter()
            .map(|&t| t as foundation::TokenId)
            .collect(),
        rate: session.rate,
    };

    let mut content = String::new();
    let mut reasoning_text = String::new();
    let mut split = ChannelSplit::new(&session.tokenizer, reasoning, &prompt_ids);
    let total = prompt_ids.len() as u32;

    let mut on_progress = |event: RawDecodeProgress| {
        let (id, text) = match event {
            RawDecodeProgress::Token { id, delta, .. } => (id, delta),
            // A withheld tail has no token id behind it, which is what the
            // tokenizer's "no such token" sentinel means.
            RawDecodeProgress::Tail(tail) => (tokenizer::NO_SUCH_TOKEN_ID, tail),
            RawDecodeProgress::Prefill { done, .. } => {
                emit(TS_EVENT_PREFILL, "", done as u32, total);
                return;
            }
        };
        // NO EARLY RETURN ON EMPTY TEXT. Special tokens decode to the empty
        // string, so every Harmony frame token arrives as `(id, "")` -- skip
        // them and the state machine never sees a single `<|channel|>` and
        // the whole turn reads as one run of content.
        let (answer, reason) = split.push(id, &text);
        if !reason.is_empty() {
            reasoning_text.push_str(&reason);
            emit(TS_EVENT_REASONING, &reason, 0, 0);
        }
        if !answer.is_empty() {
            content.push_str(&answer);
            emit(TS_EVENT_CONTENT, &answer, 0, 0);
        }
    };

    let predicate = || cancel.load(Ordering::Acquire);
    // Resolved BEFORE the mutable borrow below. Reading it inside the call's
    // argument list is E0502: `runner.as_mut()` is already a mutable borrow
    // by the time the width argument is evaluated. Same shape as the
    // re-binding rule the family flows carry.
    //
    // The width is the MODEL's, never the tokenizer's:
    // `MfTokenizer::vocab_size` is a per-DIALECT constant standing in for a
    // padded head width, which is right only while one model uses each
    // dialect (ChatML's row is Qwen 3.6's 248,320 and Qwen3-30B-A3B is also
    // ChatML at 151,936).
    let vocab_size = match &*engine {
        #[cfg(target_os = "macos")]
        Engine::Real(runner) => runner.vocab_size(),
        Engine::Scripted(_) => session.info.vocab_size,
    };
    let block = turn_block(session.speculation_block, config.shaping.is_deterministic());
    let decoded: Result<RawDecodeResult, _> = match (&mut *engine, block) {
        // The CONCRETE runner, which is why this sits inside the match:
        // `SpeculativeProducer` has an associated type and cannot be
        // reached through a `&mut dyn LogitProducer`.
        #[cfg(target_os = "macos")]
        (Engine::Real(runner), Some(block)) => runtime::run_raw_completion_speculative_cancellable(
            runner.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            block,
            &predicate,
            &mut on_progress,
        ),
        #[cfg(target_os = "macos")]
        (Engine::Real(runner), None) => runtime::run_raw_completion_cancellable(
            runner.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            &predicate,
            &mut on_progress,
        ),
        // A scripted producer implements no drafter, so `block` is always
        // `None` here and the arm is a plain wildcard rather than a case
        // this could get wrong.
        (Engine::Scripted(producer), _) => runtime::run_raw_completion_cancellable(
            producer.as_mut(),
            &session.tokenizer,
            &prompt_ids,
            &config,
            session.max_context,
            vocab_size,
            &predicate,
            &mut on_progress,
        ),
    };

    // **THE CALLER CONSUMES THE INJECTION MAP, and it is cleared on the
    // FAILURE path too.** `reset()` deliberately no longer clears it
    // (`crates/cli` Gotcha 13: clearing at the generation loop's entry landed
    // on the map for the very prompt about to be prefilled, so every image
    // run prefilled placeholder embeddings and read perfectly fluently off a
    // picture the model had not been shown). So it ends HERE, before the `?`,
    // or the next turn on this session inherits this turn's spans.
    if !image_parts.is_empty() {
        clear_vision(&mut engine);
    }
    let result: RawDecodeResult = decoded.map_err(|e| e.to_string())?;

    Ok(GenerateResult {
        prompt_tokens: result.prompt_tokens,
        new_tokens: result.new_tokens,
        prefill_seconds: result.prefill_seconds,
        decode_seconds: result.decode_seconds,
        stop_reason: stop_reason_name(result.reason).to_string(),
        // Null rather than zero when nothing was decoded, so a caller cannot
        // plot a rate that was never measured.
        tokens_per_second: (result.decode_seconds > 0.0 && result.new_tokens > 0)
            .then(|| result.new_tokens as f64 / result.decode_seconds),
        content,
        reasoning: reasoning_text,
        peak_memory_pressure: format!("{:?}", result.peak_memory_pressure).to_lowercase(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The per-turn half of the speculation decision, in all four states.
    ///
    /// Cheap enough to be exhaustive, and worth being: three of the four
    /// cells are "decode sequentially" and the one that is not is the only
    /// path in this crate that reaches a batched verify.
    #[test]
    fn a_turn_speculates_only_when_the_session_can_and_the_turn_is_greedy() {
        assert_eq!(turn_block(Some(2), true), Some(2));
        // Sampled: the session's block is DISCARDED rather than honoured,
        // because acceptance is exact only at temperature 0.
        assert_eq!(turn_block(Some(2), false), None);
        // No drafter: greedy does not conjure one.
        assert_eq!(turn_block(None, true), None);
        assert_eq!(turn_block(None, false), None);
    }

    fn parse(json: &str) -> Vec<WireMessage> {
        serde_json::from_str(json).expect("wire messages parse")
    }

    /// **THE REGRESSION GUARD FOR WIDENING `content`.** Every caller that
    /// predates images sends a bare string, and `untagged` is what keeps that
    /// decoding to the same thing. A parts-only shape would have been an ABI
    /// break dressed as a field.
    #[test]
    fn a_bare_string_content_still_decodes_and_reserializes_as_one() {
        let messages = parse(r#"[{"role":"user","content":"hi"}]"#);
        assert_eq!(messages[0].text(), "hi");
        assert!(messages[0].parts().is_none());
        assert_eq!(messages[0].image_parts().count(), 0);
        // Round-trips as a STRING, not as an object: `WindowFitOutcome`
        // hands `retained` straight back to the caller, so a re-serialized
        // message that changed shape would break every existing consumer.
        let back = serde_json::to_string(&messages).expect("serializes");
        assert!(back.contains(r#""content":"hi""#), "{back}");
    }

    /// A missing `content` is still the empty string rather than an error,
    /// which is what `#[serde(default)]` meant before this widening too.
    #[test]
    fn an_absent_content_is_the_empty_string() {
        let messages = parse(r#"[{"role":"user"}]"#);
        assert_eq!(messages[0].text(), "");
        assert!(messages[0].parts().is_none());
    }

    /// Parts keep the order they arrived in, and `text()` sees only the
    /// prose. Order is what pairs the nth picture with the nth marker run.
    #[test]
    fn ordered_parts_decode_in_order_and_text_skips_the_images() {
        let messages = parse(
            r#"[{"role":"user","content":[
                 {"type":"image","path":"/a.png"},
                 {"type":"text","text":"one"},
                 {"type":"image","base64":"QQ=="},
                 {"type":"text","text":"two"}]}]"#,
        );
        assert_eq!(messages[0].text(), "onetwo");
        assert_eq!(messages[0].parts().expect("parts").len(), 4);

        let images: Vec<_> = collect_image_parts(&messages).expect("shapes are valid");
        assert_eq!(images.len(), 2);
        // The FIRST image is the path one, because that is the order sent.
        match images[0].image_source().expect("a source") {
            crate::wire::ImageSource::Path(p) => assert_eq!(p, "/a.png"),
            crate::wire::ImageSource::Base64(_) => panic!("images were reordered"),
        }
        match images[1].image_source().expect("a source") {
            crate::wire::ImageSource::Base64(b) => assert_eq!(b, "QQ=="),
            crate::wire::ImageSource::Path(_) => panic!("images were reordered"),
        }
    }

    /// Images are collected ACROSS messages, still in order: a conversation
    /// can carry a picture in an earlier turn as well as the current one.
    #[test]
    fn image_parts_are_collected_across_messages_in_order() {
        let messages = parse(
            r#"[{"role":"user","content":[{"type":"image","path":"/first.png"}]},
                {"role":"assistant","content":"ok"},
                {"role":"user","content":[{"type":"image","path":"/second.png"}]}]"#,
        );
        let images = collect_image_parts(&messages).expect("shapes are valid");
        let paths: Vec<_> = images
            .iter()
            .map(|i| match i.image_source().expect("a source") {
                crate::wire::ImageSource::Path(p) => p.to_string(),
                crate::wire::ImageSource::Base64(_) => unreachable!(),
            })
            .collect();
        assert_eq!(paths, ["/first.png", "/second.png"]);
    }

    /// Both spellings, or neither, is REFUSED rather than resolved by
    /// precedence -- and refused BEFORE the engine lock, which is what makes
    /// it cheap. A caller that sent both meant one of them, and picking
    /// silently runs the wrong picture with no error anywhere.
    #[test]
    fn an_image_part_needs_exactly_one_source() {
        let both =
            parse(r#"[{"role":"user","content":[{"type":"image","path":"/a","base64":"QQ=="}]}]"#);
        let err = collect_image_parts(&both).expect_err("both sources is refused");
        assert!(err.contains("both"), "{err}");

        let neither = parse(r#"[{"role":"user","content":[{"type":"image"}]}]"#);
        let err = collect_image_parts(&neither).expect_err("no source is refused");
        assert!(err.contains("neither"), "{err}");
    }

    /// A conversation with no image parts collects nothing, which is the
    /// condition every text-only turn takes and the one that keeps its path
    /// byte-identical.
    #[test]
    fn a_text_only_conversation_collects_no_images() {
        let messages = parse(
            r#"[{"role":"user","content":"hi"},
                {"role":"user","content":[{"type":"text","text":"still text"}]}]"#,
        );
        assert!(collect_image_parts(&messages)
            .expect("no shapes to get wrong")
            .is_empty());
    }
}
