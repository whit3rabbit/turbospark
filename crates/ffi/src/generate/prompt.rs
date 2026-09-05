//! Message decoding, chat template rendering, tokenization, and context window fitting.

use tokenizer::{ContentPart, Message, MfTokenizer, ReasoningEffort, ReasoningSupport, Role};

use crate::session::Session;
use crate::wire::{WireMessage, WirePart};

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
pub(crate) fn collect_image_parts(messages: &[WireMessage]) -> Result<Vec<&WirePart>, String> {
    let parts: Vec<&WirePart> = messages.iter().flat_map(|m| m.image_parts()).collect();
    for part in &parts {
        part.image_source()?;
    }
    Ok(parts)
}

/// Renders the conversation and encodes it.
///
/// The checkpoint's own chat template, never a raw concatenation: an
/// instruction-tuned model fed unrendered text babbles, and that is missing
/// markup rather than a decode bug.
pub(crate) fn render(
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
