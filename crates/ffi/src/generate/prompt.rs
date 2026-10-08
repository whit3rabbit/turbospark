//! Message decoding, chat template rendering, tokenization, and context window fitting.

use std::collections::HashSet;

use tokenizer::{
    ContentPart, FunctionDefinition, HistoricalToolCall, JsonValue, Message, MfTokenizer,
    ReasoningEffort, ReasoningSupport, Role,
};

use crate::session::Session;
use crate::wire::{WireMessage, WirePart, WireToolCall, WireToolSpec};

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
            let mut message = match m.parts() {
                None => {
                    let text = m.text();
                    // An assistant turn that is ONLY a tool call has no text
                    // at all. `None` rather than an empty string, because the
                    // templates branch on presence and an empty block would
                    // render where the server's own path renders nothing.
                    if text.is_empty() && !m.tool_calls.is_empty() {
                        Message {
                            content: None,
                            ..Message::new(role, String::new())
                        }
                    } else {
                        Message::new(role, text)
                    }
                }
                Some(parts) => {
                    let mapped: Vec<ContentPart> = parts
                        .iter()
                        .map(|p| match p {
                            WirePart::Text { text } => ContentPart::Text(text.clone()),
                            WirePart::Image { .. } => ContentPart::Image,
                        })
                        .collect();
                    Message::with_parts(role, mapped)
                }
            };
            message.tool_calls = m.tool_calls.iter().map(historical_call).collect();
            message.tool_call_id = m.tool_call_id.clone();
            message.name = m.name.clone();
            Ok(message)
        })
        .collect()
}

/// A replayed tool call, its arguments as the template wants them.
///
/// An object (or any JSON value) is carried as that value. A string is
/// treated as the JSON text OpenAI carries arguments in and parsed back out;
/// one that will not parse is passed through as the string it is, because the
/// Gemma template renders `arguments` as a mapping OR as a bare string, so it
/// still renders -- the same rule the server applies.
fn historical_call(call: &WireToolCall) -> HistoricalToolCall {
    let arguments = match &call.arguments {
        serde_json::Value::String(text) => {
            JsonValue::parse(text).unwrap_or_else(|_| JsonValue::String(text.clone()))
        }
        serde_json::Value::Null => JsonValue::Object(Default::default()),
        other => JsonValue::parse(&other.to_string())
            .unwrap_or_else(|_| JsonValue::String(other.to_string())),
    };
    HistoricalToolCall {
        id: call.id.clone(),
        name: call.name.clone(),
        arguments,
    }
}

/// The functions offered for one turn: the definitions the template renders
/// and the names the reply decoder will accept a call to.
#[derive(Debug)]
pub(crate) struct ToolOffer {
    pub(crate) definitions: Vec<FunctionDefinition>,
    pub(crate) names: HashSet<String>,
}

impl ToolOffer {
    pub(crate) fn none() -> Self {
        Self {
            definitions: Vec::new(),
            names: HashSet::new(),
        }
    }
}

/// Validates and converts the offered tools. Refused by name before any
/// engine work: an empty or repeated name would make a parsed call ambiguous,
/// and a `parameters` that is not a JSON object is not a schema.
pub(crate) fn tool_offer(specs: &[WireToolSpec]) -> Result<ToolOffer, String> {
    let mut offer = ToolOffer::none();
    for spec in specs {
        if spec.name.trim().is_empty() {
            return Err("tools: a tool needs a non-empty name".to_string());
        }
        if !offer.names.insert(spec.name.clone()) {
            return Err(format!("tools: {:?} is offered more than once", spec.name));
        }
        let parameters = match &spec.parameters {
            None | Some(serde_json::Value::Null) => JsonValue::Null,
            Some(value @ serde_json::Value::Object(_)) => JsonValue::parse(&value.to_string())
                .map_err(|_| {
                    format!(
                        "tools: {:?} has parameters that are not valid JSON",
                        spec.name
                    )
                })?,
            Some(_) => {
                return Err(format!(
                    "tools: {:?} parameters must be a JSON Schema object",
                    spec.name
                ))
            }
        };
        offer.definitions.push(FunctionDefinition {
            name: spec.name.clone(),
            description: spec.description.clone().unwrap_or_default(),
            parameters,
        });
    }
    Ok(offer)
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
    tools: &[FunctionDefinition],
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
    if !tools.is_empty() {
        // With tools, the checkpoint's own `chat_template.jinja` is the only
        // renderer that can express them -- the path `crates/server` takes.
        // It tokenizes the rendered text itself (no BOS prefix, same reason
        // as below).
        let ids = tokenizer
            .encode_generic_tool_chat(&decoded, tools, reasoning)
            .map_err(|e| format!("chat template (with tools): {e}"))?;
        return Ok((ids, note));
    }
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
    tools: &[FunctionDefinition],
) -> Result<u32, String> {
    let reasoning = ReasoningEffort::parse(reasoning_str)
        .ok_or_else(|| format!("unknown reasoning level {:?}", reasoning_str))?;
    let (prompt_ids, _note) = render(&session.tokenizer, messages, reasoning, tools)?;
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
    tools: &[FunctionDefinition],
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
        // The offered tools are part of every slice's prompt, so a budget that
        // ignored them would keep turns the real prompt has no room for.
        match render(&session.tokenizer, slice, reasoning, tools) {
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
