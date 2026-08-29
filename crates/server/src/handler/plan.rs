//! Request parsing and prompt planning for `/v1/chat/completions` and `/v1/messages`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyllm_translate::openai::{
    ChatCompletionRequest, ChatMessage, ChatRole, ChatTool, ChatToolChoice, Stop, ToolCall,
};
use runtime::GenerationConfig;
use selection::ShapingConfig;
use tokenizer::{
    FunctionDefinition, HistoricalToolCall, JsonValue, Message, ReasoningEffort, Role,
};

use crate::model::ChatModel;

pub type AppState = Arc<dyn ChatModel>;

pub(crate) fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn role_from(role: &ChatRole) -> Role {
    match role {
        ChatRole::System => Role::System,
        ChatRole::Assistant => Role::Assistant,
        // `function` is OpenAI's deprecated spelling of a tool result.
        ChatRole::Tool | ChatRole::Function => Role::Tool,
        ChatRole::Developer => Role::Developer,
        ChatRole::User => Role::User,
    }
}

/// `stop` is untagged on the wire: a bare string or an array.
fn stop_strings(stop: Option<&Stop>) -> Vec<String> {
    match stop {
        Some(Stop::Single(s)) => vec![s.clone()],
        Some(Stop::Multiple(v)) => v.clone(),
        None => Vec::new(),
    }
}

fn build_config(request: &ChatCompletionRequest) -> Result<GenerationConfig, String> {
    // top_k defaults to 64 if unspecified, but can be overridden via `top_k` in extra.
    let top_k = request
        .extra
        .get("top_k")
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .unwrap_or(64);

    let repetition_penalty = request
        .extra
        .get("repetition_penalty")
        .and_then(|v| v.as_f64())
        .unwrap_or(1.0);

    let shaping = ShapingConfig::new(
        request.temperature.map(f64::from).unwrap_or(1.0),
        top_k,
        request.top_p.map(f64::from),
        repetition_penalty,
        // `seed` has no field on the OpenAI request type: it lands in the `extra` flatten map.
        request.extra.get("seed").and_then(|v| v.as_u64()),
    )
    .map_err(|e| e.to_string())?;

    Ok(GenerationConfig {
        shaping,
        max_new_tokens: request
            .max_tokens
            .or(request.max_completion_tokens)
            .unwrap_or(256),
        stop_strings: stop_strings(request.stop.as_ref()),
        extra_stop_tokens: Vec::new(),
        // Left at the default here and filled in by `plan` from the
        // backend: rate control is process-level on purpose, so there is
        // deliberately no per-request field to read off the wire.
        rate: Default::default(),
    })
}

/// The text `content` carries, with no fallback to `reasoning_content`.
///
/// `ChatMessage::effective_text` falls back to it, which is wrong here: a
/// replayed Anthropic `thinking` block would be rendered as ordinary
/// assistant prose, showing the model its own scratchpad as if it had said
/// it out loud. Everything else that method does (join the text parts of a
/// multipart content, skip images) is what this server wants.
fn visible_text(message: &ChatMessage) -> Option<String> {
    let mut stripped = message.clone();
    stripped.reasoning_content = None;
    stripped.effective_text()
}

/// A historical tool call, with its arguments parsed back out of the JSON
/// string OpenAI carries them in. A blob that will not parse is passed
/// through as a string rather than rejected: the Gemma template renders
/// `arguments` as a mapping OR as a bare string, so this still renders.
fn historical_call(call: &ToolCall) -> HistoricalToolCall {
    HistoricalToolCall {
        id: call.id.clone(),
        name: call.function.name.clone(),
        arguments: JsonValue::parse(&call.function.arguments)
            .unwrap_or_else(|_| JsonValue::String(call.function.arguments.clone())),
    }
}

/// One request message as a `tokenizer::Message`, or `None` if it carries
/// nothing to render (an image-only turn, say).
///
/// Every tool field is copied across. `tool_call_id` in particular is load
/// bearing: the Gemma template resolves a `tool` turn's function name by
/// matching that id against the preceding assistant message's `tool_calls`,
/// and renders `unknown` when it cannot.
fn to_message(message: &ChatMessage) -> Option<Message> {
    let content = visible_text(message);
    let tool_calls: Vec<HistoricalToolCall> = message
        .tool_calls
        .iter()
        .flatten()
        .map(historical_call)
        .collect();
    // An assistant turn that is ONLY a tool call has no content at all.
    // Dropping it would delete a turn from the middle of the history and
    // break user/model alternation.
    if content.is_none() && tool_calls.is_empty() {
        return None;
    }
    Some(Message {
        role: role_from(&message.role),
        content,
        // EMPTY, and still empty at M-V6: this handler drops image parts a few
        // lines up (`visible_text`), which is M-V8's job to change. Filling it
        // here without that would build a prompt whose `<|image_pad|>` run has
        // no tower rows behind it.
        content_parts: Vec::new(),
        tool_calls,
        tool_call_id: message.tool_call_id.clone(),
        name: message.name.clone(),
    })
}

fn tool_definition(tool: &ChatTool) -> FunctionDefinition {
    FunctionDefinition {
        name: tool.function.name.clone(),
        description: tool.function.description.clone().unwrap_or_default(),
        parameters: tool
            .function
            .parameters
            .as_ref()
            .and_then(|p| JsonValue::parse(&p.to_string()).ok())
            .unwrap_or(JsonValue::Null),
    }
}

/// The tool names the structured decoder will accept in generated output.
/// Respects `tool_choice`: "none" suppresses tools, named tool_choice restricts to that function.
pub(crate) fn tool_names(request: &ChatCompletionRequest) -> HashSet<String> {
    match &request.tool_choice {
        Some(ChatToolChoice::Simple(s)) if s == "none" => HashSet::new(),
        Some(ChatToolChoice::Named(n)) => {
            let mut set = HashSet::new();
            set.insert(n.function.name.clone());
            set
        }
        _ => request
            .tools
            .iter()
            .flatten()
            .map(|t| t.function.name.clone())
            .collect(),
    }
}

/// The reasoning level this request asked for.
///
/// **`reasoning_effort` HAS NO FIELD ON THE OpenAI REQUEST TYPE; it lands in
/// the `extra` flatten map**, the same place `seed` does (Gotcha 5), because
/// `anyllm_translate` gives explicit fields only to what needs translating.
/// So it arrives as a string here and a misspelling has to be REFUSED rather
/// than ignored: a request asking to think harder that quietly does not is a
/// wrong answer no client can see.
///
/// Per-REQUEST, unlike the rate control resolved once per process beside it
/// (Gotcha 10), and the contrast is the point: a power setting is a property
/// of the machine, while how hard to think is a property of the caller's
/// task. It is also what OpenAI's own API does with this field.
pub(crate) fn reasoning_effort(request: &ChatCompletionRequest) -> Result<ReasoningEffort, String> {
    let Some(value) = request.extra.get("reasoning_effort") else {
        return Ok(ReasoningEffort::Off);
    };
    let Some(text) = value.as_str() else {
        return Err(format!("reasoning_effort must be a string, got {value}"));
    };
    ReasoningEffort::parse(text).ok_or_else(|| {
        format!("unknown reasoning_effort {text:?}: expected off, low, medium, high or xhigh")
    })
}

/// Renders the chat template, encodes it, and resolves the shaping config.
pub(crate) fn plan(
    model: &AppState,
    request: &ChatCompletionRequest,
) -> Result<(Vec<foundation::TokenId>, GenerationConfig), String> {
    let messages: Vec<Message> = request.messages.iter().filter_map(to_message).collect();

    let is_none_choice =
        matches!(&request.tool_choice, Some(ChatToolChoice::Simple(s)) if s == "none");
    let tools: Vec<FunctionDefinition> = if is_none_choice {
        Vec::new()
    } else if let Some(ChatToolChoice::Named(n)) = &request.tool_choice {
        request
            .tools
            .iter()
            .flatten()
            .filter(|t| t.function.name == n.function.name)
            .map(tool_definition)
            .collect()
    } else {
        request
            .tools
            .iter()
            .flatten()
            .map(tool_definition)
            .collect()
    };

    // With tools, the checkpoint's own `chat_template.jinja` is the only
    // renderer that can express them (it already speaks OpenAI's shape:
    // `tool_calls` on an assistant turn, a forward scan of `tool` turns).
    // Without them, the text-only path stays exactly as it was.
    let reasoning = reasoning_effort(request)?;
    let prompt_ids = if tools.is_empty() {
        let prompt = model
            .tokenizer()
            .apply_chat_template_with_reasoning(&messages, reasoning)
            .map_err(|e| e.to_string())?;
        // `add_bos` is false on purpose: the Gemma template emits the literal
        // `<bos>` mark itself, so encoding with a BOS prefix would double it
        // (the CLI's proven path does the same).
        model.tokenizer().encode(&prompt, false)
    } else {
        model
            .tokenizer()
            .encode_generic_tool_chat(&messages, &tools, reasoning)
            .map_err(|e| e.to_string())?
    };

    let mut config = build_config(request)?;
    config.rate = model.rate_control();
    Ok((prompt_ids, config))
}
