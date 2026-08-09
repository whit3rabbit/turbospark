//! Request parsing and prompt planning for `/v1/chat/completions` and `/v1/messages`.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use anyllm_translate::openai::{
    ChatCompletionRequest, ChatMessage, ChatRole, ChatTool, Stop, ToolCall,
};
use runtime::GenerationConfig;
use selection::ShapingConfig;
use tokenizer::{FunctionDefinition, HistoricalToolCall, JsonValue, Message, Role};

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
    // top_k defaults to the CLI's 64 rather than 0 (unbounded): `ShapingConfig`
    // rejects a top_p below 1.0 when top_k is 0, so a plain OpenAI request
    // carrying only top_p would otherwise be a 400.
    let shaping = ShapingConfig::new(
        request.temperature.map(f64::from).unwrap_or(1.0),
        64,
        request.top_p.map(f64::from),
        1.0,
        // `seed` has no field on the OpenAI request type: it needs no
        // translation, so it lands in the `extra` flatten map. Reading it back
        // out is not optional -- a missing lookup here silently unseeds every
        // seeded request, with no deserialization error to catch it.
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
/// Kept out of [`plan`]'s return so its signature stays a pair; both call
/// sites need one line either way.
pub(crate) fn tool_names(request: &ChatCompletionRequest) -> HashSet<String> {
    request
        .tools
        .iter()
        .flatten()
        .map(|t| t.function.name.clone())
        .collect()
}

/// Renders the chat template, encodes it, and resolves the shaping config.
pub(crate) fn plan(
    model: &AppState,
    request: &ChatCompletionRequest,
) -> Result<(Vec<foundation::TokenId>, GenerationConfig), String> {
    let messages: Vec<Message> = request.messages.iter().filter_map(to_message).collect();
    let tools: Vec<FunctionDefinition> = request
        .tools
        .iter()
        .flatten()
        .map(tool_definition)
        .collect();

    // With tools, the checkpoint's own `chat_template.jinja` is the only
    // renderer that can express them (it already speaks OpenAI's shape:
    // `tool_calls` on an assistant turn, a forward scan of `tool` turns).
    // Without them, the text-only path stays exactly as it was.
    let prompt_ids = if tools.is_empty() {
        let prompt = model
            .tokenizer()
            .apply_chat_template(&messages)
            .map_err(|e| e.to_string())?;
        // `add_bos` is false on purpose: the Gemma template emits the literal
        // `<bos>` mark itself, so encoding with a BOS prefix would double it
        // (the CLI's proven path does the same).
        model.tokenizer().encode(&prompt, false)
    } else {
        model
            .tokenizer()
            .encode_generic_tool_chat(&messages, &tools, false)
            .map_err(|e| e.to_string())?
    };

    Ok((prompt_ids, build_config(request)?))
}
