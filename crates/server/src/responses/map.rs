//! Request translation and warning checks for the OpenAI Responses endpoint.
//!
//! Maps `ResponsesRequest` onto the shared `ChatCompletionRequest` that
//! `handler::plan` shapes and renders, and computes the `x-anyllm-degradation`
//! warning header values (see `crates/server/CLAUDE.md` Gotcha 24).

use anyllm_translate::openai::chat_completions::{NamedFunction, NamedToolChoice};
use anyllm_translate::openai::responses::{ResponsesInput, ResponsesRequest};
use anyllm_translate::openai::{
    ChatCompletionRequest, ChatContent, ChatMessage, ChatRole, ChatTool, ChatToolChoice,
    FunctionCall, FunctionDef, Stop, ToolCall,
};
use serde_json::Value;
use tokenizer::{ChatDialect, ReasoningEffort};

use crate::handler::AppState;

pub(crate) fn chat_message(
    role: ChatRole,
    content: Option<String>,
    tool_calls: Vec<ToolCall>,
    tool_call_id: Option<String>,
) -> ChatMessage {
    ChatMessage {
        role,
        content: content.map(ChatContent::Text),
        name: None,
        tool_calls: (!tool_calls.is_empty()).then_some(tool_calls),
        tool_call_id,
        refusal: None,
        reasoning_content: None,
        thinking_blocks: None,
    }
}

/// `content` on a Responses input item is a plain string OR an array of
/// typed parts (`input_text` / `output_text` / `input_image` / ...). Only
/// the text-bearing parts are read; an image part has no `text` field and is
/// skipped by construction rather than by name -- this endpoint has no
/// vision path in v1, the same scope `/v1/completions` has none of tools.
pub(crate) fn item_text(item: &Value) -> Option<String> {
    match item.get("content") {
        Some(Value::String(s)) => Some(s.clone()),
        Some(Value::Array(parts)) => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
                .collect();
            (!texts.is_empty()).then(|| texts.join("\n"))
        }
        _ => None,
    }
}

/// One Responses input item to one `ChatMessage`. Unlike Chat Completions, a
/// tool call and its result are ROOT-LEVEL items here (`function_call`,
/// `function_call_output`), not content blocks on a message -- the same
/// flattened shape `anyllm_translate`'s own Anthropic<->Responses mapping
/// uses (`responses_message_map::convert_blocks_to_items`), read here rather
/// than re-derived (see `crates/server/CLAUDE.md` Gotcha 24).
pub(crate) fn item_to_message(item: &Value) -> Result<ChatMessage, String> {
    let item_type = item.get("type").and_then(|v| v.as_str()).unwrap_or("");
    match item_type {
        "message" => {
            let role = match item.get("role").and_then(|v| v.as_str()) {
                Some("user") => ChatRole::User,
                Some("assistant") => ChatRole::Assistant,
                Some("system") => ChatRole::System,
                Some("developer") => ChatRole::Developer,
                other => {
                    return Err(format!(
                        "unsupported message role in a Responses input item: {other:?}"
                    ))
                }
            };
            Ok(chat_message(role, item_text(item), Vec::new(), None))
        }
        "function_call" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let name = item
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let arguments = item
                .get("arguments")
                .and_then(|v| v.as_str())
                .unwrap_or("{}")
                .to_string();
            let call = ToolCall {
                id: call_id,
                call_type: "function".to_string(),
                function: FunctionCall { name, arguments },
            };
            Ok(chat_message(ChatRole::Assistant, None, vec![call], None))
        }
        "function_call_output" => {
            let call_id = item
                .get("call_id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            let output = item
                .get("output")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string();
            Ok(chat_message(
                ChatRole::Tool,
                Some(output),
                Vec::new(),
                Some(call_id),
            ))
        }
        other => Err(format!(
            "unsupported Responses input item type {other:?}: expected message, function_call, \
             or function_call_output"
        )),
    }
}

/// A Responses tool is FLAT (`{"type":"function","name":...,"parameters":...}`)
/// where Chat Completions nests the function under its own key
/// (`{"type":"function","function":{"name":...}}`). Same information, one
/// extra layer (see `crates/server/CLAUDE.md` Gotcha 24).
pub(crate) fn flat_tool_to_chat_tool(tool: &Value) -> Result<ChatTool, String> {
    let tool_type = tool
        .get("type")
        .and_then(|v| v.as_str())
        .unwrap_or("function");
    if tool_type != "function" {
        return Err(format!(
            "unsupported Responses tool type {tool_type:?}: only function tools are supported"
        ));
    }
    let name = tool
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "a Responses tool is missing its name".to_string())?
        .to_string();
    let description = tool
        .get("description")
        .and_then(|v| v.as_str())
        .map(str::to_string);
    let parameters = tool.get("parameters").cloned();
    Ok(ChatTool {
        tool_type: "function".to_string(),
        function: FunctionDef {
            name,
            description,
            parameters,
            strict: None,
        },
    })
}

/// A Responses `tool_choice` forcing a specific function is FLAT
/// (`{"type":"function","name":"get_weather"}`), where Chat Completions
/// nests the name under its own `function` key
/// (`{"type":"function","function":{"name":"get_weather"}}`) --
/// `ChatToolChoice`'s own `Deserialize` fails on the flat shape (neither of
/// its two variants matches an object with no `function` key), which is
/// exactly how `serde_json::from_value(..).ok()` used to make a forced call
/// vanish with no error. Converted to the nested shape before falling back
/// to Chat Completions' own, so a client sending either spelling on this
/// endpoint gets the tool it asked for.
fn tool_choice_from_responses(value: Value) -> Result<ChatToolChoice, String> {
    if let Value::String(s) = &value {
        return Ok(ChatToolChoice::Simple(s.clone()));
    }
    if let Value::Object(obj) = &value {
        if obj.get("type").and_then(|t| t.as_str()) == Some("function") {
            if let Some(name) = obj.get("name").and_then(|n| n.as_str()) {
                return Ok(ChatToolChoice::Named(NamedToolChoice {
                    choice_type: "function".to_string(),
                    function: NamedFunction {
                        name: name.to_string(),
                    },
                }));
            }
        }
    }
    serde_json::from_value(value.clone()).map_err(|e| format!("invalid tool_choice ({e}): {value}"))
}

/// Folds a Responses request down onto the `ChatCompletionRequest`
/// `handler::plan` already renders and shapes -- one template path, one
/// shaping path, for all three generation endpoints this crate serves.
pub(crate) fn responses_to_chat_request(
    request: &ResponsesRequest,
) -> Result<ChatCompletionRequest, String> {
    // Stateless server, honest refusal: there is no PRIOR turn on disk to
    // continue, and silently starting a fresh conversation under the same
    // id would answer a different question than the one asked.
    if request.extra.contains_key("previous_response_id") {
        return Err(
            "previous_response_id is not supported: this server is stateless and keeps no prior \
             turn to continue"
                .to_string(),
        );
    }

    let mut messages = Vec::new();
    if let Some(instructions) = &request.instructions {
        messages.push(chat_message(
            ChatRole::System,
            Some(instructions.clone()),
            Vec::new(),
            None,
        ));
    }
    match &request.input {
        ResponsesInput::Text(text) => {
            messages.push(chat_message(
                ChatRole::User,
                Some(text.clone()),
                Vec::new(),
                None,
            ));
        }
        ResponsesInput::Items(items) => {
            for item in items {
                messages.push(item_to_message(item)?);
            }
        }
    }

    let tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(flat_tool_to_chat_tool)
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;

    // `top_p` / `tool_choice` / `stop` have no field on `ResponsesRequest`
    // (unlike Chat Completions', which has explicit ones), so they arrive in
    // Responses' OWN extra map. Pulled out here into the explicit fields
    // `handler::plan` and `build_config` read, and removed from what is
    // forwarded so nothing reads a Responses-shaped value through the wrong
    // key twice.
    //
    // Each is PROPAGATED rather than `.ok()`-swallowed: a value that fails to
    // parse is a malformed request, not one silently missing the field --
    // `.ok()` here used to mean a caller's forced `tool_choice` was dropped
    // with no error anywhere, and the model answered in prose instead of
    // calling the tool it was told to.
    let mut extra = request.extra.clone();
    let top_p = match extra.remove("top_p") {
        Some(v) => Some(
            v.as_f64()
                .map(|f| f as f32)
                .ok_or_else(|| format!("top_p must be a number, got {v}"))?,
        ),
        None => None,
    };
    let stop = match extra.remove("stop") {
        Some(v) => Some(
            serde_json::from_value::<Stop>(v.clone())
                .map_err(|e| format!("invalid stop ({e}): {v}"))?,
        ),
        None => None,
    };
    let tool_choice = match extra.remove("tool_choice") {
        Some(v) => Some(tool_choice_from_responses(v)?),
        None => None,
    };

    Ok(ChatCompletionRequest {
        model: request.model.clone(),
        messages,
        max_tokens: request.max_output_tokens,
        max_completion_tokens: None,
        temperature: request.temperature,
        top_p,
        stop,
        tools,
        tool_choice,
        stream: request.stream,
        stream_options: None,
        presence_penalty: None,
        frequency_penalty: None,
        response_format: None,
        user: None,
        parallel_tool_calls: None,
        // `top_k`, `repetition_penalty`, `seed`, `reasoning_effort`, `n`,
        // `logprobs`, and any other Responses `extra` field this server
        // reads by name pass through here unchanged -- the same map
        // `build_shaping` / `openai_request_warnings`-style readers expect.
        extra,
    })
}

/// A checkpoint whose dialect would separate reasoning from its answer on
/// THIS request -- the narrower question the streaming path needs
/// (`handler::exec::needs_decoder`'s own condition is broader: it also fires
/// on tools alone, which forces decoding for call-parsing but does not mean
/// reasoning was produced).
pub(crate) fn may_produce_reasoning(model: &AppState, effort: ReasoningEffort) -> bool {
    let dialect = model.tokenizer().dialect;
    matches!(dialect, ChatDialect::Harmony | ChatDialect::MuseGlimmer)
        || (effort != ReasoningEffort::Off
            && matches!(dialect, ChatDialect::ChatMl | ChatDialect::Gemma))
}

pub(crate) fn responses_warnings(
    request: &ResponsesRequest,
    reasoning_dropped: bool,
) -> Option<String> {
    let mut warnings = anyllm_translate::TranslationWarnings::default();
    if request
        .extra
        .get("store")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        warnings.add("store");
    }
    if reasoning_dropped {
        warnings.add("reasoning");
    }
    warnings.as_header_value()
}
