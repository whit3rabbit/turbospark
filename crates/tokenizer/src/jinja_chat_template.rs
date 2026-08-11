//! Generic Jinja-templated chat/tool-chat rendering, using the checkpoint's
//! own installed template (the `swift-transformers`
//! `applyChatTemplate(messages:tools:...)` path, ported using the `minijinja`
//! crate as the Jinja engine instead of hand-rolling one).
//!
//! This is the primary path for PLAIN TEXT chat too, not just tool chat:
//! `MfTokenizer::apply_chat_template` routes here whenever the checkpoint
//! ships a template, in either of HF's two conventions (a standalone
//! `chat_template.jinja` or the older `chat_template` key inside
//! `tokenizer_config.json`). A checkpoint that ships neither falls back to
//! `chat_template.rs`'s per-dialect renderers; DeepSeek keeps its
//! hand-rolled native tool chat there for the same reason.
//!
//! `raise_exception` (a function HF's own Python Jinja environment injects
//! for chat templates to call on malformed input) is registered manually,
//! since `minijinja` has no built-in equivalent.

use minijinja::{Environment, ErrorKind};
use serde_json::{json, Value as JsonValue};

use crate::chat_template::{FunctionDefinition, HistoricalToolCall, Message};
use crate::dialect::MfTokenizer;
use crate::error::TokenizerError;

fn raise_exception(message: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

fn message_to_json(message: &Message) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert("role".to_string(), json!(message.role.as_str()));
    obj.insert("content".to_string(), json!(message.content));
    if let Some(id) = &message.tool_call_id {
        obj.insert("tool_call_id".to_string(), json!(id));
    }
    if let Some(name) = &message.name {
        obj.insert("name".to_string(), json!(name));
    }
    if !message.tool_calls.is_empty() {
        obj.insert(
            "tool_calls".to_string(),
            JsonValue::Array(message.tool_calls.iter().map(tool_call_to_json).collect()),
        );
    }
    JsonValue::Object(obj)
}

fn tool_call_to_json(call: &HistoricalToolCall) -> JsonValue {
    json!({
        "id": call.id,
        "type": "function",
        "function": {
            "name": call.name,
            "arguments": call.arguments.to_serde_json(),
        }
    })
}

fn tool_to_json(tool: &FunctionDefinition) -> JsonValue {
    json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters.to_serde_json(),
        }
    })
}

/// Renders `tokenizer`'s installed `chat_template.jinja` over `messages`
/// and `tools`, matching the shape (keys, defaults) HF's own chat-template
/// call site passes: `messages`, `tools` (only when non-empty; absent
/// otherwise, so a template's `{% if tools %}` branch behaves the same way
/// it does upstream), `add_generation_prompt`, `enable_thinking`,
/// `bos_token`, `eos_token`, and `add_vision_id` (always `false` — this
/// port has no vision input path).
pub fn render_generic_chat_template(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    tools: &[FunctionDefinition],
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Result<String, TokenizerError> {
    let source = tokenizer.chat_template_source.as_deref().ok_or_else(|| {
        TokenizerError::InvalidChatTemplate(
            "no chat_template.jinja is installed for this tokenizer".to_string(),
        )
    })?;

    let mut env = Environment::new();
    env.add_function("raise_exception", raise_exception);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", source)
        .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;

    let mut context = serde_json::Map::new();
    context.insert(
        "messages".to_string(),
        JsonValue::Array(messages.iter().map(message_to_json).collect()),
    );
    if !tools.is_empty() {
        context.insert(
            "tools".to_string(),
            JsonValue::Array(tools.iter().map(tool_to_json).collect()),
        );
    }
    context.insert(
        "add_generation_prompt".to_string(),
        json!(add_generation_prompt),
    );
    context.insert("enable_thinking".to_string(), json!(enable_thinking));
    context.insert("add_vision_id".to_string(), json!(false));
    context.insert(
        "bos_token".to_string(),
        json!(tokenizer.id_to_token(tokenizer.bos_id)),
    );
    context.insert(
        "eos_token".to_string(),
        json!(tokenizer.id_to_token(tokenizer.eos_id)),
    );

    let template = env
        .get_template("chat")
        .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;
    template
        .render(JsonValue::Object(context))
        .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))
}

impl MfTokenizer {
    /// Encodes `messages`/`tools` through the installed Jinja
    /// `chat_template.jinja`, then tokenizes the rendered text. Returns
    /// [`TokenizerError::InvalidChatTemplate`] if no template is installed
    /// or if rendering fails (a malformed input the template itself
    /// rejects via `raise_exception`, or a template feature `minijinja`
    /// does not support).
    pub fn encode_generic_tool_chat(
        &self,
        messages: &[Message],
        tools: &[FunctionDefinition],
        enable_thinking: bool,
    ) -> Result<Vec<i32>, TokenizerError> {
        let rendered = render_generic_chat_template(self, messages, tools, true, enable_thinking)?;
        Ok(self.encode(&rendered, false))
    }
}
