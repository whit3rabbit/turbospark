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
use crate::jinja_compat::parenthesize_conditional_kwargs;
use crate::jinja_date::strftime_now;
pub use crate::jinja_date::CHAT_DATE_ENV;
use crate::reasoning::ReasoningEffort;

fn raise_exception(message: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

fn message_to_json(message: &Message) -> JsonValue {
    let mut obj = serde_json::Map::new();
    obj.insert("role".to_string(), json!(message.role.as_str()));
    // A MULTIMODAL message renders as HF's content-part LIST; a text one
    // renders as the bare string it always did (ROADMAP M-V6).
    //
    // The shape is the template's, not this port's invention: qwen3_5's
    // `render_content` macro branches on `content is string` first and falls
    // through to `content is iterable`, testing each item for an `image` key
    // or `item.type == 'image'`. Emitting `{"type": "image"}` hits that arm;
    // emitting a bare string with markup in it would put the markup through
    // the text arm and tokenize the ANGLE BRACKETS rather than the special
    // token.
    //
    // A text-only message takes the `content` branch byte for byte, which is
    // what leaves every frozen digest in `crates/bench` where it is.
    if message.content_parts.is_empty() {
        obj.insert("content".to_string(), json!(message.content));
    } else {
        let parts: Vec<JsonValue> = message
            .content_parts
            .iter()
            .map(|part| match part {
                crate::chat_template::ContentPart::Text(text) => {
                    json!({"type": "text", "text": text})
                }
                crate::chat_template::ContentPart::Image => json!({"type": "image"}),
            })
            .collect();
        obj.insert("content".to_string(), JsonValue::Array(parts));
    }
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
/// `bos_token`, `eos_token`, and `add_vision_id` (always `false`).
///
/// `add_vision_id` stays hardcoded now that a vision input path exists, and
/// that is Phase 0 item 6's finding rather than an omission: the flag only
/// controls an optional `"Picture N: "` text prefix before the marker run,
/// and it DEFAULTS to falsy when a caller sets nothing -- so `false` is
/// already what upstream sends for the unlabeled case. Threading a context
/// variable for it would change the rendered bytes of every image prompt to
/// match no reference.
///
/// `reasoning` adds the two effort keys ON TOP of that shape, and only when
/// a level is asked for: at [`ReasoningEffort::Off`] the context is exactly
/// what it was before this parameter existed, which is what keeps every
/// frozen digest in `crates/bench` where it is. See `reasoning.rs` for why
/// a level also flips `enable_thinking` and why both spellings are set.
pub fn render_generic_chat_template(
    tokenizer: &MfTokenizer,
    messages: &[Message],
    tools: &[FunctionDefinition],
    add_generation_prompt: bool,
    reasoning: ReasoningEffort,
) -> Result<String, TokenizerError> {
    let source = tokenizer.chat_template_source.as_deref().ok_or_else(|| {
        TokenizerError::InvalidChatTemplate(
            "no chat_template.jinja is installed for this tokenizer".to_string(),
        )
    })?;

    let source = parenthesize_conditional_kwargs(source);
    let source = source.as_ref();

    let mut env = Environment::new();
    env.add_function("raise_exception", raise_exception);
    // transformers' own chat-template global. gpt-oss's Harmony template is
    // the first here to call it (`Current date: ` in its system preamble);
    // without it the render fails with "value of type undefined is not
    // callable" and no generation happens at all.
    env.add_function("strftime_now", strftime_now);
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
    context.insert(
        crate::reasoning::THINKING_KEY.to_string(),
        json!(reasoning.enable_thinking()),
    );
    // ABSENT rather than null when no level is asked for: a template resolves
    // its key with `|default('xhigh')`, and a present-but-null key defeats
    // that default instead of taking it.
    if let Some(level) = reasoning.level() {
        for key in crate::reasoning::EFFORT_KEYS {
            context.insert(key.to_string(), json!(level));
        }
    }
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
        reasoning: ReasoningEffort,
    ) -> Result<Vec<i32>, TokenizerError> {
        let rendered = render_generic_chat_template(self, messages, tools, true, reasoning)?;
        Ok(self.encode(&rendered, false))
    }
}
