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

use std::io::{self, Write};

use minijinja::value::Kwargs;
use minijinja::{Environment, ErrorKind, Value};
use serde_json::{json, Value as JsonValue};

use crate::chat_template::{FunctionDefinition, HistoricalToolCall, Message};
use crate::dialect::MfTokenizer;
use crate::error::TokenizerError;
use crate::jinja_compat::{parenthesize_conditional_kwargs, rewrite_generation_tags};
use crate::jinja_date::strftime_now;
pub use crate::jinja_date::CHAT_DATE_ENV;
use crate::reasoning::ReasoningEffort;

// Templates come from model artifacts and are not trusted. Keep both limits
// comfortably above real checkpoint templates and prompts while preventing a
// small template from consuming unbounded CPU or memory before tokenization.
const TEMPLATE_FUEL: u64 = 1_000_000;
const MAX_RENDERED_BYTES: usize = 1 << 20;

struct BoundedOutput {
    bytes: Vec<u8>,
    exceeded: bool,
}

impl BoundedOutput {
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            exceeded: false,
        }
    }
}

impl Write for BoundedOutput {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.bytes.len().saturating_add(buf.len()) > MAX_RENDERED_BYTES {
            self.exceeded = true;
            return Err(io::Error::other("chat template output limit exceeded"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// `tojson`, accepting the `json.dumps` arguments `transformers` gives chat
/// templates (`sort_keys`, `separators`, `ensure_ascii`). Inkling 2's template
/// passes the first two when tools are declared, which minijinja's own filter
/// refuses, so every tool request failed to render.
///
/// **A call without those arguments still goes to minijinja's filter**, byte
/// for byte: it writes compact JSON and escapes `<>&'`, which is not what
/// Python writes, but rendered prompts and frozen digests depend on it. With
/// them, the output is Python's: no HTML escaping, non-ASCII kept, `", "` and
/// `": "` unless `separators` says otherwise.
fn tojson(value: &Value, indent: Option<Value>, args: Kwargs) -> Result<Value, minijinja::Error> {
    let python = ["sort_keys", "separators", "ensure_ascii"]
        .iter()
        .any(|k| args.has(k));
    if !python {
        return minijinja::filters::tojson(value, indent, args);
    }
    let refuse = |what: &str| {
        Err(minijinja::Error::new(
            ErrorKind::InvalidOperation,
            format!("tojson: {what}"),
        ))
    };
    if indent.is_some() || args.has("indent") {
        return refuse("indent cannot be combined with sort_keys or separators");
    }
    // The values `transformers` defaults to; the keys of a JSON context are
    // already ordered, so `sort_keys` only has to be honoured for a template's
    // own literals.
    let sort_keys = args.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    if args.get::<Option<bool>>("ensure_ascii")? == Some(true) {
        return refuse("ensure_ascii=true is not supported");
    }
    let (item, key) = match args.get::<Option<Value>>("separators")? {
        None => (", ".to_string(), ": ".to_string()),
        Some(pair) => {
            let parts: Vec<Value> = pair.try_iter()?.collect();
            match parts.as_slice() {
                [item, key] => (item.to_string(), key.to_string()),
                _ => return refuse("separators takes an (item, key) pair"),
            }
        }
    };
    args.assert_all_used()?;
    let json = serde_json::to_value(value).map_err(|e| {
        minijinja::Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON")
            .with_source(e)
    })?;
    let mut out = String::new();
    write_json(&json, &item, &key, sort_keys, &mut out);
    Ok(Value::from_safe_string(out))
}

fn write_json(value: &JsonValue, item: &str, key: &str, sort_keys: bool, out: &mut String) {
    match value {
        JsonValue::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                write_json(v, item, key, sort_keys, out);
            }
            out.push(']');
        }
        JsonValue::Object(map) => {
            let mut entries: Vec<(&String, &JsonValue)> = map.iter().collect();
            if sort_keys {
                entries.sort_by(|a, b| a.0.cmp(b.0));
            }
            out.push('{');
            for (i, (name, v)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push_str(item);
                }
                out.push_str(&JsonValue::String(name.clone()).to_string());
                out.push_str(key);
                write_json(v, item, key, sort_keys, out);
            }
            out.push('}');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}

fn raise_exception(message: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, message))
}

/// The added token a flattened image part spells.
const IMAGE_MARKER: &str = "<image>";

/// Whether a template only ever prints `content`, with no look at its shape.
///
/// Such a template renders a part list as its JSON, so the model reads
/// `[{"type": "image"}, ...]` as prose and answers fluently about nothing.
/// Unlimited-OCR-2's template is the carrier: its whole body is
/// `{{message['content']}}`, and upstream's processor hands it a string.
fn template_prints_content_raw(source: &str) -> bool {
    // `in message[`, not `in message`: every template opens with
    // `for message in messages`.
    const INSPECTS_CONTENT: [&str; 9] = [
        "is string",
        "is not string",
        "is iterable",
        "is sequence",
        "is mapping",
        "in message[",
        "in message.",
        "in content",
        "'type'",
    ];
    source.contains("content")
        && !source.contains("\"type\"")
        && !source.contains(".type")
        && !INSPECTS_CONTENT
            .iter()
            .any(|needle| source.contains(needle))
}

/// Whether a template indexes `content` as a part list with no string arm.
///
/// The Idefics3 / SmolVLM templates read `message['content'][0]['type']` and
/// loop over the parts, so a text-only turn handed over as a bare string
/// renders EMPTY. Their upstream processor wraps such a string as one text
/// part, and so does this.
fn template_requires_content_parts(source: &str) -> bool {
    source.contains("message['content'][0]['type']") && !source.contains("is string")
}

fn message_to_json(
    message: &Message,
    flatten_image_parts: bool,
    wrap_text_content: bool,
) -> JsonValue {
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
    if message.content_parts.is_empty() && wrap_text_content {
        let text = message.content.as_deref().unwrap_or_default();
        obj.insert(
            "content".to_string(),
            json!([{"type": "text", "text": text, "content": text}]),
        );
    } else if message.content_parts.is_empty() {
        obj.insert("content".to_string(), json!(message.content));
    } else if flatten_image_parts {
        // The exception to the paragraph above: this template has no part arm,
        // and this tokenizer has `<image>` as a real added token, so here the
        // markup does tokenize to the special token.
        let text: String = message
            .content_parts
            .iter()
            .map(|part| match part {
                crate::chat_template::ContentPart::Text(text) => text.as_str(),
                crate::chat_template::ContentPart::Image => IMAGE_MARKER,
            })
            .collect();
        obj.insert("content".to_string(), json!(text));
    } else {
        let parts: Vec<JsonValue> = message
            .content_parts
            .iter()
            .map(|part| match part {
                // `text` is the OpenAI and Qwen spelling. Pixtral's template
                // (`mistral-community/pixtral-12b`) reads `chunk["content"]`
                // for a text chunk instead, and an undefined key renders
                // empty: the question would vanish and the model would be
                // asked nothing about the picture. Both keys carry the same
                // string, which no other template iterates over.
                crate::chat_template::ContentPart::Text(text) => {
                    json!({"type": "text", "text": text, "content": text})
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
    let source = rewrite_generation_tags(source.as_ref());
    let source = source.as_ref();

    let mut env = Environment::new();
    // Model repositories control this source. Bound both VM work and output
    // so loading a tokenizer cannot turn a small sidecar into unbounded work.
    env.set_fuel(Some(TEMPLATE_FUEL));
    env.add_function("raise_exception", raise_exception);
    // transformers' own chat-template global. gpt-oss's Harmony template is
    // the first here to call it (`Current date: ` in its system preamble);
    // without it the render fails with "value of type undefined is not
    // callable" and no generation happens at all.
    env.add_function("strftime_now", strftime_now);
    env.add_filter("tojson", tojson);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_template("chat", source)
        .map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;

    let flatten_image_parts =
        template_prints_content_raw(source) && tokenizer.token_to_id(IMAGE_MARKER).is_some();
    let wrap_text_content = template_requires_content_parts(source);
    let mut context = serde_json::Map::new();
    context.insert(
        "messages".to_string(),
        JsonValue::Array(
            messages
                .iter()
                .map(|m| message_to_json(m, flatten_image_parts, wrap_text_content))
                .collect(),
        ),
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
    let mut output = BoundedOutput::new();
    let result = template.render_captured_to(JsonValue::Object(context), &mut output);
    if output.exceeded {
        return Err(TokenizerError::InvalidChatTemplate(format!(
            "rendered output exceeds {MAX_RENDERED_BYTES} bytes"
        )));
    }
    result.map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))?;
    String::from_utf8(output.bytes).map_err(|e| TokenizerError::InvalidChatTemplate(e.to_string()))
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

#[cfg(test)]
mod raw_content_tests {
    use super::template_prints_content_raw;

    #[test]
    fn only_a_template_that_cannot_see_parts_is_flattened() {
        // The Unlimited-OCR-2 checkpoint's whole template.
        assert!(template_prints_content_raw(
            "{% for message in messages %}{% if message['role'] == 'user' %}{% elif \
             message['role'] == 'assistant' %}{% endif %}{{message['content']}}{% if not \
             loop.last %} {% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}"
        ));
        // The shapes the other towers' templates use must keep their part list.
        for parts_aware in [
            "{% if content is string %}{{ content }}{% elif content is iterable %}{% endif %}",
            "{% for item in message['content'] %}{% if item['type'] == 'image' %}x{% endif %}{% endfor %}",
            "{% for item in message.content %}{% if item.type == \"image\" %}x{% endif %}{% endfor %}",
            "{{ message['content'] if message['content'] is not string else '' }}",
        ] {
            assert!(!template_prints_content_raw(parts_aware), "{parts_aware}");
        }
    }
}

#[cfg(test)]
mod tojson_tests {
    use super::*;

    fn render(source: &str) -> Result<String, minijinja::Error> {
        let mut env = Environment::new();
        env.add_filter("tojson", tojson);
        env.render_str(
            source,
            json!({"v": {"b": ["x<y>", 1.5, null], "a": "caf\u{e9} 'q'"}}),
        )
    }

    /// The exact calls Inkling 2's template makes, with Python's bytes.
    #[test]
    fn json_dumps_arguments_write_what_python_writes() {
        assert_eq!(
            render(r#"{{ v | tojson(sort_keys=true, separators=(",", ":")) }}"#).unwrap(),
            "{\"a\":\"caf\u{e9} 'q'\",\"b\":[\"x<y>\",1.5,null]}"
        );
        assert_eq!(
            render(r#"{{ v.b | tojson(separators=(",", ":")) }}"#).unwrap(),
            r#"["x<y>",1.5,null]"#
        );
        // `sort_keys` alone keeps Python's spaced separators.
        assert_eq!(
            render("{{ v.b | tojson(sort_keys=true) }}").unwrap(),
            r#"["x<y>", 1.5, null]"#
        );
        assert!(render("{{ v | tojson(sort_keys=true, indent=2) }}").is_err());
        assert!(render("{{ v | tojson(sort_keys=true, ensure_ascii=true) }}").is_err());
        assert!(render(r#"{{ v | tojson(separators=(",",)) }}"#).is_err());
        assert!(render("{{ v | tojson(sort_keys=true, nonsense=1) }}").is_err());
    }

    /// Without those arguments nothing changed: rendered prompts of every
    /// other family, and the digests frozen on them, stay as they were.
    #[test]
    fn a_plain_call_is_minijinjas_own_filter() {
        for source in [
            "{{ v | tojson }}",
            "{{ v | tojson(indent=2) }}",
            "{{ v | tojson(2) }}",
        ] {
            let builtin = Environment::new()
                .render_str(source, json!({"v": {"b": ["x<y>"], "a": "'q'"}}))
                .unwrap();
            let mut env = Environment::new();
            env.add_filter("tojson", tojson);
            let ours = env
                .render_str(source, json!({"v": {"b": ["x<y>"], "a": "'q'"}}))
                .unwrap();
            assert_eq!(ours, builtin, "{source}");
        }
        assert!(render("{{ v | tojson }}").unwrap().contains("\\u003c"));
    }
}
