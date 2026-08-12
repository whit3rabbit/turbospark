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

/// Env override for [`strftime_now`]'s clock, as `YYYY-MM-DD` or a full
/// `YYYY-MM-DDTHH:MM:SS`.
///
/// **THIS EXISTS BECAUSE A TEMPLATE THAT READS THE CLOCK CANNOT BE FROZEN**,
/// and this repo's entire quality apparatus is frozen digests over a fixed
/// prompt. gpt-oss's Harmony template writes `Current date: ` into its system
/// preamble, so without an override its rendered prompt changes at midnight
/// and every digest taken over it expires the next day -- which reads as a
/// numerics regression rather than as a calendar.
///
/// The DEFAULT is the real clock, because that is what transformers, vLLM and
/// llama.cpp all send and the model was trained against a real date. Pinning
/// it unconditionally would make every user's prompt differ from the reference
/// implementations' to buy a convenience only the gates need.
pub const CHAT_DATE_ENV: &str = "MFERENCE_CHAT_DATE";

/// `strftime_now(format)`, the transformers chat-template global.
///
/// Implemented here rather than pulled in with a date crate: the only thing
/// needed is a broken-down UTC timestamp, and the civil-from-days conversion
/// is a dozen lines (Howard Hinnant's algorithm, valid for any proleptic
/// Gregorian date).
///
/// TWO DEVIATIONS FROM transformers, both deliberate and both stated rather
/// than silent. It is UTC where `datetime.now()` is LOCAL, which can differ by
/// a day at the boundary and reaches the model as one field of a system
/// preamble. And it supports the specifiers real chat templates actually use
/// (`%Y %m %d %H %M %S %y %e` and `%%`); an unrecognized one is emitted
/// VERBATIM, including its `%`, so an unsupported format is visible in the
/// prompt rather than silently dropped or guessed at.
fn strftime_now(format: String) -> Result<String, minijinja::Error> {
    let secs = match std::env::var(CHAT_DATE_ENV) {
        Ok(pinned) => parse_pinned(&pinned).ok_or_else(|| {
            minijinja::Error::new(
                ErrorKind::InvalidOperation,
                format!("{CHAT_DATE_ENV}={pinned:?} is not YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS"),
            )
        })?,
        Err(_) => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0),
    };
    Ok(format_utc(secs, &format))
}

fn parse_pinned(value: &str) -> Option<i64> {
    let (date, time) = match value.split_once('T') {
        Some((d, t)) => (d, t),
        None => (value, "00:00:00"),
    };
    let mut d = date.split('-');
    let (y, m, day) = (
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
        d.next()?.parse::<i64>().ok()?,
    );
    if d.next().is_some() || !(1..=12).contains(&m) || !(1..=31).contains(&day) {
        return None;
    }
    let mut t = time.split(':');
    let (hh, mm, ss) = (
        t.next()?.parse::<i64>().ok()?,
        t.next()?.parse::<i64>().ok()?,
        t.next().unwrap_or("0").parse::<i64>().ok()?,
    );
    Some(days_from_civil(y, m, day) * 86_400 + hh * 3600 + mm * 60 + ss)
}

/// Days since 1970-01-01 for a proleptic Gregorian date (Hinnant).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The inverse, plus the time of day.
fn civil_from_secs(secs: i64) -> (i64, i64, i64, i64, i64, i64) {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    (y, m, d, rem / 3600, (rem % 3600) / 60, rem % 60)
}

fn format_utc(secs: i64, format: &str) -> String {
    let (y, m, d, hh, mm, ss) = civil_from_secs(secs);
    let mut out = String::with_capacity(format.len() + 8);
    let mut chars = format.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&y.to_string()),
            Some('y') => out.push_str(&format!("{:02}", y.rem_euclid(100))),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('e') => out.push_str(&format!("{d:2}")),
            Some('H') => out.push_str(&format!("{hh:02}")),
            Some('M') => out.push_str(&format!("{mm:02}")),
            Some('S') => out.push_str(&format!("{ss:02}")),
            Some('%') => out.push('%'),
            // VERBATIM, `%` included: an unsupported specifier should be
            // visible in the prompt rather than silently dropped.
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
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
