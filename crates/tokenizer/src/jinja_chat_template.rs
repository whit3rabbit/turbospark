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
use crate::reasoning::ReasoningEffort;

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

/// Wraps a conditional expression used as a KEYWORD ARGUMENT in parentheses,
/// so minijinja can parse a template Jinja2 accepts.
///
/// **THIS IS A COMPATIBILITY SHIM AND IT IS MEANT TO BE DELETED.** minijinja
/// 2.22.0 (the latest 2.x) rejects `f(k=a if c else d)` with
/// `syntax error: unexpected identifier, expected ","` -- after `k=a` its
/// parser wants a `,` or `)` and finds `if`. Jinja2's grammar for a keyword
/// argument's value is a full expression, conditional included, so real
/// templates use it: `mlx-community/Muse-Glimmer-30B-4bit`'s
/// `chat_template.jinja` has `namespace(name=tcid if tcid else '')`, and
/// without this the whole template fails to parse and `--messages-file`
/// cannot render a prompt at all.
///
/// **WHEN TO REMOVE IT.** If minijinja gains support (no stable release has
/// it as of 2.22.0; 3.0.0-alpha.0 is untested here and deliberately not
/// taken), delete this function and its call site and run
/// `tests/jinja_chat_template.rs` -- `the_shim_is_a_no_op_on_templates_that_
/// do_not_need_it` will still pass, and
/// `a_conditional_keyword_argument_parses` is the one that says whether the
/// engine now handles it directly. Nothing else depends on the rewrite.
///
/// **WHY THIS IS SAFE, and the two ways it could not have been.** The
/// transformation is `k=EXPR` -> `k=(EXPR)`, which is exactly Jinja2's own
/// precedence for a keyword argument value, so it changes no semantics by
/// construction -- it cannot reorder an expression, only make the existing
/// grouping explicit. The two real hazards are both handled structurally
/// rather than by pattern-matching:
///
/// 1. **It must not touch template TEXT.** A chat template is mostly prose
///    and markup, which is full of `=` and parentheses. So the scan only
///    enters `{{ ... }}` and `{% ... %}` blocks and steps over `{# ... #}`
///    comments; everything outside is copied verbatim.
/// 2. **It must not mistake a comparison for an assignment.** `==`, `!=`,
///    `<=`, `>=` are skipped, and a `=` only counts when the character
///    before it is part of an identifier and the character after it is not
///    another `=`.
///
/// String literals are tracked so a `,` or `)` inside `'...'` cannot end an
/// argument early, and nesting is tracked so a call inside a call is one
/// value rather than several.
///
/// Returns a borrowed `Cow` when nothing needed rewriting, which is the case
/// for every other template in this repo.
fn parenthesize_conditional_kwargs(source: &str) -> std::borrow::Cow<'_, str> {
    if !source.contains('=') {
        return std::borrow::Cow::Borrowed(source);
    }
    let bytes = source.as_bytes();
    let mut out = String::new();
    let mut rewrote = false;
    let mut i = 0usize;

    while i < bytes.len() {
        // Only Jinja blocks are scanned; everything else is template text.
        let block = block_at(bytes, i);
        let Some((open_len, close, is_comment)) = block else {
            out.push(bytes[i] as char);
            i += 1;
            continue;
        };
        let start = i;
        let body_start = i + open_len;
        let Some(body_end) = find_close(source, body_start, close) else {
            // Unterminated block: copy the rest verbatim and let minijinja
            // report it. Guessing at a repair here would turn a clear
            // syntax error into a confusing one.
            out.push_str(&source[start..]);
            i = bytes.len();
            continue;
        };
        out.push_str(&source[start..body_start]);
        if is_comment {
            out.push_str(&source[body_start..body_end + close.len()]);
        } else {
            let (rewritten, changed) = rewrite_expression(&source[body_start..body_end]);
            rewrote |= changed;
            out.push_str(&rewritten);
            out.push_str(close);
        }
        i = body_end + close.len();
    }

    if rewrote {
        std::borrow::Cow::Owned(out)
    } else {
        std::borrow::Cow::Borrowed(source)
    }
}

/// `(opening delimiter length, closing delimiter, is a comment)` if a Jinja
/// block starts at `i`.
fn block_at(bytes: &[u8], i: usize) -> Option<(usize, &'static str, bool)> {
    if bytes[i] != b'{' || i + 1 >= bytes.len() {
        return None;
    }
    match bytes[i + 1] {
        b'{' => Some((2, "}}", false)),
        b'%' => Some((2, "%}", false)),
        b'#' => Some((2, "#}", true)),
        _ => None,
    }
}

/// Index of `close` at or after `from`, skipping string literals.
fn find_close(source: &str, from: usize, close: &str) -> Option<usize> {
    let bytes = source.as_bytes();
    let mut i = from;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => {
                if c == b'\'' || c == b'"' {
                    quote = Some(c);
                } else if source[i..].starts_with(close) {
                    return Some(i);
                }
            }
        }
        i += 1;
    }
    None
}

/// Rewrites one Jinja expression/statement body.
fn rewrite_expression(body: &str) -> (String, bool) {
    let bytes = body.as_bytes();
    let mut out = String::with_capacity(body.len());
    let mut changed = false;
    let mut i = 0usize;
    let mut quote: Option<u8> = None;
    let mut depth = 0usize;

    while i < bytes.len() {
        let c = bytes[i];
        if let Some(q) = quote {
            out.push(c as char);
            if c == b'\\' && i + 1 < bytes.len() {
                out.push(bytes[i + 1] as char);
                i += 2;
                continue;
            }
            if c == q {
                quote = None;
            }
            i += 1;
            continue;
        }
        match c {
            b'\'' | b'"' => {
                quote = Some(c);
                out.push(c as char);
                i += 1;
            }
            b'(' | b'[' | b'{' => {
                depth += 1;
                out.push(c as char);
                i += 1;
            }
            b')' | b']' | b'}' => {
                depth = depth.saturating_sub(1);
                out.push(c as char);
                i += 1;
            }
            b'=' if depth > 0 && is_kwarg_eq(bytes, i) => {
                out.push('=');
                let value_start = i + 1;
                let value_end = kwarg_value_end(bytes, value_start);
                let value = &body[value_start..value_end];
                if contains_top_level_conditional(value) {
                    out.push('(');
                    out.push_str(value.trim_end());
                    out.push(')');
                    // Preserve trailing whitespace outside the parens so the
                    // rewrite is byte-identical apart from the two brackets.
                    out.push_str(&value[value.trim_end().len()..]);
                    changed = true;
                } else {
                    out.push_str(value);
                }
                i = value_end;
            }
            _ => {
                out.push(c as char);
                i += 1;
            }
        }
    }
    (out, changed)
}

/// True when the `=` at `i` is a keyword-argument assignment rather than a
/// comparison operator.
fn is_kwarg_eq(bytes: &[u8], i: usize) -> bool {
    if bytes.get(i + 1) == Some(&b'=') {
        return false; // ==
    }
    let Some(&prev) = bytes.get(i.wrapping_sub(1)) else {
        return false;
    };
    if matches!(prev, b'=' | b'!' | b'<' | b'>') {
        return false; // ==, !=, <=, >=
    }
    prev.is_ascii_alphanumeric() || prev == b'_'
}

/// End of a keyword argument's value: the next `,` or closing bracket at the
/// value's own nesting level, skipping string literals.
fn kwarg_value_end(bytes: &[u8], from: usize) -> usize {
    let mut i = from;
    let mut depth = 0usize;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => {
                    if depth == 0 {
                        return i;
                    }
                    depth -= 1;
                }
                b',' if depth == 0 => return i,
                _ => {}
            },
        }
        i += 1;
    }
    bytes.len()
}

/// True when `value` holds an ` if ` at its own bracket level and outside
/// string literals -- i.e. a conditional expression rather than one nested in
/// a sub-call that already has its own parentheses.
fn contains_top_level_conditional(value: &str) -> bool {
    let bytes = value.as_bytes();
    let mut i = 0usize;
    let mut depth = 0usize;
    let mut quote: Option<u8> = None;
    while i < bytes.len() {
        let c = bytes[i];
        match quote {
            Some(q) => {
                if c == b'\\' {
                    i += 2;
                    continue;
                }
                if c == q {
                    quote = None;
                }
            }
            None => match c {
                b'\'' | b'"' => quote = Some(c),
                b'(' | b'[' | b'{' => depth += 1,
                b')' | b']' | b'}' => depth = depth.saturating_sub(1),
                b'i' if depth == 0 && value[i..].starts_with("if ") => {
                    let before = i.checked_sub(1).map(|p| bytes[p]);
                    if matches!(before, Some(b' ') | Some(b'\t') | Some(b'\n')) {
                        return true;
                    }
                }
                _ => {}
            },
        }
        i += 1;
    }
    false
}
