//! GLM tool-call body parser: one
//! `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>`
//! block. The grammar is read off `zai-org/GLM-4.7-Flash`'s own
//! `chat_template.jinja` (2026-09-19), which both teaches the model the shape
//! ("output the function name and arguments within the following XML
//! format") and renders history back through it.

use std::collections::{BTreeMap, HashSet};

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

/// The `<tool_call>` / `</tool_call>` block brackets. ADDED but not special
/// in the real table (ids 154843 / 154844), so they survive detokenization
/// as literal text and the streaming decoder scans for these strings rather
/// than for ids.
pub const GLM_TOOL_CALL_OPEN_MARK: &str = "<tool_call>";
pub const GLM_TOOL_CALL_CLOSE_MARK: &str = "</tool_call>";

/// The argument pair. Inside one block the keys and values alternate:
/// `<arg_key>K</arg_key>` immediately followed by `<arg_value>V</arg_value>`.
pub const GLM_ARG_KEY_OPEN_MARK: &str = "<arg_key>";
pub const GLM_ARG_KEY_CLOSE_MARK: &str = "</arg_key>";
pub const GLM_ARG_VALUE_OPEN_MARK: &str = "<arg_value>";
pub const GLM_ARG_VALUE_CLOSE_MARK: &str = "</arg_value>";

/// GLM tool-call body parser.
pub struct GlmToolCallParser;

impl GlmToolCallParser {
    /// Creates a new `GlmToolCallParser`.
    pub fn new() -> Self {
        Self
    }

    /// Parses ONE block's body -- the text between [`GLM_TOOL_CALL_OPEN_MARK`]
    /// and [`GLM_TOOL_CALL_CLOSE_MARK`], bracketing stripped by the caller --
    /// against an allowed set of tool names. The body is the function name
    /// followed by zero or more key/value pairs.
    ///
    /// **A VALUE IS JSON-DECODED WHEN IT PARSES AS JSON AND KEPT RAW WHEN IT
    /// DOES NOT**, because the template renders string arguments RAW and
    /// every other value with `tojson` (`{{ v | tojson(ensure_ascii=False)
    /// if v is not string else v }}`). `Oslo` arrives as `Oslo`, the number
    /// `1` as `1`, and a list as `[1, 2]`; only the first is invalid JSON.
    /// The order is lossy in exactly one direction -- a string argument whose
    /// content happens to be valid JSON (the literal text `"123"`) decodes as
    /// the string `123` without its quotes -- and that ambiguity is the
    /// format's, not this parser's: the template cannot distinguish them
    /// either.
    pub fn parse(
        &self,
        body: &str,
        allowed_tools: &HashSet<String>,
        id: impl FnOnce() -> String,
    ) -> Result<ParsedToolCall, ToolCallParserError> {
        if body.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let body = body.trim();
        if body.is_empty() {
            return Err(ToolCallParserError::Malformed);
        }
        // The name runs from the top of the body to the first argument pair
        // (or the end of the body for a call with no arguments).
        let name_end = body
            .find(GLM_ARG_KEY_OPEN_MARK)
            .ok_or(ToolCallParserError::Malformed)?;
        let name = body[..name_end].trim();
        if !is_valid_function_name(name) {
            return Err(ToolCallParserError::Malformed);
        }
        if !allowed_tools.contains(name) {
            return Err(ToolCallParserError::UnknownTool(name.to_string()));
        }

        let mut rest = &body[name_end..];
        let mut arguments = BTreeMap::new();
        while !rest.is_empty() {
            let (key, value) = argument(&mut rest)?;
            arguments.insert(key, value);
        }
        let arguments_value = JsonValue::Object(arguments);
        let arguments_json = arguments_value.encoded();
        Ok(ParsedToolCall {
            id: id(),
            name: name.to_string(),
            arguments: arguments_value,
            arguments_json,
        })
    }
}

impl Default for GlmToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Consumes one `<arg_key>K</arg_key><arg_value>V</arg_value>` pair from the
/// front of `rest`, leaving `rest` at whatever follows it.
fn argument(body: &mut &str) -> Result<(String, JsonValue), ToolCallParserError> {
    if !body.starts_with(GLM_ARG_KEY_OPEN_MARK) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[GLM_ARG_KEY_OPEN_MARK.len()..];
    let close = body
        .find(GLM_ARG_KEY_CLOSE_MARK)
        .ok_or(ToolCallParserError::Malformed)?;
    let key = body[..close].trim();
    if key.is_empty() || key.contains('\n') || key.contains('<') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[close + GLM_ARG_KEY_CLOSE_MARK.len()..];

    if !body.starts_with(GLM_ARG_VALUE_OPEN_MARK) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[GLM_ARG_VALUE_OPEN_MARK.len()..];
    let close = body
        .find(GLM_ARG_VALUE_CLOSE_MARK)
        .ok_or(ToolCallParserError::Malformed)?;
    let raw = body[..close].to_string();
    *body = &body[close + GLM_ARG_VALUE_CLOSE_MARK.len()..];
    let value = match JsonValue::parse(raw.trim()) {
        Ok(value) => value,
        // Not JSON: the raw text IS the string value, exactly as the
        // template renders string arguments.
        Err(_) => JsonValue::String(raw),
    };
    Ok((key.to_string(), value))
}
