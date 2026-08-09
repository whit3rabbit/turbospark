//! DeepSeek-V4 DSML tool-call body parser: one or more
//! `<DSML|invoke name="NAME">...<DSML|parameter name="KEY" string="true|false">VALUE</DSML|parameter>...</DSML|invoke>`
//! blocks. Ported from `Tokenization/DeepseekToolCallParser.swift`.

use std::collections::{BTreeMap, HashSet};

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

/// DSML framing text, shared with the tokenizer's native tool-chat render and
/// the streaming decoder's delta scan. The bars are fullwidth (U+FF5C),
/// matching the model's training data.
pub const DSML_MARK: &str = "\u{FF5C}DSML\u{FF5C}";

/// DeepSeek DSML tool call parser.
pub struct DeepseekToolCallParser;

impl DeepseekToolCallParser {
    /// Creates a new `DeepseekToolCallParser`.
    pub fn new() -> Self {
        Self
    }

    /// Returns the opening DSML tool calls tag string.
    pub fn tool_calls_open_mark() -> String {
        format!("<{DSML_MARK}tool_calls>")
    }

    /// Returns the closing DSML tool calls tag string.
    pub fn tool_calls_close_mark() -> String {
        format!("</{DSML_MARK}tool_calls>")
    }

    /// Parses DSML tool call text against an allowed set of tool names.
    pub fn parse(
        &self,
        text: &str,
        allowed_tools: &HashSet<String>,
        mut id_generator: impl FnMut() -> String,
    ) -> Result<Vec<ParsedToolCall>, ToolCallParserError> {
        if text.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let mut body = text.trim();
        if body.is_empty() {
            return Err(ToolCallParserError::Malformed);
        }

        let mut calls = Vec::new();
        while !body.is_empty() {
            calls.push(invoke(&mut body, allowed_tools, &id_generator())?);
            body = body.trim_start();
        }
        Ok(calls)
    }
}

impl Default for DeepseekToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

fn invoke(
    body: &mut &str,
    allowed_tools: &HashSet<String>,
    id: &str,
) -> Result<ParsedToolCall, ToolCallParserError> {
    let open_prefix = format!("<{DSML_MARK}invoke name=\"");
    if !body.starts_with(&open_prefix) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[open_prefix.len()..];
    let quote = body.find('"').ok_or(ToolCallParserError::Malformed)?;
    let name = body[..quote].to_string();
    *body = &body[quote + 1..];
    if !body.starts_with('>') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[1..];
    if !is_valid_function_name(&name) {
        return Err(ToolCallParserError::Malformed);
    }
    if !allowed_tools.contains(&name) {
        return Err(ToolCallParserError::UnknownTool(name));
    }

    let close_mark = format!("</{DSML_MARK}invoke>");
    let mut arguments = BTreeMap::new();
    *body = body.trim_start();
    while !body.starts_with(&close_mark) {
        let (key, value) = parameter(body)?;
        arguments.insert(key, value);
        *body = body.trim_start();
    }
    *body = &body[close_mark.len()..];

    let arguments_value = JsonValue::Object(arguments);
    let arguments_json = arguments_value.encoded();
    Ok(ParsedToolCall {
        id: id.to_string(),
        name,
        arguments: arguments_value,
        arguments_json,
    })
}

fn parameter(body: &mut &str) -> Result<(String, JsonValue), ToolCallParserError> {
    let open_prefix = format!("<{DSML_MARK}parameter name=\"");
    if !body.starts_with(&open_prefix) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[open_prefix.len()..];
    let quote = body.find('"').ok_or(ToolCallParserError::Malformed)?;
    let key = body[..quote].to_string();
    if key.is_empty() || key.contains('\n') || key.contains('<') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[quote + 1..];
    if !body.starts_with(" string=\"") {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[" string=\"".len()..];
    let is_string = if body.starts_with("true\">") {
        *body = &body["true\">".len()..];
        true
    } else if body.starts_with("false\">") {
        *body = &body["false\">".len()..];
        false
    } else {
        return Err(ToolCallParserError::Malformed);
    };

    let close_mark = format!("</{DSML_MARK}parameter>");
    let pos = body
        .find(&close_mark)
        .ok_or(ToolCallParserError::Malformed)?;
    let raw = body[..pos].to_string();
    *body = &body[pos + close_mark.len()..];
    if is_string {
        Ok((key, JsonValue::String(raw)))
    } else {
        let value = JsonValue::parse(&raw).map_err(|_| ToolCallParserError::Malformed)?;
        Ok((key, value))
    }
}
