//! Qwen ChatML tool-call body parser:
//! `\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n`.
//! Ported from `Tokenization/QwenToolCallParser.swift`.

use std::collections::{BTreeMap, HashSet};

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

pub struct QwenToolCallParser;

impl QwenToolCallParser {
    pub fn new() -> Self {
        Self
    }

    pub fn parse(
        &self,
        text: &str,
        allowed_tools: &HashSet<String>,
        id: &str,
    ) -> Result<ParsedToolCall, ToolCallParserError> {
        if text.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let mut body = text.trim();

        let name = function_name(&mut body)?;
        if !is_valid_function_name(&name) {
            return Err(ToolCallParserError::Malformed);
        }
        if !allowed_tools.contains(&name) {
            return Err(ToolCallParserError::UnknownTool(name));
        }

        let mut arguments = BTreeMap::new();
        while !body.starts_with("</function>") {
            let (key, value) = parameter(&mut body)?;
            arguments.insert(key, value);
        }
        body = &body["</function>".len()..];
        body = body.trim();
        if !body.is_empty() {
            return Err(ToolCallParserError::Malformed);
        }

        let arguments_value = JsonValue::Object(arguments);
        let arguments_json = arguments_value.encoded();
        Ok(ParsedToolCall {
            id: id.to_string(),
            name,
            arguments: arguments_value,
            arguments_json,
        })
    }
}

impl Default for QwenToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}

fn function_name(body: &mut &str) -> Result<String, ToolCallParserError> {
    let prefix = "<function=";
    if !body.starts_with(prefix) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[prefix.len()..];
    let close = body.find('>').ok_or(ToolCallParserError::Malformed)?;
    let name = body[..close].to_string();
    *body = &body[close + 1..];
    if !body.starts_with('\n') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[1..];
    Ok(name)
}

fn parameter(body: &mut &str) -> Result<(String, JsonValue), ToolCallParserError> {
    let prefix = "<parameter=";
    if !body.starts_with(prefix) {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[prefix.len()..];
    let close = body.find('>').ok_or(ToolCallParserError::Malformed)?;
    let key = body[..close].to_string();
    if key.is_empty() || key.contains('\n') || key.contains('<') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[close + 1..];
    if !body.starts_with('\n') {
        return Err(ToolCallParserError::Malformed);
    }
    *body = &body[1..];

    let close_marker = "\n</parameter>\n";
    let (value, rest) = if let Some(pos) = body.find(close_marker) {
        (&body[..pos], &body[pos + close_marker.len()..])
    } else if let Some(stripped) = body.strip_prefix("</parameter>\n") {
        ("", stripped)
    } else {
        return Err(ToolCallParserError::Malformed);
    };
    *body = rest;
    Ok((key, parsed_value(value)))
}

fn parsed_value(raw: &str) -> JsonValue {
    let trimmed = raw.trim();
    let Some(first) = trimmed.chars().next() else {
        return JsonValue::String(raw.to_string());
    };
    if !"{[-0123456789tfn".contains(first) {
        return JsonValue::String(raw.to_string());
    }
    match JsonValue::parse(trimmed) {
        Ok(JsonValue::String(_)) => JsonValue::String(raw.to_string()),
        Ok(value) => value,
        Err(_) => JsonValue::String(raw.to_string()),
    }
}
