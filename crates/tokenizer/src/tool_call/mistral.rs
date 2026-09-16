//! Mistral `[TOOL_CALLS]` body parser: a JSON array of
//! `{"name": NAME, "arguments": {..}}` objects, the format Mistral 7B's
//! template teaches (`[TOOL_CALLS] [{"name": "get_weather", "arguments":
//! {"city": "Oslo"}}]`). Unlike every other body parser here this one is
//! ARRAY-shaped, because Mistral frames ALL of a turn's calls under the ONE
//! `[TOOL_CALLS]` marker rather than bracketing each call separately.
//!
//! A single object is accepted beside the array: the same models sometimes
//! drop the array bracket for a one-call turn, and vLLM's reference parser
//! accepts both, so refusing the object shape would rescue-worthy output
//! out of the native path for no reason.

use std::collections::HashSet;

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

/// Mistral `[TOOL_CALLS]` body parser.
pub struct MistralToolCallParser;

impl MistralToolCallParser {
    /// Creates a new `MistralToolCallParser`.
    pub fn new() -> Self {
        Self
    }

    /// Parses the span body that follows a `[TOOL_CALLS]` marker against an
    /// allowed set of tool names. Returns one call per array element, plus
    /// whatever text followed the JSON.
    ///
    /// **THE BODY IS PREFIX-PARSED, NOT WHOLE-TEXT.** A Mistral model puts
    /// every call of the turn in ONE array under the ONE marker, then
    /// sometimes keeps talking: real output on the pinned 7B install reads
    /// `[{"name": ..., "arguments": {...}}]` followed by ordinary prose. The
    /// array is the span's payload and the prose is content, so the scanner
    /// below takes the leading balanced JSON value and the caller decides
    /// what the remainder is. A value with no balanced close (a generation
    /// cut off mid-array) is malformed, which is what catches truncated
    /// output.
    pub fn parse(
        &self,
        text: &str,
        allowed_tools: &HashSet<String>,
        mut id_generator: impl FnMut() -> String,
    ) -> Result<(Vec<ParsedToolCall>, String), ToolCallParserError> {
        if text.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let trimmed = text.trim_start();
        let (json, remainder) =
            leading_json_value(trimmed).ok_or(ToolCallParserError::Malformed)?;
        let value = JsonValue::parse(json).map_err(|_| ToolCallParserError::Malformed)?;
        let entries = match value {
            JsonValue::Array(items) => items,
            JsonValue::Object(_) => vec![value],
            _ => return Err(ToolCallParserError::Malformed),
        };
        if entries.is_empty() {
            return Err(ToolCallParserError::Malformed);
        }
        let mut calls = Vec::with_capacity(entries.len());
        for entry in entries {
            let JsonValue::Object(fields) = &entry else {
                return Err(ToolCallParserError::Malformed);
            };
            let Some(name) = fields.get("name").and_then(as_name) else {
                return Err(ToolCallParserError::Malformed);
            };
            if !is_valid_function_name(&name) || !allowed_tools.contains(&name) {
                return Err(ToolCallParserError::Malformed);
            }
            // `arguments` is an object in Mistral's format (not the
            // JSON-encoded STRING ChatML carries). A missing field parses as
            // a call with no arguments, which is what the schema validator
            // downstream expects to judge.
            let arguments = fields.get("arguments").cloned().unwrap_or(JsonValue::Null);
            calls.push(ParsedToolCall {
                id: id_generator(),
                name,
                arguments_json: arguments.encoded(),
                arguments,
            });
        }
        Ok((calls, remainder.to_string()))
    }
}

impl Default for MistralToolCallParser {
    fn default() -> Self {
        Self
    }
}

/// Splits `text` at the end of its leading JSON object or array, returning
/// the slice that parses and whatever follows it. Byte-level bracket
/// counting: the value's string literals are skipped so a `}` inside quotes
/// cannot close the scan early. `None` when the text does not open with a
/// bracket or never balances.
fn leading_json_value(text: &str) -> Option<(&str, &str)> {
    let bytes = text.as_bytes();
    let open = *bytes.first()?;
    if open != b'{' && open != b'[' {
        return None;
    }
    let close = if open == b'{' { b'}' } else { b']' };
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    if b != close {
                        // A `[` closed by `}` (or the reverse) is not a
                        // balanced value; let the parser refuse it.
                        return None;
                    }
                    return Some((&text[..=i], text[i + 1..].trim_start()));
                }
            }
            _ => {}
        }
    }
    None
}

fn as_name(value: &JsonValue) -> Option<String> {
    match value {
        JsonValue::String(s) => Some(s.clone()),
        _ => None,
    }
}
