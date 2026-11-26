//! Qwen ChatML tool-call body parser:
//! `\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n`.
//! Ported from `Tokenization/QwenToolCallParser.swift`.

use std::collections::{BTreeMap, HashSet};

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

/// Parser for Qwen ChatML XML-style function call markup blocks.
pub struct QwenToolCallParser;

impl QwenToolCallParser {
    /// Creates a new Qwen tool-call parser.
    pub fn new() -> Self {
        Self
    }

    /// Parses a Qwen XML tool-call block into a [`ParsedToolCall`].
    pub fn parse(
        &self,
        text: &str,
        allowed_tools: &HashSet<String>,
        id: &str,
    ) -> Result<ParsedToolCall, ToolCallParserError> {
        self.parse_with_schemas(text, allowed_tools, id, &BTreeMap::new())
    }

    /// Like [`Self::parse`], but coerces each parameter value by the tool's
    /// JSON schema when `schemas` has one (tool name -> its `parameters`
    /// schema). The wire format carries no types, so without the schema a
    /// string parameter whose text happens to read as JSON (`true`, `null`,
    /// `1e3`, `12`) was silently turned into a bool, null or number. A tool,
    /// or a property, with no usable schema keeps the old heuristic.
    pub fn parse_with_schemas(
        &self,
        text: &str,
        allowed_tools: &HashSet<String>,
        id: &str,
        schemas: &BTreeMap<String, JsonValue>,
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

        let schema = schemas.get(&name);
        let mut arguments = BTreeMap::new();
        while !body.starts_with("</function>") {
            let (key, raw) = parameter(&mut body)?;
            let value = Self::coerce_parameter(raw, schema, &key);
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

    /// Coerces one raw parameter value by the owning tool's `parameters`
    /// schema (`tool_schema`) and the parameter's `key`. Public so the
    /// server's raw-text rescue, which reads the same markup with a regex,
    /// types values exactly as the native decoder does.
    pub fn coerce_parameter(raw: &str, tool_schema: Option<&JsonValue>, key: &str) -> JsonValue {
        coerce(raw, property_schema(tool_schema, key))
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

fn parameter<'t>(body: &mut &'t str) -> Result<(String, &'t str), ToolCallParserError> {
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
    Ok((key, value))
}

/// The `properties.<key>` schema of a tool's parameters, if there is one.
fn property_schema<'s>(schema: Option<&'s JsonValue>, key: &str) -> Option<&'s JsonValue> {
    schema?
        .as_object()?
        .get("properties")?
        .as_object()?
        .get(key)
}

/// The declared JSON types of a property, or empty when it names none.
fn declared_types(schema: &JsonValue) -> Vec<&str> {
    match schema.as_object().and_then(|o| o.get("type")) {
        Some(JsonValue::String(t)) => vec![t.as_str()],
        Some(JsonValue::Array(ts)) => ts
            .iter()
            .filter_map(|t| match t {
                JsonValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

/// Coerces a raw parameter value. A declared schema type wins over the
/// text's shape; anything the schema cannot decide falls back to
/// [`parsed_value`], so a value the schema would reject still reaches the
/// tool the way it did before schemas were consulted.
fn coerce(raw: &str, schema: Option<&JsonValue>) -> JsonValue {
    let types = schema.map(declared_types).unwrap_or_default();
    if types.is_empty() {
        return parsed_value(raw);
    }
    // `string` (optionally nullable) means the text is the value byte for
    // byte, whatever it looks like. A union that also admits a non-string
    // type is ambiguous by text alone and keeps the heuristic.
    let only_stringish = types.iter().all(|t| *t == "string" || *t == "null");
    if only_stringish && types.contains(&"string") {
        return JsonValue::String(raw.to_string());
    }
    let trimmed = raw.trim();
    let Ok(parsed) = JsonValue::parse(trimmed) else {
        return fallback(raw, &types);
    };
    let fits = types.iter().any(|t| {
        matches!(
            (*t, &parsed),
            (
                "integer",
                JsonValue::Integer(_) | JsonValue::UnsignedInteger(_)
            ) | (
                "number",
                JsonValue::Integer(_) | JsonValue::UnsignedInteger(_) | JsonValue::Number(_)
            ) | ("boolean", JsonValue::Bool(_))
                | ("null", JsonValue::Null)
                | ("array", JsonValue::Array(_))
                | ("object", JsonValue::Object(_))
        )
    });
    if fits {
        parsed
    } else {
        fallback(raw, &types)
    }
}

/// A value that does not parse as any declared type: keep the text when the
/// schema allows a string, otherwise the pre-schema heuristic.
fn fallback(raw: &str, types: &[&str]) -> JsonValue {
    if types.contains(&"string") {
        JsonValue::String(raw.to_string())
    } else {
        parsed_value(raw)
    }
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
