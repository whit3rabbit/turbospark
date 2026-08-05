//! Structural JSON value carrying integer/float distinctions through tool-call
//! argument round-trips. Ported from `Tokenization/JSONValue.swift`.
//!
//! Swift's `Decimal` case (arbitrary-precision) is folded into `Number(f64)`
//! here: the Rust port targets tool-call argument JSON, where `serde_json`
//! round-tripping through `f64` is an accepted precision limitation rather
//! than a hard parity requirement.

use std::collections::BTreeMap;

use crate::error::ToolCallParserError;

/// Nesting depth ceiling shared by the tool-call parsers and this decoder, to
/// bound recursive descent against adversarial/model-generated JSON.
pub const MAXIMUM_DEPTH: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Object(BTreeMap<String, JsonValue>),
    Array(Vec<JsonValue>),
    String(String),
    Integer(i64),
    UnsignedInteger(u64),
    Number(f64),
    Bool(bool),
    Null,
}

impl JsonValue {
    pub fn as_object(&self) -> Option<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Some(m),
            _ => None,
        }
    }

    /// Parse a JSON document into a [`JsonValue`], rejecting documents nested
    /// deeper than [`MAXIMUM_DEPTH`].
    pub fn parse(text: &str) -> Result<JsonValue, ToolCallParserError> {
        let value: serde_json::Value =
            serde_json::from_str(text).map_err(|_| ToolCallParserError::Malformed)?;
        from_serde(&value, 0)
    }

    /// Render as compact JSON text. `sorted_keys` matches Swift's
    /// `.sortedKeys` output formatting option (BTreeMap already iterates
    /// sorted, so this only controls whether callers may rely on that).
    pub fn encoded(&self) -> String {
        render(self)
    }

    /// Convert to a `serde_json::Value`, for handing this value to a
    /// generic JSON/template consumer (e.g. the Jinja chat-template
    /// renderer) that only understands `serde_json`'s type.
    pub fn to_serde_json(&self) -> serde_json::Value {
        match self {
            JsonValue::Object(map) => serde_json::Value::Object(
                map.iter()
                    .map(|(k, v)| (k.clone(), v.to_serde_json()))
                    .collect(),
            ),
            JsonValue::Array(items) => {
                serde_json::Value::Array(items.iter().map(JsonValue::to_serde_json).collect())
            }
            JsonValue::String(s) => serde_json::Value::String(s.clone()),
            JsonValue::Integer(v) => serde_json::Value::Number((*v).into()),
            JsonValue::UnsignedInteger(v) => serde_json::Value::Number((*v).into()),
            JsonValue::Number(v) => serde_json::Number::from_f64(*v)
                .map(serde_json::Value::Number)
                .unwrap_or(serde_json::Value::Null),
            JsonValue::Bool(v) => serde_json::Value::Bool(*v),
            JsonValue::Null => serde_json::Value::Null,
        }
    }

    /// Rewrites this JSON Schema fragment so every property the Gemma tool
    /// template renders carries a concrete scalar `type`, per the upstream
    /// contract documented in `JSONValue.swift`.
    pub fn gemma_schema_normalized(&self) -> JsonValue {
        let JsonValue::Object(object) = self else {
            return self.clone();
        };
        let mut object = object.clone();

        if !has_scalar_string_type(&object) {
            if let Some(JsonValue::Array(members)) = object.get("type") {
                if let Some(concrete) = first_non_null_type_name(members) {
                    object.insert("type".to_string(), JsonValue::String(concrete));
                }
            }
        }
        if !has_scalar_string_type(&object) {
            for keyword in UNION_KEYWORDS {
                if let Some(JsonValue::Array(branches)) = object.get(keyword).cloned() {
                    return if keyword == "allOf" {
                        merged_intersection(&object, &branches)
                    } else {
                        collapsed_union(&object, &branches)
                    };
                }
            }
        }
        if !has_scalar_string_type(&object) {
            let ty = if object.contains_key("properties") {
                "object"
            } else {
                "string"
            };
            object.insert("type".to_string(), JsonValue::String(ty.to_string()));
        }

        if let Some(JsonValue::Object(properties)) = object.get("properties").cloned() {
            let normalized: BTreeMap<String, JsonValue> = properties
                .into_iter()
                .map(|(k, v)| (k, v.gemma_schema_normalized()))
                .collect();
            object.insert("properties".to_string(), JsonValue::Object(normalized));
        }
        match object.get("items").cloned() {
            Some(JsonValue::Object(_)) => {
                let normalized = object["items"].gemma_schema_normalized();
                object.insert("items".to_string(), normalized);
            }
            Some(JsonValue::Array(items)) => {
                let normalized: Vec<JsonValue> =
                    items.iter().map(|i| i.gemma_schema_normalized()).collect();
                object.insert("items".to_string(), JsonValue::Array(normalized));
            }
            _ => {}
        }
        JsonValue::Object(object)
    }

    /// Gemma tool-call argument body: `key:value,key:value`, keys sorted,
    /// strings JSON-quoted, everything else rendered via `gemma_tool_value`.
    pub fn gemma_tool_argument_body(&self) -> Result<String, ToolCallParserError> {
        let object = self.as_object().ok_or(ToolCallParserError::Malformed)?;
        let mut parts = Vec::with_capacity(object.len());
        for (key, value) in object {
            if !is_representable_object_key(key) {
                return Err(ToolCallParserError::Malformed);
            }
            parts.push(format!("{key}:{}", value.gemma_tool_value()?));
        }
        Ok(parts.join(","))
    }

    fn gemma_tool_value(&self) -> Result<String, ToolCallParserError> {
        match self {
            JsonValue::Object(_) => Ok(format!("{{{}}}", self.gemma_tool_argument_body()?)),
            JsonValue::Array(items) => {
                let rendered: Result<Vec<String>, ToolCallParserError> =
                    items.iter().map(|v| v.gemma_tool_value()).collect();
                Ok(format!("[{}]", rendered?.join(",")))
            }
            JsonValue::String(s) => {
                serde_json::to_string(s).map_err(|_| ToolCallParserError::Malformed)
            }
            JsonValue::Integer(v) => Ok(v.to_string()),
            JsonValue::UnsignedInteger(v) => Ok(v.to_string()),
            JsonValue::Number(v) => {
                if !v.is_finite() {
                    return Err(ToolCallParserError::Malformed);
                }
                Ok(v.to_string())
            }
            JsonValue::Bool(v) => Ok(v.to_string()),
            JsonValue::Null => Ok("null".to_string()),
        }
    }
}

const UNION_KEYWORDS: [&str; 3] = ["anyOf", "oneOf", "allOf"];

fn has_scalar_string_type(object: &BTreeMap<String, JsonValue>) -> bool {
    matches!(object.get("type"), Some(JsonValue::String(_)))
}

fn first_non_null_type_name(members: &[JsonValue]) -> Option<String> {
    members.iter().find_map(|m| match m {
        JsonValue::String(s) if s != "null" => Some(s.clone()),
        _ => None,
    })
}

fn collapsed_union(parent: &BTreeMap<String, JsonValue>, branches: &[JsonValue]) -> JsonValue {
    let resolved: Vec<BTreeMap<String, JsonValue>> = branches
        .iter()
        .filter_map(|b| b.gemma_schema_normalized().as_object().cloned())
        .filter(has_scalar_string_type)
        .collect();
    let chosen = resolved
        .iter()
        .find(|o| o.get("type") != Some(&JsonValue::String("null".to_string())))
        .or_else(|| resolved.first())
        .cloned()
        .unwrap_or_default();
    completed(chosen, parent)
}

fn merged_intersection(parent: &BTreeMap<String, JsonValue>, branches: &[JsonValue]) -> JsonValue {
    let mut merged = BTreeMap::new();
    for branch in branches {
        if let JsonValue::Object(b) = branch {
            absorb(b, &mut merged);
        }
    }
    completed(merged, parent)
}

fn completed(
    collapsed: BTreeMap<String, JsonValue>,
    parent: &BTreeMap<String, JsonValue>,
) -> JsonValue {
    let mut result = collapsed;
    let mut siblings = parent.clone();
    for keyword in UNION_KEYWORDS {
        siblings.remove(keyword);
    }
    absorb(&siblings, &mut result);
    JsonValue::Object(result).gemma_schema_normalized()
}

fn absorb(addition: &BTreeMap<String, JsonValue>, base: &mut BTreeMap<String, JsonValue>) {
    for (key, value) in addition {
        if key == "properties" {
            if let (Some(JsonValue::Object(existing)), JsonValue::Object(incoming)) =
                (base.get("properties").cloned(), value)
            {
                let mut merged = incoming.clone();
                for (k, v) in existing {
                    merged.insert(k, v);
                }
                base.insert("properties".to_string(), JsonValue::Object(merged));
                continue;
            }
        }
        base.entry(key.clone()).or_insert_with(|| value.clone());
    }
}

fn is_representable_object_key(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_alphanumeric() || "_-.$".contains(c))
}

fn from_serde(value: &serde_json::Value, depth: usize) -> Result<JsonValue, ToolCallParserError> {
    if depth > MAXIMUM_DEPTH {
        return Err(ToolCallParserError::Malformed);
    }
    Ok(match value {
        serde_json::Value::Null => JsonValue::Null,
        serde_json::Value::Bool(b) => JsonValue::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                JsonValue::Integer(i)
            } else if let Some(u) = n.as_u64() {
                JsonValue::UnsignedInteger(u)
            } else {
                JsonValue::Number(n.as_f64().ok_or(ToolCallParserError::Malformed)?)
            }
        }
        serde_json::Value::String(s) => JsonValue::String(s.clone()),
        serde_json::Value::Array(items) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                out.push(from_serde(item, depth + 1)?);
            }
            JsonValue::Array(out)
        }
        serde_json::Value::Object(map) => {
            let mut out = BTreeMap::new();
            for (k, v) in map {
                out.insert(k.clone(), from_serde(v, depth + 1)?);
            }
            JsonValue::Object(out)
        }
    })
}

fn render(value: &JsonValue) -> String {
    match value {
        JsonValue::Object(map) => {
            let parts: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}:{}", render_string(k), render(v)))
                .collect();
            format!("{{{}}}", parts.join(","))
        }
        JsonValue::Array(items) => {
            let parts: Vec<String> = items.iter().map(render).collect();
            format!("[{}]", parts.join(","))
        }
        JsonValue::String(s) => render_string(s),
        JsonValue::Integer(v) => v.to_string(),
        JsonValue::UnsignedInteger(v) => v.to_string(),
        JsonValue::Number(v) => v.to_string(),
        JsonValue::Bool(v) => v.to_string(),
        JsonValue::Null => "null".to_string(),
    }
}

fn render_string(s: &str) -> String {
    serde_json::to_string(s).unwrap_or_else(|_| "\"\"".to_string())
}
