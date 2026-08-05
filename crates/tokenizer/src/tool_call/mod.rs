//! Tool-call parsers for the three supported dialects: Gemma's custom DSL,
//! Qwen's ChatML `<function=...>` framing, and DeepSeek's DSML markers.

mod deepseek;
mod gemma;
mod qwen;

pub use deepseek::{DeepseekToolCallParser, DSML_MARK};
pub use gemma::GemmaToolCallParser;
pub use qwen::QwenToolCallParser;

pub fn deepseek_dsml_mark() -> String {
    DSML_MARK.to_string()
}

use crate::json_value::JsonValue;

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedToolCall {
    pub id: String,
    pub name: String,
    pub arguments: JsonValue,
    pub arguments_json: String,
}

/// Byte ceiling shared by all three tool-call parsers, matching the Swift
/// `maximumBytes` constants.
pub const MAXIMUM_BYTES: usize = 256 * 1024;

fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}
