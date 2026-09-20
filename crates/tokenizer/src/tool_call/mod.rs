//! Tool-call parsers for the supported dialects: Gemma's custom DSL, Qwen's
//! ChatML `<function=...>` framing, DeepSeek's DSML markers, Mistral's
//! `[TOOL_CALLS]` JSON array, GLM's `<arg_key>`/`<arg_value>` XML, and Kimi
//! K2's section markers around a `functions.NAME:IDX` id.

mod deepseek;
mod gemma;
mod glm;
mod kimi;
mod mistral;
mod qwen;

pub use deepseek::{DeepseekToolCallParser, DSML_MARK};
pub use gemma::GemmaToolCallParser;
pub use glm::{GlmToolCallParser, GLM_TOOL_CALL_CLOSE_MARK, GLM_TOOL_CALL_OPEN_MARK};
pub use kimi::{
    KimiToolCallParser, KIMI_CALL_BEGIN_MARK, KIMI_CALL_END_MARK, KIMI_SECTION_BEGIN_MARK,
    KIMI_SECTION_END_MARK,
};
pub use mistral::MistralToolCallParser;
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

/// The shared name check every dialect applies before emitting a call. Also
/// read by the Harmony arm of [`crate::structured_decoder`], which resolves a
/// name out of a namespaced header recipient rather than out of a DSL.
pub(crate) fn is_valid_function_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
}
