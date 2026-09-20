//! Kimi K2 tool-call body parser: the text between a
//! `<|tool_call_begin|>` and `<|tool_call_end|>` marker pair, which reads
//! `functions.NAME:IDX<|tool_call_argument_begin|>{json}`. The grammar is
//! read off `moonshotai/Kimi-K2.5`'s own `chat_template.jinja`
//! (`render_toolcalls`, 2026-09-19) plus Moonshot's tool-use guidance, which
//! is where the `functions.NAME:IDX` id convention is named: the call's
//! NAME lives INSIDE its id string, joined to the per-turn index with a
//! colon.

use std::collections::HashSet;

use super::{is_valid_function_name, ParsedToolCall, MAXIMUM_BYTES};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;

/// The section wrapper around a whole turn's calls. ADDED but not special in
/// the real table (ids 163595 / 163596), so both strings survive
/// detokenization as literal text; the streaming decoder swallows them and
/// parses calls off the inner markers.
pub const KIMI_SECTION_BEGIN_MARK: &str = "<|tool_calls_section_begin|>";
pub const KIMI_SECTION_END_MARK: &str = "<|tool_calls_section_end|>";

/// One call's brackets and the marker that splits its id from its JSON
/// arguments (ids 163597 / 163598 / 163599, added but not special for the
/// same reason).
pub const KIMI_CALL_BEGIN_MARK: &str = "<|tool_call_begin|>";
pub const KIMI_CALL_ARGUMENT_MARK: &str = "<|tool_call_argument_begin|>";
pub const KIMI_CALL_END_MARK: &str = "<|tool_call_end|>";

/// Kimi K2 tool-call body parser.
pub struct KimiToolCallParser;

impl KimiToolCallParser {
    /// Creates a new `KimiToolCallParser`.
    pub fn new() -> Self {
        Self
    }

    /// Parses ONE call's body -- the text between [`KIMI_CALL_BEGIN_MARK`]
    /// and [`KIMI_CALL_END_MARK`], bracketing stripped by the caller --
    /// against an allowed set of tool names. The body is the call id, the
    /// [`KIMI_CALL_ARGUMENT_MARK`], and the arguments as one JSON object.
    ///
    /// **THE CHECKPOINT'S OWN ID IS KEPT, NOT REPLACED.** `functions.NAME:IDX`
    /// is what the model matches a tool result against (the template renders
    /// a result as `## Return of {id}`), so the parsed call carries it
    /// verbatim rather than a generated id, and the NAME is cut out of it:
    /// the optional `functions.` prefix is dropped and everything after the
    /// LAST colon is the index. An id with no name part in it is malformed.
    pub fn parse(
        &self,
        body: &str,
        allowed_tools: &HashSet<String>,
    ) -> Result<ParsedToolCall, ToolCallParserError> {
        if body.len() > MAXIMUM_BYTES {
            return Err(ToolCallParserError::Oversized);
        }
        let body = body.trim();
        let (call_id, arguments_json) = body
            .split_once(KIMI_CALL_ARGUMENT_MARK)
            .ok_or(ToolCallParserError::Malformed)?;
        let call_id = call_id.trim();
        let name = call_id
            .strip_prefix("functions.")
            .unwrap_or(call_id)
            // The index is joined by a colon and cannot contain one; the
            // LAST colon is the separator, so a namespaced name like
            // `functions.browser:open:0` still cuts correctly.
            .rsplit_once(':')
            .map(|(name, _)| name)
            .unwrap_or(call_id);
        if !is_valid_function_name(name) {
            return Err(ToolCallParserError::Malformed);
        }
        if !allowed_tools.contains(name) {
            return Err(ToolCallParserError::UnknownTool(name.to_string()));
        }
        // The arguments are the JSON OBJECT the template renders with
        // `tojson` -- not a JSON-encoded string, which is ChatML's shape. An
        // array, a bare scalar, or a truncated object is malformed, the same
        // verdict the Gemma arm reaches on an unterminated body.
        let arguments = match JsonValue::parse(arguments_json.trim()) {
            Ok(value @ JsonValue::Object(_)) => value,
            _ => return Err(ToolCallParserError::Malformed),
        };
        Ok(ParsedToolCall {
            id: call_id.to_string(),
            name: name.to_string(),
            arguments_json: arguments.encoded(),
            arguments,
        })
    }
}

impl Default for KimiToolCallParser {
    fn default() -> Self {
        Self::new()
    }
}
