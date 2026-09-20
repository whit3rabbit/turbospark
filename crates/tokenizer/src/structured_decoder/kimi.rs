//! The Kimi K2 tool-call decoder arm.
//!
//! Kimi's section and call markers are ADDED but NOT SPECIAL in the
//! checkpoint's table (`<|tool_call_begin|>` is id 163597 with
//! `special: false`), so they survive detokenization as literal text and the
//! arm is a TEXT-MARKER scan -- DeepSeek's DSML arm, not the id-bracket arm.
//! A turn's calls sit inside a `<|tool_calls_section_begin|>` /
//! `<|tool_calls_section_end|>` wrapper; each call is
//! `<|tool_call_begin|>ID<|tool_call_argument_begin|>{json}<|tool_call_end|>`
//! and its body goes to [`crate::tool_call::KimiToolCallParser`], which cuts
//! the NAME out of the `functions.NAME:IDX` id. Prose around and between the
//! markers is ordinary content.
//!
//! **THE WHOLE TURN'S CALLS LIVE UNDER ONE SECTION**, but the arm does not
//! track section state: the wrapper markers are swallowed wherever they
//! appear and calls are parsed on their OWN begin/end pairs, so a model that
//! drops the wrapper (or emits two sections) still parses -- the wrapper is
//! framing, the call pair is the call.
//!
//! Like the GLM arm the scan withholds a buffer tail that could be the front
//! of any idle mark; on the per-token path the markers arrive whole and the
//! withholding returns zero.

use super::deepseek::char_index_to_byte;
use super::glm::partial_mark_prefix_length;
use super::{Channel, StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::tool_call::{
    KimiToolCallParser, KIMI_CALL_BEGIN_MARK, KIMI_CALL_END_MARK, KIMI_SECTION_BEGIN_MARK,
    KIMI_SECTION_END_MARK,
};

/// The marks the arm scans for while no call is open. A partial match at the
/// tail of the buffer is withheld until the next delta says which mark (if
/// any) it was growing into.
const IDLE_MARKS: [&str; 4] = [
    KIMI_SECTION_BEGIN_MARK,
    KIMI_SECTION_END_MARK,
    KIMI_CALL_BEGIN_MARK,
    KIMI_CALL_END_MARK,
];

impl<'a> StructuredAssistantDecoder<'a> {
    /// The Kimi arm. Reasoning follows the ChatML rule -- EMITTED, not
    /// discarded, because this dialect's generation prompt forces the think
    /// frame open and `--reasoning` exists.
    pub(super) fn consume_kimi(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if Some(token_id) == self.tokenizer.think_start_id
            || Some(token_id) == self.tokenizer.think_end_id
        {
            self.channel = if Some(token_id) == self.tokenizer.think_start_id {
                Channel::Thought
            } else {
                Channel::Visible
            };
            if self.dsml_text.is_none() && !self.held_text.is_empty() {
                let visible = std::mem::take(&mut self.held_text);
                return Ok(vec![StructuredAssistantEvent::Content(visible)]);
            }
            return Ok(Vec::new());
        }
        if self.channel == Channel::Thought {
            return Ok(if delta.is_empty() {
                Vec::new()
            } else {
                vec![StructuredAssistantEvent::Reasoning(delta.to_string())]
            });
        }
        if delta.is_empty() {
            return Ok(Vec::new());
        }
        self.held_text.push_str(delta);
        let mut events = Vec::new();
        'scanning: while !self.held_text.is_empty() {
            if let Some(call) = &mut self.dsml_text {
                call.push_str(&self.held_text);
                self.held_text.clear();
                let Some(close_pos) = call.find(KIMI_CALL_END_MARK) else {
                    if call.len() > crate::tool_call::MAXIMUM_BYTES {
                        self.failed = true;
                        return Err(ToolCallParserError::Oversized);
                    }
                    break 'scanning;
                };
                let body = call[..close_pos].to_string();
                self.held_text = call[close_pos + KIMI_CALL_END_MARK.len()..].to_string();
                self.dsml_text = None;
                match KimiToolCallParser::new().parse(&body, &self.allowed_tools) {
                    Ok(parsed) => {
                        self.emitted_calls += 1;
                        debug_assert_eq!(
                            self.tokenizer.dialect.tool_call_support(),
                            crate::ToolCallSupport::Native,
                            "a dialect that hands a parsed tool call over must answer Native to \
                             `tool_call_support`; the two matches have drifted"
                        );
                        events.push(StructuredAssistantEvent::ToolCall(parsed));
                    }
                    // The shipped text-marker arm (DeepSeek's DSML) drops a
                    // refused body: the markup was consumed into the call
                    // buffer, so unlike the id-bracket arms there are no
                    // span TOKENS to release through
                    // `take_failed_span_text`.
                    Err(e) => {
                        self.failed = true;
                        return Err(e);
                    }
                }
                continue 'scanning;
            }
            // No call open: find the EARLIEST idle mark and act on it.
            let found = IDLE_MARKS
                .iter()
                .filter_map(|m| self.held_text.find(m).map(|pos| (pos, *m)))
                .min_by_key(|(pos, _)| *pos);
            match found {
                Some((pos, KIMI_CALL_BEGIN_MARK)) => {
                    let visible = self.held_text[..pos].to_string();
                    if !visible.is_empty() {
                        events.push(StructuredAssistantEvent::Content(visible));
                    }
                    self.held_text = self.held_text[pos + KIMI_CALL_BEGIN_MARK.len()..].to_string();
                    self.dsml_text = Some(String::new());
                    continue 'scanning;
                }
                // A wrapper marker is framing, not content and not a call:
                // swallow it and keep scanning.
                Some((pos, mark @ (KIMI_SECTION_BEGIN_MARK | KIMI_SECTION_END_MARK))) => {
                    let visible = self.held_text[..pos].to_string();
                    if !visible.is_empty() {
                        events.push(StructuredAssistantEvent::Content(visible));
                    }
                    self.held_text = self.held_text[pos + mark.len()..].to_string();
                    continue 'scanning;
                }
                // A call close with no call open: the model confused its own
                // syntax. Malformed, the same verdict the id-bracket arms
                // reach on a stray end token.
                Some((_, KIMI_CALL_END_MARK)) => {
                    self.failed = true;
                    return Err(ToolCallParserError::Malformed);
                }
                // Unreachable: the four IDLE_MARKS are all matched above.
                Some((_, _)) => unreachable!("an unmatched idle mark"),
                None => {}
            }
            let held = partial_mark_prefix_length(&self.held_text, &IDLE_MARKS);
            let total = self.held_text.chars().count();
            if held < total {
                let split_at_char = total - held;
                let byte_idx = char_index_to_byte(&self.held_text, split_at_char);
                events.push(StructuredAssistantEvent::Content(
                    self.held_text[..byte_idx].to_string(),
                ));
                self.held_text = self.held_text[byte_idx..].to_string();
            }
            break 'scanning;
        }
        Ok(events)
    }
}
