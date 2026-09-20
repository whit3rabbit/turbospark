//! The GLM `<tool_call>` / `<arg_key>` / `<arg_value>` decoder arm.
//!
//! GLM's tool markup is ADDED but NOT SPECIAL in the checkpoint's table
//! (`<tool_call>` is id 154843 with `special: false`), so unlike Gemma's or
//! ChatML's brackets it SURVIVES detokenization as literal text and the arm
//! is a TEXT-MARKER scan -- DeepSeek's DSML arm, not the id-bracket arm. One
//! call is a `<tool_call>` ... `</tool_call>` block; a turn may carry
//! several, and any prose around them is ordinary content. The body inside a
//! block goes to [`crate::tool_call::GlmToolCallParser`].
//!
//! **TEXT MARKERS SPLIT ACROSS DELTAS ON THE FLUSHED-TEXT PATH**, so the
//! scan withholds a tail that could be the front of the open mark -- the
//! same partial-prefix logic the DSML arm runs, generalized to a mark SET
//! because `</tool_call>` does not start with `<tool_call>` and a stray
//! close must be found too. On the per-token path the markers arrive whole
//! (an added token is one id) and the withholding returns zero.

use super::deepseek::char_index_to_byte;
use super::{Channel, StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::tool_call::{
    GlmToolCallParser, GLM_TOOL_CALL_CLOSE_MARK, GLM_TOOL_CALL_OPEN_MARK,
};

/// The marks the arm scans for while no block is open. A partial match at
/// the tail of the buffer is withheld until the next delta says which mark
/// (if any) it was growing into.
const IDLE_MARKS: [&str; 2] = [GLM_TOOL_CALL_OPEN_MARK, GLM_TOOL_CALL_CLOSE_MARK];

impl<'a> StructuredAssistantDecoder<'a> {
    /// The GLM arm. Reasoning follows the ChatML rule -- EMITTED, not
    /// discarded, because this dialect's generation prompt forces the think
    /// frame open and `--reasoning` exists -- and visible text runs through
    /// the block scanner below.
    pub(super) fn consume_glm(
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
            if let Some(block) = &mut self.dsml_text {
                block.push_str(&self.held_text);
                self.held_text.clear();
                let Some(close_pos) = block.find(GLM_TOOL_CALL_CLOSE_MARK) else {
                    if block.len() > crate::tool_call::MAXIMUM_BYTES {
                        self.failed = true;
                        return Err(ToolCallParserError::Oversized);
                    }
                    break 'scanning;
                };
                let body = block[..close_pos].to_string();
                self.held_text = block[close_pos + GLM_TOOL_CALL_CLOSE_MARK.len()..].to_string();
                self.dsml_text = None;
                let generator = &mut self.id_generator;
                match GlmToolCallParser::new().parse(&body, &self.allowed_tools, generator) {
                    Ok(call) => {
                        self.emitted_calls += 1;
                        debug_assert_eq!(
                            self.tokenizer.dialect.tool_call_support(),
                            crate::ToolCallSupport::Native,
                            "a dialect that hands a parsed tool call over must answer Native to \
                             `tool_call_support`; the two matches have drifted"
                        );
                        events.push(StructuredAssistantEvent::ToolCall(call));
                    }
                    Err(e) => {
                        // The shipped text-marker arm (DeepSeek's DSML)
                        // drops a refused body: the markup was consumed into
                        // the block buffer, so unlike the id-bracket arms
                        // there are no span TOKENS to release through
                        // `take_failed_span_text`.
                        self.failed = true;
                        return Err(e);
                    }
                }
                continue 'scanning;
            }
            let open_pos = self.held_text.find(GLM_TOOL_CALL_OPEN_MARK);
            let close_pos = self.held_text.find(GLM_TOOL_CALL_CLOSE_MARK);
            match (open_pos, close_pos) {
                (Some(open), Some(close)) if close < open => {
                    // A block closes with no block open: the model
                    // confused its own syntax. Malformed, the same verdict
                    // the id-bracket arms reach on a stray end token.
                    self.failed = true;
                    return Err(ToolCallParserError::Malformed);
                }
                (Some(open), _) => {
                    let visible = self.held_text[..open].to_string();
                    if !visible.is_empty() {
                        events.push(StructuredAssistantEvent::Content(visible));
                    }
                    self.held_text = self.held_text[open + GLM_TOOL_CALL_OPEN_MARK.len()..].to_string();
                    self.dsml_text = Some(String::new());
                    continue 'scanning;
                }
                (None, Some(_)) => {
                    self.failed = true;
                    return Err(ToolCallParserError::Malformed);
                }
                (None, None) => {}
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

/// Length (in chars) of the longest suffix of `text` that is a proper prefix
/// of ANY of `marks`. The DSML arm's `open_marker_prefix_length` for a set
/// of marks, which the GLM scan needs because its two marks do not share a
/// prefix (`</tool_call>` starts with `</`).
pub(super) fn partial_mark_prefix_length(text: &str, marks: &[&str]) -> usize {
    let text_chars: Vec<char> = text.chars().collect();
    let longest = text_chars
        .len()
        .min(marks.iter().map(|m| m.chars().count().saturating_sub(1)).max().unwrap_or(0));
    if longest == 0 {
        return 0;
    }
    for length in (1..=longest).rev() {
        let suffix = &text_chars[text_chars.len() - length..];
        if marks.iter().any(|m| {
            let mark_chars: Vec<char> = m.chars().collect();
            mark_chars.starts_with(suffix)
        }) {
            return length;
        }
    }
    0
}
