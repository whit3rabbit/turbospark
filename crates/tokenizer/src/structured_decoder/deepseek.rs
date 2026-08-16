//! DeepSeek stream event consumption and DSML buffering for StructuredAssistantDecoder.

use super::{Channel, StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::tool_call::{DeepseekToolCallParser, DSML_MARK};

impl<'a> StructuredAssistantDecoder<'a> {
    pub(super) fn consume_deepseek(
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
        if self.channel == Channel::Thought || delta.is_empty() {
            return Ok(Vec::new());
        }
        self.held_text.push_str(delta);
        let mut events = Vec::new();
        let open_mark = format!("<{DSML_MARK}tool_calls>");
        let close_mark = format!("</{DSML_MARK}tool_calls>");
        'scanning: while !self.held_text.is_empty() {
            if let Some(dsml) = &mut self.dsml_text {
                dsml.push_str(&self.held_text);
                self.held_text.clear();
                let Some(close_pos) = dsml.find(&close_mark) else {
                    if dsml.len() > crate::tool_call::MAXIMUM_BYTES {
                        self.failed = true;
                        return Err(ToolCallParserError::Oversized);
                    }
                    break 'scanning;
                };
                let body = dsml[..close_pos].to_string();
                self.held_text = dsml[close_pos + close_mark.len()..].to_string();
                self.dsml_text = None;
                let generator = &mut self.id_generator;
                match DeepseekToolCallParser::new().parse(&body, &self.allowed_tools, generator) {
                    Ok(calls) => {
                        self.emitted_calls += calls.len();
                        events.extend(calls.into_iter().map(StructuredAssistantEvent::ToolCall));
                    }
                    Err(e) => {
                        self.failed = true;
                        return Err(e);
                    }
                }
                continue 'scanning;
            }
            if let Some(open_pos) = self.held_text.find(&open_mark) {
                let visible = self.held_text[..open_pos].to_string();
                if !visible.is_empty() {
                    events.push(StructuredAssistantEvent::Content(visible));
                }
                self.held_text = self.held_text[open_pos + open_mark.len()..].to_string();
                self.dsml_text = Some(String::new());
                continue 'scanning;
            }
            let held = open_marker_prefix_length(&self.held_text, &open_mark);
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
/// of `open_mark`.
fn open_marker_prefix_length(text: &str, open_mark: &str) -> usize {
    let text_chars: Vec<char> = text.chars().collect();
    let mark_chars: Vec<char> = open_mark.chars().collect();
    let longest = text_chars.len().min(mark_chars.len().saturating_sub(1));
    if longest == 0 {
        return 0;
    }
    for length in (1..=longest).rev() {
        let suffix = &text_chars[text_chars.len() - length..];
        if mark_chars.starts_with(suffix) {
            return length;
        }
    }
    0
}

fn char_index_to_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}
