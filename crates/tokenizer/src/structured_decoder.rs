//! Streaming assistant-output decoder: splits generated tokens/text into
//! visible content and parsed tool calls per dialect. Ported from
//! `Tokenization/StructuredAssistantDecoder.swift`.

use std::collections::HashSet;

use crate::dialect::{ChatDialect, MfTokenizer};
use crate::error::ToolCallParserError;
use crate::tool_call::{
    DeepseekToolCallParser, GemmaToolCallParser, ParsedToolCall, QwenToolCallParser, DSML_MARK,
};

/// Decoded event emitted by the structured assistant output decoder.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuredAssistantEvent {
    /// Text content chunk for display.
    Content(String),
    /// Parsed tool call invocation.
    ToolCall(ParsedToolCall),
}

#[derive(PartialEq)]
enum Channel {
    Thought,
    Visible,
    Label,
}

/// Streaming assistant output decoder splitting tokens into visible content and tool calls.
pub struct StructuredAssistantDecoder<'a> {
    tokenizer: &'a MfTokenizer,
    allowed_tools: HashSet<String>,
    id_generator: Box<dyn FnMut() -> String + 'a>,
    channel: Channel,
    label: String,
    tool_tokens: Option<Vec<i32>>,
    held_text: String,
    dsml_text: Option<String>,
    emitted_calls: usize,
    failed: bool,
}

impl<'a> StructuredAssistantDecoder<'a> {
    /// Creates a structured assistant decoder.
    pub fn new(
        tokenizer: &'a MfTokenizer,
        allowed_tools: HashSet<String>,
        id_generator: impl FnMut() -> String + 'a,
    ) -> Self {
        Self {
            tokenizer,
            allowed_tools,
            id_generator: Box::new(id_generator),
            channel: Channel::Visible,
            label: String::new(),
            tool_tokens: None,
            held_text: String::new(),
            dsml_text: None,
            emitted_calls: 0,
            failed: false,
        }
    }

    /// Returns true if at least one tool call has been parsed and emitted.
    pub fn has_tool_calls(&self) -> bool {
        self.emitted_calls > 0
    }

    /// Consumes flushed text snippet during stream decoding.
    pub fn consume_flushed_text(
        &mut self,
        text: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if text.is_empty() {
            return Ok(Vec::new());
        }
        self.consume(-1, text)
    }

    /// Consumes a token ID and text delta, returning any parsed events.
    pub fn consume(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if self.failed {
            return Err(ToolCallParserError::Malformed);
        }
        match self.tokenizer.dialect {
            ChatDialect::ChatMl => return self.consume_chatml(token_id, delta),
            ChatDialect::Deepseek => return self.consume_deepseek(token_id, delta),
            ChatDialect::Gemma => {}
            // Nothing to decode: this checkpoint has no tool-call or
            // thinking markup, and its channel/tool ids are all
            // `NO_SUCH_TOKEN_ID`. Falling through to the Gemma arm would
            // compare every token against that sentinel, which is harmless
            // but says something untrue about the dialect.
            ChatDialect::Mistral => {
                return Ok(if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![StructuredAssistantEvent::Content(delta.to_string())]
                })
            }
            // HARMONY HAS CHANNELS AND THIS DECODER DOES NOT READ THEM YET
            // (ROADMAP M5). Content passes through, so the model's
            // `analysis` channel reaches the caller as text instead of being
            // split off as reasoning. That is a stated limitation, not an
            // oversight, and it is strictly better than the alternative:
            // Harmony's `channel_start_id` IS a real token, but it has no
            // closing counterpart (`channel_end_id` is `NO_SUCH_TOKEN_ID`),
            // so falling through to the Gemma arm would open a channel label
            // on the first `<|channel|>` and never close it -- swallowing the
            // whole reply. Wiring it needs a header parser rather than a
            // token pair, which is its own item.
            ChatDialect::Harmony => {
                return Ok(if delta.is_empty() {
                    Vec::new()
                } else {
                    vec![StructuredAssistantEvent::Content(delta.to_string())]
                })
            }
        }

        if token_id == self.tokenizer.channel_start_id {
            self.label.clear();
            self.channel = Channel::Label;
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.channel_end_id {
            self.channel = Channel::Visible;
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.tool_call_start_id {
            if self.tool_tokens.is_some() {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            self.tool_tokens = Some(Vec::new());
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.tool_call_end_id {
            let Some(tokens) = self.tool_tokens.take() else {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            };
            let text = self.tokenizer.decode(&tokens, false);
            match GemmaToolCallParser::new().parse(
                &text,
                &self.allowed_tools,
                &(self.id_generator)(),
            ) {
                Ok(call) => {
                    self.emitted_calls += 1;
                    return Ok(vec![StructuredAssistantEvent::ToolCall(call)]);
                }
                Err(e) => {
                    self.failed = true;
                    return Err(e);
                }
            }
        }
        if token_id == self.tokenizer.tool_response_id
            || token_id == self.tokenizer.tool_response_end_id
        {
            if self.emitted_calls == 0 || self.tool_tokens.is_some() {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            return Ok(Vec::new());
        }
        if let Some(tokens) = &mut self.tool_tokens {
            tokens.push(token_id);
            if tokens.len() * 4 > crate::tool_call::MAXIMUM_BYTES {
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            return Ok(Vec::new());
        }
        match self.channel {
            Channel::Thought => Ok(Vec::new()),
            Channel::Visible => {
                if delta.is_empty() {
                    Ok(Vec::new())
                } else {
                    Ok(vec![StructuredAssistantEvent::Content(delta.to_string())])
                }
            }
            Channel::Label => {
                self.label.push_str(delta);
                let Some(newline) = self.label.find('\n') else {
                    return Ok(Vec::new());
                };
                let name = self.label[..newline].trim().to_lowercase();
                let content = self.label[newline + 1..].to_string();
                self.channel = if name == "final" || name == "answer" {
                    Channel::Visible
                } else {
                    Channel::Thought
                };
                self.label.clear();
                if self.channel == Channel::Visible && !content.is_empty() {
                    Ok(vec![StructuredAssistantEvent::Content(content)])
                } else {
                    Ok(Vec::new())
                }
            }
        }
    }

    fn consume_chatml(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if token_id == self.tokenizer.tool_call_start_id {
            if self.tool_tokens.is_some() {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            self.tool_tokens = Some(Vec::new());
            return Ok(Vec::new());
        }
        if token_id == self.tokenizer.tool_call_end_id {
            let Some(tokens) = self.tool_tokens.take() else {
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            };
            let text = self.tokenizer.decode(&tokens, false);
            match QwenToolCallParser::new().parse(
                &text,
                &self.allowed_tools,
                &(self.id_generator)(),
            ) {
                Ok(call) => {
                    self.emitted_calls += 1;
                    return Ok(vec![StructuredAssistantEvent::ToolCall(call)]);
                }
                Err(e) => {
                    self.failed = true;
                    return Err(e);
                }
            }
        }
        if let Some(tokens) = &mut self.tool_tokens {
            tokens.push(token_id);
            if tokens.len() * 4 > crate::tool_call::MAXIMUM_BYTES {
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            return Ok(Vec::new());
        }
        if Some(token_id) == self.tokenizer.think_start_id {
            self.channel = Channel::Thought;
            return Ok(Vec::new());
        }
        if Some(token_id) == self.tokenizer.think_end_id {
            self.channel = Channel::Visible;
            return Ok(Vec::new());
        }
        if self.channel == Channel::Thought {
            return Ok(Vec::new());
        }
        if delta.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![StructuredAssistantEvent::Content(delta.to_string())])
        }
    }

    fn consume_deepseek(
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

    /// Release any tail withheld as a potential DSML-open prefix.
    pub fn drain(&mut self) -> Vec<StructuredAssistantEvent> {
        if self.failed || self.dsml_text.is_some() || self.held_text.is_empty() {
            return Vec::new();
        }
        let visible = std::mem::take(&mut self.held_text);
        vec![StructuredAssistantEvent::Content(visible)]
    }

    pub fn finish(&mut self) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let released = self.drain();
        if self.failed || self.tool_tokens.is_some() || self.dsml_text.is_some() {
            return Err(ToolCallParserError::Malformed);
        }
        Ok(released)
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
