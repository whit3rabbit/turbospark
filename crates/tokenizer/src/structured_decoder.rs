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
    /// Reasoning the model produced on its way to the answer, separated from
    /// the answer itself.
    ///
    /// **Only the Harmony arm emits this**, and the asymmetry is deliberate.
    /// Every other dialect here DISCARDS its thought channel (Gemma's
    /// non-final label, ChatML's `<think>`, DeepSeek's), which is what four
    /// shipped families' callers already see; turning those into events is a
    /// separate decision with its own gates. Harmony emits because
    /// `gpt-oss` puts most of its generated tokens in that channel and a
    /// caller that paid to generate them should be able to read them.
    Reasoning(String),
    /// Parsed tool call invocation.
    ToolCall(ParsedToolCall),
}

#[derive(PartialEq)]
enum Channel {
    Thought,
    Visible,
    Label,
}

/// Which Harmony channel a message body belongs to.
///
/// Only `final` is the answer. `analysis` is the model's reasoning, and
/// `commentary` carries tool calls and preambles; both are reported as
/// reasoning rather than as content, so nothing but the final channel is ever
/// presented as the reply.
#[derive(Clone, Copy, PartialEq)]
enum HarmonyChannel {
    Final,
    Reasoning,
}

/// Position in Harmony's `<|channel|>HEADER<|message|>BODY<|end|>` frame.
///
/// Harmony is the one dialect here whose channels do not BRACKET: `<|channel|>`
/// opens a header, `<|message|>` ends that header and opens the body, and
/// `<|end|>` closes the body. So the start/end token pair every other arm keys
/// on cannot express it, and reusing the Gemma arm would open a channel label
/// on the first `<|channel|>` and never close it, swallowing the whole reply.
enum HarmonyState {
    /// Before the first `<|channel|>`. Text passes through as content, so a
    /// model that never emits the frame is reported rather than silenced.
    Unframed,
    /// Inside a header, accumulating its text.
    Header,
    /// Inside a message body.
    Body(HarmonyChannel),
    /// After `<|end|>` and before the next `<|channel|>`. Everything here is
    /// dropped: what falls in this gap is the `<|start|>assistant` that opens
    /// the next message, and passing it through would emit the bare word
    /// "assistant" into the reply.
    Between,
}

/// Streaming assistant output decoder splitting tokens into visible content and tool calls.
pub struct StructuredAssistantDecoder<'a> {
    tokenizer: &'a MfTokenizer,
    allowed_tools: HashSet<String>,
    id_generator: Box<dyn FnMut() -> String + 'a>,
    channel: Channel,
    harmony: HarmonyState,
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
            harmony: HarmonyState::Unframed,
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
            ChatDialect::Harmony => return Ok(self.consume_harmony(token_id, delta)),
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

    /// `gpt-oss`'s Harmony frame. A real assistant turn reads
    ///
    /// ```text
    /// <|channel|>analysis<|message|>REASONING<|end|>
    /// <|start|>assistant<|channel|>final<|message|>ANSWER<|return|>
    /// ```
    ///
    /// **Every transition keys on a TOKEN ID, never on text**, the way the
    /// Gemma arm does: that is what makes the state machine independent of
    /// whether the detokenizer renders special tokens, and of how the header's
    /// words happen to be split into tokens.
    ///
    /// The three tokens that CLOSE a turn -- `<|return|>`, `<|call|>` and
    /// `<|endoftext|>` -- never reach here at all. They are in the dialect's
    /// stop set, and `run_raw_completion` breaks before the progress callback,
    /// so the only frame token this sees inside a body is `<|end|>`, which is
    /// deliberately not a stop (it also closes the PROMPT's system and user
    /// turns).
    ///
    /// ONE ORDERING CONSTRAINT IS INHERITED RATHER THAN ENFORCED HERE. A
    /// consumer mapping [`StructuredAssistantEvent::Reasoning`] onto an
    /// Anthropic `thinking` block gets a well-formed stream because Harmony
    /// emits `analysis` BEFORE `final` within one response. This arm reports
    /// whatever order the model produced; it does not reorder to protect a
    /// downstream state machine.
    fn consume_harmony(&mut self, token_id: i32, delta: &str) -> Vec<StructuredAssistantEvent> {
        if token_id == self.tokenizer.channel_start_id {
            self.label.clear();
            self.harmony = HarmonyState::Header;
            return Vec::new();
        }
        if token_id == self.tokenizer.message_start_id {
            self.harmony = HarmonyState::Body(harmony_channel(&self.label));
            self.label.clear();
            return Vec::new();
        }
        if token_id == self.tokenizer.message_end_id {
            self.harmony = HarmonyState::Between;
            return Vec::new();
        }
        match self.harmony {
            HarmonyState::Header => {
                self.label.push_str(delta);
                Vec::new()
            }
            HarmonyState::Between => Vec::new(),
            HarmonyState::Body(_) | HarmonyState::Unframed if delta.is_empty() => Vec::new(),
            HarmonyState::Body(HarmonyChannel::Reasoning) => {
                vec![StructuredAssistantEvent::Reasoning(delta.to_string())]
            }
            HarmonyState::Body(HarmonyChannel::Final) | HarmonyState::Unframed => {
                vec![StructuredAssistantEvent::Content(delta.to_string())]
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

/// Which channel a Harmony message header names.
///
/// A header is the text between `<|channel|>` and `<|message|>`, and it is not
/// always one word: a tool call reads `commentary to=functions.get_weather
/// <|constrain|>json`. The channel is the first whitespace-delimited word, and
/// the rest is deliberately IGNORED here -- Harmony frames a tool call as a
/// recipient in this header rather than as the bracketing token pair
/// [`StructuredAssistantDecoder`]'s tool contract describes, which is why
/// `resolve_harmony` leaves every tool id `NO_SUCH_TOKEN_ID` and why decoding
/// them is its own item.
///
/// ANYTHING THAT IS NOT `final` IS REASONING, including an unrecognized
/// channel name. The default direction matters: a new channel misreported as
/// reasoning is visible in the wrong place, while one misreported as the
/// answer corrupts the reply.
fn harmony_channel(header: &str) -> HarmonyChannel {
    match header.split_whitespace().next() {
        Some("final") => HarmonyChannel::Final,
        _ => HarmonyChannel::Reasoning,
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
