//! Streaming assistant-output decoder: splits generated tokens/text into
//! visible content and parsed tool calls per dialect. Ported from
//! `Tokenization/StructuredAssistantDecoder.swift`.

mod chatml;
mod deepseek;
mod harmony;

use std::collections::HashSet;

use self::harmony::{HarmonyChannel, HarmonyState};
use crate::dialect::{ChatDialect, MfTokenizer};
use crate::error::ToolCallParserError;
use crate::tool_call::{GemmaToolCallParser, ParsedToolCall};

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
            self.harmony = HarmonyState::Body(harmony::harmony_channel(&self.label));
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
