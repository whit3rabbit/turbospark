//! Inkling 2's reply frame, one message at a time:
//!
//! ```text
//! <|message_model|><|content_thinking|>REASONING<|end_message|>
//! <|message_model|><|content_text|>ANSWER<|end_message|>
//! <|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{..}}<|end_message|>
//! ```
//!
//! Between `<|message_model|>` and the content marker sits the recipient (the
//! tool's name on a call, nothing otherwise). It is framing, not answer text.
//! The shape is the checkpoint's template for a prior call, confirmed against
//! a real install's own output (UD-IQ1_S, 2026-11-26).

use super::{Channel, StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::json_value::JsonValue;
use crate::tool_call::{ParsedToolCall, MAXIMUM_BYTES};

const MODEL: &str = "<|message_model|>";
const TEXT: &str = "<|content_text|>";
const INVOKE: &str = "<|content_invoke_tool_json|>";

impl StructuredAssistantDecoder<'_> {
    /// Whether the generation prompt stopped right after `<|message_model|>`,
    /// so the reply opens in the recipient position.
    pub(super) fn prompt_opens_inkling2_header(&self, prompt_ids: &[i32]) -> bool {
        prompt_ids
            .last()
            .is_some_and(|&id| self.inkling2_mark(MODEL) == Some(id))
    }

    fn inkling2_mark(&self, mark: &str) -> Option<i32> {
        self.tokenizer.token_to_id(mark)
    }

    pub(super) fn consume_inkling2(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let is = |mark: &str| self.inkling2_mark(mark) == Some(token_id);
        let (model, text, invoke) = (is(MODEL), is(TEXT), is(INVOKE));
        let thinking = self.tokenizer.think_start_id == Some(token_id);
        let end = self.tokenizer.think_end_id == Some(token_id);
        if model {
            self.label.clear();
            self.inkling2_header = true;
            return Ok(Vec::new());
        }
        if text || thinking || invoke {
            self.inkling2_header = false;
            self.label.clear();
            if invoke {
                if let Some(open) = self.tool_tokens.take() {
                    self.failed_span_tokens = Some(open);
                    self.failed = true;
                    return Err(ToolCallParserError::Malformed);
                }
                self.tool_tokens = Some(Vec::new());
            } else {
                self.channel = if thinking {
                    Channel::Thought
                } else {
                    Channel::Visible
                };
            }
            return Ok(Vec::new());
        }
        if end {
            self.inkling2_header = false;
            self.channel = Channel::Visible;
            return match self.tool_tokens.take() {
                Some(tokens) => self.close_inkling2_call(tokens),
                None => Ok(Vec::new()),
            };
        }
        if let Some(tokens) = &mut self.tool_tokens {
            if (tokens.len() + 1) * 4 > MAXIMUM_BYTES {
                self.failed_span_tokens = Some(tokens.clone());
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            tokens.push(token_id);
            return Ok(Vec::new());
        }
        if self.inkling2_header {
            // Kept, not dropped: `finish` gives it back if no content marker
            // ever follows.
            self.label.push_str(delta);
            if self.label.len() > MAXIMUM_BYTES {
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            return Ok(Vec::new());
        }
        if delta.is_empty() {
            return Ok(Vec::new());
        }
        Ok(vec![if self.channel == Channel::Thought {
            StructuredAssistantEvent::Reasoning(delta.to_string())
        } else {
            StructuredAssistantEvent::Content(delta.to_string())
        }])
    }

    fn close_inkling2_call(
        &mut self,
        tokens: Vec<i32>,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let body = self.tokenizer.decode(&tokens, false);
        match parse_call(&body, &self.allowed_tools, (self.id_generator)()) {
            Ok(call) => {
                self.emitted_calls += 1;
                Ok(vec![StructuredAssistantEvent::ToolCall(call)])
            }
            Err(e) => {
                self.failed_span_tokens = Some(tokens);
                self.failed = true;
                Err(e)
            }
        }
    }

    /// At end of turn: an unclosed call is an error, and a recipient with no
    /// content after it was ordinary text all along.
    pub(super) fn finish_inkling2(
        &mut self,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if self.tool_tokens.is_some() {
            return Err(ToolCallParserError::Malformed);
        }
        let held = std::mem::take(&mut self.label);
        Ok(if self.inkling2_header && !held.is_empty() {
            vec![StructuredAssistantEvent::Content(held)]
        } else {
            Vec::new()
        })
    }
}

/// `{"name": .., "args": {..}}`, nothing else, naming an offered tool.
fn parse_call(
    body: &str,
    allowed: &std::collections::HashSet<String>,
    id: String,
) -> Result<ParsedToolCall, ToolCallParserError> {
    let JsonValue::Object(mut call) = JsonValue::parse(body.trim())? else {
        return Err(ToolCallParserError::Malformed);
    };
    let Some(JsonValue::String(name)) = call.remove("name") else {
        return Err(ToolCallParserError::Malformed);
    };
    let Some(arguments @ JsonValue::Object(_)) = call.remove("args") else {
        return Err(ToolCallParserError::Malformed);
    };
    if !call.is_empty() || !allowed.contains(&name) {
        return Err(ToolCallParserError::Malformed);
    }
    let arguments_json = arguments.encoded();
    Ok(ParsedToolCall {
        id,
        name,
        arguments,
        arguments_json,
    })
}
