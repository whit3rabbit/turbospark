//! ChatML (Qwen) stream event consumption for StructuredAssistantDecoder.

use super::{Channel, StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::tool_call::QwenToolCallParser;

impl<'a> StructuredAssistantDecoder<'a> {
    pub(super) fn consume_chatml(
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
}
