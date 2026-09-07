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
            if let Some(tokens) = self.tool_tokens.take() {
                // A second start inside an open span abandons the first one;
                // what it had buffered is still model output.
                self.failed_span_tokens = Some(tokens);
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
                    debug_assert_eq!(
                        self.tokenizer.dialect.tool_call_support(),
                        crate::ToolCallSupport::Native,
                        "a dialect that hands a parsed tool call over must answer Native to \
                         `tool_call_support`; the two matches have drifted"
                    );
                    return Ok(vec![StructuredAssistantEvent::ToolCall(call)]);
                }
                Err(e) => {
                    // The span's body was decoded for this parse and is what
                    // the caller most wants back: release it beside the
                    // error rather than drop it with the failed attempt.
                    // This is the pre-3.5 Qwen case: bare JSON inside the
                    // same special-token pair, refused here and recoverable
                    // by the server's rescue layer only if the text survives.
                    self.failed_span_tokens = Some(tokens);
                    self.failed = true;
                    return Err(e);
                }
            }
        }
        if let Some(tokens) = &mut self.tool_tokens {
            // Checked BEFORE pushing `token_id`, so the clone below -- the
            // span body `turn_stream::feed` releases on this error -- never
            // includes the token that tripped the limit. That token is
            // ordinary content, so `feed` also emits ITS OWN delta right
            // after releasing the span; including it in both would print
            // its text twice.
            if (tokens.len() + 1) * 4 > crate::tool_call::MAXIMUM_BYTES {
                if self.failed_span_tokens.is_none() {
                    self.failed_span_tokens = Some(tokens.clone());
                }
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            tokens.push(token_id);
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
        // THE THOUGHT CHANNEL IS EMITTED, NOT DISCARDED, and this arm used to
        // do the opposite. Discarding was right while nothing could turn
        // thinking on: `apply_chat_template` hardcoded `enable_thinking:
        // false`, so this checkpoint's generation prompt ended in a
        // pre-closed `<think>\n\n</think>\n\n` and a `<think>` body was
        // unreachable in practice. `--reasoning` makes it reachable, and a
        // caller who asked to see the model think should not have the answer
        // silently thrown away. The Gemma arm emits its own labelled thought
        // channel for the same reason and in the same commit.
        if self.channel == Channel::Thought {
            return Ok(if delta.is_empty() {
                Vec::new()
            } else {
                vec![StructuredAssistantEvent::Reasoning(delta.to_string())]
            });
        }
        if delta.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![StructuredAssistantEvent::Content(delta.to_string())])
        }
    }
}
