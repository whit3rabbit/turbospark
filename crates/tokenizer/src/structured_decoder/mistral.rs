//! The Mistral `[TOOL_CALLS]` decoder arm.
//!
//! Mistral's tool format has ONE special token and NO closing one: the turn
//! reads `[TOOL_CALLS][{"name": "f", "arguments": {..}}]` and the call body
//! is ordinary text that runs to end of turn. That inverts the Gemma/ChatML
//! span shape this decoder's default arm is built around -- there the end id
//! arrives mid-stream and the span closes on it -- so the arm buffers every
//! id after the marker and the call is emitted from [`Self::finish`], the
//! same terminator-blind route Harmony takes (crate Gotcha 49): end of turn
//! is `</s>`, a stop token, so `run_raw_completion` breaks before the
//! progress callback and no consume path could ever see the terminator.
//!
//! A second `[TOOL_CALLS]` inside one turn is malformed rather than a second
//! array: the format frames ALL of a turn's calls under one marker, so a
//! second one is a model confusing its own syntax and the released body is
//! what a rescue layer wants to see.

use super::{StructuredAssistantDecoder, StructuredAssistantEvent};
use crate::error::ToolCallParserError;
use crate::tool_call::MistralToolCallParser;

impl<'a> StructuredAssistantDecoder<'a> {
    /// The Mistral arm. Everything outside a `[TOOL_CALLS]` span is ordinary
    /// visible content; everything inside is buffered as token ids for the
    /// end-of-turn parse.
    pub(super) fn consume_mistral(
        &mut self,
        token_id: i32,
        delta: &str,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        if token_id == self.tokenizer.tool_call_start_id {
            if self.tool_tokens.is_some() {
                // A second marker abandons the first span, the way the
                // Gemma arm's second-start arm does.
                self.failed = true;
                return Err(ToolCallParserError::Malformed);
            }
            self.tool_tokens = Some(Vec::new());
            return Ok(Vec::new());
        }
        if let Some(tokens) = &mut self.tool_tokens {
            // The same 256 KiB ceiling the Gemma arm applies, checked BEFORE
            // the push for the same reason: the released body must not
            // double-emit the token that tripped it.
            if (tokens.len() + 1) * 4 > crate::tool_call::MAXIMUM_BYTES {
                self.failed = true;
                return Err(ToolCallParserError::Oversized);
            }
            tokens.push(token_id);
            return Ok(Vec::new());
        }
        // No span open, and this dialect has no channels: ordinary content.
        Ok(if delta.is_empty() {
            Vec::new()
        } else {
            vec![StructuredAssistantEvent::Content(delta.to_string())]
        })
    }

    /// Closes the span at end of turn, parsing the buffered body into calls.
    ///
    /// Called from `finish` and nowhere else: there is no in-stream token
    /// that closes a Mistral span. A body that will not parse releases the
    /// span's tokens the way every other failed span does, so the guardrails'
    /// rescue can still recover the calls; a turn with no open span is simply
    /// a turn with no calls.
    pub(super) fn close_mistral_tool(
        &mut self,
    ) -> Result<Vec<StructuredAssistantEvent>, ToolCallParserError> {
        let Some(tokens) = self.tool_tokens.take() else {
            return Ok(Vec::new());
        };
        let text = self.tokenizer.decode(&tokens, false);
        match MistralToolCallParser::new().parse(&text, &self.allowed_tools, &mut self.id_generator)
        {
            Ok((calls, remainder)) => {
                self.emitted_calls += calls.len();
                debug_assert_eq!(
                    self.tokenizer.dialect.tool_call_support(),
                    crate::ToolCallSupport::Native,
                    "a dialect that hands a parsed tool call over must answer Native to \
                     `tool_call_support`; the two matches have drifted"
                );
                // Prose the model wrote AFTER the call array is content, not
                // markup: the span swallowed it only because the format has
                // no closing token to stop the buffering. Calls go out first,
                // the trailing text after them, which is the order the model
                // produced them in.
                let mut events: Vec<StructuredAssistantEvent> = calls
                    .into_iter()
                    .map(StructuredAssistantEvent::ToolCall)
                    .collect();
                let remainder = remainder.trim();
                if !remainder.is_empty() {
                    events.push(StructuredAssistantEvent::Content(remainder.to_string()));
                }
                Ok(events)
            }
            Err(e) => {
                self.failed_span_tokens = Some(tokens);
                self.failed = true;
                Err(e)
            }
        }
    }
}
