//! Channel split: separates reasoning tokens from final response content.

use std::collections::HashSet;

use tokenizer::{
    ChatDialect, MfTokenizer, ReasoningEffort, StructuredAssistantDecoder, StructuredAssistantEvent,
};

/// Prefill progress event kind.
pub const TS_EVENT_PREFILL: i32 = 0;
/// Content text delta event kind.
pub const TS_EVENT_CONTENT: i32 = 1;
/// Reasoning / thought text delta event kind.
pub const TS_EVENT_REASONING: i32 = 2;

/// Splits a token's text into the answer and the reasoning that preceded it.
///
/// Ported from `crates/cli/src/generate/format.rs::ChannelSplit`, and the two
/// conditions are the same because the reasons are:
///
/// - `gpt-oss` ALWAYS needs a decoder. Harmony puts the model's reasoning in
///   an `analysis` channel before its answer whatever the caller asked for,
///   so without this the reasoning and the frame markup arrive as the reply.
/// - ChatML and Gemma need one only when a reasoning LEVEL was asked for.
///   Their thought channels are unreachable otherwise, so building a decoder
///   unconditionally would route shipped families through a state machine
///   with nothing to do.
///
/// Skipping it is not cosmetic. Measured on the real Gemma 4 install the
/// first time a level was asked for, the reply began with a bare `thought`
/// (the channel LABEL, as prose), then the model's scratch work, then its
/// answer, all as one run of content.
pub(crate) struct ChannelSplit<'a> {
    decoder: Option<StructuredAssistantDecoder<'a>>,
}

impl<'a> ChannelSplit<'a> {
    /// `prompt_ids` is the rendered generation prompt: a ChatML template opens
    /// the `<think>` frame itself when thinking is on, and the decoder cannot
    /// tell without being shown (`StructuredAssistantDecoder::new`).
    pub(crate) fn new(
        tokenizer: &'a MfTokenizer,
        reasoning: ReasoningEffort,
        prompt_ids: &[i32],
    ) -> Self {
        let wanted = matches!(
            tokenizer.dialect,
            ChatDialect::Harmony | ChatDialect::MuseGlimmer
        ) || (reasoning != ReasoningEffort::Off
            && matches!(tokenizer.dialect, ChatDialect::ChatMl | ChatDialect::Gemma));
        Self {
            decoder: wanted.then(|| {
                // An EMPTY tool allowlist. This binding has no way to run a
                // tool, so a Harmony `commentary` body stays reasoning
                // rather than being parsed as a call the caller cannot
                // service. Tools belong to the server surface.
                StructuredAssistantDecoder::new(tokenizer, HashSet::new(), String::new, prompt_ids)
            }),
        }
    }

    /// One token's `(answer, reasoning)`. Either may be empty.
    pub(crate) fn push(&mut self, id: i32, text: &str) -> (String, String) {
        let Some(decoder) = self.decoder.as_mut() else {
            return (text.to_string(), String::new());
        };
        let (mut answer, mut reasoning) = (String::new(), String::new());
        match decoder.consume(id, text) {
            Ok(events) => {
                for event in events {
                    match event {
                        StructuredAssistantEvent::Content(c) => answer.push_str(&c),
                        StructuredAssistantEvent::Reasoning(r) => reasoning.push_str(&r),
                        // Unreachable with an empty allowlist, and dropping
                        // beats inventing a rendering for it.
                        StructuredAssistantEvent::ToolCall(_) => {}
                    }
                }
            }
            // Losing a caller's output to a decoder error would be the worst
            // outcome available: pass the text through.
            Err(_) => answer.push_str(text),
        }
        (answer, reasoning)
    }
}
