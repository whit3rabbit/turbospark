//! OpenAI Chat Completions response envelopes: the non-streaming completion
//! object and the per-chunk SSE delta object.
//!
//! The wire types come from `anyllm_translate::openai` rather than being
//! hand-rolled here, because they are also what `translate_response` and
//! `new_stream_translator` consume when `/v1/messages` re-renders the same
//! generation as Anthropic. One set of types, one construction path, two
//! endpoints. This module is only the constructors: filling in the many
//! fields this server never populates (logprobs, reasoning, system
//! fingerprint) exactly once, so the handlers stay readable.

use anyllm_translate::openai::streaming::{
    ChatCompletionChunk, ChunkChoice, ChunkDelta, ChunkFunctionCall, ChunkToolCall,
};
use anyllm_translate::openai::{
    ChatCompletionResponse, ChatContent, ChatMessage, ChatRole, ChatUsage, Choice, FinishReason,
    FunctionCall, ToolCall,
};
use tokenizer::ParsedToolCall;

/// OpenAI-style stop-reason mapping: `stop` (a stop string or a normal EOS),
/// `length` (max tokens reached), or `tool_calls`.
pub fn finish_reason(reason: runtime::StopReason) -> FinishReason {
    match reason {
        runtime::StopReason::MaxTokens => FinishReason::Length,
        runtime::StopReason::ToolCalls => FinishReason::ToolCalls,
        runtime::StopReason::EndOfTurn
        | runtime::StopReason::Eos
        | runtime::StopReason::StopString => FinishReason::Stop,
    }
}

/// One parsed tool call as an OpenAI `tool_calls` entry. `arguments_json` is
/// the exact text the dialect parser accepted, so nothing here re-serializes
/// the arguments and no key order or number formatting can drift.
fn tool_call(call: ParsedToolCall) -> ToolCall {
    ToolCall {
        id: call.id,
        call_type: "function".to_string(),
        function: FunctionCall {
            name: call.name,
            arguments: call.arguments_json,
        },
    }
}

/// An assistant turn: text, plus any tool calls the structured decoder
/// parsed out of it. This server generates no reasoning content (see
/// `DEVIATIONS.md`), so the remaining fields are `None`.
pub fn assistant_message(text: String, calls: Vec<ParsedToolCall>) -> ChatMessage {
    ChatMessage {
        role: ChatRole::Assistant,
        content: Some(ChatContent::Text(text)),
        name: None,
        tool_calls: if calls.is_empty() {
            None
        } else {
            Some(calls.into_iter().map(tool_call).collect())
        },
        tool_call_id: None,
        refusal: None,
        reasoning_content: None,
        thinking_blocks: None,
    }
}

/// The full (non-streaming) completion object.
// Every argument is one wire field with no natural grouping; bundling them
// into a struct would only move the same list one line up.
#[allow(clippy::too_many_arguments)]
pub fn completion_response(
    id: String,
    created: u64,
    model: String,
    text: String,
    calls: Vec<ParsedToolCall>,
    reason: runtime::StopReason,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        model,
        choices: vec![Choice {
            index: 0,
            message: assistant_message(text, calls),
            finish_reason: Some(finish_reason(reason)),
            logprobs: None,
        }],
        usage: Some(ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..Default::default()
        }),
        created: Some(created),
        system_fingerprint: None,
        service_tier: None,
    }
}

/// One SSE chunk. `delta` carries the role on the opening chunk, the text on
/// content chunks, and nothing on the closing chunk that carries `finish`.
pub fn completion_chunk(
    id: String,
    created: u64,
    model: String,
    delta: ChunkDelta,
    finish: Option<FinishReason>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason: finish,
            logprobs: None,
        }],
        usage: None,
        created: Some(created),
        system_fingerprint: None,
        error: None,
    }
}

/// The opening chunk's delta: role only, no content.
pub fn role_delta() -> ChunkDelta {
    ChunkDelta {
        role: Some(ChatRole::Assistant),
        ..Default::default()
    }
}

/// A content delta.
pub fn text_delta(text: String) -> ChunkDelta {
    ChunkDelta {
        content: Some(text),
        ..Default::default()
    }
}

/// A whole tool call in ONE chunk, rather than the name-then-argument-
/// fragments sequence a remote OpenAI backend streams. The structured
/// decoder only yields a call once its closing marker arrives, so there are
/// no fragments to stream. `StreamingTranslator` treats any chunk carrying
/// an id or a name as opening a new call and flushes on `finish_reason`, so
/// the Anthropic side still comes out as a well-formed
/// `content_block_start` / `input_json_delta` / `content_block_stop`.
pub fn tool_call_delta(index: u32, call: ParsedToolCall) -> ChunkDelta {
    ChunkDelta {
        tool_calls: Some(vec![ChunkToolCall {
            index,
            id: Some(call.id),
            call_type: Some("function".to_string()),
            function: Some(ChunkFunctionCall {
                name: Some(call.name),
                arguments: Some(call.arguments_json),
            }),
        }]),
        ..Default::default()
    }
}

/// The half of tool calling that is testable without a model: a parsed call
/// put through these constructors has to survive translation into an
/// Anthropic `tool_use` block, in both the full and the streaming shape.
#[cfg(test)]
mod tests {
    use super::*;
    use tokenizer::JsonValue;

    fn call() -> ParsedToolCall {
        ParsedToolCall {
            id: "toolu_0".to_string(),
            name: "get_weather".to_string(),
            arguments: JsonValue::parse(r#"{"city":"Oslo"}"#).unwrap(),
            arguments_json: r#"{"city":"Oslo"}"#.to_string(),
        }
    }

    #[test]
    fn a_call_becomes_an_anthropic_tool_use_block() {
        let response = completion_response(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            String::new(),
            vec![call()],
            runtime::StopReason::ToolCalls,
            3,
            7,
        );
        let anthropic = anyllm_translate::translate_response(&response, "claude-sonnet-4-6");
        let body = serde_json::to_value(&anthropic).unwrap();

        let block = body["content"]
            .as_array()
            .unwrap()
            .iter()
            .find(|b| b["type"] == "tool_use")
            .expect("a tool_use block");
        assert_eq!(block["id"], "toolu_0");
        assert_eq!(block["name"], "get_weather");
        assert_eq!(block["input"]["city"], "Oslo");
        assert_eq!(body["stop_reason"], "tool_use");
    }

    #[test]
    fn a_streamed_call_becomes_a_tool_use_content_block() {
        let mut translator =
            anyllm_translate::new_stream_translator("claude-sonnet-4-6".to_string());
        let mut names = Vec::new();
        let mut record = |chunk| {
            for event in translator.process_chunk(&chunk) {
                names.push(serde_json::to_value(&event).unwrap());
            }
        };

        record(completion_chunk(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            role_delta(),
            None,
        ));
        record(completion_chunk(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            tool_call_delta(0, call()),
            None,
        ));
        record(completion_chunk(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            ChunkDelta::default(),
            Some(FinishReason::ToolCalls),
        ));
        for event in translator.finish() {
            names.push(serde_json::to_value(&event).unwrap());
        }

        // One complete chunk still opens a tool_use block and carries its
        // arguments as an input_json delta.
        let start = names
            .iter()
            .find(|e| e["type"] == "content_block_start")
            .expect("a content_block_start");
        assert_eq!(start["content_block"]["type"], "tool_use");
        assert_eq!(start["content_block"]["name"], "get_weather");
        assert!(
            names
                .iter()
                .any(|e| e["delta"]["type"] == "input_json_delta"
                    && e["delta"]["partial_json"] == r#"{"city":"Oslo"}"#),
            "{names:?}"
        );
        assert!(names.iter().any(|e| e["type"] == "content_block_stop"));
    }
}
