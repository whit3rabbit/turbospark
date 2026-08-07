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
