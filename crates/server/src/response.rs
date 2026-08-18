//! OpenAI Chat Completions response envelopes: the non-streaming completion
//! object and the per-chunk SSE delta object.
//!
//! The wire types come from `anyllm_translate::openai` rather than being
//! hand-rolled here, because they are also what `translate_response` and
//! `new_stream_translator` consume when `/v1/messages` re-renders the same
//! generation as Anthropic. One set of types, one construction path, two
//! endpoints. This module is only the constructors: filling in the many
//! fields this server never populates (logprobs, signed thinking blocks,
//! system fingerprint) exactly once, so the handlers stay readable.

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
        // `Cancelled` is `stop` because OpenAI has no other spelling for it,
        // and it is UNREACHABLE from this server today: nothing here calls
        // the cancellable entry points, so no generation this crate drives
        // can produce it. Named rather than left to a wildcard so a future
        // per-request cancel has to come past this line and decide.
        runtime::StopReason::EndOfTurn
        | runtime::StopReason::Eos
        | runtime::StopReason::StopString
        | runtime::StopReason::Cancelled => FinishReason::Stop,
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

/// An assistant turn: text, the reasoning that preceded it, and any tool calls
/// the structured decoder parsed out of it.
///
/// `reasoning_content` is the de-facto OpenAI field for a thinking model's
/// separated reasoning, and populating it is the WHOLE server-side cost of
/// surfacing Harmony's `analysis` channel: `translate_response` already turns
/// it into an Anthropic `thinking` block. `thinking_blocks` stays `None`
/// because that field carries Anthropic's SIGNED blocks, which only Anthropic
/// can mint; a local model has nothing to sign with.
pub fn assistant_message(
    text: String,
    reasoning: String,
    calls: Vec<ParsedToolCall>,
) -> ChatMessage {
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
        reasoning_content: (!reasoning.is_empty()).then_some(reasoning),
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
    reasoning: String,
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
            message: assistant_message(text, reasoning, calls),
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

/// An SSE chunk carrying usage statistics with empty choices, emitted when
/// `stream_options.include_usage` is true.
pub fn usage_chunk(
    id: String,
    created: u64,
    model: String,
    prompt_tokens: u32,
    completion_tokens: u32,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id,
        object: "chat.completion.chunk".to_string(),
        model,
        choices: vec![],
        usage: Some(ChatUsage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            ..Default::default()
        }),
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

/// A reasoning delta, which `StreamingTranslator` renders as an Anthropic
/// `thinking` content block.
///
/// ORDER IS LOAD-BEARING AND IS INHERITED, not enforced here. The translator
/// opens a thinking block on the first reasoning delta and CLOSES it on the
/// first content delta, without reopening; Harmony emits its `analysis`
/// channel before its `final` one, so the sequence this server produces is
/// well formed. A backend that interleaved the two would need the translator
/// to grow a second thinking block, which is upstream work rather than
/// something to paper over by reordering a caller's output.
pub fn reasoning_delta(text: String) -> ChunkDelta {
    ChunkDelta {
        reasoning_content: Some(text),
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

/// The half of tool calling and reasoning that is testable without a model: a
/// parsed call, or a separated reasoning string, put through these
/// constructors has to survive translation into the Anthropic block it claims
/// to map onto, in both the full and the streaming shape.
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

    /// Separated reasoning becomes an Anthropic `thinking` block, BEFORE the
    /// text block. This is the whole server-side claim of Harmony channel
    /// decoding: fill `reasoning_content` and `anyllm_translate` does the
    /// rest, so what is under test here is that the field really is the one
    /// its mapping reads.
    #[test]
    fn reasoning_becomes_an_anthropic_thinking_block() {
        let response = completion_response(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            "Rayleigh scattering.".to_string(),
            "The user asks why the sky is blue.".to_string(),
            Vec::new(),
            runtime::StopReason::EndOfTurn,
            3,
            7,
        );
        let anthropic = anyllm_translate::translate_response(&response, "claude-sonnet-4-6");
        let body = serde_json::to_value(&anthropic).unwrap();

        let kinds: Vec<&str> = body["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["thinking", "text"], "{body}");
        assert_eq!(
            body["content"][0]["thinking"],
            "The user asks why the sky is blue."
        );
        assert_eq!(body["content"][1]["text"], "Rayleigh scattering.");
    }

    /// A generation with no reasoning must not grow an empty thinking block:
    /// every family but `gpt-oss` produces none, and this is the guard that
    /// their responses are unchanged.
    #[test]
    fn no_reasoning_leaves_the_response_shape_alone() {
        let response = completion_response(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            "plain answer".to_string(),
            String::new(),
            Vec::new(),
            runtime::StopReason::EndOfTurn,
            3,
            7,
        );
        assert!(response.choices[0].message.reasoning_content.is_none());

        let body = serde_json::to_value(anyllm_translate::translate_response(
            &response,
            "claude-sonnet-4-6",
        ))
        .unwrap();
        let kinds: Vec<&str> = body["content"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b["type"].as_str().unwrap())
            .collect();
        assert_eq!(kinds, vec!["text"], "{body}");
    }

    /// The streaming shape: reasoning deltas open a `thinking` content block
    /// and the first text delta CLOSES it, so the two arrive as separate
    /// blocks rather than as one run of text.
    #[test]
    fn streamed_reasoning_opens_a_thinking_block_that_text_closes() {
        let mut translator =
            anyllm_translate::new_stream_translator("claude-sonnet-4-6".to_string());
        let mut events = Vec::new();
        let mut record = |chunk| {
            for event in translator.process_chunk(&chunk) {
                events.push(serde_json::to_value(&event).unwrap());
            }
        };
        let chunk =
            |delta| completion_chunk("chatcmpl-1".to_string(), 0, "m".to_string(), delta, None);

        record(chunk(role_delta()));
        record(chunk(reasoning_delta("thinking out loud".to_string())));
        record(chunk(text_delta("the answer".to_string())));
        record(completion_chunk(
            "chatcmpl-1".to_string(),
            0,
            "m".to_string(),
            ChunkDelta::default(),
            Some(FinishReason::Stop),
        ));
        for event in translator.finish() {
            events.push(serde_json::to_value(&event).unwrap());
        }

        let deltas: Vec<&str> = events
            .iter()
            .filter_map(|e| e["delta"]["type"].as_str())
            .collect();
        assert_eq!(deltas, vec!["thinking_delta", "text_delta"], "{events:?}");

        let opened: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "content_block_start")
            .map(|e| e["content_block"]["type"].as_str().unwrap())
            .collect();
        assert_eq!(opened, vec!["thinking", "text"], "{events:?}");
    }
}
