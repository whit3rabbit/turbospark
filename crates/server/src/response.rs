//! OpenAI Chat Completions response envelopes: the non-streaming completion
//! object and the per-chunk SSE delta object.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct ResponseMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct DeltaMessage {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChunkChoice {
    pub index: u32,
    pub delta: DeltaMessage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: &'static str,
    pub created: u64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
}

/// OpenAI-style stop-reason mapping: `stop` (a stop string or a normal EOS),
/// `length` (max tokens reached), or `tool_calls` (unused by this port).
pub fn finish_reason(reason: runtime::StopReason) -> &'static str {
    match reason {
        runtime::StopReason::MaxTokens => "length",
        runtime::StopReason::ToolCalls => "tool_calls",
        runtime::StopReason::EndOfTurn
        | runtime::StopReason::Eos
        | runtime::StopReason::StopString => "stop",
    }
}
