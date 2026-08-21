//! Tokenizer wrapper, chat-template rendering, streaming detokenizer, and
//! tool-call parsing for the Gemma 4, ChatML (Qwen), and DeepSeek-V4 model
//! dialects. Ported from `Sources/Mference/Tokenization` and
//! `Runtime/Generation/StreamingStopMatcher.swift`.
#![forbid(unsafe_code)]

mod chat_template;
mod detokenizer;
mod dialect;
mod error;
mod jinja_chat_template;
mod jinja_compat;
mod jinja_date;
mod json_value;
mod reasoning;
mod stop_matcher;
mod structured_decoder;
mod tool_call;

pub use chat_template::{FunctionDefinition, HistoricalToolCall, Message, Role};
pub use detokenizer::MfDetokenizer;
pub use dialect::{ChatDialect, MfTokenizer, NO_SUCH_TOKEN_ID};
pub use error::{TokenizerError, ToolCallParserError};
pub use jinja_chat_template::{render_generic_chat_template, CHAT_DATE_ENV};
pub use json_value::JsonValue;
pub use reasoning::{ReasoningEffort, ReasoningSupport};
pub use stop_matcher::StreamingStopMatcher;
pub use structured_decoder::{StructuredAssistantDecoder, StructuredAssistantEvent};
pub use tool_call::{
    DeepseekToolCallParser, GemmaToolCallParser, ParsedToolCall, QwenToolCallParser,
};

// Token id width consumed from the core primitives, keeping the dependency
// edge live and documenting the interchange type this crate uses throughout.
pub use foundation::TokenId;
