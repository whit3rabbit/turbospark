//! ChatML (Qwen) chat template fallback renderer.

use super::{Message, Role};
use crate::error::TokenizerError;

const IM_START_MARK: &str = "<|im_start|>";
const IM_END_MARK: &str = "<|im_end|>";
pub(super) const CHATML_GENERATION_SUFFIX: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

pub(super) fn chatml_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut s = String::new();
    for (index, message) in messages.iter().enumerate() {
        let Some(raw) = &message.content else {
            return Err(TokenizerError::InvalidChatTemplate(
                "text-only messages require content".to_string(),
            ));
        };
        let content = raw.trim();
        if message.role == Role::System && index != 0 {
            return Err(TokenizerError::InvalidChatTemplate(
                "system message must be first".to_string(),
            ));
        }
        s.push_str(IM_START_MARK);
        s.push_str(message.role.as_str());
        s.push('\n');
        s.push_str(content);
        s.push_str(IM_END_MARK);
        s.push('\n');
    }
    s.push_str(CHATML_GENERATION_SUFFIX);
    Ok(s)
}

pub(super) fn chatml_continuation_suffix(content: &str) -> String {
    format!("\n{IM_START_MARK}user\n{content}{IM_END_MARK}\n{CHATML_GENERATION_SUFFIX}")
}
