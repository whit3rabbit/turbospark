//! Gemma chat template fallback renderer.

use super::{Message, Role};
use crate::error::TokenizerError;

const TURN_OPEN: &str = "<|turn>";
const TURN_CLOSE: &str = "<turn|>";
const BOS_MARK: &str = "<bos>";

pub(super) fn gemma_chat_template(messages: &[Message]) -> Result<String, TokenizerError> {
    let mut s = BOS_MARK.to_string();
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
        let role = if message.role == Role::Assistant {
            "model"
        } else {
            message.role.as_str()
        };
        s.push_str(TURN_OPEN);
        s.push_str(role);
        s.push('\n');
        s.push_str(content);
        s.push_str(TURN_CLOSE);
        s.push('\n');
    }
    s.push_str(TURN_OPEN);
    s.push_str("model\n<|channel>thought\n<channel|>");
    Ok(s)
}

pub(super) fn gemma_continuation_suffix(content: &str) -> String {
    format!(
        "\n{TURN_OPEN}user\n{content}{TURN_CLOSE}\n{TURN_OPEN}model\n<|channel>thought\n<channel|>"
    )
}
