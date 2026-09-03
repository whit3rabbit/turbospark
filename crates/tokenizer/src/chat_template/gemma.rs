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

#[cfg(test)]
mod gemma_template_tests {
    use super::*;

    fn msg(role: Role, content: &str) -> Message {
        Message {
            role,
            content: Some(content.to_string()),
            content_parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }
    }

    /// **THE PAIR THAT DECIDES WHAT A TOOL LOOP MAY SEND.** A caller feeding
    /// tool results back has two spellings available and on this dialect only
    /// one of them renders: `tool` gets an ordinary turn, `system` anywhere
    /// but first is refused. That reaches past this file, because
    /// `fit_window` prices a failing render at `u64::MAX` and drops turns
    /// until it stops failing -- so the refused spelling never surfaces as an
    /// error, it silently eats the run's own history.
    ///
    /// The five fallbacks do NOT agree here, and the disagreement is
    /// deliberate rather than an oversight: `deepseek`, `llama3` and
    /// `mistral` refuse `tool` because none of the checkpoints they render
    /// has tool markup to emit, and each names itself when it does. All five
    /// are unreachable for a real install (`apply_dialect_chat_template`'s
    /// own doc: this is what a MALFORMED install gets), which is why
    /// `TurboSparkApp` settles the question on the JINJA path instead and
    /// sends `tool`.
    #[test]
    fn a_tool_result_renders_where_a_late_system_message_is_refused() {
        let out = gemma_chat_template(&[
            msg(Role::User, "list the files"),
            msg(Role::Assistant, "calling ls"),
            msg(Role::Tool, "<tool_response>a.txt</tool_response>"),
        ])
        .unwrap();
        assert!(out.contains("<|turn>tool\n<tool_response>a.txt</tool_response><turn|>"));

        assert!(gemma_chat_template(&[
            msg(Role::User, "list the files"),
            msg(Role::Assistant, "calling ls"),
            msg(Role::System, "<tool_response>a.txt</tool_response>"),
        ])
        .is_err());
    }
}
